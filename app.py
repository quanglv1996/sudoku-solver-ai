"""
app.py

Flask Backend API for Sudoku Solver Web Application

Kiến trúc Manager + Queue:
    Browser / Phone ──HTTP──▶ Flask (nhiều thread, chỉ nhận/trả dữ liệu)
                                  │ submit()
                                  ▼
                            TaskManager (PriorityQueue)
                                  │
                                  ▼
                        1 Worker thread sở hữu SudokuEngine (YOLO + PaddleOCR)

Endpoints:
    GET    /                      Giao diện web
    GET    /api/health            Trạng thái engine, hàng đợi, URL truy cập
    POST   /api/jobs              Gửi ảnh chụp → job (202)
    GET    /api/jobs/<id>         Tiến độ / kết quả job (?wait=ms để long-poll)
    DELETE /api/jobs/<id>         Huỷ job
    POST   /api/realtime          Gửi 1 frame camera → overlay lời giải (đồng bộ)
    GET    /api/screen/monitors   Danh sách màn hình của máy chủ
    POST   /api/screen            Tự giải sudoku trên màn hình máy chủ → job (202)
"""

import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

import config  # noqa: E402
from engine import build_manager, limit_size  # noqa: E402
from task_manager import JobStatus, Priority, QueueFullError  # noqa: E402
from tls import ensure_certificate, lan_ips  # noqa: E402

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logger = logging.getLogger("app")

app = Flask(__name__, template_folder=str(config.TEMPLATES_DIR), static_folder=str(config.STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = config.MAX_FILE_SIZE
app.json.ensure_ascii = False
CORS(app, origins=config.CORS_ORIGINS)

manager = build_manager(config)
ASSET_VERSION = str(int(time.time()))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "heic", "heif"}


# =====================================================================
# Helpers
# =====================================================================

def error(message: str, status: int = 400, **extra):
    return jsonify({"success": False, "message": message, **extra}), status


def read_image(field: str, max_side: int):
    """Đọc ảnh từ multipart field; trả về (image, error_response)"""
    file = request.files.get(field)
    if file is None or file.filename == "":
        return None, error("Chưa có ảnh")
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    if ext not in ALLOWED_EXTENSIONS and not (file.mimetype or "").startswith("image/"):
        return None, error("Định dạng ảnh không hỗ trợ")
    data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        return None, error("Không đọc được ảnh (thử JPG/PNG)")
    return limit_size(image, max_side), None


def engine_unavailable():
    if manager.engine_state == "error":
        return error(f"Engine lỗi: {manager.engine_error}", 500)
    if manager.engine_state != "ready":
        return error("Model đang được tải, vui lòng đợi…", 503, engine=manager.engine_state)
    return None


def job_response(job, status_code: int = 200):
    data = job.to_dict()
    data["position"] = manager.queue_position(job)
    return jsonify(data), status_code


def access_urls():
    ips = config.PUBLIC_HOSTS or lan_ips()
    urls = {"http": [f"http://localhost:{config.HTTP_PORT}"] + [f"http://{ip}:{config.HTTP_PORT}" for ip in ips]}
    if config.HTTPS_ENABLED:
        urls["https"] = [f"https://{ip}:{config.HTTPS_PORT}" for ip in ips] or [f"https://localhost:{config.HTTPS_PORT}"]
    return urls


def is_local_request() -> bool:
    return request.remote_addr in {"127.0.0.1", "::1", *lan_ips()}


# =====================================================================
# Routes
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html", v=ASSET_VERSION)


@app.route("/api/health")
def health():
    engine = manager.engine
    return jsonify({
        "status": "ok",
        "engine": manager.engine_state,
        "engine_error": manager.engine_error,
        "device": getattr(engine, "device", None),
        "queue": manager.stats(),
        "urls": access_urls(),
        "screen": {
            "enabled": config.SCREEN_CONTROL,
            "allowed": config.SCREEN_CONTROL and (not config.SCREEN_LOCAL_ONLY or is_local_request()),
            "is_server_machine": is_local_request(),
        },
        "timestamp": time.time(),
    })


@app.route("/api/jobs", methods=["POST"])
def create_job():
    if manager.engine_state == "error":  # khi đang tải model vẫn nhận job, job sẽ chờ trong hàng đợi
        return engine_unavailable()
    image, err = read_image("image", config.MAX_IMAGE_SIZE)
    if err:
        return err
    try:
        job = manager.submit("photo", {"image": image}, priority=Priority.PHOTO)
    except QueueFullError as e:
        return error(str(e), 429)
    logger.info("photo job %s queued (%dx%d)", job.id, image.shape[1], image.shape[0])
    return job_response(job, 202)


@app.route("/api/solve", methods=["POST"])
def solve_sync():
    """API cũ (đồng bộ): vẫn đi qua hàng đợi, chờ job xong rồi trả kết quả"""
    if manager.engine_state == "error":
        return engine_unavailable()
    image, err = read_image("image", config.MAX_IMAGE_SIZE)
    if err:
        return err
    try:
        job = manager.submit("photo", {"image": image}, priority=Priority.PHOTO)
    except QueueFullError as e:
        return error(str(e), 429)
    if not job.done_event.wait(config.SYNC_TIMEOUT):
        manager.cancel(job.id)
        return error("Hết thời gian chờ xử lý", 504)
    if job.status != JobStatus.DONE:
        return error(job.error or "Xử lý thất bại", 500)
    result = dict(job.result)
    if result.get("detected_grid"):
        result["difficulty"] = result.get("givens", 0)
    return jsonify(result), 200


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = manager.get(job_id)
    if job is None:
        return error("Job không tồn tại hoặc đã hết hạn", 404)
    wait_ms = min(int(request.args.get("wait", 0) or 0), 10_000)
    if wait_ms and job.status not in JobStatus.FINAL:
        # long-poll: trả về khi job xong hoặc khi stage đổi
        stage, deadline = job.stage, time.time() + wait_ms / 1000
        while time.time() < deadline and job.stage == stage and not job.done_event.is_set():
            job.done_event.wait(0.05)
    return job_response(job)


@app.route("/api/jobs/<job_id>", methods=["DELETE"])
def cancel_job(job_id):
    return jsonify({"cancelled": manager.cancel(job_id)})


@app.route("/api/realtime", methods=["POST"])
def realtime_frame():
    if resp := engine_unavailable():
        return resp
    image, err = read_image("frame", config.REALTIME_MAX_SIZE)
    if err:
        return err
    session = (request.form.get("session") or request.remote_addr or "default")[:64]
    payload = {"image": image, "reset": request.form.get("reset") == "1"}
    job = manager.submit("realtime", payload, priority=Priority.REALTIME,
                         session=session, latest_only=True)
    job.done_event.wait(config.REALTIME_TIMEOUT)

    if job.status == JobStatus.DONE:
        return jsonify({"ok": True, **job.result})
    if job.status == JobStatus.FAILED:
        return jsonify({"ok": False, "status": "error", "message": job.error})
    if job.status == JobStatus.DROPPED:
        return jsonify({"ok": False, "status": "dropped"})
    manager.cancel(job.id)
    return jsonify({"ok": False, "status": "busy"})


@app.route("/api/screen/monitors")
def screen_monitors():
    from screen_solver import list_monitors
    try:
        return jsonify({"monitors": list_monitors()})
    except Exception as e:
        return error(f"Không đọc được màn hình: {e}", 500)


@app.route("/api/screen", methods=["POST"])
def screen_solve():
    if not config.SCREEN_CONTROL:
        return error("Tính năng điều khiển màn hình đang tắt (SCREEN_CONTROL=false)", 403)
    if config.SCREEN_LOCAL_ONLY and not is_local_request():
        return error("Chỉ được kích hoạt từ máy chạy server", 403)
    if resp := engine_unavailable():
        return resp
    body = request.get_json(silent=True) or {}
    payload = {
        "monitor": int(body.get("monitor", 1)),
        "fill": bool(body.get("fill", True)),
        "type_delay": max(0.0, min(float(body.get("type_delay", 0.04)), 1.0)),
    }
    try:
        job = manager.submit("screen", payload, priority=Priority.SCREEN)
    except QueueFullError as e:
        return error(str(e), 429)
    logger.info("screen job %s queued %s", job.id, payload)
    return job_response(job, 202)


@app.errorhandler(404)
def not_found(_):
    return error("Not found", 404)


@app.errorhandler(413)
def too_large(_):
    return error(f"Ảnh quá lớn (tối đa {config.MAX_FILE_SIZE // (1024 * 1024)}MB)", 413)


# =====================================================================
# Serving (HTTP + HTTPS song song)
# =====================================================================

def serve():
    from werkzeug.serving import make_server

    manager.start()  # bắt đầu load model ngay trong worker thread

    servers = [make_server(config.SERVER_HOST, config.HTTP_PORT, app, threaded=True)]
    if config.HTTPS_ENABLED:
        try:
            cert, key = ensure_certificate(config.CERT_DIR, config.PUBLIC_HOSTS)
            servers.append(make_server(config.SERVER_HOST, config.HTTPS_PORT, app,
                                       threaded=True, ssl_context=(cert, key)))
        except Exception as e:
            logger.warning("HTTPS disabled: %s", e)

    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    urls = access_urls()
    print("\n" + "=" * 64)
    print("🧩 Sudoku Solver AI đang chạy")
    print("   Máy tính :", urls["http"][0])
    for u in urls.get("https", []):
        print("   Điện thoại (camera, cùng Wi-Fi):", u)
    print("   (Lần đầu mở HTTPS trên điện thoại: chọn 'Nâng cao' → 'Tiếp tục' vì cert tự ký)")
    if os.environ.get("IN_DOCKER") and not config.PUBLIC_HOSTS:
        print("   ⚠️  Đang chạy trong Docker: IP trên là IP nội bộ của container.")
        print("      Đặt PUBLIC_HOSTS=<IP LAN của máy host> trong .env để điện thoại truy cập đúng địa chỉ.")
    print("   Model đang tải ở nền - xem trạng thái ở góc trên giao diện")
    print("   Ctrl+C để dừng")
    print("=" * 64 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping…")
    finally:
        manager.stop()
        for srv in servers:
            srv.shutdown()


if __name__ == "__main__":
    serve()
