"""
engine.py

SudokuEngine: gom các thành phần dùng model (pipeline, realtime, screen)
và các handler cho từng loại job của TaskManager.

Mọi hàm ở đây chạy trong worker thread của TaskManager.
"""

import base64
import logging
import time
from typing import Any, Dict

import cv2
import numpy as np

from pipeline import SudokuSolverPipeline
from realtime import RealtimeTracker
from screen_solver import ScreenSolver
from task_manager import Job, TaskManager

logger = logging.getLogger(__name__)


class SudokuEngine:
    def __init__(self, cfg):
        t0 = time.time()
        self.pipeline = SudokuSolverPipeline(
            point_detection_model=str(cfg.POINT_MODEL),
            cell_size=cfg.CELL_SIZE,
            device=cfg.DEVICE,
            conf_threshold=cfg.OCR_CONF_THRESHOLD,
            ocr_model_name=cfg.OCR_MODEL_NAME,
            point_conf=cfg.POINT_CONF,
            save_debug=cfg.SAVE_DEBUG,
            debug_dir=str(cfg.DEBUG_DIR),
        )
        self.pipeline.warmup()
        self.realtime = RealtimeTracker(self.pipeline)
        self.screen = ScreenSolver(self.pipeline)
        self.device = self.pipeline.device
        logger.info("Engine loaded in %.1fs (device=%s)", time.time() - t0, self.device)


# =====================================================================
# Helpers
# =====================================================================

def to_data_url(image: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def limit_size(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def summarize(result: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
    """Chuyển kết quả pipeline sang JSON (không kèm ảnh)"""
    h, w = image.shape[:2]
    out: Dict[str, Any] = {
        "success": bool(result.get("success")),
        "stage": result.get("stage"),
        "message": "Giải thành công!" if result.get("success") else result.get("error", "Thất bại"),
        "method": result.get("method"),
        "steps": result.get("steps", {}),
        "timings": result.get("timings", {}),
        "width": w,
        "height": h,
    }
    if result.get("corners") is not None:
        out["corners"] = (result["corners"] / [w, h]).round(4).tolist()
    for key in ("detected_grid", "solved_grid"):
        if result.get(key) is not None:
            out[key] = np.asarray(result[key]).astype(int).tolist()
    if result.get("detected_grid") is not None:
        out["givens"] = int(np.count_nonzero(result["detected_grid"]))
    out["conflicts"] = [list(map(int, rc)) for rc in result.get("conflicts", [])]
    out["uncertain"] = [list(map(int, rc)) for rc in result.get("uncertain", [])]
    return out


# =====================================================================
# Job handlers
# =====================================================================

def handle_photo(job: Job, engine: SudokuEngine) -> Dict[str, Any]:
    image = job.payload["image"]
    result = engine.pipeline.run(image, progress=job.set_stage)

    job.set_stage("render")
    response = summarize(result, image)
    response["original_image"] = to_data_url(image)
    if result.get("success"):
        response["solved_image"] = to_data_url(engine.pipeline.render_solution(image, result))
    if result.get("aligned_image") is not None:
        response["aligned_image"] = to_data_url(result["aligned_image"])
    return response


def handle_realtime(job: Job, engine: SudokuEngine) -> Dict[str, Any]:
    session = job.session or "default"
    if job.payload.get("reset"):
        engine.realtime.reset(session)
    return engine.realtime.process(session, job.payload["image"])


def handle_screen(job: Job, engine: SudokuEngine) -> Dict[str, Any]:
    p = job.payload
    out = engine.screen.solve(
        monitor=p.get("monitor", 1),
        fill=p.get("fill", True),
        type_delay=p.get("type_delay", 0.04),
        progress=job.set_stage,
    )
    image, result = out["image"], out["result"]
    response = summarize(result, image)
    response["filled"] = out["filled"]
    response["fill_error"] = out["fill_error"]

    # Ảnh xem trước: chỉ vùng lưới (kèm lề) để gọn nhẹ
    preview = engine.pipeline.render_solution(image, result) if result.get("success") else image
    corners = result.get("corners")
    if corners is not None:
        x1, y1 = np.maximum(corners.min(axis=0) - 40, 0).astype(int)
        x2, y2 = corners.max(axis=0).astype(int) + 40
        preview = preview[y1:y2, x1:x2]
    response["preview_image"] = to_data_url(limit_size(preview, 900))
    return response


def build_manager(cfg) -> TaskManager:
    manager = TaskManager(
        engine_factory=lambda: SudokuEngine(cfg),
        max_queue=cfg.MAX_QUEUE,
        job_ttl=cfg.JOB_TTL,
    )
    manager.register("photo", handle_photo)
    manager.register("realtime", handle_realtime)
    manager.register("screen", handle_screen)
    return manager
