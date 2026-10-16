"""
test_integration.py - test với model thật (YOLO + PaddleOCR): pipeline, realtime, API

    pytest tests/test_integration.py -v
"""

import io
import json
import threading
import time

import cv2
import numpy as np
import pytest

from conftest import PUZZLE_CLASSIC, PUZZLE_PHOTO, TEST_IMAGE, grid_of, render_screen

pytestmark = pytest.mark.model

EXPECTED_PHOTO = grid_of(PUZZLE_PHOTO)


# =====================================================================
# Biến đổi ảnh chụp
# =====================================================================

def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def tilt(img, k):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * k, 0], [w * (1 - k), 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def noise(img, sigma):
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def pad(img, px):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_REPLICATE)


PHOTO_VARIANTS = {
    "original": lambda im: im,
    # thêm lề trước khi xoay để cả 4 góc lưới vẫn nằm trong ảnh (như khi chụp thật)
    "rotate_+8deg": lambda im: rotate(pad(im, 120), 8),
    "rotate_-15deg": lambda im: rotate(pad(im, 120), -15),
    "rotate_+25deg": lambda im: rotate(pad(im, 120), 25),
    "rotate_-35deg": lambda im: rotate(pad(im, 120), -35),
    "perspective_tilt": lambda im: tilt(im, 0.12),
    "small_0.6x": lambda im: cv2.resize(im, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA),
    "large_2x": lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    "dark_gamma_1.8": lambda im: gamma(im, 1.8),
    "bright_gamma_0.6": lambda im: gamma(im, 0.6),
    "blur_5": lambda im: cv2.GaussianBlur(im, (5, 5), 0),
    "jpeg_q20": lambda im: jpeg(im, 20),
    "sensor_noise": lambda im: noise(im, 10),
    "far_away": lambda im: cv2.resize(pad(im, 300), (640, 480), interpolation=cv2.INTER_AREA),
}


@pytest.mark.parametrize("name", list(PHOTO_VARIANTS))
def test_photo_variants(engine, photo, name):
    img = PHOTO_VARIANTS[name](photo)
    r = engine.pipeline.run(img)
    assert r.get("detected_grid") is not None, r.get("error")
    wrong = np.argwhere(r["detected_grid"] != EXPECTED_PHOTO).tolist()
    assert not wrong, f"{name}: đọc sai ô {wrong}"
    assert r["success"], r.get("error")


def test_photo_without_sudoku(engine):
    blank = np.full((480, 640, 3), 180, np.uint8)
    cv2.putText(blank, "no sudoku here", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    r = engine.pipeline.run(blank)
    assert not r["success"] and r["stage"] == "detect"


def test_render_solution_keeps_size(engine, photo):
    r = engine.pipeline.run(photo)
    out = engine.pipeline.render_solution(photo, r)
    assert out.shape == photo.shape and not np.array_equal(out, photo)


# =====================================================================
# Ảnh chụp màn hình (chế độ Màn hình PC)
# =====================================================================

SCREEN_VARIANTS = {
    "light": dict(),
    "dark_mode": dict(dark=True),
    "highlighted_cells": dict(highlight=True),
    "pencil_notes": dict(notes=True),
    "dark_highlight_notes": dict(dark=True, highlight=True, notes=True),
    "small_grid_40px": dict(cell=40, x0=200, y0=600),
    "large_grid_90px": dict(cell=90, x0=500, y0=100),
    "duplex_font": dict(font=cv2.FONT_HERSHEY_DUPLEX),
}


@pytest.mark.parametrize("name", list(SCREEN_VARIANTS))
def test_screen_variants(engine, name):
    grid = grid_of(PUZZLE_CLASSIC)
    opts = SCREEN_VARIANTS[name]
    img = render_screen(grid, **opts)
    r = engine.pipeline.run(img, prefer="contour")
    assert r.get("detected_grid") is not None, r.get("error")
    wrong = np.argwhere(r["detected_grid"] != grid).tolist()
    assert not wrong, f"{name}: đọc sai ô {wrong}"
    assert r["success"]

    # Toạ độ click phải nằm giữa ô (sai lệch < 15% cạnh ô)
    cell, x0, y0 = opts.get("cell", 60), opts.get("x0", 700), opts.get("y0", 200)
    for i, c in enumerate(r["cells"]):
        row, col = divmod(i, 9)
        ex, ey = x0 + (col + 0.5) * cell, y0 + (row + 0.5) * cell
        assert abs(c["x"] - ex) < 0.15 * cell and abs(c["y"] - ey) < 0.15 * cell, (row, col, c)


# =====================================================================
# Realtime tracker
# =====================================================================

def test_realtime_tracking_and_puzzle_switch(engine, photo):
    rt = engine.realtime
    rt.reset("t")
    first = rt.process("t", photo)
    assert first["status"] == "solved" and len(first["cells"]) == 81 - 28

    moved = rotate(photo, 3)
    t = time.perf_counter()
    second = rt.process("t", moved)
    assert second["status"] == "tracking", "cùng đề → phải dùng lại lời giải"
    assert time.perf_counter() - t < 0.5
    assert [c["d"] for c in second["cells"]] == [c["d"] for c in first["cells"]]

    other = render_screen(grid_of(PUZZLE_CLASSIC), x0=100, y0=72, cell=44, size=(480, 640))
    third = rt.process("t", other)
    assert third["status"] == "solved", "đề mới → phải giải lại"
    assert len(third["cells"]) == 81 - int(np.count_nonzero(grid_of(PUZZLE_CLASSIC)))

    empty = rt.process("t", np.full((480, 640, 3), 120, np.uint8))
    assert empty["status"] == "searching" and not empty["found"]


# =====================================================================
# HTTP API (Flask test client, đi qua TaskManager thật)
# =====================================================================

def upload(path=TEST_IMAGE, field="image"):
    return {field: (io.BytesIO(path.read_bytes()), "test.jpg")}


def test_api_health(client):
    h = client.get("/api/health").get_json()
    assert h["engine"] == "ready" and h["device"] in ("cuda", "cpu")
    assert "http" in h["urls"]


def test_api_photo_job_flow(client):
    res = client.post("/api/jobs", data=upload(), content_type="multipart/form-data")
    assert res.status_code == 202
    job = res.get_json()
    stages = set()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}?wait=1000").get_json()
        stages.add(job["stage"])
        if job["status"] in ("done", "failed"):
            break
    assert job["status"] == "done"
    r = job["result"]
    assert r["success"] and r["givens"] == 28
    assert r["detected_grid"] == EXPECTED_PHOTO.tolist()
    assert all(r[k].startswith("data:image/jpeg;base64,") for k in ("original_image", "solved_image", "aligned_image"))
    assert len(r["corners"]) == 4 and all(0 <= v <= 1 for p in r["corners"] for v in p)


def test_api_solve_legacy_sync(client):
    res = client.post("/api/solve", data=upload(), content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 200 and body["success"] and body["difficulty"] == 28


def test_api_realtime(client):
    data = upload(field="frame")
    data["session"] = "api-test"
    data["reset"] = "1"
    a = client.post("/api/realtime", data=data, content_type="multipart/form-data").get_json()
    assert a["ok"] and a["status"] == "solved"
    data = upload(field="frame")
    data["session"] = "api-test"
    b = client.post("/api/realtime", data=data, content_type="multipart/form-data").get_json()
    assert b["ok"] and b["status"] == "tracking"


def test_api_errors(client):
    assert client.post("/api/jobs", data={}, content_type="multipart/form-data").status_code == 400
    bad = {"image": (io.BytesIO(b"not an image"), "x.jpg")}
    assert client.post("/api/jobs", data=bad, content_type="multipart/form-data").status_code == 400
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.delete("/api/jobs/doesnotexist").get_json() == {"cancelled": False}
    assert client.get("/api/nope").status_code == 404


def test_api_concurrent_clients(app_module):
    """Nhiều người dùng cùng lúc: tất cả đều được xử lý qua hàng đợi, không lỗi"""
    results = []

    def worker():
        c = app_module.app.test_client()
        res = c.post("/api/solve", data=upload(), content_type="multipart/form-data")
        results.append((res.status_code, res.get_json().get("success")))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert results == [(200, True)] * 6


def test_api_screen_monitors_and_job(client):
    mons = client.get("/api/screen/monitors").get_json()["monitors"]
    assert mons and mons[0]["width"] > 0
    res = client.post("/api/screen", data=json.dumps({"monitor": 1, "fill": False}), content_type="application/json")
    assert res.status_code == 202
    job = res.get_json()
    for _ in range(60):
        job = client.get(f"/api/jobs/{job['id']}?wait=1000").get_json()
        if job["status"] in ("done", "failed"):
            break
    # Màn hình hiện tại có thể không có sudoku: chỉ cần job chạy xong, không lỗi
    assert job["status"] == "done" and job["result"]["preview_image"].startswith("data:image/jpeg")
