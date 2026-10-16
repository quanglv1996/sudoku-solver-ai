"""
conftest.py - fixtures dùng chung cho bộ test

    pytest tests -v                 # toàn bộ (cần GPU/CPU + model)
    pytest tests -v -m "not model"  # chỉ test nhanh, không load model
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

TEST_IMAGE = ROOT / "imgs" / "test.jpg"

# Đề trong imgs/test.jpg (đã kiểm tra bằng mắt)
PUZZLE_PHOTO = "046080000901300000720506000038000900100000003002000680000607051000009704000010820"
# Đề kinh điển (Wikipedia)
PUZZLE_CLASSIC = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"


def grid_of(puzzle: str) -> np.ndarray:
    return np.array([int(c) for c in puzzle]).reshape(9, 9)


def pytest_configure(config):
    config.addinivalue_line("markers", "model: test cần load model YOLO/PaddleOCR")


# =====================================================================
# Ảnh màn hình sudoku tổng hợp
# =====================================================================

def render_screen(
    grid: np.ndarray,
    x0: int = 700,
    y0: int = 200,
    cell: int = 60,
    size=(1080, 1920),
    dark: bool = False,
    highlight: bool = False,
    notes: bool = False,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> np.ndarray:
    """Vẽ một trang web sudoku giả lập (nền, thanh tiêu đề, lưới, chữ số)"""
    bg, line, ink = ((32, 32, 36), (200, 200, 200), (235, 235, 235)) if dark else ((245, 245, 245), (30, 30, 30), (40, 40, 40))
    img = np.full((*size, 3), bg, np.uint8)
    cv2.rectangle(img, (0, 0), (size[1], 60), (70, 70, 70) if not dark else (55, 55, 60), -1)
    cv2.putText(img, "My Sudoku Site", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 230, 230), 2)
    S = cell * 9

    if highlight:  # hàng/cột/khối đang chọn được tô màu như các trang web thật
        hl = (70, 60, 40) if dark else (250, 225, 200)
        cv2.rectangle(img, (x0, y0 + 4 * cell), (x0 + S, y0 + 5 * cell), hl, -1)
        cv2.rectangle(img, (x0 + 2 * cell, y0), (x0 + 3 * cell, y0 + S), hl, -1)
        cv2.rectangle(img, (x0 + 2 * cell, y0 + 4 * cell), (x0 + 3 * cell, y0 + 5 * cell), (240, 190, 140) if not dark else (110, 80, 40), -1)

    for k in range(10):
        t = 3 if k % 3 == 0 else 1
        cv2.line(img, (x0 + k * cell, y0), (x0 + k * cell, y0 + S), line, t)
        cv2.line(img, (x0, y0 + k * cell), (x0 + S, y0 + k * cell), line, t)

    scale = cell / 46
    for r in range(9):
        for c in range(9):
            if grid[r, c]:
                text = str(grid[r, c])
                (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
                org = (x0 + c * cell + (cell - tw) // 2, y0 + r * cell + (cell + th) // 2)
                cv2.putText(img, text, org, font, scale, ink, 2, cv2.LINE_AA)
            elif notes and (r + c) % 2 == 0:  # nốt ghi chú nhỏ ở góc ô trống
                for i, n in enumerate((1, 5, 9)):
                    org = (x0 + c * cell + 4 + i * cell // 3, y0 + r * cell + 14)
                    cv2.putText(img, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, cell / 190, ink, 1, cv2.LINE_AA)
    return img


# =====================================================================
# Engine thật (load 1 lần cho cả phiên test)
# =====================================================================

@pytest.fixture(scope="session")
def app_module():
    import app as app_module

    app_module.manager.start()
    deadline = time.time() + 180
    while app_module.manager.engine_state in ("idle", "loading") and time.time() < deadline:
        time.sleep(0.5)
    assert app_module.manager.engine_state == "ready", app_module.manager.engine_error
    yield app_module
    app_module.manager.stop()


@pytest.fixture(scope="session")
def engine(app_module):
    return app_module.manager.engine


@pytest.fixture(scope="session")
def client(app_module):
    app_module.app.testing = True
    return app_module.app.test_client()


@pytest.fixture(scope="session")
def photo():
    img = cv2.imread(str(TEST_IMAGE))
    assert img is not None
    return img
