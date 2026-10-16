"""
test_core.py - test nhanh, KHÔNG cần model: solver, hình học, TaskManager
"""

import threading
import time

import numpy as np
import pytest

from conftest import PUZZLE_CLASSIC, PUZZLE_PHOTO, grid_of, render_screen
from grid_geometry import detect_grid_contour, is_light_on_dark, order_corners, quad_is_valid
from sudoku_solver import SudokuSolver, SudokuValidator
from task_manager import JobStatus, Priority, QueueFullError, TaskManager

HARD = "800000000003600000070090200050007000000045700000100030001000068008500010090000400"


def is_valid_solution(sol: np.ndarray, puzzle: np.ndarray) -> bool:
    full = set(range(1, 10))
    rows = all(set(sol[r]) == full for r in range(9))
    cols = all(set(sol[:, c]) == full for c in range(9))
    boxes = all(set(sol[r:r + 3, c:c + 3].flatten()) == full for r in (0, 3, 6) for c in (0, 3, 6))
    keeps = np.all((puzzle == 0) | (puzzle == sol))
    return rows and cols and boxes and bool(keeps)


# =====================================================================
# Solver
# =====================================================================

@pytest.mark.parametrize("puzzle", [PUZZLE_PHOTO, PUZZLE_CLASSIC, HARD])
def test_solver_solves(puzzle):
    grid = grid_of(puzzle)
    sol, ok = SudokuSolver().solve_sudoku(grid)
    assert ok and is_valid_solution(sol, grid)
    assert np.array_equal(grid, grid_of(puzzle)), "không được sửa đề gốc"


def test_solver_hard_is_fast():
    t = time.perf_counter()
    SudokuSolver().solve_sudoku(grid_of(HARD))
    assert time.perf_counter() - t < 2.0


def test_solver_rejects_conflicting_givens():
    grid = grid_of(PUZZLE_CLASSIC)
    grid[0, 2] = 5  # trùng với ô (0,0)
    assert SudokuValidator.find_conflicts(grid) == [(0, 0), (0, 2)]
    _, ok = SudokuSolver().solve_sudoku(grid)
    assert not ok


def test_solver_unsolvable_terminates():
    # Không có số trùng nhưng vô nghiệm: ô (0,8) không còn ứng viên
    grid = np.zeros((9, 9), int)
    grid[0, :8] = [1, 2, 3, 4, 5, 6, 7, 8]
    grid[1, 8] = 9
    t = time.perf_counter()
    _, ok = SudokuSolver().solve_sudoku(grid)
    assert not ok and time.perf_counter() - t < 2.0


def test_solver_empty_grid_and_uniqueness():
    s = SudokuSolver()
    sol, ok = s.solve_sudoku(np.zeros((9, 9), int))
    assert ok and is_valid_solution(sol, np.zeros((9, 9), int))
    assert s.count_solutions(np.zeros((9, 9), int)) == 2
    assert s.count_solutions(grid_of(PUZZLE_CLASSIC)) == 1


def test_solver_invalid_shape():
    with pytest.raises(ValueError):
        SudokuSolver().solve_sudoku(np.zeros((8, 9), int))


# =====================================================================
# Geometry
# =====================================================================

def test_order_corners_any_input_order_and_rotation():
    quad = np.array([[100, 100], [300, 110], [310, 300], [90, 290]], np.float32)
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert np.allclose(order_corners(quad[perm]), quad)
    # xoay ~40°: cách tách trái/phải cũ sẽ hỏng
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    rot = (quad - 200) @ np.array([[c, -s], [s, c]]).T + 200
    ordered = order_corners(rot[[2, 0, 3, 1]])
    assert len({tuple(p) for p in ordered.round(3)}) == 4


def test_quad_validation():
    shape = (480, 640)
    assert quad_is_valid([[100, 70], [478, 66], [485, 446], [97, 446]], shape)
    assert not quad_is_valid([[0, 0], [10, 0], [10, 10], [0, 10]], shape)          # quá nhỏ
    assert not quad_is_valid([[100, 100], [400, 100], [150, 150], [100, 400]], shape)  # lõm


@pytest.mark.parametrize("dark", [False, True])
def test_contour_detects_screen_grid(dark):
    img = render_screen(grid_of(PUZZLE_CLASSIC), dark=dark, highlight=True)
    quad = detect_grid_contour(img)
    assert quad is not None
    assert np.abs(quad - [[700, 200], [1240, 200], [1240, 740], [700, 740]]).max() < 6


def test_dark_mode_detection():
    import cv2
    light = cv2.cvtColor(render_screen(grid_of(PUZZLE_CLASSIC)), cv2.COLOR_BGR2GRAY)
    dark = cv2.cvtColor(render_screen(grid_of(PUZZLE_CLASSIC), dark=True), cv2.COLOR_BGR2GRAY)
    assert not is_light_on_dark(light)
    assert is_light_on_dark(dark)
    # giấy chụp thiếu sáng: nền tối nhưng chữ còn tối hơn → KHÔNG phải dark mode
    dim_paper = (light.astype(float) * 0.3).astype(np.uint8)
    assert not is_light_on_dark(dim_paper)


def test_contour_none_on_blank():
    assert detect_grid_contour(np.full((600, 800, 3), 200, np.uint8)) is None


# =====================================================================
# TaskManager (engine giả)
# =====================================================================

def make_manager(handler_delay=0.0, max_queue=64, factory=None):
    order = []
    gate = threading.Event()
    gate.set()

    def handler(job, engine):
        gate.wait(5)
        time.sleep(handler_delay)
        order.append(job.payload.get("name"))
        job.set_stage("work")
        if job.payload.get("boom"):
            raise RuntimeError("boom")
        return {"name": job.payload.get("name"), "engine": engine}

    m = TaskManager(factory or (lambda: "ENGINE"), max_queue=max_queue)
    for kind in ("photo", "realtime", "screen"):
        m.register(kind, handler)
    m.start()
    deadline = time.time() + 5
    while m.engine_state not in ("ready", "error") and time.time() < deadline:
        time.sleep(0.01)
    return m, order, gate


def test_manager_runs_job():
    m, _, _ = make_manager()
    job = m.submit("photo", {"name": "a"})
    assert job.done_event.wait(5)
    assert job.status == JobStatus.DONE and job.result == {"name": "a", "engine": "ENGINE"}
    assert job.payload == {}, "ảnh đầu vào phải được giải phóng"
    m.stop()


def test_manager_priority_order():
    m, order, gate = make_manager()
    gate.clear()
    blocker = m.submit("photo", {"name": "blocker"})
    time.sleep(0.1)  # worker đang giữ blocker
    jobs = [
        m.submit("photo", {"name": "photo"}, priority=Priority.PHOTO),
        m.submit("screen", {"name": "screen"}, priority=Priority.SCREEN),
        m.submit("realtime", {"name": "rt"}, priority=Priority.REALTIME, session="s"),
    ]
    assert m.queue_position(jobs[0]) == 3
    gate.set()
    for j in [blocker, *jobs]:
        j.done_event.wait(5)
    assert order == ["blocker", "rt", "screen", "photo"]
    m.stop()


def test_manager_realtime_latest_wins():
    m, order, gate = make_manager()
    gate.clear()
    m.submit("photo", {"name": "blocker"})
    time.sleep(0.1)
    frames = [m.submit("realtime", {"name": f"f{i}"}, priority=Priority.REALTIME, session="cam", latest_only=True) for i in range(5)]
    other = m.submit("realtime", {"name": "other"}, priority=Priority.REALTIME, session="cam2", latest_only=True)
    gate.set()
    for j in [*frames, other]:
        j.done_event.wait(5)
    assert [j.status for j in frames] == [JobStatus.DROPPED] * 4 + [JobStatus.DONE]
    assert other.status == JobStatus.DONE, "session khác không bị ảnh hưởng"
    assert "f0" not in order and "f4" in order
    m.stop()


def test_manager_failure_and_cancel():
    m, _, gate = make_manager()
    bad = m.submit("photo", {"name": "bad", "boom": True})
    bad.done_event.wait(5)
    assert bad.status == JobStatus.FAILED and "boom" in bad.error

    gate.clear()
    m.submit("photo", {"name": "blocker"})
    time.sleep(0.1)
    queued = m.submit("photo", {"name": "queued"})
    assert m.cancel(queued.id)
    assert queued.status == JobStatus.CANCELLED
    gate.set()
    time.sleep(0.2)
    assert not m.cancel(queued.id), "không huỷ được job đã kết thúc"
    m.stop()


def test_manager_queue_full():
    m, _, gate = make_manager(max_queue=2)
    gate.clear()
    m.submit("photo", {"name": "blocker"})
    time.sleep(0.1)
    m.submit("photo", {"name": "q1"})
    m.submit("photo", {"name": "q2"})
    with pytest.raises(QueueFullError):
        m.submit("photo", {"name": "q3"})
    # realtime gộp theo session nên không bị giới hạn
    m.submit("realtime", {"name": "rt"}, session="s", latest_only=True)
    gate.set()
    m.stop()


def test_manager_engine_load_error():
    def broken():
        raise RuntimeError("no gpu")

    m, _, _ = make_manager(factory=broken)
    assert m.engine_state == "error" and "no gpu" in m.engine_error
    with pytest.raises(RuntimeError):
        m.submit("photo", {"name": "x"})


def test_manager_jobs_wait_while_loading():
    loaded = threading.Event()

    def slow():
        loaded.wait(5)
        return "ENGINE"

    m = TaskManager(slow)
    m.register("photo", lambda job, engine: {"ok": engine})
    m.start()
    job = m.submit("photo", {})
    time.sleep(0.2)
    assert job.status == JobStatus.QUEUED and m.engine_state == "loading"
    loaded.set()
    assert job.done_event.wait(5) and job.result == {"ok": "ENGINE"}
    m.stop()
