"""
realtime.py

RealtimeTracker: giải sudoku trên luồng camera

Mỗi frame chỉ cần detect 4 góc + phân tích ô trống (vài chục ms).
OCR + solve (chậm) chỉ chạy khi gặp đề mới; lời giải được cache theo session
và "bám" theo lưới qua các frame tiếp theo.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from sudoku_solver import SudokuValidator


@dataclass
class _SessionState:
    occupied: Optional[np.ndarray] = None
    givens: Optional[np.ndarray] = None
    solution: Optional[np.ndarray] = None
    fail_occupied: Optional[np.ndarray] = None
    fail_at: float = 0.0
    seen_at: float = 0.0


class RealtimeTracker:
    def __init__(
        self,
        pipeline,
        max_occupancy_diff: int = 2,
        retry_interval: float = 0.6,
        session_ttl: float = 120.0,
    ):
        """
        Args:
            pipeline: SudokuSolverPipeline
            max_occupancy_diff: Số ô được phép khác (ô trống/có số) để coi là cùng đề
            retry_interval: Thời gian chờ trước khi OCR lại một đề đã giải thất bại
            session_ttl: Xoá cache session không hoạt động sau N giây
        """
        self.pipeline = pipeline
        self.max_occupancy_diff = max_occupancy_diff
        self.retry_interval = retry_interval
        self.session_ttl = session_ttl
        self.sessions: Dict[str, _SessionState] = {}

    def reset(self, session: str) -> None:
        self.sessions.pop(session, None)

    def process(self, session: str, frame: np.ndarray) -> Dict[str, Any]:
        now = time.time()
        self._purge(now)
        state = self.sessions.setdefault(session, _SessionState())
        state.seen_at = now

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        response: Dict[str, Any] = {"found": False, "width": w, "height": h}

        corners, method = self.pipeline.locate(frame, "yolo")
        if corners is None:
            response["status"] = "searching"
            response["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return response

        aligned, matrix = self.pipeline.alignment.align(frame, corners)
        occupied = self.pipeline.ocr_digit.occupancy(aligned)
        response.update(found=True, method=method, corners=corners.round(1).tolist())

        if self._same(state.occupied, occupied) and state.solution is not None:
            response["status"] = "tracking"
        elif self._same(state.fail_occupied, occupied) and now - state.fail_at < self.retry_interval:
            response["status"] = "unsolved"
        else:
            self._solve(state, aligned, now)
            response["status"] = "solved" if state.solution is not None else "unsolved"

        if response["status"] in ("tracking", "solved"):
            geometry = self.pipeline.cell_geometry(matrix)
            cells = []
            for i, g in enumerate(geometry):
                r, c = divmod(i, 9)
                if state.givens[r, c] == 0:
                    cells.append({**g, "d": int(state.solution[r, c])})
            response["cells"] = cells
            response["givens"] = int(np.count_nonzero(state.givens))

        response["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return response

    def _solve(self, state: _SessionState, aligned: np.ndarray, now: float) -> None:
        reading = self.pipeline.ocr_digit.read_grid(aligned)
        grid = reading.digits
        solution, ok = None, False
        if np.count_nonzero(grid) >= 17 and not SudokuValidator.find_conflicts(grid):
            solution, ok = self.pipeline.solver.solve_sudoku(grid.copy())

        if ok:
            state.occupied = reading.occupied
            state.givens = grid
            state.solution = solution
            state.fail_occupied = None
        else:
            state.fail_occupied = reading.occupied
            state.fail_at = now
            state.solution = None
            state.occupied = None

    def _same(self, a: Optional[np.ndarray], b: np.ndarray) -> bool:
        return a is not None and int(np.count_nonzero(a != b)) <= self.max_occupancy_diff

    def _purge(self, now: float) -> None:
        stale = [k for k, s in self.sessions.items() if now - s.seen_at > self.session_ttl]
        for k in stale:
            del self.sessions[k]
