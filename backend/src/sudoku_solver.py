"""
sudoku_solver.py

Sudoku Solver: Giải ma trận sudoku
Bitmask + MRV (chọn ô ít ứng viên nhất) → giải mọi đề 9x9 trong vài ms.
Kiểm tra đề trước khi giải để OCR sai (trùng số) không làm solver treo.
"""

from typing import List, Optional, Tuple

import numpy as np

ALL_DIGITS = 0x3FE  # bit 1..9


def _box(row: int, col: int) -> int:
    return (row // 3) * 3 + col // 3


class SudokuValidator:
    """
    Validation: Kiểm tra tính hợp lệ của sudoku
    """

    GRID_SIZE = 9
    BOX_SIZE = 3

    @staticmethod
    def is_valid_placement(grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Kiểm tra xem đặt số tại (row, col) có hợp lệ không"""
        if num in grid[row, :] or num in grid[:, col]:
            return False
        br, bc = (row // 3) * 3, (col // 3) * 3
        return num not in grid[br:br + 3, bc:bc + 3]

    @staticmethod
    def validate_grid(grid: np.ndarray) -> bool:
        """Kiểm tra định dạng ma trận đầu vào"""
        return grid.shape == (9, 9) and bool(np.all((grid >= 0) & (grid <= 9)))

    @staticmethod
    def find_conflicts(grid: np.ndarray) -> List[Tuple[int, int]]:
        """
        Tìm các ô đề bài bị trùng (cùng hàng / cột / khối 3x3)

        Returns:
            Danh sách (row, col) của các ô xung đột
        """
        conflicts = set()
        for r in range(9):
            for c in range(9):
                v = grid[r, c]
                if v == 0:
                    continue
                for r2 in range(9):
                    for c2 in range(9):
                        if (r2, c2) == (r, c) or grid[r2, c2] != v:
                            continue
                        if r2 == r or c2 == c or _box(r2, c2) == _box(r, c):
                            conflicts.add((r, c))
        return sorted(conflicts)


class SudokuSolver:
    """
    Sudoku Solver: Giải sudoku bằng backtracking bitmask + MRV

    Nhận: ma trận sudoku không hoàn chỉnh (0 = trống)
    Trả về: ma trận sudoku đã giải
    """

    def __init__(self, max_nodes: int = 200_000):
        """
        Args:
            max_nodes: Giới hạn số bước tìm kiếm (chống treo với đề lỗi)
        """
        self.validator = SudokuValidator()
        self.max_nodes = max_nodes

    def solve_sudoku(self, grid: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Giải ma trận sudoku

        Args:
            grid: Ma trận input 9x9 với một số ô đã điền

        Returns:
            (solution, success): Ma trận đã giải và trạng thái thành công
        """
        if not self.validator.validate_grid(grid):
            raise ValueError("Invalid grid format")

        solutions = self._search(grid, limit=1)
        if solutions:
            return solutions[0], True
        return grid.copy(), False

    def count_solutions(self, grid: np.ndarray, limit: int = 2) -> int:
        """Đếm số lời giải (dừng ở `limit`) - dùng để kiểm tra đề có duy nhất nghiệm"""
        return len(self._search(grid, limit=limit))

    def solve(self, grid: np.ndarray) -> bool:
        """Giải tại chỗ (tương thích API cũ)"""
        solution, ok = self.solve_sudoku(grid)
        if ok:
            grid[:, :] = solution
        return ok

    def _search(self, grid: np.ndarray, limit: int) -> List[np.ndarray]:
        cells = [int(v) for v in grid.flatten()]
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9

        for i, v in enumerate(cells):
            if v == 0:
                continue
            r, c = divmod(i, 9)
            bit = 1 << v
            b = _box(r, c)
            if rows[r] & bit or cols[c] & bit or boxes[b] & bit:
                return []  # đề bài mâu thuẫn
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

        empties = [i for i, v in enumerate(cells) if v == 0]
        solutions: List[np.ndarray] = []
        nodes = [0]

        def backtrack() -> bool:
            nodes[0] += 1
            if nodes[0] > self.max_nodes:
                return True  # dừng tìm kiếm

            best_idx, best_mask, best_count = -1, 0, 10
            for k, i in enumerate(empties):
                if cells[i]:
                    continue
                r, c = divmod(i, 9)
                mask = ALL_DIGITS & ~(rows[r] | cols[c] | boxes[_box(r, c)])
                count = bin(mask).count("1")
                if count < best_count:
                    best_idx, best_mask, best_count = i, mask, count
                    if count <= 1:
                        break

            if best_idx < 0:
                solutions.append(np.array(cells, dtype=int).reshape(9, 9))
                return len(solutions) >= limit
            if best_count == 0:
                return False

            r, c = divmod(best_idx, 9)
            b = _box(r, c)
            mask = best_mask
            while mask:
                bit = mask & -mask
                mask ^= bit
                cells[best_idx] = bit.bit_length() - 1
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit
                if backtrack():
                    return True
                rows[r] ^= bit
                cols[c] ^= bit
                boxes[b] ^= bit
            cells[best_idx] = 0
            return False

        backtrack()
        return solutions

    @staticmethod
    def _find_empty(grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Tìm ô trống tiếp theo (giá trị 0)"""
        empty = np.argwhere(grid == 0)
        return (int(empty[0][0]), int(empty[0][1])) if len(empty) else None

    def print_grid(self, grid: np.ndarray) -> None:
        """In ma trận sudoku dưới dạng dễ đọc"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                row_str += f"{int(grid[i, j])} "
            print(row_str)

    def get_difficulty_score(self, grid: np.ndarray) -> int:
        """Số ô đã điền (càng ít càng khó)"""
        return int(np.sum(grid != 0))
