"""
sudoku_solver.py

Sudoku Solver: Giải ma trận sudoku
"""

import numpy as np
from typing import Optional, Tuple


class SudokuValidator:
    """
    Validation: Kiểm tra tính hợp lệ của sudoku
    """
    
    GRID_SIZE = 9
    BOX_SIZE = 3
    
    @staticmethod
    def is_valid_placement(
        grid: np.ndarray,
        row: int,
        col: int,
        num: int
    ) -> bool:
        """
        Kiểm tra xem đặt số tại (row, col) có hợp lệ không
        
        Args:
            grid: Ma trận sudoku 9x9
            row: Hàng
            col: Cột
            num: Số cần kiểm tra (1-9)
        
        Returns:
            True nếu hợp lệ
        """
        # Kiểm tra hàng
        if num in grid[row, :]:
            return False
        
        # Kiểm tra cột
        if num in grid[:, col]:
            return False
        
        # Kiểm tra ô 3x3
        box_row = (row // SudokuValidator.BOX_SIZE) * SudokuValidator.BOX_SIZE
        box_col = (col // SudokuValidator.BOX_SIZE) * SudokuValidator.BOX_SIZE
        box = grid[box_row:box_row+SudokuValidator.BOX_SIZE, 
                   box_col:box_col+SudokuValidator.BOX_SIZE]
        
        if num in box:
            return False
        
        return True
    
    @staticmethod
    def validate_grid(grid: np.ndarray) -> bool:
        """
        Kiểm tra định dạng ma trận đầu vào
        
        Args:
            grid: Ma trận cần kiểm tra
        
        Returns:
            True nếu hợp lệ
        """
        if grid.shape != (9, 9):
            return False
        
        if not np.all((grid >= 0) & (grid <= 9)):
            return False
        
        return True


class SudokuSolver:
    """
    Sudoku Solver: Giải sudoku bằng backtracking
    
    Nhận: ma trận sudoku không hoàn chỉnh (0 = trống)
    Trả về: ma trận sudoku đã giải
    """
    
    def __init__(self):
        """Khởi tạo solver"""
        self.validator = SudokuValidator()
    
    def solve(self, grid: np.ndarray) -> bool:
        """
        Giải sudoku bằng backtracking
        
        Args:
            grid: Ma trận sudoku 9x9 (0 = trống)
        
        Returns:
            True nếu giải được, False nếu không có lời giải
        """
        cell = self._find_empty(grid)
        
        if cell is None:
            return True  # Tất cả ô đã được điền
        
        row, col = cell
        
        # Thử các số 1-9
        for num in range(1, 10):
            if self.validator.is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                
                if self.solve(grid):
                    return True
                
                # Backtrack
                grid[row, col] = 0
        
        return False
    
    def solve_sudoku(self, grid: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Giải ma trận sudoku
        
        Args:
            grid: Ma trận input 9x9 với một số ô đã điền
        
        Returns:
            (solution, success): Ma trận đã giải và trạng thái thành công
        """
        # Kiểm tra input
        if not self.validator.validate_grid(grid):
            raise ValueError("Invalid grid format")
        
        # Tạo bản sao để không sửa đổi original
        solution = grid.copy()
        
        if self.solve(solution):
            return solution, True
        else:
            return grid.copy(), False
    
    @staticmethod
    def _find_empty(grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Tìm ô trống tiếp theo (giá trị 0)
        
        Args:
            grid: Ma trận sudoku 9x9
        
        Returns:
            Tuple (row, col) hoặc None nếu không có ô trống
        """
        empty_cells = np.where(grid == 0)
        
        if len(empty_cells[0]) == 0:
            return None
        
        row = empty_cells[0][0]
        col = empty_cells[1][0]
        
        return (row, col)
    
    def print_grid(self, grid: np.ndarray) -> None:
        """
        In ma trận sudoku dưới dạng dễ đọc
        
        Args:
            grid: Ma trận sudoku 9x9
        """
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                
                val = int(grid[i, j])
                row_str += f"{val} "
            
            print(row_str)
    
    def get_difficulty_score(self, grid: np.ndarray) -> int:
        """
        Ước tính độ khó của sudoku (số ô đã điền)
        Số ô điền càng nhiều, độ khó càng thấp
        
        Args:
            grid: Ma trận sudoku
        
        Returns:
            Số ô đã điền
        """
        return np.sum(grid != 0)
