"""
sudoku_solve.py

Sudoku solver with backtracking
SOLID: Separated concerns into validation and solving
"""

import numpy as np
from typing import Optional


# =====================================================================
# Validation
# =====================================================================

class SudokuValidator:
    """
    Sudoku validation rules.
    Responsibility: Validate sudoku rules only
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
        Check if placing num at (row, col) is valid.
        
        Args:
            grid: 9x9 sudoku grid
            row: Row index
            col: Column index
            num: Number to check (1-9)
            
        Returns:
            True if valid placement
        """
        # Check row
        if num in grid[row, :]:
            return False
        
        # Check column
        if num in grid[:, col]:
            return False
        
        # Check 3x3 box
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
        Validate input grid format.
        
        Args:
            grid: Input grid
            
        Returns:
            True if valid format
        """
        if grid.shape != (9, 9):
            return False
        
        if not np.all((grid >= 0) & (grid <= 9)):
            return False
        
        return True
    
    @staticmethod
    def get_solution_diff(
        original: np.ndarray,
        solution: np.ndarray
    ) -> np.ndarray:
        """
        Get difference between original and solution.
        
        Args:
            original: Original grid
            solution: Solved grid
            
        Returns:
            Binary grid (1 = filled by solver, 0 = original)
        """
        return (original == 0).astype(int) * (solution != 0).astype(int)


# =====================================================================
# Solver
# =====================================================================

class SudokuSolver:
    """
    Sudoku solver with backtracking.
    Responsibility: Solve sudoku using backtracking algorithm
    """

    def __init__(self):
        """Initialize solver."""
        self.validator = SudokuValidator()

    def solve(self, grid: np.ndarray) -> bool:
        """
        Solve sudoku using backtracking.
        
        Args:
            grid: 9x9 sudoku grid (0 for empty cells)
            
        Returns:
            True if solved, False if unsolvable
        """
        cell = self._find_empty(grid)
        
        if cell is None:
            return True  # All cells filled
        
        row, col = cell
        
        # Try numbers 1-9
        for num in range(1, 10):
            if self.validator.is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                
                if self.solve(grid):
                    return True
                
                # Backtrack
                grid[row, col] = 0
        
        return False

    def solve_sudoku(self, grid: np.ndarray) -> np.ndarray:
        """
        Solve sudoku puzzle.
        
        Args:
            grid: Input 9x9 grid with some numbers filled
            
        Returns:
            Solved 9x9 grid
        """
        # Make a copy to preserve original
        solution = grid.copy()
        
        if self.solve(solution):
            return solution
        else:
            return grid  # Return original if unsolvable

    @staticmethod
    def _find_empty(grid: np.ndarray) -> Optional[tuple]:
        """
        Find next empty cell (0).
        
        Args:
            grid: 9x9 sudoku grid
            
        Returns:
            (row, col) tuple or None if no empty cell
        """
        empty_cells = np.where(grid == 0)
        
        if len(empty_cells[0]) == 0:
            return None
        
        row = empty_cells[0][0]
        col = empty_cells[1][0]
        
        return (row, col)

    def validate_input(self, grid: np.ndarray) -> bool:
        """
        Validate input grid.
        Wrapper for SudokuValidator.validate_grid
        
        Args:
            grid: Input grid
            
        Returns:
            True if valid
        """
        return self.validator.validate_grid(grid)

    def print_grid(self, grid: np.ndarray, highlight: Optional[np.ndarray] = None):
        """
        Print sudoku grid in readable format.
        
        Args:
            grid: 9x9 sudoku grid
            highlight: Optional binary grid to highlight solved cells
        """
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                
                val = grid[i, j]
                marker = ""
                if highlight is not None and highlight[i, j]:
                    marker = "*"  # Solved cell
                
                row_str += f"{val}{marker} "
            
            print(row_str)

    @staticmethod
    def difficulty_score(grid: np.ndarray) -> int:
        """
        Estimate sudoku difficulty (number of filled cells).
        More filled = easier.
        
        Args:
            grid: Input grid
            
        Returns:
            Number of filled cells
        """
        return np.sum(grid != 0)
