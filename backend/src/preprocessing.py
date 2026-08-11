"""
preprocessing.py

Preprocessing: Cắt ảnh thành ma trận 9x9 ô
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessingResult:
    """Kết quả preprocessing"""
    aligned_image: np.ndarray      # Ảnh đã căn chỉnh
    grid_cells: List[np.ndarray]   # 81 ô (9x9), mỗi ô 64x64
    cell_size: int                 # Kích thước mỗi ô


class Preprocessing:
    """
    Preprocessing: Cắt ảnh căn chỉnh thành 9x9 ô
    
    Nhận: ảnh căn chỉnh
    Trả về: ma trận 9x9 ô ảnh
    """
    
    def __init__(self, cell_size: int = 64, grid_size: int = 9, margin: int = 4):
        """
        Khởi tạo Preprocessing
        
        Args:
            cell_size: Kích thước mỗi ô (mặc định 64x64)
            grid_size: Kích thước lưới (9x9 cho sudoku)
            margin: Margin bên trong cell để tránh viền (pixel)
        """
        self.cell_size = cell_size
        self.grid_size = grid_size
        self.margin = margin  # 🔍 Margin để tránh viền
        self.output_size = (cell_size * grid_size, cell_size * grid_size)
    
    def crop(self, aligned_image: np.ndarray) -> PreprocessingResult:
        """
        Cắt ảnh căn chỉnh thành 81 ô (9x9)
        Cắt sâu vào bên trong để tránh viền bo
        
        Args:
            aligned_image: Ảnh đã căn chỉnh (576x576 hoặc cell_size*9)
        
        Returns:
            PreprocessingResult chứa:
            - aligned_image: ảnh đã căn chỉnh
            - grid_cells: danh sách 81 ô, sắp xếp hàng theo hàng
            - cell_size: kích thước mỗi ô
        
        Raises:
            ValueError: Nếu ảnh không đúng kích thước
        """
        h, w = aligned_image.shape[:2]
        expected_size = self.cell_size * self.grid_size
        
        if h != expected_size or w != expected_size:
            raise ValueError(
                f"Expected image size {expected_size}x{expected_size}, "
                f"got {h}x{w}"
            )
        
        cells = []
        
        # Cắt thành 9x9 ô
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Tính toán vị trí với margin để tránh viền
                y1 = row * self.cell_size + self.margin
                y2 = y1 + self.cell_size - 2 * self.margin
                x1 = col * self.cell_size + self.margin
                x2 = x1 + self.cell_size - 2 * self.margin
                
                # Cắt ô bên trong, tránh viền
                cell = aligned_image[y1:y2, x1:x2]
                
                # Resize lại về kích thước đầy đủ (64x64)
                cell_resized = cv2.resize(cell, (self.cell_size, self.cell_size))
                
                cells.append(cell_resized)
        
        return PreprocessingResult(
            aligned_image=aligned_image,
            grid_cells=cells,
            cell_size=self.cell_size
        )
    
    def get_cell_at(self, grid_cells: List[np.ndarray], row: int, col: int) -> np.ndarray:
        """
        Lấy ô tại vị trí (row, col) trong lưới 9x9
        
        Args:
            grid_cells: Danh sách 81 ô
            row: Hàng (0-8)
            col: Cột (0-8)
        
        Returns:
            Ảnh ô tại vị trí đó
        """
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            raise IndexError(f"Grid index out of range: ({row}, {col})")
        
        idx = row * self.grid_size + col
        return grid_cells[idx]
    
    def reshape_cells_to_grid(self, grid_cells: List[np.ndarray]) -> np.ndarray:
        """
        Chuyển danh sách 81 ô thành lưới 2D để dễ thao tác
        
        Args:
            grid_cells: Danh sách 81 ô
        
        Returns:
            Ma trận kích thước (9, 9) chứa các ô
        """
        grid = np.empty((self.grid_size, self.grid_size), dtype=object)
        
        for idx, cell in enumerate(grid_cells):
            row = idx // self.grid_size
            col = idx % self.grid_size
            grid[row, col] = cell
        
        return grid
