"""
alignment.py

Perspective Transform and Image Alignment
Đảm bảo ảnh sudoku được căn chỉnh thành hình vuông tiêu chuẩn
"""

import cv2
import numpy as np
from typing import Tuple

from grid_geometry import order_corners


class Alignment:
    """
    Alignment: Căn chỉnh ảnh sudoku
    
    Nhận: ảnh gốc + 4 điểm góc
    Trả về: ảnh đã căn chỉnh
    """
    
    def __init__(self, output_size: Tuple[int, int] = (576, 576)):
        """
        Khởi tạo Alignment
        
        Args:
            output_size: Kích thước ảnh đầu ra (width, height)
        """
        self.output_size = output_size
    
    def align(
        self,
        image: np.ndarray,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Căn chỉnh ảnh sudoku dựa trên 4 điểm góc
        
        Args:
            image: Ảnh gốc (H, W, C)
            points: 4 điểm góc đã sắp xếp [TL, TR, BR, BL] (4, 2)
        
        Returns:
            aligned_image: Ảnh đã căn chỉnh
            transform_matrix: Ma trận perspective transform
        
        Raises:
            ValueError: Nếu không đúng 4 điểm
        """
        if len(points) != 4:
            raise ValueError(f"Expected 4 points, got {len(points)}")
        
        # Sắp xếp lại các điểm nếu cần
        points = self._sort_points(points)
        
        # Điểm đích (output vuông chuẩn)
        dst_points = np.array([
            [0, 0],
            [self.output_size[0], 0],
            [self.output_size[0], self.output_size[1]],
            [0, self.output_size[1]]
        ], dtype=np.float32)
        
        # Lấy ma trận perspective transform
        transform_matrix = cv2.getPerspectiveTransform(
            points.astype(np.float32),
            dst_points
        )
        
        # Áp dụng transform
        aligned_image = cv2.warpPerspective(
            image,
            transform_matrix,
            self.output_size
        )
        
        return aligned_image, transform_matrix
    
    @staticmethod
    def _sort_points(points: np.ndarray) -> np.ndarray:
        """Sắp xếp 4 điểm thành [TL, TR, BR, BL]"""
        return order_corners(points)

    def set_output_size(self, size: Tuple[int, int]) -> None:
        """Đặt kích thước output"""
        self.output_size = size
