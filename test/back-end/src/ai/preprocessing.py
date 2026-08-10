"""
preprocessing.py

Image preprocessing: alignment and grid extraction
SOLID: Separated concerns into distinct classes
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class PreProcessingResult:
    """Preprocessing result with aligned image and grid cells."""
    aligned_image: np.ndarray      # Aligned sudoku image
    grid_cells: List[np.ndarray]   # 81 cells (9x9), each 64x64 or configurable
    transform_matrix: np.ndarray   # Perspective transform matrix
    cell_size: int                 # Size of each cell


# =====================================================================
# Utilities
# =====================================================================

class AlignmentUtils:
    """
    Utility functions for image alignment.
    Responsibility: Point sorting, perspective transform, visualization
    """
    
    @staticmethod
    def sort_points(points: np.ndarray) -> np.ndarray:
        """
        Sort 4 points to [TL, TR, BR, BL] order.
        
        Args:
            points: 4 points (N, 2)
            
        Returns:
            Sorted points in TL, TR, BR, BL order
        """
        center = np.mean(points, axis=0)
        
        left = points[points[:, 0] < center[0]]
        right = points[points[:, 0] >= center[0]]
        
        if len(left) < 2 or len(right) < 2:
            return points
        
        tl = left[np.argmin(left[:, 1])]
        bl = left[np.argmax(left[:, 1])]
        tr = right[np.argmin(right[:, 1])]
        br = right[np.argmax(right[:, 1])]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)
    
    @staticmethod
    def get_perspective_transform(
        points: np.ndarray,
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Get perspective transform matrix.
        
        Args:
            points: 4 sorted points
            output_size: Output size (width, height)
            
        Returns:
            Perspective transform matrix
        """
        dst_points = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]]
        ], dtype=np.float32)
        
        return cv2.getPerspectiveTransform(points, dst_points)
    
    @staticmethod
    def draw_grid(
        image: np.ndarray,
        cell_size: int = 64,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw 9x9 grid lines on image.
        
        Args:
            image: Input image
            cell_size: Size of each cell
            color: Grid line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with grid lines drawn
        """
        output = image.copy()
        h, w = output.shape[:2]
        
        # Vertical lines
        for i in range(1, 9):
            x = i * cell_size
            cv2.line(output, (x, 0), (x, h), color, thickness)
        
        # Horizontal lines
        for i in range(1, 9):
            y = i * cell_size
            cv2.line(output, (0, y), (w, y), color, thickness)
        
        # Draw 3x3 box lines (thicker)
        for i in range(1, 3):
            x = i * 3 * cell_size
            y = i * 3 * cell_size
            cv2.line(output, (x, 0), (x, h), color, 2)
            cv2.line(output, (0, y), (w, y), color, 2)
        
        return output


# =====================================================================
# Cell Extraction
# =====================================================================

class CellExtractor:
    """
    Extract cells from aligned grid.
    Responsibility: Grid cell extraction only
    """
    
    def __init__(self, cell_size: int = 64, grid_size: int = 9):
        """
        Initialize extractor.
        
        Args:
            cell_size: Size of each cell
            grid_size: Grid size (9x9)
        """
        self.cell_size = cell_size
        self.grid_size = grid_size
    
    def extract(self, aligned_image: np.ndarray) -> List[np.ndarray]:
        """
        Extract 81 cells from aligned grid.
        
        Args:
            aligned_image: Aligned image (576x576 for 9x9)
            
        Returns:
            List of 81 cells, row by row (0-80)
        """
        cells = []
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = row * self.cell_size
                y2 = y1 + self.cell_size
                x1 = col * self.cell_size
                x2 = x1 + self.cell_size
                
                cell = aligned_image[y1:y2, x1:x2]
                cells.append(cell)
        
        return cells


# =====================================================================
# Preprocessing Orchestrator
# =====================================================================

class PreProcessing:
    """
    Align sudoku image and extract 9x9 grid cells.
    Responsibility: Orchestrate preprocessing steps
    """

    def __init__(self, cell_size: int = 64):
        """
        Initialize preprocessing.
        
        Args:
            cell_size: Size of each grid cell (default 64x64)
        """
        self.cell_size = cell_size
        self.grid_size = 9  # 9x9 sudoku grid
        self.output_size = (cell_size * self.grid_size, cell_size * self.grid_size)

    def preprocess(
        self,
        image: np.ndarray,
        points: np.ndarray
    ) -> PreProcessingResult:
        """
        Full preprocessing: align and extract 9x9 grid.
        
        Args:
            image: Input image
            points: 4 corner points
            
        Returns:
            PreProcessingResult with aligned image and 81 cells
        """
        # Sort points
        sorted_points = AlignmentUtils.sort_points(points)
        
        # Get transform matrix
        transform_matrix = AlignmentUtils.get_perspective_transform(sorted_points, self.output_size)
        
        # Align image
        aligned_image = cv2.warpPerspective(image, transform_matrix, self.output_size)
        
        # Extract grid cells
        extractor = CellExtractor(self.cell_size, self.grid_size)
        grid_cells = extractor.extract(aligned_image)
        
        return PreProcessingResult(
            aligned_image=aligned_image,
            grid_cells=grid_cells,
            transform_matrix=transform_matrix,
            cell_size=self.cell_size
        )

    def batch_preprocess(
        self,
        images: List[np.ndarray],
        points_list: List[np.ndarray]
    ) -> List[PreProcessingResult]:
        """
        Process multiple images.
        
        Args:
            images: List of input images
            points_list: List of point arrays
            
        Returns:
            List of PreProcessingResult
        """
        results = []
        for image, points in zip(images, points_list):
            result = self.preprocess(image, points)
            results.append(result)
        return results

    def sort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Sort 4 points to [TL, TR, BR, BL] order.
        Wrapper for AlignmentUtils.sort_points for backward compatibility.
        
        Args:
            points: 4 points (N, 2)
            
        Returns:
            Sorted points in TL, TR, BR, BL order
        """
        return AlignmentUtils.sort_points(points)

    def draw_grid(
        self,
        aligned_image: np.ndarray,
        cell_size: int = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw 9x9 grid lines on image.
        Wrapper for AlignmentUtils.draw_grid for backward compatibility.
        
        Args:
            aligned_image: Aligned image
            cell_size: Size of each cell (uses self.cell_size if None)
            color: Grid line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with grid lines drawn
        """
        if cell_size is None:
            cell_size = self.cell_size
        
        return AlignmentUtils.draw_grid(aligned_image, cell_size, color, thickness)
