"""
ocr_digist.py

OCR digit recognition using EasyOCR
SOLID: Separated concerns into distinct classes
"""

from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np
import easyocr


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class OCRResult:
    """OCR recognition result."""
    digit: int              # Recognized digit (0 if empty)
    confidence: float       # Confidence score
    text: str              # Raw text


# =====================================================================
# Preprocessing
# =====================================================================

class ImagePreprocessor:
    """
    Image preprocessing for OCR.
    Responsibility: Prepare image for processing
    """
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def resize(image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(image, (target_size, target_size))


# =====================================================================
# Cell Extraction
# =====================================================================

class GridExtractor:
    """
    Extract cells from sudoku grid.
    Responsibility: Grid cell extraction only
    """
    
    def __init__(self, grid_size: int = 9, target_size: int = 450):
        """
        Initialize extractor.
        
        Args:
            grid_size: Sudoku grid size (9x9)
            target_size: Target image size (450x450)
        """
        self.grid_size = grid_size
        self.target_size = target_size
        self.cell_size = target_size // grid_size  # 50
    
    def extract(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract cells from image.
        
        Args:
            image: Input image (target_size x target_size)
            
        Returns:
            List of 81 cell images
        """
        cells = []
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell = self._extract_cell(image, r, c)
                cells.append(cell)
        
        return cells
    
    def _extract_cell(self, image: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract single cell from grid."""
        y1 = row * self.cell_size
        y2 = (row + 1) * self.cell_size
        x1 = col * self.cell_size
        x2 = (col + 1) * self.cell_size
        
        cell = image[y1:y2, x1:x2]
        
        # Crop margins (12% to remove grid lines)
        margin = int(self.cell_size * 0.12)
        cell_cropped = cell[margin:-margin, margin:-margin]
        
        return cell_cropped


# =====================================================================
# OCR Recognition
# =====================================================================

class OCRDigist:
    """
    Sudoku digit recognition using EasyOCR.
    
    Responsibility: OCR inference and postprocessing only
    Input: Preprocessed image and cells
    Output: 9x9 grid with recognized digits
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True
    ):
        """
        Initialize OCR recognizer.
        
        Args:
            lang: Language ("en" for English)
            use_gpu: Use GPU for inference
        """
        self.lang = lang
        self.use_gpu = use_gpu
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader([lang], gpu=use_gpu)

    def recognize_grid(
        self,
        image: np.ndarray,
        grid_size: int = 9,
        target_size: int = 450
    ) -> np.ndarray:
        """
        Recognize sudoku grid from image.
        
        Args:
            image: Input image (grayscale or BGR)
            grid_size: Grid size (default 9)
            target_size: Target size (default 450)
            
        Returns:
            9x9 numpy array with recognized digits (0 for empty cells)
        """
        # Preprocess image
        preprocessor = ImagePreprocessor()
        image = preprocessor.to_grayscale(image)
        image = preprocessor.resize(image, target_size)
        
        # Extract cells
        extractor = GridExtractor(grid_size, target_size)
        cells = extractor.extract(image)
        
        # Recognize cells
        grid = self._batch_recognize(cells, grid_size)
        
        return grid

    def _batch_recognize(
        self,
        cells: List[np.ndarray],
        grid_size: int = 9
    ) -> np.ndarray:
        """
        Recognize all cells in batch.
        
        Args:
            cells: List of cell images
            grid_size: Grid size
            
        Returns:
            Grid with recognized digits
        """
        batch_results = self.reader.readtext_batched(
            cells,
            batch_size=len(cells),
            allowlist='123456789',
            detail=0,
            mag_ratio=1.2
        )
        
        # Convert to grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        for idx, res in enumerate(batch_results):
            row = idx // grid_size
            col = idx % grid_size
            
            if len(res) > 0:
                try:
                    digit = int(res[0])
                    if 1 <= digit <= 9:
                        grid[row, col] = digit
                except (ValueError, IndexError):
                    pass
        
        return grid

    def recognize_cells(self, cells: List[np.ndarray]) -> List[OCRResult]:
        """
        Recognize individual cells with confidence scores.
        
        Args:
            cells: List of cell images
            
        Returns:
            List of OCRResult objects
        """
        batch_results = self.reader.readtext_batched(
            cells,
            batch_size=len(cells),
            allowlist='123456789',
            detail=1,
            mag_ratio=1.2
        )
        
        results = []
        for res in batch_results:
            if len(res) > 0:
                text = res[0][1]
                confidence = res[0][2]
                try:
                    digit = int(text)
                    if 1 <= digit <= 9:
                        results.append(OCRResult(digit=digit, confidence=confidence, text=text))
                    else:
                        results.append(OCRResult(digit=0, confidence=0, text=""))
                except ValueError:
                    results.append(OCRResult(digit=0, confidence=0, text=""))
            else:
                results.append(OCRResult(digit=0, confidence=0, text=""))
        
        return results
