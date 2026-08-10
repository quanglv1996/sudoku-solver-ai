"""
__init__.py

Backend module: Sudoku Solver AI
Sử dụng EasyOCR cho OCR digit recognition
"""

from .alignment import Alignment
from .preprocessing import Preprocessing, PreprocessingResult
from .ocr_digit import OCRDigit, OCRResult, ImagePreprocessor, GridExtractor
from .sudoku_solver import SudokuSolver, SudokuValidator
from .point_detection import PointDetection
from .pipeline import SudokuSolverPipeline

__all__ = [
    "Alignment",
    "Preprocessing",
    "PreprocessingResult",
    "OCRDigit",
    "OCRResult",
    "ImagePreprocessor",
    "GridExtractor",
    "SudokuSolver",
    "SudokuValidator",
    "PointDetection",
    "SudokuSolverPipeline",
]
