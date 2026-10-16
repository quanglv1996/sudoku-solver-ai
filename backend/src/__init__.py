"""
__init__.py

Backend module: Sudoku Solver AI
YOLO (corner detection) + PaddleOCR (digit recognition) + bitmask solver
"""

import sys
from pathlib import Path

# Các module trong thư mục này import lẫn nhau bằng tên tuyệt đối
sys.path.insert(0, str(Path(__file__).parent))

from alignment import Alignment  # noqa: E402
from ocr_digit import OCRDigit, OCRResult, ImagePreprocessor, GridExtractor  # noqa: E402
from sudoku_solver import SudokuSolver, SudokuValidator  # noqa: E402
from point_detection import PointDetection  # noqa: E402
from pipeline import SudokuSolverPipeline  # noqa: E402
from task_manager import TaskManager, Job, JobStatus, Priority  # noqa: E402

__all__ = [
    "Alignment",
    "OCRDigit",
    "OCRResult",
    "ImagePreprocessor",
    "GridExtractor",
    "SudokuSolver",
    "SudokuValidator",
    "PointDetection",
    "SudokuSolverPipeline",
    "TaskManager",
    "Job",
    "JobStatus",
    "Priority",
]
