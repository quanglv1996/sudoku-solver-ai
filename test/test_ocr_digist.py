import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.ocr_digist import OCRDigist, OCRResult


class TestOCRResult(unittest.TestCase):
    """Test OCRResult dataclass"""

    def test_ocr_result_creation(self):
        """Test creating OCRResult instance"""
        result = OCRResult(
            digit=5,
            confidence=0.95,
            text="5"
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.digit, 5)
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.text, "5")

    def test_ocr_result_empty_cell(self):
        """Test OCRResult for empty cell"""
        result = OCRResult(
            digit=0,
            confidence=0,
            text=""
        )

        self.assertEqual(result.digit, 0)
        self.assertEqual(result.confidence, 0)

    def test_ocr_result_all_digits(self):
        """Test OCRResult with all valid digits"""
        for digit in range(1, 10):
            result = OCRResult(
                digit=digit,
                confidence=0.9,
                text=str(digit)
            )
            self.assertEqual(result.digit, digit)


class TestOCRDigist(unittest.TestCase):
    """Test OCRDigist class with EasyOCR"""

    @classmethod
    def setUpClass(cls):
        """Initialize OCR recognizer once for all tests"""
        try:
            cls.ocr = OCRDigist(lang="en", use_gpu=False)
        except Exception as e:
            cls.skipTest(f"Could not initialize OCRDigist: {e}")

    def test_ocr_initialization(self):
        """Test OCRDigist initialization"""
        self.assertIsNotNone(self.ocr)
        self.assertIsNotNone(self.ocr.reader)
        self.assertEqual(self.ocr.grid_size, 9)
        self.assertEqual(self.ocr.target_size, 450)

    def test_recognize_black_image(self):
        """Test recognize with black image"""
        black_image = np.zeros((450, 450), dtype=np.uint8)
        grid = self.ocr.recognize(black_image)
        
        self.assertIsNotNone(grid)
        self.assertEqual(grid.shape, (9, 9))

    def test_recognize_output_shape(self):
        """Test that output is always 9x9"""
        random_image = np.random.randint(0, 256, (450, 450), dtype=np.uint8)
        grid = self.ocr.recognize(random_image)
        
        self.assertEqual(grid.shape, (9, 9))
        self.assertEqual(grid.dtype, int)

    def test_recognize_with_bgr_image(self):
        """Test recognize with BGR image (not grayscale)"""
        bgr_image = np.random.randint(0, 256, (450, 450, 3), dtype=np.uint8)
        grid = self.ocr.recognize(bgr_image)
        
        self.assertEqual(grid.shape, (9, 9))

    def test_grid_values_valid(self):
        """Test that all grid values are 0-9"""
        random_image = np.random.randint(0, 256, (450, 450), dtype=np.uint8)
        grid = self.ocr.recognize(random_image)
        
        self.assertTrue(np.all(grid >= 0))
        self.assertTrue(np.all(grid <= 9))



if __name__ == "__main__":
    unittest.main()
