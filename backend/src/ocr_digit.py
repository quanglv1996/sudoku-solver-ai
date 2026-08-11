"""
ocr_digit.py

OCR Digit Recognition using EasyOCR: Nhận diện chữ số từ ô ảnh
Sử dụng EasyOCR để nhận diện chữ số sudoku với độ chính xác cao
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import cv2
import easyocr


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class OCRResult:
    """Kết quả OCR cho một ô"""
    digit: int              # Chữ số nhận diện (0 nếu trống)
    confidence: float       # Độ tin cậy
    text: str              # Text raw


# =====================================================================
# Preprocessing
# =====================================================================

class ImagePreprocessor:
    """
    Tiền xử lý ảnh để chuẩn bị cho OCR
    Trách nhiệm: Chuẩn bị ảnh để xử lý tốt hơn
    """
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Chuyển ảnh sang grayscale nếu cần
        
        Args:
            image: Ảnh input
        
        Returns:
            Ảnh grayscale
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def resize(image: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize ảnh đến kích thước target
        
        Args:
            image: Ảnh input
            target_size: Kích thước target
        
        Returns:
            Ảnh đã resize
        """
        return cv2.resize(image, (target_size, target_size))
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """
        Tăng cường độ tương phản để chữ số rõ hơn
        Sử dụng kết hợp CLAHE + Histogram Equalization
        
        Args:
            image: Ảnh grayscale
        
        Returns:
            Ảnh đã tăng cường tương phản
        """
        # Sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def denoise(image: np.ndarray) -> np.ndarray:
        """
        Giảm nhiễu để OCR chính xác hơn
        
        Args:
            image: Ảnh input
        
        Returns:
            Ảnh đã giảm nhiễu
        """
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised
    
    @staticmethod
    def apply_threshold(image: np.ndarray) -> np.ndarray:
        """
        Áp dụng ngưỡng để làm rõ chữ số (binary)
        
        Args:
            image: Ảnh grayscale
        
        Returns:
            Ảnh binary
        """
        # Otsu's binarization - tự động tìm ngưỡng tối ưu
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """
        Làm sắc các cạnh để OCR tốt hơn
        
        Args:
            image: Ảnh input
        
        Returns:
            Ảnh đã được sharpen
        """
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened


# =====================================================================
# Cell Extraction
# =====================================================================

class GridExtractor:
    """
    Trích xuất các ô từ lưới sudoku
    Trách nhiệm: Chỉ trích xuất ô ảnh từ lưới
    """
    
    def __init__(self, grid_size: int = 9, cell_size: int = 64):
        """
        Khởi tạo extractor
        
        Args:
            grid_size: Kích thước lưới (9x9)
            cell_size: Kích thước mỗi ô (64x64)
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
    
    def extract(self, grid_cells: List[np.ndarray]) -> List[np.ndarray]:
        """
        Tiền xử lý các ô trước khi OCR
        
        Args:
            grid_cells: Danh sách 81 ô ảnh
        
        Returns:
            Danh sách 81 ô ảnh đã xử lý
        """
        processed_cells = []
        
        for cell in grid_cells:
            # Convert to grayscale
            gray = ImagePreprocessor.to_grayscale(cell)
            
            # Enhance contrast mạnh hơn
            enhanced = ImagePreprocessor.enhance_contrast(gray)
            
            # Denoise
            denoised = ImagePreprocessor.denoise(enhanced)
            
            # Sharpen để làm rõ cạnh chữ số
            sharpened = ImagePreprocessor.sharpen(denoised)
            
            # Enhance contrast lần 2 để chữ số rõ hơn
            final = ImagePreprocessor.enhance_contrast(sharpened)
            
            processed_cells.append(final)
        
        return processed_cells


# =====================================================================
# OCR Recognition
# =====================================================================

class OCRDigit:
    """
    OCR Digit Recognition sử dụng EasyOCR
    
    Nhận: ma trận 9x9 ô ảnh
    Trả về: ma trận 9x9 chữ số (0-9, 0 = trống)
    """
    
    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        conf_threshold: float = 0.15
    ):
        """
        Khởi tạo OCR Digit sử dụng EasyOCR
        
        Args:
            lang: Ngôn ngữ ("en" cho tiếng Anh)
            use_gpu: Sử dụng GPU
            conf_threshold: Confidence threshold để chấp nhận kết quả (default 0.15 để nhận diện được 4, 1, 7)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.conf_threshold = conf_threshold
        
        # Khởi tạo EasyOCR reader
        try:
            self.reader = easyocr.Reader([lang], gpu=use_gpu)
        except Exception as e:
            print(f"Warning: Could not initialize GPU, falling back to CPU: {e}")
            self.reader = easyocr.Reader([lang], gpu=False)
    
    def predict(self, grid_cells: List[np.ndarray]) -> np.ndarray:
        """
        Nhận diện chữ số từ 81 ô ảnh
        
        Args:
            grid_cells: Danh sách 81 ô ảnh (9x9)
        
        Returns:
            Ma trận 9x9 chứa các chữ số (0-9, 0 = trống)
        
        Raises:
            ValueError: Nếu số lượng ô không đúng
        """
        if len(grid_cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(grid_cells)}")
        
        # Tiền xử lý các ô
        extractor = GridExtractor()
        processed_cells = extractor.extract(grid_cells)
        
        # Nhận diện chữ số
        digits = self._batch_recognize(processed_cells)
        
        return digits
    
    def _batch_recognize(self, cells: List[np.ndarray]) -> np.ndarray:
        """
        Nhận diện tất cả các ô trong batch
        Chỉ trả về chữ số 0-9 (0 = ô trống, 1-9 = chứa số)
        
        Args:
            cells: Danh sách 81 ô ảnh đã xử lý
        
        Returns:
            Ma trận 9x9 chứa chữ số (0-9 only)
        """
        # Nhận diện batch với chi tiết (để lấy confidence)
        try:
            batch_results = self.reader.readtext_batched(
                cells,
                batch_size=len(cells),
                allowlist='0123456789',  # Allow 0-9
                detail=1,                 # Lấy chi tiết (text, bbox, confidence)
                mag_ratio=2.0             # Tăng magnification ratio để cải thiện OCR cho chữ số nhỏ
            )
        except Exception as e:
            print(f"Warning: Batch recognition failed, falling back to single recognition: {e}")
            batch_results = []
            for cell in cells:
                try:
                    result = self.reader.readtext(
                        cell, 
                        allowlist='0123456789',
                        detail=1,
                        mag_ratio=2.0
                    )
                    batch_results.append(result)
                except:
                    batch_results.append([])
        
        # Chuyển đổi kết quả thành ma trận 9x9
        grid = np.zeros((9, 9), dtype=int)
        
        for idx, res in enumerate(batch_results):
            row = idx // 9
            col = idx % 9
            
            # Nhận diện chữ số từ kết quả (chỉ 0-9)
            digit = self._extract_digit_with_confidence(res, self.conf_threshold)
            grid[row, col] = digit
        
        return grid
    
    def recognize_cells(self, cells: List[np.ndarray]) -> List[OCRResult]:
        """
        Nhận diện các ô riêng lẻ với confidence score
        Chỉ trả về 0-9, reject nếu confidence < threshold
        
        Args:
            cells: Danh sách ô ảnh
        
        Returns:
            Danh sách OCRResult objects (digit 0-9)
        """
        # Tiền xử lý các ô
        extractor = GridExtractor()
        processed_cells = extractor.extract(cells)
        
        # Nhận diện batch với chi tiết
        try:
            batch_results = self.reader.readtext_batched(
                processed_cells,
                batch_size=len(processed_cells),
                allowlist='0123456789',  # Allow 0-9
                detail=1,  # Cần chi tiết để lấy confidence
                mag_ratio=1.2
            )
        except Exception as e:
            print(f"Warning: Batch recognition failed: {e}")
            batch_results = []
        
        results = []
        for res in batch_results:
            digit, confidence, text = self._extract_digit_info(res, self.conf_threshold)
            
            if digit >= 0:  # Valid digit (0-9)
                results.append(OCRResult(digit=digit, confidence=confidence, text=text))
            else:
                results.append(OCRResult(digit=0, confidence=0, text=""))
        
        return results
    
    @staticmethod
    def _extract_digit_info(result_list, conf_threshold: float) -> tuple:
        """
        Trích xuất digit, confidence, text từ kết quả OCR
        
        Args:
            result_list: List chứa tuples [(text, bbox, confidence), ...]
            conf_threshold: Confidence threshold tối thiểu
        
        Returns:
            Tuple (digit, confidence, text) hoặc (-1, 0, "") nếu fail
        """
        try:
            if not isinstance(result_list, list) or len(result_list) == 0:
                return (-1, 0, "")
            
            # Lấy kết quả đầu tiên (confidence cao nhất)
            top_result = result_list[0]
            
            if len(top_result) >= 3:
                text = str(top_result[1]).strip()  # Index 1 là text
                confidence = float(top_result[2])  # Index 2 là confidence
                
                # Chỉ chấp nhận nếu confidence >= threshold
                if confidence < conf_threshold:
                    return (-1, confidence, text)
                
                try:
                    digit = int(text)
                    # Kiểm tra digit là 0-9
                    if 0 <= digit <= 9:
                        return (digit, confidence, text)
                except (ValueError, TypeError):
                    pass
        except (IndexError, TypeError, AttributeError):
            pass
        
        return (-1, 0, "")
    
    @staticmethod
    def _extract_digit(result) -> int:
        """
        Trích xuất chữ số từ kết quả OCR (cũ - chỉ dùng cho recognize_cells)
        
        Args:
            result: Kết quả từ readtext hoặc readtext_batched
        
        Returns:
            Chữ số từ 0-9
        """
        try:
            if isinstance(result, list) and len(result) > 0:
                # Trường hợp list của strings
                text = str(result[0])
                digit = int(text)
                if 0 <= digit <= 9:
                    return digit
            elif isinstance(result, str) and result:
                # Trường hợp string
                digit = int(result)
                if 0 <= digit <= 9:
                    return digit
        except (ValueError, IndexError, TypeError):
            pass
        
        return 0  # Default: ô trống
    
    @staticmethod
    def _extract_digit_with_confidence(result_list, conf_threshold: float) -> int:
        """
        Trích xuất chữ số từ kết quả OCR với kiểm tra confidence
        Có fallback logic: thử top result, nếu không, thử result thứ 2, 3, ...
        
        Args:
            result_list: List chứa tuples [(text, bbox, confidence), ...]
            conf_threshold: Confidence threshold tối thiểu (default 0.15)
        
        Returns:
            Chữ số từ 0-9, hoặc 0 nếu không detect được
        """
        try:
            if not isinstance(result_list, list) or len(result_list) == 0:
                return 0
            
            # Thử từng result từ top đến bottom
            for result in result_list[:5]:  # Thử max 5 results
                try:
                    if len(result) >= 3:
                        text = str(result[1]).strip()  # Index 1 là text
                        confidence = float(result[2])  # Index 2 là confidence
                        
                        # Thử parse digit
                        try:
                            digit = int(text)
                            
                            # Kiểm tra digit là 0-9
                            if 0 <= digit <= 9:
                                # Nếu confidence cao, chấp nhận ngay
                                if confidence >= conf_threshold:
                                    return digit
                                # Nếu confidence thấp nhưng không quá thấp, vẫn chấp nhận
                                # (Fallback untuk những chữ số khó: 1, 4, 7)
                                elif confidence >= 0.08:
                                    return digit
                        except (ValueError, TypeError):
                            pass
                except (IndexError, TypeError, AttributeError):
                    continue
            
            # Không tìm được chữ số nào phù hợp
            return 0
            
        except Exception:
            return 0
    
    def batch_predict(
        self,
        grid_cells_list: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """
        Nhận diện chữ số cho nhiều tấm ảnh
        
        Args:
            grid_cells_list: Danh sách các danh sách 81 ô
        
        Returns:
            Danh sách ma trận 9x9 chứa chữ số
        """
        results = []
        for grid_cells in grid_cells_list:
            result = self.predict(grid_cells)
            results.append(result)
        return results
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Đặt confidence threshold
        
        Args:
            threshold: Giá trị confidence (0.0-1.0)
        """
        self.conf_threshold = threshold
    
    def set_gpu(self, use_gpu: bool) -> None:
        """
        Chuyển đổi giữa GPU/CPU
        
        Args:
            use_gpu: True để dùng GPU, False để dùng CPU
        """
        self.use_gpu = use_gpu
        try:
            self.reader = easyocr.Reader([self.lang], gpu=use_gpu)
        except Exception as e:
            print(f"Warning: Failed to switch to {'GPU' if use_gpu else 'CPU'}: {e}")
            if use_gpu:
                print("Falling back to CPU")
                self.reader = easyocr.Reader([self.lang], gpu=False)
                self.use_gpu = False
