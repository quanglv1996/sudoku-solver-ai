"""
ocr_digit.py

OCR Digit Recognition using PaddleOCR 3.x: Nhận diện chữ số từ ô ảnh

Flow cho 81 ô:
1. CellAnalyzer: phân tích "mực" trong ô → biết ô trống hay có số (rất nhanh,
   loại bỏ viền lưới và nốt ghi chú nhỏ), đồng thời lấy bbox của chữ số.
2. Chỉ những ô có số mới đưa vào model TextRecognition (1 batch duy nhất).
3. Ô nào đọc không ra chữ số → OCR lại trên ảnh nhị phân sạch.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from grid_geometry import is_light_on_dark


# Các ký tự hay bị đọc nhầm thành chữ cái
CHAR_FIXES = {
    "I": "1", "l": "1", "|": "1", "!": "1", "i": "1", "]": "1", "[": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "B": "8",
    "g": "9", "q": "9",
}


# =====================================================================
# Data Classes
# =====================================================================

@dataclass
class OCRResult:
    """Kết quả OCR cho một ô"""
    digit: int              # Chữ số nhận diện (0 nếu trống)
    confidence: float       # Độ tin cậy
    text: str               # Text raw


@dataclass
class GridReading:
    """Kết quả đọc toàn bộ lưới"""
    digits: np.ndarray                      # 9x9, 0 = trống
    confidence: np.ndarray                  # 9x9
    occupied: np.ndarray                    # 9x9 bool - ô có mực (chữ số)
    uncertain: List[Tuple[int, int]] = field(default_factory=list)


# =====================================================================
# Preprocessing
# =====================================================================

class ImagePreprocessor:
    """Tiền xử lý ảnh để chuẩn bị cho OCR"""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def normalize_polarity(gray: np.ndarray) -> np.ndarray:
        """Đảm bảo chữ tối trên nền sáng (hỗ trợ giao diện dark mode)"""
        return 255 - gray if is_light_on_dark(gray) else gray

    @staticmethod
    def flatten_illumination(gray: np.ndarray) -> np.ndarray:
        """
        Cân bằng sáng: chia cho nền ước lượng (phép đóng xoá nét chữ mảnh).
        Ảnh chụp thiếu sáng / tối góc / bóng đổ đều về nền ~230 → ngưỡng
        tương phản cố định của CellAnalyzer dùng được cho mọi điều kiện.
        """
        k = max(3, gray.shape[0] // 27) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
        return cv2.divide(gray, np.maximum(bg, 1), scale=230)

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def apply_threshold(image: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary


# =====================================================================
# Cell Analysis
# =====================================================================

class CellAnalyzer:
    """
    Phân tích một ô: có chữ số hay không, và vị trí chữ số

    Dựa trên độ chênh lệch so với nền (median) thay vì Otsu, vì Otsu luôn
    tách được 2 lớp kể cả với ô trống.
    """

    def __init__(
        self,
        border_ratio: float = 0.08,
        min_height_ratio: float = 0.35,
        min_area_ratio: float = 0.015,
        min_contrast: int = 28,
    ):
        self.border_ratio = border_ratio
        self.min_height_ratio = min_height_ratio
        self.min_area_ratio = min_area_ratio
        self.min_contrast = min_contrast

    def ink_mask(self, gray: np.ndarray) -> np.ndarray:
        bg = float(np.median(gray))
        darkest = float(np.percentile(gray, 1))
        thr = max(self.min_contrast, 0.45 * (bg - darkest))
        mask = ((bg - gray.astype(np.float32)) > thr).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def analyze(self, gray_cell: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Args:
            gray_cell: ô grayscale (chữ tối, nền sáng), gồm cả viền

        Returns:
            bbox (x1, y1, x2, y2) của chữ số trong ô, hoặc None nếu ô trống
        """
        h, w = gray_cell.shape[:2]
        by, bx = int(h * self.border_ratio), int(w * self.border_ratio)
        inner = gray_cell[by:h - by, bx:w - bx]
        ih, iw = inner.shape[:2]

        mask = self.ink_mask(inner)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        boxes = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            cx, cy = centroids[i]
            # Bỏ đường kẻ lưới còn sót (chạm biên, rất dài và mảnh)
            touches = x <= 0 or y <= 0 or x + cw >= iw or y + ch >= ih
            if touches and (cw > 0.8 * iw or ch > 0.8 * ih or area < 0.03 * ih * iw):
                continue
            if ch < self.min_height_ratio * ih or area < self.min_area_ratio * ih * iw:
                continue  # nhiễu hoặc nốt ghi chú nhỏ
            if not (0.15 * iw < cx < 0.85 * iw and 0.15 * ih < cy < 0.85 * ih):
                continue  # không nằm ở giữa ô
            boxes.append((x, y, x + cw, y + ch))

        if not boxes:
            return None

        x1 = min(b[0] for b in boxes) + bx
        y1 = min(b[1] for b in boxes) + by
        x2 = max(b[2] for b in boxes) + bx
        y2 = max(b[3] for b in boxes) + by
        return x1, y1, x2, y2


# =====================================================================
# Cell Extraction (giữ tên cũ để tương thích)
# =====================================================================

class GridExtractor:
    """Cắt các ô từ ảnh lưới đã căn chỉnh"""

    def __init__(self, grid_size: int = 9, cell_size: int = 64):
        self.grid_size = grid_size
        self.cell_size = cell_size

    def split(self, aligned_gray: np.ndarray) -> List[np.ndarray]:
        h, w = aligned_gray.shape[:2]
        cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                y1, y2 = r * h // 9, (r + 1) * h // 9
                x1, x2 = c * w // 9, (c + 1) * w // 9
                cells.append(aligned_gray[y1:y2, x1:x2])
        return cells

    def extract(self, grid_cells: List[np.ndarray]) -> List[np.ndarray]:
        """Chuyển các ô sang grayscale chuẩn hóa"""
        return [ImagePreprocessor.to_grayscale(c) for c in grid_cells]


# =====================================================================
# OCR Recognition
# =====================================================================

class OCRDigit:
    """
    OCR Digit Recognition sử dụng PaddleOCR TextRecognition

    Nhận: ảnh lưới đã căn chỉnh (hoặc 81 ô ảnh)
    Trả về: ma trận 9x9 chữ số (0 = trống)
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        conf_threshold: float = 0.5,
        model_name: str = "PP-OCRv6_medium_rec",
        verbose: bool = False,
    ):
        """
        Args:
            lang: (giữ cho tương thích) ngôn ngữ
            use_gpu: Sử dụng GPU
            conf_threshold: Ngưỡng tin cậy chấp nhận chữ số
            model_name: Tên model recognition của PaddleOCR
            verbose: In log chi tiết
        """
        from paddleocr import TextRecognition

        self.lang = lang
        self.conf_threshold = conf_threshold
        self.verbose = verbose
        self.analyzer = CellAnalyzer()

        self.use_gpu = use_gpu
        device = "gpu:0" if use_gpu else "cpu"
        try:
            self.reader = TextRecognition(model_name=model_name, device=device)
        except Exception as e:
            if not use_gpu:
                raise
            print(f"⚠️  PaddleOCR GPU init failed ({e}), falling back to CPU")
            self.use_gpu = False
            self.reader = TextRecognition(model_name=model_name, device="cpu")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def read_grid(self, aligned_image: np.ndarray) -> GridReading:
        """
        Đọc toàn bộ lưới từ ảnh đã căn chỉnh (vuông, chứa đúng lưới 9x9)
        """
        return self._read_cells(GridExtractor().split(self._prepare(aligned_image)))

    def occupancy(self, aligned_image: np.ndarray) -> np.ndarray:
        """Chỉ tính ô nào có số (không chạy OCR) - dùng cho realtime cache"""
        cells = GridExtractor().split(self._prepare(aligned_image))
        occ = np.array([self.analyzer.analyze(c) is not None for c in cells])
        return occ.reshape(9, 9)

    @staticmethod
    def _prepare(aligned_image: np.ndarray) -> np.ndarray:
        """Grayscale, chữ tối trên nền sáng, cân bằng sáng toàn lưới"""
        gray = ImagePreprocessor.normalize_polarity(ImagePreprocessor.to_grayscale(aligned_image))
        return ImagePreprocessor.flatten_illumination(gray)

    def predict(self, grid_cells: List[np.ndarray]) -> np.ndarray:
        """
        Nhận diện chữ số từ 81 ô ảnh (API cũ)

        Returns:
            Ma trận 9x9 chứa các chữ số (0 = trống)
        """
        if len(grid_cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(grid_cells)}")
        cells = [
            ImagePreprocessor.normalize_polarity(ImagePreprocessor.to_grayscale(c))
            for c in grid_cells
        ]
        return self._read_cells(cells).digits

    def recognize_cells(self, cells: List[np.ndarray]) -> List[OCRResult]:
        """Nhận diện các ô riêng lẻ, trả về chi tiết từng ô"""
        gray = [
            ImagePreprocessor.normalize_polarity(ImagePreprocessor.to_grayscale(c))
            for c in cells
        ]
        boxes = [self.analyzer.analyze(c) for c in gray]
        idx = [i for i, b in enumerate(boxes) if b is not None]
        results = [OCRResult(0, 0.0, "") for _ in cells]
        for i, (d, conf, text) in zip(idx, self._recognize([self._crop(gray[i], boxes[i]) for i in idx])):
            results[i] = OCRResult(d, conf, text)
        return results

    def batch_predict(self, grid_cells_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        return [self.predict(cells) for cells in grid_cells_list]

    def set_confidence_threshold(self, threshold: float) -> None:
        self.conf_threshold = threshold

    def warmup(self) -> None:
        dummy = np.full((48, 48, 3), 255, np.uint8)
        cv2.putText(dummy, "5", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        self._recognize([dummy] * 4)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _read_cells(self, gray_cells: List[np.ndarray]) -> GridReading:
        digits = np.zeros(81, dtype=int)
        conf = np.zeros(81, dtype=float)
        boxes = [self.analyzer.analyze(c) for c in gray_cells]
        occupied = np.array([b is not None for b in boxes])
        idx = [i for i in range(81) if occupied[i]]

        uncertain = []
        if idx:
            crops = [self._crop(gray_cells[i], boxes[i]) for i in idx]
            first = self._recognize(crops)

            retry = [k for k, (d, _, _) in enumerate(first) if d == 0]
            if retry:
                binary_crops = [self._binarize(crops[k]) for k in retry]
                for k, res in zip(retry, self._recognize(binary_crops)):
                    first[k] = res

            for k, i in enumerate(idx):
                d, c, text = first[k]
                digits[i], conf[i] = d, c
                if d == 0:
                    uncertain.append(divmod(i, 9))
                if self.verbose:
                    print(f"  cell {divmod(i, 9)}: '{text}' -> {d} ({c:.2f})")

        return GridReading(
            digits=digits.reshape(9, 9),
            confidence=conf.reshape(9, 9),
            occupied=occupied.reshape(9, 9),
            uncertain=uncertain,
        )

    @staticmethod
    def _crop(gray: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Cắt vùng chữ số, thêm lề và đặt vào nền trắng hình vuông"""
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = box
        pad = max(2, int(0.12 * h))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = gray[y1:y2, x1:x2]

        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        bg = int(np.median(gray))
        canvas = np.full((side, side), bg, dtype=np.uint8)
        oy, ox = (side - ch) // 2, (side - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop
        if side < 48:
            canvas = cv2.resize(canvas, (48, 48), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = ImagePreprocessor.apply_threshold(gray)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _recognize(self, images: List[np.ndarray]) -> List[Tuple[int, float, str]]:
        if not images:
            return []
        try:
            outputs = list(self.reader.predict(input=images, batch_size=len(images)))
        except Exception as e:
            print(f"⚠️  OCR failed: {e}")
            return [(0, 0.0, "") for _ in images]

        results = []
        for out in outputs:
            text = str(out.get("rec_text", "") or "")
            score = float(out.get("rec_score", 0.0) or 0.0)
            results.append((self._parse_digit(text, score), score, text))
        return results

    def _parse_digit(self, text: str, score: float) -> int:
        cleaned = "".join(ch for ch in text.strip() if ch.isalnum() or ch in "|!][")
        if len(cleaned) != 1:
            return 0
        ch = CHAR_FIXES.get(cleaned, cleaned)
        if ch in "123456789" and score >= self.conf_threshold:
            return int(ch)
        return 0
