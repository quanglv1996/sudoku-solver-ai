"""
point_detection.py

YOLO Point Detector - Sudoku Grid Corner Detection

Single Responsibility:
- Load YOLO model
- Detect corner objects in sudoku image
- Extract bbox centers
- Return 4 sorted corner points (TL, TR, BR, BL)
"""
from itertools import combinations
from typing import Optional

import numpy as np
from ultralytics import YOLO

from grid_geometry import order_corners, quad_area, quad_is_valid


# =====================================================================
# Utility Functions
# =====================================================================

def _get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Center point [cx, cy] of bbox [x1, y1, x2, y2]"""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def _sort_points(points: np.ndarray) -> np.ndarray:
    """Sort 4 points to TL, TR, BR, BL order."""
    if len(points) != 4:
        raise ValueError(f"Expected 4 points, got {len(points)}")
    return order_corners(points)


# =====================================================================
# Point Detection
# =====================================================================

class PointDetection:
    """
    YOLO-based sudoku corner detection.

    Detects corner objects in sudoku image.
    Returns their bbox centers sorted as: [TL, TR, BR, BL]
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_candidates: int = 8,
        retry_conf: float = 0.1,
    ):
        """
        Args:
            model_path: Path to YOLO model
            device: Device ("cuda" or "cpu")
            imgsz: Input image size for model
            conf: Confidence threshold
            iou: IOU threshold
            max_candidates: Số box tối đa xét khi có nhiều hơn 4 detection
        """
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_candidates = max_candidates
        self.retry_conf = retry_conf

        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        self.model = YOLO(self.model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Detect 4 corners in sudoku image.

        If more than 4 corners detected, choose the 4 boxes (among the most
        confident ones) that form the best valid quadrilateral, so two boxes
        on the same corner can't both be selected.

        Returns:
            Sorted corner points (4, 2): [TL, TR, BR, BL]

        Raises:
            ValueError: If no valid 4-corner grid found
        """
        centers, confs = self._detect(image, self.conf)
        quad = self._best_quad(centers, confs, image.shape)
        if quad is None and self.retry_conf < self.conf:
            # Ảnh nghiêng/xoay: model thường kém tự tin hơn → thử lại ngưỡng thấp
            centers, confs = self._detect(image, self.retry_conf)
            quad = self._best_quad(centers, confs, image.shape)
        if quad is None:
            raise ValueError(f"No valid 4-corner grid (detected {len(centers)} corners)")
        return quad

    def _detect(self, image: np.ndarray, conf: float):
        result = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 2), np.float32), np.zeros(0, np.float32)
        confidences = result.boxes.conf.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        order = np.argsort(confidences)[::-1][: self.max_candidates]
        centers = np.array([_get_bbox_center(boxes[i]) for i in order], dtype=np.float32)
        return centers, confidences[order]

    @staticmethod
    def _best_quad(centers: np.ndarray, confs: np.ndarray, shape) -> Optional[np.ndarray]:
        """
        Chọn tứ giác hợp lệ tốt nhất từ các góc phát hiện được:
        - tổ hợp 4 góc (tránh chọn 2 box cùng một góc)
        - nếu chỉ thấy 3 góc: suy ra góc thứ 4 theo hình bình hành
        """
        img_area = float(shape[0] * shape[1])
        best, best_score = None, -1.0

        def consider(points, conf, penalty=1.0):
            nonlocal best, best_score
            quad = order_corners(points)
            if not quad_is_valid(quad, shape):
                return
            score = conf * penalty * (quad_area(quad) / img_area) ** 0.25
            if score > best_score:
                best, best_score = quad, score

        for combo in combinations(range(len(centers)), 4):
            consider(centers[list(combo)], float(np.mean(confs[list(combo)])))

        if best is None:
            for combo in combinations(range(len(centers)), 3):
                pts = centers[list(combo)]
                for m in range(3):
                    a, b = pts[(m + 1) % 3], pts[(m + 2) % 3]
                    va, vb = a - pts[m], b - pts[m]
                    cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6))
                    if abs(cos) > 0.45:  # góc ở đỉnh giữa phải gần vuông (~63°-117°)
                        continue
                    fourth = a + b - pts[m]
                    consider(np.vstack([pts, fourth]), float(np.mean(confs[list(combo)])), penalty=0.8)
        return best

    def set_conf(self, conf: float) -> None:
        self.conf = conf

    def set_iou(self, iou: float) -> None:
        self.iou = iou

    def set_device(self, device: str) -> None:
        self.device = device

    def warmup(self) -> None:
        """Warmup model with dummy input."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.predict(dummy)
        except ValueError:
            pass  # Expected - dummy image has no detections
