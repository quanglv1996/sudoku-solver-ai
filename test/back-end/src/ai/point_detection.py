"""
point_detection.py

YOLO Point Detector - Simplified for Sudoku Grid Corner Detection

Single Responsibility:
- Load YOLO model
- Detect 4 corner objects in sudoku image
- Extract bbox centers
- Return 4 sorted corner points (TL, TR, BR, BL)
"""

from typing import Optional
import numpy as np
from ultralytics import YOLO


# =====================================================================
# Utility Functions
# =====================================================================

def _get_bbox_center(bbox: np.ndarray) -> np.ndarray:
    """
    Extract center point from bounding box.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Center point [center_x, center_y]
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return np.array([center_x, center_y], dtype=np.float32)


def _sort_points(points: np.ndarray) -> np.ndarray:
    """
    Sort 4 points to TL, TR, BR, BL order.
    
    TL ---- TR
    |        |
    BL ---- BR
    
    Args:
        points: Array of 4 points (4, 2)
        
    Returns:
        Sorted points (4, 2)
    """
    if len(points) != 4:
        raise ValueError(f"Expected 4 points, got {len(points)}")
    
    # Find center
    center = np.mean(points, axis=0)
    
    # Split left and right
    left = points[points[:, 0] < center[0]]
    right = points[points[:, 0] >= center[0]]
    
    if len(left) != 2 or len(right) != 2:
        raise ValueError("Could not split points into left/right")
    
    # Sort each side vertically
    tl = left[np.argmin(left[:, 1])]
    bl = left[np.argmax(left[:, 1])]
    
    tr = right[np.argmin(right[:, 1])]
    br = right[np.argmax(right[:, 1])]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)


# =====================================================================
# Point Detection
# =====================================================================

class PointDetection:
    """
    YOLO-based sudoku corner detection.
    
    Detects 4 corner objects in sudoku image.
    Returns their bbox centers sorted as: [TL, TR, BR, BL]
    
    Simple and focused: inference only, no batch processing.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model
            device: Device ("cuda" or "cpu")
            imgsz: Input image size for model
            conf: Confidence threshold
            iou: IOU threshold
        """
        self.model_path = model_path
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        self.model = YOLO(self.model_path)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Detect 4 corners in sudoku image.
        
        Returns bbox centers of detected objects sorted as:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Sorted corner points (4, 2) - ready for perspective transform
            
        Raises:
            ValueError: If not exactly 4 objects detected
        """
        # Inference
        results = self.model.predict(
            source=image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        # Check detections
        if result.boxes is None or len(result.boxes) == 0:
            raise ValueError("No objects detected in image")
        
        if len(result.boxes) != 4:
            raise ValueError(
                f"Expected exactly 4 corners, detected {len(result.boxes)}"
            )
        
        # Extract bbox centers
        centers = []
        for i in range(4):
            bbox = result.boxes.xyxy[i].cpu().numpy()
            center = _get_bbox_center(bbox)
            centers.append(center)
        
        centers = np.array(centers, dtype=np.float32)
        
        # Sort to [TL, TR, BR, BL]
        sorted_centers = _sort_points(centers)
        
        return sorted_centers

    def set_conf(self, conf: float) -> None:
        """Set confidence threshold."""
        self.conf = conf

    def set_iou(self, iou: float) -> None:
        """Set IOU threshold."""
        self.iou = iou

    def set_device(self, device: str) -> None:
        """Set device."""
        self.device = device

    def warmup(self) -> None:
        """Warmup model with dummy input."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.predict(dummy)
        except ValueError:
            # Expected - dummy image has no detections
            pass


def main():
    """Test PointDetection."""
    import cv2
    
    # Load model
    model_path = r"D:\Projects-Persional\sudoku-solver-ai\src\weights\best.pt"
    detector = PointDetection(model_path=model_path, device="cpu")
    
    # Load test image
    image = cv2.imread(r"D:\Projects-Persional\sudoku-solver-ai\input\2.jpg")
    if image is None:
        print("Error: Could not load image")
        return
    
    # Predict - returns 4 sorted corner points
    try:
        corners = detector.predict(image)
        print(f"✅ Detected 4 corners:\n{corners}")
        
        # Draw corners on image
        output = image.copy()
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(corners):
            cv2.circle(output, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(output, labels[i], (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Sudoku Corners", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except ValueError as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()