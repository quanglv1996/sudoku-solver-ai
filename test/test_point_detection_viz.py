"""
test_point_detection.py

Test class for Point Detection
- Load images from input folder
- Detect points using YOLO
- Visualize detected points
- Save results to output folder
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.point_detection import PointDetection, PointDetectionResult, PointUtils


class PointDetectionTester:
    """
    Test and visualize point detection results.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        input_dir: str = "input",
        output_dir: str = "output"
    ):
        """
        Initialize tester.
        
        Args:
            model_path: Path to YOLO model
            device: Device ("cuda" or "cpu")
            input_dir: Input images directory
            output_dir: Output results directory
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize detector
        self.detector = PointDetection(
            model_path=model_path,
            device=device
        )
        
        print(f"✓ PointDetectionTester initialized")
        print(f"  Input: {self.input_dir}")
        print(f"  Output: {self.output_dir}")

    def detect_image(self, image_path: str) -> Optional[PointDetectionResult]:
        """
        Detect points in a single image.
        
        Args:
            image_path: Path to image
            
        Returns:
            First PointDetectionResult or None if no detection
        """
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        # Detect
        results = self.detector.predict(image)
        
        if not results:
            print(f"⚠️  No detection found")
            return None
        
        return results[0]

    def visualize_detection(
        self,
        image_path: str,
        detection: PointDetectionResult,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw detected points and bbox on image.
        
        Args:
            image_path: Path to input image
            detection: PointDetectionResult
            output_path: Optional path to save result
            
        Returns:
            Image with visualizations
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        viz_image = image.copy()
        
        # Draw bounding box
        bbox = detection.bbox.astype(int)
        cv2.rectangle(
            viz_image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            (0, 255, 0),  # Green
            2
        )
        
        # Draw detected points
        points = detection.points.astype(int)
        point_scores = detection.point_scores
        
        for i, (pt, score) in enumerate(zip(points, point_scores)):
            x, y = pt
            
            # Color based on confidence
            if score > 0.8:
                color = (0, 255, 0)  # Green
            elif score > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw circle
            cv2.circle(viz_image, (x, y), 5, color, -1)
            
            # Draw point index
            cv2.putText(
                viz_image,
                f"P{i}",
                (x + 10, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            
            # Draw score
            cv2.putText(
                viz_image,
                f"{score:.2f}",
                (x + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        # Draw detection score
        score_text = f"Confidence: {detection.score:.2f}"
        cv2.putText(
            viz_image,
            score_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Draw number of points
        points_text = f"Points: {len(points)}"
        cv2.putText(
            viz_image,
            points_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, viz_image)
            print(f"✓ Saved visualization: {output_path}")
        
        return viz_image

    def test_single_image(self, image_path: str) -> bool:
        """
        Test detection on single image and save result.
        
        Args:
            image_path: Path to image
            
        Returns:
            True if successful
        """
        print(f"\n{'='*60}")
        print(f"Testing: {image_path}")
        print(f"{'='*60}")
        
        # Detect
        detection = self.detect_image(image_path)
        
        if detection is None:
            return False
        
        print(f"✓ Detection successful")
        print(f"  Confidence: {detection.score:.2f}")
        print(f"  Points: {len(detection.points)}")
        print(f"  Point scores: {detection.point_scores}")
        print(f"  BBox: {detection.bbox}")
        
        # Visualize
        image_name = Path(image_path).stem
        output_path = str(self.output_dir / f"{image_name}_detection.jpg")
        
        self.visualize_detection(image_path, detection, output_path)
        
        return True

    def test_all_images(self) -> dict:
        """
        Test detection on all images in input directory.
        
        Returns:
            Dictionary with results
        """
        print(f"\n{'='*60}")
        print(f"Testing all images in {self.input_dir}")
        print(f"{'='*60}")
        
        # Find image files
        image_files = list(self.input_dir.glob("*.jpg")) + \
                     list(self.input_dir.glob("*.png")) + \
                     list(self.input_dir.glob("*.JPG")) + \
                     list(self.input_dir.glob("*.PNG"))
        
        if not image_files:
            print(f"⚠️  No images found in {self.input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"Found {len(image_files)} images\n")
        
        results = {
            'total': len(image_files),
            'success': 0,
            'failed': 0,
            'files': []
        }
        
        for image_path in image_files:
            success = self.test_single_image(str(image_path))
            
            if success:
                results['success'] += 1
                results['files'].append({
                    'path': str(image_path),
                    'status': 'success'
                })
            else:
                results['failed'] += 1
                results['files'].append({
                    'path': str(image_path),
                    'status': 'failed'
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {results['total']}")
        print(f"Success: {results['success']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['success']/results['total']*100:.1f}%")
        
        return results

    def draw_comparison(
        self,
        image_path: str,
        detection: PointDetectionResult,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Draw detailed comparison visualization.
        
        Args:
            image_path: Path to image
            detection: PointDetectionResult
            output_path: Optional output path
            
        Returns:
            Visualization image
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Create canvas
        canvas = np.ones((h + 100, w, 3), dtype=np.uint8) * 255
        canvas[:h, :] = image
        
        # Draw info panel
        y = h + 20
        
        info = [
            f"Detection Results",
            f"Confidence: {detection.score:.2f}",
            f"Num Points: {len(detection.points)}",
            f"BBox: ({detection.bbox[0]:.0f}, {detection.bbox[1]:.0f}, {detection.bbox[2]:.0f}, {detection.bbox[3]:.0f})"
        ]
        
        for i, text in enumerate(info):
            cv2.putText(
                canvas,
                text,
                (10, y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        # Draw detection
        viz_image = self.visualize_detection(image_path, detection)
        
        # Combine
        result = np.vstack([viz_image, canvas[h:, :]])
        
        if output_path:
            cv2.imwrite(output_path, result)
        
        return result


def main():
    """
    Main test function.
    """
    print("="*60)
    print("POINT DETECTION TESTER")
    print("="*60)
    
    # Configuration
    MODEL_PATH = r"D:\Projects-Persional\sudoku-solver-ai\src\weights\testttt_1_best.pt"
    INPUT_DIR = r"D:\Projects-Persional\sudoku-solver-ai\input"
    OUTPUT_DIR = "output"
    DEVICE = "cuda"
    
    # Create tester
    try:
        tester = PointDetectionTester(
            model_path=MODEL_PATH,
            device=DEVICE,
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR
        )
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        return
    
    # Test all images
    results = tester.test_all_images()
    
    # Display results
    if results['success'] > 0:
        print(f"\n✓ Test completed successfully!")
        print(f"  Results saved to: {OUTPUT_DIR}")
    else:
        print(f"\n⚠️  No successful detections")


if __name__ == "__main__":
    main()
