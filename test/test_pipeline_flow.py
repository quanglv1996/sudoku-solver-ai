"""
test_pipeline_flow.py

Test the complete pipeline flow:
Image → Point Detection → Preprocessing (align) → OCR Digit → Sudoku Solve
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.sudoku_pipeline import SudokuPipeline


def test_pipeline_with_image(image_path: str):
    """
    Test the complete pipeline with an image.
    
    Pipeline Flow:
    1. Image Input (numpy array)
    2. Point Detection (detect 4 corner points)
    3. Preprocessing (align sudoku grid)
    4. OCR Digit (recognize digits in 9x9 grid)
    5. Sudoku Solve (solve the puzzle)
    """
    print("=" * 60)
    print("SUDOKU SOLVER PIPELINE TEST")
    print("=" * 60)
    
    # Load image
    print(f"\n[Step 1] Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return
    
    print(f"✓ Image loaded: {image.shape}")
    
    # Initialize pipeline
    print("\n[Init] Initializing pipeline...")
    try:
        pipeline = SudokuPipeline(
            point_model_path="path/to/model.pt",  # Configure your model path
            device="cuda",
            use_gpu=True
        )
        print("✓ Pipeline initialized")
    except Exception as e:
        print(f"ERROR initializing pipeline: {e}")
        return
    
    # Run the complete pipeline
    print("\n[Processing] Running complete pipeline...")
    print("-" * 60)
    
    success, results = pipeline.process(image)
    
    if not success:
        print(f"\n✗ Pipeline failed: {results.get('error', 'Unknown error')}")
        return
    
    print("✓ Pipeline completed successfully")
    
    # Display results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Detection points
    if results['detection_points'] is not None:
        print(f"\n[Point Detection]")
        print(f"✓ Detected 4 corner points:")
        for i, point in enumerate(results['detection_points']):
            print(f"  Point {i}: ({point[0]:.1f}, {point[1]:.1f})")
    
    # OCR Grid
    if results['input_grid'] is not None:
        print(f"\n[OCR Recognition]")
        filled = np.sum(results['input_grid'] != 0)
        print(f"✓ Recognized: {filled}/81 cells filled")
        print("Input Grid (OCR):")
        print(results['input_grid'])
    
    # Solved Grid
    if results['solution_grid'] is not None:
        print(f"\n[Sudoku Solution]")
        print("✓ Puzzle solved")
        print("Solution Grid:")
        print(results['solution_grid'])
    
    # Visualizations
    print(f"\n[Visualizations]")
    if results['viz_original'] is not None:
        print("✓ Original image with detected points")
    if results['viz_aligned'] is not None:
        print("✓ Aligned image with solution")
    
    print("\n" + "=" * 60)
    print("PIPELINE FLOW SUMMARY")
    print("=" * 60)
    print("""
    Image (numpy array)
        ↓
    Point Detection (YOLO)
        ↓
    Preprocessing (Align & Perspective Transform)
        ↓
    OCR Digit Recognition (EasyOCR Batch Processing)
        ↓
    Sudoku Solver (Backtracking Algorithm)
        ↓
    Results & Visualizations
    """)


if __name__ == "__main__":
    # Test image path
    test_image = r"D:\Projects-Persional\sudoku-solver-ai\input\1.png"
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    test_pipeline_with_image(test_image)
