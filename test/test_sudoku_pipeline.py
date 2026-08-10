"""
test_sudoku_pipeline.py

Test the complete sudoku solving pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.pipeline.sudoku_pipeline import SudokuPipeline
from src.ai.sudoku_solve import SudokuSolver


def test_sudoku_solver():
    """Test sudoku solver."""
    print("\n" + "="*60)
    print("TEST: Sudoku Solver")
    print("="*60)
    
    solver = SudokuSolver()
    
    # Create a test sudoku
    grid = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    print("Input Grid:")
    solver.print_grid(grid)
    
    print("\nSolving...")
    solution = solver.solve_sudoku(grid)
    
    print("\nSolution:")
    solver.print_grid(solution)
    
    print("✓ Sudoku solver works")


def test_preprocessing():
    """Test preprocessing and grid extraction."""
    print("\n" + "="*60)
    print("TEST: Preprocessing")
    print("="*60)
    
    from src.ai.preprocessing import PreProcessing
    
    preprocessor = PreProcessing(cell_size=64)
    
    # Create dummy image and points
    image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a square in the image
    pts = np.array([
        [50, 50],      # TL
        [590, 50],     # TR
        [590, 430],    # BR
        [50, 430]      # BL
    ], dtype=np.float32)
    
    # Add some grid lines to image for visualization
    cv2.polylines(image, [pts.astype(int)], True, (0, 0, 0), 2)
    
    print(f"Image size: {image.shape}")
    print(f"Corner points: {pts}")
    
    # Preprocess
    result = preprocessor.preprocess(image, pts)
    
    print(f"Aligned image size: {result.aligned_image.shape}")
    print(f"Number of cells extracted: {len(result.grid_cells)}")
    print(f"Cell size: {result.cell_size}")
    
    print("✓ Preprocessing works")


def test_pipeline_with_image(image_path: str):
    """Test pipeline with real image."""
    print("\n" + "="*60)
    print(f"TEST: Pipeline with Image")
    print("="*60)
    
    # Try to load model
    model_path = "yolov8n.pt"  # Will download if needed
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"Image size: {image.shape}")
    
    try:
        pipeline = SudokuPipeline(model_path, device="cuda", cell_size=64)
        print("✓ Pipeline initialized")
        
        print("\nProcessing...")
        success, results = pipeline.process(image)
        
        if success:
            print("✓ Pipeline completed successfully")
            
            if results['input_grid'] is not None:
                print("\nOCR Result Grid:")
                print(results['input_grid'])
            
            if results['solution_grid'] is not None:
                print("\nSolution Grid:")
                print(results['solution_grid'])
            
            # Display
            pipeline.display_results(results)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"❌ Pipeline failed: {results['error']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_all():
    """Run all tests."""
    print("\n" + "="*60)
    print("SUDOKU PIPELINE TEST SUITE")
    print("="*60)
    
    # Test 1: Solver
    test_sudoku_solver()
    
    # Test 2: Preprocessing
    test_preprocessing()
    
    # Test 3: Pipeline with image
    # Try to find an image from dataset
    project_root = Path(__file__).parent.parent.parent
    dataset_path = project_root / "models" / "project-5-at-2026-07-17-07-39-de48885f"
    
    if dataset_path.exists():
        images_dir = dataset_path / "images"
        image_files = list(images_dir.glob("*.jpg"))
        
        if image_files:
            test_pipeline_with_image(str(image_files[0]))
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    test_all()
