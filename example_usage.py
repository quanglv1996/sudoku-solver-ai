"""
example_usage.py

Ví dụ sử dụng Pipeline Sudoku Solver
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from pipeline import SudokuSolverPipeline


def example_basic_pipeline():
    """
    Ví dụ cơ bản: Chạy pipeline trên một ảnh
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*60)
    
    # Đường dẫn models
    point_model = r"backend\weights\best.pt"
    
    # Khởi tạo pipeline
    pipeline = SudokuSolverPipeline(
        point_detection_model=point_model,
        ocr_model=None,  # Thêm model OCR nếu có
        cell_size=64,
        device="cpu"  # Thay thành "cuda" nếu có GPU
    )
    
    # Warmup models
    pipeline.warmup()
    
    # Load ảnh test
    image_path = r"imgs\test.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Chạy pipeline
    result = pipeline.run(image)
    
    # In tóm tắt
    print(pipeline.get_pipeline_summary(result))
    
    if result["success"]:
        print("\n✓ Sudoku solved successfully!")
        print(f"Original filled cells: {pipeline.solver.get_difficulty_score(result['detected_grid'])}")
    else:
        print("\n✗ Could not solve sudoku")


def example_batch_processing():
    """
    Ví dụ xử lý batch: Chạy pipeline trên nhiều ảnh
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)
    
    point_model = r"backend\weights\best.pt"
    
    pipeline = SudokuSolverPipeline(
        point_detection_model=point_model,
        ocr_model=None,
        cell_size=64,
        device="cpu"
    )
    
    # Giả sử có nhiều ảnh
    image_paths = [
        r"imgs\test.jpg",
        # r"imgs\test2.jpg",
        # r"imgs\test3.jpg",
    ]
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if images:
        results = pipeline.run_batch(images)
        
        # Tổng kết
        successful = sum(1 for r in results if r.get("success", False))
        print(f"\nTotal images: {len(results)}")
        print(f"Successfully solved: {successful}")
        print(f"Failed: {len(results) - successful}")


def example_step_by_step():
    """
    Ví dụ từng bước: Hiển thị kết quả của mỗi bước
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Step-by-step Processing")
    print("="*60)
    
    from alignment import Alignment
    from preprocessing import Preprocessing
    from ocr_digit import OCRDigit
    from sudoku_solver import SudokuSolver
    from point_detection import PointDetection
    
    point_model = r"backend\weights\best.pt"
    
    # Step 1: Load image
    image = cv2.imread(r"imgs\test.jpg")
    if image is None:
        print("Error: Could not load image")
        return
    
    print(f"✓ Loaded image: {image.shape}")
    
    # Step 2: Detect corners
    print("\nStep 1: Point Detection")
    detector = PointDetection(point_model, device="cpu")
    try:
        corners = detector.predict(image)
        print(f"✓ Detected 4 corners: {corners.shape}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
    
    # Step 3: Align image
    print("\nStep 2: Alignment")
    alignment = Alignment(output_size=(576, 576))
    aligned_image, _ = alignment.align(image, corners)
    print(f"✓ Aligned image: {aligned_image.shape}")
    
    # Step 4: Crop into grid
    print("\nStep 3: Preprocessing (Crop)")
    preprocessing = Preprocessing(cell_size=64)
    result = preprocessing.crop(aligned_image)
    print(f"✓ Extracted {len(result.grid_cells)} cells")
    
    # Step 5: OCR (using EasyOCR)
    print("\nStep 4: OCR Digit Recognition (EasyOCR)")
    ocr = OCRDigit(lang="en", use_gpu=False)  # EasyOCR-based recognition
    detected_grid = ocr.predict(result.grid_cells)
    print(f"✓ Detected grid shape: {detected_grid.shape}")
    print("Detected digits:")
    print(detected_grid)
    
    # Step 6: Solve
    print("\nStep 5: Sudoku Solving")
    solver = SudokuSolver()
    solved_grid, success = solver.solve_sudoku(detected_grid.copy())
    
    if success:
        print("✓ Sudoku solved!")
        print("\nSolution:")
        solver.print_grid(solved_grid)
    else:
        print("✗ Could not solve sudoku")


def example_custom_configuration():
    """
    Ví dụ cấu hình tùy chỉnh: Sử dụng những thiết lập khác nhau
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Configuration")
    print("="*60)
    
    point_model = r"backend\weights\best.pt"
    
    # Cấu hình 1: CPU, cell size nhỏ, EasyOCR
    print("\nConfiguration 1: CPU, cell_size=64, EasyOCR")
    pipeline1 = SudokuSolverPipeline(
        point_detection_model=point_model,
        cell_size=64,
        device="cpu",
        use_easyocr=True,
        conf_threshold=0.3
    )
    print(f"✓ Created pipeline (device: {pipeline1.device}, cell_size: {pipeline1.cell_size})")
    
    # Cấu hình 2: CUDA (nếu có), cell size lớn
    print("\nConfiguration 2: CUDA (if available), cell_size=128, EasyOCR")
    pipeline2 = SudokuSolverPipeline(
        point_detection_model=point_model,
        cell_size=128,
        device="cuda",
        use_easyocr=True,
        conf_threshold=0.5
    )
    print(f"✓ Created pipeline (device: {pipeline2.device}, cell_size: {pipeline2.cell_size})")
    
    # Thay đổi configuration runtime
    print("\nConfiguration 3: Runtime changes")
    pipeline1.set_verbose(False)
    pipeline1.ocr_digit.set_confidence_threshold(0.7)
    print("✓ Set verbose mode to False")
    print("✓ Set OCR confidence threshold to 0.7")


def example_result_analysis():
    """
    Ví dụ phân tích kết quả: Trích xuất và phân tích thông tin từ kết quả
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Result Analysis")
    print("="*60)
    
    point_model = r"backend\weights\best.pt"
    
    pipeline = SudokuSolverPipeline(
        point_detection_model=point_model,
        cell_size=64,
        device="cpu"
    )
    
    image = cv2.imread(r"imgs\test.jpg")
    if image is None:
        print("Error: Could not load image")
        return
    
    result = pipeline.run(image)
    
    # Phân tích kết quả
    print("\n" + "="*60)
    print("RESULT ANALYSIS")
    print("="*60)
    
    if result.get("success"):
        original_grid = result["detected_grid"]
        solved_grid = result["solved_grid"]
        
        filled_cells = np.sum(original_grid != 0)
        total_cells = 81
        empty_cells = total_cells - filled_cells
        
        print(f"\nStatistics:")
        print(f"  Filled cells: {filled_cells}/81")
        print(f"  Empty cells: {empty_cells}/81")
        print(f"  Difficulty: {'Easy' if filled_cells > 50 else 'Medium' if filled_cells > 30 else 'Hard'}")
        
        print(f"\nCorners detected:")
        for i, corner in enumerate(result.get("corners", [])):
            print(f"  Corner {i+1}: {corner}")
        
        print(f"\nPipeline steps:")
        for step, status in result["steps"].items():
            symbol = "✓" if status == "Success" else "✗"
            print(f"  {symbol} {step}: {status}")
    else:
        print("\nFailed to solve sudoku")
        if "error" in result:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    # Chọn ví dụ nào để chạy
    examples = {
        "1": ("Basic Pipeline", example_basic_pipeline),
        "2": ("Batch Processing", example_batch_processing),
        "3": ("Step by step", example_step_by_step),
        "4": ("Custom Configuration", example_custom_configuration),
        "5": ("Result Analysis", example_result_analysis),
    }
    
    print("\n" + "="*60)
    print("SUDOKU SOLVER PIPELINE - EXAMPLES")
    print("="*60)
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print("  0. Run all examples")
    
    choice = input("\nChoose example (0-5): ").strip()
    
    if choice == "0":
        for key in sorted(examples.keys()):
            try:
                examples[key][1]()
            except Exception as e:
                print(f"Error in {examples[key][0]}: {e}")
            print("\n")
    elif choice in examples:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice")
