"""
pipeline.py

Pipeline: Kết hợp toàn bộ các lớp để giải sudoku từ ảnh
Flow: Ảnh gốc → Detect điểm → Align → Crop → OCR → Solve → Kết quả
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from point_detection import PointDetection
from alignment import Alignment
from preprocessing import Preprocessing
from ocr_digit import OCRDigit
from sudoku_solver import SudokuSolver


class SudokuSolverPipeline:
    """
    Pipeline hoàn chỉnh: Từ ảnh sudoku đến kết quả giải
    
    Các bước:
    1. Detect 4 góc của sudoku
    2. Align ảnh dựa trên 4 góc
    3. Crop ảnh thành 9x9 ô
    4. Nhận diện chữ số (OCR)
    5. Giải sudoku
    """
    
    def __init__(
        self,
        point_detection_model: str,
        ocr_model: Optional[str] = None,
        cell_size: int = 64,
        device: str = "cuda",
        use_easyocr: bool = True,
        conf_threshold: float = 0.3
    ):
        """
        Khởi tạo pipeline
        
        Args:
            point_detection_model: Đường dẫn YOLO model detect góc
            ocr_model: (Deprecated) Đường dẫn YOLO model nhận diện chữ số
            cell_size: Kích thước mỗi ô (default 64x64)
            device: Thiết bị ("cuda" hoặc "cpu")
            use_easyocr: Sử dụng EasyOCR (default True)
            conf_threshold: Confidence threshold cho OCR (0.0-1.0)
        """
        self.device = device
        self.cell_size = cell_size
        self.use_easyocr = use_easyocr
        
        # Khởi tạo các component
        self.point_detector = PointDetection(
            model_path=point_detection_model,
            device=device
        )
        self.alignment = Alignment(output_size=(cell_size * 9, cell_size * 9))
        self.preprocessing = Preprocessing(cell_size=cell_size)
        
        # Khởi tạo OCR sử dụng EasyOCR
        use_gpu = (device == "cuda")
        self.ocr_digit = OCRDigit(
            lang="en",
            use_gpu=use_gpu,
            conf_threshold=conf_threshold
        )
        
        self.solver = SudokuSolver()
        
        self.verbose = True
    
    def run(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline
        
        Args:
            image: Ảnh sudoku gốc
        
        Returns:
            Dict chứa:
            - original_image: Ảnh gốc
            - aligned_image: Ảnh đã align
            - grid_cells: 81 ô ảnh
            - detected_grid: Ma trận chữ số detect được (9x9)
            - solved_grid: Ma trận sudoku đã giải (9x9)
            - success: True nếu giải được
            - steps: Chi tiết các bước
        
        Raises:
            ValueError: Nếu có lỗi trong các bước
        """
        result = {
            "original_image": image,
            "success": False,
            "steps": {}
        }
        
        try:
            # Step 1: Detect 4 góc
            if self.verbose:
                print("Step 1: Detecting 4 corners...")
            
            corners = self.point_detector.predict(image)
            result["steps"]["point_detection"] = "Success"
            result["corners"] = corners
            
            if self.verbose:
                print(f"  ✓ Found 4 corners: TL, TR, BR, BL")
            
            # Step 2: Align ảnh
            if self.verbose:
                print("Step 2: Aligning image...")
            
            aligned_image, transform_matrix = self.alignment.align(image, corners)
            result["aligned_image"] = aligned_image
            result["transform_matrix"] = transform_matrix
            result["steps"]["alignment"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Image aligned to {self.cell_size*9}x{self.cell_size*9}")
            
            # Step 3: Crop ảnh thành 9x9 ô
            if self.verbose:
                print("Step 3: Cropping into 9x9 grid...")
            
            preprocess_result = self.preprocessing.crop(aligned_image)
            grid_cells = preprocess_result.grid_cells
            result["grid_cells"] = grid_cells
            result["steps"]["preprocessing"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Extracted 81 cells ({len(grid_cells)} cells)")
            
            # Step 4: Nhận diện chữ số
            if self.verbose:
                print("Step 4: Recognizing digits (OCR)...")
            
            detected_grid = self.ocr_digit.predict(grid_cells)
            result["detected_grid"] = detected_grid
            result["steps"]["ocr"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Recognized digits")
                self._print_grid("Detected digits", detected_grid)
            
            # Step 5: Giải sudoku
            if self.verbose:
                print("Step 5: Solving sudoku...")
            
            solved_grid, solved = self.solver.solve_sudoku(detected_grid.copy())
            result["solved_grid"] = solved_grid
            result["success"] = solved
            result["steps"]["solver"] = "Success" if solved else "Failed"
            
            if self.verbose:
                if solved:
                    print(f"  ✓ Sudoku solved!")
                    self._print_grid("Solution", solved_grid)
                else:
                    print(f"  ✗ Could not solve sudoku")
            
            return result
        
        except Exception as e:
            result["error"] = str(e)
            if self.verbose:
                print(f"  ✗ Error: {e}")
            raise
    
    def run_batch(self, images: list) -> list:
        """
        Xử lý nhiều ảnh
        
        Args:
            images: Danh sách ảnh
        
        Returns:
            Danh sách kết quả
        """
        results = []
        for i, image in enumerate(images):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Processing image {i+1}/{len(images)}")
                print(f"{'='*50}")
            
            try:
                result = self.run(image)
                results.append(result)
            except Exception as e:
                print(f"Error processing image {i+1}: {e}")
                results.append({"error": str(e), "success": False})
        
        return results
    
    def _print_grid(self, title: str, grid: np.ndarray) -> None:
        """In ma trận sudoku"""
        print(f"\n{title}:")
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("------+-------+------")
            
            row_str = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                
                val = int(grid[i, j])
                row_str += f"{val} "
            
            print(row_str)
    
    def set_verbose(self, verbose: bool) -> None:
        """Đặt chế độ verbose"""
        self.verbose = verbose
    
    def get_pipeline_summary(self, result: Dict) -> str:
        """
        Tóm tắt kết quả pipeline
        
        Args:
            result: Kết quả từ run()
        
        Returns:
            Chuỗi tóm tắt
        """
        lines = []
        lines.append("=" * 50)
        lines.append("PIPELINE SUMMARY")
        lines.append("=" * 50)
        
        for step, status in result.get("steps", {}).items():
            lines.append(f"  {step}: {status}")
        
        lines.append(f"Overall success: {result['success']}")
        
        if result['success']:
            lines.append(f"Difficulty score: {self.solver.get_difficulty_score(result['detected_grid'])}")
        
        if "error" in result:
            lines.append(f"Error: {result['error']}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def warmup(self) -> None:
        """Warmup các models"""
        if self.verbose:
            print("Warming up models...")
        
        self.point_detector.warmup()
        
        if self.verbose:
            print("  ✓ Models warmed up")


def main():
    """Test pipeline"""
    import cv2
    
    # Đường dẫn models
    point_model = r"D:\Projects-Persional\sudoku-solver-ai\backend\weights\best.pt"
    
    # Khởi tạo pipeline
    pipeline = SudokuSolverPipeline(
        point_detection_model=point_model,
        ocr_model=None,  # Nếu có model OCR, đặt đường dẫn ở đây
        cell_size=64,
        device="cpu"
    )
    
    # Load ảnh test
    image_path = r"D:\Projects-Persional\sudoku-solver-ai\imgs\test.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Chạy pipeline
    result = pipeline.run(image)
    
    # In tóm tắt
    print(pipeline.get_pipeline_summary(result))


if __name__ == "__main__":
    main()
