"""
pipeline.py

Pipeline: Kết hợp toàn bộ các lớp để giải sudoku từ ảnh
Flow: Ảnh gốc → Detect điểm → Align → Crop → OCR → Solve → Kết quả
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

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
        conf_threshold: float = 0.3,
        save_debug: bool = False,
        debug_dir: str = "debug_output"
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
            save_debug: Lưu ảnh debug ở từng bước
            debug_dir: Thư mục lưu ảnh debug
        """
        self.device = device
        self.cell_size = cell_size
        self.use_easyocr = use_easyocr
        self.save_debug = save_debug
        self.debug_dir = Path(debug_dir)
        
        # Tạo thư mục debug nếu cần
        if self.save_debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            # Tạo subfolder cho session hiện tại
            self.debug_session_dir = self.debug_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_session_dir.mkdir(parents=True, exist_ok=True)
            print(f"Debug images will be saved to: {self.debug_session_dir}")
        
        # Khởi tạo các component
        self.point_detector = PointDetection(
            model_path=point_detection_model,
            device=device
        )
        self.alignment = Alignment(output_size=(cell_size * 9, cell_size * 9))
        self.preprocessing = Preprocessing(
            cell_size=cell_size,
            margin=4  # 🔍 Cắt sâu vào bên trong, tránh viền (4 pixel)
        )
        
        # Khởi tạo OCR sử dụng EasyOCR
        use_gpu = (device == "cuda")
        self.ocr_digit = OCRDigit(
            lang="en",
            use_gpu=use_gpu,
            conf_threshold=conf_threshold
        )
        
        self.solver = SudokuSolver()
        
        self.verbose = True
    
    # =====================================================================
    # Debug Functions
    # =====================================================================
    
    def _save_debug_image(self, image: np.ndarray, step_name: str) -> None:
        """
        Lưu ảnh debug
        
        Args:
            image: Ảnh cần lưu
            step_name: Tên bước (e.g., "01_original", "02_corners", etc.)
        """
        if not self.save_debug:
            return
        
        output_path = self.debug_session_dir / f"{step_name}.jpg"
        cv2.imwrite(str(output_path), image)
        if self.verbose:
            print(f"  📸 Saved: {step_name}")
    
    def _draw_corners_on_image(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Vẽ 4 điểm corner lên ảnh
        
        Args:
            image: Ảnh gốc
            corners: Array 4x2 chứa 4 điểm [TL, TR, BR, BL]
        
        Returns:
            Ảnh với các điểm được vẽ
        """
        img_copy = image.copy()
        
        # Tên các điểm
        point_names = ['TL', 'TR', 'BR', 'BL']
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR
        
        for i, (point, name, color) in enumerate(zip(corners, point_names, colors)):
            x, y = int(point[0]), int(point[1])
            
            # Vẽ circle tại điểm
            cv2.circle(img_copy, (x, y), 8, color, -1)
            cv2.circle(img_copy, (x, y), 10, (255, 255, 255), 2)
            
            # Vẽ text tên điểm
            cv2.putText(img_copy, name, (x + 15, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Vẽ đường nối các điểm
        for i in range(4):
            p1 = tuple(map(int, corners[i]))
            p2 = tuple(map(int, corners[(i + 1) % 4]))
            cv2.line(img_copy, p1, p2, (0, 255, 255), 3)
        
        return img_copy
    
    def _draw_ocr_results_on_cells(
        self, 
        grid_cells: list,
        detected_grid: np.ndarray
    ) -> list:
        """
        Vẽ kết quả OCR lên các ô ảnh
        
        Args:
            grid_cells: Danh sách 81 ô ảnh
            detected_grid: Ma trận 9x9 chứa các chữ số OCR detected
        
        Returns:
            Danh sách 81 ô ảnh với chữ số được vẽ
        """
        annotated_cells = []
        
        for idx, cell in enumerate(grid_cells):
            row = idx // 9
            col = idx % 9
            digit = int(detected_grid[row, col])
            
            cell_copy = cell.copy()
            
            # Nếu là ảnh grayscale, convert sang BGR để vẽ màu
            if len(cell_copy.shape) == 2:
                cell_copy = cv2.cvtColor(cell_copy, cv2.COLOR_GRAY2BGR)
            
            # Vẽ chữ số nếu detect được
            if digit > 0:
                h, w = cell_copy.shape[:2]
                
                # Vẽ background tối để chữ số rõ ràng hơn
                cv2.rectangle(cell_copy, (0, 0), (w, h), (0, 0, 0), -1)
                
                # Vẽ chữ số
                text = str(digit)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 2
                color = (0, 255, 0)  # Green
                
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                x = (w - text_size[0]) // 2
                y = (h + text_size[1]) // 2
                
                cv2.putText(cell_copy, text, (x, y), font, font_scale, color, thickness)
            else:
                # Ô trống: vẽ hình vuông xám
                h, w = cell_copy.shape[:2]
                cv2.rectangle(cell_copy, (5, 5), (w-5, h-5), (100, 100, 100), 2)
            
            annotated_cells.append(cell_copy)
        
        return annotated_cells
    
    def _save_cells_grid(self, cells: list, grid_name: str) -> None:
        """
        Lưu 81 ô ảnh thành một ảnh 9x9 grid
        
        Args:
            cells: Danh sách 81 ô ảnh
            grid_name: Tên file (e.g., "04_ocr_cells")
        """
        if not self.save_debug:
            return
        
        # Tạo grid 9x9
        cell_h, cell_w = cells[0].shape[:2]
        grid_img = np.zeros((cell_h * 9, cell_w * 9, 3), dtype=np.uint8)
        
        for idx, cell in enumerate(cells):
            row = idx // 9
            col = idx % 9
            
            # Convert grayscale to BGR nếu cần
            if len(cell.shape) == 2:
                cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            
            grid_img[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w
            ] = cell
        
        output_path = self.debug_session_dir / f"{grid_name}.jpg"
        cv2.imwrite(str(output_path), grid_img)
        if self.verbose:
            print(f"  📸 Saved: {grid_name}")
    
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
            
            # Lưu ảnh gốc
            self._save_debug_image(image, "01_original_image")
            
            # Lưu ảnh với corners được vẽ
            corners_img = self._draw_corners_on_image(image, corners)
            self._save_debug_image(corners_img, "02_detected_corners")
            
            # Step 2: Align ảnh
            if self.verbose:
                print("Step 2: Aligning image...")
            
            aligned_image, transform_matrix = self.alignment.align(image, corners)
            result["aligned_image"] = aligned_image
            result["transform_matrix"] = transform_matrix
            result["steps"]["alignment"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Image aligned to {self.cell_size*9}x{self.cell_size*9}")
            
            # Lưu ảnh sau align
            self._save_debug_image(aligned_image, "03_aligned_image")
            
            # Step 3: Crop ảnh thành 9x9 ô
            if self.verbose:
                print("Step 3: Cropping into 9x9 grid...")
            
            preprocess_result = self.preprocessing.crop(aligned_image)
            grid_cells = preprocess_result.grid_cells
            result["grid_cells"] = grid_cells
            result["steps"]["preprocessing"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Extracted 81 cells ({len(grid_cells)} cells)")
            
            # Lưu grid cells (raw)
            self._save_cells_grid(grid_cells, "04_cropped_cells_raw")
            
            # Step 4: Nhận diện chữ số
            if self.verbose:
                print("Step 4: Recognizing digits (OCR)...")
            
            detected_grid = self.ocr_digit.predict(grid_cells)
            result["detected_grid"] = detected_grid
            result["steps"]["ocr"] = "Success"
            
            if self.verbose:
                print(f"  ✓ Recognized digits")
                self._print_grid("Detected digits", detected_grid)
            
            # Lưu cells với OCR results được vẽ
            ocr_cells = self._draw_ocr_results_on_cells(grid_cells, detected_grid)
            self._save_cells_grid(ocr_cells, "05_ocr_results_drawn")
            
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
