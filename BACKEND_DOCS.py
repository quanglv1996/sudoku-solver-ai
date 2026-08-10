"""
Sudoku Solver AI - Backend Documentation
Hướng dẫn kiến trúc và sử dụng
"""

# =====================================================================
# KIẾN TRÚC HỆ THỐNG
# =====================================================================

ARCHITECTURE = """
SUDOKU SOLVER AI - SYSTEM ARCHITECTURE

┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Sudoku Image                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  1. POINT DETECTION (PointDetection)                            │
│     - Sử dụng YOLO model                                        │
│     - Detect 4 góc của sudoku                                   │
│     Output: 4 điểm góc [TL, TR, BR, BL]                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  2. ALIGNMENT (Alignment)                                       │
│     - Perspective transform                                     │
│     - Thẳng hóa ảnh sudoku                                      │
│     Output: Ảnh 576x576 (hoặc tùy chỉnh)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  3. PREPROCESSING (Preprocessing)                               │
│     - Cắt ảnh thành 9x9 ô                                       │
│     - Mỗi ô: 64x64 pixels                                       │
│     Output: 81 ô ảnh                                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  4. OCR DIGIT (OCRDigit)                                        │
│     - Nhận diện chữ số từ mỗi ô                                │
│     - Sử dụng YOLO hoặc phương pháp đơn giản                   │
│     Output: Ma trận 9x9 chữ số (0-9, 0=trống)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  5. SUDOKU SOLVER (SudokuSolver)                                │
│     - Giải sudoku bằng backtracking                             │
│     - Validate rules (hàng, cột, ô 3x3)                        │
│     Output: Ma trận sudoku đã giải                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    OUTPUT: Solved Grid                          │
│                    + Intermediate Results                       │
└─────────────────────────────────────────────────────────────────┘

ORCHESTRATION: SudokuSolverPipeline
├─ Kết hợp tất cả các component
├─ Quản lý luồng dữ liệu
├─ Xử lý lỗi và validation
└─ Cung cấp interface thống nhất
"""

# =====================================================================
# CÁC LỚP (CLASSES)
# =====================================================================

CLASSES = """
1. Alignment (alignment.py)
   ─────────────────────────
   Trách nhiệm: Căn chỉnh ảnh sudoku thành hình vuông tiêu chuẩn
   
   Hàm chính:
   - align(image, points) → (aligned_image, transform_matrix)
     * Nhập: ảnh gốc + 4 điểm góc
     * Xuất: ảnh đã align (576x576)
   
   Phương thức nội bộ:
   - _sort_points(points): Sắp xếp 4 điểm theo thứ tự TL, TR, BR, BL


2. Preprocessing (preprocessing.py)
   ────────────────────────────────
   Trách nhiệm: Cắt ảnh thành 9x9 ô (64x64 mỗi ô)
   
   Hàm chính:
   - crop(aligned_image) → PreprocessingResult
     * Nhập: ảnh đã align
     * Xuất: 81 ô ảnh + metadata
   
   Phương thức hỗ trợ:
   - get_cell_at(grid_cells, row, col): Lấy ô tại vị trí
   - reshape_cells_to_grid(grid_cells): Chuyển đổi format


3. OCRDigit (ocr_digit.py)
   ─────────────────────────
   Trách nhiệm: Nhận diện chữ số từ ô ảnh sử dụng EasyOCR
   
   Hàm chính:
   - predict(grid_cells) → digits_grid (9x9)
     * Nhập: 81 ô ảnh
     * Xuất: ma trận chữ số (0-9, 0=trống)
   
   Sử dụng:
   - EasyOCR: Công cụ OCR chuyên nghiệp, chính xác cao
   - Hỗ trợ GPU/CPU tự động
   - Tiền xử lý ảnh tự động (contrast enhancement, grayscale)


4. SudokuSolver (sudoku_solver.py)
   ─────────────────────────────────
   Trách nhiệm: Giải sudoku bằng backtracking
   
   Hàm chính:
   - solve_sudoku(grid) → (solution, success)
     * Nhập: ma trận chưa giải (0 = trống)
     * Xuất: ma trận đã giải + trạng thái
   
   Helper class:
   - SudokuValidator: Kiểm tra tính hợp lệ của placement


5. SudokuSolverPipeline (pipeline.py)
   ───────────────────────────────────
   Trách nhiệm: Kết hợp tất cả component
   
   Hàm chính:
   - run(image) → result_dict
     * Thực hiện tất cả 5 bước
     * Trả về kết quả chi tiết
   
   - run_batch(images) → list[result_dict]
     * Xử lý nhiều ảnh
   
   Tiện ích:
   - get_pipeline_summary(result): Tóm tắt kết quả
   - warmup(): Pre-load models vào GPU/CPU
"""

# =====================================================================
# CÁCH SỦDỤNG
# =====================================================================

USAGE = """
1. CÁC BƯỚC CỰC KỲ ĐƠN GIẢN
   ═══════════════════════════

   from backend.src.pipeline import SudokuSolverPipeline
   import cv2
   
   # Khởi tạo pipeline
   pipeline = SudokuSolverPipeline(
       point_detection_model=r"backend\weights\best.pt",
       ocr_model=None,  # Optional OCR model
       cell_size=64,
       device="cpu"  # Hoặc "cuda"
   )
   
   # Load ảnh
   image = cv2.imread("sudoku.jpg")
   
   # Chạy pipeline
   result = pipeline.run(image)
   
   # Kiểm tra kết quả
   if result["success"]:
       solved_grid = result["solved_grid"]
       print("Sudoku solved!")
   else:
       print("Could not solve sudoku")


2. SỬ DỤNG TỪNG COMPONENT RIÊNG Lẻ
   ════════════════════════════════

   from backend.src import Alignment, Preprocessing, SudokuSolver
   from backend.src import PointDetection, OCRDigit
   
   # Point Detection
   detector = PointDetection("best.pt", device="cpu")
   corners = detector.predict(image)
   
   # Alignment
   alignment = Alignment()
   aligned_image, transform = alignment.align(image, corners)
   
   # Preprocessing
   preprocessing = Preprocessing(cell_size=64)
   result = preprocessing.crop(aligned_image)
   grid_cells = result.grid_cells
   
   # OCR
   ocr = OCRDigit()
   detected_grid = ocr.predict(grid_cells)
   
   # Solve
   solver = SudokuSolver()
   solved_grid, success = solver.solve_sudoku(detected_grid)


3. BATCH PROCESSING
   ═════════════════

   images = [cv2.imread(f"img{i}.jpg") for i in range(10)]
   
   pipeline = SudokuSolverPipeline(point_model, device="cuda")
   results = pipeline.run_batch(images)
   
   # Kiểm tra kết quả
   successful = sum(1 for r in results if r["success"])
   print(f"Solved {successful}/{len(results)}")


4. CUSTOM CONFIGURATION
   ═════════════════════

   # Cấu hình tùy chỉnh
   pipeline = SudokuSolverPipeline(
       point_detection_model=point_model,
       ocr_model=ocr_model,           # Cung cấp OCR model
       cell_size=128,                  # Ô lớn hơn để OCR chính xác hơn
       device="cuda"                   # Dùng GPU
   )
   
   # Điều chỉnh runtime
   pipeline.point_detector.set_conf(0.5)   # Thay confidence
   pipeline.ocr_digit.set_confidence(0.7)
   pipeline.set_verbose(False)              # Tắt output
"""

# =====================================================================
# CẤU TRÚC KẾT QUẢ
# =====================================================================

RESULT_STRUCTURE = """
result = pipeline.run(image)
─────────────────────────────

result là một dict với các keys:

{
    # Trạng thái chung
    "success": bool,                    # True nếu giải được
    "original_image": np.ndarray,       # Ảnh gốc
    "aligned_image": np.ndarray,        # Ảnh đã align
    "grid_cells": List[np.ndarray],     # 81 ô ảnh
    "corners": np.ndarray,              # 4 điểm góc detect được
    
    # Kết quả từng bước
    "detected_grid": np.ndarray,        # Ma trận chữ số detect (9x9)
    "solved_grid": np.ndarray,          # Ma trận đã giải (9x9)
    
    # Transform matrix
    "transform_matrix": np.ndarray,     # Perspective transform
    
    # Chi tiết các bước
    "steps": {
        "point_detection": "Success/Failed",
        "alignment": "Success/Failed",
        "preprocessing": "Success/Failed",
        "ocr": "Success/Failed",
        "solver": "Success/Failed"
    },
    
    # Lỗi (nếu có)
    "error": str (optional)
}
"""

# =====================================================================
# YÊUCẦU ĐẦU VÀO
# =====================================================================

REQUIREMENTS = """
1. DEPENDENCIES
   ────────────
   - opencv-python
   - numpy
   - ultralytics (YOLO)
   - torch (for YOLO)

2. MODELS
   ──────
   - Point Detection: YOLO model (best.pt) - REQUIRED
   - OCR Digit: YOLO model - OPTIONAL (mặc định dùng simple method)

3. HARDWARE
   ────────
   - CPU: Tối thiểu, nhưng chậm
   - GPU: Khuyến cáo (NVIDIA với CUDA support)
   - RAM: Tối thiểu 2GB, khuyến cáo 8GB+

4. INPUT IMAGE
   ──────────
   - Format: BGR (opencv format)
   - Kích thước: Tối thiểu 400x400 pixels
   - Chất lượng: Sudoku rõ ràng, ánh sáng tốt
"""

# =====================================================================
# PARAMETERS ĐIỀU CHỈNH
# =====================================================================

PARAMETERS = """
Các tham số có thể điều chỉnh:

1. Pipeline Level
   ──────────────
   - cell_size: Kích thước mỗi ô (64, 128, 256, ...)
     * Nhỏ hơn → nhanh nhưng ít chính xác
     * Lớn hơn → chậm nhưng chính xác hơn
   
   - device: "cpu" hoặc "cuda"
     * CUDA nhanh hơn nếu có GPU
   
   - verbose: True/False
     * Hiển thị chi tiết xử lý
   
   - use_easyocr: True (mặc định)
     * Sử dụng EasyOCR cho OCR digit
   
   - conf_threshold: 0.0-1.0 (mặc định 0.3)
     * OCR confidence threshold
     * Cao hơn → chỉ chấp nhận kết quả chắc chắn

2. Point Detector Level
   ───────────────────
   - conf: Confidence threshold (0.0-1.0)
     * Cao hơn → ít false positive
     * Thấp hơn → detect nhiều hơn
   
   - iou: IOU threshold (0.0-1.0)
   - imgsz: Input size cho YOLO (640, 1280, ...)

3. OCR Level (EasyOCR)
   ──────────────────
   - lang: Ngôn ngữ (mặc định "en")
     * "en": English (khuyến cáo)
     * "vi": Vietnamese
     * "ch_sim": Chinese Simplified
   
   - use_gpu: True/False
     * True: Sử dụng GPU (nhanh)
     * False: Sử dụng CPU (chậm nhưng không cần GPU)
   
   - conf_threshold: 0.0-1.0 (mặc định 0.3)
     * Confidence threshold để chấp nhận kết quả
     * Cao hơn (0.7) → chỉ chấp nhận kết quả rất chắc chắn
     * Thấp hơn (0.2) → chấp nhận kết quả ít chắc chắn

4. Solver Level
   ────────────
   - Không có tham số (backtracking pure)

Ví dụ điều chỉnh:
   pipeline.point_detector.set_conf(0.6)
   pipeline.ocr_digit.set_confidence_threshold(0.8)
   pipeline.ocr_digit.set_gpu(False)
"""

# =====================================================================
# TROUBLESHOOTING
# =====================================================================

TROUBLESHOOTING = """
Problem 1: "No objects detected" error
────────────────────────────────────
Nguyên nhân: YOLO không tìm được 4 góc
Giải pháp:
  1. Kiểm tra chất lượng ảnh (phải rõ, ánh sáng tốt)
  2. Tăng confidence threshold (set_conf thấp hơn)
  3. Đảm bảo model weights được tải đúng


Problem 2: "Expected 4 corners, detected N" error
──────────────────────────────────────────────
Nguyên nhân: Model detect nhiều hơn/ít hơn 4 điểm
Giải pháp:
  1. Crop ảnh để chỉ chứa sudoku grid
  2. Tăng IoU threshold (set_iou cao hơn)
  3. Kiểm tra model quality


Problem 3: OCR accuracy thấp
─────────────────────────────
Nguyên nhân: Model nhận diện chữ số không chính xác
Giải pháp:
  1. Sử dụng cell_size lớn hơn
  2. Huấn luyện model OCR với dataset tương tự
  3. Tăng confidence threshold OCR


Problem 4: Sudoku không giải được
───────────────────────────────
Nguyên nhân: 
  - Chữ số detect sai → puzzle input không hợp lệ
  - Puzzle input không có solution
Giải pháp:
  1. Kiểm tra detected_grid (so với ảnh gốc)
  2. Debug OCR accuracy
  3. Validate puzzle (phải có unique solution)


Problem 5: Chậm/GPU memory issue
────────────────────────────────
Nguyên nhân: Model quá lớn, batch quá lớn
Giải pháp:
  1. Giảm cell_size
  2. Xử lý từng ảnh thay vì batch
  3. Dùng CPU thay GPU (nếu VRAM hết)
"""

# =====================================================================
# FLOW ĐỀ XUẤT
# =====================================================================

RECOMMENDED_FLOW = """
Recommended Workflow:

1. PREPARATION
   ────────────
   ✓ Chuẩn bị YOLO model cho point detection (best.pt)
   ✓ Chuẩn bị ảnh sudoku (chất lượng tốt)
   ✓ (Optional) Chuẩn bị YOLO model cho OCR digit
   ✓ Cài đặt dependencies (pip install -r requirements.txt)

2. INITIALIZATION
   ──────────────
   ✓ Import pipeline
   ✓ Tạo SudokuSolverPipeline instance
   ✓ Warmup models (nếu dùng GPU)

3. PROCESSING
   ──────────
   ✓ Load ảnh
   ✓ Chạy pipeline.run()
   ✓ Kiểm tra result["success"]

4. RESULT ANALYSIS
   ────────────────
   ✓ Nếu success: Sử dụng result["solved_grid"]
   ✓ Nếu failed: Debug từng component
     - Kiểm tra detected corners
     - Kiểm tra aligned image
     - Kiểm tra detected_grid
     - Tùy chỉnh parameters

5. OPTIMIZATION
   ─────────────
   ✓ Điều chỉnh parameters dựa vào results
   ✓ Nếu cần OCR tốt hơn: Train OCR model
   ✓ Nếu cần nhanh hơn: Dùng GPU, giảm cell_size
   ✓ Nếu cần chính xác hơn: Tăng cell_size, điều chỉnh confidence
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("ARCHITECTURE")
    print("="*70)
    print(ARCHITECTURE)
    print("\n" + "="*70)
    print("CLASSES")
    print("="*70)
    print(CLASSES)
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print(USAGE)
    print("\n" + "="*70)
    print("RESULT STRUCTURE")
    print("="*70)
    print(RESULT_STRUCTURE)
    print("\n" + "="*70)
    print("REQUIREMENTS")
    print("="*70)
    print(REQUIREMENTS)
    print("\n" + "="*70)
    print("PARAMETERS")
    print("="*70)
    print(PARAMETERS)
    print("\n" + "="*70)
    print("TROUBLESHOOTING")
    print("="*70)
    print(TROUBLESHOOTING)
    print("\n" + "="*70)
    print("RECOMMENDED WORKFLOW")
    print("="*70)
    print(RECOMMENDED_FLOW)
