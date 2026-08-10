import cv2
import numpy as np
import easyocr

# 1. Khởi tạo EasyOCR với GPU
reader = easyocr.Reader(['en'], gpu=True)

def process_sudoku_grid(image_path):
    # 2. Đọc ảnh và tiền xử lý
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (450, 450)) # Resize về kích thước chuẩn dễ chia ô (450x450)

    cells = []
    grid_size = 9
    cell_width = 450 // grid_size  # Mỗi ô rộng 50px
    cell_height = 450 // grid_size # Mỗi ô cao 50px

    # 3. Cắt bảng thành 81 ô vuông và loại bỏ viền đen
    for r in range(grid_size):
        for c in range(grid_size):
            y1, y2 = r * cell_height, (r + 1) * cell_height
            x1, x2 = c * cell_width, (c + 1) * cell_width
            
            cell = img[y1:y2, x1:x2]
            
            # Cắt bỏ 12% viền xung quanh để loại bỏ đường lưới Sudoku (Tránh nhiễu OCR)
            margin_h = int(cell_height * 0.12)
            margin_w = int(cell_width * 0.12)
            cell_cropped = cell[margin_h:-margin_h, margin_w:-margin_w]
            
            cells.append(cell_cropped)

    # 4. Đẩy toàn bộ 81 ô vào GPU xử lý cùng lúc (Batch Inference)
    # Tăng batch_size lên 81 để chạy 1 lần duy nhất
    batch_results = reader.readtext_batched(
        cells,
        batch_size=81,
        allowlist='123456789',
        detail=0,
        mag_ratio=1.2
    )

    # 5. Chuyển kết quả thành ma trận Sudoku 9x9 (Ô trống = 0)
    sudoku_matrix = np.zeros((9, 9), dtype=int)
    
    for idx, res in enumerate(batch_results):
        row = idx // 9
        col = idx % 9
        if len(res) > 0:
            sudoku_matrix[row][col] = int(res[0])
            
    return sudoku_matrix

# --- CHẠY THỬ NGHIỆM ---
# Thay 'sudoku.jpg' bằng đường dẫn ảnh của bạn
matrix = process_sudoku_grid(r'D:\Projects-Persional\sudoku-solver-ai\input\1.png')
print("Ma trận Sudoku nhận diện được:")
print(matrix)
