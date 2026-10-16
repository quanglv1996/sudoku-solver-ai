# 🧩 Sudoku Vision AI

Giải sudoku từ camera/ảnh/màn hình bằng YOLO (dò 4 góc lưới) + PaddleOCR (đọc số) + solver bitmask.
Truy cập qua trình duyệt trên máy tính hoặc điện thoại (cùng Wi-Fi).

## Ba chế độ

| Chế độ | Mô tả |
|---|---|
| 📷 **Chụp & Giải** | Chụp bằng camera (hoặc tải ảnh / kéo-thả / Ctrl+V). Giao diện "quét" hiển thị từng bước thật của server: dò khung → căn chỉnh → đọc số → giải, rồi chữ số lời giải hiện lên đúng phối cảnh trên ảnh gốc. |
| 🎥 **Realtime** | Chỉ cần quay camera: lời giải được vẽ thẳng lên hình và bám theo lưới khi di chuyển (~12 fps, ~50ms/frame trên GPU). |
| 🖥️ **Màn hình PC** | Chụp màn hình máy chạy server, tìm lưới sudoku trên trang web, giải và **tự click + gõ số** vào từng ô trống. Có thể kích hoạt từ điện thoại. |

## Chạy

```bash
conda activate yolo11_paddle        # Python 3.10
pip install -r requirements.txt
python run_web.py
```

Server in ra địa chỉ truy cập:

- Máy tính: `http://localhost:5000`
- Điện thoại: `https://<IP-LAN>:5443` — trình duyệt điện thoại **chỉ cho mở camera qua HTTPS**, nên server tự sinh chứng chỉ tự ký (thư mục `certs/`). Lần đầu mở, chọn *Nâng cao → Tiếp tục*.

Model được tải nền khi khởi động (~10–15s); trạng thái hiện ở góc trên giao diện.

## Kiến trúc: Manager + Queue

```
Browser / Phone ──HTTP──▶ Flask (nhiều thread: chỉ nhận/trả dữ liệu)
                              │ submit()
                              ▼
                     TaskManager (PriorityQueue)
                     ưu tiên: realtime > screen > photo
                              │
                              ▼
               1 worker thread sở hữu SudokuEngine (YOLO + PaddleOCR trên GPU)
```

- Model không thread-safe → chỉ 1 worker giữ model, xử lý tuần tự; HTTP thread không bao giờ chạm vào model.
- **Realtime "latest wins"**: mỗi session chỉ giữ frame mới nhất đang chờ, frame cũ bị bỏ → không dồn độ trễ. Client luôn chỉ có 1 request đang bay nên tự điều tiết theo tốc độ server.
- **RealtimeTracker**: OCR + giải chỉ chạy khi gặp đề mới; các frame sau chỉ dò góc + so mẫu ô trống, dùng lại lời giải đã cache.
- Job lưu `stage` (detect/align/ocr/solve/render/fill) → frontend hiển thị tiến độ thật; hỗ trợ huỷ, giới hạn hàng đợi, tự dọn job cũ.

### API

| Method | Path | Mô tả |
|---|---|---|
| GET | `/api/health` | Trạng thái engine, thiết bị, hàng đợi, URL truy cập |
| POST | `/api/jobs` | `multipart image` → job ảnh (202) |
| GET | `/api/jobs/<id>?wait=ms` | Tiến độ/kết quả (long-poll tới khi đổi stage) |
| DELETE | `/api/jobs/<id>` | Huỷ job |
| POST | `/api/realtime` | `multipart frame, session` → overlay (góc lưới + chữ số từng ô) |
| GET | `/api/screen/monitors` | Danh sách màn hình |
| POST | `/api/screen` | `{monitor, fill, type_delay}` → job tự giải màn hình (202) |

## Cấu trúc

```
app.py                  Flask routes + chạy HTTP/HTTPS song song
config.py               Cấu hình (biến môi trường / .env)
run_web.py              Kiểm tra môi trường rồi chạy
backend/src/
  task_manager.py       Hàng đợi ưu tiên + worker + trạng thái job
  engine.py             SudokuEngine + handler cho từng loại job
  pipeline.py           Detect → Align → OCR → Solve → overlay/render
  point_detection.py    YOLO dò 4 góc (chọn tứ giác hợp lệ tốt nhất)
  grid_geometry.py      Sắp xếp góc, kiểm tra tứ giác, dò lưới bằng contour
  ocr_digit.py          Phân tích ô trống + PaddleOCR TextRecognition
  sudoku_solver.py      Solver bitmask + MRV, phát hiện số trùng
  realtime.py           Tracker có cache cho chế độ realtime
  screen_solver.py      Chụp màn hình (mss) + tự điền (pyautogui)
  tls.py                Chứng chỉ HTTPS tự ký cho LAN
backend/weights/best.pt Model YOLO dò 4 góc
static/js/              main.js, photo.js, live.js, screen.js, common.js (ES modules)
tests/                  pytest (test_core.py nhanh, test_integration.py dùng model thật)
tools/convert_seg2det.py  Chuyển nhãn segmentation 4 điểm → 4 box YOLO detection (huấn luyện model góc)
imgs/test.jpg           Ảnh mẫu dùng trong test
```

## Cấu hình chính (`.env`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `HTTP_PORT` / `HTTPS_PORT` | `5000` / `5443` | Cổng |
| `HTTPS_ENABLED` | `true` | Bật HTTPS cho điện thoại |
| `OCR_MODEL_NAME` | `PP-OCRv6_medium_rec` | Model nhận dạng của PaddleOCR |
| `OCR_CONF_THRESHOLD` | `0.5` | Ngưỡng tin cậy chữ số |
| `SCREEN_CONTROL` | `true` | Cho phép chế độ Màn hình PC |
| `SCREEN_LOCAL_ONLY` | `false` | Chỉ cho kích hoạt từ chính máy server |
| `SAVE_DEBUG` | `false` | Lưu ảnh từng bước vào `debug_output/` |

## Kiểm thử

```bash
pytest tests -v                  # toàn bộ (load model thật, ~30s trên GPU)
pytest tests -v -m "not model"   # chỉ test nhanh: solver, hình học, TaskManager
```

- `tests/test_core.py`: solver (đề khó, đề mâu thuẫn, đề vô nghiệm), sắp xếp góc, dò lưới contour, dark mode, TaskManager (ưu tiên, latest-wins, huỷ, hàng đợi đầy, lỗi tải model).
- `tests/test_integration.py`: ảnh chụp biến dạng (xoay ±35°, nghiêng, tối/sáng, mờ, nhiễu, nén JPEG, chụp xa), ảnh màn hình (dark mode, tô sáng, ghi chú bút chì, nhiều cỡ lưới) kèm kiểm tra toạ độ click, realtime tracker, toàn bộ HTTP API và 6 client đồng thời.

Pipeline "xác minh bằng cách giải": nếu khung lưới đầu tiên cho ra đề không giải được, hệ thống thử ứng viên tiếp theo (YOLO → contour → YOLO trên ảnh xoay).

## Lưu ý chế độ Màn hình PC

- Điều khiển chuột/bàn phím của **máy chạy server**. Dừng khẩn cấp: đưa chuột vào một góc màn hình.
- Bất kỳ ai truy cập được server trong mạng LAN đều có thể kích hoạt; đặt `SCREEN_LOCAL_ONLY=true` hoặc `SCREEN_CONTROL=false` nếu không muốn.
- Trang sudoku phải cho phép chọn ô bằng click và nhập bằng phím số (đa số trang đều vậy). Nếu trang phản hồi chậm, chọn tốc độ điền "Vừa"/"Chậm".
