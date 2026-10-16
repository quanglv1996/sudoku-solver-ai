"""
run_web.py

Kiểm tra môi trường rồi chạy web app (HTTP + HTTPS cho điện thoại)
    python run_web.py
"""

import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

REQUIRED = {
    "flask": "flask",
    "flask_cors": "flask-cors",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "paddleocr": "paddleocr",
    "paddle": "paddlepaddle / paddlepaddle-gpu",
}
OPTIONAL = {
    "cryptography": "HTTPS cho điện thoại (camera)",
    "mss": "chế độ Màn hình PC (chụp màn hình)",
    "pyautogui": "chế độ Màn hình PC (tự điền)",
}


def check_requirements() -> bool:
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print("❌ Thiếu package:", ", ".join(missing))
        print("   pip install -r requirements.txt")
        return False
    print("✅ Package bắt buộc: OK")
    for mod, purpose in OPTIONAL.items():
        if importlib.util.find_spec(mod) is None:
            print(f"⚠️  Thiếu '{mod}' → tắt {purpose}")
    return True


def check_model() -> bool:
    import config

    if not config.POINT_MODEL.exists():
        print(f"❌ Không thấy model: {config.POINT_MODEL}")
        return False
    print(f"✅ Model: {config.POINT_MODEL}")
    return True


def main():
    print("\n🧩 Sudoku Solver AI - Web Application\n")
    if not check_requirements() or not check_model():
        sys.exit(1)

    import app
    app.serve()


if __name__ == "__main__":
    main()
