"""
config.py

Configuration file for Sudoku Solver Web Application
Mọi giá trị đều có thể ghi đè bằng biến môi trường (hoặc file .env)
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    """Đọc .env đơn giản (KEY=VALUE), không ghi đè biến môi trường đã có"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


# =====================================================================
# Paths
# =====================================================================

BACKEND_DIR = PROJECT_ROOT / "backend"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
CERT_DIR = PROJECT_ROOT / "certs"
DEBUG_DIR = PROJECT_ROOT / os.environ.get("DEBUG_DIR", "debug_output")

# =====================================================================
# Server
# =====================================================================

SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("HTTP_PORT", os.environ.get("SERVER_PORT", 5000)))
# HTTPS bắt buộc để trình duyệt điện thoại cho phép mở camera (getUserMedia)
HTTPS_ENABLED = _bool("HTTPS_ENABLED", True)
HTTPS_PORT = int(os.environ.get("HTTPS_PORT", 5443))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
# IP/tên máy mà điện thoại dùng để truy cập (phân tách bằng dấu phẩy). Cần khi chạy
# trong Docker: IP tự dò được là IP nội bộ của container, không truy cập được từ LAN
PUBLIC_HOSTS = [h.strip() for h in os.environ.get("PUBLIC_HOSTS", "").split(",") if h.strip()]

# =====================================================================
# Models
# =====================================================================

POINT_MODEL = Path(os.environ.get("POINT_MODEL_PATH", BACKEND_DIR / "weights" / "best.pt"))
if not POINT_MODEL.is_absolute():
    POINT_MODEL = PROJECT_ROOT / POINT_MODEL
POINT_CONF = float(os.environ.get("POINT_CONF", 0.25))

# auto | cuda | cpu
DEVICE = os.environ.get("DEVICE", "auto")

OCR_MODEL_NAME = os.environ.get("OCR_MODEL_NAME", "PP-OCRv6_medium_rec")
OCR_CONF_THRESHOLD = float(os.environ.get("OCR_CONF_THRESHOLD", 0.5))
CELL_SIZE = int(os.environ.get("CELL_SIZE", 64))

# =====================================================================
# Queue manager
# =====================================================================

MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 32))
JOB_TTL = float(os.environ.get("JOB_TTL", 600))
REALTIME_TIMEOUT = float(os.environ.get("REALTIME_TIMEOUT", 5))
# /api/solve (đồng bộ): chờ tối đa N giây, gồm cả thời gian tải model lúc khởi động
SYNC_TIMEOUT = float(os.environ.get("SYNC_TIMEOUT", 120))

# =====================================================================
# Upload
# =====================================================================

MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE", 15 * 1024 * 1024))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", 1280))
REALTIME_MAX_SIZE = int(os.environ.get("REALTIME_MAX_SIZE", 800))

# =====================================================================
# Screen auto-solve (điều khiển chuột/bàn phím của máy chạy server)
# =====================================================================

SCREEN_CONTROL = _bool("SCREEN_CONTROL", True)
# True: chỉ cho phép kích hoạt từ chính máy chạy server
SCREEN_LOCAL_ONLY = _bool("SCREEN_LOCAL_ONLY", False)

# =====================================================================
# Misc
# =====================================================================

SAVE_DEBUG = _bool("SAVE_DEBUG", False)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


def get_config_summary():
    return {
        "server": {"host": SERVER_HOST, "http_port": HTTP_PORT,
                   "https": HTTPS_ENABLED, "https_port": HTTPS_PORT},
        "model": {"point_detection": str(POINT_MODEL), "device": DEVICE,
                  "ocr_model": OCR_MODEL_NAME, "ocr_conf": OCR_CONF_THRESHOLD},
        "queue": {"max_queue": MAX_QUEUE, "job_ttl": JOB_TTL},
        "screen": {"enabled": SCREEN_CONTROL, "local_only": SCREEN_LOCAL_ONLY},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_config_summary(), indent=2))
