"""
config.py

Configuration file for Sudoku Solver Web Application
Tập tin cấu hình cho ứng dụng web Sudoku Solver
"""

import os
from pathlib import Path

# =====================================================================
# Project Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).parent
BACKEND_DIR = PROJECT_ROOT / "backend"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Create uploads directory if not exists
UPLOADS_DIR.mkdir(exist_ok=True)

# =====================================================================
# Server Configuration
# =====================================================================

# Flask
FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
FLASK_DEBUG = FLASK_ENV == 'development'
SERVER_HOST = os.environ.get('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.environ.get('SERVER_PORT', 5000))

# CORS
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

# =====================================================================
# Model Configuration
# =====================================================================

# Point Detection Model (YOLO)
POINT_MODEL = os.environ.get(
    'POINT_MODEL_PATH',
    str(BACKEND_DIR / "weights" / "best.pt")
)

# Device (cuda/cpu)
DEVICE = os.environ.get('DEVICE', 'cpu')

# =====================================================================
# OCR Configuration
# =====================================================================

# EasyOCR Settings
OCR_LANG = os.environ.get('OCR_LANG', 'en')
OCR_GPU = os.environ.get('OCR_GPU', 'false').lower() == 'true'
OCR_CONF_THRESHOLD = float(os.environ.get('OCR_CONF_THRESHOLD', 0.3))

# =====================================================================
# File Upload Configuration
# =====================================================================

# Max file size (bytes)
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 10 * 1024 * 1024))  # 10MB

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Image resize max size (pixels)
MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', 800))

# =====================================================================
# Pipeline Configuration
# =====================================================================

# Cell size for preprocessing
CELL_SIZE = int(os.environ.get('CELL_SIZE', 64))

# Verbose logging
VERBOSE = os.environ.get('VERBOSE', 'false').lower() == 'true'

# =====================================================================
# Timeout Configuration
# =====================================================================

# Request timeout (seconds)
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', 60))

# Processing timeout (seconds)
PROCESSING_TIMEOUT = int(os.environ.get('PROCESSING_TIMEOUT', 120))

# =====================================================================
# Logging Configuration
# =====================================================================

# Log level
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Log file
LOG_FILE = os.environ.get('LOG_FILE', 'sudoku_solver.log')

# =====================================================================
# Feature Flags
# =====================================================================

# Enable debug mode
DEBUG_MODE = FLASK_DEBUG

# Enable result caching
CACHE_RESULTS = os.environ.get('CACHE_RESULTS', 'false').lower() == 'true'

# Enable result download
ENABLE_DOWNLOAD = os.environ.get('ENABLE_DOWNLOAD', 'true').lower() == 'true'

# =====================================================================
# Display Configuration
# =====================================================================

# UI Theme (dark/light)
UI_THEME = os.environ.get('UI_THEME', 'dark')

# Page title
PAGE_TITLE = os.environ.get('PAGE_TITLE', 'Sudoku Solver AI - High Tech')

# Page subtitle
PAGE_SUBTITLE = os.environ.get('PAGE_SUBTITLE', 'Upload a sudoku image, let AI solve it instantly')

# =====================================================================
# Utility Functions
# =====================================================================

def get_config_summary():
    """Get summary of current configuration"""
    return {
        'server': {
            'host': SERVER_HOST,
            'port': SERVER_PORT,
            'debug': FLASK_DEBUG
        },
        'model': {
            'point_detection': POINT_MODEL,
            'device': DEVICE
        },
        'ocr': {
            'language': OCR_LANG,
            'use_gpu': OCR_GPU,
            'confidence': OCR_CONF_THRESHOLD
        },
        'upload': {
            'max_size_mb': MAX_FILE_SIZE / (1024 * 1024),
            'max_image_size': MAX_IMAGE_SIZE,
            'allowed_extensions': list(ALLOWED_EXTENSIONS)
        },
        'pipeline': {
            'cell_size': CELL_SIZE,
            'verbose': VERBOSE
        }
    }

if __name__ == '__main__':
    import json
    config = get_config_summary()
    print(json.dumps(config, indent=2))
