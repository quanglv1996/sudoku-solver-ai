"""
run_web.py

Simple script to run the web application
Tập lệnh đơn giản để chạy ứng dụng web
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'flask',
        'flask_cors',
        'cv2',
        'numpy',
        'easyocr',
        'ultralytics'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All requirements satisfied")
    return True

def check_model():
    """Check if YOLO model exists"""
    model_path = project_root / "backend" / "weights" / "best.pt"
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        print("Download the YOLO model and place it at: backend/weights/best.pt")
        return False
    
    print(f"✅ Model found: {model_path}")
    return True

def main():
    print("\n" + "="*60)
    print("🧩 Sudoku Solver AI - Web Application")
    print("="*60 + "\n")
    
    # Check requirements
    print("Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    print()
    
    # Check model
    print("Checking model...")
    if not check_model():
        print("Warning: Continuing without model (some features will fail)")
    
    print()
    
    # Run Flask app
    print("Starting Flask server...")
    print("-" * 60)
    print("Opening app at: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    print()
    
    try:
        from app import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=True
        )
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
