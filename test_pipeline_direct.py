"""
test_pipeline_direct.py

Direct test of pipeline - check where it fails
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

print("=" * 60)
print("DIRECT PIPELINE TEST")
print("=" * 60)

# Step 1: Test imports
print("\n[1/4] Testing imports...")
try:
    from pipeline import SudokuSolverPipeline
    print("✅ SudokuSolverPipeline imported successfully")
except Exception as e:
    print(f"❌ Failed to import SudokuSolverPipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Check model file
print("\n[2/4] Checking model file...")
point_model = Path(__file__).parent / "backend" / "weights" / "best.pt"
if point_model.exists():
    size_mb = point_model.stat().st_size / (1024 * 1024)
    print(f"✅ Model file found: {point_model} ({size_mb:.1f} MB)")
else:
    print(f"❌ Model file not found at: {point_model}")
    sys.exit(1)

# Step 3: Initialize pipeline
print("\n[3/4] Initializing pipeline...")
try:
    logger.info("Creating SudokuSolverPipeline...")
    pipeline = SudokuSolverPipeline(
        point_detection_model=str(point_model),
        cell_size=64,
        device="cpu",
        use_easyocr=True,
        conf_threshold=0.3
    )
    pipeline.set_verbose(True)
    print("✅ Pipeline initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize pipeline:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test with sample image (if exists)
print("\n[4/4] Testing with sample image...")
test_images = list(Path("imgs").glob("*.jpg")) + list(Path("imgs").glob("*.png"))
if test_images:
    test_image_path = test_images[0]
    print(f"Found test image: {test_image_path}")
    try:
        image = cv2.imread(str(test_image_path))
        if image is None:
            print(f"❌ Could not read image: {test_image_path}")
        else:
            print(f"✅ Image loaded: {image.shape}")
            
            print("\n⏳ Running pipeline on test image...")
            result = pipeline.run(image)
            
            if result.get('success'):
                print("✅ PIPELINE TEST PASSED!")
                print(f"   - Detected grid:\n{result['detected_grid']}")
                print(f"   - Solved grid:\n{result['solved_grid']}")
            else:
                print(f"⚠️  Pipeline ran but failed to solve: {result.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  No test images found in 'imgs' directory")
    print("   But pipeline initialization passed!")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
