"""
test_debug_images.py

Test debug image saving feature
"""
import sys
from pathlib import Path
import cv2

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from pipeline import SudokuSolverPipeline

print("=" * 60)
print("TEST DEBUG IMAGE SAVING")
print("=" * 60)

# Check model
point_model = Path(__file__).parent / "backend" / "weights" / "best.pt"
if not point_model.exists():
    print(f"❌ Model not found: {point_model}")
    sys.exit(1)

print(f"✅ Model found: {point_model}")

# Initialize pipeline with debug enabled
print("\n[1/3] Initializing pipeline with debug=True...")
try:
    pipeline = SudokuSolverPipeline(
        point_detection_model=str(point_model),
        cell_size=64,
        device="cpu",
        use_easyocr=True,
        conf_threshold=0.3,
        save_debug=True,  # Enable debug
        debug_dir="debug_output"
    )
    print("✅ Pipeline initialized")
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with sample image
print("\n[2/3] Loading test image...")
test_images = list(Path("imgs").glob("*.jpg")) + list(Path("imgs").glob("*.png"))
if not test_images:
    print("❌ No test images found in 'imgs/'")
    sys.exit(1)

test_image_path = test_images[0]
image = cv2.imread(str(test_image_path))
if image is None:
    print(f"❌ Could not read image: {test_image_path}")
    sys.exit(1)

print(f"✅ Loaded: {test_image_path} ({image.shape})")

# Run pipeline
print("\n[3/3] Running pipeline with debug image saving...")
try:
    result = pipeline.run(image)
    
    if result.get('success'):
        print("\n✅ PIPELINE SUCCESSFUL!")
        print(f"\nDebug images saved to: debug_output/")
        print("\nFiles generated:")
        debug_dir = Path("debug_output")
        if debug_dir.exists():
            for subdir in sorted(debug_dir.iterdir()):
                if subdir.is_dir():
                    print(f"\n  📁 {subdir.name}/")
                    for img_file in sorted(subdir.glob("*.jpg")):
                        size_kb = img_file.stat().st_size / 1024
                        print(f"     - {img_file.name} ({size_kb:.1f} KB)")
    else:
        print(f"⚠️  Pipeline ran but did not solve")
        if 'error' in result:
            print(f"Error: {result['error']}")
            
except Exception as e:
    print(f"❌ Pipeline error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
