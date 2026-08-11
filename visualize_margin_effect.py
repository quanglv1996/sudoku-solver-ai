"""
visualize_margin_effect.py

Visualize the difference between cutting with and without margin
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from preprocessing import Preprocessing

print("=" * 60)
print("VISUALIZE MARGIN EFFECT ON CELL CUTTING")
print("=" * 60)

# Load aligned image (or create dummy if not exists)
aligned_img_path = Path("debug_output")
if aligned_img_path.exists():
    # Look for aligned image in debug output
    for subdir in sorted(aligned_img_path.iterdir(), reverse=True):
        if subdir.is_dir():
            aligned_imgs = list(subdir.glob("03_aligned_image.jpg"))
            if aligned_imgs:
                print(f"\n✅ Found aligned image: {aligned_imgs[0]}")
                aligned_image = cv2.imread(str(aligned_imgs[0]))
                
                if aligned_image is not None:
                    print(f"Image size: {aligned_image.shape}")
                    
                    # Test with different margins
                    margins = [0, 4, 6, 8]
                    
                    for margin in margins:
                        print(f"\n[Testing margin={margin} pixels]")
                        
                        # Without margin
                        preproc = Preprocessing(cell_size=64, grid_size=9, margin=margin)
                        result = preproc.crop(aligned_image)
                        cells = result.grid_cells
                        
                        # Get first cell and show stats
                        first_cell = cells[0]
                        print(f"  - Cell shape: {first_cell.shape}")
                        print(f"  - Cell min/max pixel: {first_cell.min()}/{first_cell.max()}")
                        
                        # Create comparison grid (first 4 cells)
                        grid_img = np.zeros((128, 128, 3), dtype=np.uint8)
                        for i in range(2):
                            for j in range(2):
                                cell = cells[i * 9 + j]
                                if len(cell.shape) == 2:
                                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
                                grid_img[i*64:(i+1)*64, j*64:(j+1)*64] = cell
                        
                        # Save comparison
                        output_path = Path(f"margin_comparison_{margin}.jpg")
                        cv2.imwrite(str(output_path), grid_img)
                        print(f"  ✅ Saved: {output_path}")
                    
                    print("\n" + "=" * 60)
                    print("COMPARISON SAVED")
                    print("=" * 60)
                    print("\nFiles generated:")
                    for margin in margins:
                        print(f"  - margin_comparison_{margin}.jpg")
                    print("\nMargin effect:")
                    print("  - margin=0:  Include borders, more noise")
                    print("  - margin=4:  Clean interior, no borders ✓")
                    print("  - margin=6+: May lose content if cell too small")
                    sys.exit(0)
                    
print("\n❌ Could not find aligned image in debug output")
print("Run Flask and upload an image first to generate debug images")
