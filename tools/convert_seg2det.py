"""Convert segmentation (4 points) to YOLO detection (4 objects)."""
import os
from pathlib import Path
from PIL import Image
import numpy as np


def parse_seg_line(line):
    """Parse segmentation line: class_id x1 y1 x2 y2 x3 y3 x4 y4"""
    parts = line.strip().split()
    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    points = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
    return class_id, points


def expand_point(point, img_w, img_h, expand_px=10):
    """Expand point by expand_px pixels from center."""
    px_x = point[0] * img_w
    px_y = point[1] * img_h
    
    left = max(0, px_x - expand_px)
    right = min(img_w, px_x + expand_px)
    top = max(0, px_y - expand_px)
    bottom = min(img_h, px_y + expand_px)
    
    center_x = (left + right) / 2 / img_w
    center_y = (top + bottom) / 2 / img_h
    width = (right - left) / img_w
    height = (bottom - top) / img_h
    
    return center_x, center_y, width, height


def convert_segmentation_to_detection(project_root, expand_px=20):
    """Convert all segmentation labels to YOLO detection format."""
    labels_in = Path(project_root) / "labels"
    images_dir = Path(project_root) / "images"
    labels_out = Path(project_root) / "labels_detection"
    labels_out.mkdir(exist_ok=True)
    
    label_files = sorted(labels_in.glob("*.txt"))
    print(f"Converting {len(label_files)} files (4 objects per segmentation)...")
    
    for i, label_file in enumerate(label_files):
        img_name = label_file.stem
        img_file = images_dir / f"{img_name}.jpg"
        
        if not img_file.exists():
            continue
        
        img = Image.open(img_file)
        img_w, img_h = img.size
        
        with open(label_file) as f:
            line = f.readline()
        
        class_id, points = parse_seg_line(line)
        
        # Convert each of 4 points to detection object
        det_lines = []
        point_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
        
        for j, point in enumerate(points):
            cx, cy, w, h = expand_point(point, img_w, img_h, expand_px)
            det_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Write output
        out_file = labels_out / label_file.name
        with open(out_file, 'w') as f:
            f.writelines(det_lines)
        
        if (i + 1) % 100 == 0:
            print(f"  ✓ {i + 1}/{len(label_files)}")
    
    print(f"✅ Converted to: {labels_out}")
    return labels_out


if __name__ == "__main__":
    project_root = r"c:\Users\lequa\Desktop\sudoku-solver-ai\models\project-5-at-2026-07-17-07-39-de48885f"
    convert_segmentation_to_detection(project_root, expand_px=10)
