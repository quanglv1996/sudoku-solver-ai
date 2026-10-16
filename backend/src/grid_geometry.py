"""
grid_geometry.py

Tiện ích hình học cho lưới sudoku:
- Sắp xếp 4 góc [TL, TR, BR, BL]
- Kiểm tra tứ giác hợp lệ (lồi, đủ lớn, không quá méo)
- Phát hiện lưới bằng contour (fallback khi YOLO thất bại; rất tốt với ảnh chụp màn hình)
- Chiếu điểm từ ảnh căn chỉnh về ảnh gốc
"""

from typing import Optional

import cv2
import numpy as np


def is_light_on_dark(gray: np.ndarray) -> bool:
    """
    Nét (chữ, đường kẻ) sáng trên nền tối - giao diện dark mode?

    Nét luôn là phần thiểu số: so độ lệch của phần sáng nhất và tối nhất so với
    nền (median). Không dùng riêng độ sáng trung bình vì ảnh giấy chụp thiếu
    sáng cũng có nền tối nhưng chữ vẫn tối hơn nền.
    """
    m = float(np.median(gray))
    lo, hi = np.percentile(gray, (3, 97))
    return m < 140 and (hi - m) > 1.5 * (m - lo)


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 điểm thành [TL, TR, BR, BL] theo góc quanh tâm
    (ổn định kể cả khi lưới bị xoay ~45°)
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # thuận chiều kim đồng hồ (trục y hướng xuống)
    start = int(np.argmin(pts.sum(axis=1)))  # TL có x+y nhỏ nhất
    return np.roll(pts, -start, axis=0)


def quad_area(quad: np.ndarray) -> float:
    return float(cv2.contourArea(np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)))


def quad_is_valid(
    quad: np.ndarray,
    image_shape,
    min_area_ratio: float = 0.02,
    max_side_ratio: float = 3.0,
) -> bool:
    """Tứ giác lồi, đủ lớn so với ảnh và các cạnh không chênh lệch quá nhiều"""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    if not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
        return False
    h, w = image_shape[:2]
    if quad_area(quad) < min_area_ratio * h * w:
        return False
    sides = [np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 0 and max(sides) / min(sides) <= max_side_ratio


def detect_grid_contour(image: np.ndarray, min_area_ratio: float = 0.02) -> Optional[np.ndarray]:
    """
    Tìm lưới sudoku là contour 4 cạnh lớn nhất, gần vuông

    Returns:
        4 góc [TL, TR, BR, BL] hoặc None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if is_light_on_dark(gray):
        # Dark mode: đảo về nét tối trên nền sáng, nếu không contour sẽ bám vào
        # "quầng tối" bên ngoài đường kẻ sáng và lệch ra ngoài lưới
        gray = 255 - gray
    scale = 1.0
    if max(gray.shape) > 1600:
        scale = 1600 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

    # RETR_LIST: xét cả viền trong của khung lưới - trên ảnh chụp, viền ngoài
    # hay dính vào chữ/đường kẻ xung quanh nên không còn là tứ giác
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    best, best_area = None, 0.0
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * h * w:
            break
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue
        quad = order_corners(approx.reshape(4, 2))
        if not quad_is_valid(quad, gray.shape, min_area_ratio, max_side_ratio=1.6):
            continue
        if area > best_area:
            best, best_area = quad, area

    if best is None:
        return None
    return (best / scale).astype(np.float32)


def project_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Áp dụng perspective transform cho mảng điểm (N, 2)"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)


def cell_centers_in_source(inv_matrix: np.ndarray, aligned_size: int) -> np.ndarray:
    """
    Tâm 81 ô (theo thứ tự hàng) chiếu về ảnh gốc

    Returns:
        (81, 2) toạ độ trong ảnh gốc
    """
    step = aligned_size / 9.0
    grid = np.array(
        [[(c + 0.5) * step, (r + 0.5) * step] for r in range(9) for c in range(9)],
        dtype=np.float32,
    )
    return project_points(grid, inv_matrix)
