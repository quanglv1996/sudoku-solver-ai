"""
pipeline.py

Pipeline: Kết hợp toàn bộ các lớp để giải sudoku từ ảnh
Flow: Ảnh gốc → Detect 4 góc → Align → Đọc lưới (OCR) → Solve → Overlay

Pipeline KHÔNG thread-safe (model YOLO/Paddle) → chỉ được dùng bởi 1 worker
của TaskManager.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from point_detection import PointDetection
from grid_geometry import detect_grid_contour, order_corners, project_points, quad_is_valid
from alignment import Alignment
from ocr_digit import OCRDigit
from sudoku_solver import SudokuSolver, SudokuValidator

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, Dict[str, Any]], None]

STAGES = ["detect", "align", "ocr", "solve", "render"]


def resolve_device(device: str) -> str:
    """'auto' → 'cuda' nếu có GPU, ngược lại 'cpu'"""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class SudokuSolverPipeline:
    """
    Pipeline hoàn chỉnh: Từ ảnh sudoku đến kết quả giải

    Các bước:
    1. Detect 4 góc của sudoku (YOLO, fallback contour)
    2. Align ảnh dựa trên 4 góc
    3. Đọc chữ số (phân tích ô + PaddleOCR)
    4. Giải sudoku
    5. Tính overlay (vị trí từng ô trên ảnh gốc) / vẽ lời giải
    """

    def __init__(
        self,
        point_detection_model: str,
        ocr_model: Optional[str] = None,
        cell_size: int = 64,
        device: str = "auto",
        use_easyocr: bool = True,
        conf_threshold: float = 0.5,
        save_debug: bool = False,
        debug_dir: str = "debug_output",
        ocr_model_name: str = "PP-OCRv6_medium_rec",
        point_conf: float = 0.25,
    ):
        """
        Args:
            point_detection_model: Đường dẫn YOLO model detect góc
            ocr_model: (Deprecated, bỏ qua)
            cell_size: Kích thước mỗi ô sau khi align
            device: "cuda" / "cpu" / "auto"
            use_easyocr: (Deprecated, bỏ qua - dùng PaddleOCR)
            conf_threshold: Ngưỡng tin cậy OCR
            save_debug: Lưu ảnh debug ở từng bước
            debug_dir: Thư mục lưu ảnh debug
            ocr_model_name: Model recognition của PaddleOCR
            point_conf: Ngưỡng confidence của YOLO
        """
        self.device = resolve_device(device)
        self.cell_size = cell_size
        self.aligned_size = cell_size * 9
        self.save_debug = save_debug
        self.debug_dir = Path(debug_dir)
        self.verbose = False

        logger.info("Loading YOLO corner model on %s ...", self.device)
        self.point_detector = PointDetection(
            model_path=point_detection_model, device=self.device, conf=point_conf
        )
        self.alignment = Alignment(output_size=(self.aligned_size, self.aligned_size))

        logger.info("Loading PaddleOCR recognition model (%s) ...", ocr_model_name)
        self.ocr_digit = OCRDigit(
            use_gpu=(self.device == "cuda"),
            conf_threshold=conf_threshold,
            model_name=ocr_model_name,
        )
        self.solver = SudokuSolver()

    # =====================================================================
    # Main API
    # =====================================================================

    def candidate_quads(self, image: np.ndarray, prefer: str = "yolo", thorough: bool = False):
        """
        Sinh lần lượt các ứng viên khung lưới (4 góc), bỏ qua ứng viên trùng nhau

        Args:
            prefer: "yolo" (ảnh chụp camera) hoặc "contour" (ảnh chụp màn hình)
            thorough: thêm ứng viên từ ảnh xoay (chậm hơn; không dùng cho realtime)

        Yields:
            (corners (4,2), method)
        """
        seen: List[np.ndarray] = []
        tol = 0.02 * max(image.shape[:2])

        def fresh(quad):
            if any(np.abs(quad - q).max() < tol for q in seen):
                return False
            seen.append(quad)
            return True

        order = ["yolo", "contour"] if prefer == "yolo" else ["contour", "yolo"]
        for method in order:
            try:
                quad = self.point_detector.predict(image) if method == "yolo" else detect_grid_contour(image)
            except ValueError as e:
                logger.debug("locate[%s] failed: %s", method, e)
                quad = None
            if quad is not None and fresh(quad):
                yield quad, method

        if thorough:
            for quad, method in self._rotated_quads(image):
                if fresh(quad):
                    yield quad, method

    def locate(self, image: np.ndarray, prefer: str = "yolo", thorough: bool = False):
        """Ứng viên khung lưới đầu tiên: (corners | None, method)"""
        return next(self.candidate_quads(image, prefer, thorough), (None, None))

    def _rotated_quads(self, image: np.ndarray):
        """Model YOLO chỉ quen lưới gần thẳng đứng → xoay ảnh về các góc khác rồi dò"""
        h, w = image.shape[:2]
        pad = int(0.2 * max(h, w))
        padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        ph, pw = padded.shape[:2]
        for angle in (12, -12, 24, -24, 36, -36):
            M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
            rotated = cv2.warpAffine(padded, M, (pw, ph), borderMode=cv2.BORDER_REPLICATE)
            try:
                corners = self.point_detector.predict(rotated)
            except ValueError:
                continue
            inv = cv2.invertAffineTransform(M)
            back = order_corners(cv2.transform(corners.reshape(-1, 1, 2), inv).reshape(4, 2) - pad)
            if quad_is_valid(back, image.shape):
                yield back.astype(np.float32), f"yolo-rot{angle:+d}"

    def run(
        self,
        image: np.ndarray,
        progress: Optional[ProgressFn] = None,
        prefer: str = "yolo",
        max_attempts: int = 4,
    ) -> Dict[str, Any]:
        """
        Chạy toàn bộ pipeline

        "Xác minh bằng cách giải": nếu khung lưới tìm được cho ra đề không giải
        được (thường do YOLO bắt nhầm điểm), thử ứng viên khung tiếp theo.

        Returns:
            Dict chứa: success, stage (bước dừng lại), error, corners, method,
            transform_matrix, aligned_image, detected_grid, solved_grid,
            confidence, conflicts, uncertain, cells (overlay), timings, steps, attempts
        """
        notify = progress or (lambda stage, info: None)
        t_start = time.perf_counter()
        h, w = image.shape[:2]

        notify("detect", {})
        best: Optional[Dict[str, Any]] = None
        attempts = 0
        t_detect = time.perf_counter()
        for corners, method in self.candidate_quads(image, prefer, thorough=True):
            attempts += 1
            detect_ms = round((time.perf_counter() - t_detect) * 1000, 1)
            notify("align", {"corners": (corners / [w, h]).round(4).tolist()})
            result = self._attempt(image, corners, method, notify)
            result["timings"]["detect"] = detect_ms
            if result["success"]:
                best = result
                break
            if best is None or self._quality(result) > self._quality(best):
                best = result
            if attempts >= max_attempts:
                break
            t_detect = time.perf_counter()

        if best is None:
            best = {
                "success": False, "stage": "detect", "steps": {"detect": "Failed"},
                "timings": {"detect": round((time.perf_counter() - t_start) * 1000, 1)},
                "error": "Không tìm thấy lưới sudoku trong ảnh",
            }
        best["attempts"] = attempts
        best["timings"]["total"] = round((time.perf_counter() - t_start) * 1000, 1)
        if self.save_debug:
            self._save_debug(image, best)
        return best

    @staticmethod
    def _quality(result: Dict[str, Any]) -> float:
        """Xếp hạng lần thử thất bại: ít xung đột, nhiều số đọc được"""
        grid = result.get("detected_grid")
        givens = int(np.count_nonzero(grid)) if grid is not None else 0
        return givens - 10 * len(result.get("conflicts", []))

    def _attempt(self, image, corners, method, notify) -> Dict[str, Any]:
        """Align → OCR → Solve với một khung lưới cụ thể"""
        timings: Dict[str, float] = {}
        result: Dict[str, Any] = {
            "success": False, "steps": {"detect": "Success"}, "timings": timings,
            "corners": corners, "method": method,
        }

        def timed(name, fn):
            t = time.perf_counter()
            out = fn()
            timings[name] = round((time.perf_counter() - t) * 1000, 1)
            return out

        # 2. Align
        result["stage"] = "align"
        aligned, matrix = timed("align", lambda: self.alignment.align(image, corners))
        result.update(aligned_image=aligned, transform_matrix=matrix)
        result["steps"]["align"] = "Success"

        # 3. OCR
        result["stage"] = "ocr"
        notify("ocr", {})
        reading = timed("ocr", lambda: self.ocr_digit.read_grid(aligned))
        detected = reading.digits
        result.update(detected_grid=detected, confidence=reading.confidence,
                      occupied=reading.occupied, uncertain=reading.uncertain)
        result["steps"]["ocr"] = "Success"
        notify("solve", {"detected_grid": detected.tolist()})

        # 4. Solve
        result["stage"] = "solve"
        t = time.perf_counter()
        conflicts = SudokuValidator.find_conflicts(detected)
        result["conflicts"] = conflicts
        givens = int(np.count_nonzero(detected))
        if conflicts:
            result["error"] = f"Đề bài nhận diện bị trùng số ở {len(conflicts)} ô (OCR có thể đọc sai)"
        elif givens < 17:
            result["error"] = f"Chỉ nhận diện được {givens} số (cần ≥ 17)"
        else:
            solved, ok = self.solver.solve_sudoku(detected.copy())
            result["solved_grid"] = solved
            if ok:
                result["success"] = True
                result["stage"] = "done"
            else:
                result["error"] = "Không tìm được lời giải (OCR có thể đọc sai)"
        timings["solve"] = round((time.perf_counter() - t) * 1000, 1)
        result["steps"]["solve"] = "Success" if result["success"] else "Failed"

        # 5. Overlay geometry (dùng được cả khi thất bại để hiển thị lưới)
        result["cells"] = self.cell_geometry(matrix)
        return result

    # =====================================================================
    # Geometry / Rendering
    # =====================================================================

    def cell_geometry(self, matrix: np.ndarray) -> List[Dict[str, float]]:
        """
        Vị trí từng ô trên ảnh gốc: tâm (x, y), kích thước s, góc xoay a (độ)
        """
        inv = np.linalg.inv(matrix)
        step = self.aligned_size / 9.0
        pts = []
        for r in range(9):
            for c in range(9):
                cx, cy = (c + 0.5) * step, (r + 0.5) * step
                pts += [(cx, cy), (cx - step / 2, cy), (cx + step / 2, cy),
                        (cx, cy - step / 2), (cx, cy + step / 2)]
        src = project_points(np.array(pts, dtype=np.float32), inv).reshape(81, 5, 2)

        cells = []
        for i in range(81):
            center, left, right, top, bottom = src[i]
            width = float(np.linalg.norm(right - left))
            height = float(np.linalg.norm(bottom - top))
            angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
            cells.append({
                "x": round(float(center[0]), 1),
                "y": round(float(center[1]), 1),
                "s": round(min(width, height), 1),
                "a": round(angle, 1),
            })
        return cells

    def render_solution(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        color=(60, 220, 120),
    ) -> np.ndarray:
        """
        Vẽ chữ số lời giải lên ảnh gốc theo đúng phối cảnh
        (vẽ trên ảnh căn chỉnh rồi warp ngược về ảnh gốc)
        """
        out = image.copy()
        matrix = result.get("transform_matrix")
        solved = result.get("solved_grid")
        detected = result.get("detected_grid")
        if matrix is None or solved is None or detected is None:
            return out

        size = self.aligned_size
        step = size / 9.0
        layer = np.zeros((size, size, 3), np.uint8)
        mask = np.zeros((size, size), np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = step / 42.0
        thick = max(2, int(step / 22))
        for r in range(9):
            for c in range(9):
                if detected[r, c] != 0:
                    continue
                text = str(int(solved[r, c]))
                (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
                org = (int((c + 0.5) * step - tw / 2), int((r + 0.5) * step + th / 2))
                # viền tối + chữ màu để dễ đọc trên mọi nền
                cv2.putText(layer, text, org, font, scale, (20, 40, 20), thick + 3, cv2.LINE_AA)
                cv2.putText(layer, text, org, font, scale, color, thick, cv2.LINE_AA)
                cv2.putText(mask, text, org, font, scale, 255, thick + 3, cv2.LINE_AA)

        h, w = image.shape[:2]
        inv = np.linalg.inv(matrix)
        warped = cv2.warpPerspective(layer, inv, (w, h))
        alpha = cv2.warpPerspective(mask, inv, (w, h)).astype(np.float32)[..., None] / 255.0
        out = (out * (1 - alpha) + warped * alpha).astype(np.uint8)

        corners = result.get("corners")
        if corners is not None:
            cv2.polylines(out, [corners.astype(np.int32).reshape(-1, 1, 2)], True,
                          (255, 200, 0), max(1, int(min(h, w) / 300)), cv2.LINE_AA)
        return out

    # =====================================================================
    # Misc
    # =====================================================================

    def warmup(self) -> None:
        """Warmup các models (lần chạy đầu luôn chậm)"""
        logger.info("Warming up models ...")
        self.point_detector.warmup()
        self.ocr_digit.warmup()
        logger.info("Models warmed up")

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose
        self.ocr_digit.verbose = verbose

    def run_batch(self, images: list) -> list:
        return [self.run(img) for img in images]

    def get_pipeline_summary(self, result: Dict) -> str:
        lines = ["=" * 50, "PIPELINE SUMMARY", "=" * 50]
        for step_name, status in result.get("steps", {}).items():
            lines.append(f"  {step_name}: {status} ({result['timings'].get(step_name, '-')} ms)")
        lines.append(f"Overall success: {result['success']}")
        if "detected_grid" in result:
            lines.append(f"Givens: {self.solver.get_difficulty_score(result['detected_grid'])}")
        if result.get("error"):
            lines.append(f"Error: {result['error']}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def _save_debug(self, image: np.ndarray, result: Dict[str, Any]) -> None:
        folder = self.debug_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "01_original.jpg"), image)
        if "aligned_image" in result:
            cv2.imwrite(str(folder / "02_aligned.jpg"), result["aligned_image"])
        if result.get("success"):
            cv2.imwrite(str(folder / "03_solved.jpg"), self.render_solution(image, result))


def main():
    """Test pipeline: python pipeline.py <image>"""
    import sys

    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[2]
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(root / "imgs" / "test.jpg")
    pipeline = SudokuSolverPipeline(str(root / "backend" / "weights" / "best.pt"))
    pipeline.set_verbose(True)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    result = pipeline.run(image)
    print(pipeline.get_pipeline_summary(result))
    if result["success"]:
        pipeline.solver.print_grid(result["solved_grid"])


if __name__ == "__main__":
    main()
