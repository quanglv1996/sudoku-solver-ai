"""
screen_solver.py

ScreenSolver: tự giải sudoku đang hiển thị trên màn hình máy tính
1. Chụp màn hình (mss)
2. Tìm lưới (contour trước - ảnh màn hình rất sạch, fallback YOLO)
3. OCR + giải
4. Tự click từng ô trống và gõ chữ số (pyautogui)

An toàn: pyautogui FAILSAFE - đưa chuột vào góc màn hình để dừng ngay lập tức.
"""

import ctypes
import logging
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    # Toạ độ chụp màn hình và toạ độ click phải cùng hệ pixel vật lý
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def list_monitors() -> List[Dict[str, int]]:
    import mss

    with mss.mss() as sct:
        return [
            {"index": i, "left": m["left"], "top": m["top"], "width": m["width"], "height": m["height"]}
            for i, m in enumerate(sct.monitors)
            if i > 0
        ]


def capture_screen(monitor: int = 1):
    """
    Returns:
        (ảnh BGR, (left, top) của màn hình trong hệ toạ độ desktop)
    """
    import mss

    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor < 1 or monitor >= len(monitors):
            monitor = 1
        mon = monitors[monitor]
        shot = np.array(sct.grab(mon))
    return np.ascontiguousarray(shot[:, :, :3]), (mon["left"], mon["top"])


class ScreenSolver:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def solve(
        self,
        monitor: int = 1,
        fill: bool = True,
        type_delay: float = 0.04,
        progress: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            monitor: Màn hình cần chụp (1 = màn hình chính)
            fill: Tự động điền kết quả
            type_delay: Nghỉ giữa các ô (giây) - tăng nếu trang web phản hồi chậm
            progress: callback(stage, info); có thể raise để huỷ

        Returns:
            dict: image (ảnh màn hình), pipeline result, filled, fill_error
        """
        notify = progress or (lambda stage, info: None)
        notify("capture", {})
        image, (left, top) = capture_screen(monitor)

        result = self.pipeline.run(image, progress=notify, prefer="contour")
        out: Dict[str, Any] = {"image": image, "result": result, "filled": 0, "fill_error": None}
        if not result["success"] or not fill:
            return out

        notify("fill", {"total": int(np.count_nonzero(result["detected_grid"] == 0))})
        try:
            out["filled"] = self._fill(result, left, top, type_delay, notify)
        except Exception as e:  # FailSafeException, huỷ, ...
            out["fill_error"] = f"{type(e).__name__}: {e}"
            if type(e).__name__ == "JobCancelled":
                raise
        return out

    def _fill(self, result, left: int, top: int, delay: float, notify) -> int:
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0

        detected = result["detected_grid"]
        solved = result["solved_grid"]
        cells = result["cells"]
        total = int(np.count_nonzero(detected == 0))
        filled = 0
        for i, cell in enumerate(cells):
            r, c = divmod(i, 9)
            if detected[r, c] != 0:
                continue
            x, y = int(round(left + cell["x"])), int(round(top + cell["y"]))
            pyautogui.click(x, y)
            time.sleep(delay)
            pyautogui.press(str(int(solved[r, c])))
            time.sleep(delay)
            filled += 1
            if filled % 5 == 0 or filled == total:
                notify("fill", {"filled": filled, "total": total})
        return filled
