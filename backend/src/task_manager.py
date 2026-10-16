"""
task_manager.py

TaskManager: kiến trúc Manager + Queue cho mọi tác vụ dùng model

    HTTP threads ──submit()──▶ PriorityQueue ──▶ Worker thread (sở hữu engine/GPU)
         ▲                                            │
         └──────── Job (status/stage/result) ◀────────┘

- Model YOLO/Paddle không thread-safe → chỉ 1 worker giữ engine, xử lý tuần tự.
- Ưu tiên: realtime (frame camera) > screen > photo.
- Realtime dùng "latest wins": mỗi session chỉ giữ frame mới nhất đang chờ,
  frame cũ bị bỏ (DROPPED) → không bao giờ dồn độ trễ.
- Job lưu tiến độ (stage) để frontend hiển thị hiệu ứng scan theo thời gian thực.
- Job đã xong được dọn sau `job_ttl` giây.
"""

import itertools
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DROPPED = "dropped"
    CANCELLED = "cancelled"

    FINAL = {DONE, FAILED, DROPPED, CANCELLED}


class Priority:
    REALTIME = 0
    SCREEN = 1
    PHOTO = 2


class QueueFullError(Exception):
    pass


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    kind: str
    payload: Dict[str, Any]
    priority: int
    session: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = JobStatus.QUEUED
    stage: str = "queued"
    stage_info: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    cancel_requested: bool = False
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def set_stage(self, stage: str, info: Optional[Dict[str, Any]] = None) -> None:
        """Gọi từ handler để báo tiến độ; cũng là điểm kiểm tra huỷ"""
        if self.cancel_requested:
            raise JobCancelled()
        self.stage = stage
        if info:
            self.stage_info.update(info)

    def finish(self, status: str, result=None, error=None) -> None:
        self.status = status
        self.result = result
        self.error = error
        self.finished_at = time.time()
        self.payload = {}  # giải phóng ảnh đầu vào
        self.done_event.set()

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "stage": self.stage,
            "stage_info": self.stage_info,
            "error": self.error,
            "created_at": self.created_at,
            "wait_ms": round(((self.started_at or time.time()) - self.created_at) * 1000),
        }
        if self.started_at:
            end = self.finished_at or time.time()
            data["run_ms"] = round((end - self.started_at) * 1000)
        if include_result and self.result is not None:
            data["result"] = self.result
        return data


Handler = Callable[[Job, Any], Dict[str, Any]]


class TaskManager:
    def __init__(
        self,
        engine_factory: Callable[[], Any],
        max_queue: int = 64,
        job_ttl: float = 600.0,
    ):
        """
        Args:
            engine_factory: Hàm tạo engine (load model) - chạy trong worker thread
            max_queue: Số job chờ tối đa (ngoài realtime đã gộp theo session)
            job_ttl: Thời gian giữ job đã hoàn tất (giây)
        """
        self.engine_factory = engine_factory
        self.max_queue = max_queue
        self.job_ttl = job_ttl

        self.handlers: Dict[str, Handler] = {}
        self.engine: Any = None
        self.engine_state = "idle"  # idle | loading | ready | error
        self.engine_error: Optional[str] = None

        self._queue: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._jobs: Dict[str, Job] = {}
        self._pending_by_session: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._running: Optional[Job] = None
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._stats: Dict[str, Dict[str, float]] = {}

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="sudoku-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put((-1, next(self._seq), None))

    # -----------------------------------------------------------------
    # Public API (gọi từ HTTP threads)
    # -----------------------------------------------------------------

    def submit(
        self,
        kind: str,
        payload: Dict[str, Any],
        priority: int = Priority.PHOTO,
        session: Optional[str] = None,
        latest_only: bool = False,
    ) -> Job:
        """
        Đưa job vào hàng đợi

        Args:
            latest_only: Nếu True, job đang chờ cùng session sẽ bị thay thế (DROPPED)
        """
        if kind not in self.handlers:
            raise ValueError(f"Unknown job kind: {kind}")
        if self.engine_state == "error":
            raise RuntimeError(f"Engine failed to load: {self.engine_error}")

        job = Job(kind=kind, payload=payload, priority=priority, session=session)
        with self._lock:
            self._cleanup_locked()
            if latest_only and session:
                old = self._pending_by_session.get(session)
                if old is not None and old.status == JobStatus.QUEUED:
                    old.finish(JobStatus.DROPPED, error="Superseded by newer frame")
                self._pending_by_session[session] = job
            elif self._queued_count_locked() >= self.max_queue:
                raise QueueFullError("Server đang bận, vui lòng thử lại sau")
            self._jobs[job.id] = job
        self._queue.put((priority, next(self._seq), job))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in JobStatus.FINAL:
                return False
            job.cancel_requested = True
            if job.status == JobStatus.QUEUED:
                job.finish(JobStatus.CANCELLED, error="Cancelled")
        return True

    def queue_position(self, job: Job) -> int:
        """Số job đứng trước (0 = đang chạy hoặc sắp chạy)"""
        if job.status != JobStatus.QUEUED:
            return 0
        with self._lock:
            ahead = [
                j for j in self._jobs.values()
                if j.status == JobStatus.QUEUED and (j.priority, j.created_at) < (job.priority, job.created_at)
            ]
        return len(ahead) + (1 if self._running else 0)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            queued = self._queued_count_locked()
            running = self._running.kind if self._running else None
        return {
            "engine": self.engine_state,
            "engine_error": self.engine_error,
            "queued": queued,
            "running": running,
            "per_kind": {
                k: {"count": int(v["count"]), "avg_ms": round(v["total_ms"] / max(1, v["count"]), 1)}
                for k, v in self._stats.items()
            },
        }

    # -----------------------------------------------------------------
    # Worker
    # -----------------------------------------------------------------

    def _loop(self) -> None:
        self.engine_state = "loading"
        try:
            self.engine = self.engine_factory()
            self.engine_state = "ready"
            logger.info("✅ Engine ready - worker started")
        except Exception as e:
            self.engine_state = "error"
            self.engine_error = f"{type(e).__name__}: {e}"
            logger.exception("❌ Engine failed to load")
            self._fail_all_pending()
            return

        while not self._stop.is_set():
            _, _, job = self._queue.get()
            if job is None:
                break

            with self._lock:
                if job.status != JobStatus.QUEUED:
                    continue  # đã bị drop / cancel
                if job.session and self._pending_by_session.get(job.session) is job:
                    del self._pending_by_session[job.session]
                self._running = job
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

            try:
                result = self.handlers[job.kind](job, self.engine)
                job.finish(JobStatus.DONE, result=result)
            except JobCancelled:
                job.finish(JobStatus.CANCELLED, error="Cancelled")
            except Exception as e:
                logger.error("Job %s (%s) failed: %s\n%s", job.id, job.kind, e, traceback.format_exc())
                job.finish(JobStatus.FAILED, error=str(e))
            finally:
                with self._lock:
                    self._running = None
                self._record(job)

    def _fail_all_pending(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job.status == JobStatus.QUEUED:
                    job.finish(JobStatus.FAILED, error=f"Engine failed to load: {self.engine_error}")

    def _record(self, job: Job) -> None:
        if job.started_at and job.finished_at:
            s = self._stats.setdefault(job.kind, {"count": 0, "total_ms": 0.0})
            s["count"] += 1
            s["total_ms"] += (job.finished_at - job.started_at) * 1000

    def _queued_count_locked(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [
            jid for jid, j in self._jobs.items()
            if j.status in JobStatus.FINAL and now - (j.finished_at or now) > self.job_ttl
            or (j.kind == "realtime" and j.status in JobStatus.FINAL and now - (j.finished_at or now) > 10)
        ]
        for jid in expired:
            del self._jobs[jid]
