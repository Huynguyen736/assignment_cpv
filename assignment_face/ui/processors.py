from __future__ import annotations

import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Any, Callable

import cv2
import numpy as np

from assignment_face.config.settings import AppSettings
from assignment_face.services.live_attendance import build_live_attendance_status

try:
    import av
except Exception:  # pragma: no cover - optional runtime dependency
    av = None


def _annotate_frame(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    headline: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    annotated = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.putText(annotated, headline, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return annotated


class BackgroundTask:
    def __init__(self, target: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._done = Event()
        self._lock = Lock()
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread = Thread(target=self._run, args=(target, args, kwargs), daemon=True)
        self._thread.start()

    def _run(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            result = target(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - defensive UI surface
            with self._lock:
                self._error = exc
        else:
            with self._lock:
                self._result = result
        finally:
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self) -> Any:
        if not self.done():
            raise RuntimeError("Background task is still running.")
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._result


class LiveAttendanceProcessor:
    def __init__(self, settings: AppSettings, processing_interval_seconds: float = 0.25) -> None:
        self.settings = settings
        self.processing_interval_seconds = processing_interval_seconds
        self.last_status: dict[str, Any] = {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": None,
            "status": "Waiting for video",
            "bounding_boxes": [],
            "reference_image_path": None,
            "reference_lbp_input": None,
            "query_lbp_input": None,
        }
        self._lock = Lock()
        self._pending_frame: np.ndarray | None = None
        self._frame_available = Event()
        self._stop_requested = Event()
        self._worker = Thread(target=self._run_status_worker, daemon=True)
        self._worker.start()

    def _schedule_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame.copy()
        self._frame_available.set()

    def _take_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            frame = self._pending_frame
            self._pending_frame = None
            self._frame_available.clear()
            return frame

    def _set_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self.last_status = status

    def _run_status_worker(self) -> None:
        last_processed_at = 0.0
        while not self._stop_requested.is_set():
            if not self._frame_available.wait(0.1):
                continue

            wait_time = self.processing_interval_seconds - (time.monotonic() - last_processed_at)
            if wait_time > 0 and self._stop_requested.wait(wait_time):
                break

            frame = self._take_latest_frame()
            if frame is None:
                continue

            last_processed_at = time.monotonic()
            try:
                status = build_live_attendance_status(settings=self.settings, frame=frame)
            except Exception as exc:  # pragma: no cover - defensive UI surface
                status = {
                    "recognized": False,
                    "student_id": None,
                    "student_name": None,
                    "confidence": None,
                    "status": f"Processing error: {exc}",
                    "bounding_boxes": [],
                    "reference_image_path": None,
                    "reference_lbp_input": None,
                    "query_lbp_input": None,
                }
            self._set_status(status)

    def process_ndarray(
        self,
        frame: np.ndarray,
        detected_faces: list[np.ndarray] | None = None,
        bounding_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray:
        if detected_faces is None and bounding_boxes is None:
            self._schedule_frame(frame)
            status = self.get_status()
        else:
            status = build_live_attendance_status(
                settings=self.settings,
                frame=frame,
                detected_faces=detected_faces,
                bounding_boxes=bounding_boxes,
            )
            self._set_status(status)

        color = (0, 180, 0) if status.get("recognized") else (0, 165, 255)
        return _annotate_frame(
            frame,
            status.get("bounding_boxes", []),
            status.get("status", "Processing video"),
            color,
        )

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        if av is None:
            raise RuntimeError("PyAV is required for WebRTC video processing.")
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, self.settings.frame_size)
        annotated = self.process_ndarray(resized)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.last_status)

    def stop(self) -> None:
        self._stop_requested.set()
        self._frame_available.set()
        self._worker.join(timeout=1.0)


class RegistrationRecorderProcessor:
    def __init__(self, settings: AppSettings, max_seconds: int = 12, history_fps: int = 30) -> None:
        self.settings = settings
        self._lock = Lock()
        self.max_history = max_seconds * history_fps
        self.frame_history: deque[dict[str, Any]] = deque(maxlen=self.max_history)
        self.last_preview = np.zeros((settings.frame_height, settings.frame_width, 3), dtype=np.uint8)

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        if av is None:
            raise RuntimeError("PyAV is required for WebRTC video processing.")
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, self.settings.frame_size)
        timestamp = time.monotonic()
        with self._lock:
            self.frame_history.append({"timestamp": timestamp, "frame": resized.copy()})
            self.last_preview = resized.copy()
        return av.VideoFrame.from_ndarray(resized, format="bgr24")

    def get_recent_recording(self, duration_seconds: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            history = list(self.frame_history)
        if not history:
            return []
        end_time = float(history[-1]["timestamp"])
        start_time = end_time - duration_seconds
        trimmed = [item for item in history if float(item["timestamp"]) >= start_time]
        if not trimmed:
            return []
        base_time = float(trimmed[0]["timestamp"])
        return [{"timestamp": float(item["timestamp"]) - base_time, "frame": item["frame"]} for item in trimmed]

    def get_preview_frame(self) -> np.ndarray:
        with self._lock:
            return self.last_preview.copy()
