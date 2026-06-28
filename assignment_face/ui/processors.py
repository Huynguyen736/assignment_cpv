from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

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


class LiveAttendanceProcessor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
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

    def process_ndarray(
        self,
        frame: np.ndarray,
        detected_faces: list[np.ndarray] | None = None,
        bounding_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray:
        status = build_live_attendance_status(
            settings=self.settings,
            frame=frame,
            detected_faces=detected_faces,
            bounding_boxes=bounding_boxes,
        )
        color = (0, 180, 0) if status["recognized"] else (0, 165, 255)
        annotated = _annotate_frame(frame, status["bounding_boxes"], status["status"], color)
        with self._lock:
            self.last_status = status
        return annotated

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
