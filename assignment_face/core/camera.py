from __future__ import annotations

import asyncio
from collections.abc import Callable
from queue import Empty, Queue
from threading import Event, Lock, Thread
import time
from typing import Protocol

import cv2
import numpy as np

from assignment_face.config.settings import AppSettings

try:
    import av
    from aiortc import VideoStreamTrack
except Exception:  # pragma: no cover - runtime dependency fallback
    av = None
    VideoStreamTrack = object  # type: ignore[assignment,misc]


class Capture(Protocol):
    def isOpened(self) -> bool:
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    def release(self) -> None:
        ...


class Camera:
    def __init__(
        self,
        settings: AppSettings,
        capture_factory: Callable[[object], Capture] | None = None,
        open_attempts: int = 3,
        retry_delay_seconds: float = 0.05,
        open_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 5.0,
    ) -> None:
        self.settings = settings
        self.capture_factory = capture_factory or self._default_capture_factory
        self.open_attempts = max(1, open_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self.open_timeout_seconds = max(0.01, open_timeout_seconds)
        self.read_timeout_seconds = max(0.01, read_timeout_seconds)
        self.capture: Capture | None = None
        self.source: str | None = None
        self._first_frame: np.ndarray | None = None
        # Lock prevents concurrent cap.read() calls from asyncio threads,
        # which causes libavcodec assertion failures in FFmpeg.
        self._read_lock = Lock()

    @staticmethod
    def _default_capture_factory(source: object) -> "cv2.VideoCapture":
        """Open camera with FFMPEG backend hint for RTSP URLs."""
        if isinstance(source, str) and source.startswith("rtsp"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            # Limit internal FFmpeg decode threads to 1 to avoid
            # pthread_frame assertion errors under multi-threaded access.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)  # type: ignore[arg-type]
        return cap

    def start(self) -> str:
        if self.capture is not None and self.source is not None:
            return self.source

        for source_name, source_value in self._source_candidates():
            capture = self._open_capture(source_value)
            if capture is None:
                continue
            if not capture.isOpened():
                capture.release()
                continue

            first_frame = self._read_opening_frame(capture)
            if first_frame is None:
                capture.release()
                continue

            self.capture = capture
            self.source = source_name
            self._first_frame = first_frame
            return source_name

        raise RuntimeError("Unable to connect to RTSP camera or fallback webcam.")

    def read(self) -> tuple[bool, np.ndarray]:
        with self._read_lock:
            if self.capture is None:
                self.start()

            if self._first_frame is not None:
                frame = self._first_frame
                self._first_frame = None
                return True, self._resize_frame(frame)

            if self.capture is None:
                return False, self._blank_frame()

            ok, frame = self.capture.read()
            if not ok or frame is None:
                return False, self._blank_frame()
            return True, self._resize_frame(frame)

    def release(self) -> None:
        with self._read_lock:
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            self.source = None
            self._first_frame = None

    def _source_candidates(self) -> list[tuple[str, object]]:
        candidates: list[tuple[str, object]] = []
        if self.settings.rtsp_url:
            candidates.append(("rtsp", self.settings.rtsp_url))
        candidates.append(("webcam", self.settings.webcam_index))
        return candidates

    def _open_capture(self, source: object) -> Capture | None:
        result_queue: Queue[tuple[str, Capture | BaseException]] = Queue(maxsize=1)
        timed_out = Event()
        decision_lock = Lock()

        def open_worker() -> None:
            try:
                capture = self.capture_factory(source)
            except BaseException as exc:
                with decision_lock:
                    if not timed_out.is_set():
                        result_queue.put(("error", exc))
                return

            with decision_lock:
                if timed_out.is_set():
                    capture.release()
                    return
                result_queue.put(("capture", capture))

        Thread(target=open_worker, daemon=True).start()
        try:
            result_type, payload = result_queue.get(timeout=self.open_timeout_seconds)
        except Empty:
            with decision_lock:
                timed_out.set()
                self._release_queued_capture(result_queue)
            return None

        if result_type == "error":
            return None
        return payload  # type: ignore[return-value]

    @staticmethod
    def _release_queued_capture(result_queue: Queue[tuple[str, Capture | BaseException]]) -> None:
        try:
            result_type, payload = result_queue.get_nowait()
        except Empty:
            return
        if result_type == "capture":
            payload.release()  # type: ignore[union-attr]

    def _read_opening_frame(self, capture: Capture) -> np.ndarray | None:
        for attempt in range(self.open_attempts):
            ok, frame = self._read_frame_with_timeout(capture)
            if ok and frame is not None:
                return frame
            if attempt + 1 < self.open_attempts and self.retry_delay_seconds:
                time.sleep(self.retry_delay_seconds)
        return None

    def _read_frame_with_timeout(self, capture: Capture) -> tuple[bool, np.ndarray | None]:
        result_queue: Queue[tuple[bool, np.ndarray | None]] = Queue(maxsize=1)

        def read_worker() -> None:
            try:
                result_queue.put(capture.read())
            except Exception:
                result_queue.put((False, None))

        Thread(target=read_worker, daemon=True).start()
        try:
            return result_queue.get(timeout=self.read_timeout_seconds)
        except Empty:
            return False, None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1::-1] == self.settings.frame_size:
            return frame
        return cv2.resize(frame, self.settings.frame_size)

    def _blank_frame(self) -> np.ndarray:
        return np.zeros((self.settings.frame_height, self.settings.frame_width, 3), dtype=np.uint8)


class OpenCVCameraVideoTrack(VideoStreamTrack):  # type: ignore[misc,valid-type]
    kind = "video"

    def __init__(self, camera: Camera) -> None:
        if av is None or VideoStreamTrack is object:
            raise RuntimeError("PyAV and aiortc are required for camera streaming.")
        super().__init__()
        self.camera = camera
        self._last_frame = np.zeros(
            (camera.settings.frame_height, camera.settings.frame_width, 3),
            dtype=np.uint8,
        )

    async def recv(self):  # type: ignore[no-untyped-def]
        pts, time_base = await self.next_timestamp()
        try:
            ok, image = await asyncio.to_thread(self.camera.read)
        except Exception:
            ok = False
            image = self._last_frame.copy()
        if ok:
            self._last_frame = image.copy()
        else:
            image = self._last_frame.copy()

        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self) -> None:
        self.camera.release()
        super().stop()


class CameraPlayer:
    def __init__(self, settings: AppSettings) -> None:
        self.audio = None
        self.video = OpenCVCameraVideoTrack(Camera(settings))


def camera_player_factory(settings: AppSettings) -> Callable[[], CameraPlayer]:
    return lambda: CameraPlayer(settings)
