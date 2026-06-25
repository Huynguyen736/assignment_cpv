from __future__ import annotations

import cv2
import numpy as np

from assignment_face.config.settings import AppSettings


class Camera:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.capture = None
        self.source_name: str | None = None

    def start(self) -> str:
        attempted_sources: list[tuple[object, str]] = []
        if self.settings.rtsp_url:
            attempted_sources.append((self.settings.rtsp_url, "rtsp"))
        attempted_sources.append((self.settings.webcam_index, "webcam"))

        for source, source_name in attempted_sources:
            capture = cv2.VideoCapture(source)
            if capture.isOpened():
                self.capture = capture
                self.source_name = source_name
                return source_name
            capture.release()

        raise RuntimeError("Unable to open RTSP stream or webcam.")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.capture is None:
            self.start()
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        resized = cv2.resize(frame, self.settings.frame_size)
        return True, resized

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

