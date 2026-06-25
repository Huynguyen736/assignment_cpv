from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from assignment_face.core.preprocess import to_grayscale


def load_face_cascade(cascade_path: str | Path) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise FileNotFoundError(f"Unable to load Haar cascade from {cascade_path}")
    return cascade


def extract_face_region(gray_frame: np.ndarray, bbox: tuple[int, int, int, int], face_size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = bbox
    face = gray_frame[y : y + h, x : x + w]
    return cv2.resize(face, face_size)


class FaceDetector:
    def __init__(self, cascade_path: str | Path, face_size: tuple[int, int]) -> None:
        self.cascade = load_face_cascade(cascade_path)
        self.face_size = face_size

    def detect(self, frame: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        gray = to_grayscale(frame)
        detections = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        boxes = [tuple(int(value) for value in detection) for detection in detections]
        faces = [extract_face_region(gray, bbox, self.face_size) for bbox in boxes]
        return faces, boxes

