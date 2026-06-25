from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype("uint8")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def save_face_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), ensure_grayscale(image))


def load_face_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

