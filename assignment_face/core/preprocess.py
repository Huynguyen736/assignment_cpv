from __future__ import annotations

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(frame, frame_size)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def equalize_histogram(gray_frame: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_frame)


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.0 + contrast / 100.0, beta=brightness)


def apply_gaussian_blur(image: np.ndarray, blur_kernel: int = 3) -> np.ndarray:
    kernel = max(1, int(blur_kernel))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def preprocess_frame(
    frame: np.ndarray,
    frame_size: tuple[int, int],
    brightness: int = 0,
    contrast: int = 0,
    blur_kernel: int = 3,
) -> np.ndarray:
    resized = resize_frame(frame, frame_size)
    gray = to_grayscale(resized)
    equalized = equalize_histogram(gray)
    adjusted = adjust_brightness_contrast(equalized, brightness=brightness, contrast=contrast)
    return apply_gaussian_blur(adjusted, blur_kernel=blur_kernel)

