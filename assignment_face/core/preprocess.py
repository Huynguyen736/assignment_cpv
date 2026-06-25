from __future__ import annotations

import numpy as np


def resize_frame(frame: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """Resize image using Bilinear Interpolation with pure math (numpy)."""
    target_w, target_h = frame_size
    orig_h, orig_w = frame.shape[:2]
    
    x_ratio = float(orig_w - 1) / (target_w - 1) if target_w > 1 else 0
    y_ratio = float(orig_h - 1) / (target_h - 1) if target_h > 1 else 0
    
    x_dst = np.arange(target_w)
    y_dst = np.arange(target_h)
    
    x_src = x_dst * x_ratio
    y_src = y_dst * y_ratio
    
    x0 = np.floor(x_src).astype(int)
    y0 = np.floor(y_src).astype(int)
    
    x1 = np.minimum(x0 + 1, orig_w - 1)
    y1 = np.minimum(y0 + 1, orig_h - 1)
    
    wx = x_src - x0
    wy = y_src - y0
    
    if frame.ndim == 3:
        wx = wx[np.newaxis, :, np.newaxis]
        wy = wy[:, np.newaxis, np.newaxis]
        
        y0_idx = y0[:, np.newaxis]
        y1_idx = y1[:, np.newaxis]
        x0_idx = x0[np.newaxis, :]
        x1_idx = x1[np.newaxis, :]
        
        I00 = frame[y0_idx, x0_idx]
        I01 = frame[y0_idx, x1_idx]
        I10 = frame[y1_idx, x0_idx]
        I11 = frame[y1_idx, x1_idx]
    else:
        wx = wx[np.newaxis, :]
        wy = wy[:, np.newaxis]
        
        y0_idx = y0[:, np.newaxis]
        y1_idx = y1[:, np.newaxis]
        x0_idx = x0[np.newaxis, :]
        x1_idx = x1[np.newaxis, :]
        
        I00 = frame[y0_idx, x0_idx]
        I01 = frame[y0_idx, x1_idx]
        I10 = frame[y1_idx, x0_idx]
        I11 = frame[y1_idx, x1_idx]
        
    top = I00 * (1 - wx) + I01 * wx
    bottom = I10 * (1 - wx) + I11 * wx
    
    result = top * (1 - wy) + bottom * wy
    return np.clip(result, 0, 255).astype(frame.dtype)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR image to Grayscale mathematically."""
    if frame.ndim == 2:
        return frame
    # Formula: Y = 0.114 * B + 0.587 * G + 0.299 * R (BGR format)
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    gray = 0.114 * b + 0.587 * g + 0.299 * r
    return gray.astype(np.uint8)


def equalize_histogram(gray_frame: np.ndarray) -> np.ndarray:
    """Histogram Equalization using Cumulative Distribution Function."""
    hist, _ = np.histogram(gray_frame.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    
    # Mask out zeros to avoid min value problems
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    return cdf_final[gray_frame]


def adjust_brightness_contrast(image: np.ndarray, brightness: int = 0, contrast: int = 0) -> np.ndarray:
    """Adjust brightness and contrast linearly."""
    alpha = 1.0 + contrast / 100.0
    beta = float(brightness)
    
    adjusted = alpha * image.astype(np.float32) + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def get_gaussian_kernel(kernel_size: int, sigma: float = 0) -> np.ndarray:
    """Generate a 2D Gaussian kernel."""
    if sigma <= 0:
        # OpenCV formula for default sigma
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def apply_gaussian_blur(image: np.ndarray, blur_kernel: int = 3) -> np.ndarray:
    """Apply Gaussian Blur using 2D Convolution (sliding window view)."""
    kernel_size = max(1, int(blur_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    if kernel_size == 1:
        return image.copy()
        
    kernel = get_gaussian_kernel(kernel_size)
    pad_w = kernel_size // 2
    
    def convolve2d_numpy(ch: np.ndarray) -> np.ndarray:
        # Padding edge pixels
        padded = np.pad(ch, pad_width=pad_w, mode='reflect')
        # Check if sliding_window_view is available
        if hasattr(np.lib.stride_tricks, 'sliding_window_view'):
            view = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
            return np.sum(view * kernel, axis=(2, 3))
        else:
            # Fallback naive convolution
            h, w = ch.shape
            out = np.zeros_like(ch, dtype=np.float32)
            for i in range(h):
                for j in range(w):
                    out[i, j] = np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel)
            return out

    if image.ndim == 3:
        result = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d_numpy(image[:, :, c])
    else:
        result = convolve2d_numpy(image)
        
    return np.clip(result, 0, 255).astype(image.dtype)


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
