from __future__ import annotations
from typing import Any
import numpy as np

def sample_recording_frames(
    frames: list[dict[str, Any]],
    duration_seconds: int = 10,
    frames_per_second: int = 5,
) -> list[dict[str, Any]]:
    """
    Tách luồng video (danh sách các frames) thành các khung hình riêng biệt 
    dựa trên fps được chỉ định (mặc định 5 fps).
    """
    if not frames:
        return []

    sampled: list[dict[str, Any]] = []
    for second_index in range(duration_seconds):
        start = float(second_index)
        end = float(second_index + 1)
        bucket = [item for item in frames if start <= float(item["timestamp"]) < end]
        if not bucket:
            continue

        max_items = min(frames_per_second, len(bucket))
        positions = np.linspace(0, len(bucket) - 1, num=max_items, dtype=int)
        for position in positions:
            sampled.append(bucket[int(position)])
    return sampled
