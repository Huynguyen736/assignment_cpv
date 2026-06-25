from __future__ import annotations

from typing import Any


def create_recording_capture_state(duration_seconds: int = 10, frames_per_second: int = 5) -> dict[str, Any]:
    return {
        "duration_seconds": duration_seconds,
        "frames_per_second": frames_per_second,
        "started_at": None,
        "recording": False,
        "processed": False,
        "saved_images": [],
        "captured_total": 0,
    }


def create_registration_ui_state() -> dict[str, bool]:
    return {
        "camera_enabled": False,
        "capture_enabled": False,
    }
