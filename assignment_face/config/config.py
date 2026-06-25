from __future__ import annotations

import os
from pathlib import Path

from assignment_face.config.settings import AppSettings

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - optional dependency fallback
    dotenv_values = None


def load_settings(base_dir: str | Path | None = None, env_file: str | Path | None = None) -> AppSettings:
    resolved_base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent
    resolved_env = Path(env_file) if env_file else resolved_base_dir / ".env"
    env_data: dict[str, str | None] = {}

    if dotenv_values is not None and resolved_env.exists():
        env_data.update(dotenv_values(resolved_env))

    def get_value(name: str, default: str) -> str:
        return str(os.getenv(name) or env_data.get(name) or default)

    return AppSettings(
        base_dir=resolved_base_dir,
        rtsp_url=os.getenv("RTSP_URL") or env_data.get("RTSP_URL"),
        webcam_index=int(get_value("WEBCAM_INDEX", "0")),
        frame_width=int(get_value("FRAME_WIDTH", "640")),
        frame_height=int(get_value("FRAME_HEIGHT", "480")),
        face_width=int(get_value("FACE_WIDTH", "200")),
        face_height=int(get_value("FACE_HEIGHT", "200")),
        confidence_threshold=float(get_value("CONFIDENCE_THRESHOLD", "70")),
        blur_kernel=int(get_value("BLUR_KERNEL", "3")),
    )

