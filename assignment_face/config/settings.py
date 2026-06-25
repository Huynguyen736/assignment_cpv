from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    base_dir: Path
    rtsp_url: str | None = None
    webcam_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    face_width: int = 200
    face_height: int = 200
    confidence_threshold: float = 70.0
    blur_kernel: int = 3

    @property
    def config_dir(self) -> Path:
        return self.base_dir / "config"

    @property
    def core_dir(self) -> Path:
        return self.base_dir / "core"

    @property
    def database_dir(self) -> Path:
        return self.base_dir / "database"

    @property
    def face_db_dir(self) -> Path:
        return self.database_dir / "face_db"

    @property
    def students_path(self) -> Path:
        return self.database_dir / "students.json"

    @property
    def attendance_path(self) -> Path:
        return self.database_dir / "attendance.csv"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def model_path(self) -> Path:
        return self.models_dir / "lbph_model.npz"

    @property
    def fallback_model_path(self) -> Path:
        return self.models_dir / "lbph_fallback.npz"

    @property
    def label_map_path(self) -> Path:
        return self.models_dir / "label_map.json"

    @property
    def cascade_path(self) -> Path:
        return self.models_dir / "haarcascade_frontalface_default.xml"

    @property
    def assets_dir(self) -> Path:
        return self.base_dir / "assets"

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self.frame_width, self.frame_height)

    @property
    def face_size(self) -> tuple[int, int]:
        return (self.face_width, self.face_height)
