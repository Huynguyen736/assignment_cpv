from __future__ import annotations

from assignment_face.config.config import load_settings
from assignment_face.config.settings import AppSettings
from assignment_face.utils.file_utils import ensure_project_structure


def load_app_settings(settings: AppSettings | None = None) -> AppSettings:
    app_settings = settings or load_settings()
    ensure_project_structure(app_settings)
    return app_settings
