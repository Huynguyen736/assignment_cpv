from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

import cv2

from assignment_face_v2.config.settings import AppSettings

ATTENDANCE_COLUMNS = ["StudentID", "StudentName", "Date", "Time", "Confidence"]


def ensure_project_structure(settings: AppSettings) -> None:
    for directory in (
        settings.config_dir,
        settings.core_dir,
        settings.database_dir,
        settings.face_db_dir,
        settings.models_dir,
        settings.assets_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not settings.students_path.exists():
        settings.students_path.write_text("[]", encoding="utf-8")

    if not settings.attendance_path.exists():
        with settings.attendance_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ATTENDANCE_COLUMNS)
            writer.writeheader()

    if not settings.cascade_path.exists():
        source = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if source.exists():
            shutil.copyfile(source, settings.cascade_path)


def load_students(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def save_students(path: Path, students: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(students, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_student(path: Path, student_id: str, student_name: str) -> dict[str, str]:
    students = load_students(path)
    normalized = {"id": student_id, "name": student_name}
    for index, student in enumerate(students):
        if str(student.get("id")) == student_id:
            students[index] = normalized
            save_students(path, students)
            return normalized
    students.append(normalized)
    save_students(path, students)
    return normalized


def load_attendance_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_attendance_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTENDANCE_COLUMNS)
        if not file_exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

