from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from assignment_face.config.settings import AppSettings
from assignment_face.core.recognizer import FaceRecognizer
from assignment_face.utils.file_utils import (
    delete_student,
    load_students,
    update_student_name,
)


def get_all_students_with_images(settings: AppSettings) -> list[dict[str, Any]]:
    """Return all students with their face image paths."""
    students = load_students(settings.students_path)
    result: list[dict[str, Any]] = []
    for student in students:
        student_id = str(student.get("id", ""))
        student_dir = settings.face_db_dir / student_id
        images: list[str] = []
        if student_dir.is_dir():
            images = sorted(
                str(p) for p in student_dir.glob("*.jpg")
            ) + sorted(
                str(p) for p in student_dir.glob("*.png")
            )
        result.append({
            "id": student_id,
            "name": student.get("name", ""),
            "image_count": len(images),
            "image_paths": images,
        })
    return result


def _retrain_model(settings: AppSettings) -> dict[str, Any]:
    """Retrain the LBPH model. If no faces remain, remove model files."""
    face_dirs = [d for d in settings.face_db_dir.iterdir() if d.is_dir()]
    has_faces = any(
        list(d.glob("*.jpg")) + list(d.glob("*.png")) for d in face_dirs
    )

    if not has_faces:
        # No faces left — remove model files
        for model_file in (settings.model_path, settings.fallback_model_path, settings.label_map_path):
            if model_file.exists():
                model_file.unlink()
        return {"retrained": False, "message": "Không còn dữ liệu khuôn mặt. Đã xóa model."}

    recognizer = FaceRecognizer(settings)
    trained = recognizer.train()
    return {
        "retrained": True,
        "message": f"Model đã được retrain với {trained.trained_images} ảnh, {trained.students} sinh viên.",
    }


def _update_label_map(settings: AppSettings) -> None:
    """Rebuild label_map.json from current students.json."""
    students = load_students(settings.students_path)
    label_map = {str(s["id"]): s["name"] for s in students}
    settings.label_map_path.write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def service_update_student(
    settings: AppSettings,
    student_id: str,
    new_name: str,
) -> dict[str, Any]:
    """Update a student's name and sync DB + model."""
    if not new_name.strip():
        return {"ok": False, "message": "Tên sinh viên không được để trống."}

    found = update_student_name(settings.students_path, student_id, new_name.strip())
    if not found:
        return {"ok": False, "message": f"Không tìm thấy sinh viên {student_id}."}

    _update_label_map(settings)
    retrain_result = _retrain_model(settings)
    return {
        "ok": True,
        "message": f"Đã cập nhật tên sinh viên {student_id} thành \"{new_name.strip()}\". {retrain_result['message']}",
    }


def service_delete_student(
    settings: AppSettings,
    student_id: str,
) -> dict[str, Any]:
    """Delete a student and all their face data, then retrain model."""
    found = delete_student(settings.students_path, student_id)
    if not found:
        return {"ok": False, "message": f"Không tìm thấy sinh viên {student_id}."}

    # Remove face images directory
    student_face_dir = settings.face_db_dir / student_id
    if student_face_dir.is_dir():
        shutil.rmtree(student_face_dir)

    _update_label_map(settings)
    retrain_result = _retrain_model(settings)
    return {
        "ok": True,
        "message": f"Đã xóa sinh viên {student_id}. {retrain_result['message']}",
    }
