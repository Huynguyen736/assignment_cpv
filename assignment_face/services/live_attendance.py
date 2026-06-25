from __future__ import annotations

from typing import Any

import numpy as np

from assignment_face.config.settings import AppSettings
from assignment_face.core.attendance import AttendanceManager
from assignment_face.core.detector import FaceDetector
from assignment_face.core.preprocess import preprocess_frame
from assignment_face.core.recognizer import FaceRecognizer
from assignment_face.utils.file_utils import ensure_project_structure


def build_live_attendance_status(
    settings: AppSettings,
    frame: np.ndarray,
    detected_faces: list[np.ndarray] | None = None,
    bounding_boxes: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, Any]:
    ensure_project_structure(settings)
    if not settings.model_path.exists() and not settings.fallback_model_path.exists():
        return {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": None,
            "status": "Recognizer model missing. Register students and train first.",
            "bounding_boxes": [],
        }

    faces = detected_faces
    boxes = bounding_boxes or []
    if faces is None:
        processed = preprocess_frame(frame, frame_size=settings.frame_size, blur_kernel=settings.blur_kernel)
        detector = FaceDetector(settings.cascade_path, settings.face_size)
        faces, boxes = detector.detect(processed)

    if not faces:
        return {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": None,
            "status": "No face detected",
            "bounding_boxes": boxes,
        }

    if len(faces) > 1:
        return {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": None,
            "status": "Multiple faces detected",
            "bounding_boxes": boxes,
        }

    recognizer = FaceRecognizer(settings)
    prediction = recognizer.recognize(faces[0])
    if not prediction.recognized:
        return {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": prediction.confidence,
            "status": "Unknown student",
            "bounding_boxes": boxes,
        }

    attendance = AttendanceManager(settings.attendance_path)
    result = attendance.mark_attendance(
        student_id=prediction.student_id or "",
        student_name=prediction.student_name or "",
        confidence=prediction.confidence,
    )
    return {
        "recognized": True,
        "student_id": prediction.student_id,
        "student_name": prediction.student_name,
        "confidence": prediction.confidence,
        "status": result.status,
        "bounding_boxes": boxes,
    }
