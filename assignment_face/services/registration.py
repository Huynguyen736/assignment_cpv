from __future__ import annotations

from typing import Any

import numpy as np

from assignment_face.config.settings import AppSettings
from assignment_face.core.detector import FaceDetector
from assignment_face.core.preprocess import preprocess_frame
from assignment_face.core.recognizer import FaceRecognizer
from assignment_face.core.register import StudentRegistrar
from assignment_face.core.crop_video import sample_recording_frames
from assignment_face.utils.file_utils import ensure_project_structure


def _next_student_image_index(settings: AppSettings, student_id: str) -> int:
    existing = sorted((settings.face_db_dir / student_id).glob("*.jpg"))
    return len(existing) + 1


def _register_face_sample(
    settings: AppSettings,
    student_id: str,
    student_name: str,
    face_image: np.ndarray,
    retrain: bool,
) -> dict[str, Any]:
    registrar = StudentRegistrar(settings)
    image_index = _next_student_image_index(settings, student_id)
    result = registrar.register(
        student_id=student_id.strip(),
        student_name=student_name.strip(),
        face_image=face_image,
        image_index=image_index,
    )

    training_summary = ""
    if retrain:
        try:
            recognizer = registrar.recognizer_factory(settings)
            trained = recognizer.train()
            training_summary = f" Recognizer trained with {trained.trained_images} images."
        except ValueError:
            training_summary = " Add more registered faces to enable training."

    return {
        "ok": True,
        "message": f"Student registered successfully.{training_summary}".strip(),
        "student_id": result.student_id,
        "student_name": result.student_name,
        "image_path": result.image_path,
    }





def deduplicate_face_samples(face_images: list[np.ndarray], similarity_threshold: float = 5.0) -> list[np.ndarray]:
    unique_faces: list[np.ndarray] = []
    previous_face: np.ndarray | None = None
    for face_image in face_images:
        if previous_face is None:
            unique_faces.append(face_image)
            previous_face = face_image
            continue

        difference = float(np.mean(np.abs(face_image.astype(np.float32) - previous_face.astype(np.float32))))
        if difference > similarity_threshold:
            unique_faces.append(face_image)
            previous_face = face_image
    return unique_faces


def process_registration_recording(
    settings: AppSettings,
    student_id: str,
    student_name: str,
    recorded_frames: list[dict[str, Any]],
    duration_seconds: int = 10,
    frames_per_second: int = 5,
    similarity_threshold: float = 5.0,
) -> dict[str, Any]:
    ensure_project_structure(settings)
    if not student_id.strip() or not student_name.strip():
        return {"ok": False, "message": "Student ID and name are required.", "saved_images": [], "captured_total": 0}

    detector = FaceDetector(settings.cascade_path, settings.face_size)
    sampled_frames = sample_recording_frames(
        recorded_frames,
        duration_seconds=duration_seconds,
        frames_per_second=frames_per_second,
    )

    valid_faces: list[np.ndarray] = []
    for sample in sampled_frames:
        processed = preprocess_frame(sample["frame"], frame_size=settings.frame_size, blur_kernel=settings.blur_kernel)
        faces, _ = detector.detect(processed)
        if len(faces) == 1:
            valid_faces.append(faces[0])

    unique_faces = deduplicate_face_samples(valid_faces, similarity_threshold=similarity_threshold)
    if not unique_faces:
        return {
            "ok": False,
            "message": "No valid distinct face frames found in the 10-second recording.",
            "saved_images": [],
            "captured_total": 0,
        }

    saved_images: list[str] = []
    for face_image in unique_faces:
        status = _register_face_sample(
            settings=settings,
            student_id=student_id,
            student_name=student_name,
            face_image=face_image,
            retrain=False,
        )
        saved_images.append(status["image_path"])

    recognizer = FaceRecognizer(settings)
    trained = recognizer.train()
    return {
        "ok": True,
        "message": (
            f"Saved {len(saved_images)} distinct face samples from the 10-second recording. "
            f"Recognizer trained with {trained.trained_images} images."
        ),
        "saved_images": saved_images,
        "captured_total": len(saved_images),
        "student_id": student_id,
        "student_name": student_name,
    }
