from __future__ import annotations

from dataclasses import dataclass
import json

import cv2
import numpy as np

from assignment_face_v2.config.settings import AppSettings
from assignment_face_v2.utils.file_utils import load_students
from assignment_face_v2.utils.image_utils import ensure_grayscale, load_face_image


def _compute_lbp_codes(face_image: np.ndarray) -> np.ndarray:
    image = face_image.astype(np.uint8)
    center = image[1:-1, 1:-1]
    neighbors = [
        image[:-2, :-2],
        image[:-2, 1:-1],
        image[:-2, 2:],
        image[1:-1, 2:],
        image[2:, 2:],
        image[2:, 1:-1],
        image[2:, :-2],
        image[1:-1, :-2],
    ]
    lbp = np.zeros(center.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)
    return lbp


def compute_lbp_descriptor(face_image: np.ndarray, grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    lbp = _compute_lbp_codes(face_image)
    histograms: list[np.ndarray] = []
    for row in np.array_split(lbp, grid_size[1], axis=0):
        for cell in np.array_split(row, grid_size[0], axis=1):
            histogram = np.bincount(cell.ravel(), minlength=256).astype(np.float32)
            histogram /= histogram.sum() + 1e-6
            histograms.append(histogram)
    return np.concatenate(histograms)


def _chi_square_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    numerator = (vector_a - vector_b) ** 2
    denominator = vector_a + vector_b + 1e-6
    return float(0.5 * np.sum(numerator / denominator))


@dataclass(frozen=True)
class TrainingResult:
    trained_images: int
    students: int


@dataclass(frozen=True)
class RecognitionResult:
    student_id: str | None
    student_name: str | None
    confidence: float
    recognized: bool


class FaceRecognizer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.grid_size = (8, 8)
        self.descriptors: np.ndarray | None = None
        self.student_ids: list[str] = []
        self.student_names: dict[str, str] = {}
        self.label_to_student_id: dict[int, str] = {}
        self.backend = "fallback-lbp"

    def _load_students(self) -> None:
        self.student_names = {student["id"]: student["name"] for student in load_students(self.settings.students_path)}

    def _iter_training_images(self) -> list[tuple[str, np.ndarray]]:
        records: list[tuple[str, np.ndarray]] = []
        for student_dir in sorted(self.settings.face_db_dir.glob("*")):
            if not student_dir.is_dir():
                continue
            for image_path in sorted(list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))):
                image = load_face_image(image_path)
                if image is None:
                    continue
                records.append((student_dir.name, image))
        return records

    def train(self) -> TrainingResult:
        self._load_students()
        records = self._iter_training_images()
        if not records:
            raise ValueError("No face images found in the face database.")

        grouped_records: dict[str, list[np.ndarray]] = {}
        for student_id, image in records:
            grouped_records.setdefault(student_id, []).append(ensure_grayscale(image))

        self.student_ids = [student_id for student_id, _ in records]
        self.descriptors = np.vstack(
            [compute_lbp_descriptor(ensure_grayscale(image), grid_size=self.grid_size) for _, image in records]
        )
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.label_map_path.write_text(
            json.dumps(self.student_names, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        np.savez(
            self.settings.fallback_model_path,
            descriptors=self.descriptors,
            student_ids=np.array(self.student_ids, dtype=object),
            grid_size=np.array(self.grid_size),
        )

        if hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces: list[np.ndarray] = []
            labels: list[int] = []
            self.label_to_student_id = {}
            for label, student_id in enumerate(sorted(grouped_records)):
                self.label_to_student_id[label] = student_id
                for image in grouped_records[student_id]:
                    faces.append(image)
                    labels.append(label)
            recognizer.train(faces, np.array(labels, dtype=np.int32))
            recognizer.save(str(self.settings.model_path))
            self.backend = "opencv-lbph"
        else:
            np.savez(
                self.settings.model_path,
                descriptors=self.descriptors,
                student_ids=np.array(self.student_ids, dtype=object),
                grid_size=np.array(self.grid_size),
            )
            self.backend = "fallback-lbp"

        return TrainingResult(trained_images=len(records), students=len(set(self.student_ids)))

    def _ensure_model_loaded(self) -> None:
        self._load_students()
        if self.descriptors is not None and self.student_ids and self.backend in {"opencv-lbph", "fallback-lbp"}:
            return
        if not self.settings.model_path.exists() and not self.settings.fallback_model_path.exists():
            raise FileNotFoundError("Model file not found. Please train the recognizer first.")

        if self.settings.fallback_model_path.exists():
            payload = np.load(self.settings.fallback_model_path, allow_pickle=True)
            self.descriptors = payload["descriptors"]
            self.student_ids = [str(value) for value in payload["student_ids"].tolist()]
            self.grid_size = tuple(int(value) for value in payload["grid_size"].tolist())

        try:
            payload = np.load(self.settings.model_path, allow_pickle=True)
            self.backend = "fallback-lbp"
        except Exception:
            if not (hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")):
                raise FileNotFoundError("Model requires OpenCV contrib LBPH support, but cv2.face is unavailable.")
            self.backend = "opencv-lbph"
            self.label_to_student_id = {
                index: student_id for index, student_id in enumerate(sorted(self.student_names))
            }

    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        self._ensure_model_loaded()
        grayscale_face = ensure_grayscale(face_image)
        fallback_student_id: str | None = None
        fallback_distance: float | None = None

        if self.descriptors is not None:
            descriptor = compute_lbp_descriptor(grayscale_face, grid_size=self.grid_size)
            best_by_student: dict[str, float] = {}
            for student_id, reference in zip(self.student_ids, self.descriptors, strict=True):
                distance = _chi_square_distance(descriptor, reference)
                current = best_by_student.get(student_id)
                if current is None or distance < current:
                    best_by_student[student_id] = distance
            fallback_student_id, fallback_distance = min(best_by_student.items(), key=lambda item: item[1])

        if self.backend == "opencv-lbph":
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(self.settings.model_path))
            label, distance = recognizer.predict(grayscale_face)
            student_id = self.label_to_student_id.get(int(label))
            recognized = student_id is not None and distance <= self.settings.confidence_threshold
            if not recognized and fallback_student_id is not None and fallback_distance is not None:
                student_id = fallback_student_id
                distance = fallback_distance
                recognized = distance <= self.settings.confidence_threshold
        else:
            if fallback_student_id is None or fallback_distance is None:
                raise FileNotFoundError("Fallback face descriptor model is unavailable.")
            student_id = fallback_student_id
            distance = fallback_distance
            recognized = distance <= self.settings.confidence_threshold

        return RecognitionResult(
            student_id=student_id if recognized else None,
            student_name=self.student_names.get(student_id) if recognized else None,
            confidence=round(float(distance), 4),
            recognized=recognized,
        )
