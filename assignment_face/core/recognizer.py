from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from assignment_face.config.settings import AppSettings
from assignment_face.utils.file_utils import load_students
from assignment_face.utils.image_utils import ensure_grayscale, load_face_image


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
        self.backend = "internal-lbp"

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

        self.student_ids = [student_id for student_id, _ in records]
        self.descriptors = np.vstack(
            [compute_lbp_descriptor(ensure_grayscale(image), grid_size=self.grid_size) for _, image in records]
        )
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.label_map_path.write_text(
            json.dumps(self.student_names, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        model_payload = {
            "descriptors": self.descriptors,
            "student_ids": np.array(self.student_ids, dtype=object),
            "grid_size": np.array(self.grid_size),
        }
        np.savez(
            self.settings.model_path,
            **model_payload,
        )
        np.savez(
            self.settings.fallback_model_path,
            **model_payload,
        )

        self.backend = "internal-lbp"
        return TrainingResult(trained_images=len(records), students=len(set(self.student_ids)))

    def _load_descriptor_payload(self, model_path: str) -> bool:
        try:
            payload = np.load(model_path, allow_pickle=True)
        except Exception:
            return False

        self.descriptors = payload["descriptors"]
        self.student_ids = [str(value) for value in payload["student_ids"].tolist()]
        self.grid_size = tuple(int(value) for value in payload["grid_size"].tolist())
        self.backend = "internal-lbp"
        return True

    def _ensure_model_loaded(self) -> None:
        self._load_students()
        if self.descriptors is not None and self.student_ids and self.backend == "internal-lbp":
            return
        if not self.settings.model_path.exists() and not self.settings.fallback_model_path.exists():
            raise FileNotFoundError("Model file not found. Please train the recognizer first.")

        model_loaded = False
        if self.settings.model_path.exists():
            model_loaded = self._load_descriptor_payload(str(self.settings.model_path))
        if not model_loaded and self.settings.fallback_model_path.exists():
            model_loaded = self._load_descriptor_payload(str(self.settings.fallback_model_path))
        if not model_loaded:
            raise FileNotFoundError("Stored recognizer model is incompatible. Please retrain the recognizer.")

    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        self._ensure_model_loaded()
        grayscale_face = ensure_grayscale(face_image)
        descriptor = compute_lbp_descriptor(grayscale_face, grid_size=self.grid_size)
        best_by_student: dict[str, float] = {}
        for student_id, reference in zip(self.student_ids, self.descriptors, strict=True):
            distance = _chi_square_distance(descriptor, reference)
            current = best_by_student.get(student_id)
            if current is None or distance < current:
                best_by_student[student_id] = distance

        if not best_by_student:
            raise FileNotFoundError("Face descriptor model is unavailable.")

        student_id, distance = min(best_by_student.items(), key=lambda item: item[1])
        recognized = distance <= self.settings.confidence_threshold

        return RecognitionResult(
            student_id=student_id if recognized else None,
            student_name=self.student_names.get(student_id) if recognized else None,
            confidence=round(float(distance), 4),
            recognized=recognized,
        )
