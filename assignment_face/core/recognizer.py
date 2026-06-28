from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import cv2

from assignment_face.config.settings import AppSettings
from assignment_face.utils.file_utils import load_students
from assignment_face.utils.image_utils import ensure_grayscale, load_face_image

UNIFORM_LBP_BINS = 59
NON_UNIFORM_LBP_BIN = UNIFORM_LBP_BINS - 1
LBP_VARIANT = "uniform-lbp-u2-p8-r1-center-crop-0.76"


def _is_uniform_lbp_code(code: int) -> bool:
    transitions = 0
    previous_bit = (code >> 7) & 1
    for bit in range(8):
        current_bit = (code >> bit) & 1
        if current_bit != previous_bit:
            transitions += 1
        previous_bit = current_bit
    return transitions <= 2


def _build_uniform_lbp_lookup() -> np.ndarray:
    lookup = np.full(256, NON_UNIFORM_LBP_BIN, dtype=np.uint8)
    next_bin = 0
    for code in range(256):
        if _is_uniform_lbp_code(code):
            lookup[code] = next_bin
            next_bin += 1
    return lookup


_UNIFORM_LBP_LOOKUP = _build_uniform_lbp_lookup()


def prepare_lbp_input(face_image: np.ndarray, crop_ratio: float = 0.76) -> np.ndarray:
    grayscale = ensure_grayscale(face_image)
    height, width = grayscale.shape[:2]
    clamped_ratio = min(max(crop_ratio, 0.5), 1.0)
    crop_w = max(1, int(round(width * clamped_ratio)))
    crop_h = max(1, int(round(height * clamped_ratio)))
    x1 = max(0, (width - crop_w) // 2)
    y1 = max(0, (height - crop_h) // 2)
    cropped = grayscale[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if cropped.shape == grayscale.shape:
        return cropped.astype(np.uint8)
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


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
    return _UNIFORM_LBP_LOOKUP[lbp]


def compute_lbp_descriptor(face_image: np.ndarray, grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    lbp = _compute_lbp_codes(face_image)
    histograms: list[np.ndarray] = []
    for row in np.array_split(lbp, grid_size[1], axis=0):
        for cell in np.array_split(row, grid_size[0], axis=1):
            histogram = np.bincount(cell.ravel(), minlength=UNIFORM_LBP_BINS).astype(np.float32)
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
    reference_image_path: str | None = None
    reference_lbp_input: np.ndarray | None = None
    query_lbp_input: np.ndarray | None = None


class FaceRecognizer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.grid_size = (8, 8)
        self.descriptors: np.ndarray | None = None
        self.student_ids: list[str] = []
        self.image_paths: list[str] = []
        self.student_names: dict[str, str] = {}
        self.backend = "internal-uniform-lbp"

    def _load_students(self) -> None:
        self.student_names = {student["id"]: student["name"] for student in load_students(self.settings.students_path)}

    def _iter_training_images(self) -> list[tuple[str, str, np.ndarray]]:
        records: list[tuple[str, str, np.ndarray]] = []
        for student_dir in sorted(self.settings.face_db_dir.glob("*")):
            if not student_dir.is_dir():
                continue
            for image_path in sorted(list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))):
                image = load_face_image(image_path)
                if image is None:
                    continue
                records.append((student_dir.name, str(image_path), image))
        return records

    def train(self) -> TrainingResult:
        self._load_students()
        records = self._iter_training_images()
        if not records:
            raise ValueError("No face images found in the face database.")

        self.student_ids = [student_id for student_id, _, _ in records]
        self.image_paths = [image_path for _, image_path, _ in records]
        self.descriptors = np.vstack(
            [compute_lbp_descriptor(prepare_lbp_input(image), grid_size=self.grid_size) for _, _, image in records]
        )
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        self.settings.label_map_path.write_text(
            json.dumps(self.student_names, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        model_payload = {
            "descriptors": self.descriptors,
            "student_ids": np.array(self.student_ids, dtype=object),
            "image_paths": np.array(self.image_paths, dtype=object),
            "grid_size": np.array(self.grid_size),
            "lbp_variant": np.array(LBP_VARIANT),
        }
        np.savez(
            self.settings.model_path,
            **model_payload,
        )
        np.savez(
            self.settings.fallback_model_path,
            **model_payload,
        )

        self.backend = "internal-uniform-lbp"
        return TrainingResult(trained_images=len(records), students=len(set(self.student_ids)))

    def _load_descriptor_payload(self, model_path: str) -> bool:
        try:
            payload = np.load(model_path, allow_pickle=True)
            if str(payload["lbp_variant"].tolist()) != LBP_VARIANT:
                return False
            grid_size = tuple(int(value) for value in payload["grid_size"].tolist())
            descriptors = payload["descriptors"]
            expected_descriptor_length = grid_size[0] * grid_size[1] * UNIFORM_LBP_BINS
            if descriptors.shape[1] != expected_descriptor_length:
                return False
        except Exception:
            return False

        self.descriptors = descriptors
        self.student_ids = [str(value) for value in payload["student_ids"].tolist()]
        if "image_paths" in payload:
            self.image_paths = [str(value) for value in payload["image_paths"].tolist()]
        else:
            records = self._iter_training_images()
            self.image_paths = [image_path for _, image_path, _ in records] if len(records) == len(self.student_ids) else []
        self.grid_size = grid_size
        self.backend = "internal-uniform-lbp"
        return True

    def _ensure_model_loaded(self) -> None:
        self._load_students()
        if self.descriptors is not None and self.student_ids and self.backend == "internal-uniform-lbp":
            return
        if not self.settings.model_path.exists() and not self.settings.fallback_model_path.exists():
            raise FileNotFoundError("Model file not found. Please train the recognizer first.")

        model_loaded = False
        if self.settings.model_path.exists():
            model_loaded = self._load_descriptor_payload(str(self.settings.model_path))
        if not model_loaded and self.settings.fallback_model_path.exists():
            model_loaded = self._load_descriptor_payload(str(self.settings.fallback_model_path))
        if not model_loaded:
            self.train()

    def recognize(self, face_image: np.ndarray) -> RecognitionResult:
        self._ensure_model_loaded()
        query_lbp_input = prepare_lbp_input(face_image)
        descriptor = compute_lbp_descriptor(query_lbp_input, grid_size=self.grid_size)
        best_by_student: dict[str, float] = {}
        best_reference_index_by_student: dict[str, int] = {}
        for reference_index, (student_id, reference) in enumerate(zip(self.student_ids, self.descriptors, strict=True)):
            distance = _chi_square_distance(descriptor, reference)
            current = best_by_student.get(student_id)
            if current is None or distance < current:
                best_by_student[student_id] = distance
                best_reference_index_by_student[student_id] = reference_index

        if not best_by_student:
            raise FileNotFoundError("Face descriptor model is unavailable.")

        student_id, distance = min(best_by_student.items(), key=lambda item: item[1])
        recognized = distance <= self.settings.confidence_threshold
        reference_index = best_reference_index_by_student[student_id]
        reference_image_path = self.image_paths[reference_index] if reference_index < len(self.image_paths) else None
        reference_image = load_face_image(reference_image_path) if reference_image_path else None
        reference_lbp_input = prepare_lbp_input(reference_image) if reference_image is not None else None

        return RecognitionResult(
            student_id=student_id if recognized else None,
            student_name=self.student_names.get(student_id) if recognized else None,
            confidence=round(float(distance), 4),
            recognized=recognized,
            reference_image_path=reference_image_path,
            reference_lbp_input=reference_lbp_input,
            query_lbp_input=query_lbp_input,
        )
