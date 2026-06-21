from __future__ import annotations

from dataclasses import dataclass

from assignment_face_v2.config.settings import AppSettings
from assignment_face_v2.utils.file_utils import upsert_student
from assignment_face_v2.utils.image_utils import save_face_image


@dataclass(frozen=True)
class RegistrationResult:
    student_id: str
    student_name: str
    image_path: str


class StudentRegistrar:
    def __init__(self, settings: AppSettings, recognizer_factory=None) -> None:
        self.settings = settings
        self.recognizer_factory = recognizer_factory or self._default_recognizer_factory

    @staticmethod
    def _default_recognizer_factory(settings: AppSettings):
        from assignment_face_v2.core.recognizer import FaceRecognizer

        return FaceRecognizer(settings)

    def save_face(self, student_id: str, face_image, image_index: int) -> str:
        output_path = self.settings.face_db_dir / student_id / f"{image_index:03d}.jpg"
        save_face_image(output_path, face_image)
        return str(output_path)

    def save_student(self, student_id: str, student_name: str) -> dict[str, str]:
        return upsert_student(self.settings.students_path, student_id=student_id, student_name=student_name)

    def register(self, student_id: str, student_name: str, face_image, image_index: int = 1) -> RegistrationResult:
        student = self.save_student(student_id, student_name)
        image_path = self.save_face(student_id=student["id"], face_image=face_image, image_index=image_index)
        return RegistrationResult(student_id=student["id"], student_name=student["name"], image_path=image_path)

