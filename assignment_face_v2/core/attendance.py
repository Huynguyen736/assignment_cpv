from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from assignment_face_v2.utils.file_utils import append_attendance_row, load_attendance_rows
from assignment_face_v2.utils.time_utils import format_date, format_time


@dataclass(frozen=True)
class AttendanceResult:
    inserted: bool
    status: str
    row: dict[str, str] | None


class AttendanceManager:
    def __init__(self, attendance_path: Path) -> None:
        self.attendance_path = attendance_path

    def load_records(self) -> list[dict[str, str]]:
        return load_attendance_rows(self.attendance_path)

    def already_checked(self, student_id: str, date_text: str) -> bool:
        return any(
            str(row.get("StudentID")) == student_id and str(row.get("Date")) == date_text for row in self.load_records()
        )

    def mark_attendance(
        self,
        student_id: str,
        student_name: str,
        confidence: float,
        now: datetime | None = None,
    ) -> AttendanceResult:
        current_time = now or datetime.now()
        date_text = format_date(current_time)
        if self.already_checked(student_id, date_text):
            return AttendanceResult(inserted=False, status="Already checked today", row=None)

        row = {
            "StudentID": student_id,
            "StudentName": student_name,
            "Date": date_text,
            "Time": format_time(current_time),
            "Confidence": f"{float(confidence):.2f}",
        }
        self.save(row)
        return AttendanceResult(inserted=True, status="Attendance saved", row=row)

    def save(self, row: dict[str, str]) -> None:
        append_attendance_row(self.attendance_path, row)
