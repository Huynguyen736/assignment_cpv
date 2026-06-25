from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment_face_v2.config.config import load_settings
from assignment_face_v2.config.settings import AppSettings
from assignment_face_v2.core.attendance import AttendanceManager
from assignment_face_v2.core.camera import Camera
from assignment_face_v2.core.detector import FaceDetector
from assignment_face_v2.core.preprocess import preprocess_frame
from assignment_face_v2.core.recognizer import FaceRecognizer
from assignment_face_v2.core.register import StudentRegistrar
from assignment_face_v2.utils.file_utils import ensure_project_structure

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover - runtime dependency fallback
    av = None
    WebRtcMode = None
    webrtc_streamer = None
def create_recording_capture_state(duration_seconds: int = 10, frames_per_second: int = 5) -> dict[str, Any]:
    return {
        "duration_seconds": duration_seconds,
        "frames_per_second": frames_per_second,
        "started_at": None,
        "recording": False,
        "processed": False,
        "saved_images": [],
        "captured_total": 0,
    }


def create_registration_ui_state() -> dict[str, bool]:
    return {
        "camera_enabled": False,
        "capture_enabled": False,
    }


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

def _annotate_frame(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    headline: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    annotated = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv2.putText(annotated, headline, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return annotated


def _release_camera(camera: Camera | None) -> None:
    if camera is not None:
        camera.release()


def sample_recording_frames(
    frames: list[dict[str, Any]],
    duration_seconds: int = 10,
    frames_per_second: int = 5,
) -> list[dict[str, Any]]:
    if not frames:
        return []

    sampled: list[dict[str, Any]] = []
    for second_index in range(duration_seconds):
        start = float(second_index)
        end = float(second_index + 1)
        bucket = [item for item in frames if start <= float(item["timestamp"]) < end]
        if not bucket:
            continue

        max_items = min(frames_per_second, len(bucket))
        positions = np.linspace(0, len(bucket) - 1, num=max_items, dtype=int)
        for position in positions:
            sampled.append(bucket[int(position)])
    return sampled


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


class LiveAttendanceProcessor:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.last_status: dict[str, Any] = {
            "recognized": False,
            "student_id": None,
            "student_name": None,
            "confidence": None,
            "status": "Waiting for video",
            "bounding_boxes": [],
        }
        self._lock = Lock()

    def process_ndarray(
        self,
        frame: np.ndarray,
        detected_faces: list[np.ndarray] | None = None,
        bounding_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> np.ndarray:
        status = build_live_attendance_status(
            settings=self.settings,
            frame=frame,
            detected_faces=detected_faces,
            bounding_boxes=bounding_boxes,
        )
        color = (0, 180, 0) if status["recognized"] else (0, 165, 255)
        annotated = _annotate_frame(frame, status["bounding_boxes"], status["status"], color)
        with self._lock:
            self.last_status = status
        return annotated

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, self.settings.frame_size)
        annotated = self.process_ndarray(resized)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.last_status)


class RegistrationRecorderProcessor:
    def __init__(self, settings: AppSettings, max_seconds: int = 12, history_fps: int = 30) -> None:
        self.settings = settings
        self._lock = Lock()
        self.max_history = max_seconds * history_fps
        self.frame_history: deque[dict[str, Any]] = deque(maxlen=self.max_history)
        self.last_preview = np.zeros((settings.frame_height, settings.frame_width, 3), dtype=np.uint8)

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, self.settings.frame_size)
        timestamp = time.monotonic()
        with self._lock:
            self.frame_history.append({"timestamp": timestamp, "frame": resized.copy()})
            self.last_preview = resized.copy()
        return av.VideoFrame.from_ndarray(resized, format="bgr24")

    def get_recent_recording(self, duration_seconds: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            history = list(self.frame_history)
        if not history:
            return []
        end_time = float(history[-1]["timestamp"])
        start_time = end_time - duration_seconds
        trimmed = [item for item in history if float(item["timestamp"]) >= start_time]
        if not trimmed:
            return []
        base_time = float(trimmed[0]["timestamp"])
        return [{"timestamp": float(item["timestamp"]) - base_time, "frame": item["frame"]} for item in trimmed]

    def get_preview_frame(self) -> np.ndarray:
        with self._lock:
            return self.last_preview.copy()


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
        return {"ok": False, "message": "No valid distinct face frames found in the 10-second recording.", "saved_images": [], "captured_total": 0}

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
        "message": f"Saved {len(saved_images)} distinct face samples from the 10-second recording. Recognizer trained with {trained.trained_images} images.",
        "saved_images": saved_images,
        "captured_total": len(saved_images),
        "student_id": student_id,
        "student_name": student_name,
    }


def build_registration_status(
    settings: AppSettings,
    student_id: str,
    student_name: str,
    face_image: np.ndarray | None,
) -> dict[str, Any]:
    ensure_project_structure(settings)
    if not student_id.strip() or not student_name.strip():
        return {"ok": False, "message": "Student ID and name are required."}
    if face_image is None:
        return {"ok": False, "message": "A detected face is required before saving."}
    return _register_face_sample(
        settings=settings,
        student_id=student_id,
        student_name=student_name,
        face_image=face_image,
        retrain=True,
    )


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
        detected, boxes = detector.detect(processed)
        faces = detected

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


def _decode_uploaded_image(raw_bytes: bytes) -> np.ndarray | None:
    buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def _prepare_registration_face(settings: AppSettings, raw_image: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    detector = FaceDetector(settings.cascade_path, settings.face_size)
    processed = preprocess_frame(raw_image, frame_size=settings.frame_size, blur_kernel=settings.blur_kernel)
    faces, _ = detector.detect(processed)
    if len(faces) != 1:
        return None, "Exactly one face must be visible for registration."
    return faces[0], None


def render_home_page(settings: AppSettings) -> None:
    import streamlit as st

    st.title("Attendance System")
    st.write("Computer Vision attendance system with RTSP/webcam fallback, Haar Cascade detection, and LBPH-style recognition.")
    st.json(
        {
            "rtsp_url": settings.rtsp_url,
            "webcam_index": settings.webcam_index,
            "frame_size": settings.frame_size,
            "face_size": settings.face_size,
            "confidence_threshold": settings.confidence_threshold,
        }
    )


def render_live_attendance_page(settings: AppSettings) -> None:
    import streamlit as st

    ensure_project_structure(settings)
    st.title("Live Attendance")
    st.write("Use the live WebRTC stream below to detect faces, recognize students, and record attendance without page reruns.")

    if webrtc_streamer is None or av is None or WebRtcMode is None:
        st.error("streamlit-webrtc is not installed. Install dependencies again to enable stable live streaming.")
        return

    status_placeholder = st.empty()
    info_placeholder = st.empty()

    webrtc_ctx = webrtc_streamer(
        key="live-attendance-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: LiveAttendanceProcessor(settings),
        async_processing=True,
    )

    @st.fragment(run_every=0.3)
    def render_live_status_fragment() -> None:
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            status = webrtc_ctx.video_processor.get_status()
            if status["recognized"]:
                status_placeholder.success(
                    f"Student: {status['student_id']} | Name: {status['student_name']} | "
                    f"Confidence: {status['confidence']} | Attendance Status: {status['status']}"
                )
            else:
                status_placeholder.warning(
                    f"Student: - | Name: - | Confidence: {status['confidence']} | Attendance Status: {status['status']}"
                )
            info_placeholder.caption("Camera stream is handled in the WebRTC component to avoid visible page refreshes.")
        else:
            status_placeholder.info("Camera is stopped.")
            info_placeholder.write(
                {
                    "Student": "",
                    "Name": "",
                    "Confidence": "",
                    "Attendance Status": "",
                }
            )

    render_live_status_fragment()


def render_register_student_page(settings: AppSettings) -> None:
    import streamlit as st

    ensure_project_structure(settings)
    st.title("Register Student")
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student Name")
    st.caption("Bam start, quay video 10 giay. Sau do he thong se tach 5 frame moi giay, bo cac frame qua giong nhau va dang ky cac mau khuon mat hop le.")

    if webrtc_streamer is None or av is None or WebRtcMode is None:
        st.error("streamlit-webrtc is not installed. Install dependencies again to enable stable live registration.")
        return

    if "registration_capture_state" not in st.session_state:
        st.session_state.registration_capture_state = create_recording_capture_state()
    if "registration_ui_state" not in st.session_state:
        st.session_state.registration_ui_state = create_registration_ui_state()
    if "registration_feedback" not in st.session_state:
        st.session_state.registration_feedback = "Nhap thong tin va bam Start Capture."
    if "registration_student_id" not in st.session_state:
        st.session_state.registration_student_id = ""
    if "registration_student_name" not in st.session_state:
        st.session_state.registration_student_name = ""

    camera_col, capture_col = st.columns(2)

    camera_button_label = "Stop Camera" if st.session_state.registration_ui_state["camera_enabled"] else "Start Camera"

    if camera_col.button(camera_button_label, use_container_width=True):
        ui_state = st.session_state.registration_ui_state
        ui_state["camera_enabled"] = not ui_state["camera_enabled"]
        ui_state["capture_enabled"] = ui_state["camera_enabled"]
        if ui_state["camera_enabled"]:
            st.session_state.registration_feedback = "Camera da bat. Ban co the bam Start Capture."
        else:
            st.session_state.registration_capture_state["recording"] = False
            st.session_state.registration_feedback = "Camera da tat."

    if capture_col.button(
        "Start Capture",
        use_container_width=True,
        disabled=(
            not st.session_state.registration_ui_state["capture_enabled"]
            or st.session_state.registration_capture_state["recording"]
        ),
    ):
        if not student_id.strip() or not student_name.strip():
            st.error("Student ID and name are required.")
            return
        st.session_state.registration_student_id = student_id.strip()
        st.session_state.registration_student_name = student_name.strip()
        st.session_state.registration_capture_state = create_recording_capture_state()
        st.session_state.registration_capture_state["started_at"] = time.monotonic()
        st.session_state.registration_capture_state["recording"] = True
        st.session_state.registration_capture_state["processed"] = False
        st.session_state.registration_capture_state["saved_images"] = []
        st.session_state.registration_capture_state["captured_total"] = 0
        st.session_state.registration_feedback = "Dang quay video 10 giay. Hay di chuyen khuon mat tu nhien trong khung hinh."

    feedback_placeholder = st.empty()
    progress_placeholder = st.empty()
    preview_placeholder = st.empty()

    webrtc_ctx = webrtc_streamer(
        key="register-attendance-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=st.session_state.registration_ui_state["camera_enabled"],
        video_processor_factory=lambda: RegistrationRecorderProcessor(settings),
        async_processing=True,
    )

    @st.fragment(run_every=0.3)
    def render_registration_status_fragment() -> None:
        capture_state = st.session_state.registration_capture_state
        if st.session_state.registration_ui_state["camera_enabled"] and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            preview_frame = webrtc_ctx.video_processor.get_preview_frame()
            if preview_frame.size > 0:
                preview_placeholder.image(preview_frame[:, :, ::-1], channels="RGB", use_container_width=True)

            if capture_state["recording"] and capture_state["started_at"] is not None:
                elapsed = time.monotonic() - float(capture_state["started_at"])
                remaining = max(0.0, float(capture_state["duration_seconds"]) - elapsed)
                feedback_placeholder.info(f"Dang quay video. Con {remaining:.1f} giay.")
                progress_placeholder.progress(min(elapsed / float(capture_state["duration_seconds"]), 1.0))

                if elapsed >= float(capture_state["duration_seconds"]):
                    recorded_frames = webrtc_ctx.video_processor.get_recent_recording(
                        duration_seconds=int(capture_state["duration_seconds"])
                    )
                    result = process_registration_recording(
                        settings=settings,
                        student_id=st.session_state.registration_student_id,
                        student_name=st.session_state.registration_student_name,
                        recorded_frames=recorded_frames,
                        duration_seconds=int(capture_state["duration_seconds"]),
                        frames_per_second=int(capture_state["frames_per_second"]),
                    )
                    capture_state["recording"] = False
                    capture_state["processed"] = True
                    capture_state["saved_images"] = result.get("saved_images", [])
                    capture_state["captured_total"] = result.get("captured_total", 0)
                    st.session_state.registration_feedback = result["message"]
                    if result["ok"]:
                        feedback_placeholder.success(result["message"])
                    else:
                        feedback_placeholder.warning(result["message"])
            else:
                feedback_placeholder.write(st.session_state.registration_feedback)
                progress_placeholder.write(
                    {
                        "Captured total": capture_state["captured_total"],
                        "Saved images": capture_state["saved_images"],
                    }
                )
        else:
            feedback_placeholder.info("Camera is stopped. Bat camera bang nut Start/Stop Camera.")
            progress_placeholder.write(
                {
                    "Captured total": capture_state["captured_total"],
                    "Saved images": capture_state["saved_images"],
                }
            )

    render_registration_status_fragment()


def render_streamlit_app(settings: AppSettings | None = None) -> None:
    import streamlit as st

    app_settings = settings or load_settings()
    ensure_project_structure(app_settings)

    st.set_page_config(page_title="Attendance System", layout="wide")
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    page = st.session_state.page

    if page == "Home":
        render_home_page(app_settings)
    elif page == "Live Attendance":
        render_live_attendance_page(app_settings)
    elif page == "Register Student":
        render_register_student_page(app_settings)


if __name__ == "__main__":
    render_streamlit_app()
