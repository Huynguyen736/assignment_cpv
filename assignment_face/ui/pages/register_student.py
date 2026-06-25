from __future__ import annotations

import time

from assignment_face.config.settings import AppSettings
from assignment_face.services.registration import process_registration_recording
from assignment_face.ui.processors import RegistrationRecorderProcessor
from assignment_face.ui.state import create_recording_capture_state, create_registration_ui_state

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover - runtime dependency fallback
    av = None
    WebRtcMode = None
    webrtc_streamer = None


def render_register_student_page(settings: AppSettings) -> None:
    import streamlit as st

    st.title("Register Student")
    student_id = st.text_input("Student ID")
    student_name = st.text_input("Student Name")
    st.caption(
        "Bam start, quay video 10 giay. Sau do he thong se tach 5 frame moi giay, "
        "bo cac frame qua giong nhau va dang ky cac mau khuon mat hop le."
    )

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
