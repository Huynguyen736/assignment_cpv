from __future__ import annotations

from assignment_face.config.settings import AppSettings
from assignment_face.ui.processors import LiveAttendanceProcessor

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover - runtime dependency fallback
    av = None
    WebRtcMode = None
    webrtc_streamer = None


def render_live_attendance_page(settings: AppSettings) -> None:
    import streamlit as st

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
