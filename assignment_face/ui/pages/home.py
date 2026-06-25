from __future__ import annotations

from assignment_face.config.settings import AppSettings


def render_home_page(settings: AppSettings) -> None:
    import streamlit as st

    st.title("Attendance System")
    st.write("Computer Vision attendance system with Streamlit pages, Haar Cascade detection, and LBPH-style recognition.")
    st.info("Use the sidebar navigation to open Live Attendance or Register Student.")
    st.json(
        {
            "rtsp_url": settings.rtsp_url,
            "webcam_index": settings.webcam_index,
            "frame_size": settings.frame_size,
            "face_size": settings.face_size,
            "confidence_threshold": settings.confidence_threshold,
        }
    )
