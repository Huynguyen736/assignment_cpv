"""Shared camera singleton for Streamlit pages.

Both Live Attendance and Register Student reuse the same Camera instance
stored in session_state, so only one RTSP connection is opened regardless
of which page is active.
"""
from __future__ import annotations

from collections.abc import Callable

from assignment_face.config.settings import AppSettings
from assignment_face.core.camera import Camera, CameraPlayer

_SESSION_KEY = "_shared_camera"


def get_shared_camera(settings: AppSettings) -> Camera:
    """Return the shared Camera from session_state, creating it if needed."""
    import streamlit as st

    if _SESSION_KEY not in st.session_state or st.session_state[_SESSION_KEY] is None:
        st.session_state[_SESSION_KEY] = Camera(settings)
    return st.session_state[_SESSION_KEY]  # type: ignore[return-value]


def shared_camera_player_factory(settings: AppSettings) -> Callable[[], CameraPlayer]:
    """Return a player factory that wraps the shared Camera instance.

    Each call to the returned factory creates a new CameraPlayer/VideoTrack,
    but all tracks share the same underlying Camera (and therefore the same
    RTSP connection).
    """

    def _factory() -> CameraPlayer:
        import streamlit as st

        camera = get_shared_camera(settings)
        player = object.__new__(CameraPlayer)
        player.audio = None

        from assignment_face.core.camera import OpenCVCameraVideoTrack

        player.video = OpenCVCameraVideoTrack(camera)
        return player

    return _factory
