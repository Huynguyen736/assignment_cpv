from __future__ import annotations

from assignment_face.config.settings import AppSettings
from assignment_face.ui.bootstrap import load_app_settings
from assignment_face.ui.pages.home import render_home_page


def run(settings: AppSettings | None = None) -> None:
    import streamlit as st

    app_settings = load_app_settings(settings)
    st.set_page_config(page_title="Attendance System", layout="wide")
    render_home_page(app_settings)
