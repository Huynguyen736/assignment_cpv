import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment_face.ui.bootstrap import load_app_settings
from assignment_face.ui.pages.live_attendance import render_live_attendance_page

render_live_attendance_page(load_app_settings())
