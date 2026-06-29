import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment_face.ui.bootstrap import load_app_settings
from assignment_face.ui.pages.manage_students import render_manage_students_page

render_manage_students_page(load_app_settings())
