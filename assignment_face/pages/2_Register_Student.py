import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assignment_face.app import render_register_student_page
from assignment_face.config.config import load_settings
from assignment_face.utils.file_utils import ensure_project_structure

settings = load_settings()
ensure_project_structure(settings)
render_register_student_page(settings)
