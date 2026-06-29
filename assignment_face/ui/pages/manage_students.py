from __future__ import annotations

from pathlib import Path

from assignment_face.config.settings import AppSettings
from assignment_face.services.student_management import (
    get_all_students_with_images,
    service_delete_student,
    service_update_student,
)


def render_manage_students_page(settings: AppSettings) -> None:
    import streamlit as st

    st.title("Quản lý Sinh viên")
    st.caption("Xem, chỉnh sửa thông tin và xóa sinh viên khỏi hệ thống.")

    # ── Load students ──────────────────────────────────────────────
    students = get_all_students_with_images(settings)

    if not students:
        st.info("Chưa có sinh viên nào trong hệ thống. Hãy đăng ký sinh viên trước.")
        return

    # ── Sidebar-like selector ──────────────────────────────────────
    student_options = {
        f"{s['id']} — {s['name']} ({s['image_count']} ảnh)": s["id"]
        for s in students
    }

    st.subheader(f"📋 Danh sách sinh viên ({len(students)} người)")

    selected_label = st.selectbox(
        "Chọn sinh viên để xem chi tiết:",
        options=list(student_options.keys()),
    )

    if selected_label is None:
        return

    selected_id = student_options[selected_label]
    selected_student = next((s for s in students if s["id"] == selected_id), None)
    if selected_student is None:
        st.error("Không tìm thấy sinh viên.")
        return

    st.divider()

    # ── Student detail ─────────────────────────────────────────────
    info_col, action_col = st.columns([2, 1])

    with info_col:
        st.subheader(f"👤 {selected_student['name']}")
        st.markdown(f"**Mã sinh viên:** `{selected_student['id']}`")
        st.markdown(f"**Số ảnh khuôn mặt:** {selected_student['image_count']}")

    # ── Face images gallery ────────────────────────────────────────
    image_paths = selected_student["image_paths"]
    if image_paths:
        st.subheader("🖼️ Ảnh khuôn mặt trong DB")
        cols_per_row = 6
        for row_start in range(0, len(image_paths), cols_per_row):
            row_images = image_paths[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_idx, img_path in enumerate(row_images):
                with cols[col_idx]:
                    st.image(img_path, caption=Path(img_path).name, use_container_width=True)
    else:
        st.warning("Sinh viên này chưa có ảnh khuôn mặt nào trong database.")

    st.divider()

    # ── Edit name ──────────────────────────────────────────────────
    st.subheader("✏️ Chỉnh sửa thông tin")

    new_name = st.text_input(
        "Tên mới:",
        value=selected_student["name"],
        key=f"edit_name_{selected_id}",
    )

    if st.button("💾 Cập nhật tên", key=f"update_{selected_id}", use_container_width=True):
        if new_name.strip() == selected_student["name"]:
            st.warning("Tên không thay đổi.")
        else:
            result = service_update_student(settings, selected_id, new_name)
            if result["ok"]:
                st.success(result["message"])
                st.rerun()
            else:
                st.error(result["message"])

    st.divider()

    # ── Delete student ─────────────────────────────────────────────
    st.subheader("🗑️ Xóa sinh viên")
    st.warning(
        f"Xóa sinh viên **{selected_student['name']}** ({selected_student['id']}) "
        f"sẽ xóa toàn bộ {selected_student['image_count']} ảnh khuôn mặt và "
        f"retrain lại model. Hành động này không thể hoàn tác!"
    )

    confirm_delete = st.checkbox(
        f"Tôi xác nhận muốn xóa sinh viên {selected_student['id']}",
        key=f"confirm_delete_{selected_id}",
    )

    if st.button(
        "🗑️ Xóa sinh viên",
        key=f"delete_{selected_id}",
        disabled=not confirm_delete,
        type="primary",
        use_container_width=True,
    ):
        result = service_delete_student(settings, selected_id)
        if result["ok"]:
            st.success(result["message"])
            st.rerun()
        else:
            st.error(result["message"])
