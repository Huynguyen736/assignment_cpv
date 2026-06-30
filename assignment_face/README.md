# Assignment Face - Face Attendance System

Ứng dụng điểm danh bằng nhận diện khuôn mặt, chạy bằng Streamlit, OpenCV và NumPy.

Hướng dẫn này dành cho máy mới chỉ có sẵn Python, không cần cài `uv`, Conda hay công cụ phụ khác.

## 1. Yêu cầu

- Python 3.13 trở lên.
- Webcam nếu muốn dùng camera của máy.
- Trình duyệt web để mở giao diện Streamlit.

Kiểm tra phiên bản Python:

```bash
python --version
```

Nếu lệnh `python` không chạy trên Windows, thử:

```bash
py --version
```

## 2. Mở đúng thư mục dự án

Mở Terminal, PowerShell hoặc Command Prompt tại thư mục chứa `pyproject.toml` và thư mục `assignment_face`.

Ví dụ nếu đang đứng trong thư mục `assignment_face`, quay ra thư mục cha:

```bash
cd ..
```

## 3. Tạo môi trường ảo

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn activate script, chạy thêm:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sau khi activate thành công, dòng lệnh thường sẽ có tiền tố `(.venv)`.

## 4. Cài thư viện cần thiết

Cập nhật pip trước:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Cài dependency từ `pyproject.toml` của dự án:

```bash
python -m pip install -e .
```

Nếu chỉ muốn cài nhóm thư viện tối thiểu để chạy app `assignment_face`, có thể dùng:

```bash
python -m pip install streamlit streamlit-webrtc opencv-contrib-python numpy python-dotenv
```

## 5. Tạo file cấu hình `.env`

Tạo file cấu hình từ file mẫu.

### Windows PowerShell

```powershell
Copy-Item assignment_face\.env.example assignment_face\.env
```

### macOS / Linux

```bash
cp assignment_face/.env.example assignment_face/.env
```

Mở `assignment_face/.env` và chỉnh nếu cần:

```env
RTSP_URL=rtsp://username:password@192.168.1.100:554/stream
WEBCAM_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480
FACE_WIDTH=200
FACE_HEIGHT=200
CONFIDENCE_THRESHOLD=70
BLUR_KERNEL=3
```

Ghi chú:

- Nếu dùng webcam máy tính, thường giữ `WEBCAM_INDEX=0`.
- Nếu dùng camera IP/RTSP, thay `RTSP_URL` bằng URL thật của camera.
- Không chia sẻ hoặc commit file `.env` nếu trong đó có username/password camera.

## 6. Chạy ứng dụng

Từ thư mục root của dự án, chạy:

```bash
python -m streamlit run assignment_face/app.py
```

Streamlit sẽ mở trình duyệt. Nếu không tự mở, truy cập:

```text
http://localhost:8501
```

Trong sidebar của Streamlit có các trang:

- `Live Attendance`: mở camera và điểm danh trực tiếp.
- `Register Student`: đăng ký sinh viên mới bằng khuôn mặt.
- `Manage Students`: xem, sửa tên hoặc xóa sinh viên.

## 7. Dừng ứng dụng

Trong terminal đang chạy Streamlit, nhấn:

```text
Ctrl + C
```

## 8. Lỗi thường gặp

### `ModuleNotFoundError`

Dependency chưa được cài đúng môi trường ảo. Activate lại `.venv`, rồi chạy:

```bash
python -m pip install -e .
```

### `streamlit-webrtc is not installed`

Cài lại dependency WebRTC:

```bash
python -m pip install streamlit-webrtc
```

### Không mở được webcam

Kiểm tra các điểm sau:

- Trình duyệt đã được cấp quyền camera.
- Webcam không bị ứng dụng khác chiếm.
- Thử đổi `WEBCAM_INDEX=1` trong `assignment_face/.env`, rồi chạy lại app.

### OpenCV hoặc `cv2` lỗi khi import

Cài lại OpenCV:

```bash
python -m pip install --upgrade --force-reinstall opencv-contrib-python
```

### PowerShell không chạy được activate script

Chạy lệnh này trong cùng cửa sổ PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó activate lại:

```powershell
.\.venv\Scripts\Activate.ps1
```
