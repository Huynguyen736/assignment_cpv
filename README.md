# Hướng dẫn sử dụng `assignment_face_v2`

`assignment_face_v2` là ứng dụng điểm danh bằng nhận diện khuôn mặt viết bằng Streamlit. Ứng dụng hỗ trợ:

- đăng ký sinh viên bằng video 10 giây,
- tách và lưu các mẫu khuôn mặt hợp lệ,
- train lại model nhận diện,
- điểm danh trực tiếp từ camera và lưu kết quả vào file CSV.

## 1. Cài đặt

Yêu cầu:

- Python `>= 3.13`
- webcam hoặc nguồn video từ trình duyệt

Nếu dùng `uv`:

```bash
uv sync
```

Nếu dùng `pip`:

```bash
pip install -r assignment_face_v2/requirements.txt
```

## 2. Cấu hình

Tạo file `.env` cho module:

```bash
copy assignment_face_v2/.env.example assignment_face_v2/.env
```

Nội dung mẫu:

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

Lưu ý:

- `WEBCAM_INDEX=0` là webcam mặc định.
- `CONFIDENCE_THRESHOLD` càng nhỏ thì điều kiện nhận diện càng chặt.
- App hiện tại stream trực tiếp bằng `streamlit-webrtc`, nên bạn cần cấp quyền camera cho trình duyệt.

## 3. Chạy ứng dụng

Chạy toàn bộ app:

```bash
streamlit run assignment_face_v2/app.py
```

Sau khi chạy, app có 3 màn hình chính:

- `Home`
- `Register Student`
- `Live Attendance`

## 4. Đăng ký khuôn mặt

Vào trang `Register Student`, sau đó:

1. Nhập `Student ID`.
2. Nhập `Student Name`.
3. Bấm `Start Camera`.
4. Bấm `Start Capture`.
5. Giữ mặt trong khung hình và di chuyển tự nhiên trong 10 giây.

Hệ thống sẽ:

- quay video trong 10 giây,
- lấy tối đa 5 frame mỗi giây,
- chỉ giữ các frame có đúng 1 khuôn mặt,
- bỏ các frame quá giống nhau,
- lưu các ảnh khuôn mặt vào `assignment_face_v2/database/face_db/<student_id>/`,
- cập nhật `assignment_face_v2/database/students.json`,
- train lại recognizer sau khi lưu xong.

Nếu không thu được frame hợp lệ, app sẽ báo lỗi `No valid distinct face frames found in the 10-second recording`.

## 5. Điểm danh trực tiếp

Vào trang `Live Attendance` để bật stream camera và nhận diện khuôn mặt theo thời gian thực.

Kết quả có thể rơi vào các trạng thái:

- `No face detected`: không phát hiện khuôn mặt.
- `Multiple faces detected`: có hơn 1 khuôn mặt trong khung hình.
- `Unknown student`: khuôn mặt không khớp model.
- `Attendance saved`: nhận diện thành công và đã lưu điểm danh.
- `Already checked today`: sinh viên đã được điểm danh trong ngày.

Dữ liệu điểm danh được ghi vào:

```text
assignment_face_v2/database/attendance.csv
```

Mỗi sinh viên chỉ được ghi 1 lần mỗi ngày.

## 6. Dữ liệu và model

Thư mục/file quan trọng:

- `assignment_face_v2/database/students.json`: danh sách sinh viên.
- `assignment_face_v2/database/face_db/`: ảnh khuôn mặt đã đăng ký.
- `assignment_face_v2/database/attendance.csv`: lịch sử điểm danh.
- `assignment_face_v2/models/lbph_model.npz`: model nhận diện.
- `assignment_face_v2/models/lbph_fallback.npz`: fallback descriptor model.
- `assignment_face_v2/models/label_map.json`: map giữa `student_id` và tên.

Nếu chưa có model, trang `Live Attendance` sẽ yêu cầu đăng ký sinh viên và train trước khi nhận diện.

## 7. Lưu ý sử dụng

- Nên đăng ký mỗi sinh viên với nhiều góc mặt để tăng độ ổn định.
- Trong lúc capture, chỉ nên để 1 khuôn mặt trong khung hình.
- Nếu camera không lên, kiểm tra lại quyền camera của trình duyệt và dependency `streamlit-webrtc`.
- App tự động tạo `students.json`, `attendance.csv` và file Haar Cascade nếu chưa tồn tại.
