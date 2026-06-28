# Version Detect 2 - Cập nhật Register Face và LBP

Ngày 2026-06-28, luồng đăng ký và nhận diện khuôn mặt được chỉnh để giảm treo, giảm nhiễu nền trong LBP và dễ debug hơn.

## 1. Sửa lỗi Register Face bị treo

Nguyên nhân là `FaceDetector.detect()` dùng `HaarCascade` tự viết bằng Python, quét cửa sổ ảnh rất chậm. Benchmark trước khi sửa: một frame 640x480 mất khoảng `95.85s`.

Đã đổi runtime detector sang `cv2.CascadeClassifier.detectMultiScale` trong `assignment_face/core/detector.py`, vẫn giữ các bước crop, align landmark và resize sau detect.

Kết quả sau sửa: một frame 640x480 còn khoảng `0.68s`; luồng register 50 frame trống xử lý khoảng `5s`.

## 2. Sửa cách lấy frame khi đăng ký

Trong `assignment_face/core/crop_video.py`, `sample_recording_frames()` trước đây có thể bỏ mất frame cuối mỗi bucket.

Đã đổi sang cách lấy mẫu phân bố đều từ đầu đến cuối bucket, giúp video 10 giây có mẫu đại diện tốt hơn.

## 3. Thêm debug ảnh đầu vào LBP

Trong `assignment_face/core/recognizer.py`, `RecognitionResult` trả thêm:

- `reference_image_path`
- `reference_lbp_input`
- `query_lbp_input`

Trang Live Attendance có thể hiển thị hai ảnh:

- ảnh đăng ký gần nhất được dùng làm reference
- ảnh live đang được dùng để dự đoán

Các field này được truyền qua `assignment_face/services/live_attendance.py` và `assignment_face/ui/processors.py`.

## 4. Thêm nút bật/tắt ảnh debug

Trong `assignment_face/ui/pages/live_attendance.py`, thêm toggle `Show LBP debug images`.

Mặc định tắt. Khi bật, UI mới hiển thị hai ảnh đầu vào LBP.

## 5. Giảm nền lọt vào ảnh LBP

LBP lấy texture trên toàn ảnh crop, nên nền ở viền có thể làm cùng một mặt bị lệch vector khi chụp ở địa điểm khác.

Đã tách hai loại crop trong `assignment_face/core/detector.py`:

- `FACE_CROP_EXPANSION = 0.15`: dùng cho ảnh đưa vào LBP
- `BOX_DISPLAY_EXPANSION = 0.4`: chỉ dùng cho box hiển thị trên camera

Trong `assignment_face/core/recognizer.py`, thêm `prepare_lbp_input()`:

- chuyển ảnh sang grayscale
- center-crop 76% vùng giữa
- resize lại về kích thước ban đầu

Cả ảnh đăng ký và ảnh dự đoán đều đi qua bước này trước khi trích vector LBP.

## 6. Version lại model LBP

Đổi `LBP_VARIANT` thành:

```text
uniform-lbp-u2-p8-r1-center-crop-0.76
```

Nếu app gặp model cũ, recognizer sẽ tự retrain từ `face_db` để tránh so ảnh query đã crop mới với descriptor cũ.

## 7. Ghi chú về confidence

`confidence` hiện tại thực chất là Chi-square distance giữa hai vector LBP:

- càng thấp: càng giống
- càng cao: càng khác
- nhận diện thành công khi `distance <= CONFIDENCE_THRESHOLD`

Nên hiểu giá trị này là `LBP distance`, không phải phần trăm độ tin cậy.

## 8. Kiểm chứng

Đã chạy:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_assignment_face_app.py -q
```

Kết quả:

```text
18 passed
```
