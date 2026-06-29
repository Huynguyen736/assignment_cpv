import cv2
import time
import numpy as np
from typing import List, Dict, Any, Union

class CameraCapture:
    """
    Class xử lý đầu vào từ camera (Webcam local hoặc RTSP IP Camera).
    Có thể sử dụng để quay video độc lập đưa vào backend xử lý.
    """
    def __init__(self, source: Union[int, str] = 0):
        """
        Khởi tạo kết nối camera.
        
        Args:
            source: 0 cho webcam mặc định, hoặc URL RTSP (vd: "rtsp://user:pass@192.168.1.100/stream")
        """
        self.source = source

    def record_video(self, duration_seconds: int = 10, fps: int = 5) -> List[Dict[str, Any]]:
        """
        Quay video trong khoảng thời gian nhất định và trả về danh sách frame.
        Cấu trúc trả về khớp 100% với yêu cầu của backend (vd: `sample_recording_frames`).
        
        Args:
            duration_seconds: Số giây cần quay (mặc định 10s cho đăng ký)
            fps: Tần số lấy mẫu (mặc định 5 fps là đủ cho nhận diện khuôn mặt)
            
        Returns:
            list[dict]: Mỗi phần tử là {"timestamp": float, "frame": np.ndarray}
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể kết nối đến camera: {self.source}")

        frames = []
        start_time = time.monotonic()
        last_saved_time = 0.0
        frame_interval = 1.0 / fps

        print(f"🎥 Bắt đầu quay video từ {self.source} trong {duration_seconds} giây...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Không thể đọc frame từ camera (có thể đã mất kết nối hoặc hết video).")
                break

            now = time.monotonic()
            elapsed = now - start_time

            if elapsed >= duration_seconds:
                break

            # Chỉ lưu frame theo đúng định mức FPS yêu cầu
            if (now - last_saved_time) >= frame_interval:
                frames.append({
                    "timestamp": elapsed,
                    "frame": frame.copy()
                })
                last_saved_time = now

        cap.release()
        print(f"✅ Quay xong! Đã thu thập được {len(frames)} frames.")
        return frames

    def stream_live(self):
        """
        Generator trả về frame liên tục, dùng cho Live Attendance nếu muốn chạy CLI/Backend thuần
        không dùng Streamlit WebRTC.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể kết nối đến camera: {self.source}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
