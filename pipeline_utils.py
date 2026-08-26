import os
import cv2
import csv
from datetime import datetime

def init_directories():
    """Tự động tạo thư mục lưu trữ nếu chưa tồn tại"""
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def save_data(frame_rgb, emotion, gesture):
    """Lưu ảnh snapshot và ghi log vào file CSV"""
    init_directories()
    
    # 1. Chuẩn bị thời gian thực
    now = datetime.now()
    timestamp_file = now.strftime("%Y%m%d_%H%M%S")
    timestamp_log = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. Xử lý lưu ảnh (Chuyển ngược RGB về BGR để OpenCV lưu đúng màu)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    image_path = f"snapshots/snap_{timestamp_file}.jpg"
    cv2.imwrite(image_path, frame_bgr)
    
    # 3. Xử lý lưu file log CSV
    log_file = "logs/recognition_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Nếu file chưa có thì tạo dòng Tiêu đề (Header) trước
        if not file_exists:
            writer.writerow(["Thời gian", "Cảm xúc (Emotion)", "Cử chỉ (Gesture)", "File ảnh"])
        # Ghi dữ liệu của khoảnh khắc hiện tại
        writer.writerow([timestamp_log, emotion, gesture, image_path])
        
    return image_path