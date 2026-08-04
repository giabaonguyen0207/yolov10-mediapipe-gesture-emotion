import streamlit as st
import cv2
import pandas as pd
from datetime import datetime

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(
    page_title="AI Gesture & Emotion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Hệ thống nhận diện Cảm xúc & Cử chỉ tay Real-time")
st.markdown("Tích hợp song song MediaPipe Face Mesh và YOLOv10")

# ==========================================
# 2. XÂY DỰNG SIDEBAR (BẢNG ĐIỀU KHIỂN)
# ==========================================
st.sidebar.header("⚙️ Bảng Điều Khiển")

run_camera = st.sidebar.checkbox("🟢 Bật Camera & AI")

conf_threshold = st.sidebar.slider(
    "Ngưỡng nhạy (Confidence Threshold)", 
    min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.markdown("---")

if st.sidebar.button("📸 Chụp ảnh khoảnh khắc (Snapshot)"):
    st.sidebar.success("Đã chụp ảnh và lưu log CSV!")

# ==========================================
# 3. BỐ CỤC MÀN HÌNH CHÍNH (CHIA CỘT)
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Luồng Video Webcam")
    # Khung chứa video
    video_placeholder = st.empty()
    
    if not run_camera:
        video_placeholder.info("Vui lòng check vào ô Bật Camera bên trái để bắt đầu nhận diện.")
    else:
        # 1. Khởi động Webcam
        cap = cv2.VideoCapture(0)
        
        # CHỈ DÙNG MỘT VÒNG LẶP DUY NHẤT NHƯ SAU:
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Lỗi: Không thể kết nối với Webcam!")
                break
                
            # Mẹo nhỏ: OpenCV dùng hệ màu BGR, trong khi Streamlit hiển thị web cần RGB
            # Nên ta phải đảo ngược hệ màu lại để mặt không bị "xanh lè"
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Đẩy khung hình lên khung giao diện rỗng đã tạo
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
        # Giải phóng camera khi tắt (dòng này phải thụt lề bằng với chữ 'while')
        cap.release()

with col2:
    st.subheader("📊 Kết quả nhận diện")
    
    st.markdown("#### Khuôn mặt (Emotion)")
    emotion_text = st.empty()
    emotion_text.markdown("**Nhãn:** Đang chờ...")
    face_conf_bar = st.progress(0.0)
    
    st.markdown("---")
    
    st.markdown("#### Cử chỉ tay (Gesture)")
    gesture_text = st.empty()
    gesture_text.markdown("**Nhãn:** Đang chờ...")
    hand_conf_bar = st.progress(0.0)