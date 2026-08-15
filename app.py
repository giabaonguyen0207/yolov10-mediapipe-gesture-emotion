import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import random 
import mediapipe as mp
from smoothing import LabelSmoother
from pipeline_utils import save_data
from ultralytics import YOLO

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(
    page_title="AI Gesture & Emotion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Hệ thống nhận diện Cảm xúc & Cử chỉ tay Real-time")
st.markdown("Tích hợp song song MediaPipe Face Mesh và Bounding Box Động")

# 2. XÂY DỰNG SIDEBAR (BẢNG ĐIỀU KHIỂN)
st.sidebar.header("⚙️ Bảng Điều Khiển")
hand_model = YOLO("hand_yolov10_best.pt")

run_camera = st.sidebar.checkbox("🟢 Bật Camera")

conf_threshold = st.sidebar.slider(
    "Ngưỡng nhạy (Confidence Threshold)", 
    min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.markdown("---")

if st.sidebar.button("📸 Chụp ảnh khoảnh khắc (Snapshot)"):
    st.session_state['take_snapshot'] = True

# 3. BỐ CỤC MÀN HÌNH CHÍNH
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Luồng Video Webcam")
    video_placeholder = st.empty()

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

# 4. VÒNG LẶP CAMERA & XỬ LÝ AI
if not run_camera:
    video_placeholder.info("Vui lòng check vào ô Bật Camera bên trái để bắt đầu nhận diện.")
else:
    cap = cv2.VideoCapture(0)
    smoother = LabelSmoother(window_size=10)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4, min_detection_confidence=conf_threshold)
    
    all_emotions = ["angry", "confused", "disgust", "fear", "happy", "neutral", "sad", "shy", "surprise"]

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi: Không thể kết nối với Webcam!")
            break
            
        # A. NHẬN DIỆN TAY BẰNG YOLOV10
        # Đưa frame gốc (BGR) vào YOLO và lấy kết quả
        yolo_results = hand_model(frame, conf=conf_threshold, imgsz=320, verbose=False)
        
        # YOLO tự động vẽ Bounding Box và tên Cử chỉ đè lên luôn khung hình
        frame = yolo_results[0].plot()

        # Rút trích tên Cử chỉ
        raw_gesture = "None"
        if len(yolo_results[0].boxes) > 0:
            class_id = int(yolo_results[0].boxes.cls[0].item())
            raw_gesture = hand_model.names[class_id]

        # Sau khi YOLO vẽ xong, chuyển sang hệ màu RGB để MediaPipe và Streamlit xài
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # B. NHẬN DIỆN MẶT BẰNG MEDIAPIPE
        results_face = face_mesh.process(frame_rgb)
        
        # Tạm thời vẫn dùng random cho Cảm xúc (Đợi TV4)
        raw_emotion = random.choices(all_emotions, weights=[0.025, 0.025, 0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025])[0]
        
        # Làm mượt cả 2 nhãn
        final_emotion, final_gesture = smoother.update(raw_emotion, raw_gesture)
        
        # VẼ BOUNDING BOX KHUÔN MẶT
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                h, w, _ = frame_rgb.shape
                f_x_min, f_y_min = w, h
                f_x_max, f_y_max = 0, 0
                
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < f_x_min: f_x_min = x
                    if y < f_y_min: f_y_min = y
                    if x > f_x_max: f_x_max = x
                    if y > f_y_max: f_y_max = y
                
                f_x_min = max(0, f_x_min - 20)
                f_y_min = max(0, f_y_min - 20)
                f_x_max = min(w, f_x_max + 20)
                f_y_max = min(h, f_y_max + 20)
                
                cv2.rectangle(frame_rgb, (f_x_min, f_y_min), (f_x_max, f_y_max), (255, 0, 0), 2)
                cv2.putText(frame_rgb, f"{final_emotion} ({conf_threshold:.2f})", 
                            (f_x_min, f_y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)
                
        # ==========================================
        # C. CẬP NHẬT GIAO DIỆN VÀ ĐẨY ẢNH LÊN WEB
        # ==========================================
        emotion_text.markdown(f"**Nhãn:** {final_emotion}")
        gesture_text.markdown(f"**Nhãn:** {final_gesture}")
        face_conf_bar.progress(random.uniform(0.75, 0.95))
        hand_conf_bar.progress(random.uniform(0.70, 0.90))

        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        if st.session_state.get('take_snapshot'):
            saved_path = save_data(frame_rgb, final_emotion, final_gesture)
            st.sidebar.success(f"Đã lưu ảnh tại: {saved_path}")
            st.session_state['take_snapshot'] = False
            
    cap.release()