import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
import random 
import mediapipe as mp
from smoothing import LabelSmoother
from pipeline_utils import save_data

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(
    page_title="AI Gesture & Emotion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Hệ thống nhận diện Cảm xúc & Cử chỉ tay Real-time")
st.markdown("Tích hợp song song MediaPipe Face Mesh và Bounding Box Động")

# ==========================================
# 2. XÂY DỰNG SIDEBAR (BẢNG ĐIỀU KHIỂN)
# ==========================================
st.sidebar.header("⚙️ Bảng Điều Khiển")

run_camera = st.sidebar.checkbox("🟢 Bật Camera")

# THANH TRƯỢT NÀY GIỜ ĐÃ CÓ TÁC DỤNG THẬT SỰ VÀO MÔ HÌNH!
conf_threshold = st.sidebar.slider(
    "Ngưỡng nhạy (Confidence Threshold)", 
    min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.markdown("---")

if st.sidebar.button("📸 Chụp ảnh khoảnh khắc (Snapshot)"):
    st.session_state['take_snapshot'] = True

# ==========================================
# 3. BỐ CỤC MÀN HÌNH CHÍNH
# ==========================================
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

# ==========================================
# 4. VÒNG LẶP CAMERA & XỬ LÝ AI
# ==========================================
if not run_camera:
    video_placeholder.info("Vui lòng check vào ô Bật Camera bên trái để bắt đầu nhận diện.")
else:
    cap = cv2.VideoCapture(0)
    smoother = LabelSmoother(window_size=10)
    
    # KHỞI TẠO BỘ NÃO AI (MEDIAPIPE)
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Nạp thanh trượt conf_threshold vào AI
    hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=conf_threshold)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=4, min_detection_confidence=conf_threshold)
    
    all_emotions = ["angry", "confused", "disgust", "fear", "happy", "neutral", "sad", "shy", "surprise"]
    all_gestures = ["Heart", "Like", "Dislike", "OK", "Stop"]

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi: Không thể kết nối với Webcam!")
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ------------------------------------------
        # A. CHO AI QUÉT HÌNH ẢNH THỰC TẾ
        # ------------------------------------------
        results_hands = hands.process(frame_rgb)
        results_face = face_mesh.process(frame_rgb)
        
        # (Vẫn giữ giả lập nhãn vì chưa có file classification của TV3, TV4)
        raw_emotion = random.choices(all_emotions, weights=[0.025, 0.025, 0.025, 0.025, 0.8, 0.025, 0.025, 0.025, 0.025])[0]
        raw_gesture = random.choices(all_gestures, weights=[0.8, 0.05, 0.05, 0.05, 0.05])[0]
        final_emotion, final_gesture = smoother.update(raw_emotion, raw_gesture)
        
        # ------------------------------------------
        # B. VẼ BOUNDING BOX KHUÔN MẶT (KHÔNG LƯỚI)
        # ------------------------------------------
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                
                # 1. Thuật toán tự động tính toán Bounding Box ôm sát khuôn mặt
                h, w, _ = frame_rgb.shape
                f_x_min, f_y_min = w, h
                f_x_max, f_y_max = 0, 0
                
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < f_x_min: f_x_min = x
                    if y < f_y_min: f_y_min = y
                    if x > f_x_max: f_x_max = x
                    if y > f_y_max: f_y_max = y
                
                # Mở rộng khung ra 20 pixel để không bị lẹm
                f_x_min = max(0, f_x_min - 20)
                f_y_min = max(0, f_y_min - 20)
                f_x_max = min(w, f_x_max + 20)
                f_y_max = min(h, f_y_max + 20)
                
                # 2. Vẽ Bounding Box màu Đỏ (R=255, G=0, B=0)
                cv2.rectangle(frame_rgb, (f_x_min, f_y_min), (f_x_max, f_y_max), (255, 0, 0), 2)
                
                # 3. In nhãn Cảm xúc TO VÀ RÕ HƠN (Font DUPLEX, size 1.0, nét 2)
                cv2.putText(frame_rgb, f"{final_emotion} ({conf_threshold:.2f})", 
                            (f_x_min, f_y_min - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)
                
        # ------------------------------------------
        # C. VẼ KHUNG XƯƠNG TAY & BOUNDING BOX ĐỘNG
        # ------------------------------------------
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # 1. Vẽ các đường nối ngón tay (Skeleton)
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 2. Thuật toán tự động tính toán Bounding Box ôm sát tay
                h, w, _ = frame_rgb.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y
                
                # Mở rộng khung ra 20 pixel cho đẹp, không bị lẹm ngón tay
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 20)
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)
                
                # 3. Vẽ Bounding Box màu xanh lá
                cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 4. In nhãn bám sát khung
                cv2.putText(frame_rgb, f"{final_gesture} ({conf_threshold:.2f})", 
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # CẬP NHẬT GIAO DIỆN VÀ ĐẨY ẢNH LÊN WEB
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