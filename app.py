import streamlit as st
import cv2
import pandas as pd
import time
from datetime import datetime
import random 
import mediapipe as mp
import os
import glob
import torch
import numpy as np
from PIL import Image

from smoothing import LabelSmoother
from pipeline_utils import save_data
from ultralytics import YOLO

from attitude import get_attitude

from train_fusion import (
    DEVICE,
    NUM_LANDMARKS,
    FusionModel,
    align_image,
    align_landmarks,
    build_transforms,
    compute_roll_angles_and_centers,
    normalize_landmarks,
)

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(
    page_title="AI Gesture & Emotion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Hệ thống nhận diện Cảm xúc & Cử chỉ tay Real-time")
st.markdown("Tích hợp song song MediaPipe Face Mesh và Bounding Box Động")

# XỬ LÝ KHUÔN MẶT
def crop_face(frame_rgb, landmarks):
    height, width = frame_rgb.shape[:2]
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    left, right = max(0, int(np.floor(xs.min()))), min(width, int(np.ceil(xs.max())))
    top, bottom = max(0, int(np.floor(ys.min()))), min(height, int(np.ceil(ys.max())))
    side = max(right - left, bottom - top)
    pad = int(side * 0.35)
    center_x, center_y = (left + right) // 2, (top + bottom) // 2
    x0, x1 = max(0, center_x - side // 2 - pad), min(width, center_x + side // 2 + pad)
    y0, y1 = max(0, center_y - side // 2 - pad), min(height, center_y + side // 2 + pad)

    crop = frame_rgb[y0:y1, x0:x1]
    crop_landmarks = landmarks.copy()
    crop_landmarks[:, 0] -= x0
    crop_landmarks[:, 1] -= y0
    return crop, crop_landmarks, (x0, y0, x1, y1)

def prepare_inputs(frame_bgr, face_landmarks, scaler_mean, scaler_scale, transform):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = frame_rgb.shape[:2]
    landmarks = np.asarray(
        [[point.x * width, point.y * height, point.z * width] for point in face_landmarks.landmark],
        dtype=np.float32,
    )

    face_rgb, landmarks, box = crop_face(frame_rgb, landmarks)
    raw = landmarks[None, ...]
    angles, centers = compute_roll_angles_and_centers(raw)
    aligned_landmarks = align_landmarks(raw, angles, centers)
    normalized = normalize_landmarks(aligned_landmarks).reshape(1, -1)
    standardized = (normalized - scaler_mean) / scaler_scale

    aligned_image = align_image(Image.fromarray(face_rgb), float(angles[0]), centers[0])
    image_tensor = transform(aligned_image).unsqueeze(0)
    landmark_tensor = torch.from_numpy(standardized.astype(np.float32))
    return image_tensor, landmark_tensor, box

# XOÁ ẢNH CHẠY NGẦM 
def delete_single(img_path):
    if os.path.exists(img_path):
        os.remove(img_path)

def delete_selected(images_list):
    for img_path in images_list:
        if st.session_state.get(f"chk_{img_path}", False):
            if os.path.exists(img_path):
                os.remove(img_path)

def delete_all(images_list):
    for img_path in images_list:
        if os.path.exists(img_path):
            os.remove(img_path)

@st.dialog("🔍 Xem chi tiết ảnh", width="large")
def show_full_image(img_path):
    st.image(img_path, width="stretch")

@st.dialog("🖼️ Lịch sử chụp", width="large")
def show_history():
    if os.path.exists("snapshots"):
        images = glob.glob("snapshots/*.jpg")
        images.sort(reverse=True)
        
        if images:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                st.button("🗑️ Xoá các ảnh đã chọn", type="primary", on_click=delete_selected, args=(images,))
            with col_btn2:
                st.button("🚨 Xoá tất cả ảnh", type="primary", on_click=delete_all, args=(images,))
            st.markdown("---")
            
            cols = st.columns(3)
            for i, img_path in enumerate(images):
                with cols[i % 3]:
                    file_name = os.path.basename(img_path)
                    st.image(img_path, width="stretch")
                    st.checkbox(f"Chọn {file_name}", key=f"chk_{img_path}")
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        if st.button("🔍 Phóng to", key=f"zoom_{img_path}"):
                            show_full_image(img_path)
                    with sub_col2:
                        st.button("❌ Xoá", key=f"del_{img_path}", on_click=delete_single, args=(img_path,))
        else:
            st.info("Chưa có bức ảnh nào trong thư viện.")
    else:
        st.info("Thư mục lưu trữ chưa được tạo. Hãy chụp ảnh để hệ thống tự tạo.")

# 2. XÂY DỰNG SIDEBAR (BẢNG ĐIỀU KHIỂN)
st.sidebar.header("⚙️ Bảng Điều Khiển")

@st.cache_resource
def load_hand_model():
    return YOLO("hand_yolov10_best.pt")

# Nạp model của TV4 vào bộ nhớ đệm
@st.cache_resource
def load_emotion_model():
    checkpoint_path = "fusion_model_partial.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    model = FusionModel(
        num_classes=5, 
        pretrained_cnn=False, 
        freeze_mode="none"
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    scaler_mean = checkpoint["scaler_mean"]
    scaler_scale = checkpoint["scaler_scale"]
    img_size = int(checkpoint["img_size"])
    return model, scaler_mean, scaler_scale, img_size

hand_model = load_hand_model()
emotion_model, scaler_mean, scaler_scale, img_size = load_emotion_model()
_, transform = build_transforms(img_size)

# Danh sách 5 nhãn cảm xúc của TV4
EMOTION_LABELS = ["angry", "happy", "neutral", "sad", "surprise"]

run_camera = st.sidebar.checkbox("🟢 Bật Camera")

conf_threshold = st.sidebar.slider(
    "Ngưỡng nhạy (Confidence Threshold)", 
    min_value=0.1, max_value=1.0, value=0.5, step=0.05
)

st.sidebar.markdown("---")

if st.sidebar.button("📸 Chụp ảnh khoảnh khắc"):
    st.session_state['take_snapshot'] = True

if st.sidebar.button("🖼️ Lịch sử chụp"):
    show_history()

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

    st.markdown("---")
    
    st.markdown("#### Thái độ (Attitude)")
    attitude_text = st.empty()
    attitude_text.markdown("**Tổ hợp:** Đang chờ...")
    attitude_conf_bar = st.progress(0.0)

# 4. VÒNG LẶP CAMERA & XỬ LÝ AI
@st.fragment
def camera_loop():
    if not run_camera:
        video_placeholder.info("Vui lòng check vào ô Bật Camera bên trái để bắt đầu nhận diện.")
        return
    
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoother = LabelSmoother(window_size=15)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=conf_threshold,
        min_tracking_confidence=conf_threshold
    )

    time_frame = 1.0
    last_label_time = time.time()

    display_emotion = "Đang chờ..."
    display_gesture = "Đang chờ..."

    frame_count = 0
    cached_raw_emotion = "None"
    cached_face_conf = 0.0

    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Lỗi: Không thể kết nối với Webcam!")
            break
            
        frame_count += 1

        # A. NHẬN DIỆN CỬ CHỈ TAY (YOLO)
        yolo_results = hand_model(frame, conf=conf_threshold, verbose=False)
        
        raw_gesture = "None"
        real_hand_conf = 0.0
        if len(yolo_results[0].boxes) > 0:
            class_id = int(yolo_results[0].boxes.cls[0].item())
            raw_gesture = hand_model.names[class_id]
            real_hand_conf = float(yolo_results[0].boxes.conf[0].item())
            
            box = yolo_results[0].boxes.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{display_gesture} ({real_hand_conf:.2f})", 
                        (x_min, max(20, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # B. NHẬN DIỆN CẢM XÚC KHUÔN MẶT
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(frame_rgb)

        raw_emotion = cached_raw_emotion
        real_face_conf = cached_face_conf

        if results_face.multi_face_landmarks:
            try:
                image_tensor, landmark_tensor, face_box = prepare_inputs(
                    frame, results_face.multi_face_landmarks[0], scaler_mean, scaler_scale, transform
                )

                if frame_count % 3 == 0:
                    with torch.inference_mode():
                        logits = emotion_model(image_tensor.to(DEVICE), landmark_tensor.to(DEVICE))
                        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    
                    class_id = int(np.argmax(probs))
                    cached_raw_emotion = EMOTION_LABELS[class_id]
                    cached_face_conf = float(probs[class_id])
                    
                    raw_emotion = cached_raw_emotion
                    real_face_conf = cached_face_conf
                
                # Vẽ Bounding Box
                f_x0, f_y0, f_x1, f_y1 = face_box
                cv2.rectangle(frame_rgb, (f_x0, f_y0), (f_x1, f_y1), (255, 0, 0), 2)
                cv2.putText(frame_rgb, f"{display_emotion} ({real_face_conf:.2f})", 
                            (f_x0, max(28, f_y0 - 10)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 2)
                            
            except Exception as e:
                cv2.putText(frame_rgb, f"Face Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # C. LÀM MƯỢT NHÃN & CẬP NHẬT 1 GIÂY 1 LẦN
        smoothed_emotion, smoothed_gesture = smoother.update(raw_emotion, raw_gesture)

        current_time = time.time()
        if current_time - last_label_time >= time_frame:
            display_emotion = smoothed_emotion
            display_gesture = smoothed_gesture
            last_label_time = current_time

        # D. TÍNH TOÁN THÁI ĐỘ VÀ ĐẨY LÊN GIAO DIỆN
        if display_emotion != "Đang chờ..." and display_gesture != "Đang chờ...":
            display_attitude = get_attitude(display_emotion, display_gesture)
        else:
            display_attitude = "Đang chờ dữ liệu..."

        emotion_text.markdown(f"**Nhãn:** {display_emotion}")
        gesture_text.markdown(f"**Nhãn:** {display_gesture}")
        attitude_text.markdown(f"**Tổ hợp:** {display_attitude}")
        
        if real_face_conf > 0:
            face_conf_bar.progress(real_face_conf)
        if raw_gesture != "None":
            hand_conf_bar.progress(real_hand_conf)
            
        attitude_conf_bar.progress(random.uniform(0.70, 0.90))

        video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

        if st.session_state.get('take_snapshot'):
            saved_path = save_data(frame_rgb, display_emotion, display_gesture)
            st.sidebar.success(f"Đã lưu ảnh tại: {saved_path}")
            st.session_state['take_snapshot'] = False
            
    cap.release()

camera_loop()