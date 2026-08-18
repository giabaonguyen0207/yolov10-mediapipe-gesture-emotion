# YOLOv10 & EfficientNetB0 + MediaPipe Face Mesh Gesture + Emotion Recognition

Dự án kết hợp nhận diện cử chỉ tay (Hand Gesture Recognition) bằng **YOLOv10** và nhận diện cảm xúc khuôn mặt Real-time (Facial Emotion Recognition) bằng mô hình Fusion giữa **EfficientNet-B0** và **MediaPipe Face Mesh (478 Landmarks)**.

---

## Cấu trúc Thư mục Dự án

```text
yolov10-mediapipe-gesture-emotion/
├── config/                     # File cấu hình (dữ liệu, dataset YAML)
│   └── data.yaml               # Cấu hình dataset YOLO
│
├── data/                       # Dữ liệu dự án
│   ├── raw/                    # Dữ liệu thô (Roboflow, Kaggle datasets)
│   ├── processed/              # Dữ liệu đã xử lý & gán nhãn
│   └── README.md               # Mô tả & link các bộ dữ liệu
│
├── docs/                       # Tài liệu hướng dẫn & Workflow
│   ├── GIT_WORKFLOW.md         # Quy trình làm việc với Git
│   └── Class_ID.txt            # Danh sách nhãn lớp
│
├── models/                     # Trọng số mô hình đã huấn luyện (.pt)
│   ├── fusion_model_partial.pt # Trọng số mô hình Fusion Emotion
│   ├── hand_yolov10_best.pt    # Trọng số YOLOv10 nhận diện bàn tay
│   └── YOLOv10n_gestures.pt    # Trọng số YOLOv10 nhận diện cử chỉ
│
├── notebooks/                  # Các bài toán thử nghiệm (Jupyter Notebooks)
│   └── 01_data_to_csv.ipynb    # Trích xuất landmark MediaPipe sang CSV
│
├── src/                        # Mã nguồn xử lý & huấn luyện mô hình
│   ├── preprocess_labels.py    # Tiền xử lý & làm sạch nhãn YOLO
│   ├── train_yolo.py           # Pipeline huấn luyện YOLOv10
│   └── train_fusion.py         # Pipeline huấn luyện Fusion (EfficientNet + Landmark)
│
├── demo_emotion_fusion.py      # DEMO 1: Dự đoán Cảm xúc Real-time (Fusion Model)
├── demo_gesture_yolo.py        # DEMO 2: Nhận diện Cử chỉ tay qua Webcam (YOLOv10)
├── requirements.txt            # Danh sách thư viện phụ thuộc
└── README.md                   # Tài liệu tổng quan dự án
```

---

## Hướng dẫn Nhanh

### 1. Cài đặt môi trường & thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy Demo 1: Dự đoán Cảm xúc Real-time (Fusion Model)
```bash
python demo_emotion_fusion.py
```
*(Tự động nhận diện khuôn mặt qua webcam, trích xuất 478 landmarks và kết hợp ảnh crop mặt để dự đoán cảm xúc: angry, happy, neutral, sad, surprise).*

### 3. Chạy Demo 2: Nhận diện Cử chỉ tay (YOLOv10)
```bash
python demo_gesture_yolo.py
```

---

## Các Bộ dữ liệu Sử dụng (Datasets)

1. **Fine-tune YOLOv10 (Hand Gesture Recognition)**:
   - [Roboflow Common Hand Gestures Emoji v4](https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/4)

2. **Huấn luyện Fusion Emotion Model**:
   - [Facial Emotion Dataset (Kaggle)](https://www.kaggle.com/datasets/himanshuydv11/facial-emotion-dataset)
   - [Human Emotions Dataset with Real-World Images (Kaggle)](https://www.kaggle.com/datasets/tasneembinmahmood/human-emotions-dataset-with-real-world-images)

Chi tiết cấu trúc và cách cài đặt tham khảo thêm tại [data/README.md](file:///d:/Bao%20Lap%20Trinh%20AI/EE_MLIOT/yolov10-mediapipe-gesture-emotion/data/README.md).

---

## Công nghệ Sử dụng
- **PyTorch** & **Torchvision**: Xây dựng kiến trúc mô hình Fusion (EfficientNet-B0 + MLP Landmark Encoder).
- **Ultralytics YOLOv10**: Nhận diện cử chỉ tay với tốc độ cao.
- **MediaPipe Face Mesh**: Trích xuất 478 điểm mốc khuôn mặt (face landmarks).
- **OpenCV & PIL**: Xử lý hình ảnh, xoay chỉnh góc mặt (face alignment) và hiển thị luồng webcam.
