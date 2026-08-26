# Quản lý Bộ dữ liệu (Dataset Management)

Thư mục này là nơi quản lý và lưu trữ dữ liệu đầu vào phục vụ huấn luyện mô hình **YOLOv10** và mô hình **Fusion Emotion Model**.

---

## 🖐️ 1. Bộ dữ liệu Fine-tune YOLOv10 (Hand Gesture Recognition)

Mô hình YOLOv10 nhận diện cử chỉ tay được fine-tune từ bộ dữ liệu Roboflow Universe:

- 🔗 **Link Dataset (Roboflow Universe v4)**: [Common Hand Gestures Emoji Dataset v4](https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/4)
- **Vị trí lưu trữ**: Tải về và giải nén vào `data/raw/roboflow_raw/`
- **Quy trình xử lý**: Chạy [src/preprocess_labels.py](file:///d:/Bao%20Lap%20Trinh%20AI/EE_MLIOT/yolov10-mediapipe-gesture-emotion/src/preprocess_labels.py) để lọc nhãn và tạo tập `data/processed/Preprocessed_DTS`.

---

## 🎭 2. Bộ dữ liệu Huấn luyện Fusion Model (Facial Emotion Recognition)

Mô hình Fusion (EfficientNet-B0 + MediaPipe Face Mesh) được huấn luyện kết hợp từ **2 bộ dữ liệu trên Kaggle**:

1. 🔗 **Facial Emotion Dataset**:
   - Link Kaggle: [https://www.kaggle.com/datasets/himanshuydv11/facial-emotion-dataset](https://www.kaggle.com/datasets/himanshuydv11/facial-emotion-dataset)
   - Mô tả: Chứa hình ảnh các biểu cảm khuôn mặt cơ bản.

2. 🔗 **Human Emotions Dataset with Real-World Images**:
   - Link Kaggle: [https://www.kaggle.com/datasets/tasneembinmahmood/human-emotions-dataset-with-real-world-images](https://www.kaggle.com/datasets/tasneembinmahmood/human-emotions-dataset-with-real-world-images)
   - Mô tả: Ảnh khuôn mặt thực tế trong nhiều điều kiện ánh sáng và góc quay khác nhau.

---

## 📁 Cấu trúc lưu trữ dữ liệu

```text
data/
├── emotion/                     # Chứa hình ảnh được tải về từ kaggle
├── gesture/
│    ├────raw/                   # Chứa dữ liệu gốc tải về (chưa tiền xử lý)
│    │    └── roboflow_raw/      # Dataset Roboflow thô
│    └────processed/             # Dữ liệu đã làm sạch & phân chia train/val/test
│         └── Preprocessed_DTS/     # Dataset YOLOv10 đã chuẩn hóa
└── README.md                 # Hướng dẫn quản lý dữ liệu
```
