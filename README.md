# Concurrent Hand Gesture & Facial Emotion Recognition System for Attitude Analysis

> **Hệ thống nhận diện đồng thời Cử chỉ tay và Cảm xúc khuôn mặt thời gian thực trên Web (Streamlit) nhằm đánh giá Ma trận Thái độ (Attitude Matrix - 70 trạng thái).**

Dự án phát triển bởi **Nhóm 2 - Machine Learning & IoT Lab (HCMUT)** trong học phần *Machine Learning & Deep Learning*.

---

## Cấu Trúc Thư Mục Dự Án

```text
yolov10-mediapipe-gesture-emotion/
├── app.py                          # Ứng dụng Web chính (Streamlit Dashboard & Real-time Inference)
├── smoothing.py                    # Bộ lọc làm mượt nhãn (Label Smoother)
├── pipeline_utils.py               # Tiện ích chụp snapshot & ghi log dữ liệu
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── Dockerfile                      # Cấu hình containerization
│
├── models/                         # Trọng số mô hình đã huấn luyện (.pt)
│   ├── hand_yolov10_best.pt        # Trọng số YOLOv10 fine-tune 14 cử chỉ
│   ├── fusion_model_partial.pt     # Trọng số Fusion Emotion (EfficientNet-B0 + MLP)
│   └── YOLOv10n_gestures.pt        # Trọng số gốc pre-trained HaGRID
│
├── Train_model/                    # Mã nguồn huấn luyện & tiền xử lý
│   ├── train_yolo.py               # Pipeline huấn luyện fine-tune YOLOv10
│   ├── train_fusion.py             # Pipeline huấn luyện kiến trúc Fusion & Roll Alignment
│   └── preprocess_labels.py        # Làm sạch và lọc cân bằng nhãn dataset
│
├── docs/                           # Tài liệu & định nghĩa hệ thống
│   ├── attitude.py                 # Ma trận 70 trạng thái thái độ (Attitude Lookup Table)
│   ├── Class_ID.txt                # Danh sách 14 nhãn cử chỉ tay
│   └── GIT_WORKFLOW.md             # Quy trình phát triển nhóm qua Git
│
├── TEMPLATE_PRESENTATION_LAB/      # Mã nguồn LaTeX Beamer báo cáo bảo vệ đồ án
│   ├── main.tex                    # File slide báo cáo chính (12 slides hoàn chỉnh)
│   └── beamerthemeMLIOT.sty        # Theme Beamer chuẩn ML IoT Lab HCMUT
│
├── data/                           # Quản lý tập dữ liệu (Emotion & Gesture)
│   └── README.md                   # Hướng dẫn chi tiết tải & chuẩn bị dataset
└── README.md                       # Tài liệu tổng quan dự án
```

---

## Hướng Dẫn Cài Đặt & Chạy Ứng Dụng

### 1. Chuẩn bị môi trường
Khuyến nghị sử dụng Python 3.10 hoặc 3.11 trong môi trường ảo (venv hoặc conda):

```bash
# Tạo môi trường ảo (Conda)
conda create -n gesture_emotion python=3.10 -y
conda activate gesture_emotion

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Dashboard Real-Time (Khuyên dùng)
Khởi chạy toàn bộ hệ thống phát hiện song song trên giao diện Web:

```bash
streamlit run app.py
```
*Truy cập đường dẫn cục bộ trên trình duyệt: `http://localhost:8501`*

### 3. Chạy các kịch bản kiểm thử đơn thức (Standalone Scripts)
- **Nhận diện Cử chỉ tay (YOLOv10):**
  ```bash
  python demo_gesture_predict.py
  ```
- **Nhận diện Cảm xúc khuôn mặt (Fusion Model):**
  ```bash
  python demo_emotion_predict.py
  ```

---

## Bộ Dữ Liệu Sử Dụng (Datasets)

1. **Cử chỉ tay (Hand Gestures):**
   - Bộ dữ liệu: [Roboflow Common Hand Gestures (Emoji) v4](https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/4).
   - Lọc nhãn: 1.356 ảnh gốc $\rightarrow$ Lọc cân bằng còn **1.041 ảnh** (Train: 834, Val: 101, Test: 106) trải đều trên 14 cử chỉ: `call`, `palm`, `stop`, `hand_heart`, `fist`, `middle_finger`, `ok`, `peace`, `point`, `one`, `holy`, `rock`, `dislike`, `like`.
2. **Cảm xúc khuôn mặt (Facial Emotions):**
   - Kết hợp từ: [Facial Emotion Dataset](https://www.kaggle.com/datasets/himanshuydv11/facial-emotion-dataset) và [Human Emotions Dataset](https://www.kaggle.com/datasets/tasneembinmahmood/human-emotions-dataset-with-real-world-images).
   - Trích xuất thành công 31.454 / 34.439 ảnh ($91.33\%$) với 5 lớp: `angry`, `happy`, `neutral`, `sad`, `surprise`.

---

## Thành Viên Nhóm Thực Hiện (Nhóm 2 - MLIOT Lab)

- **Nguyễn Lê Gia Bảo (Lead)** – Email: `bao.nguyenlegia@hcmut.edu.vn`
- **Dương Gia Bảo**
- **Huỳnh Nguyễn Huy Hoàng**
- **Nguyễn Huỳnh Gia Bảo**
- **Trần Trung Quân**

*Đơn vị: Machine Learning & IoT Lab - Khoa Điện - Điện Tử, Trường Đại học Bách Khoa - ĐHQG TP.HCM.*
