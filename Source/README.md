# EE_MLIOT_F5_FinalProject

# Tiền xử lý dữ liệu cảm xúc khuôn mặt

## Giới thiệu

Module này có nhiệm vụ tiền xử lý bộ dữ liệu **FANE - Facial Expressions and Emotion Dataset** bằng cách sử dụng **MediaPipe Face Mesh** để trích xuất **468 điểm đặc trưng (facial landmarks)** trên khuôn mặt.

Sau khi xử lý, toàn bộ dữ liệu sẽ được lưu thành **một file CSV**, trong đó mỗi dòng tương ứng với một ảnh và bao gồm:

- Nhãn cảm xúc (Emotion Label)
- Tọa độ `(x, y, z)` của 468 điểm trên khuôn mặt

File CSV này được sử dụng làm đầu vào cho các mô hình Machine Learning trong bước huấn luyện.

---

# Tải bộ dữ liệu

Bộ dữ liệu được lấy từ Kaggle:

> https://www.kaggle.com/datasets/furcifer/fane-facial-expressions-and-emotion-dataset

Có thể tải trực tiếp bằng thư viện `kagglehub`:

```bash
kaggle datasets download -d furcifer/fane-facial-expressions-and-emotion-dataset -p ./Dts/ --unzip
```

---

# Cài đặt thư viện

Cài đặt các thư viện cần thiết:

```bash
pip install kagglehub mediapipe opencv-python pandas numpy tqdm
```

---

# Cấu trúc thư mục

Ví dụ:

```text
Source/
│
├── preprocess.py
├── Output/
│   └── face_landmarks.csv
└── README.md
```

---

# Quy trình xử lý

```text
Ảnh khuôn mặt
        │
        ▼
MediaPipe Face Mesh
        │
        ▼
Trích xuất 468 landmarks
        │
        ▼
Lấy tọa độ (x, y, z)
        │
        ▼
Ghép với nhãn cảm xúc
        │
        ▼
Xuất thành file CSV
```

---

# Định dạng dữ liệu đầu ra

Mỗi dòng trong file CSV tương ứng với **một ảnh**.

Các cột bao gồm:

- x1, y1, z1
- x2, y2, z2
- ...
- x467, y467, z467
- Label

Ví dụ:

| x0  | y0  | z0  | ... | x467 | y467 | z467 | Label |
| --- | --- | --- | --- | ---- | ---- | ---- | ----- |

---

# Đặc điểm

- Tự động phát hiện khuôn mặt bằng MediaPipe Face Mesh.
- Trích xuất đầy đủ 468 điểm đặc trưng.
- Lưu dữ liệu dưới dạng CSV để thuận tiện cho việc huấn luyện mô hình.
- Giữ nguyên nhãn cảm xúc của bộ dữ liệu gốc.
- Bỏ qua những ảnh không phát hiện được khuôn mặt.
