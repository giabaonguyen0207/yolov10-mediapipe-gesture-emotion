import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm

DATASET_DIR = "Dts/fane_data"          
OUTPUT_CSV = "Output/face_landmarks.csv"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

mp_face_mesh = mp.solutions.face_mesh

NUM_LANDMARKS = 468


def danhsotoado():
    header = []
    for i in range(NUM_LANDMARKS):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    header.append("label")
    return header


def trich_xuat_toa_do_tu_anh(face_mesh, image_path):
    """Trả về list 1404 giá trị (x,y,z * 468) hoặc None nếu không detect được mặt."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # CV2 đọc ảnh theo Blue-Green-Red, nhưng mà Mediapipe đọc theo Red-Green-Blue dẫn đến quá trình trích xuất sai lệch.
    # Do vậy cần phải convert ảnh từ BGR sang RGB để trích xuất đúng.
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks: # Không tìm được mặt thì return None
        return None

    face_landmarks = results.multi_face_landmarks[0] # Nếu detect từ 2 mặt trở lên thì lấy cái mặt được detect đầu tiên để gắn toạ độ

    row = []
    for lm in face_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z]) # Lấy toạ độ (x,y,z) của 468 điểm

    return row

def load_label_id(class_id_path):
    """
    Đọc file Class_ID.txt dạng:
        0: angry
        1: disgust
        2: fear
        ...
    Trả về dict: {"angry": 0, "disgust": 1, "fear": 2, ...}
    """
    name_to_id = {}
    with open(class_id_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()          # bỏ khoảng trắng/xuống dòng thừa
            if not line:
                continue                 # bỏ qua dòng trống
            id_str, name = line.split(":", 1)   # tách theo dấu ":"
            class_id = int(id_str.strip())      # "0" → 0 (số nguyên)
            class_name = name.strip()           # " angry" → "angry"
            name_to_id[class_name] = class_id

    return name_to_id

def main():
    header = danhsotoado()
    class_mapping = load_label_id("Class_ID.txt")
    rows_written = 0
    rows_skipped = 0
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,      # bắt buộc True vì xử lý ảnh tĩnh, không phải xử lí theo thời gian thực
        max_num_faces=1,
        refine_landmarks=True,       # bật thêm landmark quanh mắt/môi cho chính xác hơn
        min_detection_confidence=0.5,
    ) as face_mesh, open(OUTPUT_CSV, "w", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(header) # Tạo hàng tiêu đề cho file CSV

        labels = sorted(
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        )
        print(f"Tìm thấy {len(labels)} nhãn: {labels}")

        for label in labels:
            label_dir = os.path.join(DATASET_DIR, label)
            image_files = [
                fn for fn in os.listdir(label_dir)
                if fn.lower().endswith(IMAGE_EXTENSIONS)
            ]

            for fn in tqdm(image_files, desc=f"Đang xử lý '{label}'"):
                image_path = os.path.join(label_dir, fn)
                landmarks = trich_xuat_toa_do_tu_anh(face_mesh, image_path)

                if landmarks is None:
                    rows_skipped += 1
                    continue
                label_id = class_mapping.get(label, -1)  # mặc định -1 nếu không có trong mapping
                writer.writerow(landmarks + [label_id])
                rows_written += 1

    print(f"\nXong! Ghi được {rows_written} dòng vào '{OUTPUT_CSV}'.")
    print(f"Bỏ qua {rows_skipped} ảnh (không detect được mặt).")


if __name__ == "__main__":
    main()