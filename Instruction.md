# Hướng Dẫn Cài Đặt và Sử Dụng Hệ Thống

## 1. Cài đặt môi trường
* **Yêu cầu hệ thống:** Máy tính cần cài đặt sẵn Python.
* **Tải Source Code:** Clone toàn bộ dự án từ nhánh `main` trên Github về máy tính.
* **Cài đặt thư viện:** Mở Terminal tại thư mục chứa source code và chạy lệnh sau để tải các gói công cụ:
  ```bash
  pip install -r requirements.txt
  ```

## 2. Khởi chạy ứng dụng
* Tại Terminal, gõ lệnh sau để khởi động server Streamlit:
  ```bash
  streamlit run app.py
  ```
* Ứng dụng sẽ khởi chạy và tự động mở giao diện web trên trình duyệt tại địa chỉ mặc định: `http://localhost:8501`.

## 3. Thao tác trên giao diện
* **Bật luồng nhận diện:** Tại Bảng Điều Khiển (Sidebar) bên trái, đánh dấu vào ô **"🟢 Bật Camera"** và cấp quyền cho trình duyệt truy cập Webcam.
* **Tinh chỉnh hiệu năng:** Sử dụng thanh trượt để điều chỉnh **Ngưỡng nhạy (Confidence)**.
* **Các tiện ích mở rộng:** Sử dụng các nút bấm chức năng để:
  * **"📸 Chụp ảnh khoảnh khắc"**: Lưu ngay khung hình hiện tại cùng kết quả phân tích.
  * **"🖼️ Lịch sử chụp"**: Mở cửa sổ quản lý, xem chi tiết hoặc xóa ảnh đã chụp.
  * **"📖 Xem danh sách Nhãn"**: Mở bảng tra cứu chéo 70 tổ hợp Thái độ dựa trên Cảm xúc và Cử chỉ tay.

## 4. Kết quả nhận diện
* **Cảm xúc khuôn mặt (Emotion):** Nhận diện 5 trạng thái cơ bản (angry, happy, neutral, sad, surprise).
* **Cử chỉ tay (Gesture):** Phát hiện và phân loại 14 loại cử chỉ tay khác nhau (ok, like, peace, hand_heart, stop,...).
* **Thái độ (Attitude):** Thực hiện tra cứu chéo giữa Cảm xúc và Cử chỉ tay để đưa ra đánh giá tổng quan (gồm 70 tổ hợp). Nếu cảm xúc và cử chỉ mâu thuẫn nhau, hệ thống sẽ trả về kết quả là Undefined.
