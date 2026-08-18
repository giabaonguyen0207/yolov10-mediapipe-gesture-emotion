# GIT WORKFLOW

Tài liệu này quy định quy trình quản lý mã nguồn, chia nhánh và gửi mã nguồn cho dự án Real-time Dual-Task Gesture & Emotion Recognition.   
Tất cả 5 thành viên trong nhóm bắt buộc tuân thủ quy trình này để tránh xung đột code (conflict) và giữ cho nhánh chính (main) luôn ổn định.

---

## 1. Cấu Trúc Nhánh (Branching Strategy)

* **main**: Nhánh chính của dự án. CẤM PUSH TRỰC TIẾP LÊN NHÁNH NÀY. Code trên main phải luôn là code sạch, chạy được và đã qua kiểm thử (review) bởi Leader.
* **feature/<ten-tinh-nang>**: Nhánh cá nhân để mỗi thành viên phát triển nhiệm vụ của mình.

### Quy ước đặt tên nhánh cho từng thành viên:
* TV 1 (Leader): feature/setup-architecture
* TV 2 (Data): feature/data-landmarks-prep
* TV 3 (Hand): feature/train-yolo-hand
* TV 4 (Emotion): feature/train-emotion-ml
* TV 5 (UI/Integration): feature/streamlit-ui-integration

---

## 2. Quy Trình Làm Việc Hàng Ngày (Step-by-Step Workflow)

### Bước 1: Khởi tạo dự án về máy cá nhân (Chỉ làm 1 lần đầu)
Mở Terminal/PowerShell và clone repo về máy:  
```powershell
git clone https://github.com/giabaonguyen0207/yolov10-mediapipe-gesture-emotion.git  
cd yolov10-mediapipe-gesture-emotion
```

---

### Bước 2: Bắt đầu làm tính năng mới (Bắt đầu mỗi ngày làm việc)
Trước khi làm việc, luôn chuyển sang nhánh main và kéo code mới nhất từ các bạn khác về:  
```powershell
git checkout main  
git pull origin main  
git checkout -b feature/ten-nhanh-cua-ban  
```

---

### Bước 3: Lưu tiến độ làm việc (Commit Code)
Sau khi viết xong 1 đoạn code hoặc 1 chức năng hoạt động tốt, hãy lưu lại commit:  
``` powershell
git status  
git add .  
git commit -m "feat: mo ta ngan gọn cong viec da lam"  
```

#### Quy ước viết Commit Message:
* feat: Thêm một tính năng mới (VD: feat: thêm streamlit ui layout)
* fix: Sửa lỗi code (VD: fix: sửa lỗi phép chia cho số không trong chuẩn hóa)
* chore: Việc lặt vặt (VD: chore: update .gitignore)
* refactor: Tối ưu/cấu trúc lại code mà không đổi tính năng

---

### Bước 4: Đẩy code lên GitHub (Push Code)
Khi muốn lưu code lên GitHub, đẩy nhánh cá nhân lên:  
```powershell
git push -u origin feature/ten-nhanh-cua-ban
```
(Các lần push sau trên cùng nhánh đó chỉ cần gõ: `git push`)  

---

### Bước 5: Tạo Pull Request (PR) & Gộp Code vào main
1. Truy cập vào giao diện Web của Repository trên GitHub.
2. Bấm vào nút màu vàng "Compare & pull request".
3. Đặt tiêu đề PR rõ ràng (VD: [TV3] Fine-tune thanh cong YOLOv10n).
4. Gán Reviewer là Leader (TV1).
5. Leader kiểm tra code: Nếu ổn bấm Approve và Merge Pull Request.

---

## 3. Các Quy Tắc "Vàng" Cần Ghi Nhớ (Strict Rules)

1. KHÔNG PUSH FILE NẶNG LÊN GIT: KHÔNG push các file dataset gốc (.jpg, .png, .zip), KHÔNG push môi trường ảo (venv/), các file trọng số lớn (.pt, .pkl) phải qua sự đồng ý của Leader.
2. XỬ LÝ KHI BỊ XUNG ĐỘT (GIT CONFLICT): Báo ngay cho Leader để cùng giải quyết, tuyệt đối không dùng `git push --force.`
3. LUÔN PULL MỚI TRƯỚC KHI TẠO NHÁNH MỚI: Giúp hạn chế tối đa việc bị xung đột code.
