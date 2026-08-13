# 1. Sử dụng đúng phiên bản Python 3.10.11 của nhóm
FROM python:3.10.11-slim

# 2. Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Đặt thư mục làm việc mặc định trong Container
WORKDIR /app

# 4. Copy file cấu hình thư viện vào trước để cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ code của dự án vào Container
COPY . .

# 6. Mở cổng 8501 mặc định của Streamlit
EXPOSE 8501

# 7. Lệnh khởi chạy ứng dụng khi bật Container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]