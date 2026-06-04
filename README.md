# ĐỒ ÁN MÔN HỌC: NHẬN DẠNG ĐƯỜNG VỆ TINH (DEEPGLOBE)
**Sinh viên:** Phạm Minh Quân
**Đồ án:** Nhận dạng đường bộ từ ảnh vệ tinh (Satellite Imagery Road Extraction bằng Mạng Nơ-ron Tích Chập)

---

## 1. LIÊN KẾT TRUY CẬP ỨNG DỤNG (LIVE DEMO)
Toàn bộ mô hình đã được triển khai thành công thành một ứng dụng Web hoàn chỉnh trên Streamlit Cloud. Kính mời Thầy/Cô và các bạn xem trực tiếp tại đây:

**[Live Web App - DeepGlobe Road Segmentation](https://road-segmentation-cnn-7rdsmsaq3uchna355jbjcr.streamlit.app/)**

### Các tính năng chính của Web App:
- **Gallery Demo:** Trực quan hóa kết quả phân loại và dự đoán đường của mô hình (tốt, trung bình, xấu) từ bộ ảnh vệ tinh mẫu.
- **Upload Ảnh:** Cho phép tải lên hình ảnh vệ tinh từ máy tính cá nhân để mô hình phân tích tự động.
- **Cắt Bản Đồ (Map Cropping):** Tích hợp bản đồ vệ tinh trực tuyến (Esri Satellite, OpenStreetMap), cho phép tự do vẽ khung chọn một khu vực bất kỳ trên thế giới, ứng dụng sẽ cắt vùng ảnh vệ tinh thực tế và dùng AI để trích xuất đường xá bên trong đó.

---

## 2. KỸ THUẬT VÀ PHƯƠNG PHÁP SỬ DỤNG
Dự án áp dụng nhiều kỹ thuật và phương pháp Xử lý Ảnh và Học Sâu tiên tiến để đảm bảo tốc độ và độ chính xác:

### Xử lý Dữ liệu và Tối ưu bộ nhớ
- **tf.data.Dataset API:** Xây dựng Data Pipeline song song (parallel computing), tự động map dữ liệu thành các batch nhằm loại bỏ hoàn toàn giới hạn tràn bộ nhớ (Out-Of-Memory) so với việc load mảng Numpy truyền thống.
- **Tối ưu RAM Hệ thống:** Sử dụng lệnh `tf.keras.backend.clear_session()` và thư viện `gc` để thu gom rác hệ thống (garbage collection) ngay sau mỗi vòng lặp huấn luyện mô hình.
- **Tiền xử lý ảnh:** Áp dụng kỹ thuật chuyển đổi sang ảnh xám (Grayscale), chuẩn hóa ma trận điểm ảnh (Normalization) về khoảng [0, 1].

### Kiến trúc Mô hình (CNN + U-Net)
Dự án sử dụng 2 kiến trúc CNN được tinh chỉnh kết hợp với cơ chế U-Net Decoder để giải quyết bài toán phân đoạn nhị phân (Binary Segmentation):

1. **TinyVGG-UNet:**
   - Dựa trên kiến trúc TinyVGG.
   - Phần mã hóa (Encoder): [Conv -> ReLU -> Conv -> ReLU -> MaxPool] x 3.
   - Có khả năng trích xuất đặc trưng sâu và hội tụ nhanh, cho ra mặt nạ đường xá rất sắc nét.
   
2. **LeNet-UNet:**
   - Dựa trên kiến trúc tối giản LeNet.
   - Phần mã hóa (Encoder): Conv(32) -> Pool -> Conv(64) -> Pool -> Dropout.
   - Ưu điểm: Khối lượng nhẹ, tốc độ dự đoán tính bằng mili-giây, cực kì phù hợp khi deploy trên server không có GPU.

### Huấn luyện và Đánh giá (Training & Evaluation)
- **Hàm mất mát (Loss Function):** Kết hợp Binary Crossentropy và Dice Loss để giải quyết vấn đề mất cân bằng lớp (đường xá chiếm diện tích rất nhỏ so với hậu cảnh).
- **Siêu tham số (Hyperparameters):** Batch Size = 32, Image Size = 128x128, thuật toán tối ưu Adam với Learning Rate = 0.001.
- **Callbacks thông minh:** 
   - `ReduceLROnPlateau`: Tự động giảm tốc độ học (learning rate) nếu độ mất mát không còn cải thiện.
   - `EarlyStopping`: Dừng quá trình huấn luyện sớm nếu độ chính xác trên tập validation ngừng tăng, chống hiện tượng học vẹt (overfitting).
- **Đánh giá (Metrics):** Binary Accuracy, Dice Coefficient, và Intersection over Union (IoU).

### Triển khai (Deployment)
- Tách biệt UI: Mã nguồn giao diện được tách khỏi file Notebook thành `app.py`.
- Tải nhanh dữ liệu: Lưu trạng thái của mô hình và các mảng hình ảnh thư viện dưới dạng `.pkl` và chuỗi `Base64` để đọc ngay lập tức, tiết kiệm tối đa thời gian khởi tạo trên máy chủ Cloud.
- Tương tác HTML/JS tĩnh: Loại bỏ các tính năng xung đột JSON khi render bản đồ trực tuyến (Folium) trên server Linux.

---

## 3. HƯỚNG DẪN CÀI ĐẶT CỤC BỘ (LOCAL)

### Cài đặt thư viện
Cài đặt các gói thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng Web (Streamlit)
Mở terminal/command prompt và chạy lệnh sau để khởi động giao diện trên trình duyệt web:
```bash
streamlit run app.py
```

### Khởi chạy quá trình Huấn Luyện (Training)
Nếu muốn huấn luyện lại từ đầu toàn bộ mô hình:
- Mở file Jupyter Notebook `DOG_PhamMinhQuan_KT02_v5_fast.ipynb`.
- Chạy toàn bộ các dòng code (Run all). Dữ liệu vệ tinh sẽ tự động được tải xuống thông qua thư viện `kagglehub`.

---
*Phát triển bởi Phạm Minh Quân*
