# ĐỒ ÁN MÔN HỌC: NHẬN DẠNG ĐƯỜNG VỆ TINH (DEEPGLOBE)

**Danh sách nhóm sinh viên thực hiện:**
1. **Phạm Minh Quân** - 2045230079
2. **Dương Diệu Linh** - 2045230055
3. **Trần Thị Bảo Trâm** - 2045230106

**Đồ án:** Nhận dạng đường bộ từ ảnh vệ tinh (Satellite Imagery Road Extraction bằng Mạng Nơ-ron Tích Chập)

---

## 1. LIÊN KẾT TRUY CẬP ỨNG DỤNG (LIVE DEMO)
Toàn bộ mô hình đã được triển khai thành công thành một ứng dụng Web hoàn chỉnh trên Streamlit Cloud. Kính mời Thầy/Cô xem trực tiếp tại đây:

**[Live Web App - DeepGlobe Road Segmentation](https://road-segmentation-cnn-7rdsmsaq3uchna355jbjcr.streamlit.app/)**

### Các tính năng chính của Web App:
- **Gallery Demo:** Trực quan hóa kết quả phân loại và dự đoán đường của mô hình (tốt, trung bình, xấu) từ bộ ảnh vệ tinh mẫu.
- **Upload Ảnh:** Cho phép tải lên hình ảnh vệ tinh từ máy tính cá nhân để mô hình phân tích tự động.
- **Cắt Bản Đồ (Map Cropping):** Tích hợp bản đồ vệ tinh trực tuyến (Esri Satellite, OpenStreetMap), cho phép tự do vẽ khung chọn một khu vực bất kỳ trên thế giới, ứng dụng sẽ cắt vùng ảnh vệ tinh thực tế và dùng AI để trích xuất đường xá bên trong đó.

---

## 2. KỸ THUẬT VÀ SIÊU THAM SỐ (HYPERPARAMETERS)
Dự án áp dụng nhiều kỹ thuật Học Sâu tiên tiến để đảm bảo tốc độ và tối đa hóa độ chính xác cho bài toán phân đoạn nhị phân (Binary Segmentation):

### Siêu tham số thuật toán (Đúng với bản V5)
- **Kích thước ảnh đầu vào (IMG_SIZE):** 128x128. Cân bằng hoàn hảo giữa tốc độ và khả năng hiển thị.
- **Kích thước Batch (BATCH_SIZE):** 128. Giúp tối ưu hóa việc sử dụng RAM và GPU.
- **Số vòng huấn luyện (EPOCHS):** 50. Thời gian học đủ dài để mô hình hội tụ.
- **Thuật toán Tối ưu (Optimizer):** Adam với Learning Rate (LR) ban đầu = 0.001.

### Các Kỹ thuật Tối ưu Mô hình Khác
- **Mixed Precision Training:** Khởi chạy chính sách `mixed_float16` giúp tăng tốc độ huấn luyện lên đáng kể mà không suy giảm độ chính xác.
- **Mask Dilation (MASK_DILATE_ITER = 3):** Làm dày (dilate) nhãn đường nhỏ trước khi train để đảm bảo đường không bị biến mất khi thu nhỏ kích thước (resize) ảnh về 128x128.
- **Weighted BCE (POS_WEIGHT = 15.0):** Trọng số phạt nặng hơn 15 lần đối với các lỗi bỏ sót đường (false negative) nhằm chống lại hiện tượng mất cân bằng dữ liệu (đường xá chiếm diện tích cực nhỏ so với background).
- **Callbacks thông minh:** 
   - `EarlyStopping (PATIENCE = 15)`: Dừng quá trình huấn luyện sớm nếu độ chính xác trên tập validation ngừng tăng sau 15 epochs, chống hiện tượng học vẹt (overfitting).
   - `ReduceLROnPlateau`: Tự động giảm tốc độ học nếu độ mất mát (loss) không còn cải thiện.

### Xử lý Dữ liệu và Tối ưu bộ nhớ
- **tf.data.Dataset API:** Xây dựng Data Pipeline song song (parallel computing), tự động map dữ liệu thành các batch nhằm loại bỏ hoàn toàn giới hạn tràn bộ nhớ (Out-Of-Memory) so với việc load mảng Numpy truyền thống.
- **Tối ưu RAM Hệ thống:** Sử dụng lệnh `tf.keras.backend.clear_session()` và thư viện `gc` để thu gom rác hệ thống (garbage collection) ngay sau mỗi vòng lặp huấn luyện mô hình.

### Kiến trúc Mô hình (CNN + U-Net)
1. **TinyVGG-UNet:**
   - Dựa trên kiến trúc TinyVGG.
   - Trích xuất đặc trưng sâu, hội tụ cực nhanh, xuất ra mặt nạ đường xá rất sắc nét.
2. **LeNet-UNet:**
   - Dựa trên kiến trúc tối giản LeNet.
   - Tốc độ suy luận (inference) tính bằng mili-giây, tiêu tốn rất ít tài nguyên, phù hợp cho Web App Cloud.

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
*Phát triển bởi nhóm sinh viên Phạm Minh Quân, Dương Diệu Linh, Trần Thị Bảo Trâm.*
