# 🛰️ Mạng Nơ-ron Tích Chập (CNN) - Nhận Dạng Đường Vệ Tinh (DeepGlobe)
**Sinh viên:** Phạm Minh Quân
**Đồ án môn học:** Nhận dạng đường bộ từ ảnh vệ tinh (Satellite Imagery Road Extraction)

![DeepGlobe Road Extraction](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11227-020-03525-5/MediaObjects/11227_2020_3525_Fig1_HTML.png) *(Ảnh minh họa)*

---

## 🚀 Live Demo (Streamlit Cloud)
Toàn bộ mô hình đã được triển khai (deploy) thành công thành một ứng dụng Web tương tác đầy đủ tính năng. Thầy/Cô và các bạn có thể xem trực tiếp tại đây:

👉 **[Live Web App - DeepGlobe Road Segmentation](https://road-segmentation-cnn-7rdsmsaq3uchna355jbjcr.streamlit.app/)**

### 🌟 Các tính năng chính của Web App:
- **Gallery Demo:** Trực quan hóa kết quả phân loại và dự đoán đường của mô hình (tốt, trung bình, xấu) từ bộ ảnh vệ tinh chuẩn.
- **Upload Ảnh:** Cho phép tải lên bất kỳ hình ảnh vệ tinh nào từ máy tính để mô hình phân tích tự động.
- **Cắt Bản Đồ (Map Cropping):** Tích hợp Google Maps / OpenStreetMap, cho phép người dùng tự vẽ khung chọn một khu vực bất kỳ trên bản đồ thế giới, ứng dụng sẽ cắt vùng ảnh vệ tinh thực tế và dùng AI để tìm ra đường xá bên trong đó.

---

## 🧠 Kiến Trúc Mô Hình (Model Architectures)
Dự án sử dụng **2 kiến trúc CNN** được tinh chỉnh kết hợp với cơ chế **U-Net** Decoder để giải quyết bài toán phân đoạn nhị phân (Binary Segmentation):

1. **TinyVGG-UNet:**
   - Dựa trên kiến trúc `TinyVGG`.
   - Encoder: `[Conv -> ReLU -> Conv -> ReLU -> MaxPool] x 3`.
   - Có khả năng trích xuất đặc trưng sâu và hội tụ nhanh, cho ra mặt nạ đường xá rất sắc nét.
   
2. **LeNet-UNet:**
   - Dựa trên kiến trúc tối giản `LeNet`.
   - Encoder: `Conv(32) -> Pool -> Conv(64) -> Pool -> Dropout`.
   - Ưu điểm: Siêu nhẹ, tốc độ suy luận (inference) nhanh tính bằng mili-giây, tiêu tốn rất ít RAM.

**Thông số kỹ thuật:**
- Đầu vào (Input): Ảnh màu RGB kích thước `128 x 128`.
- Đầu ra (Output): Ảnh Binary Mask (1 kênh) dự đoán xác suất điểm ảnh là đường xá.
- Hàm mất mát (Loss): `Binary Crossentropy` + `Dice Loss`.
- Đánh giá (Metrics): `Binary Accuracy`, `Dice Coefficient`, `IoU`.

---

## ⚙️ Cải Tiến Kỹ Thuật Nổi Bật (Version 5)
Mã nguồn (`DOG_PhamMinhQuan_KT02_v5_fast.ipynb`) đã được tối ưu hóa toàn diện để xử lý bộ dữ liệu ảnh vệ tinh khổng lồ:

- **Tối ưu RAM:** Loại bỏ cách nạp toàn bộ ảnh vào RAM. Sử dụng `tf.data.Dataset` API để xây dựng data pipeline tải dữ liệu song song và theo từng batch.
- **Dọn Rác Tự Động:** Kết hợp `tf.keras.backend.clear_session()` và thư viện `gc` để thu hồi bộ nhớ ngay sau khi train xong từng mô hình, ngăn chặn hoàn toàn lỗi văng bộ nhớ (Out-Of-Memory) trên các máy cấu hình thấp.
- **Auto-Learning Rate:** Tích hợp cơ chế `ReduceLROnPlateau` (giảm tốc độ học khi loss bị kẹt) và `EarlyStopping` (chống quá khớp).
- **Fast Deploy:** Mô hình và toàn bộ biến trạng thái (state) được đóng gói vào file nén `.pkl` và `.keras` độc lập, giúp Web App load dữ liệu chỉ trong chưa tới 3 giây mà không cần train lại.

---

## 💻 Hướng Dẫn Chạy Cục Bộ (Local)

### 1. Cài đặt thư viện
Bạn cần cài đặt các thư viện cần thiết trước khi chạy:
```bash
pip install -r requirements.txt
```

### 2. Chạy Web App (Streamlit)
Bạn có thể mở giao diện bằng dòng lệnh sau:
```bash
streamlit run app.py
```

### 3. Huấn Luyện Lại (Training)
Nếu muốn huấn luyện lại (train) mô hình từ đầu hoặc xem luồng xử lý chi tiết, hãy mở file Jupyter Notebook:
- `DOG_PhamMinhQuan_KT02_v5_fast.ipynb`
- Chạy lần lượt các cell từ trên xuống dưới. Dữ liệu sẽ tự động được tải về qua `kagglehub`.

---
*Được phát triển bởi Phạm Minh Quân.*
