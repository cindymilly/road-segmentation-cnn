# ĐỒ ÁN MÔN HỌC: NHẬN DẠNG ĐƯỜNG TỪ ẢNH VỆ TINH (DEEPGLOBE)

**Danh sách nhóm sinh viên thực hiện:**
1. Phạm Minh Quân - 2045230079
2. Dương Diệu Linh - 2045230055
3. Trần Thị Bảo Trâm - 2045230106

---

## 1. LINK DEMO GIAO DIỆN HOÀN CHỈNH (STREAMLIT CLOUD)
Thầy vui lòng bấm vào đường link dưới đây để xem trực tiếp giao diện Web App đã được Deploy thành công lên Cloud:
**[Nhấn vào đây để xem Giao diện Web App](https://road-segmentation-cnn-7rdsmsaq3uchna355jbjcr.streamlit.app/)**

*(Web App bao gồm đầy đủ tính năng: Xem thư viện ảnh mẫu, Tải ảnh từ máy tính, và Đặc biệt: Tính năng vẽ khung cắt bản đồ thực tế với Google Maps/OpenStreetMap)*

---

## 2. CÁC CẢI TIẾN VƯỢT BẬC TRONG PHIÊN BẢN (SO VỚI V4):
Nhóm đã thực hiện tối ưu hóa toàn diện để giải quyết hoàn toàn lỗi tràn RAM (OOM) và tăng tốc độ xử lý:

### Tối ưu hóa Quản lý Bộ nhớ & Dữ liệu:
- Chuyển đổi hoàn toàn từ việc load dữ liệu mảng numpy sang sử dụng `tf.data.Dataset`, giúp load dữ liệu theo từng batch, tiết kiệm RAM tuyệt đối.
- Tự động dọn dẹp bộ nhớ Keras (`tf.keras.backend.clear_session()`) và thu gom rác hệ thống (`gc.collect()`) sau mỗi lần huấn luyện mô hình.
- Nén bộ biến trạng thái vào file `nb_state_vars.pkl` để Deploy lên Cloud mà không cần train lại.

### Tinh chỉnh Siêu tham số (Hyperparameters):
- **[THAY ĐỔI 1] Kích thước ảnh (IMG_SIZE) = 128:** Giúp cân bằng hoàn hảo giữa tốc độ huấn luyện và khả năng giữ lại các chi tiết đường nhỏ.
- **[THAY ĐỔI 2] Tăng EPOCHS = 50:** Học đủ lâu, kết hợp EarlyStopping sẽ dừng sớm nếu hội tụ.
- **[THAY ĐỔI 3] Kích thước Batch (BATCH_SIZE) = 128:** Tránh OOM khi GPU xử lý lượng ảnh lớn.
- **[THAY ĐỔI 4] Tốc độ học (LR) = 0.001:** Thêm warmup và aggressive decay. Thiết lập Patience = 15 (Dừng sớm nếu không cải thiện sau 15 epochs).
- **[THAY ĐỔI 5] Mask Dilation (MASK_DILATE_ITER = 3):** Làm "dày" thêm đường nhỏ trước khi resize. Dilate mask trước khi train giúp model học được cả đường mỏng (không bị biến mất khi thu nhỏ kích thước).
- **[THAY ĐỔI 6] Weighted BCE (POS_WEIGHT = 15.0):** Chống class imbalance. Đường road chiếm phần trăm pixel rất ít so với background. Phạt lỗi false negative (bỏ sót đường) nặng gấp 15 lần để tăng độ bao phủ.
- Mức binarize mask (THRESH) giữ nguyên ở mức 128.

### Kiến trúc Mô hình (CNN + U-Net):
- **TinyVGG-UNet:** Nhẹ, hội tụ cực nhanh, trích xuất đặc trưng tốt.
- **LeNet-UNet:** Phiên bản tối giản, tốc độ suy luận (inference) tính bằng mili-giây.

### Giao diện Đồ họa (Streamlit):
- Tách rời mã nguồn UI ra file `app.py` thuần túy.
- Nâng cấp giao diện chuẩn Dark Mode sang trọng, hiển thị đầy đủ thông số độ chính xác (Dice, IoU, Accuracy).

---

## 3. HƯỚNG DẪN CHẠY CODE

**Cài đặt thư viện:**
Mở Terminal và chạy lệnh:
pip install -r requirements.txt

**Chạy Web App Cục Bộ:**
Mở Terminal và chạy lệnh:
streamlit run app.py

**Huấn Luyện Lại (Train):**
Mở file Jupyter Notebook `DOG_PhamMinhQuan_KT02_Update.ipynb` và ấn "Run All" để máy tự động tải dữ liệu từ Kaggle và tiến hành train lại toàn bộ 2 model.
