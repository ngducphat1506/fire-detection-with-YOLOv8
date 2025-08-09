## Phát hiện lửa với YOLOv8 (Streamlit)

Ứng dụng Streamlit chạy mô hình YOLOv8 đã huấn luyện để phát hiện lửa trên ảnh, video, hoặc ảnh chụp từ webcam.

### Tính năng
- Tải ảnh từ tệp hoặc nhập URL ảnh (http/https)
- Tải video để xử lý và tải về video kết quả
- Chụp ảnh trực tiếp từ webcam
- Điều chỉnh tham số suy luận: Confidence và IoU
- Sử dụng trọng số mô hình cố định

### Yêu cầu hệ thống
- Python 3.9+ (khuyến nghị)
- Nếu dùng GPU: cài PyTorch phiên bản hỗ trợ CUDA phù hợp

### Cài đặt
```bash
pip install -r requirements.txt
```

### Chạy ứng dụng
```bash
streamlit run fire_detection/app.py
```
Sau khi chạy, truy cập đường dẫn hiển thị trên terminal (thường là `http://localhost:8501`).

### Trọng số mô hình (đã cố định)
Ứng dụng sử dụng một đường dẫn trọng số cố định trong `fire_detection/app.py`:
```
/workspace/ai_intern/PHAT/fire_detection/runs/detect/train2/weights/best.pt
```
- Nếu bạn muốn thay đổi, sửa hằng `WEIGHTS_FIXED_PATH` trong `fire_detection/app.py`.

### Cách sử dụng nhanh
1) Vào mục "Nguồn dữ liệu" chọn loại nguồn:
   - Ảnh: Chọn "Tải tệp" để upload ảnh hoặc "URL" để nhập đường dẫn ảnh.
   - Video: Upload tệp video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
   - Webcam (chụp ảnh): Chụp ảnh trực tiếp trên trình duyệt.
2) Điều chỉnh tham số ở thanh bên nếu cần:
   - Confidence, IoU, và thiết bị (auto/cpu/cuda)
3) Nhấn nút chạy tương ứng. Kết quả hiển thị ở cột bên phải và có nút tải về.

### Giải thích tham số
- Confidence: Mức độ chắc chắn tối thiểu để giữ lại một dự đoán.
  - Cao hơn → ít cảnh báo giả hơn nhưng dễ bỏ sót.
  - Gợi ý: 0.25–0.5 (nhạy) hoặc 0.5–0.7 (chặt).
- IoU: Ngưỡng chồng lắp cho bước NMS để loại box trùng lặp.
  - Thấp hơn → loại trùng lặp mạnh tay (ít box hơn). Cao hơn → nương tay (nhiều box gần nhau).
  - Gợi ý: 0.4–0.6 (mặc định 0.45).

### Lưu ý & khắc phục sự cố
- Không tìm thấy trọng số: Kiểm tra file tồn tại đúng tại đường dẫn cố định hoặc chỉnh `WEIGHTS_FIXED_PATH`.
- GPU không khả dụng: Chọn thiết bị `cpu` trong thanh bên.
- Video không phát/ghi: Thử dùng định dạng `.mp4`. Trên một số hệ thống có thể cần codec/phụ thuộc hệ thống.
- Ảnh URL không tải được: Kiểm tra URL hợp lệ và có thể truy cập, hoặc thử tải ảnh về máy rồi upload.

### Cấu trúc thư mục chính
```
PHAT/
├─ fire_detection/
│  ├─ app.py                # Ứng dụng Streamlit
│  └─ runs/detect/train2/weights/best.pt  # Trọng số mô hình (đường dẫn ví dụ)
└─ requirements.txt
```

### Bản quyền
Dùng cho mục đích học tập/thử nghiệm. Vui lòng kiểm tra giấy phép của tập trọng số và dữ liệu bạn sử dụng. 