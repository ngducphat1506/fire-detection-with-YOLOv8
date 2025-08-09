import os
import io
from tempfile import NamedTemporaryFile
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import requests

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None


st.set_page_config(page_title="Phát hiện lửa - YOLOv8", page_icon="🔥", layout="wide")
st.title("🔥 Phát hiện lửa (YOLOv8)")
st.caption("Tải ảnh/video hoặc chụp ảnh để chạy mô hình đã huấn luyện.")

# -------------------- Sidebar --------------------
st.sidebar.header("Cấu hình mô hình")

WEIGHTS_FIXED_PATH = "/workspace/ai_intern/PHAT/fire_detection/runs/detect/train2/weights/best.pt"
st.sidebar.info(f"Trọng số cố định: {WEIGHTS_FIXED_PATH}")

confidence = st.sidebar.slider("Độ tin cậy (confidence)", 0.0, 1.0, 0.5, 0.01)
iou = st.sidebar.slider("Ngưỡng IoU", 0.0, 1.0, 0.45, 0.01)

device = st.sidebar.selectbox("Thiết bị", ["auto", "cpu", "cuda"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("Nguồn dữ liệu")
source_type = st.sidebar.radio(
    "Chọn nguồn",
    ["Ảnh", "Video", "Webcam (chụp ảnh)"],
    index=0,
)

# -------------------- Helpers --------------------
@st.cache_resource(show_spinner=False)
def load_model(weights_file_path: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO chưa được cài đặt. Hãy cài đặt bằng: pip install ultralytics")
    return YOLO(weights_file_path)


def ensure_uploaded_weights_to_path(uploaded) -> Optional[str]:
    if uploaded is None:
        return None
    suffix = os.path.splitext(uploaded.name)[1] or ".pt"
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def annotate_image_array(image_bgr: np.ndarray, model) -> np.ndarray:
    results = model.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou,
        device=device,
        verbose=False,
    )
    annotated = results[0].plot()  # BGR ndarray
    return annotated


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_rgb_img(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# -------------------- Resolve weights --------------------
resolved_weights_path: Optional[str] = WEIGHTS_FIXED_PATH

model = None
if resolved_weights_path and os.path.exists(resolved_weights_path):
    with st.spinner("Đang tải mô hình..."):
        try:
            model = load_model(resolved_weights_path)
            st.success(f"Đã tải mô hình từ: {resolved_weights_path}")
        except Exception as e:
            st.error(f"Không thể tải mô hình: {e}")
else:
    st.error(
        f"Không tìm thấy tệp trọng số tại: {WEIGHTS_FIXED_PATH}"
    )

# -------------------- Main content --------------------
col_left, col_right = st.columns([1, 1])

if source_type == "Ảnh":
    with col_left:
        image_source_mode = st.radio("Nguồn ảnh", ["Tải tệp", "URL"], index=0)
        img = None
        image_url = ""
        if image_source_mode == "Tải tệp":
            img_file = st.file_uploader(
                "Tải ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False
            )
            if img_file is not None:
                img = Image.open(img_file)
        else:
            image_url = st.text_input("Nhập URL ảnh (http/https)")
            if image_url:
                try:
                    img = load_image_from_url(image_url)
                except Exception as e:
                    st.error(f"Không tải được ảnh từ URL: {e}")
        run_btn = st.button("Chạy nhận dạng")

    if img is not None:
        col_left.image(img, caption="Ảnh gốc", use_container_width=True)

    if run_btn and img is not None and model is not None:
        bgr = pil_to_bgr(img)
        with st.spinner("Đang suy luận..."):
            annotated_bgr = annotate_image_array(bgr, model)
        annotated_img = bgr_to_rgb_img(annotated_bgr)

        with col_right:
            st.image(annotated_img, caption="Kết quả", use_container_width=True)

            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Tải ảnh kết quả",
                data=byte_im,
                file_name="prediction.png",
                mime="image/png",
            )

elif source_type == "Video":
    with col_left:
        vid_file = st.file_uploader(
            "Tải video", type=["mp4", "mov", "avi", "mkv", "webm"], accept_multiple_files=False
        )
        run_btn = st.button("Xử lý video")

    if run_btn and vid_file is not None and model is not None:
        # Lưu video nguồn ra file tạm
        src_tmp = NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1])
        src_tmp.write(vid_file.read())
        src_tmp.flush()
        src_tmp.close()

        cap = cv2.VideoCapture(src_tmp.name)
        if not cap.isOpened():
            st.error("Không thể mở video.")
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_tmp = NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tmp.name
            out_tmp.close()

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0)
            status = st.empty()

            frame_idx = 0
            with st.spinner("Đang xử lý video..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated = annotate_image_array(frame, model)
                    writer.write(annotated)

                    frame_idx += 1
                    if total > 0:
                        progress.progress(min(frame_idx / total, 1.0))
                    if frame_idx % max(int(fps // 2), 1) == 0:
                        status.image(
                            bgr_to_rgb_img(annotated), caption=f"Xử lý khung {frame_idx}/{total}", use_container_width=True
                        )

            cap.release()
            writer.release()

            with col_right:
                st.success("Hoàn tất xử lý.")
                st.video(out_path)
                with open(out_path, "rb") as f:
                    st.download_button(
                        label="Tải video kết quả",
                        data=f.read(),
                        file_name="prediction.mp4",
                        mime="video/mp4",
                    )

elif source_type == "Webcam (chụp ảnh)":
    with col_left:
        shot = st.camera_input("Chụp ảnh từ webcam")
        run_btn = st.button("Nhận dạng ảnh chụp")

    if run_btn and shot is not None and model is not None:
        img = Image.open(shot)
        bgr = pil_to_bgr(img)
        with st.spinner("Đang suy luận..."):
            annotated_bgr = annotate_image_array(bgr, model)
        annotated_img = bgr_to_rgb_img(annotated_bgr)

        with col_right:
            st.image(annotated_img, caption="Kết quả", use_container_width=True)
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            st.download_button(
                label="Tải ảnh kết quả",
                data=buf.getvalue(),
                file_name="prediction.png",
                mime="image/png",
            )

