# app.py - Military Drone Detection (Inference Only)
# Requires: best.pt in the same directory or specify MODEL_PATH below

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH = "best.pt"   # Change this if your model lives elsewhere
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Drone Detection",
    page_icon="🚁",
    layout="wide",
)


@st.cache_resource
def load_model(path: str) -> YOLO | None:
    if not os.path.exists(path):
        st.error(f"❌ Model not found at: **{path}**\n\nPlace `best.pt` next to `app.py` or update `MODEL_PATH`.")
        return None
    model = YOLO(path)
    st.success(f"✅ Model loaded from `{path}`")
    return model


def draw_stats_overlay(img: np.ndarray, boxes, inference_ms: float, track_ids=None) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DETECTION STATUS", (20, 38), font, 0.65, (255, 255, 255), 2)

    y = 70
    if len(boxes) > 0:
        color = (0, 255, 80)
        cv2.putText(img, f"Drones detected: {len(boxes)}", (20, y), font, 0.65, color, 2)
        avg_conf = float(boxes.conf.mean())
        cv2.putText(img, f"Avg confidence:  {avg_conf:.1%}", (20, y + 28), font, 0.65, color, 2)
        cv2.putText(img, f"Inference:       {inference_ms:.1f} ms", (20, y + 56), font, 0.65, color, 2)
        if track_ids is not None:
            cv2.putText(img, f"Tracked IDs:     {len(set(track_ids))}", (20, y + 84), font, 0.65, (0, 210, 255), 2)
    else:
        color = (160, 160, 160)
        cv2.putText(img, "Drones: 0  —  CLEAR", (20, y), font, 0.65, color, 2)
        cv2.putText(img, f"Inference: {inference_ms:.1f} ms", (20, y + 28), font, 0.65, color, 2)

    return img


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.title("🚁 Military Drone Detection")
st.markdown("Upload an image or video — detections are run against your trained `best.pt` model.")
st.markdown("---")

st.sidebar.title("⚙️ Settings")
model_path_input = st.sidebar.text_input("Model path", value=MODEL_PATH)
confidence = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.40, 0.05)
st.sidebar.info("Recommended threshold: 0.35 – 0.45")

model = load_model(model_path_input)
if model is None:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📷 Image", "🎥 Video"])


# ── Image tab ─────────────────────────────────────────────────────────────────
with tab_img:
    st.header("Image Detection")
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="img_upload",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")

        with st.spinner("🔍 Running detection…"):
            results = model(image, conf=confidence)
            annotated = results[0].plot()                          # BGR numpy array
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            boxes = results[0].boxes
            ms = results[0].speed["inference"]
            annotated = draw_stats_overlay(annotated, boxes, ms)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detections")
            st.image(annotated, use_container_width=True)

        if len(boxes) > 0:
            st.markdown("---")
            st.subheader("🎯 Detection Details")
            for i, box in enumerate(boxes):
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                st.write(f"**Drone {i+1}:** confidence = {conf_val:.2%}  |  bbox = [{x1}, {y1}, {x2}, {y2}]")
        else:
            st.info("✅ No drones detected in this image.")


# ── Video tab ─────────────────────────────────────────────────────────────────
with tab_vid:
    st.header("Video Detection")
    st.info("ℹ️ If the tab switches after upload, click **Video** again.")

    video_file = st.file_uploader(
        "Upload a video (MP4 / AVI / MOV / MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
    enable_tracking = st.checkbox(
        "Enable object tracking",
        value=True,
        help="Track drones across frames and draw trajectories.",
    )

    if video_file and st.button("▶️ Process Video"):
        # Save upload to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps        = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Fallback codec
        if not out.isOpened():
            output_path = tempfile.mktemp(suffix=".avi")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        progress = st.progress(0)
        status_text = st.empty()

        frame_count      = 0
        unique_ids: set  = set()
        track_history: dict = {}

        with st.spinner("Processing video…"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if enable_tracking:
                    results = model.track(frame, conf=confidence, persist=True, verbose=False)
                else:
                    results = model(frame, conf=confidence, verbose=False)

                annotated_frame = results[0].plot()

                # Draw trajectories when tracking
                if enable_tracking and results[0].boxes.id is not None:
                    boxes_xywh = results[0].boxes.xywh.cpu()
                    ids        = results[0].boxes.id.int().cpu().tolist()

                    for box, tid in zip(boxes_xywh, ids):
                        cx, cy = float(box[0]), float(box[1])
                        track_history.setdefault(tid, []).append((cx, cy))

                        # Keep last 50 points
                        if len(track_history[tid]) > 50:
                            track_history[tid].pop(0)

                        pts = np.array(track_history[tid], dtype=np.int32).reshape(-1, 1, 2)
                        if len(pts) > 1:
                            cv2.polylines(annotated_frame, [pts], False, (0, 0, 0), 5)        # shadow
                            cv2.polylines(annotated_frame, [pts], False, (0, 230, 255), 3)    # trail

                    unique_ids.update(ids)

                out.write(annotated_frame)
                frame_count += 1
                progress.progress(min(frame_count / max(total_frames, 1), 1.0))
                status_text.text(f"Frame {frame_count} / {total_frames}")

        cap.release()
        out.release()
        os.unlink(tfile.name)

        status_text.empty()
        st.success("✅ Video processed!")

        if enable_tracking and unique_ids:
            st.info(f"🎯 Tracked **{len(unique_ids)}** unique drone(s) across {frame_count} frames.")

        st.video(output_path)

        with open(output_path, "rb") as f:
            ext = os.path.splitext(output_path)[1]
            mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
            st.download_button("📥 Download annotated video", f, f"drone_detection{ext}", mime)


if __name__ == "__main__":
    pass