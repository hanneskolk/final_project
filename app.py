# app.py - Object Detection (Vehicles & Humans) — Inference Only
# Supports TensorRT (.engine) with automatic fallback to .pt

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# ── Configuration ─────────────────────────────────────────────────────────────
PT_MODEL_PATH  = "best.pt"       # Source weights — always required
TRT_MODEL_PATH = "best.engine"   # TensorRT export target / auto-detected
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Object Detection", page_icon="🎯", layout="wide")


# ── TensorRT export ───────────────────────────────────────────────────────────
def export_to_tensorrt(pt_path: str, engine_path: str, imgsz: int = 640) -> bool:
    """Export best.pt → best.engine. Returns True on success."""
    try:
        with st.spinner("⚙️ Exporting to TensorRT — this takes a few minutes on first run…"):
            model = YOLO(pt_path)
            model.export(format="engine", imgsz=imgsz, half=True)  # FP16 for speed

        # Ultralytics writes the .engine next to the .pt file
        auto_path = pt_path.replace(".pt", ".engine")
        if os.path.exists(auto_path) and auto_path != engine_path:
            os.rename(auto_path, engine_path)

        return os.path.exists(engine_path)
    except Exception as e:
        st.error(f"TensorRT export failed: {e}")
        return False


# ── Model loader with TRT → PT fallback ──────────────────────────────────────
@st.cache_resource
def load_model(pt_path: str, trt_path: str, prefer_trt: bool) -> tuple:
    """
    Returns (model, backend_label).
    Tries TensorRT first when prefer_trt=True; falls back to .pt automatically.
    """
    if prefer_trt:
        if os.path.exists(trt_path):
            try:
                model = YOLO(trt_path)
                return model, "TensorRT (.engine)"
            except Exception as e:
                st.warning(f"⚠️ TensorRT load failed ({e}) — falling back to PyTorch.")
        else:
            st.warning(f"⚠️ No `.engine` found at `{trt_path}` — falling back to PyTorch.")

    # PyTorch fallback
    if not os.path.exists(pt_path):
        st.error(f"❌ Model not found at `{pt_path}`.")
        return None, ""

    model = YOLO(pt_path)
    return model, "PyTorch (.pt)"


# ── Overlay helper ────────────────────────────────────────────────────────────
def draw_stats_overlay(img: np.ndarray, boxes, inference_ms: float,
                       backend: str, track_ids=None) -> np.ndarray:
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (340, 175), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DETECTION STATUS", (20, 38), font, 0.65, (255, 255, 255), 2)

    badge_color = (0, 200, 255) if "TensorRT" in backend else (180, 180, 180)
    cv2.putText(img, backend, (20, 62), font, 0.50, badge_color, 1)

    y = 90
    if len(boxes) > 0:
        color = (0, 255, 80)
        cv2.putText(img, f"Objects:   {len(boxes)}", (20, y), font, 0.65, color, 2)
        avg_conf = float(boxes.conf.mean())
        cv2.putText(img, f"Avg conf:  {avg_conf:.1%}", (20, y + 28), font, 0.65, color, 2)
        cv2.putText(img, f"Inference: {inference_ms:.1f} ms", (20, y + 56), font, 0.65, color, 2)
        if track_ids is not None:
            cv2.putText(img, f"Track IDs: {len(set(track_ids))}", (20, y + 84), font, 0.65, (0, 210, 255), 2)
    else:
        color = (160, 160, 160)
        cv2.putText(img, "Objects: 0  —  CLEAR", (20, y), font, 0.65, color, 2)
        cv2.putText(img, f"Inference: {inference_ms:.1f} ms", (20, y + 28), font, 0.65, color, 2)

    return img


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.title("🎯 Object Detection")
st.markdown("Detects **vehicles and humans** using your trained YOLO model.")
st.markdown("---")

st.sidebar.title("⚙️ Settings")

pt_path_input  = st.sidebar.text_input("PyTorch model (.pt)",      value=PT_MODEL_PATH)
trt_path_input = st.sidebar.text_input("TensorRT model (.engine)",  value=TRT_MODEL_PATH)
confidence     = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.40, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 TensorRT")

prefer_trt = st.sidebar.toggle(
    "Prefer TensorRT",
    value=os.path.exists(TRT_MODEL_PATH),
    help="Use .engine if available; fall back to .pt automatically.",
)

if st.sidebar.button("Export → TensorRT", help="Converts best.pt → best.engine (requires NVIDIA GPU + TensorRT)"):
    if not os.path.exists(pt_path_input):
        st.sidebar.error("❌ .pt file not found.")
    else:
        success = export_to_tensorrt(pt_path_input, trt_path_input)
        if success:
            st.sidebar.success("✅ Export complete — toggle 'Prefer TensorRT' and reload.")
            st.cache_resource.clear()
        else:
            st.sidebar.error("Export failed. Ensure TensorRT + CUDA are installed.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**TensorRT requirements:** NVIDIA GPU, CUDA toolkit, and the `tensorrt` "
    "Python package matching your CUDA version. Export runs once; the `.engine` "
    "is then reused on every start."
)

# Load model (result is cached — changes to path/toggle require a rerun)
model, backend = load_model(pt_path_input, trt_path_input, prefer_trt)
if model is None:
    st.stop()

st.sidebar.success(f"Active backend: **{backend}**")

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
            results   = model(image, conf=confidence)
            annotated = results[0].plot(labels=False, conf=False)
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            boxes     = results[0].boxes
            ms        = results[0].speed["inference"]
            annotated = draw_stats_overlay(annotated, boxes, ms, backend)

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
            names = model.names  # {0: 'person', 1: 'car', ...}
            for i, box in enumerate(boxes):
                conf_val = float(box.conf[0])
                cls_id   = int(box.cls[0])
                cls_name = names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                st.write(
                    f"**{cls_name.capitalize()} {i+1}:** "
                    f"confidence = {conf_val:.2%}  |  "
                    f"bbox = [{x1}, {y1}, {x2}, {y2}]"
                )
        else:
            st.info("✅ No objects detected in this image.")


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
        help="Track objects across frames with unique IDs and trajectories.",
    )

    if video_file and st.button("▶️ Process Video"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        cap          = cv2.VideoCapture(tfile.name)
        fps          = max(int(cap.get(cv2.CAP_PROP_FPS)), 1)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            output_path = tempfile.mktemp(suffix=".avi")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        progress    = st.progress(0)
        status_text = st.empty()
        frame_count = 0
        unique_ids: set     = set()
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

                annotated_frame = results[0].plot(labels=False, conf=False)

                if enable_tracking and results[0].boxes.id is not None:
                    boxes_xywh = results[0].boxes.xywh.cpu()
                    ids        = results[0].boxes.id.int().cpu().tolist()

                    for box, tid in zip(boxes_xywh, ids):
                        cx, cy = float(box[0]), float(box[1])
                        track_history.setdefault(tid, []).append((cx, cy))

                        if len(track_history[tid]) > 50:
                            track_history[tid].pop(0)

                        pts = np.array(track_history[tid], dtype=np.int32).reshape(-1, 1, 2)
                        if len(pts) > 1:
                            cv2.polylines(annotated_frame, [pts], False, (0, 0, 0), 5)
                            cv2.polylines(annotated_frame, [pts], False, (0, 230, 255), 3)

                    unique_ids.update(ids)

                out.write(annotated_frame)
                frame_count += 1
                progress.progress(min(frame_count / max(total_frames, 1), 1.0))
                status_text.text(f"Frame {frame_count} / {total_frames}  |  {backend}")

        cap.release()
        out.release()
        os.unlink(tfile.name)

        status_text.empty()
        st.success(f"✅ Video processed with **{backend}**!")

        if enable_tracking and unique_ids:
            st.info(f"🎯 Tracked **{len(unique_ids)}** unique object(s) across {frame_count} frames.")

        st.video(output_path)

        with open(output_path, "rb") as f:
            ext  = os.path.splitext(output_path)[1]
            mime = "video/mp4" if ext == ".mp4" else "video/x-msvideo"
            st.download_button("📥 Download annotated video", f, f"detection_output{ext}", mime)


if __name__ == "__main__":
    pass