# app.py - Military Drone Detection with Tracking

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Military Drone Detection", page_icon="🚁", layout="wide")

@st.cache_resource
def load_model():
    model_path = "results/drone_model/weights/best.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Train first: python main.py")
        return None
    st.success("✅ Model loaded")
    return YOLO(model_path)

def add_overlay_stats(img, boxes, inference_time, tracker_id=None):
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (310, 150), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "DETECTION STATUS", (20, 35), font, 0.7, (255, 255, 255), 2)
    
    y = 65
    color = (0, 255, 0) if len(boxes) > 0 else (128, 128, 128)
    
    if len(boxes) > 0:
        cv2.putText(img, f"Drones: {len(boxes)}", (20, y), font, 0.7, color, 2)
        avg_conf = float(boxes.conf.mean())
        cv2.putText(img, f"Confidence: {avg_conf:.1%}", (20, y + 30), font, 0.7, color, 2)
        cv2.putText(img, f"Time: {inference_time:.1f}ms", (20, y + 60), font, 0.7, color, 2)
        
        if tracker_id is not None:
            unique_ids = len(set(tracker_id.tolist()))
            cv2.putText(img, f"Tracked IDs: {unique_ids}", (20, y + 90), font, 0.7, (0, 200, 255), 2)
    else:
        cv2.putText(img, "Drones: 0", (20, y), font, 0.7, color, 2)
        cv2.putText(img, "Status: CLEAR", (20, y + 30), font, 0.7, color, 2)
        cv2.putText(img, f"Time: {inference_time:.1f}ms", (20, y + 60), font, 0.7, color, 2)
    
    return img

def main():
    st.title("🚁 Military Drone Detection with YOLO11")
    st.markdown("**Target UAVs:** Shahed-136, Lancet, Orlan-10, ZALA, Forpost, and others")
    st.markdown("---")
    
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
    st.sidebar.info("💡 Recommended: 0.35-0.45 to reduce false positives")
    
    model = load_model()
    if model is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📷 Image Upload", "🎥 Video Upload", "📹 Webcam"])
    
    # TAB 1: Image Upload
    with tab1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image_uploader")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            with st.spinner("🔍 Detecting drones..."):
                results = model(image, conf=confidence)
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                boxes = results[0].boxes
                inference_time = results[0].speed['inference']
                annotated_img = add_overlay_stats(annotated_img, boxes, inference_time)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_img, use_container_width=True)
            
            if len(boxes) > 0:
                st.markdown("---")
                st.subheader("🎯 Detection Details")
                for i, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    st.write(f"**Drone {i+1}:** Confidence = {conf:.2%}")
            else:
                st.info("✅ No drones detected")
    
    # TAB 2: Video Upload with Tracking
    with tab2:
        st.header("Upload Video")
        st.info("ℹ️ Note: Tab may switch after upload - click Video tab again if needed")
        video_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'], key="video_uploader")
        
        enable_tracking = st.checkbox("Enable Tracking", value=True, key="enable_tracking", 
                                      help="Track drones across frames with unique IDs and trajectories")
        
        if video_file and st.button("🎬 Process Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            tfile.close()
            
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture(tfile.name)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Suppress OpenH264 warnings
                import sys
                import io
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                
                output_path = "output_detection.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                sys.stderr = old_stderr
                
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    output_path = "output_detection.avi"
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                progress_bar = st.progress(0)
                frame_count = 0
                unique_drone_ids = set()
                track_history = {}
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if enable_tracking:
                        results = model.track(frame, conf=confidence, persist=True, verbose=False)
                    else:
                        results = model(frame, conf=confidence, verbose=False)
                    
                    annotated_frame = results[0].plot()
                    
                    # Draw trajectories
                    if enable_tracking and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xywh.cpu()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        
                        for box, track_id in zip(boxes, track_ids):
                            x, y, w, h = box
                            
                            if track_id not in track_history:
                                track_history[track_id] = []
                            
                            track_history[track_id].append((float(x), float(y)))
                            
                            if len(track_history[track_id]) > 50:
                                track_history[track_id].pop(0)
                            
                            # Draw trajectory with shadow
                            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                            if len(points) > 1:
                                cv2.polylines(annotated_frame, [points], False, (0, 0, 0), 5)
                                cv2.polylines(annotated_frame, [points], False, (0, 255, 255), 3)
                        
                        unique_drone_ids.update(track_ids)
                    
                    out.write(annotated_frame)
                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                
                cap.release()
                out.release()
                
                st.success("✅ Video processed!")
                
                if enable_tracking and unique_drone_ids:
                    st.info(f"🎯 Tracked {len(unique_drone_ids)} unique drone(s) across {frame_count} frames")
                
                st.video(output_path)
                
                with open(output_path, 'rb') as f:
                    st.download_button("📥 Download Video", f, "drone_detection.mp4", "video/mp4")

if __name__ == "__main__":
    main()