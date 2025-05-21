import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
import pickle
from collections import deque
import os

# ✅ Set page config first
st.set_page_config(page_title="ASL Recognition", layout="centered")

# === PATHS ===
MODEL_PATH = r"C:\Users\HP\Desktop\sign detection\mobilenetv2_asl_model.h5"
LABEL_ENCODER_PATH = r"C:\Users\HP\Desktop\sign detection\label_encoder.pkl"

# === SAFE MODEL LOADING ===
def load_keras_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"❌ Failed to load model.\n\nError: {e}")
        return None

# === SAFE LABEL ENCODER LOADING ===
def load_label_encoder(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Failed to load label encoder.\n\nError: {e}")
        return None

# === LOAD MODEL AND ENCODER ===
model = load_keras_model(MODEL_PATH)
label_map = load_label_encoder(LABEL_ENCODER_PATH)

# === VALIDATE LOADING ===
if model is None or label_map is None:
    st.stop()
else:
    st.success("✅ Model and label encoder loaded successfully.")

# === CONSTANTS ===
img_size = 224
smooth_buffer = deque(maxlen=5)

# === IMAGE PREPROCESSING ===
def preprocess_image(img):
    img = cv2.resize(img, (img_size, img_size))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img.astype(np.float32))
    return tf.expand_dims(img, axis=0)

def enhance_contrast(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

# === PREDICTION ===
def predict(img):
    processed = preprocess_image(img)
    probs = model.predict(processed, verbose=0)[0]
    smooth_buffer.append(probs)
    avg_probs = np.mean(smooth_buffer, axis=0)
    class_idx = np.argmax(avg_probs)
    label = label_map[class_idx]
    confidence = float(avg_probs[class_idx]) * 100
    return label, confidence

# === STREAMLIT UI ===
st.title("🤟 ASL Sign Language Recognition")
st.markdown("Upload an image or use webcam to predict A–Z signs.")

tab1, tab2 = st.tabs(["📤 Upload", "📷 Webcam"])

# === TAB 1: IMAGE UPLOAD ===
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        enhanced = enhance_contrast(image_rgb)
        label, confidence = predict(enhanced)
        st.image(image_rgb, caption=f"Prediction: {label} ({confidence:.2f}%)", use_column_width=True)
        st.success(f"Predicted: **{label}** with **{confidence:.2f}%** confidence")

# === TAB 2: WEBCAM PREDICTION ===
with tab2:
    run = st.toggle("Start Webcam")
    FRAME_WINDOW = st.image([])

    if 'cap' not in st.session_state:
        st.session_state.cap = None

    if run:
        st.session_state.cap = cv2.VideoCapture(0)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while run:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.warning("⚠️ Webcam not detected.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            enhanced = enhance_contrast(rgb)
            label, confidence = predict(enhanced)
            text = f"{label} ({confidence:.1f}%)" if confidence > 60 else "Uncertain"
            cv2.putText(rgb, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            FRAME_WINDOW.image(rgb)
            time.sleep(0.1)

    if not run and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
