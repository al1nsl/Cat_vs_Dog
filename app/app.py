import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module="PIL"
)

import streamlit as st
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from PIL import Image

from src.config import INFER_IMG_SIZE, THRESHOLD, MODEL_PATH
from src.inference import preprocess_image
from src.metrics import BinaryF1


# Page config
st.set_page_config(
    page_title="🐶🐱 Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# Check model existence
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

# Styles
st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }
        .block-container { padding-top: 2rem; }
        .title { text-align: center; font-size: 40px; font-weight: 700; }
        .subtitle { text-align: center; color: #aaaaaa; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    '<div class="title">🐾 Cat vs Dog Classifier</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle"© 2025 Gatilov Igor, Alina Karnaukhova, Denis Grigoriev. All rights reserved. </div>',
    unsafe_allow_html=True
)
st.divider()

# Load model (cached)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BinaryF1": BinaryF1},
        compile=False
    )

model = load_model()

# Image uploader
uploaded_file = st.file_uploader(
    "📸 Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Basic validation
    if image.size[0] < 64 or image.size[1] < 64:
        st.error("❌ Image is too small")
        st.stop()

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    # Preprocess
    img = preprocess_image(image)

    # Predict
    with st.spinner("🧠 Running inference..."):
        pred = model.predict(img, verbose=0)[0][0]

    label = "🐶 Dog" if pred > THRESHOLD else "🐱 Cat"
    confidence = float(pred if pred > THRESHOLD else 1 - pred)

    st.divider()

    # Result
    st.markdown(f"### Prediction: **{label}**")
    st.progress(confidence)
    st.caption(f"Confidence: **{confidence * 100:.2f}%**")
