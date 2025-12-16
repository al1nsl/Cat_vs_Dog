import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple

from src.config import INFER_IMG_SIZE, THRESHOLD, MODEL_PATH


# =========================
# Model loading
# =========================

_model = None


def load_model() -> tf.keras.Model:
    """
    Lazy-load model (loaded once per process).
    """
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model


# =========================
# Preprocessing
# =========================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Prepare PIL image for MobileNetV2 inference.
    Returns shape: (1, H, W, 3)
    """
    image = image.convert("RGB")
    image = image.resize(INFER_IMG_SIZE)

    img = np.array(image, dtype=np.float32)
    img = img / 255.0

    return np.expand_dims(img, axis=0)


# =========================
# Prediction
# =========================

def predict(image: Image.Image) -> Tuple[str, float]:
    """
    Run inference and return label + confidence.

    Returns:
        label: "cat" | "dog"
        confidence: float in [0, 1]
    """
    model = load_model()
    x = preprocess_image(image)

    prob = model.predict(x, verbose=0)[0][0]
    prob = float(prob)

    label = "dog" if prob >= THRESHOLD else "cat"
    confidence = prob if label == "dog" else 1.0 - prob

    return label, confidence
