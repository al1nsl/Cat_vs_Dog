import os
import warnings
from pathlib import Path
from PIL import ImageFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings(
    "ignore",
    message="Truncated File Read",
    category=UserWarning,
    module="PIL"
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import mixed_precision

from src.config import IMG_SIZE, BATCH, DATASET_DIR, INITIAL_MODEL_PATH
from src.metrics import BinaryF1

# PERFORMANCE
mixed_precision.set_global_policy("mixed_float16")

DATA_DIR = DATASET_DIR / "PetImages"
EPOCHS = 1


# DATA
def prepare_dataset():
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ])

    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        verbose=0,              # ⬅ hide "Found X files"
    )

    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        verbose=0,              # ⬅ hide "Found X files"
    )

    train = train.map(lambda x, y: (preprocess_input(aug(x)), y))
    val = val.map(lambda x, y: (preprocess_input(x), y))

    train = train.cache().prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)

    return train, val


# LOSS
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * tf.pow(1 - pt, gamma) * bce
    return loss


# MODEL
def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(base.input, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=focal_loss(),
        metrics=[
            "accuracy",
            BinaryF1(),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


# TRAIN
if __name__ == "__main__":
    train_ds, val_ds = prepare_dataset()
    model = build_model()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(
                monitor="val_roc_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    model.save(INITIAL_MODEL_PATH)

