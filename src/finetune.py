import os
import warnings
from PIL import ImageFile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import IMG_SIZE, BATCH, DATASET_DIR, INITIAL_MODEL_PATH, FINETUNED_MODEL_PATH
from src.metrics import BinaryF1


DATA_DIR = DATASET_DIR / "PetImages"
EPOCHS = 1
UNFREEZE_LAYERS = 8
LR = 1e-5

STEPS_PER_EPOCH = 60
VAL_STEPS = 20


# DATASET
def prepare_dataset():
    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        verbose=0,
    )

    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
        verbose=0,
    )

    train = train.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val = val.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return train, val


# LOAD & FINETUNE
def load_model():
    model = tf.keras.models.load_model(
        INITIAL_MODEL_PATH,
        custom_objects={"BinaryF1": BinaryF1},
        compile=False,
    )

    for layer in model.layers:
        layer.trainable = False

    count = 0
    for layer in reversed(model.layers):
        if count >= UNFREEZE_LAYERS:
            break
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        layer.trainable = True
        count += 1

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            BinaryF1(),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model


# TRAIN
def main():
    train_ds, val_ds = prepare_dataset()
    model = load_model()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        callbacks=[
            EarlyStopping(
                monitor="val_roc_auc",
                mode="max",
                patience=0,
                restore_best_weights=True,
                verbose=1,
            )
        ],
    )

    model.save(FINETUNED_MODEL_PATH)


if __name__ == "__main__":
    main()
