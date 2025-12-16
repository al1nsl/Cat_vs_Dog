from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = Path("dataset/PetImages")
IMG_SIZE = (96, 96)
BATCH = 32

def get_datasets(
    max_train_batches=None,
    max_val_batches=None,
):
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
    )

    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="binary",
    )

    train = train.map(lambda x, y: (preprocess_input(aug(x)), y))
    val = val.map(lambda x, y: (preprocess_input(x), y))

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    if max_train_batches:
        train = train.take(max_train_batches)
    if max_val_batches:
        val = val.take(max_val_batches)

    return train, val
