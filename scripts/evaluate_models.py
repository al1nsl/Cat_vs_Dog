import os
import sys
import warnings
import contextlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# Suppress STDERR (libjpeg spam)
@contextlib.contextmanager
def suppress_stderr():
    devnull = open(os.devnull, "w")
    old_stderr = sys.stderr
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

# Imports AFTER suppression
with suppress_stderr():
    import tensorflow as tf
    from src.train import prepare_dataset
    from src.metrics import BinaryF1

tf.get_logger().setLevel("ERROR")

# CONFIG
MODELS = {
    "initial": "models/cat_dog_initial.h5",
    "finetuned": "models/cat_dog_finetuned.keras",
}

OUT_FILE = "reports/metrics.txt"
MAX_EVAL_BATCHES = 30


def evaluate(model_path, val_ds):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"BinaryF1": BinaryF1},
        compile=False,
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            BinaryF1(name="f1"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model.evaluate(
        val_ds.take(MAX_EVAL_BATCHES),
        verbose=1
    )


def main():
    # Dataset
    with suppress_stderr():
        _, val_ds = prepare_dataset()

    # Evaluation
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for name, path in MODELS.items():
            print(f"\n🔍 Evaluating: {name}")

            with suppress_stderr():
                results = evaluate(path, val_ds)

            metrics = dict(zip(
                ["loss", "accuracy", "f1", "roc_auc"],
                results
            ))

            f.write(f"[{name}]\n")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
                f.write(f"{k}: {v:.6f}\n")
            f.write("\n")

    print(f"\n✅ Metrics saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
