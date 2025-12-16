from pathlib import Path

# =========================
# PROJECT PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# =========================
# GLOBAL CONFIG
# =========================
IMG_SIZE = (96, 96)
BATCH = 64

# =========================
# TRAINING
# =========================
INITIAL_MODEL_PATH = MODELS_DIR / "cat_dog_initial.h5"
FINETUNED_MODEL_PATH = MODELS_DIR / "cat_dog_finetuned.keras"

# =========================
# INFERENCE
# =========================
INFER_IMG_SIZE = IMG_SIZE
THRESHOLD = 0.5
MODEL_PATH = FINETUNED_MODEL_PATH

