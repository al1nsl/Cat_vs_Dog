import os
from PIL import Image, ImageFile
from pathlib import Path


ImageFile.LOAD_TRUNCATED_IMAGES = True
DATASET_DIR = Path("dataset/PetImages")
VALID_EXT = (".jpg", ".jpeg", ".png")


def clean_image(path):
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        img.save(path, "JPEG", quality=95, subsampling=0)
        return True
    except Exception:
        return False


def main():
    removed = 0
    fixed = 0

    for root, _, files in os.walk(DATASET_DIR):
        for name in files:
            if not name.lower().endswith(VALID_EXT):
                continue

            path = os.path.join(root, name)

            if clean_image(path):
                fixed += 1
            else:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass

    print(f"Fixed images: {fixed}")
    print(f"Removed images: {removed}")


if __name__ == "__main__":
    main()
