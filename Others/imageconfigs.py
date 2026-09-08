from PIL import Image
import os

# ===========================
# Configuration
# ===========================

SIZE = (512, 512)

LOW_INPUT = "dataset/train/low"
HIGH_INPUT = "dataset/train/high"

LOW_OUTPUT = "dataset_processed/train/low"
HIGH_OUTPUT = "dataset_processed/train/high"

# ===========================
# Create output folders
# ===========================

os.makedirs(LOW_OUTPUT, exist_ok=True)
os.makedirs(HIGH_OUTPUT, exist_ok=True)

# ===========================
# Supported image formats
# ===========================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)

# ===========================
# Get image lists
# ===========================

low_images = sorted([
    f for f in os.listdir(LOW_INPUT)
    if f.lower().endswith(VALID_EXTENSIONS)
])

high_images = sorted([
    f for f in os.listdir(HIGH_INPUT)
    if f.lower().endswith(VALID_EXTENSIONS)
])

if len(low_images) != len(high_images):
    print("WARNING:")
    print(f"Low images : {len(low_images)}")
    print(f"High images: {len(high_images)}")
    print("Pairing will stop at the smaller folder.\n")

num_images = min(len(low_images), len(high_images))

# ===========================
# Resize & Rename
# ===========================

for i in range(num_images):

    filename = f"{i:05d}.png"

    low_path = os.path.join(
        LOW_INPUT,
        low_images[i]
    )

    high_path = os.path.join(
        HIGH_INPUT,
        high_images[i]
    )

    low_image = Image.open(low_path).convert("RGB")
    high_image = Image.open(high_path).convert("RGB")

    low_image = low_image.resize(
        SIZE,
        Image.LANCZOS
    )

    high_image = high_image.resize(
        SIZE,
        Image.LANCZOS
    )

    low_image.save(
        os.path.join(
            LOW_OUTPUT,
            filename
        )
    )

    high_image.save(
        os.path.join(
            HIGH_OUTPUT,
            filename
        )
    )

    if (i + 1) % 100 == 0 or i == num_images - 1:
        print(f"Processed {i + 1}/{num_images}")

print("\nFinished!")