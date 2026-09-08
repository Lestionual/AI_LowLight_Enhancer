import torch
from PIL import Image
from torchvision import transforms
import os

from models.unet import UNet
from config.config import IMAGE_SIZE


# ==============================
# Paths
# ==============================

MODEL_PATH = "checkpoints/best_model.pth"

INPUT_IMAGE = "input.jpg"

OUTPUT_IMAGE = "enhanced_output.png"


# ==============================
# Device
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print("Using device:", device)


# ==============================
# Load Model
# ==============================

model = UNet().to(device)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print("Model loaded successfully")


# ==============================
# Image preprocessing
# ==============================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


# ==============================
# Load Image
# ==============================

image = Image.open(
    INPUT_IMAGE
).convert("RGB")
image_tensor = transform(image)
# -----------------------------
# Brightness Analysis
# -----------------------------
mean_brightness = image_tensor.mean().item()

dark_pixel_ratio = (
    (image_tensor < 0.20)
    .float()
    .mean()
    .item()
)

print(f"Mean Brightness : {mean_brightness:.3f}")
print(f"Dark Pixel Ratio: {dark_pixel_ratio:.3f}")
mean_brightness = image_tensor.mean().item()

print(f"Mean brightness: {mean_brightness:.3f}")


original_size = image.size


image_tensor = transform(image)


# -----------------------------
# Decide whether enhancement is needed
# -----------------------------

BRIGHTNESS_THRESHOLD = 0.35
DARK_PIXEL_THRESHOLD = 0.45

if (
    mean_brightness > BRIGHTNESS_THRESHOLD
    and
    dark_pixel_ratio < DARK_PIXEL_THRESHOLD
):

    print("Image already bright. Skipping enhancement.")

    save_image = image

else:

    print("Low-light image detected. Enhancing...")

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        enhanced = model(image_tensor)
    print(
        "Input Mean :",
        image_tensor.mean().item()
    )

    print(
        "Output Mean:",
        enhanced.mean().item()
    )

    print(
        "Input Max  :",
        image_tensor.max().item()
    )

    print(
        "Output Max :",
        enhanced.max().item()
    )

    print(
        "Output Mean:",
        enhanced.mean().item()
    )


    alpha = 1

    enhanced = (
        alpha * enhanced +
        (1 - alpha) * image_tensor
    )

    enhanced = torch.clamp(
        enhanced,
        0,
        1
    )
    input_mean = image_tensor.mean()
    output_mean = enhanced.mean()

    max_increase = 0.20

    if output_mean > input_mean + max_increase:

        target_mean = input_mean + max_increase

        scale = target_mean / output_mean

        enhanced = enhanced * scale
    # Increase contrast while keeping roughly the same mean brightness
    contrast_strength = 1.20

    current_mean = enhanced.mean(
        dim=(2, 3),
        keepdim=True
    )

    enhanced = (
        current_mean
        + contrast_strength * (enhanced - current_mean)
    )

    enhanced = torch.clamp(
        enhanced,
        0.0,
        1.0
    )
    print(f"Input Mean : {input_mean:.3f}")
    print(f"Output Mean: {output_mean:.3f}")
    print(f"Scale      : {scale:.3f}" if output_mean > input_mean + max_increase else "Scale      : 1.000")
    enhanced = enhanced.squeeze(0)
    save_image = transforms.ToPILImage()(
        enhanced.cpu()
    )


# Resize back to original dimensions

save_image = save_image.resize(
    original_size
)


save_image.save(
    OUTPUT_IMAGE
)


print(
    f"Saved enhanced image as {OUTPUT_IMAGE}"
)
print("Final Output Mean:", enhanced.mean().item())