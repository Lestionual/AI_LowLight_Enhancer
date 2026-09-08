import os
import time

import torch
from PIL import Image
from torchvision import transforms

from models.unet import UNet
from config.config import IMAGE_SIZE
from utils.metrics import calculate_psnr, calculate_ssim
MODEL_PATH = "checkpoints/ModelV5.pth"

INPUT_FOLDER = "dataset/test/low"

TARGET_FOLDER = "dataset/test/high"

OUTPUT_FOLDER = "outputs/test_results"
os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)
model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model Loaded.")
total_psnr = 0.0
total_ssim = 0.0
total_time = 0.0
image_count = 0
transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])
for filename in sorted(os.listdir(INPUT_FOLDER)):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    print(f"Processing {filename}...")
    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    target_path = os.path.join(
        TARGET_FOLDER,
        filename
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )
    input_image = Image.open(
        input_path
    ).convert("RGB")

    target_image = Image.open(
        target_path
    ).convert("RGB")

    original_size = input_image.size
    input_tensor = transform(
        input_image
    ).unsqueeze(0).to(device)

    target_tensor = transform(
        target_image
    ).unsqueeze(0).to(device)
    start = time.time()

    with torch.no_grad():

        prediction = model(input_tensor)

    elapsed = time.time() - start
    total_psnr += calculate_psnr(
        prediction,
        target_tensor
    )

    total_ssim += calculate_ssim(
        prediction,
        target_tensor
    )

    total_time += elapsed

    image_count += 1
    prediction = prediction.squeeze(0)

    prediction = torch.clamp(
        prediction,
        0,
        1
    )

    save_image = transforms.ToPILImage()(
        prediction.cpu()
    )

    save_image = save_image.resize(
        original_size
    )

    save_image.save(output_path)
    print(
        f"Done {image_count} | "
        f"PSNR: {total_psnr / image_count:.2f} | "
        f"SSIM: {total_ssim / image_count:.4f}"
    )
print("\n===================================")
print(f"Images Tested : {image_count}")
print(f"Average PSNR  : {total_psnr / image_count:.2f} dB")
print(f"Average SSIM  : {total_ssim / image_count:.4f}")
print(f"Average Time  : {(total_time / image_count) * 1000:.2f} ms")
print("===================================")