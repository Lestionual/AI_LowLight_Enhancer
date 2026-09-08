import torch
from torch.utils.data import DataLoader
from utils.save_images import save_sample
from config.config import *
from dataloader.dataset import LowLightDataset
from models.unet import UNet
from utils.metrics import (
    calculate_psnr,
    calculate_ssim
)
import os
import time

from utils.checkpoint import save_checkpoint

def main():

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using:", device)

    # -----------------------------
    # Training Dataset
    # -----------------------------
    train_dataset = LowLightDataset(
        TRAIN_LOW,
        TRAIN_HIGH,
        IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # -----------------------------
    # Validation Dataset
    # -----------------------------
    validation_dataset = LowLightDataset(
        VAL_LOW,
        VAL_HIGH,
        IMAGE_SIZE
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = UNet().to(device)
    checkpoint_path = "checkpoints/modelV5.pth"

    if os.path.exists(checkpoint_path):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(f"Loaded model from {checkpoint_path}")

    # -----------------------------
    # Loss
    # -----------------------------
    criterion = torch.nn.L1Loss()

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # -----------------------------
    # Epoch Loop
    # -----------------------------
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        # =========================
        # Training
        # =========================
        model.train()

        running_train_loss = 0.0

        for batch_index, (low, high) in enumerate(
            train_loader
        ):

            low = low.to(device)
            high = high.to(device)

            optimizer.zero_grad()

            prediction = model(low)

            if not torch.isfinite(prediction).all():
                print("Prediction contains NaN or Inf.")
                return

            loss = criterion(
                prediction,
                high
            )

            if not torch.isfinite(loss):
                print("Training loss contains NaN or Inf.")
                return

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

            if batch_index % 20 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Batch {batch_index}/{len(train_loader)} "
                    f"Loss: {loss.item():.6f}"
                )

        average_train_loss = (
            running_train_loss
            / len(train_loader)
        )

        # =========================
        # Validation
        # =========================
        model.eval()

        running_validation_loss = 0.0

        with torch.no_grad():
            running_psnr = 0.0
            running_ssim = 0.0
            for batch_index, (low, high) in enumerate(validation_loader):
                

                low = low.to(device)
                high = high.to(device)

                prediction = model(low)
                if batch_index == 0:

                    validation_low = low.detach().cpu()

                    validation_prediction = prediction.detach().cpu()

                    validation_high = high.detach().cpu()
                if not torch.isfinite(prediction).all():
                    print(
                        "Validation prediction contains "
                        "NaN or Inf."
                    )
                    return

                validation_loss = criterion(
                    prediction,
                    high
                )

                if not torch.isfinite(validation_loss):
                    print(
                        "Validation loss contains "
                        "NaN or Inf."
                    )
                    return

                running_validation_loss += (
                    validation_loss.item()
                )
                running_psnr += calculate_psnr(
                    prediction,
                    high
                )

                running_ssim += calculate_ssim(
                    prediction,
                    high
                )
        
        average_validation_loss = (
            running_validation_loss
            / len(validation_loader)
        )
        average_psnr = (
            running_psnr /
            len(validation_loader)
        )

        average_ssim = (
            running_ssim /
            len(validation_loader)
        )
        save_sample(
            validation_low,
            validation_prediction,
            validation_high,
            epoch + 1
        )
        elapsed = time.time() - epoch_start

        minutes = int(elapsed // 60)

        seconds = int(elapsed % 60)
        # =========================
        # Epoch Results
        # =========================
        print("\n===================================")
        print(f"Epoch           : {epoch + 1}/{EPOCHS}")
        print(
            f"Train Loss      : "
            f"{average_train_loss:.6f}"
        )
        print(
            f"Validation Loss : "
            f"{average_validation_loss:.6f}"
        )
        print(
            f"PSNR            : "
            f"{average_psnr:.2f} dB"
        )

        print(
            f"SSIM            : "
            f"{average_ssim:.4f}"
        )
        print(f"Epoch Time      : {minutes}m {seconds:02d}s")
        print("===================================\n")
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{epoch + 1}.pth"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            loss=average_validation_loss,
            filename=checkpoint_path
        )
if __name__ == "__main__":
    main()