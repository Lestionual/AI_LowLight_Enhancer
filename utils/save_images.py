import os
import torch
import torchvision.utils as vutils


def save_sample(low, prediction, target, epoch):
    """
    Save one comparison image for each epoch.

    Layout:
    -----------------------------------------
    | Low-Light | Prediction | Ground Truth |
    -----------------------------------------
    """

    os.makedirs("outputs", exist_ok=True)

    # Clamp predictions to valid image range
    prediction = torch.clamp(prediction, 0.0, 1.0)

    comparison = torch.cat(
        [
            low.cpu(),
            prediction.cpu(),
            target.cpu()
        ],
        dim=0
    )

    vutils.save_image(
        comparison,
        f"outputs/epoch_{epoch}.png",
        nrow=low.size(0),
        normalize=True
    )