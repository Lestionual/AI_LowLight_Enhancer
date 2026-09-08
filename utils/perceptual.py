import torch
import torch.nn as nn

from torchvision.models import (
    vgg16,
    VGG16_Weights
)


class PerceptualLoss(nn.Module):

    def __init__(self):

        super().__init__()

        vgg = vgg16(
            weights=VGG16_Weights.DEFAULT
        )

        self.features = nn.Sequential(
            *list(vgg.features.children())[:16]
        )

        # Freeze VGG
        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.features.eval()

        self.loss = nn.L1Loss()

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def forward(self, prediction, target):

        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std

        prediction_features = self.features(
            prediction
        )

        with torch.no_grad():

            target_features = self.features(
                target
            )

        return self.loss(
            prediction_features,
            target_features
        )