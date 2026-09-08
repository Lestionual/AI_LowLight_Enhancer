import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from utils.perceptual import PerceptualLoss
from utils.color_loss import ColorLoss
class HybridLoss(nn.Module):

    def __init__(
        self,
        l1_weight=0.5,
        ssim_weight=0.2,
        perceptual_weight=0.3
    ):

        super().__init__()

        self.l1 = nn.L1Loss()

        self.ssim = SSIM(
            data_range=1.0,
            size_average=True,
            channel=3
        )
        self.perceptual = PerceptualLoss()
        self.color = ColorLoss()

        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, prediction, target):
        l1_loss = self.l1(
            prediction,
            target
        )

        ssim_loss = 1 - self.ssim(
            prediction,
            target
        )

        perceptual_loss = self.perceptual(
            prediction,
            target
        )
        color_loss = self.color(
            prediction,
            target
        )

        total_loss = (
            self.l1_weight * l1_loss +
            self.ssim_weight * ssim_loss +
            self.perceptual_weight * perceptual_loss +
            0.05 * color_loss
        )

        return (
            total_loss,
            l1_loss,
            ssim_loss,
            perceptual_loss,
            color_loss
        )