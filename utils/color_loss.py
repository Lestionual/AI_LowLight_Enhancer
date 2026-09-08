import torch
import torch.nn as nn


class ColorLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = nn.L1Loss()

    def rgb_to_hsv(self, image):

        r = image[:, 0]
        g = image[:, 1]
        b = image[:, 2]

        max_rgb, _ = image.max(dim=1)
        min_rgb, _ = image.min(dim=1)

        delta = max_rgb - min_rgb

        # -------------------------
        # Hue
        # -------------------------

        hue = torch.zeros_like(max_rgb)

        mask = delta != 0

        red = (max_rgb == r) & mask
        green = (max_rgb == g) & mask
        blue = (max_rgb == b) & mask

        hue[red] = (
            (g[red] - b[red]) / delta[red]
        ) % 6

        hue[green] = (
            (b[green] - r[green]) / delta[green]
        ) + 2

        hue[blue] = (
            (r[blue] - g[blue]) / delta[blue]
        ) + 4

        hue = hue / 6.0

        # -------------------------
        # Saturation
        # -------------------------

        saturation = torch.zeros_like(max_rgb)

        saturation[max_rgb != 0] = (
            delta[max_rgb != 0]
            / max_rgb[max_rgb != 0]
        )

        # -------------------------
        # Value
        # -------------------------

        value = max_rgb

        hsv = torch.stack(
            [
                hue,
                saturation,
                value
            ],
            dim=1
        )

        return hsv

    def forward(
        self,
        prediction,
        target
    ):

        prediction_hsv = self.rgb_to_hsv(
            prediction
        )

        target_hsv = self.rgb_to_hsv(
            target
        )

        hue_loss = self.l1(
            prediction_hsv[:, 0:1],
            target_hsv[:, 0:1]
        )

        saturation_loss = self.l1(
            prediction_hsv[:, 1:2],
            target_hsv[:, 1:2]
        )

        return (
            hue_loss +
            saturation_loss
        )