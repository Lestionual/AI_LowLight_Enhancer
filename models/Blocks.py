import torch
import torch.nn as nn
from models.Attention import AttentionGate

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels)

        )

        # Match dimensions for residual connection
        if in_channels != out_channels:

            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.double_conv(x)

        x = x + residual

        return self.relu(x)
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):

        return self.block(x)
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.attention = AttentionGate(
            gate_channels=in_channels // 2,
            skip_channels=in_channels // 2,
            inter_channels=out_channels // 2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(self, x_decoder, x_encoder):

        # Upsample decoder features
        x_decoder = self.up(x_decoder)

        # Apply attention to encoder features
        x_encoder = self.attention(
            x_decoder,
            x_encoder
        )

        # Concatenate
        x = torch.cat(
            [x_encoder, x_decoder],
            dim=1
        )

        return self.conv(x)
class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        return self.conv(x)