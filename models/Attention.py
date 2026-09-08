import torch
import torch.nn as nn


class AttentionGate(nn.Module):

    def __init__(self, gate_channels, skip_channels, inter_channels):

        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):

        g = self.gate_conv(gate)

        x = self.skip_conv(skip)

        attention = self.relu(g + x)

        attention = self.psi(attention)

        return skip * attention