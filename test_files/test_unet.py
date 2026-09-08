import torch
from torch.utils.data import DataLoader

from models.unet import UNet
from dataloader.dataset import LowLightDataset
from config.config import *


def main():

    dataset = LowLightDataset(
        TRAIN_LOW,
        TRAIN_HIGH,
        IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model = UNet()

    low, high = next(iter(loader))

    output = model(low)

    print("Input :", low.shape)
    print("Target:", high.shape)
    print("Output:", output.shape)


if __name__ == "__main__":
    main()