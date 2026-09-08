from torch.utils.data import DataLoader

from dataloader.dataset import LowLightDataset
from config.config import *
from utils.visualize import show_image


def main():

    dataset = LowLightDataset(
        TRAIN_LOW,
        TRAIN_HIGH,
        IMAGE_SIZE
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    print("Dataset Size:", len(dataset))

    for low, high in loader:

        print("Low:", low.shape)
        print("High:", high.shape)

        show_image(low[0])
        show_image(high[0])

        break


if __name__ == "__main__":
    main()