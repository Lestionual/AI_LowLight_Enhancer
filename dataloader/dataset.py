import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class LowLightDataset(Dataset):

    def __init__(self, low_dir, high_dir, image_size=512):

        self.low_dir = low_dir
        self.high_dir = high_dir

        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))

        # Check number of images
        assert len(self.low_images) == len(self.high_images), \
            "Number of low and high images do not match."

        # Check filenames
        for low_name, high_name in zip(self.low_images, self.high_images):
            assert low_name == high_name, \
                f"Filename mismatch: {low_name} != {high_name}"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.low_images)

    def __getitem__(self, index):

        low_path = os.path.join(
            self.low_dir,
            self.low_images[index]
        )

        high_path = os.path.join(
            self.high_dir,
            self.high_images[index]
        )

        low_image = Image.open(low_path).convert("RGB")
        high_image = Image.open(high_path).convert("RGB")

        low_image = self.transform(low_image)
        high_image = self.transform(high_image)

        return low_image, high_image