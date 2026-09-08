from PIL import Image
import os

LOW_PATH = "dataset/Train/Low"
HIGH_PATH = "dataset/Train/High"

SIZE = (512, 512)


def resize_images(folder):

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        image = Image.open(path)

        image = image.resize(SIZE)

        image.save(path)

    print(folder, "completed")


resize_images(LOW_PATH)
resize_images(HIGH_PATH)
