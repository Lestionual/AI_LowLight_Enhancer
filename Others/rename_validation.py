import os

LOW_FOLDER = "dataset/validation/low"
HIGH_FOLDER = "dataset/validation/high"

# Get image filenames
low_images = sorted(os.listdir(LOW_FOLDER))
high_images = sorted(os.listdir(HIGH_FOLDER))

# Make sure both folders contain the same number of images
if len(low_images) != len(high_images):
    raise Exception("The number of low and high images does not match!")

print(f"Found {len(low_images)} image pairs.")

for index, (low_name, high_name) in enumerate(zip(low_images, high_images), start=1):

    extension = os.path.splitext(low_name)[1]

    new_name = f"low{index:05d}{extension}"

    os.rename(
        os.path.join(LOW_FOLDER, low_name),
        os.path.join(LOW_FOLDER, new_name)
    )

    os.rename(
        os.path.join(HIGH_FOLDER, high_name),
        os.path.join(HIGH_FOLDER, new_name)
    )

print("Done!")