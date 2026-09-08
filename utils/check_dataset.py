import os

low = sorted(os.listdir("dataset/train/low"))
high = sorted(os.listdir("dataset/train/high"))

print("Low images :", len(low))
print("High images:", len(high))

assert len(low) == len(high)

for l, h in zip(low, high):

    assert l == h

print("Dataset Verified")