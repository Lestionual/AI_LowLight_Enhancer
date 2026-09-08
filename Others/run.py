import torch

checkpoint = torch.load(
    "checkpoints/modelV4.pth",
    map_location="cpu"
)

print(checkpoint.keys())