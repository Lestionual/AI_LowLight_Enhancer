import torch

from utils.losses import HybridLoss

criterion = HybridLoss()

prediction = torch.rand(
    1, 3, 224, 224
)

target = torch.rand(
    1, 3, 224, 224
)

loss = criterion(
    prediction,
    target
)

print("Loss:", loss)