import torch

from utils.color_loss import ColorLoss

loss = ColorLoss()

prediction = torch.rand(1, 3, 256, 256)

target = torch.rand(1, 3, 256, 256)

print(loss(prediction, target))