import torch

from utils.perceptual import PerceptualLoss

loss = PerceptualLoss()

x = torch.rand(1, 3, 224, 224)

y = torch.rand(1, 3, 224, 224)

print(loss(x, y))