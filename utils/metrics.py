import torch
import math
from pytorch_msssim import ssim

def calculate_psnr(prediction, target):

    mse = torch.mean((prediction - target) ** 2)

    if mse == 0:
        return 100

    return 20 * math.log10(1.0 / math.sqrt(mse.item()))
def calculate_ssim(prediction, target):

    score = ssim(
        prediction,
        target,
        data_range=1.0,
        size_average=True
    )

    return score.item() 