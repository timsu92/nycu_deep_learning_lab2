import torch
import torch.nn as nn
from torch.utils.data import DataLoader as DataLoader

from .utils import dice_score


def evaluate(net: nn.Module, data: DataLoader, device: torch.device):
    with torch.set_grad_enabled(False):
        net.eval()
        net.to(device)
        dice = 0
        for batch in data:
            images = batch[:, :-1, ...].to(device)
            masks = batch[:, -1:, ...].to(device)
            pred = net(images)
            dice += dice_score(torch.round(torch.sigmoid(pred)), masks)
    return dice / len(data)
