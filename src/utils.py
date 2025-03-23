import torch


# https://chih-sheng-huang821.medium.com/%E5%BD%B1%E5%83%8F%E5%88%87%E5%89%B2%E4%BB%BB%E5%8B%99%E5%B8%B8%E7%94%A8%E7%9A%84%E6%8C%87%E6%A8%99-iou%E5%92%8Cdice-coefficient-3fcc1a89cd1c
def dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, epsilon: float = 1e-6):
    # turn into uint8 for bitwise operations
    pred_mask = pred_mask.to(torch.uint8)
    gt_mask = gt_mask.to(torch.uint8)
    dim = (-1, -2, -3)  # CHW
    same = 2 * (pred_mask == gt_mask).sum(dim=dim)
    whole = 2 * torch.tensor(pred_mask[0].numel(), device=pred_mask.device)
    whole = whole.expand(pred_mask.size(0))

    dice = same / whole
    return dice.mean()


# https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py
def dice_loss(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    return 1 - dice_score(pred_mask, gt_mask)
