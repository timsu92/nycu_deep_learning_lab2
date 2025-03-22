import torch


def dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    # turn into uint8 for bitwise operations
    pred_mask = pred_mask.to(torch.uint8)
    gt_mask = gt_mask.to(torch.uint8)
    return 2 * (pred_mask & gt_mask).sum() / (pred_mask.numel() + gt_mask.numel())
