import torch

def dice_loss(pred, gt_mask, eps=1):
    """
    Computes the Dice Loss for binary segmentation.
    NOTE: 
        - If your masks are not strictly binary (e.g., contain 255 instead of 1), normalize them.
        - eps helps us avoid division by zero
    """
    dice_coeff = dice_metric(pred=pred, gt_mask=gt_mask, eps=eps)
    return 1 - dice_coeff.mean()

def dice_metric(pred, gt_mask, eps=1):
    pred = torch.sigmoid(pred) # the sigmoid converts the logits into probabilities
    intersection = (pred * gt_mask).sum(dim=(2, 3)) # dim 2, 3 = H, W
    union = pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    if union == 0: union += eps  # eps to avoid division by zero
    dice_coeff = (2. * intersection) / union
    return dice_coeff