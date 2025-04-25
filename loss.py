import torch

def dice_loss(pred, gt_mask, eps=1):
    """
    Computes the Dice Loss for binary segmentation.
    NOTE: 
        - If your masks are not strictly binary (e.g., contain 255 instead of 1), normalize them.
        - eps helps us avoid division by zero
    """
    dice_coeff = dice_metric(pred=pred, gt_mask=gt_mask, eps=eps)
    return 1 - dice_coeff

def dice_metric(pred, gt_mask, eps=1e-6): # plus haut => meilleur
    assert set(gt_mask.unique().tolist()) <= {0, 1}, "Le masque de vérité doit être binaire, 0 ou 1"

    pred = torch.sigmoid(pred)

    intersection = (pred * gt_mask).sum(dim=(2,3)) # flatten?
    union = pred.sum(dim=(2,3)) + gt_mask.sum(dim=(2,3))

    dice_coeff = (2. * intersection + eps) / (union + eps)
    return dice_coeff.mean() # moyenne sur le batch