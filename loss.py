import torch

def dice_loss_2d(pred, gt_mask, eps=1):
    dice_coeff = dice_metric_2d(pred=pred, gt_mask=gt_mask, eps=eps)
    return 1 - dice_coeff

def dice_metric_2d(pred, gt_mask, eps=1e-6): # plus haut => meilleur
    assert set(gt_mask.unique().tolist()) <= {0, 1}, "Le masque de vérité doit être binaire, 0 ou 1"

    pred = torch.sigmoid(pred)

    intersection = (pred * gt_mask).sum(dim=(2,3)) # flatten?
    union = pred.sum(dim=(2,3)) + gt_mask.sum(dim=(2,3))

    dice_coeff = (2. * intersection + eps) / (union + eps)
    return dice_coeff.mean() # moyenne sur le batch

def dice_loss_3d(pred, gt_mask, eps=1):
    dice_coeff = dice_metric_3d(pred=pred, gt_mask=gt_mask, eps=eps)
    return 1 - dice_coeff

def dice_metric_3d(pred, gt_mask, eps=1e-6): # plus haut => meilleur
    assert set(gt_mask.unique().tolist()) <= {0, 1}, "Le masque de vérité doit être binaire, 0 ou 1"

    pred = torch.sigmoid(pred)

    intersection = (pred * gt_mask).sum(dim=(2,3,4)) # flatten?
    union = pred.sum(dim=(2,3,4)) + gt_mask.sum(dim=(2,3,4))

    dice_coeff = (2. * intersection + eps) / (union + eps)
    return dice_coeff.mean() # moyenne sur le batch