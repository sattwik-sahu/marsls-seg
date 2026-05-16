import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCoshDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        """
        Log-Cosh Dice Loss for robust semantic segmentation.
        Args:
            smooth: Smoothing factor to prevent division by zero and handle small masks.
            eps: Small constant for numerical stability.
        """
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted logits [B, C, H, W]
            y_true: Ground truth masks [B, H, W] or [B, C, H, W]
        """
        # 1. Convert logits to probabilities
        num_classes = y_pred.size(1)
        y_pred = F.softmax(y_pred, dim=1)

        # 2. One-hot encode ground truth if necessary
        if len(y_true.shape) == 3:  # [B, H, W] -> [B, C, H, W]
            y_true = F.one_hot(y_true.long(), num_classes).permute(0, 3, 1, 2).float()

        # 3. Calculate Dice Coefficient per class
        # Flatten spatial dimensions
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)

        intersection = (y_pred * y_true).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth + self.eps
        )

        # 4. Dice Loss
        dice_loss = 1.0 - dice_coeff

        # 5. Apply Log-Cosh: L = log(cosh(DiceLoss))
        # Log-Cosh is approximated as x + softplus(-2x) - log(2) for numerical stability
        return torch.log(torch.cosh(dice_loss + self.eps))
