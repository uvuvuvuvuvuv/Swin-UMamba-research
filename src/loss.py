import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    """Ours：忽略 ignore_index(默认255) 的联合损失 (CE + Dice)"""

    def __init__(self, ignore_index=255, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def dice_loss(self, probs, target):
        valid = (target != self.ignore_index)
        if valid.sum() == 0:
            return probs.new_tensor(0.0)

        probs_fg = probs[:, 1, :, :][valid].float()
        target_fg = (target == 1)[valid].float()

        smooth = 1e-5
        inter = (probs_fg * target_fg).sum()
        union = probs_fg.sum() + target_fg.sum()
        return 1.0 - (2.0 * inter + smooth) / (union + smooth)

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)
        loss_dice = self.dice_loss(probs, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


class JointLossFull(nn.Module):
    """Baseline / Upper：不忽略任何像素的联合损失 (CE + Dice)"""

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def dice_loss(self, probs, target):
        probs_fg = probs[:, 1, :, :].contiguous().view(-1)
        target_fg = (target == 1).float().contiguous().view(-1)

        smooth = 1e-5
        inter = (probs_fg * target_fg).sum()
        union = probs_fg.sum() + target_fg.sum()
        return 1.0 - (2.0 * inter + smooth) / (union + smooth)

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        probs = F.softmax(logits, dim=1)
        loss_dice = self.dice_loss(probs, target)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice
