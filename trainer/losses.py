import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for sequence labeling with class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for sequence labeling"""
    def __init__(self, pos_weight=10.0, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, seq_len, 2) - logits
        # targets: (batch_size, seq_len) - 0 or 1
        logits = inputs.view(-1, 2)  # (batch_size * seq_len, 2)
        targets = targets.view(-1)   # (batch_size * seq_len)
        
        # Convert to binary classification
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        # Calculate weighted loss
        pos_loss = -torch.log(torch.softmax(logits, dim=1)[:, 1] + 1e-8) * pos_mask
        neg_loss = -torch.log(torch.softmax(logits, dim=1)[:, 0] + 1e-8) * neg_mask
        
        total_loss = pos_loss * self.pos_weight + neg_loss
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class DiceLoss(nn.Module):
    """Dice Loss for sequence labeling with class imbalance"""
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (batch_size, seq_len, 2) - logits
        # targets: (batch_size, seq_len) - 0 or 1
        probs = torch.softmax(inputs, dim=2)[:, :, 1]  # (batch_size, seq_len)
        targets = targets.float()  # (batch_size, seq_len)
        
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss

class CombinedLoss(nn.Module):
    """Combined Focal + BCE Loss for better sequence labeling performance"""
    def __init__(self, focal_weight=0.7, bce_weight=0.3, alpha=1, gamma=2, pos_weight=10.0):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.weighted_bce = WeightedBCELoss(pos_weight=pos_weight)
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        weighted_bce = self.weighted_bce(inputs, targets)
        
        return self.focal_weight * focal + self.bce_weight * weighted_bce

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=1, reduction='mean', device="cuda"):
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss