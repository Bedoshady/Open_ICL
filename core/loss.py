import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterContrastiveLoss(nn.Module):
    """
    Contrastive Loss based on Semantic Feature Centers (SFC).
    Pulls samples to their class centers and pushes them away from other centers.
    """
    def __init__(self, margin=1.0):
        super(CenterContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, distances, labels):
        # Ignore unknown classes (label -1) during known-class training
        valid_idx = labels >= 0
        distances = distances[valid_idx]
        labels = labels[valid_idx]
        
        if len(distances) == 0:
            return torch.tensor(0.0, device=distances.device, requires_grad=True)
            
        labels_expanded = labels.view(-1, 1)
        dist_to_true_center = distances.gather(1, labels_expanded).squeeze(-1)
        
        # Contrastive part: push away from nearest wrong center
        mask = torch.ones_like(distances).scatter_(1, labels_expanded, 0.)
        dist_to_wrong_center = (distances + (1. - mask) * 1e5).min(dim=1)[0]
        
        # Margin-based loss
        loss = dist_to_true_center.pow(2) + F.relu(self.margin - dist_to_wrong_center).pow(2)
        return loss.mean()

class OpenICLLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(OpenICLLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.contrastive_loss = CenterContrastiveLoss()
        self.alpha = alpha
        
    def forward(self, logits, distances, labels):
        # Cross Entropy for Classification Path (L_cl)
        loss_ce = self.ce_loss(logits, labels)
        
        # Contrastive Loss for Contrast Path (L_ct)
        loss_con = self.contrastive_loss(distances, labels)
        
        # Following Equation (8) from the Open-ICL paper:
        # L = \lambda * L_cl + (1 - \lambda) * L_ct
        return self.alpha * loss_ce + (1 - self.alpha) * loss_con
