import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterContrastiveLoss(nn.Module):
    """
    Contrastive Loss based on Semantic Feature Centers (SFC).
    Pulls samples to their class centers and pushes them away from other centers.
    """
    def __init__(self, num_classes, feature_dim, margin=1.0):
        super(CenterContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.margin = margin
        
        # Trainable Semantic Feature Centers (SFCs)
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
    def forward(self, features, labels):
        # Ignore unknown classes (label -1) during known-class training
        valid_idx = labels >= 0
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) == 0:
            return torch.tensor(0.0, device=self.centers.device, requires_grad=True)
            
        # Normalize
        features = F.normalize(features, p=2, dim=1)
        centers = F.normalize(self.centers, p=2, dim=1)
        
        # Compute distances
        dist_mat = torch.cdist(features, centers, p=2)
        
        labels_expanded = labels.view(-1, 1)
        dist_to_true_center = dist_mat.gather(1, labels_expanded).squeeze(-1)
        
        # Contrastive part: push away from nearest wrong center
        mask = torch.ones_like(dist_mat).scatter_(1, labels_expanded, 0.)
        dist_to_wrong_center = (dist_mat + (1. - mask) * 1e5).min(dim=1)[0]
        
        # Margin-based loss
        loss = dist_to_true_center.pow(2) + F.relu(self.margin - dist_to_wrong_center).pow(2)
        return loss.mean()
        
    def add_classes(self, new_num_classes):
        """
        Add new centers for new classes during incremental learning.
        """
        if new_num_classes <= self.num_classes:
            return
            
        new_centers = torch.randn(new_num_classes, self.feature_dim).to(self.centers.device)
        new_centers[:self.num_classes] = self.centers.data
        
        self.num_classes = new_num_classes
        self.centers = nn.Parameter(new_centers)

class OpenICLLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.5):
        super(OpenICLLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.contrastive_loss = CenterContrastiveLoss(num_classes, feature_dim)
        self.alpha = alpha
        
    def forward(self, logits, contrast_features, labels):
        # Cross Entropy for Classification Path
        loss_ce = self.ce_loss(logits, labels)
        
        # Contrastive Loss for Contrast Path
        loss_con = self.contrastive_loss(contrast_features, labels)
        
        return loss_ce + self.alpha * loss_con
