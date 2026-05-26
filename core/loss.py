import torch
import torch.nn as nn
import torch.nn.functional as F


def get_margin(epoch, start_margin=0.5, end_margin=1.0, ramp_epochs=10):
    """
    Linearly ramp the triplet-loss margin from ``start_margin`` to
    ``end_margin`` over the first ``ramp_epochs`` epochs.

    A smaller margin at the start allows easier optimisation
    (the model only needs to push negatives slightly away);
    a larger margin later forces crisper separation, helping
    the loss continue to decrease past the initial plateau.
    """
    if epoch >= ramp_epochs:
        return end_margin
    t = epoch / ramp_epochs
    return start_margin + (end_margin - start_margin) * t


class BatchHardTripletLoss(nn.Module):
    """
    Online Batch-Hard Triplet Loss for Metric Learning.
    For each anchor in the batch, it finds the hardest positive (furthest sample of the same class)
    and the hardest negative (closest sample of a different class).
    """
    def __init__(self, margin=1.0, squared=False):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels, margin_override=None):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embed_dim)
            labels: Tensor of shape (batch_size,)
            margin_override: If given, overrides self.margin for this call
                             (used by the margin curriculum).
        """
        margin = margin_override if margin_override is not None else self.margin

        # Ignore unknown classes (label -1) during known-class training
        valid_idx = labels >= 0
        embeddings = embeddings[valid_idx]
        labels = labels[valid_idx]

        if len(embeddings) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Normalize embeddings to lie on a hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise distance matrix
        # d(x,y)^2 = ||x||^2 - 2<x,y> + ||y||^2
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        dist_sq = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        dist_sq = torch.clamp(dist_sq, min=0.0) # Ensure non-negative

        if self.squared:
            dist = dist_sq
        else:
            dist = torch.sqrt(dist_sq + 1e-8) # Add epsilon for numerical stability

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Don't compare an anchor to itself
        eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        pos_mask = labels_equal & ~eye
        neg_mask = ~labels_equal

        # Hardest positive for each anchor (maximum distance to same class)
        pos_dist_masked = dist * pos_mask.float()
        hardest_positive_dist, _ = pos_dist_masked.max(dim=1)

        # Hardest negative for each anchor (minimum distance to different class)
        # Add a large number to non-negative distances so they are ignored by min()
        max_dist, _ = dist.max(dim=1, keepdim=True)
        neg_dist_masked = dist + max_dist * (~neg_mask).float()
        hardest_negative_dist, _ = neg_dist_masked.min(dim=1)

        # Triplet loss: max(0, d(A, P) - d(A, N) + margin)
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)

        # Only compute mean over anchors that have both valid positives and negatives
        valid_triplets = (hardest_positive_dist > 0) & (hardest_negative_dist < max_dist.squeeze(-1))
        
        if not valid_triplets.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return triplet_loss[valid_triplets].mean()


class CenterLoss(nn.Module):
    """
    Center Loss — pulls embeddings of each class toward a learnable centroid.

    Reduces intra-class variance and complements the Triplet Loss which
    only controls *relative* distances.

    Reference: Wen et al., "A Discriminative Feature Learning Approach
    for Deep Face Recognition", ECCV 2016.
    """
    def __init__(self, num_classes, feat_dim, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Learnable class centres
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim, device=device))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (N, feat_dim)
            labels: (N,) — only non-negative labels are used.
        """
        valid = labels >= 0
        embeddings = embeddings[valid]
        labels = labels[valid]

        if len(embeddings) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        batch_centers = self.centers[labels]
        loss = (embeddings - batch_centers).pow(2).sum(dim=1).mean()
        return loss
