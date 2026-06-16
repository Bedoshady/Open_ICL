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
    def __init__(self, margin=1.0, squared=False, use_soft_margin=True):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        self.use_soft_margin = use_soft_margin

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

        # Compute pairwise distance matrix using cdist for numerical stability on unnormalized features
        dist = torch.cdist(embeddings, embeddings, p=2)
        
        if self.squared:
            dist = dist ** 2

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

        # Triplet loss
        if self.use_soft_margin:
            # Soft-margin: ln(1 + exp(d(A, P) - d(A, N)))
            triplet_loss = F.softplus(hardest_positive_dist - hardest_negative_dist)
        else:
            # Hard-margin: max(0, d(A, P) - d(A, N) + margin)
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)

        # Only compute mean over anchors that have both valid positives and negatives in the batch
        # We determine this using the masks (based on labels) rather than distances to avoid the collapse trap.
        has_pos = pos_mask.any(dim=1)
        has_neg = neg_mask.any(dim=1)
        valid_triplets = has_pos & has_neg
        
        if not valid_triplets.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return triplet_loss[valid_triplets].mean()


class BatchAllTripletLoss(nn.Module):
    """
    Online Batch-All Triplet Loss for Metric Learning.
    Enumerates all valid (anchor, positive, negative) triplets in the batch,
    computes softplus(d(a,p) - d(a,n)) for each, and averages only over
    non-zero-loss triplets (semi-hard + hard).
    """
    def __init__(self, squared=False):
        super(BatchAllTripletLoss, self).__init__()
        self.squared = squared

    def forward(self, embeddings, labels, margin_override=None):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embed_dim)
            labels: Tensor of shape (batch_size,)
            margin_override: Ignored, retained for API compatibility with curriculum.
        """
        # Ignore unknown classes (label -1) during known-class training
        valid_idx = labels >= 0
        embeddings = embeddings[valid_idx]
        labels = labels[valid_idx]

        if len(embeddings) < 3:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute pairwise distance matrix
        dist = torch.cdist(embeddings, embeddings, p=2)
        if self.squared:
            dist = dist ** 2

        # Broadcast pairwise distances into a 3D tensor of shape (N, N, N)
        # triplet_dist[a, p, n] = d(a, p) - d(a, n)
        d_ap = dist.unsqueeze(2)  # Shape: (N, N, 1)
        d_an = dist.unsqueeze(1)  # Shape: (N, 1, N)
        triplet_dist = d_ap - d_an  # Shape: (N, N, N)

        # Build the 3D valid triplet mask
        # 1. Anchor and Positive must have the same label, Anchor and Negative must have different labels
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1) # (N, N)
        
        mask_ap = labels_equal.unsqueeze(2)      # (N, N, 1) -> True where label(a) == label(p)
        mask_an = (~labels_equal).unsqueeze(1)    # (N, 1, N) -> True where label(a) != label(n)

        # 2. Exclude self-comparisons (Anchor != Positive, Anchor != Negative)
        eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        mask_a_not_p = (~eye).unsqueeze(2)        # (N, N, 1)
        mask_a_not_n = (~eye).unsqueeze(1)        # (N, 1, N)

        # Combine into a single valid triplet mask of shape (N, N, N)
        valid_triplet_mask = mask_ap & mask_an & mask_a_not_p & mask_a_not_n

        if not valid_triplet_mask.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Extract distances for valid triplets
        valid_distances = triplet_dist[valid_triplet_mask]

        # Compute loss using softplus: ln(1 + exp(d(a,p) - d(a,n)))
        loss_values = F.softplus(valid_distances)

        # Filter for non-zero-loss triplets (semi-hard + hard)
        # In practice, softplus(x) > 0 for all real numbers, but we mathematically 
        # isolate positive signals or non-negligible losses based on standard thresholds.
        positive_loss_mask = loss_values > 1e-6
        active_losses = loss_values[positive_loss_mask]

        if active_losses.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return active_losses.mean()
        
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

class BCEContrastLoss(nn.Module):
    """
    Binary Cross Entropy Contrastive Loss as defined in the Open-ICL paper (Equation 6).
    Pairs signal features with Semantic Feature Centers (SFCs).
    A pair is 'positive' (label 0) if they belong to the same class,
    and 'negative' (label 1) otherwise.
    """
    def __init__(self):
        super(BCEContrastLoss, self).__init__()

    def forward(self, contrast_probs, labels, num_classes):
        """
        Args:
            contrast_probs: Tensor of shape (batch_size, num_classes) representing \tilde{y}_i^{k*}
                            This is the novelty probability output of the DM module.
            labels: Tensor of shape (batch_size,)
            num_classes: int, the number of known classes (K) corresponding to the SFCs.
        """
        # Filter valid labels (ignore -1)
        valid_idx = labels >= 0
        if not valid_idx.any():
            return torch.tensor(0.0, device=contrast_probs.device, requires_grad=True)
            
        contrast_probs = contrast_probs[valid_idx]
        labels = labels[valid_idx]
        
        batch_size = labels.size(0)
        # Initialize all contrast labels to 1 (negative pairs)
        contrast_labels = torch.ones((batch_size, num_classes), device=labels.device)
        
        # For known samples (label < num_classes), set the matching SFC label to 0 (positive pair)
        known_mask = labels < num_classes
        if known_mask.any():
            known_labels = labels[known_mask]
            # Scatter 0 into the correct class index
            zeros = torch.zeros((known_labels.size(0), 1), device=labels.device)
            contrast_labels[known_mask] = contrast_labels[known_mask].scatter(1, known_labels.unsqueeze(1), zeros)
            
        # Compute standard Binary Cross Entropy
        # PyTorch BCE includes the negative sign inherently missing from the paper's simplified equation
        loss = F.binary_cross_entropy(contrast_probs, contrast_labels, reduction='mean')
        return loss
