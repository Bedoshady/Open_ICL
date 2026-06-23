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

"BatchAllTripleLoss By https://github.com/h1yuol/pytorch-triplet-loss"
class BatchAllTripletLoss(nn.Module):
    """
    Online Batch-All Triplet Loss for Metric Learning.
    Enumerates all valid (anchor, positive, negative) triplets in the batch,
    computes softplus(d(a,p) - d(a,n)) for each, and averages only over
    non-zero-loss triplets (semi-hard + hard).
    """
    def __init__(self, margin=1.0, squared=False):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def pairwise_distances(self, embeddings, squared=False):
        """
        ||a-b||^2 = |a|^2 - 2*<a,b> + |b|^2
        """
        # get dot product (batch_size, batch_size)
        dot_product = embeddings.mm(embeddings.t())

        # a vector
        square_sum = dot_product.diag()

        distances = square_sum.unsqueeze(1) - 2*dot_product + square_sum.unsqueeze(0)

        distances = distances.clamp(min=0)

        if not squared:
            epsilon=1e-16
            mask = torch.eq(distances, 0).float()
            distances = distances + mask * epsilon
            distances = torch.sqrt(distances)
            distances = distances * (1-mask)

        return distances

    def get_valid_triplets_mask(self, labels):
        """
        To be valid, a triplet (a,p,n) has to satisfy:
            - a,p,n are distinct embeddings
            - a and p have the same label, while a and n have different label
        """
        indices_equal = torch.eye(labels.size(0)).byte().to(labels.device)
        indices_not_equal = ~indices_equal
        i_ne_j = indices_not_equal.unsqueeze(2)
        i_ne_k = indices_not_equal.unsqueeze(1)
        j_ne_k = indices_not_equal.unsqueeze(0)
        distinct_indices = i_ne_j & i_ne_k & j_ne_k

        label_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).to(labels.device)
        i_eq_j = label_equal.unsqueeze(2)
        i_eq_k = label_equal.unsqueeze(1)
        i_ne_k = ~i_eq_k
        valid_labels = i_eq_j & i_ne_k

        mask = distinct_indices & valid_labels
        return mask

    def batch_all_triplet_loss(self, labels, embeddings, margin, squared=False):
        """
        get triplet loss for all valid triplets and average over those triplets whose loss is positive.
        """

        distances = self.pairwise_distances(embeddings, squared=squared)

        anchor_positive_dist = distances.unsqueeze(2)
        anchor_negative_dist = distances.unsqueeze(1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

        # get a 3D mask to filter out invalid triplets
        mask = self.get_valid_triplets_mask(labels)

        triplet_loss = triplet_loss * mask.float()
        triplet_loss.clamp_(min=0)

        # count the number of positive triplets
        epsilon = 1e-16
        num_positive_triplets = (triplet_loss > 0).float().sum()
        num_valid_triplets = mask.float().sum()
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + epsilon)

        triplet_loss = triplet_loss.sum() / (num_positive_triplets + epsilon)

        return triplet_loss

    def forward(self, embeddings, labels, margin_override=None):
        return self.batch_all_triplet_loss(labels, embeddings, self.margin, self.squared)
        
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
