import torch

class DynamicAdaptiveThreshold:
    """
    Dynamic Adaptive Threshold (DAT)
    Calculates the distance threshold for open-set recognition based on the 
    distribution of distances from known class samples to their respective SFCs.
    """
    def __init__(self, alpha=0.95):
        # alpha is the confidence level (e.g., 95th percentile of distances)
        self.alpha = alpha
        self.threshold = 0.0
        
    def update(self, distances, labels):
        """
        distances: [Batch_size, num_known_classes]
        labels: [Batch_size]
        """
        # Only consider known classes
        valid_idx = labels >= 0
        if not valid_idx.any():
            return
            
        valid_dists = distances[valid_idx]
        valid_labels = labels[valid_idx]
        
        # Get distances to the true class centers
        labels_expanded = valid_labels.view(-1, 1)
        true_dists = valid_dists.gather(1, labels_expanded).squeeze(-1)
        
        # Set threshold to the alpha-percentile of true distances
        if len(true_dists) > 0:
            q = torch.quantile(true_dists, self.alpha)
            # Smooth moving average update
            if self.threshold == 0.0:
                self.threshold = q.item()
            else:
                self.threshold = 0.9 * self.threshold + 0.1 * q.item()
                
    def get_threshold(self):
        return self.threshold
