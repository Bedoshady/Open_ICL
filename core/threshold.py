import torch

class DynamicAdaptiveThreshold:
    """
    Dynamic Adaptive Threshold (DAT)
    Calculates the distance threshold for open-set recognition based on the 
    distribution of distances from known class samples to their respective SFCs.
    """
    def __init__(self, alpha=0.95):
        # alpha is the confidence level
        self.alpha = alpha
        self.threshold = {}
        self.epoch_true_dists = []
        self.epoch_labels = []
        
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
        
        # Accumulate the true distances for the current epoch
        self.epoch_true_dists.append(true_dists.detach().cpu())
        self.epoch_labels.append(valid_labels.detach().cpu())
        
    def compute_epoch_threshold(self):
        if len(self.epoch_true_dists) > 0:
            all_true_dists = torch.cat(self.epoch_true_dists, dim=0)
            all_labels = torch.cat(self.epoch_labels, dim=0)
            
            unique_classes = torch.unique(all_labels)
            
            for c in unique_classes:
                c_item = c.item()
                class_dists = all_true_dists[all_labels == c]
                if len(class_dists) > 1:
                    mu = torch.mean(class_dists)
                    sigma = torch.std(class_dists)
                    self.threshold[c_item] = (mu + 3 * sigma).item()
                
        # Clear accumulated distances for the next epoch
        self.epoch_true_dists = []
        self.epoch_labels = []
                
    def get_threshold(self):
        return self.threshold
