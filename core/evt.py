import torch
import numpy as np
from scipy.stats import weibull_min

class DynamicEVT:
    """
    Dynamic Extreme Value Theory (EVT) for Open-Set Recognition.
    Fits a Weibull distribution to the tail distances of each known class.
    """
    def __init__(self, tail_size=0.05):
        self.tail_size = tail_size
        self.models = {} # dict mapping class_idx -> (shape, loc, scale)
        self.centroids = {} # dict mapping class_idx -> centroid tensor

    def fit(self, features, labels):
        """
        Fits EVT Weibull models dynamically.
        Args:
            features: Tensor of shape (N, feature_dim)
            labels: Tensor of shape (N,)
        """
        features_np = features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        unique_classes = np.unique(labels_np)
        
        for cls in unique_classes:
            if cls < 0:
                continue # Skip unknowns
                
            cls_features = features_np[labels_np == cls]
            if len(cls_features) < 10:
                continue # Not enough samples to fit
                
            # Compute centroid
            centroid = np.mean(cls_features, axis=0)
            self.centroids[cls] = torch.tensor(centroid, device=features.device)
            
            # Compute distances to centroid
            dists = np.linalg.norm(cls_features - centroid, axis=1)
            
            # Extract tail (largest distances)
            tail_count = max(5, int(len(dists) * self.tail_size))
            tail_dists = np.sort(dists)[-tail_count:]
            
            # Fit Weibull using scipy
            # We fix loc to 0 because distances are inherently >= 0
            shape, loc, scale = weibull_min.fit(tail_dists, floc=0)
            
            # Safeguard to prevent division by zero or NaN values under model collapse
            scale = max(scale, 1e-4)
            shape = max(shape, 0.1)
            
            self.models[cls] = (shape, loc, scale)
            
    def predict_prob(self, features):
        """
        Predicts the probability of belonging to the known classes.
        Runs entirely on the GPU for maximum speed (saving GPU/CPU transfers and scipy overhead).
        Args:
            features: Tensor of shape (N, feature_dim) on GPU/CPU
        Returns:
            probs: Tensor of shape (N,), the max probability of belonging to any known class.
            pred_classes: Tensor of shape (N,), the predicted known class.
        """
        if not self.centroids or not self.models:
            # If not fitted yet, return dummy probabilities (all 1.0)
            return torch.ones(len(features), device=features.device), torch.zeros(len(features), dtype=torch.long, device=features.device)
            
        N = len(features)
        num_classes = max(self.centroids.keys()) + 1
        
        probs_matrix = torch.zeros((N, num_classes), device=features.device)
        
        for cls, centroid in self.centroids.items():
            centroid = centroid.to(features.device)
            
            # Compute Euclidean distance on the device
            dists = torch.norm(features - centroid, p=2, dim=1)
            
            if cls in self.models:
                shape, loc, scale = self.models[cls]
                # EVT probability: 1 - CDF of Weibull, which is exp(-(d/scale)**shape) for loc=0
                # Computed entirely on GPU
                probs = torch.exp(- (dists / scale) ** shape)
            else:
                probs = torch.ones(N, device=features.device)
                
            probs_matrix[:, cls] = probs
            
        max_probs, pred_classes = torch.max(probs_matrix, dim=1)
        
        return max_probs, pred_classes
