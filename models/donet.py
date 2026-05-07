import torch
import torch.nn as nn
import torch.nn.functional as F

class DONet(nn.Module):
    """
    Dual-Path 1-D Network (DONet)
    """
    def __init__(self, num_known_classes, feature_dim=128):
        super(DONet, self).__init__()
        
        # Shared Feature Extractor (1D CNN for time-series IQ data)
        self.shared_conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Classification Path (CLP)
        self.clp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_known_classes)
        )
        
        # Contrast Path (COP)
        self.cop = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Semantic Feature Centers (SFCs) integrated into the model
        self.sfcs = nn.Parameter(torch.randn(num_known_classes, feature_dim))
        self.num_classes = num_known_classes
        self.feature_dim = feature_dim
        
    def forward(self, x):
        # x shape: [Batch, 2, 128]
        features = self.shared_conv(x)
        features = features.view(features.size(0), -1) # Flatten -> [Batch, 256]
        
        logits = self.clp(features)
        contrast_features = self.cop(features)
        
        # Normalize contrast features and SFCs for distance computing
        contrast_features = F.normalize(contrast_features, p=2, dim=1)
        sfc_normalized = F.normalize(self.sfcs, p=2, dim=1)
        
        # DM: Calculate Euclidean distances between signals and all known class centers
        distances = torch.cdist(contrast_features, sfc_normalized, p=2)
        
        return logits, contrast_features, distances

    def update_num_classes(self, new_num_classes):
        """
        Dynamically expand the classification path and SFCs for incremental learning
        without forgetting old classes.
        """
        old_clp_weight = self.clp[-1].weight.data
        old_clp_bias = self.clp[-1].bias.data
        old_num_classes = old_clp_weight.size(0)
        
        if new_num_classes <= old_num_classes:
            return
            
        in_features = self.clp[-1].in_features
        new_layer = nn.Linear(in_features, new_num_classes).to(old_clp_weight.device)
        
        # Keep old weights to retain memory of previous classes
        new_layer.weight.data[:old_num_classes] = old_clp_weight
        new_layer.bias.data[:old_num_classes] = old_clp_bias
        self.clp[-1] = new_layer
        
        # Update SFCs
        new_sfcs = torch.randn(new_num_classes, self.feature_dim).to(self.sfcs.device)
        new_sfcs[:old_num_classes] = self.sfcs.data
        self.sfcs = nn.Parameter(new_sfcs)
        
        self.num_classes = new_num_classes
