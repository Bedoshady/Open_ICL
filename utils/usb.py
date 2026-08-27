import numpy as np
from sklearn.cluster import DBSCAN

class UnknownSignalBank:
    """
    Unknown Signal Bank (USB)
    Stores high-confidence unknown signals to be used later for incremental learning
    of new modulation types.
    """
    def __init__(self, max_size=40000):
        self.max_size = max_size
        
        # Buffers
        self.signals = []
        self.features = []
        
    def add_signals(self, signals, features):
        """
        Add new unknown signals to the bank.
        """
        for sig, feat in zip(signals, features):
            if len(self.signals) >= self.max_size:
                # FIFO replacement
                self.signals.pop(0)
                self.features.pop(0)
                
            self.signals.append(sig.cpu().numpy())
            self.features.append(feat.detach().cpu().numpy())
            
    def get_all(self):
        if not self.signals:
            return None, None
        return np.array(self.signals), np.array(self.features)

    def discover_new_classes(self, n_clusters=None):
        """
        Cluster the stored unknown features to discover new classes using default DBSCAN.
        Returns pseudo-labels and the number of discovered clusters.
        """
        features_np = np.array(self.features)
        
        if len(self.features) < 10:
            return np.zeros(len(self.features), dtype=int), 1
            
        # To detect fewer clusters:
        # 1. INCREASE eps (points farther apart will merge into the same cluster)
        # 2. INCREASE min_samples (smaller clusters will be discarded as noise)
        dbscan = DBSCAN(eps=0.3, min_samples=200) 
        best_labels = dbscan.fit_predict(features_np)
        
        # Check number of valid clusters (excluding noise)
        mask = best_labels != -1
        unique_clusters = set(best_labels[mask])
        best_n_clusters = len(unique_clusters)
                        
        if best_n_clusters <= 1:
            # Fallback if no clustering found multiple clusters
            return np.zeros(len(self.features), dtype=int), 1
            
        # Filter out noise points
        self.signals = [self.signals[i] for i in range(len(self.signals)) if mask[i]]
        self.features = [self.features[i] for i in range(len(self.features)) if mask[i]]
        
        filtered_labels = best_labels[mask]
        
        return filtered_labels, best_n_clusters

    def clear(self):
        self.signals = []
        self.features = []
