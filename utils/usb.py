import numpy as np
from sklearn.cluster import KMeans

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

    def discover_new_classes(self, eps=0.5, min_samples=10):
        """
        Cluster the stored unknown features to discover new classes using DBSCAN.
        Returns pseudo-labels (including -1 for noise) and the optimal number of clusters.
        """
        features_np = np.array(self.features)
        
        from sklearn.cluster import DBSCAN
        
        if len(self.features) < min_samples:
            return np.zeros(len(self.features), dtype=int), 0
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        pseudo_labels = dbscan.fit_predict(features_np)
        
        # Calculate number of valid clusters (excluding noise label -1)
        if -1 in pseudo_labels:
            n_clusters = len(set(pseudo_labels)) - 1
        else:
            n_clusters = len(set(pseudo_labels))
            
        return pseudo_labels, n_clusters

    def clear(self):
        self.signals = []
        self.features = []
