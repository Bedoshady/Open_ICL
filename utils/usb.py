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
        Ablation: No clustering. Assign all unknown features to a single novel class.
        Returns pseudo-labels (all 0) and the number of clusters (1).
        """
        if len(self.features) == 0:
            return np.zeros(0, dtype=int), 0
            
        pseudo_labels = np.zeros(len(self.features), dtype=int)
        n_clusters = 1
            
        return pseudo_labels, n_clusters

    def clear(self):
        self.signals = []
        self.features = []
