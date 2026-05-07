import numpy as np
from sklearn.cluster import KMeans

class UnknownSignalBank:
    """
    Unknown Signal Bank (USB)
    Stores high-confidence unknown signals to be used later for incremental learning
    of new modulation types.
    """
    def __init__(self, max_size=10000):
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

    def discover_new_classes(self, n_clusters=2):
        """
        Cluster the stored unknown features to discover new classes.
        Returns pseudo-labels for the stored signals.
        """
        if len(self.features) < n_clusters:
            return None
            
        features_np = np.array(self.features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        pseudo_labels = kmeans.fit_predict(features_np)
        
        return pseudo_labels
        
    def clear(self):
        self.signals = []
        self.features = []
