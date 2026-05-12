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

    def discover_new_classes(self, n_clusters=None):
        """
        Cluster the stored unknown features to discover new classes.
        If n_clusters is None, it uses the Silhouette score to find the optimal number dynamically.
        Returns pseudo-labels and the optimal number of clusters.
        """
        features_np = np.array(self.features)
        
        if n_clusters is not None:
            if len(self.features) < n_clusters:
                return None, n_clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            pseudo_labels = kmeans.fit_predict(features_np)
            return pseudo_labels, n_clusters
            
        from sklearn.metrics import silhouette_score
        
        best_score = -1
        best_k = 2
        best_labels = None
        
        # Test clusters from 2 to 10
        max_k = min(10, len(self.features) - 1)
        if max_k < 2:
            return np.zeros(len(self.features), dtype=int), 1
            
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(features_np)
            score = silhouette_score(features_np, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
        return best_labels, best_k

    def clear(self):
        self.signals = []
        self.features = []
