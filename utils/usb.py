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

    def discover_new_classes(self, n_clusters=None, max_k=15, random_state=42):
        """
        Cluster the stored unknown features to discover new classes.
        Returns pseudo-labels and the number of discovered clusters.
        """
        features_np = np.array(self.features)
        
        if len(self.features) < 10:
            return np.zeros(len(self.features), dtype=int), 1
            
        # Dynamic K-Means Optimization using Silhouette Score
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        best_score = -float('inf')
        best_labels = None
        best_n_clusters = 1
        
        # Cap max_k to avoid searching beyond the data size
        upper = min(max_k, len(features_np) - 1)
        
        for k in range(2, upper + 1):
            # We use n_init=10 to avoid warnings and ensure stability
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(features_np)
            
            sil_score = silhouette_score(features_np, labels)
            
            if sil_score > best_score:
                best_score = sil_score
                best_labels = labels
                best_n_clusters = k
                        
        # Hypothesis test: Does the data strongly support multiple clusters?
        # A score < 0.15 indicates overlapping or weak artificial clusters.
        if best_labels is None or best_score < 0.15:
            print("Dynamically Selected K-Means: 1 cluster discovered (No strong multi-cluster structure)")
            return np.zeros(len(self.features), dtype=int), 1
        else:
            print(f"Dynamically Selected K-Means: {best_n_clusters} clusters discovered (Score: {best_score:.4f})")
            
        # K-Means assigns all points to a cluster (no noise points to filter)
        return best_labels, best_n_clusters

    def clear(self):
        self.signals = []
        self.features = []
