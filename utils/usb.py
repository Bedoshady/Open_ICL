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
        
        # Dynamic HDBSCAN Optimization using Silhouette Score
        # HDBSCAN has no eps parameter, making it far more robust to varying densities.
        # We only need to search over min_cluster_size.
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            raise ImportError("HDBSCAN not found. Please install scikit-learn>=1.3.0 or run `pip install hdbscan`.")
                
        min_cluster_sizes = [50, 100, 200, 300]
        
        best_score = -float('inf')
        best_labels = None
        best_n_clusters = 1
        
        # Track the best 1-cluster solution separately
        min_1cluster_noise = float('inf')
        best_1cluster_labels = None
        
        from sklearn.metrics import silhouette_score
        
        for m in min_cluster_sizes:
            hdbscan_model = HDBSCAN(min_cluster_size=m)
            labels = hdbscan_model.fit_predict(features_np)
            
            mask = labels != -1
            valid_count = mask.sum()
            noise_fraction = 1.0 - (valid_count / len(labels))
            
            unique_clusters = set(labels[mask])
            n_clusters = len(unique_clusters)
            
            if n_clusters == 1 and noise_fraction < min_1cluster_noise:
                min_1cluster_noise = noise_fraction
                best_1cluster_labels = labels
            
            # Require at least 2 clusters to compute silhouette, and keep at least 40% of the data
            elif n_clusters >= 2 and noise_fraction < 0.60:
                sil_score = silhouette_score(features_np[mask], labels[mask])
                # Penalize high noise to encourage retaining data for incremental learning
                adjusted_score = sil_score * (1.0 - noise_fraction)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_labels = labels
                    best_n_clusters = n_clusters
                        
        # Hypothesis test: Does the data strongly support multiple clusters?
        # A score < 0.15 indicates overlapping or weak artificial clusters.
        if best_labels is None or best_score < 0.15:
            if best_1cluster_labels is not None:
                print(f"Dynamically Selected HDBSCAN: 1 cluster discovered (Min Noise: {min_1cluster_noise:.2%})")
                best_labels = best_1cluster_labels
                best_n_clusters = 1
            else:
                print("HDBSCAN failed to find dense regions. Falling back to single class.")
                return np.zeros(len(self.features), dtype=int), 1
        else:
            print(f"Dynamically Selected HDBSCAN: {best_n_clusters} clusters discovered (Score: {best_score:.4f})")
        
        # Filter out noise points
        mask = best_labels != -1
        self.signals = [self.signals[i] for i in range(len(self.signals)) if mask[i]]
        self.features = [self.features[i] for i in range(len(self.features)) if mask[i]]
        
        filtered_labels = best_labels[mask]
        
        return filtered_labels, best_n_clusters

    def clear(self):
        self.signals = []
        self.features = []
