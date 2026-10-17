import numpy as np
from sklearn.metrics import silhouette_score

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

    def discover_new_classes(self, min_cluster_size=None, min_samples=None):
        """
        Cluster the stored unknown features to discover new classes using HDBSCAN.

        If min_cluster_size is provided, HDBSCAN is run with that fixed parameter.
        Otherwise, min_cluster_size is dynamically selected using Silhouette Score 
        (penalized by noise fraction).

        Returns pseudo-labels and the number of discovered clusters.
        """
        features_np = np.array(self.features)
        n_samples = len(features_np)
        
        # If fewer than 51 samples, cannot evaluate even the minimum cluster size (50)
        if n_samples <= 50:
            return np.zeros(n_samples, dtype=int), 1

        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            raise ImportError("HDBSCAN not found. Please install scikit-learn>=1.3.0 or run `pip install hdbscan`.")

        # --- Fixed parameter mode ---
        if min_cluster_size is not None:
            if n_samples < min_cluster_size:
                return np.zeros(n_samples, dtype=int), 1
            hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = hdbscan_model.fit_predict(features_np)
            mask = labels != -1
            if mask.sum() == 0:
                print("HDBSCAN labeled all points as noise. Falling back to single class.")
                return np.zeros(n_samples, dtype=int), 1

            self.signals = [self.signals[i] for i in range(len(self.signals)) if mask[i]]
            self.features = [self.features[i] for i in range(len(self.features)) if mask[i]]
            filtered_labels = labels[mask]
            n_clusters = len(set(filtered_labels))
            print(f"Fixed HDBSCAN (min_cluster_size={min_cluster_size}): {n_clusters} clusters discovered.")
            return filtered_labels, n_clusters

        # --- Dynamic Silhouette-guided search ---
        candidate_sizes = [50, 100, 200, 300]
        min_cluster_sizes = [m for m in candidate_sizes if m < n_samples]
        
        best_score = -float('inf')
        best_labels = None
        best_n_clusters = 1
        
        # Track the best 1-cluster solution separately
        min_1cluster_noise = float('inf')
        best_1cluster_labels = None
        
        from sklearn.metrics import silhouette_score
        
        for m in min_cluster_sizes:
            hdbscan_model = HDBSCAN(min_cluster_size=m, min_samples=min_samples)
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
