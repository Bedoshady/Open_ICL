import numpy as np
from sklearn.cluster import KMeans
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

    def discover_new_classes(self, n_clusters=None, max_k=15, random_state=42):
        """
        Cluster the stored unknown features to discover new classes.

        If n_clusters is provided, KMeans is run with that fixed k.
        Otherwise, the optimal k is selected by sweeping k in [2, max_k]
        and choosing the k that maximises the Silhouette Score.

        Returns pseudo-labels and the number of discovered clusters.
        """
        features_np = np.array(self.features)

        if len(self.features) < 10:
            return np.zeros(len(self.features), dtype=int), 1

        # --- Fixed k mode ---
        if n_clusters is not None:
            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
            best_labels = km.fit_predict(features_np)
            return best_labels, n_clusters

        # --- Silhouette-guided k search ---
        best_score = -1.0
        best_labels = None
        best_n_clusters = 1

        # Cap max_k to avoid searching beyond the data size
        upper = min(max_k, len(features_np) - 1)

        for k in range(2, upper + 1):
            km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
            labels = km.fit_predict(features_np)
            try:
                score = silhouette_score(features_np, labels)
            except ValueError:
                continue
            print(f"  k={k:>2d}  silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_n_clusters = k

        if best_labels is None:
            # Fallback: no valid clustering found
            return np.zeros(len(self.features), dtype=int), 1

        print(f"Selected k={best_n_clusters} (silhouette={best_score:.4f})")
        return best_labels, best_n_clusters

    def clear(self):
        self.signals = []
        self.features = []
