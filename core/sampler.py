import torch
import numpy as np
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    P-K Batch Sampler for Metric Learning.
    
    Each batch contains exactly P classes with K samples per class,
    guaranteeing that every anchor has at least (K-1) same-class
    partners for hard-positive mining.
    
    Args:
        labels: array-like of integer labels for the entire dataset.
        P: number of classes per batch.
        K: number of samples per class per batch.
    """
    def __init__(self, labels, P=8, K=4):
        super().__init__()
        self.P = P
        self.K = K
        self.batch_size = P * K

        # Build a mapping: class_label -> list of dataset indices
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            label = int(label)
            if label < 0:
                continue  # skip unknowns (label -1)
            self.label_to_indices.setdefault(label, [])
            self.label_to_indices[label].append(idx)

        # Only keep classes that have at least K samples
        self.valid_classes = [
            cls for cls, indices in self.label_to_indices.items()
            if len(indices) >= self.K
        ]

        if len(self.valid_classes) < self.P:
            raise ValueError(
                f"PKSampler requires at least P={self.P} classes with >= K={self.K} "
                f"samples each, but only {len(self.valid_classes)} classes qualify."
            )

        # Precompute total number of batches per epoch
        # Each epoch iterates until every sample in every class has been
        # visited at least once (approximately).
        total_samples = sum(len(self.label_to_indices[c]) for c in self.valid_classes)
        self.num_batches = max(1, total_samples // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            # Pick P random classes
            chosen_classes = np.random.choice(
                self.valid_classes, size=self.P, replace=False
            )
            batch_indices = []
            for cls in chosen_classes:
                pool = self.label_to_indices[cls]
                # Sample K indices with replacement if pool < K (shouldn't happen
                # because we filtered, but just in case)
                replace = len(pool) < self.K
                chosen = np.random.choice(pool, size=self.K, replace=replace)
                batch_indices.extend(chosen.tolist())
            yield batch_indices

    def __len__(self):
        return self.num_batches
