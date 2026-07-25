import torch
class MovingIntersectionAlgorithm:
    """
    Moving Intersection Algorithm (MIA)
    Records the index of candidate unknown signals over epochs.
    Signals that are detected as candidates for L consecutive times 
    are considered reliable and added to the USB.
    """
    def __init__(self, L=5, confidence_threshold=0.5):
        self.L = L
        self.confidence_threshold = confidence_threshold
        # Stores the set of candidate indices for the last L epochs
        self.epoch_candidates = []
        # Keep track of indices already yielded to prevent adding them multiple times
        self.already_discovered = set()
        
    def detect_candidates(self, distances_to_sfcs, dat_threshold, batch_indices):
        """
        Identify candidates in the current batch and return their global indices.
        distances_to_sfcs: [Batch_size, num_known_classes]
        dat_threshold: scalar threshold from DAT
        batch_indices: [Batch_size] global indices of the signals
        """
        min_dists, closest_class = distances_to_sfcs.min(dim=1)
        # strict confidence is no longer needed since we do intersection, but we keep it for extra reliability
        if isinstance(dat_threshold, dict):
            thresholds = torch.tensor([dat_threshold.get(c.item(), float('inf')) for c in closest_class], device=min_dists.device)
            candidate_mask = min_dists > thresholds
        else:
            candidate_mask = min_dists > dat_threshold
        candidate_indices = batch_indices[candidate_mask]
        return set(candidate_indices.cpu().numpy())

    def update_epoch(self, current_epoch_candidates):
        """
        Called at the end of an epoch to add the new candidates.
        Returns the intersection of the last L epochs (excluding those already yielded).
        """
        self.epoch_candidates.append(current_epoch_candidates)
        if len(self.epoch_candidates) > self.L:
            self.epoch_candidates.pop(0)
            
        if len(self.epoch_candidates) == self.L:
            # Calculate intersection
            reliable_indices = set.intersection(*self.epoch_candidates)
            
            # Filter out indices we've already discovered and yielded
            new_reliable = reliable_indices - self.already_discovered
            self.already_discovered.update(new_reliable)
            
            return new_reliable
        else:
            return set()

