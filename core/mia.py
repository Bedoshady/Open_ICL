class MovingIntersectionAlgorithm:
    """
    Moving Intersection Algorithm (MIA)
    Filters out noise and unreliable signals from being classified as 'unknown'
    by ensuring they exhibit high confidence in their 'unknown' status.
    
    In a sequential data stream, this would track a signal over time. 
    For dataset batches, we use a strict confidence multiplier on the DAT threshold.
    """
    def __init__(self, confidence_threshold=0.8):
        self.confidence_threshold = confidence_threshold
        
    def filter_unknowns(self, distances_to_sfcs, dat_threshold):
        """
        distances_to_sfcs: [Batch_size, num_known_classes]
        dat_threshold: scalar threshold from DAT
        
        Returns a boolean mask of shape [Batch_size] where True means reliable unknown.
        """
        # Find the minimum distance to any known class SFC
        min_dists, _ = distances_to_sfcs.min(dim=1)
        
        # A reliable unknown should be significantly farther than the DAT threshold
        reliable_unknown_mask = min_dists > (dat_threshold * (1 + self.confidence_threshold))
        
        return reliable_unknown_mask
