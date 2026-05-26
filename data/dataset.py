import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.sampler import PKSampler

class RadioMLDataset(Dataset):
    """
    Dataset loader for RadioML 2016.10a / 2018.01a.
    """
    def __init__(self, file_path, known_classes=None, unknown_classes=None, min_snr=0):
        super().__init__()
        # Using latin1 encoding is required for loading older Python 2 pickle files (like RML2016.10a_dict.pkl)
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        self.data = []
        self.labels = []
        self.snrs = []
        
        # Extract unique modulations
        all_mods = sorted(list(set([k[0] for k in data.keys()])))
        
        self.known_classes = known_classes if known_classes else all_mods
        self.unknown_classes = unknown_classes if unknown_classes else []
        
        self.class_to_idx = {mod: idx for idx, mod in enumerate(self.known_classes)}
        
        for (mod, snr), samples in data.items():
            if snr < min_snr:
                continue
                
            if mod in self.known_classes:
                label = self.class_to_idx[mod]
            elif mod in self.unknown_classes:
                label = -1
            else:
                continue
                
            self.data.append(samples)
            self.labels.extend([label] * samples.shape[0])
            self.snrs.extend([snr] * samples.shape[0])
            
        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)
        self.snrs = np.array(self.snrs)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # RML2016.10a shapes are usually (2, 128) per sample
        x = torch.FloatTensor(self.data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, idx

def get_dataloaders(file_path, known_classes, unknown_classes=None,
                    batch_size=128, min_snr=0, use_pk_sampler=True,
                    P=6, K=8):
    """
    Build train / validation DataLoaders.

    Args:
        use_pk_sampler: If True the training loader uses a PKSampler
            that guarantees each batch has P classes × K samples for
            effective batch-hard triplet mining.
        P: Number of classes per batch (must be <= number of known classes).
        K: Number of samples per class per batch.
    """
    dataset = RadioMLDataset(file_path, known_classes, unknown_classes, min_snr)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use manual seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    if use_pk_sampler:
        # Extract labels for the training subset indices
        train_labels = [dataset.labels[i] for i in train_dataset.indices]
        pk_sampler = PKSampler(train_labels, P=P, K=K)

        # PKSampler yields full batch index-lists, so we use batch_sampler
        # and remap the local PKSampler indices back to the dataset indices.
        class _RemappedSampler:
            """Thin wrapper that maps PKSampler local indices → Subset global indices."""
            def __init__(self, pk, subset_indices):
                self._pk = pk
                self._map = subset_indices  # list[int]

            def __iter__(self):
                for batch in self._pk:
                    yield [self._map[i] for i in batch]

            def __len__(self):
                return len(self._pk)

        remapped = _RemappedSampler(pk_sampler, train_dataset.indices)
        train_loader = DataLoader(dataset, batch_sampler=remapped)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
