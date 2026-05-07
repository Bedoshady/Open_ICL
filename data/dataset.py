import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
        return x, y

def get_dataloaders(file_path, known_classes, unknown_classes=None, batch_size=128, min_snr=0):
    dataset = RadioMLDataset(file_path, known_classes, unknown_classes, min_snr)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use manual seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
