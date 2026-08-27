import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from core.sampler import PKSampler

class RadioMLDataset(Dataset):
    """
    Dataset loader for RadioML 2016.10a / 2018.01a.
    """
    def __init__(self, file_path, known_classes=None, unknown_classes=None, min_snr=0, dataset_type='rml2016'):
        super().__init__()
        self.dataset_type = dataset_type
        
        if self.dataset_type == 'rml2018':
            import h5py
            self.h5_file = h5py.File(file_path, 'r')
            self.X = self.h5_file['X']
            self.Y = self.h5_file['Y']
            self.Z = self.h5_file['Z']
            
            all_mods = ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32', 
                        'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64', 
                        'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC', 
                        'FM', 'GMSK', 'OQPS']
                        
            self.known_classes = known_classes if known_classes else all_mods
            self.unknown_classes = unknown_classes if unknown_classes else []
            self.class_to_idx = {mod: idx for idx, mod in enumerate(self.known_classes)}
            
            print("Reading labels and SNRs from HDF5...")
            Y_data = self.Y[:]
            Z_data = self.Z[:]
            mod_indices = np.argmax(Y_data, axis=1)
            
            self.valid_indices = []
            self.labels = []
            self.snrs = []
            self.true_class_names = []
            
            print("Filtering dataset based on SNR and classes...")
            for i in range(len(mod_indices)):
                snr = Z_data[i][0]
                if snr < min_snr:
                    continue
                mod_str = all_mods[mod_indices[i]]
                if mod_str in self.known_classes:
                    self.valid_indices.append(i)
                    self.labels.append(self.class_to_idx[mod_str])
                    self.snrs.append(snr)
                    self.true_class_names.append(mod_str)
                elif mod_str in self.unknown_classes:
                    self.valid_indices.append(i)
                    self.labels.append(-1)
                    self.snrs.append(snr)
                    self.true_class_names.append(mod_str)
                    
            self.valid_indices = np.array(self.valid_indices)
            self.labels = np.array(self.labels)
            self.snrs = np.array(self.snrs)
            self.true_class_names = np.array(self.true_class_names)
            print(f"Loaded {len(self.valid_indices)} valid samples.")
            
        else:
            # Using latin1 encoding is required for loading older Python 2 pickle files (like RML2016.10a_dict.pkl)
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                
            self.data = []
            self.labels = []
            self.snrs = []
            self.true_class_names = []
            
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
                self.true_class_names.extend([mod] * samples.shape[0])
                
            self.data = np.vstack(self.data)
            self.labels = np.array(self.labels)
            self.snrs = np.array(self.snrs)
            self.true_class_names = np.array(self.true_class_names)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if self.dataset_type == 'rml2018':
            real_idx = self.valid_indices[idx]
            # X shape is (N, 1024, 2). Need (2, 1024)
            x_data = self.X[real_idx]
            x_data = np.transpose(x_data, (1, 0))
            x = torch.FloatTensor(x_data)
        else:
            # RML2016.10a shapes are usually (2, 128) per sample
            x = torch.FloatTensor(self.data[idx])
            
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y, idx

def get_dataloaders(file_path, known_classes, unknown_classes=None,
                    batch_size=128, min_snr=0, use_pk_sampler=False,
                    P=6, K=8, dataset_type='rml2016', seed=42):
    """
    Build train / validation DataLoaders.

    Args:
        use_pk_sampler: If True the training loader uses a PKSampler
            that guarantees each batch has P classes × K samples for
            effective batch-hard triplet mining.
        P: Number of classes per batch (must be <= number of known classes).
        K: Number of samples per class per batch.
    """
    dataset = RadioMLDataset(file_path, known_classes, unknown_classes, min_snr, dataset_type=dataset_type)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use manual seed for reproducible splits
    generator = torch.Generator().manual_seed(seed)
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
