import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from models.donet import DONet
from core.loss import OpenICLLoss
from core.threshold import DynamicAdaptiveThreshold
from core.mia import MovingIntersectionAlgorithm
from utils.usb import UnknownSignalBank
from data.dataset import get_dataloaders

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = 'RML2016.10a_dict.pkl'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # RadioML 2016.10a modulations
    all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    # Split into known and unknown to minimize confusion (test theory)
    # Known: All Phase/Amplitude shift keying
    known_classes = ['8PSK', 'BPSK', 'QPSK', 'QAM16', 'QAM64', 'PAM4']
    # Unknown: All AM/FM/FSK continuous modulations
    unknown_classes = ['AM-DSB', 'AM-SSB', 'CPFSK', 'GFSK', 'WBFM']
    
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    
    train_loader, val_loader = get_dataloaders(
        dataset_path, 
        known_classes=known_classes, 
        unknown_classes=unknown_classes,
        batch_size=512
    )
    
    num_known = len(known_classes)
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    
    criterion = OpenICLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dat = DynamicAdaptiveThreshold(alpha=0.95)
    mia = MovingIntersectionAlgorithm(L=5)
    usb = UnknownSignalBank(max_size=40000)
    
    num_epochs = 50
    
    # Create directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_candidates = set()
        epoch_candidate_data = {}
        
        for batch_x, batch_y, batch_idx in train_loader:
            batch_x, batch_y, batch_idx = batch_x.to(device), batch_y.to(device), batch_idx.to(device)
            
            optimizer.zero_grad()
            
            # The model now natively computes the Distance Metric (DM) against SFCs
            logits, contrast_features, distances = model(batch_x)
            
            # Calculate loss (only applied to known classes where label >= 0)
            loss = criterion(logits, distances, batch_y)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Open-Set Recognition and USB population
            with torch.no_grad():
                # Update DAT threshold with known samples
                dat.update(distances, batch_y)
                
                # Use MIA to find candidate unknowns in this batch
                current_threshold = dat.get_threshold()
                if current_threshold > 0:
                    candidates = mia.detect_candidates(distances, current_threshold, batch_idx)
                    epoch_candidates.update(candidates)
                    
                    # Temporarily store the signals/features of candidates for this epoch
                    for idx_val in candidates:
                        # Find the local batch position for this global index
                        local_pos = (batch_idx == idx_val).nonzero(as_tuple=True)[0][0]
                        epoch_candidate_data[idx_val] = (batch_x[local_pos].cpu(), contrast_features[local_pos].cpu())
                        
        # End of epoch: evaluate MIA intersection
        reliable_indices = mia.update_epoch(epoch_candidates)
        if reliable_indices:
            new_signals = []
            new_features = []
            for idx_val in reliable_indices:
                if idx_val in epoch_candidate_data:
                    new_signals.append(epoch_candidate_data[idx_val][0])
                    new_features.append(epoch_candidate_data[idx_val][1])
            if new_signals:
                usb.add_signals(new_signals, new_features)
                
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, DAT Threshold: {dat.get_threshold():.4f}")
        print(f"Signals currently in Unknown Signal Bank: {len(usb.signals)}")

    # Save initial model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes,
        'dat_threshold': dat.get_threshold(),
        'usb_signals': usb.signals,
        'usb_features': usb.features
    }, "checkpoints/phase1_model.pth")
    
    print("\nPhase 1 Training Complete! Model saved.")
    print("Now ready for Incremental Learning (Phase 2).")

if __name__ == '__main__':
    main()
