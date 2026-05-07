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
    
    # Split into known and unknown to simulate open-set scenarios
    known_classes = all_classes[:8]
    unknown_classes = all_classes[8:]
    
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    
    train_loader, val_loader = get_dataloaders(
        dataset_path, 
        known_classes=known_classes, 
        unknown_classes=unknown_classes,
        batch_size=128
    )
    
    num_known = len(known_classes)
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    
    criterion = OpenICLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    dat = DynamicAdaptiveThreshold(alpha=0.95)
    mia = MovingIntersectionAlgorithm(confidence_threshold=0.5)
    usb = UnknownSignalBank(max_size=5000)
    
    num_epochs = 10
    
    # Create directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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
                
                # Use MIA to find reliable unknowns in this batch
                current_threshold = dat.get_threshold()
                if current_threshold > 0:
                    reliable_unknown_mask = mia.filter_unknowns(distances, current_threshold)
                    
                    # Store these reliable unknown signals in USB
                    if reliable_unknown_mask.any():
                        unknown_signals = batch_x[reliable_unknown_mask]
                        unknown_features = contrast_features[reliable_unknown_mask]
                        usb.add_signals(unknown_signals, unknown_features)
                
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, DAT Threshold: {dat.get_threshold():.4f}")
        print(f"Signals currently in Unknown Signal Bank: {len(usb.signals)}")

    # Save initial model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes,
        'dat_threshold': dat.get_threshold()
    }, "checkpoints/phase1_model.pth")
    
    print("\nPhase 1 Training Complete! Model saved.")
    print("Now ready for Incremental Learning (Phase 2).")

if __name__ == '__main__':
    main()
