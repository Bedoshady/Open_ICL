import torch
import torch.optim as optim
import os
import numpy as np

from models.donet import DONet
from core.loss import OpenICLLoss
from utils.usb import UnknownSignalBank
from data.dataset import get_dataloaders

def run_incremental_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Phase 1 model
    checkpoint_path = "checkpoints/phase1_model.pth"
    if not os.path.exists(checkpoint_path):
        print("Please run train.py first to generate the phase1_model.pth checkpoint.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    known_classes = checkpoint['known_classes']
    num_known = len(known_classes)
    
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Load USB from checkpoint
    print("\n--- Step 1: Discovering Novel Classes from USB ---")
    usb = UnknownSignalBank(max_size=5000)
    
    if 'usb_signals' in checkpoint and 'usb_features' in checkpoint:
        usb.signals = checkpoint['usb_signals']
        usb.features = checkpoint['usb_features']
        print(f"Loaded {len(usb.signals)} unknown signals from USB.")
    else:
        print("Error: No USB data found in checkpoint! Please re-run train.py to generate a checkpoint with USB data.")
        return
    
    # Cluster the unknowns dynamically
    print("\n--- Step 2: Dynamic Clustering of Unknowns ---")
    pseudo_labels, n_new_classes = usb.discover_new_classes(n_clusters=None)
    print(f"Dynamically discovered {n_new_classes} new classes using Silhouette Score.")
    
    # Offset pseudo labels by the number of known classes
    pseudo_labels = pseudo_labels + num_known
    
    # 3. Model Expansion
    print("\n--- Step 2: Expanding DONet ---")
    total_classes = num_known + n_new_classes
    model.update_num_classes(total_classes)
    print(f"Model expanded to {total_classes} classes.")
    
    # 4. Fine-Tuning Phase with Sample Replay (As per Open-ICL Paper Section III.F)
    print("\n--- Step 3: Incremental Fine-Tuning with Sample Replay ---")
    criterion = OpenICLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    
    # Load the original known data (D_trin) for Sample Replay
    # We use a small batch size from the original training set to combine with USB
    train_loader, _ = get_dataloaders(
        'RML2016.10a_dict.pkl', 
        known_classes=known_classes, 
        unknown_classes=[],
        batch_size=128
    )
    
    # Get USB signals
    usb_signals, _ = usb.get_all()
    if usb_signals is not None:
        usb_signals = torch.tensor(usb_signals, dtype=torch.float32)
        usb_labels = torch.tensor(pseudo_labels, dtype=torch.long)
    else:
        usb_signals = torch.empty((0, 2, 128))
        usb_labels = torch.empty((0,), dtype=torch.long)
        
    num_epochs = 5
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0
        batches = 0
        
        # Iterate over the original dataset (Sample Replay)
        for batch_known_x, batch_known_y, _ in train_loader:
            
            # Combine known samples with a random batch of USB unknown samples
            if len(usb_signals) > 0:
                indices = torch.randperm(len(usb_signals))[:batch_size]
                batch_unknown_x = usb_signals[indices]
                batch_unknown_y = usb_labels[indices]
                
                batch_x = torch.cat([batch_known_x, batch_unknown_x], dim=0).to(device)
                batch_y = torch.cat([batch_known_y, batch_unknown_y], dim=0).to(device)
            else:
                batch_x = batch_known_x.to(device)
                batch_y = batch_known_y.to(device)
            
            optimizer.zero_grad()
            logits, _, distances = model(batch_x)
            
            loss = criterion(logits, distances, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        print(f"Fine-Tuning Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / max(1, batches):.4f}")
        
    # Save the expanded model
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes + [f"Novel_{i}" for i in range(n_new_classes)],
        'dat_threshold': checkpoint['dat_threshold']
    }, "checkpoints/phase2_incremental_model.pth")
    
    print("\nIncremental Learning Complete. New model saved!")

if __name__ == '__main__':
    run_incremental_learning()
