from core.loss import BatchAllTripletLoss
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import argparse

from models.donet import DONet
from utils.usb import UnknownSignalBank
from data.dataset import get_dataloaders

def run_incremental_learning():
    parser = argparse.ArgumentParser(description='Phase 2 Incremental Training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing phase1 checkpoint')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the Phase 1 model
    checkpoint_path = os.path.join(args.checkpoint_dir, "phase1_model.pth")
    if not os.path.exists(checkpoint_path):
        print("Please run train.py first to generate the phase1_model.pth checkpoint.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    known_classes = checkpoint['known_classes']
    num_known = len(known_classes)
    
    use_simple_proj = checkpoint.get('use_simple_projection', True)
    model = DONet(num_known_classes=num_known, feature_dim=128, use_simple_projection=use_simple_proj).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Load USB from checkpoint
    print("\n--- Step 1: Discovering Novel Classes from USB ---")
    usb = UnknownSignalBank(max_size=40000)
    
    if 'usb_signals' in checkpoint and 'usb_features' in checkpoint:
        usb.signals = checkpoint['usb_signals']
        usb.features = checkpoint['usb_features']
        print(f"Loaded {len(usb.signals)} unknown signals from USB.")
    else:
        print("Error: No USB data found in checkpoint! Please re-run train.py to generate a checkpoint with USB data.")
        return
    
    if len(usb.signals) == 0:
        print("USB is empty — no unknown signals were discovered during training. Exiting.")
        return

    # 3. Cluster the unknowns dynamically
    print("\n--- Step 2: Dynamic Clustering of Unknowns ---")
    pseudo_labels, n_new_classes = usb.discover_new_classes(n_clusters=None)
    print(f"Dynamically discovered {n_new_classes} new classes using Silhouette Score.")
    
    if n_new_classes == 0:
        print("No new classes discovered from clustering. Exiting.")
        return

    # Offset pseudo labels by the number of known classes
    pseudo_labels = pseudo_labels + num_known
    
    # 4. Model Update
    # Since we removed the CLP, update_num_classes just tracks the count.
    print("\n--- Step 3: Updating DONet class count ---")
    total_classes = num_known + n_new_classes
    model.update_num_classes(total_classes)
    print(f"Model now tracking {total_classes} total classes ({num_known} known + {n_new_classes} novel).")
    
    # 5. Fine-Tuning Phase with Sample Replay using Triplet Loss
    print("\n--- Step 4: Incremental Fine-Tuning with Sample Replay (Triplet Loss) ---")
    criterion = BatchAllTripletLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-4)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    # Logger
    #inc_logger = TrainingLogger(log_path="logs/incremental_training_log.csv")
    #early_stop = EarlyStopping(patience=5, min_delta=1e-4)
    
    model.train()
    
    # Load the original known data for Sample Replay
    train_loader, _ = get_dataloaders(
        'RML2016.10a_dict.pkl', 
        known_classes=known_classes, 
        unknown_classes=[],
        batch_size=128,
        use_pk_sampler=False   # simpler sampling for short fine-tune
    )
    
    # Prepare USB signals as tensors
    usb_signals, _ = usb.get_all()
    if usb_signals is not None:
        usb_signals = torch.tensor(usb_signals, dtype=torch.float32)
        usb_labels = torch.tensor(pseudo_labels, dtype=torch.long)
    else:
        usb_signals = torch.empty((0, 2, 128))
        usb_labels = torch.empty((0,), dtype=torch.long)
        
    num_epochs = 50  # more epochs to benefit from scheduler
    usb_batch_size = 32
    
    for epoch in range(num_epochs):
        # Update SFCs dynamically 
        model.update_sfcs(train_loader, device, extra_x=usb_signals, extra_y=usb_labels)
        
        #inc_logger.epoch_start()
        total_loss = 0
        batches = 0

        # Margin curriculum: ramp 0.5 → 1.0 over first 5 epochs
        #current_margin = get_margin(epoch, 0.5, 1.0, ramp_epochs=5)
        
        # Iterate over the original dataset (Sample Replay)
        for batch_known_x, batch_known_y, _ in train_loader:
            
            # Combine known samples with a random batch of USB unknown samples
            if len(usb_signals) > 0:
                indices = torch.randperm(len(usb_signals))[:usb_batch_size]
                batch_unknown_x = usb_signals[indices]
                batch_unknown_y = usb_labels[indices]
                
                batch_x = torch.cat([batch_known_x, batch_unknown_x], dim=0).to(device)
                batch_y = torch.cat([batch_known_y, batch_unknown_y], dim=0).to(device)
            else:
                batch_x = batch_known_x.to(device)
                batch_y = batch_known_y.to(device)
            
            optimizer.zero_grad()
            #center_optimizer.zero_grad()
            
            # Forward pass: model returns (logits, contrast_features, distances)
            logits, contrast_features, distances = model(batch_x)
            
            # Filter known samples for loss calculation (in case -1 labels exist)
            valid_mask = (batch_y != -1)
            
            if valid_mask.sum() > 0:
                # Joint Loss: alpha * ce_loss + (1 - alpha) * triplet_loss
                alpha = 0.5
                ce_loss = F.cross_entropy(logits[valid_mask], batch_y[valid_mask])
                triplet_loss = criterion(contrast_features[valid_mask], batch_y[valid_mask])
                loss = alpha * ce_loss + (1.0 - alpha) * triplet_loss
                
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batches += 1

        avg_loss = total_loss / max(1, batches)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Incremental Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    # 6. Re-calculate DAT threshold on updated features (known + newly discovered)
    print("\n--- Step 5: Re-calculating DAT Threshold on Incremental Features ---")
    from core.threshold import DynamicAdaptiveThreshold
    dat = DynamicAdaptiveThreshold(alpha=0.95)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            _, _, distances = model(batch_x)
            
            valid_mask = (batch_y != -1)
            if valid_mask.sum() > 0:
                dat.update(distances[valid_mask], batch_y[valid_mask])
            
        if len(usb_signals) > 0:
            usb_batch = usb_signals.to(device)
            _, _, usb_distances = model(usb_batch)
            dat.update(usb_distances, usb_labels.to(device))

    # Save the expanded model
    novel_class_names = [f"Novel_{i}" for i in range(n_new_classes)]
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes + novel_class_names,
        'dat_threshold': dat.get_threshold(),
        'usb_signals': usb.signals,
        'usb_features': usb.features,
        'use_simple_projection': use_simple_proj
    }, os.path.join(args.checkpoint_dir, "phase2_incremental_model.pth"))
    
    print(f"\nIncremental Learning Complete!")
    print(f"New model saved with {total_classes} classes: {known_classes + novel_class_names}")

if __name__ == '__main__':
    run_incremental_learning()
