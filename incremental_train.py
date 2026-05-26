import torch
import torch.optim as optim
import os
import numpy as np

from models.donet import DONet
from core.loss import BatchHardTripletLoss, CenterLoss, get_margin
from core.evt import DynamicEVT
from utils.usb import UnknownSignalBank
from utils.logger import TrainingLogger, EarlyStopping
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
    criterion = BatchHardTripletLoss(margin=1.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Optional Center Loss for incremental phase
    center_loss_fn = CenterLoss(num_classes=total_classes,
                                 feat_dim=128,
                                 device=device).to(device)
    center_optimizer = optim.SGD(center_loss_fn.parameters(), lr=0.5)
    LAMBDA_CENTER = 0.01

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, verbose=True
    )

    # Logger
    inc_logger = TrainingLogger(log_path="logs/incremental_training_log.csv")
    early_stop = EarlyStopping(patience=5, min_delta=1e-4)
    
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
        
    num_epochs = 10  # more epochs to benefit from scheduler
    usb_batch_size = 32
    
    for epoch in range(num_epochs):
        inc_logger.epoch_start()
        total_loss = 0
        batches = 0

        # Margin curriculum: ramp 0.5 → 1.0 over first 5 epochs
        current_margin = get_margin(epoch, 0.5, 1.0, ramp_epochs=5)
        
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
            center_optimizer.zero_grad()
            
            # Forward pass: model now returns only contrast_features
            contrast_features = model(batch_x)
            
            # Triplet Loss handles both known and novel pseudo-labels
            loss = criterion(contrast_features, batch_y, margin_override=current_margin)
            c_loss = center_loss_fn(contrast_features, batch_y)
            loss = loss + LAMBDA_CENTER * c_loss
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                center_optimizer.step()
                total_loss += loss.item()
                batches += 1

        avg_loss = total_loss / max(1, batches)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)

        inc_logger.epoch_end(
            epoch=epoch + 1, loss=avg_loss, lr=current_lr,
            margin=current_margin, usb_size=len(usb.signals)
        )
        print(f"Fine-Tuning Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.6f}, Margin: {current_margin:.2f}")

        if early_stop.step(avg_loss):
            print(f"Incremental fine-tuning stopped early at epoch {epoch+1}.")
            break

    # 6. Re-fit EVT on updated features (known + newly discovered)
    print("\n--- Step 5: Re-fitting EVT on Incremental Features ---")
    evt = DynamicEVT(tail_size=0.05)
    
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            contrast_features = model(batch_x)
            known_mask = batch_y >= 0
            if known_mask.any():
                all_features.append(contrast_features[known_mask.to(device)].cpu())
                all_labels.append(batch_y[known_mask])
                
        # Also add USB signals (novel classes)
        if len(usb_signals) > 0:
            usb_batch = usb_signals.to(device)
            usb_features = model(usb_batch)
            all_features.append(usb_features.cpu())
            all_labels.append(usb_labels)

    all_features_tensor = torch.cat(all_features, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    evt.fit(all_features_tensor, all_labels_tensor)
    print("EVT re-fitted on all known + novel classes.")
    
    # Save the expanded model
    novel_class_names = [f"Novel_{i}" for i in range(n_new_classes)]
    torch.save({
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes + novel_class_names,
        'evt_models': evt.models,
        'evt_centroids': evt.centroids,
        'usb_signals': usb.signals,
        'usb_features': usb.features
    }, "checkpoints/phase2_incremental_model.pth")
    
    print(f"\nIncremental Learning Complete!")
    print(f"New model saved with {total_classes} classes: {known_classes + novel_class_names}")

if __name__ == '__main__':
    run_incremental_learning()
