from core.loss import BCEContrastLoss
import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse

from models.donet import DONet
from core.threshold import DynamicAdaptiveThreshold
from core.mia import MovingIntersectionAlgorithm
from utils.usb import UnknownSignalBank
from data.dataset import get_dataloaders

# ── Toggles ─────────────────────────────────────────────────────────────
USE_SIMPLE_PROJ   = False      # use full clp and cop
ALPHA             = 0.5       # Joint loss weighting: alpha * ce + (1-alpha) * contrast
# ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Training')
    parser.add_argument('--known_classes', type=str, required=True, help='Comma separated list of known classes')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = 'RML2016.10a_dict.pkl'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    # RadioML 2016.10a modulations
    all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    # Split into known and unknown from args
    known_classes = [c.strip() for c in args.known_classes.split(',')]
    unknown_classes = [c for c in all_classes if c not in known_classes]
    
    print(f"Known classes: {known_classes}")
    print(f"Unknown classes: {unknown_classes}")
    
    # PKSampler: P=6 (all known classes), K=8 samples each → batch=48
    train_loader, val_loader = get_dataloaders(
        dataset_path, 
        known_classes=known_classes, 
        unknown_classes=unknown_classes,
        batch_size=512,          # fallback if PKSampler disabled
        use_pk_sampler=False,
        P=6, K=32
    )
    
    num_known = len(known_classes)
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    
    # ── Loss ────────────────────────────────────────────────────────────
    criterion = BCEContrastLoss().to(device)

    # ── Optimiser & Scheduler ───────────────────────────────────────────
    # Paper uses Adam optimizer with learning rate of 0.0006
    optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    
    # Dynamic Adaptive Threshold (DAT) for signal novelty detection
    dat = DynamicAdaptiveThreshold(alpha=0.95)
    mia = MovingIntersectionAlgorithm(L=5)
    usb = UnknownSignalBank(max_size=40000)
    
    num_epochs = 50
    warmup_epochs = 5
    
    # ── Logging & Early Stop ────────────────────────────────────────────
   # logger = TrainingLogger(log_path="logs/training_log.csv")
    #early_stop = EarlyStopping(patience=7, min_dpythelta=1e-4)
    
    # Create directory for checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # Update SFCs as per Algorithm 1 in Open-ICL paper
        model.update_sfcs(train_loader, device)
        
        model.train()
        #logger.epoch_start()
        total_loss = 0
        num_batches = 0
        epoch_candidates = set()
        epoch_candidate_data = {}
        
        for batch_x, batch_y, batch_idx in train_loader:
            batch_x, batch_y, batch_idx = batch_x.to(device), batch_y.to(device), batch_idx.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: DONet returns (logits, contrast_features, contrast_probs, y_novelty, distances)
            logits, contrast_features, contrast_probs, y_novelty, distances = model(batch_x)
            
            # Filter known samples for loss calculation
            known_mask = (batch_y != -1)
            
            if known_mask.sum() > 0:
                # Joint Loss: ALPHA * ce_loss + (1 - ALPHA) * contrast_loss
                ce_loss = F.cross_entropy(logits[known_mask], batch_y[known_mask])
                contrast_loss = criterion(contrast_probs[known_mask], batch_y[known_mask], num_known)
                loss = ALPHA * ce_loss + (1.0 - ALPHA) * contrast_loss
     
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                
            # Accumulate features and detect unknowns via CLP/COP distance metric & DAT
            with torch.no_grad():
                # Update DAT threshold with known samples
                if known_mask.sum() > 0:
                    dat.update(distances[known_mask], batch_y[known_mask])
                
                # USB Population (after warmup) using DAT threshold
                if epoch >= warmup_epochs:
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
        if epoch >= warmup_epochs:
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
                    
        # ── Epoch stats ─────────────────────────────────────────────────
        avg_loss = total_loss / max(1, num_batches)
        current_lr = optimizer.param_groups[0]['lr']

        # Step the plateau-aware scheduler
        scheduler.step(avg_loss)


        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.6f}")
        print(f"Signals currently in Unknown Signal Bank: {len(usb.signals)}")

    # Save initial model state
    save_dict = {
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes,
        'dat_threshold': dat.get_threshold(),
        'usb_signals': usb.signals,
        'usb_features': usb.features,
        'use_simple_projection': False,  # legacy key, architecture now always uses full COP/CLP
    }
    checkpoint_path = os.path.join(args.checkpoint_dir, "phase1_model.pth")
    torch.save(save_dict, checkpoint_path)
    
    print("\nPhase 1 Training Complete! Model saved.")
    print("Now ready for Incremental Learning (Phase 2).")

if __name__ == '__main__':
    main()
