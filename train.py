import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from models.donet import DONet
from core.loss import BatchHardTripletLoss, CenterLoss, get_margin
from core.evt import DynamicEVT
from core.mia import MovingIntersectionAlgorithm
from utils.usb import UnknownSignalBank
from utils.logger import TrainingLogger, EarlyStopping
from data.dataset import get_dataloaders

# ── Toggles ─────────────────────────────────────────────────────────────
USE_CENTER_LOSS   = True      # combine triplet + center loss
LAMBDA_CENTER     = 0.01      # weight for center loss term
USE_MARGIN_SCHED  = True      # linearly ramp margin from 0.5→1.0
MARGIN_START      = 0.5
MARGIN_END        = 1.0
MARGIN_RAMP_EPOCHS = 10
# ─────────────────────────────────────────────────────────────────────────

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
    
    # PKSampler: P=6 (all known classes), K=8 samples each → batch=48
    train_loader, val_loader = get_dataloaders(
        dataset_path, 
        known_classes=known_classes, 
        unknown_classes=unknown_classes,
        batch_size=512,          # fallback if PKSampler disabled
        use_pk_sampler=True,
        P=6, K=8
    )
    
    num_known = len(known_classes)
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    
    # ── Loss ────────────────────────────────────────────────────────────
    criterion = BatchHardTripletLoss(margin=MARGIN_END).to(device)
    center_loss_fn = None
    center_optimizer = None
    if USE_CENTER_LOSS:
        center_loss_fn = CenterLoss(num_classes=num_known,
                                     feat_dim=128,
                                     device=device).to(device)
        # Separate optimizer for centers (higher LR so they track quickly)
        center_optimizer = optim.SGD(center_loss_fn.parameters(), lr=0.5)

    # ── Optimiser & Scheduler ───────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    
    evt = DynamicEVT(tail_size=0.05)
    mia = MovingIntersectionAlgorithm(L=5)
    usb = UnknownSignalBank(max_size=40000)
    
    num_epochs = 30
    warmup_epochs = 5
    
    # ── Logging & Early Stop ────────────────────────────────────────────
    logger = TrainingLogger(log_path="logs/training_log.csv")
    early_stop = EarlyStopping(patience=7, min_delta=1e-4)
    
    # Create directory for checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        logger.epoch_start()

        # Current margin for this epoch
        if USE_MARGIN_SCHED:
            current_margin = get_margin(epoch, MARGIN_START, MARGIN_END, MARGIN_RAMP_EPOCHS)
        else:
            current_margin = MARGIN_END

        total_loss = 0
        num_batches = 0
        epoch_candidates = set()
        epoch_candidate_data = {}
        
        # Accumulate known features for EVT fitting at epoch end
        epoch_known_features = []
        epoch_known_labels = []
        
        for batch_x, batch_y, batch_idx in train_loader:
            batch_x, batch_y, batch_idx = batch_x.to(device), batch_y.to(device), batch_idx.to(device)
            
            optimizer.zero_grad()
            if center_optimizer is not None:
                center_optimizer.zero_grad()
            
            # The model now natively computes purely the Contrast Features
            contrast_features = model(batch_x)
            
            # Calculate Triplet Loss with the epoch-dependent margin
            loss = criterion(contrast_features, batch_y, margin_override=current_margin)

            # Optionally add Center Loss
            if USE_CENTER_LOSS and center_loss_fn is not None:
                c_loss = center_loss_fn(contrast_features, batch_y)
                loss = loss + LAMBDA_CENTER * c_loss
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                if center_optimizer is not None:
                    center_optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                
            # Accumulate features for EVT and detect unknowns
            with torch.no_grad():
                known_mask = batch_y >= 0
                if known_mask.any():
                    epoch_known_features.append(contrast_features[known_mask].detach())
                    epoch_known_labels.append(batch_y[known_mask].detach())
                
                # USB Population (after warmup)
                if epoch >= warmup_epochs and evt.models:
                    # Evaluate EVT probability
                    probs, _ = evt.predict_prob(contrast_features)
                    
                    # Prob < 0.05 indicates extreme distance -> Unknown
                    anomalies = probs < 0.05
                    
                    if anomalies.any():
                        anomaly_indices = anomalies.nonzero(as_tuple=True)[0]
                        for idx_pos in anomaly_indices:
                            global_idx = batch_idx[idx_pos].item()
                            epoch_candidates.add(global_idx)
                            epoch_candidate_data[global_idx] = (batch_x[idx_pos].cpu(), contrast_features[idx_pos].cpu())
                            
        # End of epoch: Fit EVT
        with torch.no_grad():
            if epoch_known_features:
                all_features = torch.cat(epoch_known_features, dim=0)
                all_labels = torch.cat(epoch_known_labels, dim=0)
                evt.fit(all_features, all_labels)
                
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

        # Log to CSV
        logger.epoch_end(
            epoch=epoch + 1,
            loss=avg_loss,
            lr=current_lr,
            margin=current_margin,
            usb_size=len(usb.signals),
        )

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"LR: {current_lr:.6f}, Margin: {current_margin:.2f}")
        print(f"Signals currently in Unknown Signal Bank: {len(usb.signals)}")

        # Early stopping check
        if early_stop.step(avg_loss):
            print(f"Stopping early at epoch {epoch+1}.")
            break

    # Save initial model state
    save_dict = {
        'model_state_dict': model.state_dict(),
        'known_classes': known_classes,
        'evt_models': evt.models,
        'evt_centroids': evt.centroids,
        'usb_signals': usb.signals,
        'usb_features': usb.features,
    }
    if USE_CENTER_LOSS and center_loss_fn is not None:
        save_dict['center_loss_state'] = center_loss_fn.state_dict()

    torch.save(save_dict, "checkpoints/phase1_model.pth")
    
    print("\nPhase 1 Training Complete! Model saved.")
    print("Now ready for Incremental Learning (Phase 2).")

if __name__ == '__main__':
    main()
