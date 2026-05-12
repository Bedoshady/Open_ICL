import torch
import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from models.donet import DONet
from data.dataset import get_dataloaders

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    checkpoint_path = "checkpoints/phase1_model.pth"
    if not os.path.exists(checkpoint_path):
        print("Model checkpoint not found. Run train.py first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    known_classes = checkpoint['known_classes']
    dat_threshold = checkpoint['dat_threshold']
    num_known = len(known_classes)
    
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation data (including unknown classes)
    all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    unknown_classes = [c for c in all_classes if c not in known_classes]
    
    _, val_loader = get_dataloaders(
        'RML2016.10a_dict.pkl', 
        known_classes=known_classes, 
        unknown_classes=unknown_classes,
        batch_size=128
    )
    
    print("\n--- Running Evaluation ---")
    print(f"DAT Threshold from training: {dat_threshold:.4f}")
    
    y_true_binary = []  # 0 for known, 1 for unknown
    y_scores = []       # min distance to any SFC
    
    correct_known = 0
    total_known = 0
    
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits, _, distances = model(batch_x)
            
            # Minimum distance to any known class SFC
            min_dists, pred_classes = distances.min(dim=1)
            
            # Record for Open-Set Metrics
            is_unknown = (batch_y == -1).cpu().numpy()
            y_true_binary.extend(is_unknown.astype(int))
            y_scores.extend(min_dists.cpu().numpy())
            
            # Classification Accuracy (only on known classes)
            known_mask = batch_y >= 0
            if known_mask.any():
                true_known_labels = batch_y[known_mask]
                pred_known_labels = pred_classes[known_mask]
                correct_known += (pred_known_labels == true_known_labels).sum().item()
                total_known += known_mask.sum().item()

    y_true_binary = np.array(y_true_binary)
    y_scores = np.array(y_scores)
    
    # Predictions based on DAT threshold
    y_pred_binary = (y_scores > dat_threshold).astype(int)
    
    # Metrics calculation
    os_f1 = f1_score(y_true_binary, y_pred_binary)
    if len(np.unique(y_true_binary)) > 1:
        os_auc = roc_auc_score(y_true_binary, y_scores)
    else:
        os_auc = 0.0
        
    acc = correct_known / total_known if total_known > 0 else 0
    
    print("\nResults:")
    print(f"Closed-Set Classification Accuracy (Knowns): {acc*100:.2f}%")
    print(f"Open-Set Detection F1-Score: {os_f1:.4f}")
    print(f"Open-Set Detection AUROC: {os_auc:.4f}")

if __name__ == '__main__':
    evaluate_model()
