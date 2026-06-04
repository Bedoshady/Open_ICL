import torch
import torch.nn.functional as F
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
    dat_threshold = checkpoint.get('dat_threshold', 0.0)
    num_classes = len(known_classes)
    
    # Original known classes (before incremental learning)
    original_known_classes = ['8PSK', 'BPSK', 'QPSK', 'QAM16', 'QAM64', 'PAM4']
    num_original_known = len(original_known_classes)
    
    use_simple_proj = checkpoint.get('use_simple_projection', True)
    model = DONet(num_known_classes=num_classes, feature_dim=128, use_simple_projection=use_simple_proj).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation data (including unknown classes)
    all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    unknown_classes = [c for c in all_classes if c not in original_known_classes]
    
    _, val_loader = get_dataloaders(
        'RML2016.10a_dict.pkl', 
        known_classes=original_known_classes, 
        unknown_classes=unknown_classes,
        batch_size=128
    )
    
    print("\n--- Running Evaluation with CLP/COP DAT Anomaly Detection ---")
    print(f"DAT Threshold from training: {dat_threshold:.4f}")
    
    y_true_binary = []  # 0 for known, 1 for unknown
    y_scores = []       # Anomaly score (min distance to SFC)
    y_pred_binary = []  # Binary prediction based on DAT threshold
    
    correct_known = 0
    total_known = 0
    
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # The model computes logits, contrast features, and distances
            logits, contrast_features, distances = model(batch_x)
            
            # Compute class predictions via CLP logits
            pred_classes = logits.argmax(dim=1)
            pred_classes_np = pred_classes.cpu().numpy()
            
            # Minimum distance to any known class SFC
            min_dists, _ = distances.min(dim=1)
            min_dists_np = min_dists.cpu().numpy()
            
            # Ground Truth Binary: batch_y is -1 for any unknown signal sample
            is_unknown = (batch_y == -1).cpu().numpy()
            y_true_binary.extend(is_unknown.astype(int))
            
            # 1. Anomaly Score: Force max score (2.0) if it hits a novel cluster, else use min distance
            anomaly_scores = np.where(pred_classes_np >= num_original_known, 2.0, min_dists_np)
            y_scores.extend(anomaly_scores)
            
            # 2. Binary Prediction: Flag as Unknown (1) if min distance > dat_threshold OR if it falls into a novel cluster
            predictions = ((min_dists_np > dat_threshold) | (pred_classes_np >= num_original_known)).astype(int)
            y_pred_binary.extend(predictions)
            
            # 3. Closed-Set Accuracy: Evaluated ONLY on the original known classes
            known_mask = batch_y >= 0
            if known_mask.any():
                true_known_labels = batch_y[known_mask].cpu().numpy()
                pred_known_labels = pred_classes_np[known_mask.cpu().numpy()]
                
                # Must match the correct original known index perfectly
                correct_known += (pred_known_labels == true_known_labels).sum()
                total_known += known_mask.sum().item()

    y_true_binary = np.array(y_true_binary)
    y_scores = np.array(y_scores)
    y_pred_binary = np.array(y_pred_binary)
    
    # Metrics calculation
    os_f1 = f1_score(y_true_binary, y_pred_binary)
    if len(np.unique(y_true_binary)) > 1:
        os_auc = roc_auc_score(y_true_binary, y_scores)
    else:
        os_auc = 0.0
        
    acc = correct_known / total_known if total_known > 0 else 0
    
    print("\nResults:")
    print(f"Closed-Set Classification Accuracy (Knowns): {acc*100:.2f}%")
    print(f"Open-Set Detection F1-Score (DAT Threshold = {dat_threshold:.4f}): {os_f1:.4f}")
    print(f"Open-Set Detection AUROC: {os_auc:.4f}")

if __name__ == '__main__':
    evaluate_model()
