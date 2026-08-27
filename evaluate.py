import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse
from sklearn.metrics import f1_score, roc_auc_score

from models.donet import DONet
from data.dataset import get_dataloaders

def evaluate_model():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory containing phase1 checkpoint')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to dataset file')
    parser.add_argument('--dataset_type', type=str, default='', help='Dataset format to use (overrides checkpoint value)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset splits')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    checkpoint_path_p2 = os.path.join(args.checkpoint_dir, "phase2_incremental_model.pth")
    checkpoint_path_p1 = os.path.join(args.checkpoint_dir, "phase1_model.pth")
    
    if os.path.exists(checkpoint_path_p2):
        checkpoint_path = checkpoint_path_p2
        print(f"Loading Phase 2 Incremental Model from {checkpoint_path}")
    elif os.path.exists(checkpoint_path_p1):
        checkpoint_path = checkpoint_path_p1
        print(f"Loading Phase 1 Model from {checkpoint_path}")
    else:
        print("Model checkpoint not found. Run train.py first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    original_known_classes = checkpoint['known_classes']
    discovered_classes = checkpoint.get('discovered_classes', [])
    dat_threshold = checkpoint.get('dat_threshold', 0.0)
    
    num_original_known = len(original_known_classes)
    num_classes = num_original_known + len(discovered_classes)
    
    print(f"Original Classes are {original_known_classes} and the number of original known classes is {num_original_known}")
    if discovered_classes:
        print(f"Discovered Novel Classes: {discovered_classes}")
    print("")
    model = DONet(num_known_classes=num_classes, feature_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if args.dataset_type:
        dataset_type = args.dataset_type
    else:
        dataset_type = checkpoint.get('dataset_type', 'rml2016')
    
    if args.dataset_path:
        dataset_path = args.dataset_path
    else:
        dataset_path = 'RML2016.10a_dict.pkl' if dataset_type == 'rml2016' else 'GOLD_XYZ_OSC.0001_1024.hdf5'

    # Load validation data (including unknown classes)
    if dataset_type == 'rml2018':
        all_classes = ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32', 
                       'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64', 
                       'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC', 
                       'FM', 'GMSK', 'OQPS']
    else:
        all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
        
    unknown_classes = [c for c in all_classes if c not in original_known_classes]
    
    _, val_loader = get_dataloaders(
        dataset_path, 
        known_classes=original_known_classes, 
        unknown_classes=unknown_classes,
        batch_size=128,
        dataset_type=dataset_type,
        seed=args.seed
    )
    
    print("\n--- Running Evaluation with CLP/COP DAT Anomaly Detection ---")
    print(f"DAT Thresholds from training: {dat_threshold}")
    
    y_true_binary = []  # 0 for known, 1 for unknown
    y_scores = []       # Anomaly score (min distance to SFC)
    y_pred_binary = []  # Binary prediction based on DAT threshold
    y_true_class = []
    y_pred_class = []
    
    correct_known = 0
    total_known = 0
    
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # The model computes logits, contrast features, contrast probs, novelty score, and distances
            logits, contrast_features, contrast_probs, y_novelty, distances = model(batch_x)
            
            # Compute class predictions via CLP logits
            pred_classes = logits.argmax(dim=1)
            pred_classes_np = pred_classes.cpu().numpy()
            
            # Anomaly score: use y_novelty (min contrast prob) instead of Euclidean distance
            min_probs, closest_classes = contrast_probs.min(dim=1)
            min_probs_np = min_probs.cpu().numpy()
            closest_classes_np = closest_classes.cpu().numpy()
            
            # Ground Truth Binary: batch_y is -1 for any unknown signal sample
            is_unknown = (batch_y == -1).cpu().numpy()
            y_true_binary.extend(is_unknown.astype(int))
            
            y_true_class.extend(batch_y.cpu().numpy())
            y_pred_class.extend(pred_classes_np)
            
            # 1. Anomaly Score: Force max score (2.0) if it hits a novel cluster, else use min prob
            anomaly_scores = np.where(pred_classes_np >= num_original_known, 2.0, min_probs_np)
            y_scores.extend(anomaly_scores)
            
            # 2. Binary Prediction: Flag as Unknown (1) if min prob > dat_threshold OR if it falls into a novel cluster
            if isinstance(dat_threshold, dict):
                thresholds_arr = np.array([dat_threshold.get(c, float('inf')) for c in closest_classes_np])
                dist_cond = min_probs_np > thresholds_arr
            else:
                dist_cond = min_probs_np > dat_threshold
                
            predictions = (dist_cond | (pred_classes_np >= num_original_known)).astype(int)
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
    y_true_class = np.array(y_true_class)
    y_pred_class = np.array(y_pred_class)
    
    # Metrics calculation
    os_f1 = f1_score(y_true_binary, y_pred_binary)
    if len(np.unique(y_true_binary)) > 1:
        os_auc = roc_auc_score(y_true_binary, y_scores)
    else:
        os_auc = 0.0
        
    acc = correct_known / total_known if total_known > 0 else 0
    
    # Overall Accuracy (OA) Calculation
    # TP: Unknowns correctly identified as unknowns
    tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
    # TN: Knowns correctly identified as knowns AND correctly classified to their specific class
    tn = ((y_true_binary == 0) & (y_pred_binary == 0) & (y_true_class == y_pred_class)).sum()
    
    total_samples = len(y_true_binary)
    oa = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    print("\nResults:")
    print(f"Closed-Set Classification Accuracy (Knowns): {acc*100:.2f}%")
    print(f"Open-Set Detection F1-Score (DAT Threshold = {dat_threshold}): {os_f1:.4f}")
    print(f"Open-Set Detection AUROC: {os_auc:.4f}")
    print(f"Overall Accuracy (OA): {oa*100:.2f}%")
    
    # Print parseable line for wrapper script
    print(f"METRICS: OA={oa*100:.4f}, F1={os_f1:.4f}, CS_ACC={acc*100:.4f}")

if __name__ == '__main__':
    evaluate_model()
