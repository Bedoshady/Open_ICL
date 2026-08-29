import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from models.donet import DONet
from core.evt import DynamicEVT
from data.dataset import RadioMLDataset

class RadioMLDatasetWithTrueLabels(RadioMLDataset):
    def __init__(self, file_path, known_classes=None, unknown_classes=None, min_snr=0):
        super().__init__(file_path, known_classes, unknown_classes, min_snr)
        
        # Re-parse to get the original class name for each sample
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        self.true_class_names = []
        for (mod, snr), samples in data.items():
            if snr < min_snr:
                continue
            if mod in self.known_classes or mod in self.unknown_classes:
                self.true_class_names.extend([mod] * samples.shape[0])
        self.true_class_names = np.array(self.true_class_names)

def generate_plots():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating plots on device: {device}")
    
    checkpoint_path = "checkpoints/phase1_model.pth"
    if not os.path.exists(checkpoint_path):
        print("Model checkpoint checkpoints/phase1_model.pth not found. Run train.py first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    known_classes = checkpoint['known_classes']
    num_known = len(known_classes)
    
    print(f"Loaded Phase 1 model. Known classes: {known_classes}")
    
    model = DONet(num_known_classes=num_known, feature_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load EVT
    evt = DynamicEVT(tail_size=0.05)
    if 'evt_models' in checkpoint and 'evt_centroids' in checkpoint:
        evt.models = checkpoint['evt_models']
        evt.centroids = checkpoint['evt_centroids']
    else:
        print("Error: EVT parameters not found in checkpoint!")
        return
        
    all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    unknown_classes = [c for c in all_classes if c not in known_classes]
    
    # Instantiate dataset with true labels tracked
    dataset = RadioMLDatasetWithTrueLabels(
        'RML2016.10a_dict.pkl', 
        known_classes=known_classes, 
        unknown_classes=unknown_classes
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Collect lists to store evaluations
    y_true_binary = []  # 0 for known, 1 for unknown
    y_scores = []       # Anomaly score (1 - EVT probability of being known)
    
    all_pred_known = []
    all_true_known = []
    
    all_features = []
    all_true_classes = [] # original class names
    
    # Collect indices of val dataset to fetch true class names
    val_indices = val_dataset.indices
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_x, batch_y, batch_idx in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            contrast_features = model(batch_x)
            
            # Predict EVT probabilities
            probs, pred_classes = evt.predict_prob(contrast_features)
            
            # Open-set binary true labels (0 = known, 1 = unknown)
            is_unknown = (batch_y == -1).cpu().numpy()
            y_true_binary.extend(is_unknown.astype(int))
            
            # Anomaly score is 1.0 - probability
            y_scores.extend((1.0 - probs).cpu().numpy())
            
            # Classification true/pred for known classes
            known_mask = batch_y >= 0
            if known_mask.any():
                all_true_known.extend(batch_y[known_mask].cpu().numpy())
                all_pred_known.extend(pred_classes[known_mask].cpu().numpy())
                
            # Collect for t-SNE
            all_features.append(contrast_features.cpu().numpy())
            
            # Fetch true class names using indices
            for idx in batch_idx.cpu().numpy():
                all_true_classes.append(dataset.true_class_names[idx])
                
    y_true_binary = np.array(y_true_binary)
    y_scores = np.array(y_scores)
    all_features = np.concatenate(all_features, axis=0)
    all_true_classes = np.array(all_true_classes)
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # ------------------ PLOT 1: Confusion Matrix ------------------
    print("Plotting Confusion Matrix...")
    cm = confusion_matrix(all_true_known, all_pred_known, labels=range(num_known))
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), row_sums, out=cm_norm, where=row_sums > 0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=known_classes, yticklabels=known_classes)
    plt.title("Closed-Set Classification Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png", dpi=300)
    plt.close()
    
    # ------------------ PLOT 2: ROC Curve ------------------
    print("Plotting ROC Curve...")
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Find the threshold closest to the 0.95 anomaly score (EVT prob = 0.05)
    evt_threshold = 0.95
    idx_thresh = np.argmin(np.abs(thresholds - evt_threshold))
    plt.plot(fpr[idx_thresh], tpr[idx_thresh], 'ro', markersize=8, 
             label=f'EVT Threshold 0.05 (Anomaly Score {evt_threshold})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Knowns classified as Unknown)')
    plt.ylabel('True Positive Rate (Unknowns correctly detected)')
    plt.title('Open-Set Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/roc_curve.png", dpi=300)
    plt.close()
    
    # ------------------ PLOT 3: t-SNE Feature Visualization ------------------
    print("Running t-SNE dimensionality reduction (subsampled for speed)...")
    num_samples = len(all_features)
    subsample_idx = np.random.choice(num_samples, min(2000, num_samples), replace=False)
    features_sub = all_features[subsample_idx]
    classes_sub = all_true_classes[subsample_idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_sub)
    
    plt.figure(figsize=(10, 8))
    unique_classes = sorted(list(set(classes_sub)))
    
    palette = sns.color_palette("tab10", len(unique_classes))
    color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    
    for cls in unique_classes:
        mask = classes_sub == cls
        is_unknown = cls in unknown_classes
        marker = 'x' if is_unknown else 'o'
        alpha = 0.8 if is_unknown else 0.5
        label = f"{cls} (Unknown)" if is_unknown else cls
        
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    label=label, marker=marker, alpha=alpha, s=40)
                    
    plt.title("t-SNE Visualization of Shared Feature Extractor Space (Triplet Loss + EVT)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("plots/tsne_features.png", dpi=300)
    plt.close()
    
    print("Plots generated successfully and saved in the 'plots/' directory!")
    print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    generate_plots()
