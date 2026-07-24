import torch
import numpy as np
import argparse
import os
from data.dataset import RadioMLDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase1_model.pth')
    parser.add_argument('--dataset', type=str, default='RML2016.10a_dict.pkl')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
        
    print(f"Loading dataset: {args.dataset}")
    # Load all classes to get true labels. 
    # By default, RadioMLDataset loads all modulations if known_classes is None
    dataset = RadioMLDataset(args.dataset)
    
    print("Loading all dataset signals into memory for comparison...")
    # RadioMLDataset's .data is already a numpy array of shape (N, 2, 128)
    # We flatten the spatial dimensions to compare them easily
    dataset_signals = dataset.data.reshape(len(dataset), -1)
    dataset_labels = dataset.labels
        
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    if 'usb_signals' not in checkpoint:
        print("No usb_signals found in checkpoint.")
        return
        
    usb_signals = checkpoint['usb_signals']
    known_classes = checkpoint.get('known_classes', [])
    # RadioMLDataset assigns labels based on the sorted list of all classes found
    all_classes = dataset.known_classes 
    
    label_counts = {}
    total_found = 0
    not_found = 0
    
    print(f"Analyzing {len(usb_signals)} USB signals against {len(dataset_signals)} dataset signals...")
    
    # Reshape USB signals for broadcasting
    # usb_signals is list or array of shape (M, 2, 128) -> (M, 256)
    usb_signals_flat = np.array(usb_signals).reshape(len(usb_signals), -1)
    
    for i, sig in enumerate(usb_signals_flat):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{len(usb_signals)} signals...")
            
        # Compute Mean Absolute Error (MAE) between this USB signal and all dataset signals
        # This is robust to minor floating point serialization differences
        mae = np.mean(np.abs(dataset_signals - sig), axis=1)
        
        # Find the best match
        best_match_idx = np.argmin(mae)
        min_error = mae[best_match_idx]
        
        # Threshold for considering it a match
        if min_error < 1e-4:
            label_idx = dataset_labels[best_match_idx]
            label_name = all_classes[label_idx]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
            total_found += 1
        else:
            not_found += 1
            
    print("\n--- True Distribution of Signals in the USB ---")
    
    # Sort by count descending
    sorted_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    total_usb = len(usb_signals)
    
    for label_name, count in sorted_counts:
        status = "KNOWN" if label_name in known_classes else "NOVEL"
        pct = (count / total_usb) * 100
        print(f"{label_name:<10} ({status}): {count} samples ({pct:.2f}%)")
        
    if not_found > 0:
        print(f"\nWarning: {not_found} signals in USB could not be exactly matched in the dataset.")
        print("This could be due to floating point precision changes during serialization.")

if __name__ == '__main__':
    main()
