import os
import sys
import argparse
import pickle
import numpy as np
import torch
from sklearn.cluster import DBSCAN

def load_rml2016(path, min_snr=0):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    lookup = {}
    for (mod, snr), samples in data.items():
        if snr < min_snr:
            continue
        for sig in samples:
            sig_f32 = sig.astype(np.float32)
            key = sig_f32.tobytes()
            lookup[key] = mod
    return lookup

def recover_true_labels(usb_signals, lookup):
    true_labels = []
    found_mask  = []
    for sig in usb_signals:
        key = sig.astype(np.float32).flatten().tobytes()
        mod = lookup.get(key, None)
        true_labels.append(mod)
        found_mask.append(mod is not None)
    return true_labels, np.array(found_mask)

def cluster_purity(cluster_labels, true_labels_arr):
    noise_mask       = cluster_labels == -1
    valid_mask       = ~noise_mask
    n_total          = len(cluster_labels)
    n_noise          = int(noise_mask.sum())
    n_valid          = int(valid_mask.sum())
    unique_clusters  = sorted(set(cluster_labels[valid_mask]))
    n_clusters       = len(unique_clusters)

    per_cluster = []
    for cid in unique_clusters:
        cmask  = cluster_labels == cid
        c_true = true_labels_arr[cmask]
        classes, counts = np.unique(c_true, return_counts=True)
        dominant_class  = classes[np.argmax(counts)]
        dominant_count  = counts.max()
        purity          = dominant_count / len(c_true)
        per_cluster.append({
            'cluster_id'     : cid,
            'size'           : len(c_true),
            'purity'         : purity,
        })

    if per_cluster:
        sizes           = np.array([c['size']   for c in per_cluster])
        purities        = np.array([c['purity'] for c in per_cluster])
        weighted_purity = float((purities * sizes).sum() / sizes.sum())
    else:
        weighted_purity = 0.0

    return {
        'n_total'         : n_total,
        'noise_fraction'  : n_noise / n_total if n_total else 0.0,
        'n_clusters'      : n_clusters,
        'weighted_purity' : weighted_purity,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='RML2016.10a_dict.pkl')
    parser.add_argument('--split_dir', type=str, default='research_papers/new_method/rml2016/set2_random')
    parser.add_argument('--checkpoint', type=str, default='phase1_model.pth')
    args = parser.parse_args()

    lookup = load_rml2016(args.dataset_path)
    
    ckpt_path = os.path.join(args.split_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    usb_signals  = ckpt.get('usb_signals',  None)
    usb_features = ckpt.get('usb_features', None)
    
    usb_signals_np  = np.array(usb_signals,  dtype=np.float32)
    usb_features_np = np.array(usb_features, dtype=np.float32)
    
    true_labels, found_mask = recover_true_labels(usb_signals_np, lookup)
    features_eval    = usb_features_np[found_mask]
    true_labels_eval = np.array(true_labels)[found_mask]
    
    print(f"Running grid search on {len(features_eval)} valid signals...")
    print(f"{'eps':<8} {'min_samples':<12} {'d_hat':<8} {'Noise%':<10} {'Wtd Purity':<12}")
    print("-" * 55)
    
    results = []
    
    # Grid search ranges
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    min_samples_values = [20, 50, 100, 200, 500]
    
    for e in eps_values:
        for m in min_samples_values:
            db = DBSCAN(eps=e, min_samples=m)
            labels = db.fit_predict(features_eval)
            res = cluster_purity(labels, true_labels_eval)
            
            # Only keep results with a reasonable number of clusters (not 0 or too many)
            # and not completely classifying everything as noise
            if 1 <= res['n_clusters'] <= 10 and res['noise_fraction'] < 0.9:
                results.append((e, m, res['n_clusters'], res['noise_fraction'], res['weighted_purity']))
                
    # Sort by weighted purity descending
    results.sort(key=lambda x: x[4], reverse=True)
    
    for r in results:
        print(f"{r[0]:<8.2f} {r[1]:<12d} {r[2]:<8d} {r[3]*100:<9.1f}% {r[4]*100:<11.1f}%")

if __name__ == '__main__':
    main()
