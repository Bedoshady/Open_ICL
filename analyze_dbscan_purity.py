"""
analyze_dbscan_purity.py
========================
Loads the phase1_model.pth checkpoint from every split folder, extracts
the USB features, re-runs DBSCAN with the same hyperparameters as training
(eps=0.5, min_samples=100), then recovers the *true* modulation label for
each USB signal by matching it back to the RML2016 dataset.

Metrics reported per split
--------------------------
  - Total USB signals collected
  - Noise fraction  (fraction flagged as -1 by DBSCAN; not used for learning)
  - Discovered clusters  d_hat
  - Per-cluster purity   (fraction of cluster belonging to the dominant class)
  - Mean cluster purity  (average over all clusters)
  - Weighted mean purity (weighted by cluster size)

True labels are used *only* for evaluation, never during clustering.

Usage
-----
    python analyze_dbscan_purity.py \
        --dataset_path RML2016.10a_dict.pkl \
        --splits_root research_papers/new_method/rml2016 \
        [--eps 0.5] [--min_samples 100] [--min_snr 0]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_rml2016(path, min_snr=0):
    """
    Returns a lookup dict: raw bytes of flattened float32 signal -> mod string.
    """
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
    """
    For each USB signal, find its true modulation class by exact-byte match.

    Returns (true_labels list[str|None], found_mask bool array)
    """
    true_labels = []
    found_mask  = []
    for sig in usb_signals:
        key = sig.astype(np.float32).flatten().tobytes()
        mod = lookup.get(key, None)
        true_labels.append(mod)
        found_mask.append(mod is not None)
    return true_labels, np.array(found_mask)


def cluster_purity(cluster_labels, true_labels_arr):
    """
    Compute per-cluster purity given DBSCAN labels (noise = -1).
    """
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
            'dominant_class' : dominant_class,
            'dominant_count' : dominant_count,
            'purity'         : purity,
            'class_breakdown': dict(zip(classes.tolist(), counts.tolist())),
        })

    if per_cluster:
        sizes           = np.array([c['size']   for c in per_cluster])
        purities        = np.array([c['purity'] for c in per_cluster])
        mean_purity     = float(purities.mean())
        weighted_purity = float((purities * sizes).sum() / sizes.sum())
    else:
        mean_purity = weighted_purity = float('nan')

    return {
        'n_total'         : n_total,
        'n_noise'         : n_noise,
        'noise_fraction'  : n_noise / n_total if n_total else float('nan'),
        'n_valid'         : n_valid,
        'valid_fraction'  : n_valid / n_total if n_total else float('nan'),
        'n_clusters'      : n_clusters,
        'mean_purity'     : mean_purity,
        'weighted_purity' : weighted_purity,
        'per_cluster'     : per_cluster,
    }


def print_split_summary(split_name, result):
    print(f"\n{'='*70}")
    print(f"  Split: {split_name}")
    print(f"  Known classes  : {result.get('known_classes', [])}")
    print(f"  Unknown (true) : {result.get('discovered_classes', [])}")
    print(f"{'='*70}")
    print(f"  USB signals total   : {result['n_total']}")
    print(f"  Noise (not used)    : {result['n_noise']}  ({result['noise_fraction']*100:.1f}%)")
    print(f"  Used in clusters    : {result['n_valid']}  ({result['valid_fraction']*100:.1f}%)")
    print(f"  Discovered clusters : {result['n_clusters']}")
    print(f"  Mean cluster purity : {result['mean_purity']*100:.1f}%")
    print(f"  Weighted purity     : {result['weighted_purity']*100:.1f}%")
    print()
    for c in result['per_cluster']:
        breakdown_str = ', '.join(
            f"{cls}: {cnt}"
            for cls, cnt in sorted(c['class_breakdown'].items(), key=lambda x: -x[1])
        )
        print(f"  Cluster {c['cluster_id']:>2d}  "
              f"size={c['size']:>6d}  "
              f"dominant={c['dominant_class']:>12s}  "
              f"purity={c['purity']*100:5.1f}%  "
              f"[ {breakdown_str} ]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyse DBSCAN clustering purity on USB embeddings (evaluation only)')
    parser.add_argument('--dataset_path', type=str, default='RML2016.10a_dict.pkl',
                        help='Path to RML2016.10a_dict.pkl')
    parser.add_argument('--splits_root', type=str,
                        default=r'research_papers/new_method/rml2016',
                        help='Root folder containing set1_random, set2_random, ... subdirs')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='DBSCAN eps (same as used during training, default=0.5)')
    parser.add_argument('--min_samples', type=int, default=100,
                        help='DBSCAN min_samples (default=100)')
    parser.add_argument('--min_snr', type=float, default=0,
                        help='Minimum SNR threshold used during training (default=0)')
    parser.add_argument('--checkpoint', type=str, default='phase1_model.pth',
                        choices=['phase1_model.pth', 'phase2_incremental_model.pth'],
                        help='Checkpoint file to load USB from (default: phase1_model.pth)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Build dataset lookup table (used only for label recovery)
    # ------------------------------------------------------------------
    if not os.path.exists(args.dataset_path):
        sys.exit(f"ERROR: Dataset not found at '{args.dataset_path}'.\n"
                 f"Run from the project root or pass --dataset_path explicitly.")

    print(f"Loading dataset from: {args.dataset_path} (min_snr={args.min_snr})")
    lookup = load_rml2016(args.dataset_path, min_snr=args.min_snr)
    print(f"  Lookup built: {len(lookup)} unique signals indexed.")

    # ------------------------------------------------------------------
    # 2. Iterate over split folders
    # ------------------------------------------------------------------
    splits_root = args.splits_root
    split_dirs  = sorted([
        d for d in os.listdir(splits_root)
        if os.path.isdir(os.path.join(splits_root, d))
    ])

    if not split_dirs:
        sys.exit(f"No split directories found under '{splits_root}'.")

    all_results = {}

    for split_name in split_dirs:
        ckpt_path = os.path.join(splits_root, split_name, args.checkpoint)
        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] {split_name}: '{args.checkpoint}' not found.")
            continue

        print(f"\nProcessing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # ---- Extract USB -----------------------------------------------
        usb_signals  = ckpt.get('usb_signals',  None)
        usb_features = ckpt.get('usb_features', None)

        if not usb_signals or usb_features is None:
            print(f"  [SKIP] No USB data in checkpoint.")
            continue

        usb_signals_np  = np.array(usb_signals,  dtype=np.float32)
        usb_features_np = np.array(usb_features, dtype=np.float32)
        print(f"  USB: {len(usb_signals_np)} signals,  feature_dim={usb_features_np.shape[1]}")

        # ---- Recover true labels by matching raw IQ to dataset ---------
        true_labels, found_mask = recover_true_labels(usb_signals_np, lookup)
        n_found = found_mask.sum()
        pct     = n_found / len(usb_signals_np) * 100
        print(f"  True-label recovery: {n_found}/{len(usb_signals_np)} matched ({pct:.1f}%)")

        if n_found < 10:
            print("  [SKIP] Too few matched signals.")
            continue

        # Evaluate only on matched signals
        features_eval    = usb_features_np[found_mask]
        true_labels_eval = np.array(true_labels)[found_mask]

        # ---- Run DBSCAN (evaluation only) ------------------------------
        print(f"  DBSCAN  eps={args.eps}  min_samples={args.min_samples} ...")
        db     = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        labels = db.fit_predict(features_eval)

        # ---- Purity analysis -------------------------------------------
        result = cluster_purity(labels, true_labels_eval)
        result['known_classes']      = ckpt.get('known_classes', [])
        result['discovered_classes'] = ckpt.get('discovered_classes', [])
        all_results[split_name]      = result

        print_split_summary(split_name, result)

    # ------------------------------------------------------------------
    # 3. Aggregate summary table
    # ------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*80}")
    header = f"{'Split':<18} {'USB':>7} {'Noise%':>8} {'Used%':>7} {'d_hat':>6} {'MeanPurity':>12} {'WtdPurity':>11}"
    print(header)
    print('-' * 80)
    for split_name, r in all_results.items():
        print(f"{split_name:<18} "
              f"{r['n_total']:>7} "
              f"{r['noise_fraction']*100:>7.1f}% "
              f"{r['valid_fraction']*100:>6.1f}% "
              f"{r['n_clusters']:>6} "
              f"{r['mean_purity']*100:>11.1f}% "
              f"{r['weighted_purity']*100:>10.1f}%")

    print(f"\nAnalysed {len(all_results)} split(s). Done.")


if __name__ == '__main__':
    main()
