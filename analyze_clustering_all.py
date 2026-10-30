#!/usr/bin/env python
"""
analyze_clustering_all.py
=========================
Runs the three clustering implementations used in the paper on the SAME Unknown
Signal Bank (USB) and measures how well the discovered groups correspond to the
true unseen modulation classes.

One job, one USB, three algorithms, so the comparison is controlled.

The three implementations are transcribed verbatim from the branch that produced
the corresponding row of the results tables:

    kmeans   <- openicl_kmeans      utils/usb.py  (Silhouette-guided k in [2,15])
    dbscan   <- openicl_clustering  utils/usb.py  (eps=0.5, min_samples=100)
    hdbscan  <- openicl_hdbscan     utils/usb.py  (min_cluster_size grid,
                                                   silhouette*(1-noise), 0.15)

Each is applied to the USB contrast features stored in a phase-1 checkpoint, and
returns the group labels, the number of groups, and which samples were retained
(DBSCAN and HDBSCAN delete noise; K-Means keeps everything).

Reported per checkpoint and per method:
    d_hat  : number of groups discovered
    keep   : % of USB samples retained
    purity : size-weighted fraction of the dominant TRUE unseen class per group
    ACC    : clustering accuracy with Hungarian matching to the true classes
    NMI    : normalised mutual information

True labels are recovered by matching each stored USB signal back to the dataset,
so labels are used for EVALUATION ONLY and never inside the clustering.

Usage (CPU only, no GPU needed):

    python analyze_clustering_all.py --split_dir checkpoints/set1_random \
        --dataset_type rml2016 --dataset_path RML2016.10a_dict.pkl

Same --split_dir / --seeds convention as run_experiments.py. Drive it with
run_clustering_purity_cc.sh for one slurm array task per split.
"""

import argparse
import glob
import hashlib
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

RML2018_MODS = ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
                'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64',
                'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC',
                'FM', 'GMSK', 'OQPS']


def log(m=""):
    print(m, flush=True)


def banner(m, ch="="):
    log(""); log(ch * 78); log(m); log(ch * 78)


# ===========================================================================
#  Clustering implementations, transcribed from each branch's utils/usb.py
# ===========================================================================

def discover_kmeans(features_np, max_k=15, random_state=42, sil_sample=0):
    """openicl_kmeans : Silhouette-guided k search over [2, max_k], argmax."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(features_np)
    if n < 10:
        return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    best_score = -1.0
    best_labels = None
    best_n_clusters = 1
    upper = min(max_k, n - 1)

    for k in range(2, upper + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = km.fit_predict(features_np)
        try:
            if sil_sample and sil_sample < n:
                score = silhouette_score(features_np, labels,
                                         sample_size=sil_sample, random_state=42)
            else:
                score = silhouette_score(features_np, labels)
        except ValueError:
            continue
        log(f"     k={k:>2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_labels = labels
            best_n_clusters = k

    if best_labels is None:
        return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    # K-Means assigns every point to a cluster: nothing is filtered
    return best_labels, best_n_clusters, np.ones(n, dtype=bool)


def discover_dbscan(features_np, eps=0.5, min_samples=100):
    """openicl_clustering : DBSCAN(eps, min_samples), noise dropped."""
    from sklearn.cluster import DBSCAN

    n = len(features_np)
    if n < 10:
        return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features_np)
    mask = labels != -1
    n_clusters = len(set(labels[mask]))

    if n_clusters <= 1:
        # branch fallback: single class, and no noise filtering is applied
        return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    return labels[mask], n_clusters, mask


def discover_hdbscan(features_np, min_cluster_size=None, min_samples=None):
    """openicl_hdbscan : min_cluster_size grid, score = silhouette*(1-noise), 0.15."""
    from sklearn.metrics import silhouette_score
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:  # older scikit-learn
        from hdbscan import HDBSCAN

    n = len(features_np)
    if n <= 50:
        return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    # ---- fixed parameter mode ----
    if min_cluster_size is not None:
        if n < min_cluster_size:
            return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)
        labels = HDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples).fit_predict(features_np)
        mask = labels != -1
        if mask.sum() == 0:
            return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)
        return labels[mask], len(set(labels[mask])), mask

    # ---- dynamic Silhouette-guided search ----
    candidate_sizes = [50, 100, 200, 300]
    sizes = [m for m in candidate_sizes if m < n]

    best_score = -float('inf')
    best_labels = None
    best_n_clusters = 1
    min_1cluster_noise = float('inf')
    best_1cluster_labels = None

    for m in sizes:
        labels = HDBSCAN(min_cluster_size=m,
                         min_samples=min_samples).fit_predict(features_np)
        mask = labels != -1
        valid = mask.sum()
        if valid == 0:
            continue
        noise_fraction = 1.0 - valid / len(labels)
        n_clusters = len(set(labels[mask]))

        if n_clusters == 1 and noise_fraction < min_1cluster_noise:
            min_1cluster_noise = noise_fraction
            best_1cluster_labels = labels
        elif n_clusters >= 2 and noise_fraction < 0.60:
            s = silhouette_score(features_np[mask], labels[mask])
            adjusted = s * (1.0 - noise_fraction)
            log(f"     mcs={m:>4d}  n_clusters={n_clusters:>3d}  "
                f"noise={noise_fraction:.2%}  score={adjusted:.4f}")
            if adjusted > best_score:
                best_score = adjusted
                best_labels = labels
                best_n_clusters = n_clusters

    if best_labels is None or best_score < 0.15:
        if best_1cluster_labels is not None:
            best_labels = best_1cluster_labels
            best_n_clusters = 1
        else:
            return np.zeros(n, dtype=int), 1, np.ones(n, dtype=bool)

    mask = best_labels != -1
    return best_labels[mask], best_n_clusters, mask


METHODS = {
    'kmeans': discover_kmeans,
    'dbscan': discover_dbscan,
    'hdbscan': discover_hdbscan,
}


# ===========================================================================
#  Label recovery (evaluation only)
# ===========================================================================

def _bytes_c(x):
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32)).tobytes()


def build_lookup_rml2016(path, min_snr=0):
    log(f"  loading {path} ...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    lut = {}
    for (mod, snr), samples in data.items():
        if snr < min_snr:
            continue
        for sig in samples:
            lut[_bytes_c(sig)] = mod
    log(f"  lookup: {len(lut):,} signals")
    return lut


def build_lookup_rml2018(path, min_snr=0, chunk=20000):
    import h5py
    log(f"  loading {path} (streaming) ...")
    lut = {}
    with h5py.File(path, 'r') as h:
        X, Y, Z = h['X'], h['Y'], h['Z']
        n = X.shape[0]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            xs = X[s:e]
            zs = np.asarray(Z[s:e]).reshape(-1)
            ys = np.argmax(np.asarray(Y[s:e]), axis=1)
            for j in range(e - s):
                if zs[j] < min_snr:
                    continue
                arr = np.ascontiguousarray(np.transpose(xs[j], (1, 0)), dtype=np.float32)
                lut[hashlib.blake2b(arr.tobytes(), digest_size=16).digest()] = RML2018_MODS[ys[j]]
            if (s // chunk) % 25 == 0:
                log(f"    ... {e:,}/{n:,}")
    log(f"  lookup: {len(lut):,} signals")
    return lut


def recover(signals, lut, dtype):
    out, found = [], []
    for sig in signals:
        if dtype == 'rml2018':
            arr = np.ascontiguousarray(np.asarray(sig, dtype=np.float32))
            if arr.ndim > 1 and arr.shape[0] != 2:
                arr = np.ascontiguousarray(np.transpose(arr, (1, 0)))
            key = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
        else:
            key = _bytes_c(sig)
        m = lut.get(key)
        out.append(m)
        found.append(m is not None)
    return np.asarray(out, dtype=object), np.asarray(found, dtype=bool)


# ===========================================================================
#  Metrics
# ===========================================================================

def cluster_metrics(labels, true_labels, unknown_mask):
    """Purity / ACC / NMI computed on TRUE-UNKNOWN samples."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    res = dict(d_hat=int(len(set(labels))), purity=float('nan'),
               acc=float('nan'), nmi=float('nan'), ari=float('nan'), n_eval=0)

    keep = unknown_mask
    if keep.sum() < 20:
        return res
    lp, lt = labels[keep], true_labels[keep]
    res['n_eval'] = int(keep.sum())

    tot = hit = 0
    for c in np.unique(lp):
        m = lp == c
        if m.any():
            _, cnt = np.unique(lt[m], return_counts=True)
            tot += int(m.sum())
            hit += int(cnt.max())
    res['purity'] = hit / tot if tot else float('nan')

    t_idx = {v: i for i, v in enumerate(sorted(set(lt.tolist())))}
    c_idx = {v: i for i, v in enumerate(sorted(set(lp.tolist())))}
    cm = np.zeros((len(t_idx), len(c_idx)), dtype=np.int64)
    for a, b in zip(lt, lp):
        cm[t_idx[a], c_idx[b]] += 1
    r, c = linear_sum_assignment(-cm)
    res['acc'] = float(cm[r, c].sum() / cm.sum())
    res['nmi'] = float(normalized_mutual_info_score(lt, lp))
    res['ari'] = float(adjusted_rand_score(lt, lp))
    return res


# ===========================================================================
#  Per checkpoint
# ===========================================================================

def analyze(path, split_name, seed, args, lut_cache):
    import torch

    ck = torch.load(path, map_location='cpu', weights_only=False)
    known = list(ck.get('known_classes', []))
    dtype = args.dataset_type or ck.get('dataset_type') or 'rml2016'
    usb_sig, usb_feat = ck.get('usb_signals'), ck.get('usb_features')

    banner(f"CHECKPOINT  split={split_name}  seed={seed}\n{path}", "-")
    if not usb_sig:
        log("  [SKIP] no USB stored in this checkpoint"); return None

    F = np.asarray(usb_feat, dtype=np.float32)
    n_orig = len(F)
    log(f"  known classes : {known}")
    log(f"  USB samples   : {n_orig:,}   dim={F.shape[1]}")

    key = (dtype, args.dataset_path)
    if key not in lut_cache:
        lut_cache[key] = (build_lookup_rml2018(args.dataset_path)
                          if dtype == 'rml2018' else build_lookup_rml2016(args.dataset_path))
    true_all, found = recover(usb_sig, lut_cache[key], dtype)
    log(f"  label recovery: {found.sum():,}/{len(found):,} ({found.mean()*100:.1f}%)")
    if found.sum() < 50:
        log("  [SKIP] too few matched signals"); return None

    # evaluate only on matched signals; all methods see the same subset
    F = F[found]
    true_all = true_all[found]
    known_set = set(known)
    unk = np.array([t not in known_set for t in true_all])
    leak = 1.0 - unk.mean()
    vals, cnt = np.unique(true_all, return_counts=True)
    n_unk_present = int(sum(1 for v in vals if v not in known_set))
    true_d = (11 - len(known)) if dtype == 'rml2016' else (24 - len(known))
    log(f"  composition   : {len(vals)} true classes, {n_unk_present}/{true_d} "
        f"unseen present, dominant share {cnt.max()/len(true_all)*100:.1f}%, "
        f"KNOWN leakage {leak*100:.1f}%")

    rows = {}
    for name in [m.strip() for m in args.methods.split(',') if m.strip()]:
        if name not in METHODS:
            log(f"  [warn] unknown method '{name}', skipping"); continue
        log("")
        log(f"  --- {name} ---")
        try:
            labels, d_hat, mask = METHODS[name](F, sil_sample=args.sil_sample) \
                if name == 'kmeans' else METHODS[name](F)
            m = cluster_metrics(labels, true_all[mask], unk[mask])
            m['keep'] = float(mask.mean())
            m['d_reported'] = int(d_hat)
            rows[name] = m
        except Exception as e:
            log(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    log("")
    log(f"  {'method':<10} {'d_hat':>6} {'keep%':>8} {'purity':>8} {'ACC':>8} "
        f"{'NMI':>8} {'ARI':>8} {'n_eval':>8}")
    log("  " + "-" * 70)
    for name, m in rows.items():
        log(f"  {name:<10} {m['d_hat']:>6} {m['keep']*100:>7.1f}% "
            f"{m['purity']:>8.4f} {m['acc']:>8.4f} {m['nmi']:>8.4f} "
            f"{m['ari']:>8.4f} {m['n_eval']:>8}")

    return dict(seed=seed, rows=rows, n_unk_present=n_unk_present,
                leak=leak, true_d=true_d, usb=n_orig)


# ===========================================================================
#  Summary
# ===========================================================================

def summarize(all_res, split_name, seeds):
    banner(f"FINAL AVERAGED RESULTS  (split: {split_name})")
    if not all_res:
        log("No checkpoint produced a result."); return

    methods = [m for m in METHODS if any(m in r['rows'] for r in all_res)]

    log(f"Averaged over seeds: {seeds}   ({len(all_res)} checkpoint(s))")
    log("")
    log(f"  {'method':<10} {'d_hat':>7} {'keep%':>8} {'purity':>8} "
        f"{'ACC':>8} {'NMI':>8}")
    log("  " + "-" * 56)
    for m in methods:
        rs = [r['rows'][m] for r in all_res if m in r['rows']]
        if not rs:
            continue
        log(f"  {m:<10} {np.mean([x['d_hat'] for x in rs]):>7.2f} "
            f"{np.mean([x['keep'] for x in rs])*100:>7.1f}% "
            f"{np.mean([x['purity'] for x in rs]):>8.4f} "
            f"{np.mean([x['acc'] for x in rs]):>8.4f} "
            f"{np.mean([x['nmi'] for x in rs]):>8.4f}")

    log("")
    log("  per seed (d_hat / purity):")
    for r in all_res:
        parts = []
        for m in methods:
            if m in r['rows']:
                x = r['rows'][m]
                parts.append(f"{m}={x['d_hat']}/{x['purity']:.3f}")
        log(f"    seed {r['seed']:<5} " + "  ".join(parts))

    log("")
    log(f"  True number of unseen classes d : {all_res[0]['true_d']}")
    log(f"  Unseen classes present in USB   : "
        f"{np.mean([r['n_unk_present'] for r in all_res]):.1f}")
    log(f"  KNOWN-class leakage in USB      : "
        f"{np.mean([r['leak'] for r in all_res])*100:.1f}%")
    log("")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--split_dir', required=True,
                   help='base dir for one split, e.g. checkpoints/set1_random '
                        '(same convention as run_experiments.py)')
    p.add_argument('--dataset_type', default='', choices=['', 'rml2016', 'rml2018'])
    p.add_argument('--dataset_path', default='RML2016.10a_dict.pkl')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--methods', default='kmeans,dbscan,hdbscan')
    p.add_argument('--sil_sample', type=int, default=0,
                   help='subsample for the K-Means Silhouette (0 = exact, matching '
                        'the branch code; set e.g. 10000 if the job is too slow)')
    p.add_argument('--list_only', action='store_true')
    args = p.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    split_name = os.path.basename(os.path.normpath(args.split_dir))

    work = []
    for s in seeds:
        hit = sorted(glob.glob(os.path.join(
            args.split_dir, f"seed_{s}", "phase1_model*.pth")))
        if hit:
            work.append((hit[0], str(s)))
    if not work:
        for h in sorted(glob.glob(os.path.join(args.split_dir, "phase1_model*.pth"))):
            work.append((os.path.normpath(h), '?'))

    banner(f"SPLIT {split_name}: found {len(work)} phase-1 checkpoint(s)")
    for path, sd in work:
        log(f"  seed={sd:<5} {path}")
    if not work:
        log("none found. Expected <split_dir>/seed_<42|43|44>/phase1_model.pth")
        sys.exit(1)
    if args.list_only:
        return

    lut_cache, out = {}, []
    for path, sd in work:
        log("")
        log("=" * 78)
        log(f"  Seed {sd}")
        log("=" * 78)
        try:
            r = analyze(path, split_name, sd, args, lut_cache)
            if r:
                out.append(r)
        except Exception as e:
            log(f"  [ERROR] {path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    summarize(out, split_name, seeds)


if __name__ == '__main__':
    main()
