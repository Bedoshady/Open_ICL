#!/usr/bin/env python
"""
analyze_k_selection.py
======================
Diagnostic: which criterion should choose the number of novel classes (d_hat)
from the Unknown Signal Bank (USB) embeddings?

This is READ-ONLY. It does not train anything and does not modify checkpoints.

It walks the checkpoint tree the same way run_experiments.py / run_cc_experiments.sh
create it, i.e.:

    <root>/<split_name>/seed_<seed>/phase1_model.pth      (current layout)
    <root>/<split_name>/phase1_model.pth                  (older layout)

For every phase-1 checkpoint it loads the stored USB embeddings, recovers the
TRUE modulation label of each USB signal by matching it back to the dataset, and
for k = k_min .. k_max reports:

  (a) SELECTORS  - what you would pick WITHOUT labels
        sil_euc  : silhouette, Euclidean, raw 128-D  <-- what utils/usb.py uses now
        sil_cos  : silhouette, cosine distance
        sil_pca  : silhouette, Euclidean, on PCA-reduced features
        bic      : Gaussian mixture BIC (lower is better)
        gap      : Tibshirani gap statistic
        ps       : prediction strength (stability under a train/test split)

  (b) GROUND TRUTH - how good that k actually is (labels used for EVALUATION ONLY)
        purity   : weighted purity (dominant true class per cluster)
        acc      : clustering accuracy with Hungarian matching
        nmi, ari : clustering agreement
        cov      : fraction of USB points assigned to a cluster

The point: find the selector whose pick lands closest to the ORACLE k (the k with
the best ground-truth ACC). That tells you what to put into utils/usb.py instead
of `argmax silhouette`.

Usage (Compute Canada - CPU is enough, no GPU needed):

    module load StdEnv/2023
    module load python/3.12
    module load scipy-stack
    source myDRLenv/bin/activate

    python analyze_k_selection.py \
        --root checkpoints \
        --dataset_type rml2018 \
        --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5 \
        --k_min 2 --k_max 15

    # or point it at an explicit list of split dirs / checkpoints
    python analyze_k_selection.py \
        --root checkpoints/set1_2018 --root checkpoints/set2_2018 \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5

    # quick smoke test (skips the slow selectors, few k)
    python analyze_k_selection.py --root checkpoints/set1_2018 \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5 \
        --k_min 2 --k_max 6 --skip_gap --skip_ps

Outputs plain text to stdout (so it ends up in the slurm log) and optionally a CSV.
"""

import argparse
import glob
import hashlib
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np

# Old RML2016 pkl files trip a numpy>=2.4 dtype(align=) deprecation warning and
# sklearn emits convergence chatter; keep the slurm logs readable.
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- utilities

def log(msg=""):
    print(msg, flush=True)


def banner(msg, ch="="):
    log("")
    log(ch * 78)
    log(msg)
    log(ch * 78)


def human(n):
    return f"{n:,}"


# ------------------------------------------------------- checkpoint discovery

def find_checkpoints(roots, filename_glob="phase1_model*.pth"):
    """Return a sorted list of checkpoint paths found under the given roots."""
    found = []
    for root in roots:
        if os.path.isfile(root):
            found.append(root)
            continue
        if not os.path.isdir(root):
            log(f"  [WARN] root not found, skipping: {root}")
            continue
        pattern = os.path.join(root, "**", filename_glob)
        found.extend(glob.glob(pattern, recursive=True))
    # newest layout first is not needed; sort for stable logs
    return sorted(set(os.path.normpath(p) for p in found))


def describe_ckpt(path):
    """Infer (split_name, seed) from a checkpoint path for readable logs."""
    parts = os.path.normpath(path).split(os.sep)
    seed = "?"
    for p in reversed(parts):
        if p.startswith("seed_"):
            seed = p.replace("seed_", "")
            break
    # split name = directory just above seed_XX, else the parent dir
    if seed != "?":
        idx = parts.index(f"seed_{seed}")
        split = parts[idx - 1] if idx >= 1 else "?"
    else:
        split = parts[-2] if len(parts) >= 2 else "?"
    return split, seed


# ------------------------------------------------------------ label recovery

RML2018_MODS = ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
                'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64',
                'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC', 'AM_DSB_SC',
                'FM', 'GMSK', 'OQPS']


def _sig_bytes(sig):
    """C-order float32 bytes of one USB signal, matching dataset.py's tensor."""
    return np.ascontiguousarray(np.asarray(sig, dtype=np.float32)).tobytes()


def build_lookup_rml2016(path, min_snr=0):
    """raw float32 bytes -> modulation name, for RML2016.10a .pkl"""
    log(f"  loading {path} (this takes ~1 min for the 640MB pkl) ...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    lookup = {}
    for (mod, snr), samples in data.items():
        if snr < min_snr:
            continue
        for sig in samples:
            lookup[_sig_bytes(sig)] = mod
    log(f"  lookup built: {human(len(lookup))} signals")
    return lookup


def build_lookup_rml2018(path, min_snr=0, chunk=20000):
    """
    blake2b digest of the transposed sample -> modulation name, for RML2018.01a.
    dataset.py does: x = transpose(X[i], (1,0)) -> FloatTensor, so we reproduce
    exactly that byte layout. Digest instead of raw bytes keeps memory sane.
    """
    import h5py
    log(f"  loading {path} (streaming, this can take a few minutes) ...")
    lookup = {}
    with h5py.File(path, 'r') as h:
        X, Y, Z = h['X'], h['Y'], h['Z']
        n = X.shape[0]
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xs = X[start:end]
            zs = np.asarray(Z[start:end]).reshape(-1)
            ys = np.argmax(np.asarray(Y[start:end]), axis=1)
            for j in range(end - start):
                if zs[j] < min_snr:
                    continue
                arr = np.ascontiguousarray(np.transpose(xs[j], (1, 0)), dtype=np.float32)
                lookup[hashlib.blake2b(arr.tobytes(), digest_size=16).digest()] = RML2018_MODS[ys[j]]
            if (start // chunk) % 20 == 0:
                log(f"    ... {human(end)}/{human(n)}")
    log(f"  lookup built: {human(len(lookup))} signals")
    return lookup


def recover_labels(signals, lookup, dataset_type):
    """Return (labels list[str|None], found_mask bool array)."""
    labels, found = [], []
    if dataset_type == 'rml2018':
        for sig in signals:
            # dataset.py stored transpose(X[i], (1,0)) -> (2, 1024); that is what
            # build_lookup_rml2018 hashed, so no further transpose is needed.
            arr = np.ascontiguousarray(np.asarray(sig, dtype=np.float32))
            if arr.ndim > 1 and arr.shape[0] != 2:
                arr = np.ascontiguousarray(np.transpose(arr, (1, 0)))
            d = hashlib.blake2b(arr.tobytes(), digest_size=16).digest()
            m = lookup.get(d)
            labels.append(m)
            found.append(m is not None)
    else:
        for sig in signals:
            m = lookup.get(_sig_bytes(sig))
            labels.append(m)
            found.append(m is not None)
    return labels, np.asarray(found, dtype=bool)


# ----------------------------------------------------------- k-selection rules

def kmeans_labels(F, k, seed=42):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(F)


def sel_silhouette(F, k, metric='euclidean', sample_size=None, seed=42):
    from sklearn.metrics import silhouette_score
    lab = kmeans_labels(F, k, seed)
    if sample_size and sample_size < len(F):
        return silhouette_score(F, lab, metric=metric,
                                sample_size=sample_size, random_state=seed)
    return silhouette_score(F, lab, metric=metric)


def sel_silhouette_pca(F, k, pca_dim=20, sample_size=None, seed=42):
    from sklearn.decomposition import PCA
    d = min(pca_dim, F.shape[1], F.shape[0] - 1)
    Fr = PCA(n_components=d, random_state=seed).fit_transform(F)
    return sel_silhouette(Fr, k, 'euclidean', sample_size, seed), Fr


def sel_gmm_bic(F, k, seed=42):
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, covariance_type='diag',
                         random_state=seed, max_iter=200, n_init=1)
    gm.fit(F)
    return gm.bic(F)


def _within_dispersion(F, labels):
    tot = 0.0
    for c in np.unique(labels):
        m = labels == c
        if m.sum() > 1:
            tot += ((F[m] - F[m].mean(axis=0)) ** 2).sum()
        elif m.sum() == 1:
            pass
    return tot


def sel_gap(F, k, refs=5, seed=42, fit_sample=0):
    """
    Tibshirani gap statistic. Returns (gap_k, s_k) so the caller can apply the
    standard 'smallest k with Gap(k) >= Gap(k+1) - s_{k+1}' rule.
    """
    rng = np.random.default_rng(seed)
    if fit_sample and fit_sample < len(F):
        idx = rng.choice(len(F), fit_sample, replace=False)
        Ff = F[idx]
    else:
        Ff = F
    lab = kmeans_labels(Ff, k, seed)
    logW = np.log(_within_dispersion(Ff, lab) + 1e-12)

    mins, maxs = Ff.min(axis=0), Ff.max(axis=0)
    ref_logW = []
    for b in range(refs):
        R = rng.uniform(mins, maxs, size=Ff.shape).astype(np.float32)
        rlab = kmeans_labels(R, k, seed + 1000 + b)
        ref_logW.append(np.log(_within_dispersion(R, rlab) + 1e-12))
    ref_logW = np.asarray(ref_logW)
    gap = ref_logW.mean() - logW
    s = ref_logW.std(ddof=1) * np.sqrt(1.0 + 1.0 / refs) if refs > 1 else 0.0
    return float(gap), float(s)


def sel_prediction_strength(F, k, repeats=5, max_pts=2000, seed=42):
    """
    Tibshirani & Walther prediction strength. Cluster half the data, assign the
    other half to the nearest centroid, then check whether pairs of test points
    that land in the same cluster are within the training cluster's own spread.
    Returns the mean over repeats of min_j PS_j.
    """
    rng = np.random.default_rng(seed)
    scores = []
    n = len(F)
    for r in range(repeats):
        perm = rng.permutation(n)
        half = n // 2
        tr, te = F[perm[:half]], F[perm[half:2 * half]]
        km = kmeans_labels(tr, k, seed + r)
        from sklearn.metrics import pairwise_distances
        centers = np.vstack([tr[km == c].mean(axis=0) for c in range(k)])
        assign = pairwise_distances(te, centers).argmin(axis=1)

        # training-cluster spread: max pairwise distance inside each training cluster
        max_tr = np.zeros(k)
        for c in range(k):
            pts = tr[km == c]
            if len(pts) > 1:
                max_tr[c] = pairwise_distances(pts).max()

        ps_j = []
        for c in range(k):
            pts = te[assign == c]
            if len(pts) > max_pts:
                pts = pts[rng.choice(len(pts), max_pts, replace=False)]
            if len(pts) < 2:
                continue
            D = pairwise_distances(pts)
            iu = np.triu_indices(len(pts), k=1)
            ps_j.append(float((D[iu] <= max_tr[c]).mean()))
        if ps_j:
            scores.append(min(ps_j))
    return float(np.mean(scores)) if scores else np.nan


# --------------------------------------------------------- ground-truth grade

def grade(labels_pred, labels_true, unknown_mask):
    """Ground-truth clustering quality, computed on the true-UNKNOWN points only."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from scipy.optimize import linear_sum_assignment

    lp = np.asarray(labels_pred)[unknown_mask]
    lt = np.asarray(labels_true)[unknown_mask]

    out = dict(n=int(unknown_mask.sum()), purity=np.nan, acc=np.nan,
               nmi=np.nan, ari=np.nan)
    if len(lp) == 0:
        return out

    # purity (weighted by cluster size)
    tot, hit = 0, 0
    for c in np.unique(lp):
        m = lp == c
        if m.sum() == 0:
            continue
        _, cnt = np.unique(lt[m], return_counts=True)
        tot += m.sum()
        hit += cnt.max()
    out['purity'] = hit / tot if tot else np.nan

    # accuracy with Hungarian matching
    tru = {v: i for i, v in enumerate(sorted(set(lt.tolist())))}
    clu = {v: i for i, v in enumerate(sorted(set(lp.tolist())))}
    cm = np.zeros((len(tru), len(clu)), dtype=np.int64)
    for a, b in zip(lt, lp):
        cm[tru[a], clu[b]] += 1
    r, c = linear_sum_assignment(-cm)
    out['acc'] = cm[r, c].sum() / cm.sum()

    out['nmi'] = normalized_mutual_info_score(lt, lp)
    out['ari'] = adjusted_rand_score(lt, lp)
    return out


# ------------------------------------------------------------------- per ckpt

def analyze_checkpoint(path, args, lookup_cache):
    split, seed = describe_ckpt(path)
    banner(f"CHECKPOINT  split={split}  seed={seed}\n{path}", "-")

    import torch
    ck = torch.load(path, map_location='cpu', weights_only=False)
    known = list(ck.get('known_classes', []))
    dtype = args.dataset_type or ck.get('dataset_type') or 'rml2016'
    usb_sig, usb_feat = ck.get('usb_signals'), ck.get('usb_features')

    log(f"  dataset_type   : {dtype}")
    log(f"  known_classes  : {known}")
    log(f"  dat_threshold  : {ck.get('dat_threshold')}")
    if not usb_sig:
        log("  [SKIP] no USB stored in this checkpoint.")
        return None
    F_all = np.asarray(usb_feat, dtype=np.float32)
    log(f"  USB size       : {human(len(F_all))}   (max_size cap is 40000)")
    log(f"  feature dim    : {F_all.shape[1]}   ||h|| mean={np.linalg.norm(F_all,axis=1).mean():.4f}")

    # ---- true labels: from the .npz cache if present, else from the dataset
    tag = re.sub(r'[^A-Za-z0-9._-]+', '_', f"{split}_seed{seed}")
    npz_path = os.path.join(args.save_npz, f"usb_{tag}.npz") if args.save_npz else None

    if args.reuse_npz and npz_path and os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        F = z['features'].astype(np.float32)
        true = z['true_labels'].astype(object)
        log(f"  loaded cached USB from {npz_path}  ({human(len(F))} points)")
    else:
        key = (dtype, args.dataset_path, args.min_snr)
        if key not in lookup_cache:
            if dtype == 'rml2018':
                lookup_cache[key] = build_lookup_rml2018(args.dataset_path, args.min_snr)
            else:
                lookup_cache[key] = build_lookup_rml2016(args.dataset_path, args.min_snr)
        true_any, found = recover_labels(usb_sig, lookup_cache[key], dtype)
        log(f"  label recovery : {human(int(found.sum()))}/{human(len(found))} "
            f"({found.mean()*100:.1f}%)")
        if found.sum() < 20:
            log("  [SKIP] too few USB signals matched back to the dataset.")
            return None
        # keep only matched points from here on
        F = F_all[found]
        true = np.asarray(true_any, dtype=object)[found]
        if npz_path:
            os.makedirs(args.save_npz, exist_ok=True)
            np.savez_compressed(npz_path, features=F, true_labels=true.astype(str))
            log(f"  cached USB features+labels -> {npz_path}")
            log("  (later passes: add --reuse_npz to skip the dataset lookup)")

    known_set = set(known)
    unknown_mask = np.array([t not in known_set for t in true])
    leak = 1.0 - unknown_mask.mean()

    uniq, cnt = np.unique(true, return_counts=True)
    order = np.argsort(-cnt)
    log("")
    log("  USB TRUE-CLASS COMPOSITION (matched points only):")
    for i in order:
        tag = 'KNOWN' if uniq[i] in known_set else 'unknown'
        log(f"     {uniq[i]:>12s}  {cnt[i]:>7,d}  {cnt[i]/len(true)*100:5.1f}%   [{tag}]")
    n_unk_present = int(sum(1 for u in uniq if u not in known_set))
    log(f"  -> distinct true classes in USB     : {len(uniq)}")
    log(f"  -> distinct true UNKNOWN classes    : {n_unk_present}  "
        f"(true d = {24 - len(known) if dtype == 'rml2018' else 11 - len(known)})")
    log(f"  -> dominant true class share        : {cnt.max()/len(true)*100:.1f}%")
    log(f"  -> known-class leakage in USB       : {leak*100:.2f}%")

    # optional subsample for the expensive fits
    if args.fit_sample and args.fit_sample < len(F):
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(F), args.fit_sample, replace=False)
    else:
        sub = np.arange(len(F))
    Fs, trues = F[sub], true[sub]
    unk_s = np.array([t not in known_set for t in trues])
    log(f"  fitting selectors on a subsample of {human(len(Fs))} points")

    # ---- sweep k -----------------------------------------------------------
    log("")
    log(f"  {'k':>3} {'sil_euc':>8} {'sil_cos':>8} {'sil_pca':>8} "
        f"{'GMM_BIC':>12} {'gap':>8} {'gap_s':>7} {'PS':>6} | "
        f"{'purity':>7} {'ACC':>6} {'NMI':>6} {'ARI':>6}")
    log("  " + "-" * 104)

    rows = []
    Fr_pca = None
    for k in range(args.k_min, args.k_max + 1):
        if k >= len(Fs):
            break
        t0 = time.time()
        row = {'k': k}
        row['sil_euc'] = sel_silhouette(Fs, k, 'euclidean', args.sil_sample, args.seed)
        row['sil_cos'] = sel_silhouette(Fs, k, 'cosine', args.sil_sample, args.seed)
        row['sil_pca'], Fr_pca = sel_silhouette_pca(Fs, k, args.pca_dim,
                                                    args.sil_sample, args.seed)
        row['bic'] = sel_gmm_bic(Fs, k, args.seed)
        if args.skip_gap:
            row['gap'], row['gap_s'] = np.nan, np.nan
        else:
            row['gap'], row['gap_s'] = sel_gap(F, k, args.gap_refs, args.seed,
                                               args.fit_sample)
        if args.skip_ps:
            row['ps'] = np.nan
        else:
            row['ps'] = sel_prediction_strength(Fs, k, args.ps_repeats,
                                                args.ps_max_pts, args.seed)

        g = grade(kmeans_labels(Fs, k, args.seed), trues, unk_s)
        row.update(purity=g['purity'], acc=g['acc'], nmi=g['nmi'], ari=g['ari'],
                   n_unk=g['n'])

        log(f"  {k:>3} {row['sil_euc']:>8.4f} {row['sil_cos']:>8.4f} "
            f"{row['sil_pca']:>8.4f} {row['bic']:>12.1f} {row['gap']:>8.4f} "
            f"{row['gap_s']:>7.4f} {row['ps']:>6.3f} | "
            f"{row['purity']:>7.4f} {row['acc']:>6.4f} {row['nmi']:>6.4f} "
            f"{row['ari']:>6.4f}   ({time.time()-t0:.0f}s)")
        rows.append(row)

    if not rows:
        return None

    # ---- which k does each selector pick, and how good is it? --------------
    def pick_max(col):
        vals = np.array([r[col] for r in rows], dtype=float)
        if np.all(np.isnan(vals)):
            return None
        return rows[int(np.nanargmax(vals))]['k']

    def pick_min(col):
        vals = np.array([r[col] for r in rows], dtype=float)
        if np.all(np.isnan(vals)):
            return None
        return rows[int(np.nanargmin(vals))]['k']

    def pick_gap():
        vals = [(r['k'], r['gap'], r['gap_s']) for r in rows
                if not np.isnan(r['gap'])]
        for i in range(len(vals) - 1):
            if vals[i][1] >= vals[i + 1][1] - vals[i + 1][2]:
                return vals[i][0]
        return vals[-1][0] if vals else None

    def pick_ps(threshold=0.8):
        ok = [r['k'] for r in rows if not np.isnan(r['ps']) and r['ps'] >= threshold]
        return max(ok) if ok else None

    by_k = {r['k']: r for r in rows}
    picks = {
        'sil_euc (CURRENT)': pick_max('sil_euc'),
        'sil_cos': pick_max('sil_cos'),
        'sil_pca': pick_max('sil_pca'),
        'GMM BIC': pick_min('bic'),
        'gap': pick_gap(),
        'pred strength': pick_ps(0.8),
    }
    oracle_k = max(rows, key=lambda r: r['acc'])['k']
    oracle_acc = by_k[oracle_k]['acc']
    oracle_pur = by_k[oracle_k]['purity']

    log("")
    log("  SELECTOR COMPARISON  (does the selector's pick land on the oracle k?)")
    log(f"  {'selector':<20} {'k picked':>9} {'purity@k':>10} {'ACC@k':>8} "
        f"{'NMI@k':>8}   {'ok?':>4}")
    log("  " + "-" * 66)
    for name, k in picks.items():
        if k is None:
            log(f"  {name:<20} {'n/a':>9}")
            continue
        r = by_k[k]
        ok = 'YES' if abs(r['acc'] - oracle_acc) < 1e-9 else (
            'near' if oracle_acc - r['acc'] < 0.05 else 'no')
        log(f"  {name:<20} {k:>9} {r['purity']:>10.4f} {r['acc']:>8.4f} "
            f"{r['nmi']:>8.4f}   {ok:>4}")
    log("  " + "-" * 66)
    log(f"  {'ORACLE (best k)':<20} {oracle_k:>9} {oracle_pur:>10.4f} "
        f"{oracle_acc:>8.4f} {by_k[oracle_k]['nmi']:>8.4f}")
    log("")
    log(f"  NOTE: utils/usb.py forces k=1 when silhouette < 0.15. "
        f"Best silhouette here = {max(r['sil_euc'] for r in rows):.4f} "
        f"-> {'TRIGGERS the k=1 fallback' if max(r['sil_euc'] for r in rows) < 0.15 else 'does not trigger'}")

    return dict(split=split, seed=seed, path=path, rows=rows, picks=picks,
                oracle_k=oracle_k, oracle_acc=oracle_acc,
                n_unk_present=n_unk_present, leak=leak, usb=len(F),
                max_sil=max(r['sil_euc'] for r in rows))


# ------------------------------------------------------------------ summary

def summarize(all_res):
    banner("AGGREGATE SUMMARY ACROSS ALL CHECKPOINTS")
    if not all_res:
        log("nothing to summarise.")
        return
    log(f"checkpoints analysed: {len(all_res)}")
    log("")
    log(f"  {'split':<16} {'seed':>5} {'USB':>7} {'#unk':>5} {'oracle k':>9} "
        f"{'oracleACC':>10} {'sil pick':>9} {'silACC':>8} {'max_sil':>8}")
    log("  " + "-" * 92)
    for r in all_res:
        sk = r['picks']['sil_euc (CURRENT)']
        sacc = [x['acc'] for x in r['rows'] if x['k'] == sk][0] if sk else float('nan')
        log(f"  {r['split']:<16} {r['seed']:>5} {human(r['usb']):>7} "
            f"{r['n_unk_present']:>5} {r['oracle_k']:>9} {r['oracle_acc']:>10.4f} "
            f"{str(sk):>9} {sacc:>8.4f} {r['max_sil']:>8.4f}")

    log("")
    log("  MEAN ground-truth ACC of each selector's pick "
        "(higher = the selector found a better k):")
    names = list(all_res[0]['picks'].keys())
    for name in names:
        accs, ks = [], []
        for r in all_res:
            k = r['picks'][name]
            if k is None:
                continue
            accs.append([x['acc'] for x in r['rows'] if x['k'] == k][0])
            ks.append(k)
        if accs:
            log(f"    {name:<20} mean k = {np.mean(ks):5.2f}   "
                f"mean ACC = {np.mean(accs):.4f}   (n={len(accs)})")
    log(f"    {'ORACLE':<20} mean k = "
        f"{np.mean([r['oracle_k'] for r in all_res]):5.2f}   "
        f"mean ACC = {np.mean([r['oracle_acc'] for r in all_res]):.4f}")
    log("")
    n_forced = sum(1 for r in all_res if r['max_sil'] < 0.15)
    log(f"  current silhouette<0.15 gate would force k=1 in "
        f"{n_forced}/{len(all_res)} checkpoints")
    log("")
    log("  HOW TO READ THIS")
    log("  ----------------")
    log("  * ACC (Hungarian) is dominated by the majority true class, so it usually")
    log("    favours SMALL k. NMI/ARI are much less majority-dominated. Compare both:")
    log("      - oracle ACC and oracle NMI at the SAME k  -> the embedding separates")
    log("        the classes, and only the k-selector needs fixing.")
    log("      - oracle NMI at large k but oracle ACC at small k -> structure exists")
    log("        but clusters are badly unbalanced; a selector swap will NOT be enough")
    log("        (you need class-balanced / density-based clustering or better features).")
    log("  * If no selector ever beats the ORACLE by much, the selector is not your")
    log("    bottleneck - the novelty gate / USB coverage is.")
    log("  * 'pred strength' uses the standard rule: largest k with PS >= 0.8. If PS")
    log("    stays ~1.0 for every k, the criterion is not discriminating here and it")
    log("    will return k_max; read the raw PS column instead.")
    log("")


def write_csv(all_res, path):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'seed', 'k', 'sil_euc', 'sil_cos', 'sil_pca', 'gmm_bic',
                    'gap', 'gap_s', 'pred_strength', 'purity', 'acc', 'nmi', 'ari'])
        for r in all_res:
            for row in r['rows']:
                w.writerow([r['split'], r['seed'], row['k'],
                            row['sil_euc'], row['sil_cos'], row['sil_pca'],
                            row['bic'], row['gap'], row['gap_s'], row['ps'],
                            row['purity'], row['acc'], row['nmi'], row['ari']])
    log(f"CSV written: {path}")


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(
        description="Grade k-selection criteria for novel-class discovery on the USB.")
    p.add_argument('--root', action='append', required=True,
                   help='dir to search recursively for phase1_model*.pth '
                        '(repeatable; files also accepted)')
    p.add_argument('--dataset_type', default='', choices=['', 'rml2016', 'rml2018'],
                   help='override; otherwise taken from the checkpoint')
    p.add_argument('--dataset_path', default='RML2016.10a_dict.pkl')
    p.add_argument('--min_snr', type=float, default=0)
    p.add_argument('--k_min', type=int, default=2)
    p.add_argument('--k_max', type=int, default=15)
    p.add_argument('--sil_sample', type=int, default=10000,
                   help='subsample used by silhouette (O(n^2)); 0 = all')
    p.add_argument('--fit_sample', type=int, default=20000,
                   help='subsample used for the repeated fits; 0 = all')
    p.add_argument('--pca_dim', type=int, default=20)
    p.add_argument('--gap_refs', type=int, default=5)
    p.add_argument('--skip_gap', action='store_true')
    p.add_argument('--ps_repeats', type=int, default=5)
    p.add_argument('--ps_max_pts', type=int, default=2000)
    p.add_argument('--skip_ps', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--csv', default='')
    p.add_argument('--save_npz', default='',
                   help='optional dir: cache matched USB features + true labels '
                        'per checkpoint, so you never rebuild the dataset lookup')
    p.add_argument('--reuse_npz', action='store_true',
                   help='with --save_npz: load the cached USB instead of the dataset '
                        '(skips the lookup build entirely)')
    p.add_argument('--list_only', action='store_true',
                   help='just list the checkpoints found, then exit')
    args = p.parse_args()

    ckpts = find_checkpoints(args.root)
    banner(f"FOUND {len(ckpts)} CHECKPOINT(S)")
    for c in ckpts:
        s, sd = describe_ckpt(c)
        log(f"  split={s:<16} seed={sd:<4} {c}")
    if not ckpts:
        log("")
        log("No phase1_model*.pth found. Check --root. Expected layout:")
        log("    checkpoints/set1_2018/seed_42/phase1_model.pth")
        sys.exit(1)
    if args.list_only:
        return

    lookup_cache = {}
    all_res = []
    for c in ckpts:
        try:
            r = analyze_checkpoint(c, args, lookup_cache)
            if r:
                all_res.append(r)
        except Exception as e:
            log(f"  [ERROR] {c}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    summarize(all_res)
    if args.csv and all_res:
        write_csv(all_res, args.csv)


if __name__ == '__main__':
    main()
