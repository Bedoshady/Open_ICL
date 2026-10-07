#!/usr/bin/env python
"""
gate_ablation.py
================
Answers one question: does the CLP novel-class gate (Gate B) actually add
open-set F1 / OA on top of the DM novelty gate (Gate A), and is the current
operating point already at the ceiling?

This is READ-ONLY: it loads checkpoints and evaluates. No training.

The decision rule in evaluate.py is an OR of two gates:

    Gate A :  min_prob > tau                  (DM novelty score)
    Gate B :  argmax(logits) >= K             (classifier picks a novel class)

    pred_unknown = GateA(tau) OR GateB

This script separates them and sweeps tau, reporting F1 / OA / precision /
recall / AUROC for:

    A_only      : Gate A alone, best tau
    B_only      : Gate B alone (no threshold)
    OR_current  : the OR at the tau saved in the checkpoint  (what you report now)
    OR_best     : the OR at its best tau                    (the ceiling of this rule)

plus several candidate novelty SCORES, each swept on its own threshold:

    dm_minprob      : min over known SFCs of the DM output      (Gate A's score)
    euclid_min      : min Euclidean distance to a known SFC
    novel_maxprob   : max softmax over the discovered novel classes   <- continuous Gate B
    novel_margin    : best novel logit - best known logit             <- continuous Gate B
    fused_rank      : rank-average of dm_minprob and novel_maxprob    <- fusion test
    novel_centroid  : min distance to a discovered novel centroid     (optional, --novel_k)

HOW TO READ THE VERDICT
-----------------------
  OR_best  ~=  A_only        -> Gate B adds nothing. The rule is at its ceiling.
                                Fix = add a new signal / fuse the gates (Step 3).
  OR_best  >   A_only        -> headroom already exists in the combination and the
                                fixed tau is not capturing it. You can take it for
                                free by fixing how tau is chosen (Step 2).
  OR_best  >   A_only  AND
  OR_best  >   B_only        -> the combination genuinely beats both singles once the
                                operating point is right.

Usage
-----
    python gate_ablation.py --root checkpoints --dataset_type rml2018 \
        --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5

    # quick look on a subset of the validation set
    python gate_ablation.py --root checkpoints/set1_2018 \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5 \
        --max_batches 200

    # include the novel-centroid distance signal (clusters the stored USB)
    python gate_ablation.py --root checkpoints/set1_2018 --novel_k 2 \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')


def log(msg=""):
    print(msg, flush=True)


def banner(msg, ch="="):
    log("")
    log(ch * 78)
    log(msg)
    log(ch * 78)


# ------------------------------------------------------------ ckpt discovery

def find_checkpoints(roots):
    """Prefer phase2 (post-incremental) checkpoints; fall back to phase1."""
    out = []
    for root in roots:
        if os.path.isfile(root):
            out.append(root)
            continue
        for name in ("phase2_incremental_model.pth", "phase1_model.pth"):
            out.extend(glob.glob(os.path.join(root, "**", name), recursive=True))
    # keep phase2 when both exist for the same dir
    seen, keep = set(), []
    for p in sorted(set(os.path.normpath(x) for x in out), reverse=True):
        d = os.path.dirname(p)
        if d in seen:
            continue
        seen.add(d)
        keep.append(p)
    return sorted(keep)


def describe_ckpt(path):
    parts = os.path.normpath(path).split(os.sep)
    seed = "?"
    for p in reversed(parts):
        if p.startswith("seed_"):
            seed = p.replace("seed_", "")
            break
    if seed != "?":
        i = parts.index(f"seed_{seed}")
        split = parts[i - 1] if i >= 1 else "?"
    else:
        split = parts[-2] if len(parts) >= 2 else "?"
    return split, seed


# --------------------------------------------------------------- score sweep

def _best_over_threshold(score, y_true, y_class, pred_class, grid):
    """Sweep a threshold on `score`; pred_unknown = score > thr. Return best F1."""
    from sklearn.metrics import f1_score
    best = dict(f1=-1, thr=np.nan, p=np.nan, r=np.nan, oa=np.nan)
    for thr in grid:
        pred = (score > thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best['f1']:
            tp = ((y_true == 1) & (pred == 1)).sum()
            fp = ((y_true == 0) & (pred == 1)).sum()
            fn = ((y_true == 1) & (pred == 0)).sum()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            oa = _oa(pred, y_true, y_class, pred_class)
            best = dict(f1=f1, thr=thr, p=p, r=r, oa=oa)
    return best


def _oa(pred_binary, y_true_binary, y_class, pred_class):
    """Exactly evaluate.py's OA: TP(unknown) + knowns unflagged AND correctly classified."""
    tp = ((y_true_binary == 1) & (pred_binary == 1)).sum()
    tn = ((y_true_binary == 0) & (pred_binary == 0) & (y_class == pred_class)).sum()
    n = len(y_true_binary)
    return (tp + tn) / n if n else 0.0


def _f1_pr(pred_binary, y_true_binary):
    from sklearn.metrics import f1_score
    tp = ((y_true_binary == 1) & (pred_binary == 1)).sum()
    fp = ((y_true_binary == 0) & (pred_binary == 1)).sum()
    fn = ((y_true_binary == 1) & (pred_binary == 0)).sum()
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return f1_score(y_true_binary, pred_binary, zero_division=0), p, r


def rank_norm(x):
    """Rank-normalise to [0,1] - scale free, robust for fusing scores."""
    order = np.argsort(np.argsort(x))
    return order / max(1, len(x) - 1)


def flags(r, floor):
    """'*' = best F1 is essentially the all-unknown floor, '!' = recall ~ 1 corner."""
    s = '*' if r['f1'] <= floor + 0.02 else ' '
    if r.get('r', 0.0) > 0.95:
        s += '!'
    else:
        s += ' '
    return s


# --------------------------------------------------------------- collect pass

def collect(model, loader, K, device, max_batches, novel_centroids=None):
    import torch
    import torch.nn.functional as F

    acc = {k: [] for k in ('minprob', 'eumindist', 'maxnovel', 'margin',
                           'true_bin', 'true_cls', 'pred_cls', 'ncent')}
    model.eval()
    with torch.no_grad():
        for bi, (x, y, _) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x = x.to(device)
            logits, feats, probs, _, dists = model(x)

            acc['minprob'].append(probs.min(dim=1).values.cpu().numpy())
            acc['eumindist'].append(dists.min(dim=1).values.cpu().numpy())
            acc['true_bin'].append((y == -1).numpy().astype(int))
            acc['true_cls'].append(y.numpy())
            acc['pred_cls'].append(logits.argmax(dim=1).cpu().numpy())

            sm = F.softmax(logits, dim=1)
            if K < sm.shape[1]:
                acc['maxnovel'].append(sm[:, K:].max(dim=1).values.cpu().numpy())
                best_novel = logits[:, K:].max(dim=1).values
                best_known = logits[:, :K].max(dim=1).values
                acc['margin'].append((best_novel - best_known).cpu().numpy())
            else:
                acc['maxnovel'].append(np.zeros(len(y)))
                acc['margin'].append(np.full(len(y), -1e9))

            if novel_centroids is not None:
                c = torch.as_tensor(novel_centroids, dtype=feats.dtype, device=feats.device)
                d = torch.cdist(feats, c).min(dim=1).values
                acc['ncent'].append(d.cpu().numpy())

            if (bi + 1) % 200 == 0:
                log(f"      ... {bi+1} batches")

    return {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}


# ------------------------------------------------------------------ per ckpt

def analyze(path, args):
    import torch
    from models.donet import DONet
    from data.dataset import get_dataloaders

    split, seed = describe_ckpt(path)
    banner(f"CHECKPOINT  split={split}  seed={seed}\n{path}", "-")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  device         : {device}")
    ck = torch.load(path, map_location=device, weights_only=False)
    known = list(ck['known_classes'])
    K = len(known)
    discovered = ck.get('discovered_classes', [])
    n_total = K + len(discovered)
    dtype = args.dataset_type or ck.get('dataset_type') or 'rml2016'
    tau_saved = ck.get('dat_threshold', 0.0)

    log(f"  known classes  : {K}   discovered novel classes: {len(discovered)}")
    log(f"  dat_threshold  : {tau_saved}   (this is OR_current's tau)")

    model = DONet(num_known_classes=n_total, feature_dim=128).to(device)
    try:
        model.load_state_dict(ck['model_state_dict'])
    except RuntimeError as e:
        first = str(e).splitlines()[0]
        log("  [SKIP] checkpoint architecture does not match the current DONet.")
        log(f"         {first}")
        log("         This checkpoint came from an older/other model definition")
        log("         (e.g. the legacy simple-projection DONet). Re-run train.py with")
        log("         the current code, or evaluate it with the code that made it.")
        return None
    model.eval()

    # ---- novel centroids (optional signal) --------------------------------
    if str(args.novel_k).lower() == 'auto':
        novel_k = len(discovered)
    else:
        try:
            novel_k = int(args.novel_k)
        except ValueError:
            novel_k = 0
    novel_centroids = None
    if novel_k >= 2 and ck.get('usb_features'):
        from sklearn.cluster import KMeans
        U = np.asarray(ck['usb_features'], dtype=np.float32)
        if len(U) >= novel_k:
            km = KMeans(n_clusters=novel_k, random_state=42, n_init=10).fit(U)
            novel_centroids = km.cluster_centers_
            log(f"  novel centroids: {novel_k} (from {len(U)} USB embeddings)")
        else:
            log(f"  [note] USB has only {len(U)} points, skipping novel-centroid test")

    # ---- data -------------------------------------------------------------
    if dtype == 'rml2018':
        all_classes = ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
                       'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64',
                       'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC',
                       'AM_DSB_SC', 'FM', 'GMSK', 'OQPS']
    else:
        all_classes = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4',
                       'QAM16', 'QAM64', 'QPSK', 'WBFM']
    unknown = [c for c in all_classes if c not in known]
    _, val_loader = get_dataloaders(args.dataset_path, known_classes=known,
                                   unknown_classes=unknown, batch_size=128,
                                   dataset_type=dtype, seed=args.seed)
    log(f"  validating on the standard val split (seed={args.seed})"
        f"{f', first {args.max_batches} batches' if args.max_batches else ''}")

    S = collect(model, val_loader, K, device, args.max_batches, novel_centroids)
    y_true = S['true_bin']
    y_cls = S['true_cls']
    pred_cls = S['pred_cls']
    gateB = (pred_cls >= K).astype(int)
    u = y_true.mean()
    log(f"  val samples    : {len(y_true):,}   unknown fraction u = {u:.4f}"
        f"   trivial all-unknown F1 floor = {2*u/(1+u):.4f}")

    grid = np.linspace(0.0, 1.0, 201)
    res = {}

    # ---- Gate A only -------------------------------------------------------
    a = _best_over_threshold(S['minprob'], y_true, y_cls, pred_cls, grid)
    res['A_only'] = a

    # ---- Gate B only -------------------------------------------------------
    f1b, pb, rb = _f1_pr(gateB, y_true)
    res['B_only'] = dict(f1=f1b, p=pb, r=rb,
                         oa=_oa(gateB, y_true, y_cls, pred_cls), thr=np.nan)

    # ---- OR at saved tau and at its best tau -------------------------------
    orc = (S['minprob'] > tau_saved).astype(int) | gateB
    f1c, pc, rc = _f1_pr(orc, y_true)
    res['OR_current'] = dict(f1=f1c, p=pc, r=rc,
                             oa=_oa(orc, y_true, y_cls, pred_cls), thr=tau_saved)

    best_or = dict(f1=-1, thr=np.nan, p=np.nan, r=np.nan, oa=np.nan)
    for thr in grid:
        pred = (S['minprob'] > thr).astype(int) | gateB
        f1, p, r = _f1_pr(pred, y_true)
        if f1 > best_or['f1']:
            best_or = dict(f1=f1, thr=thr, p=p, r=r,
                           oa=_oa(pred, y_true, y_cls, pred_cls))
    res['OR_best'] = best_or

    # ---- single-score candidates ------------------------------------------
    from sklearn.metrics import roc_auc_score
    cands = {
        'dm_minprob': S['minprob'],
        'euclid_min': S['eumindist'],
        'novel_maxprob': S['maxnovel'],
        'novel_margin': S['margin'],
        'fused_rank': 0.5 * rank_norm(S['minprob']) + 0.5 * rank_norm(S['maxnovel']),
    }
    if novel_centroids is not None and len(S['ncent']):
        cands['novel_centroid'] = S['ncent']

    scores = {}
    for name, sc in cands.items():
        sc = np.asarray(sc, dtype=float)
        if np.all(np.isnan(sc)) or np.nanmax(sc) == np.nanmin(sc):
            continue
        try:
            auc = roc_auc_score(y_true, sc)
        except Exception:
            auc = np.nan
        # threshold grid scaled to this score's range
        g = np.linspace(np.nanmin(sc), np.nanmax(sc), 201)
        b = _best_over_threshold(sc, y_true, y_cls, pred_cls, g)
        scores[name] = dict(auc=auc, **b)

    # ---- report -----------------------------------------------------------
    floor = 2 * u / (1 + u)
    log("")
    log("  GATE ABLATION   (F1 / OA on the standard val split)")
    log(f"  trivial all-unknown F1 floor = {floor:.4f}")
    log("  flags: * = best F1 is essentially the all-unknown floor (degenerate),")
    log("         ! = operating point has recall > 0.95 (flags nearly everything unknown)")
    log(f"  {'variant':<14} {'tau':>7} {'F1':>7}{'':2} {'OA':>7} {'P':>7} {'R':>7}")
    log("  " + "-" * 56)
    for name in ('A_only', 'B_only', 'OR_current', 'OR_best'):
        r = res[name]
        thr = 'n/a' if np.isnan(r['thr']) else f"{r['thr']:.3f}"
        log(f"  {name:<14} {thr:>7} {r['f1']:>7.4f}{flags(r, floor)} {r['oa']:>7.4f} "
            f"{r['p']:>7.4f} {r['r']:>7.4f}")
    log("")
    log("  CANDIDATE NOVELTY SCORES   (each swept on its own threshold)")
    log(f"  {'score':<16} {'AUROC':>7} {'thr@bF1':>8} {'bestF1':>8}{'':2} {'bestOA':>8} "
        f"{'P@bF1':>7} {'R@bF1':>7}")
    log("  " + "-" * 70)
    for name, r in scores.items():
        log(f"  {name:<16} {r['auc']:>7.4f} {r['thr']:>8.4f} {r['f1']:>8.4f}"
            f"{flags(r, floor)} {r['oa']:>8.4f} {r['p']:>7.4f} {r['r']:>7.4f}")

    # ---- verdict ----------------------------------------------------------
    d_f1 = best_or['f1'] - a['f1']
    d_oa = best_or['oa'] - a['oa']
    log("")
    log("  VERDICT")
    log("  " + "-" * 60)
    log(f"  OR_best - A_only  :  dF1 = {d_f1:+.4f}   dOA = {d_oa:+.4f}")
    log(f"  OR_best - B_only  :  dF1 = {best_or['f1']-f1b:+.4f}")
    log(f"  OR_best - OR_current (free win from fixing tau): "
        f"dF1 = {best_or['f1']-f1c:+.4f}   dOA = {best_or['oa']-res['OR_current']['oa']:+.4f}")
    if d_f1 <= 0.005:
        log("  -> Gate B adds nothing. The OR rule is at its ceiling.")
        log("     Next: fuse Gate B as a continuous score, or add a new signal.")
    elif best_or['f1'] - f1b > 0.005:
        log("  -> The combination BEATS both singles once tau is right.")
        log("     Next: fix how tau is chosen, then re-report.")
    else:
        log("  -> Gate B helps, but not beyond Gate A alone at its best tau.")
        log("     Next: strengthen Gate B (coverage/purity) or fuse the scores.")
    dm = scores.get('dm_minprob')
    if dm and dm['auc'] < 0.60:
        log(f"  !! Gate A's DM score has AUROC {dm['auc']:.4f} (near chance) on this")
        log("     checkpoint: it is not detecting novelty at all, and its 'best F1' is")
        log("     just the all-unknown corner. Anything the combination achieves here is")
        log("     coming from the classifier, not from the DM novelty score.")
    best_single = max(scores.items(), key=lambda kv: kv[1]['f1']) if scores else None
    if best_single:
        log(f"  -> best single candidate score: {best_single[0]} "
            f"(AUROC {best_single[1]['auc']:.4f}, bestF1 {best_single[1]['f1']:.4f})")
    ncent = scores.get('novel_centroid')
    if ncent and ncent['auc'] < 0.45:
        log("  !! novel_centroid AUROC < 0.5: test-time unknowns are NOT closer to the")
        log("     discovered novel centroids than knowns are -> the USB clusters do not")
        log("     represent the unknowns. This is the COVERAGE problem, not a selector bug.")

    return dict(split=split, seed=seed, u=u, res=res, scores=scores,
                n=n_total, tau_saved=tau_saved)


# ------------------------------------------------------------------ summary

def summarize(all_res):
    banner("AGGREGATE ACROSS CHECKPOINTS")
    if not all_res:
        log("nothing to summarise.")
        return
    keys = ('A_only', 'B_only', 'OR_current', 'OR_best')
    log(f"  {'split':<14} {'seed':>5} | " + " ".join(f"{k:>11}" for k in keys))
    log("  " + "-" * 74)
    for r in all_res:
        vals = " ".join(f"{r['res'][k]['f1']:>11.4f}" for k in keys)
        log(f"  {r['split']:<14} {r['seed']:>5} | {vals}")
    log("")
    log("  mean F1:")
    for k in keys:
        log(f"    {k:<12} {np.mean([r['res'][k]['f1'] for r in all_res]):.4f}")
    log("  mean OA:")
    for k in keys:
        log(f"    {k:<12} {np.mean([r['res'][k]['oa'] for r in all_res]):.4f}")
    log("")
    log("  mean AUROC / bestF1 per candidate score:")
    names = set()
    for r in all_res:
        names |= set(r['scores'].keys())
    for n in sorted(names):
        aucs = [r['scores'][n]['auc'] for r in all_res if n in r['scores']]
        f1s = [r['scores'][n]['f1'] for r in all_res if n in r['scores']]
        if aucs:
            log(f"    {n:<16} AUROC {np.mean(aucs):.4f}   bestF1 {np.mean(f1s):.4f}")
    log("")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', action='append', required=True,
                   help='dir with checkpoints (repeatable); prefers phase2 over phase1')
    p.add_argument('--dataset_type', default='', choices=['', 'rml2016', 'rml2018'])
    p.add_argument('--dataset_path', default='RML2016.10a_dict.pkl')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_batches', type=int, default=0,
                   help='0 = full validation set (matches evaluate.py exactly)')
    p.add_argument('--novel_k', default='auto',
                   help="novel centroids to cluster the USB into: 'auto' (use the "
                        "checkpoint's own discovered count), an int, or 0 to skip")
    p.add_argument('--list_only', action='store_true')
    args = p.parse_args()

    ck = find_checkpoints(args.root)
    banner(f"FOUND {len(ck)} CHECKPOINT(S)")
    for c in ck:
        s, sd = describe_ckpt(c)
        log(f"  split={s:<16} seed={sd:<4} {os.path.basename(c)}")
    if not ck:
        log("none found - check --root")
        sys.exit(1)
    if args.list_only:
        return

    all_res = []
    for c in ck:
        try:
            r = analyze(c, args)
            if r:
                all_res.append(r)
        except Exception as e:
            log(f"  [ERROR] {c}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    summarize(all_res)


if __name__ == '__main__':
    main()
