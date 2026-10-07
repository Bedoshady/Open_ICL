#!/usr/bin/env python
"""
evaluate_calibrated.py
======================
Evaluation-only variant of evaluate.py.

The ONLY change is how the decision threshold (tau) is chosen. The network, the
weights, the USB and the trained model are all untouched -- nothing needs
retraining.

    evaluate.py            : tau = mu + 0.5*sigma   (computed from training data,
                                                      unsupervised heuristic)
    this script            : tau = the (1-alpha) percentile of the novelty scores
                                 of KNOWN TRAINING samples, i.e. tau is set so
                                 that alpha of the known signals the model was
                                 trained on would be flagged as unknown.

Because the calibration signals come from the TRAINING split, there is no overlap
whatsoever with the evaluation set. The realised false-alarm rate on validation
knowns will be somewhat ABOVE alpha (the model has seen the calibration signals,
so their scores are optimistically low) -- that is conservative and is reported
explicitly so it is transparent.

The decision rule itself is unchanged:

    pred_unknown = (min_contrast_prob > tau)  OR  (argmax(logits) >= K)

and for every checkpoint the script prints the OLD rule (using the tau saved in
the checkpoint) next to the NEW calibrated rule, so the two can be compared
directly from the same forward pass.

Usage
-----
    python evaluate_calibrated.py --checkpoint_dir checkpoints/set1_2018/seed_42 \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5

    # sweep a few false-alarm rates
    python evaluate_calibrated.py --checkpoint_dir ... --alphas 0.01,0.05,0.10

    # walk a whole tree (all splits/seeds at once)
    python evaluate_calibrated.py --root checkpoints \
        --dataset_type rml2018 --dataset_path GOLD_XYZ_OSC.0001_1024.hdf5

    # cap the calibration pass for speed (200k is ample for a 1% quantile)
    python evaluate_calibrated.py --root checkpoints --calib_samples 200000 ...

Reporting
---------
For each checkpoint: F1, OA, precision, recall, plus the realised false-alarm
rate on validation knowns and the realised detection rate on validation unknowns.
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

ALL_CLASSES = {
    'rml2018': ['OOK', 'ASK4', 'ASK8', 'BPSK', 'QPSK', 'PSK8', 'PSK16', 'PSK32',
                'APSK16', 'APSK32', 'APSK64', 'APSK128', 'QAM16', 'QAM32', 'QAM64',
                'QAM128', 'QAM256', 'AM_SSB_WC', 'AM_SSB_SC', 'AM_DSB_WC',
                'AM_DSB_SC', 'FM', 'GMSK', 'OQPS'],
    'rml2016': ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4',
                'QAM16', 'QAM64', 'QPSK', 'WBFM'],
}


def log(m=""):
    print(m, flush=True)


def banner(m, ch="="):
    log(""); log(ch * 78); log(m); log(ch * 78)


# ---------------------------------------------------------------- metrics

def metrics(pred_binary, y_true_binary, y_class, pred_class):
    """Exactly evaluate.py's F1 / OA / P / R definitions."""
    from sklearn.metrics import f1_score
    tp = int(((y_true_binary == 1) & (pred_binary == 1)).sum())
    fp = int(((y_true_binary == 0) & (pred_binary == 1)).sum())
    fn = int(((y_true_binary == 1) & (pred_binary == 0)).sum())
    tn = int(((y_true_binary == 0) & (pred_binary == 0) & (y_class == pred_class)).sum())
    n = len(y_true_binary)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    oa = (tp + tn) / n if n else 0.0
    return dict(f1=f1_score(y_true_binary, pred_binary, zero_division=0),
                oa=oa, p=p, r=r,
                fpr=fp / max(1, int((y_true_binary == 0).sum())),
                tpr=r)


# --------------------------------------------------------------- forward pass

def collect(model, loader, K, device, max_batches=0, want_all=True, tag=""):
    """
    Returns dict of arrays. Always collects the DM novelty score and the
    classifier prediction; labels when want_all (evaluation pass).
    """
    import torch

    out = {k: [] for k in ('minprob', 'pred_class', 'true_bin', 'true_class',
                           'maxnovel', 'margin')}
    model.eval()
    with torch.no_grad():
        for bi, (x, y, _) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x = x.to(device)
            logits, _, probs, _, _ = model(x)

            out['minprob'].append(probs.min(dim=1).values.cpu().numpy())
            pred = logits.argmax(dim=1)
            out['pred_class'].append(pred.cpu().numpy())
            if want_all:
                out['true_bin'].append((y == -1).numpy().astype(int))
                out['true_class'].append(y.numpy())
                import torch.nn.functional as F
                sm = F.softmax(logits, dim=1)
                out['maxnovel'].append(sm[:, K:].max(dim=1).values.cpu().numpy())
                bn = logits[:, K:].max(dim=1).values if logits.shape[1] > K else None
                bk = logits[:, :K].max(dim=1).values
                out['margin'].append((bn - bk).cpu().numpy() if bn is not None
                                     else np.full(len(y), -1e9))

            if tag and (bi + 1) % 1000 == 0:
                log(f"      {tag}: {bi+1} batches")

    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


# ------------------------------------------------------------- per checkpoint

def load_model(path, device):
    import torch
    from models.donet import DONet
    ck = torch.load(path, map_location=device, weights_only=False)
    known = list(ck['known_classes'])
    K = len(known)
    discovered = ck.get('discovered_classes', [])
    n_total = K + len(discovered)
    model = DONet(num_known_classes=n_total, feature_dim=128).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    return model, ck, known, K, discovered, n_total


def describe(path):
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


def analyze(ckpt_path, args):
    import torch
    from data.dataset import get_dataloaders

    split, seed = describe(ckpt_path)
    banner(f"CHECKPOINT  split={split}  seed={seed}\n{ckpt_path}", "-")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck, known, K, discovered, n_total = load_model(ckpt_path, device)
    dtype = args.dataset_type or ck.get('dataset_type') or 'rml2016'
    tau_saved = ck.get('dat_threshold', 0.0)

    log(f"  device            : {device}")
    log(f"  known classes     : {K}    discovered novel classes: {len(discovered)}")
    log(f"  saved tau (old)   : {tau_saved}")

    unknown = [c for c in ALL_CLASSES[dtype] if c not in known]
    train_loader, val_loader = get_dataloaders(
        args.dataset_path, known_classes=known, unknown_classes=unknown,
        batch_size=128, dataset_type=dtype, seed=args.seed)

    # ---- calibration: novelty scores of KNOWN TRAINING samples -------------
    log("")
    log("  --- calibration pass over KNOWN TRAINING samples ---")
    cap = 0
    if args.calib_samples:
        # train set is 80% of the data; cap by batches so the pass stays bounded
        cap = args.calib_samples // 128
        log(f"  capping at ~{args.calib_samples:,} samples ({cap} batches)")
    cal = collect(model, train_loader, K, device, max_batches=cap,
                  want_all=True, tag="calib")
    # the training loader ALSO carries -1 for the unknown classes, so we must
    # filter on labels to keep only KNOWN training signals
    keep = cal['true_bin'] == 0
    cal_scores = cal['minprob'][keep]
    log(f"  calibration samples used : {len(cal_scores):,} known training signals "
        f"(of {len(cal['minprob']):,} training samples seen)")
    if len(cal_scores) < 1000:
        log("  [SKIP] too few calibration samples.")
        return None

    # ---- evaluation pass ---------------------------------------------------
    log("")
    log("  --- evaluation pass over the validation set ---")
    ev = collect(model, val_loader, K, device, want_all=True, tag="eval")
    y = ev['true_bin']
    ycls = ev['true_class']
    pred_cls = ev['pred_class']
    gateB = (pred_cls >= K).astype(int)
    n_val = len(y)
    u = y.mean()
    log(f"  val samples : {n_val:,}   unknown fraction u = {u:.4f}   "
        f"trivial all-unknown F1 floor = {2*u/(1+u):.4f}")

    # optional calibration on a held-out half of the val set instead
    if args.calib_from == 'val_half':
        half = n_val // 2
        keep_val = (y[:half] == 0)
        cal_scores = ev['minprob'][:half][keep_val]
        sl = slice(half, n_val)
        y, ycls, pred_cls, gateB = y[sl], ycls[sl], pred_cls[sl], gateB[sl]
        ev = {k: (v[sl] if len(v) == n_val else v) for k, v in ev.items()}
        log(f"  [calib_from=val_half] calibration on {len(cal_scores):,} known "
            f"signals from the first half; reporting on the second half "
            f"({len(y):,} samples)")

    # ---- table -------------------------------------------------------------
    alphas = [float(a) for a in args.alphas.split(',')]
    floor = 2 * u / (1 + u)
    log("")
    log("  RESULTS   (* = F1 is at/below the trivial all-unknown floor)")
    log(f"  {'rule':<26} {'tau':>8} {'F1':>7}{'':2} {'OA':>7} {'P':>7} {'R':>7} "
        f"{'FPR':>7} {'TPR':>7}")
    log("  " + "-" * 84)

    # old rule (what evaluate.py currently reports)
    pred_old = (ev['minprob'] > tau_saved).astype(int) | gateB
    m = metrics(pred_old, y, ycls, pred_cls)
    log(f"  {'OLD: mu+0.5sigma':<26} {tau_saved:>8.4f} {m['f1']:>7.4f}"
        f"{'*' if m['f1'] <= floor + 0.01 else ' ':>2} {m['oa']:>7.4f} "
        f"{m['p']:>7.4f} {m['r']:>7.4f} {m['fpr']:>7.4f} {m['tpr']:>7.4f}")

    rows = {'old': m}
    for a in alphas:
        tau = float(np.quantile(cal_scores, 1.0 - a))
        pred = (ev['minprob'] > tau).astype(int) | gateB
        m = metrics(pred, y, ycls, pred_cls)
        rows[f'a={a}'] = m
        log(f"  {f'NEW: pct alpha={a:<.2f}':<26} {tau:>8.4f} {m['f1']:>7.4f}"
            f"{'*' if m['f1'] <= floor + 0.01 else ' ':>2} {m['oa']:>7.4f} "
            f"{m['p']:>7.4f} {m['r']:>7.4f} {m['fpr']:>7.4f} {m['tpr']:>7.4f}")

    # ---- secondary (OFF by default): alternative scores --------------------
    if args.include_alt_scores and args.calib_from == 'val_half':
        log("")
        log("  [note] --include_alt_scores is skipped in calib_from=val_half mode "
            "(the two calibration sources would be mixed).")
    if args.include_alt_scores and args.calib_from == 'train':
        log("")
        log("  SECONDARY - alternative novelty SCORES (this is a METHOD change, "
            "not just evaluation).")
        log("  Calibrated with the identical percentile rule on known training data.")
        log(f"  {'score / rule':<26} {'tau':>8} {'F1':>7} {'OA':>7} {'P':>7} "
            f"{'R':>7} {'FPR':>7} {'TPR':>7}")
        log("  " + "-" * 82)
        for name, key in (('classifier max-novel prob', 'maxnovel'),
                          ('classifier margin', 'margin')):
            cal_sc = np.asarray(cal[key])[keep]
            if len(cal_sc) < 1000:
                continue
            for a in alphas:
                tau = float(np.quantile(cal_sc, 1.0 - a))
                pred = (np.asarray(ev[key]) > tau).astype(int)
                m = metrics(pred, y, ycls, pred_cls)
                log(f"  {f'{name} a={a:<.2f}':<26} {tau:>8.4f} {m['f1']:>7.4f} "
                    f"{m['oa']:>7.4f} {m['p']:>7.4f} {m['r']:>7.4f} {m['fpr']:>7.4f} "
                    f"{m['tpr']:>7.4f}")
        # simple fixed-weight rank fusion, identical percentile rule
        from scipy.stats import rankdata
        cm = np.asarray(cal['minprob'])[keep]
        cn = np.asarray(cal['maxnovel'])[keep]
        for w in (0.5, 0.7, 0.8):
            cal_f = ((1 - w) * rankdata(cm) / len(cm)
                     + w * rankdata(cn) / len(cn))
            f = ((1 - w) * rankdata(ev['minprob']) / len(ev['minprob'])
                 + w * rankdata(ev['maxnovel']) / len(ev['maxnovel']))
            for a in alphas:
                tau = float(np.quantile(cal_f, 1.0 - a))
                pred = (f > tau).astype(int)
                m = metrics(pred, y, ycls, pred_cls)
                log(f"  {f'fused w_clf={w} a={a:<.2f}':<26} {tau:>8.4f} "
                    f"{m['f1']:>7.4f} {m['oa']:>7.4f} {m['p']:>7.4f} {m['r']:>7.4f} "
                    f"{m['fpr']:>7.4f} {m['tpr']:>7.4f}")

    return dict(split=split, seed=seed, tau_saved=tau_saved, rows=rows, u=u,
                alphas=alphas)


def summarize(all_res):
    banner("AGGREGATE (mean over checkpoints)")
    if not all_res:
        log("nothing to summarise."); return
    keys = ['old'] + [f'a={a}' for a in all_res[0]['alphas']]
    log(f"  mean F1 over {len(all_res)} checkpoint(s):")
    for k in keys:
        vals = [r['rows'][k]['f1'] for r in all_res if k in r['rows']]
        if vals:
            log(f"    {k:<12} {np.mean(vals):.4f}")
    log(f"  mean OA:")
    for k in keys:
        vals = [r['rows'][k]['oa'] for r in all_res if k in r['rows']]
        if vals:
            log(f"    {k:<12} {np.mean(vals):.4f}")
    log("")
    log(f"  mean FPR (realised false-alarm rate on val knowns):")
    for k in keys:
        vals = [r['rows'][k]['fpr'] for r in all_res if k in r['rows']]
        if vals:
            log(f"    {k:<12} {np.mean(vals):.4f}")
    log("")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint_dir', default='',
                   help="dir containing phase2_incremental_model.pth / phase1_model.pth")
    p.add_argument('--root', action='append', default=None,
                   help='walk a tree for checkpoints (repeatable)')
    p.add_argument('--ckpt', default='', help='explicit checkpoint file')
    p.add_argument('--dataset_type', default='', choices=['', 'rml2016', 'rml2018'])
    p.add_argument('--dataset_path', default='RML2016.10a_dict.pkl')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alphas', default='0.01,0.05,0.10',
                   help='false-alarm rates for the percentile threshold')
    p.add_argument('--calib_from', default='train', choices=['train', 'val_half'],
                   help="'train' = known training samples (default, no test overlap); "
                        "'val_half' = known samples from the first half of val, "
                        "report on the second half")
    p.add_argument('--calib_samples', type=int, default=200000,
                   help='cap on calibration samples (0 = use all of the train pass)')
    p.add_argument('--include_alt_scores', action='store_true',
                   help='ALSO test classifier-score variants. This changes the '
                        'METHOD, not just the threshold - only enable if you will '
                        'describe it that way.')
    p.add_argument('--list_only', action='store_true')
    p.add_argument('--seeds', default='',
                   help='only use checkpoints under seed_<s> for these seeds, '
                        'e.g. 42,43,44 (empty = all, including unseeded dirs)')
    args = p.parse_args()

    ckpts = []
    if args.ckpt:
        ckpts.append(args.ckpt)
    if args.checkpoint_dir:
        for name in ('phase2_incremental_model.pth', 'phase1_model.pth'):
            ckpts.extend(glob.glob(os.path.join(args.checkpoint_dir, name)))
    if args.root:
        for r in args.root:
            for name in ('phase2_incremental_model.pth', 'phase1_model.pth'):
                ckpts.extend(glob.glob(os.path.join(r, '**', name), recursive=True))
    # one per directory, phase2 preferred
    seen, keep = set(), []
    for c in sorted(set(os.path.normpath(x) for x in ckpts), reverse=True):
        d = os.path.dirname(c)
        if d in seen:
            continue
        seen.add(d); keep.append(c)
    ckpts = sorted(keep)

    if args.seeds:
        want = {f"seed_{s.strip()}" for s in args.seeds.split(',') if s.strip()}
        ckpts = [c for c in ckpts
                 if any(w in os.path.normpath(c).split(os.sep) for w in want)]
        log(f"[--seeds {args.seeds}] filtered to {len(ckpts)} checkpoint(s)")

    banner(f"FOUND {len(ckpts)} CHECKPOINT(S)")
    for c in ckpts:
        s, sd = describe(c)
        log(f"  split={s:<18} seed={sd:<5} {os.path.basename(c)}")
    if not ckpts:
        log('none found - pass --checkpoint_dir, --root or --ckpt'); sys.exit(1)
    if args.list_only:
        return

    all_res = []
    for c in ckpts:
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
