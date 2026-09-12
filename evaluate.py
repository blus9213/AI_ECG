"""
Evaluate a checkpoint on fold 10 (the held-out test set).

    python evaluate.py --ckpt checkpoints/best.pt --cache data/ptbxl/cache_100hz.npy

Reports results split by how many leads were visible. A single average over
1..7 visible leads blends a nearly-trivial interpolation problem with a genuinely
hard one; the split is what tells you whether the model is usable for your
electrode budget.

Also reports ST-segment error, because Pearson r does not measure it. r is
driven by variance, and almost all the variance in an ECG is the QRS complex.
A model can score r = 0.9 while getting ST deviation — the thing that indicates
ischaemia — completely wrong.
"""

import argparse

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from ecg_recon import (ECGReconstructor, pearson_matrix, LEAD_NAMES,
                       INDEPENDENT_IDX, INDEPENDENT_NAMES)
from data import CachedPTBXL

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def load_labels(root, cache, split):
    """Map each test record to diagnostic superclasses via scp_statements.csv."""
    import os
    import ast
    import pandas as pd

    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    folds = {"val": [9], "test": [10]}[split]
    df = df[df.strat_fold.isin(folds)].reset_index(drop=True)

    out = {c: np.zeros(len(df), bool) for c in ["NORM", "MI", "STTC", "CD", "HYP"]}
    for i, codes in enumerate(df.scp_codes):
        for code in ast.literal_eval(codes):
            if code in scp.index:
                cls = scp.loc[code].diagnostic_class
                if cls in out:
                    out[cls][i] = True
    return out


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    model = ECGReconstructor(
        seq_len=a["seq_len"], patch=a["patch"], d_model=a["d_model"],
        n_heads=a["heads"], n_layers=a["layers"], dropout=0.0,
        n_in=16, direct_12=a.get("direct_12", False),
        stem_kernel=a.get("stem_kernel", 15),
        refine_kernel=a.get("refine_kernel", 9)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"loaded {ckpt_path}  (epoch {ck.get('epoch')}, "
          f"hidden r {ck.get('hidden_r', float('nan')):.4f})")
    return model, a


@torch.no_grad()
def st_error_per_record(pred, true, fs, lead=1, win=(0.08, 0.12)):
    """Mean absolute ST-level error, in whatever units the data is in.

    Finds R peaks on the TRUE signal, then compares the mean amplitude in the
    ST window (80-120 ms after R) between prediction and truth. Uses lead II
    for detection since its R waves are usually tallest.
    """
    p = pred.cpu().numpy()
    t = true.cpu().numpy()
    a, b = int(win[0] * fs), int(win[1] * fs)
    out = np.full(t.shape[0], np.nan)
    for i in range(t.shape[0]):
        ref = t[i, lead]
        thr = np.percentile(np.abs(ref), 98) * 0.5
        peaks, _ = find_peaks(np.abs(ref), height=max(thr, 1e-6),
                              distance=int(0.25 * fs))
        errs = [abs(p[i, c, r + a:r + b].mean() - t[i, c, r + a:r + b].mean())
                for c in range(t.shape[1]) for r in peaks
                if r + b < t.shape[2]]
        if errs:
            out[i] = float(np.mean(errs))
    return out


@torch.no_grad()
def run(model, loader, device, fs):
    idx = torch.tensor(INDEPENDENT_IDX, device=device)
    hn = torch.zeros(8); hd = torch.zeros(8)
    rmse_n, n = 0.0, 0
    st_all = []

    for x, _, y12, mask in tqdm(loader, ncols=80, ascii=True, leave=False):
        x, y12 = x.to(device), y12.to(device)
        pred12 = model(x).float()
        b = x.size(0)

        r8 = pearson_matrix(pred12[:, idx], y12[:, idx]).cpu()
        m = mask.cpu()
        hn += (r8 * (1 - m)).sum(0)
        hd += (1 - m).sum(0)

        # RMSE over hidden leads only, to match the correlation figure.
        se = ((pred12[:, idx] - y12[:, idx]) ** 2).mean(-1).cpu()
        hidden_mask = (1 - m)
        rmse_n += float((se.sqrt() * hidden_mask).sum())
        st_all.append(st_error_per_record(pred12[:, idx], y12[:, idx], fs))
        n += b

    st = np.concatenate(st_all)
    return (hn / hd.clamp(min=1)), rmse_n / max(float(hd.sum()), 1), st


def search(model, a, fs, device, k, limit=None, require=()):
    """Exhaustively rank every k-lead subset.

    require: leads forced into every combination. Use --require I II for a
    hardware-realistic search. Precordial leads are referenced to the Wilson
    central terminal (the average of RA, LA, LL), so you cannot record V2
    without the three limb electrodes — and once they are placed, I and II cost
    nothing. A subset like V2+V5 is not a 2-electrode design; it is a
    5-electrode design discarding two free leads.
    """
    from itertools import combinations

    rows = []
    pool = [l for l in INDEPENDENT_NAMES if l not in require]
    combos = [tuple(require) + c
              for c in combinations(pool, max(k - len(require), 0))]
    print(f"\nsearching {len(combos)} {k}-lead combinations"
          + (f" containing {'+'.join(require)}" if require else "")
          + (f" on {limit} records" if limit else ""))
    for leads in tqdm(combos, ncols=80, ascii=True):
        ds = CachedPTBXL(a.cache, split=a.split, seq_len=fs * 10,
                         augment=False, mask_mode="fixed", input_leads=leads)
        if limit:
            ds.rows = ds.rows[:limit]
        dl = DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
        r, rmse, st = run(model, dl, device, fs)
        hid = [INDEPENDENT_NAMES.index(h) for h in INDEPENDENT_NAMES
               if h not in leads]
        rows.append((float(r[hid].mean()), float(np.nanmean(st)),
                     float(np.nanpercentile(st, 95)), "+".join(leads)))

    rows.sort(key=lambda x: -x[0])
    print(f"\nbest {k}-lead subsets (ranked by hidden r)")
    print("=" * 66)
    print(f"{'leads':>26}  {'hidden r':>9}  {'ST mean':>8}  {'ST p95':>8}")
    print("-" * 66)
    for r_, st_, p95, name in rows[:10]:
        print(f"{name:>26}  {r_:>9.4f}  {st_:>8.4f}  {p95:>8.4f}")
    print("  ...")
    for r_, st_, p95, name in rows[-3:]:
        print(f"{name:>26}  {r_:>9.4f}  {st_:>8.4f}  {p95:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--search", type=int, default=None, metavar="K",
                    help="Rank every K-lead subset instead of the standard "
                         "report. Try --search 3.")
    ap.add_argument("--require", nargs="*", default=[],
                    help="Leads present in every searched subset, e.g. "
                         "--require I II for an electrode-realistic search.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Records per combination during search. 500 is plenty "
                         "to rank them; drop it for final numbers.")
    ap.add_argument("--root", default=None,
                    help="PTB-XL folder. If given, ST error is also broken down "
                         "by diagnostic superclass — PTB-XL is ~44%% normal ECGs "
                         "with flat ST segments, which flatters the average.")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, targs = load_model(a.ckpt, device)
    fs = targs["fs"]

    if a.search:
        search(model, a, fs, device, a.search, a.limit, tuple(a.require))
        return

    labels = None
    if a.root:
        labels = load_labels(a.root, a.cache, a.split)
        print("diagnostic labels loaded")

    print(f"\nfold 10 ({a.split} split), by number of visible leads")
    print("=" * 74)
    print(f"{'visible':>8}  {'hidden r':>9}  {'RMSE':>8}  {'ST err':>8}   per-lead")
    print("-" * 74)

    for k in range(1, 8):
        ds = CachedPTBXL(a.cache, split=a.split, seq_len=fs * 10,
                         augment=False, mask_mode="random", force_count=k)
        dl = DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
        r, rmse, st = run(model, dl, device, fs)
        worst = INDEPENDENT_NAMES[int(r.argmin())]
        print(f"{k:>8}  {r.mean():>9.4f}  {rmse:>8.4f}  {np.nanmean(st):>8.4f}   "
              f"worst {worst} {r.min():.3f}")

    print("\nnamed configurations")
    print("=" * 74)
    for leads in [("I",), ("II",), ("I", "II"), ("I", "II", "V2"),
                  ("I", "II", "V2", "V5"), ("I", "II", "V1", "V3", "V5")]:
        ds = CachedPTBXL(a.cache, split=a.split, seq_len=fs * 10,
                         augment=False, mask_mode="fixed", input_leads=leads)
        dl = DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
        r, rmse, st = run(model, dl, device, fs)
        hidden = [n for n in INDEPENDENT_NAMES if n not in leads]
        vals = {n: float(v) for n, v in zip(INDEPENDENT_NAMES, r)}
        p50, p90, p95 = np.nanpercentile(st, [50, 90, 95])
        over = float(np.nanmean(st > 0.1)) * 100
        print(f"{'+'.join(leads):>22}  r {r[[INDEPENDENT_NAMES.index(h) for h in hidden]].mean():.4f}"
              f"  RMSE {rmse:.4f}  ST mean {np.nanmean(st):.4f}")
        print(f"      ST p50 {p50:.4f}  p90 {p90:.4f}  p95 {p95:.4f}  "
              f"| {over:.1f}% of records exceed the 0.1 mV threshold")
        print("      " + "  ".join(f"{h}:{vals[h]:.3f}" for h in hidden))

        if labels is not None:
            for cls in ["NORM", "MI", "STTC", "CD", "HYP"]:
                sel = labels[cls][:len(st)]
                if sel.sum() > 20:
                    sub = st[sel]
                    print(f"        {cls:5s} n={int(sel.sum()):5d}  "
                          f"ST mean {np.nanmean(sub):.4f}  "
                          f"p95 {np.nanpercentile(sub, 95):.4f}")

    print("\nNote: r is dominated by the QRS complex. Two models with the same r")
    print("can differ substantially on ST deviation, which is what matters for")
    print("ischaemia. Read the ST column alongside it.")


if __name__ == "__main__":
    main()