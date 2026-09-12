"""
Overlay two models against ground truth on the same axes.

    python compare_models.py --ckpt-a checkpoints_fixed/best.pt \
                             --ckpt-b checkpoints_amp/best.pt \
                             --cache data/ptbxl/cache_100hz_fixed.npy

Two separate 12-panel figures cannot be compared by eye — the difference is a
few percent of amplitude, well below what you can judge across two images.
This plots both predictions on one axis, restricted to the leads that actually
differ, with the per-lead amplitude ratio printed on each panel.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_recon import INDEPENDENT_IDX, INDEPENDENT_NAMES, LEAD_NAMES
from data import CachedPTBXL
from evaluate import load_model


def cleanest(ds, n_scan=200, k=3):
    """Pick the least noisy records.

    High-frequency content in an isoelectric segment is mostly noise, so rank
    by the ratio of short-lag differencing energy to total energy. A noisy trace
    makes amplitude differences impossible to judge visually — which is the
    whole point of the plot.
    """
    scores = []
    for i in range(min(n_scan, len(ds))):
        sig = np.asarray(ds._arr()[ds.rows[i]], dtype=np.float32)
        hf = np.abs(np.diff(sig, axis=-1)).mean()
        amp = np.abs(sig).max() + 1e-9
        scores.append((hf / amp, i))
    scores.sort()
    return [i for _, i in scores[:k]]


def scatter(model, a, fs, device, lead="V4", n=800):
    """Amplitude ratio vs true amplitude, per record.

    Tests whether the model is regressing toward population-average amplitude.
    A flat cloud near 1.0 means errors are random. A downward trend means the
    model systematically shrinks large deflections and inflates small ones —
    the Bayes-optimal response to information it does not have, and unfixable
    by any reweighting of the loss.
    """
    from torch.utils.data import DataLoader
    ds = CachedPTBXL(a.cache, split="test", seq_len=fs * 10, augment=False,
                     mask_mode="fixed", input_leads=tuple(a.input_leads))
    ds.rows = ds.rows[:n]
    dl = DataLoader(ds, 128, shuffle=False)
    c = LEAD_NAMES.index(lead)

    ts, rs = [], []
    with torch.no_grad():
        for x, _, y12, _ in dl:
            p = model(x.to(device)).float().cpu()
            t_std = y12[:, c].std(dim=-1)
            p_std = p[:, c].std(dim=-1)
            ts.append(t_std.numpy()); rs.append((p_std / (t_std + 1e-8)).numpy())
    ts = np.concatenate(ts); rs = np.concatenate(rs)

    q = np.quantile(ts, [0, .2, .4, .6, .8, 1.0])
    print(f"\n{lead}: amplitude ratio by true-amplitude quintile")
    print(f"{'true std (mV)':>18}  {'n':>5}  {'mean ratio':>11}")
    for i in range(5):
        m = (ts >= q[i]) & (ts <= q[i + 1])
        print(f"{q[i]:.3f} - {q[i+1]:.3f}   {m.sum():>5}  {rs[m].mean():>11.3f}")
    corr = np.corrcoef(ts, rs)[0, 1]
    print(f"\ncorrelation(true amplitude, ratio) = {corr:+.3f}")
    print("strongly negative => regression toward the population mean amplitude")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ts, rs, s=7, alpha=0.3, color="tab:blue")
    ax.axhline(1.0, color="0.3", lw=1.2, ls="--")
    ax.set_xlabel(f"true {lead} amplitude, std (mV)")
    ax.set_ylabel("predicted / true amplitude")
    ax.set_title(f"{lead}: does the model shrink large deflections?  r = {corr:+.3f}")
    ax.grid(alpha=0.25); ax.set_ylim(0, 2)
    fig.tight_layout()
    out = os.path.join(a.outdir, f"amplitude_bias_{lead}.png")
    os.makedirs(a.outdir, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label-a", default="baseline")
    ap.add_argument("--label-b", default="amplitude term")
    ap.add_argument("--leads", nargs="+", default=["V2", "V3", "V4"],
                    help="Leads to plot. Default: the three with the largest "
                         "amplitude-ratio gap.")
    ap.add_argument("--input-leads", nargs="+", default=["I", "II"])
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="Window to show. Full 10 s compresses the QRS until "
                         "amplitude differences are invisible.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--outdir", default="plots_compare")
    ap.add_argument("--scatter", nargs="*", default=None, metavar="LEAD",
                    help="Run the amplitude-bias test instead of plotting "
                         "traces. Defaults to V3 V4 if no leads given.")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ma, ta = load_model(a.ckpt_a, device)
    mb, _ = load_model(a.ckpt_b, device)
    fs = ta["fs"]

    if a.scatter is not None:
        for lead in (a.scatter or ["V3", "V4"]):
            print(f"\n=== {a.label_a} ==="); scatter(ma, a, fs, device, lead)
            print(f"\n=== {a.label_b} ==="); scatter(mb, a, fs, device, lead)
        return

    ds = CachedPTBXL(a.cache, split="test", seq_len=fs * 10, augment=False,
                     mask_mode="fixed", input_leads=tuple(a.input_leads))
    picks = cleanest(ds, k=a.n)
    nsamp = int(a.seconds * fs)

    for rank, idx in enumerate(picks):
        x, _, y12, mask = ds[idx]
        with torch.no_grad():
            pa = ma(x[None].to(device)).float()[0].cpu().numpy()
            pb = mb(x[None].to(device)).float()[0].cpu().numpy()
        true = y12.numpy()
        t = np.arange(nsamp) / fs

        fig, axes = plt.subplots(len(a.leads), 1, figsize=(13, 3.0 * len(a.leads)),
                                 sharex=True)
        if len(a.leads) == 1:
            axes = [axes]
        for ax, name in zip(axes, a.leads):
            c = LEAD_NAMES.index(name)
            T, A, B = true[c, :nsamp], pa[c, :nsamp], pb[c, :nsamp]
            ax.plot(t, T, color="0.45", lw=2.4, label="true", zorder=1)
            ax.plot(t, A, color="tab:blue", lw=1.3, label=a.label_a, zorder=2)
            ax.plot(t, B, color="tab:red", lw=1.3, label=a.label_b, zorder=3)
            ra = T.std() and A.std() / T.std()
            rb = T.std() and B.std() / T.std()
            ax.set_ylabel(f"{name}\n(mV)", rotation=0, ha="right", va="center",
                          fontweight="bold")
            ax.set_title(f"amplitude ratio   {a.label_a} {ra:.3f}   |   "
                         f"{a.label_b} {rb:.3f}", fontsize=9, loc="right")
            ax.grid(alpha=0.25)
            ax.margins(x=0)
        axes[0].legend(loc="upper right", fontsize=9, ncol=3)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"input: {', '.join(a.input_leads)}   |   "
                     f"reconstructed leads   |   record #{idx}", fontsize=11)
        fig.tight_layout()
        out = os.path.join(a.outdir, f"compare_{rank}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()