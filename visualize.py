"""
Plot reconstructions against ground truth.

    python visualize.py --ckpt checkpoints/best.pt --cache data/ptbxl/cache_100hz.npy

Writes one PNG per requested lead configuration. Look at these. A scalar
correlation cannot tell you that the model is flattening T waves, rounding QRS
peaks, or inventing a P wave that is not there — a plot shows all three in a
couple of seconds.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ecg_recon import LEAD_NAMES, INDEPENDENT_IDX, INDEPENDENT_NAMES
from data import CachedPTBXL
from evaluate import load_model


def plot_record(true12, pred12, mask, fs, out, title=""):
    """Grey = ground truth, colour = reconstruction. Visible leads are green,
    hidden leads (the actual test) are red."""
    vis = {INDEPENDENT_NAMES[i] for i, v in enumerate(mask) if v > 0}
    t = np.arange(true12.shape[1]) / fs

    fig, ax = plt.subplots(6, 2, figsize=(15, 12), sharex=True)
    ax = ax.T.reshape(-1)
    for i, name in enumerate(LEAD_NAMES):
        derived = name not in INDEPENDENT_NAMES
        if derived:
            kind, colour = "derived", "tab:blue"
        elif name in vis:
            kind, colour = "MEASURED", "tab:green"
        else:
            kind, colour = "reconstructed", "tab:red"

        r = np.corrcoef(true12[i], pred12[i])[0, 1]
        ax[i].plot(t, true12[i], color="0.55", lw=1.6, label="true")
        ax[i].plot(t, pred12[i], color=colour, lw=1.0, label="pred")
        ax[i].set_ylabel(name, rotation=0, ha="right", va="center",
                         fontweight="bold")
        ax[i].set_title(f"{kind}   r={r:.3f}", fontsize=8, loc="right",
                        color=colour)
        ax[i].margins(x=0)
        ax[i].grid(alpha=0.25)
    ax[0].legend(loc="upper right", fontsize=8)
    for j in (5, 11):
        ax[j].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3, help="records per configuration")
    ap.add_argument("--outdir", default="plots")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, targs = load_model(a.ckpt, device)
    fs = targs["fs"]

    configs = [("I",), ("I", "II"), ("I", "II", "V2", "V5")]
    for leads in configs:
        tag = "-".join(leads)
        ds = CachedPTBXL(a.cache, split=a.split, seq_len=fs * 10,
                         augment=False, mask_mode="fixed", input_leads=leads)
        print(f"\nconfiguration: {tag}")
        for k in range(a.n):
            x, _, y12, mask = ds[k * 37]        # spread across the fold
            with torch.no_grad():
                pred = model(x[None].to(device)).float()[0].cpu().numpy()
            plot_record(y12.numpy(), pred, mask.numpy(), fs,
                        os.path.join(a.outdir, f"recon_{tag}_{k}.png"),
                        title=f"input leads: {', '.join(leads)}   "
                              f"(grey = true, red = reconstructed)")


if __name__ == "__main__":
    main()
