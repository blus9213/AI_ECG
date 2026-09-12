"""
Preprocess PTB-XL once into a single memory-mapped array.

Per-item wfdb parsing plus scipy filtering cannot keep a GPU fed — you end up
at 0% utilisation with the model waiting on the CPU. The filtering is
deterministic, so there is no reason to redo it every epoch.

At 100 Hz the whole dataset is 21799 x 12 x 1000 float32 = ~1.0 GB, which the
OS will hold in page cache after the first epoch.

    python prepare_cache.py --root data/ptbxl --fs 100
"""

import argparse
import os
import numpy as np


def main():
    import pandas as pd
    import wfdb
    from data import preprocess, LEAD_NAMES

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fs", type=int, default=None,
                    help="Output rate. Default: native rate of what you have.")
    ap.add_argument("--mains", type=float, default=50.0,
                    help="50 for Europe/Asia/Africa, 60 for the Americas.")
    ap.add_argument("--norm", choices=["fixed", "percentile", "robust"],
                    default="fixed",
                    help="fixed = keep millivolts (recommended). See "
                         "data.normalise for why 'robust' is a trap.")
    ap.add_argument("--clip", type=float, default=10.0,
                    help="Clamp to +/- this many mV; kills saturation artifacts.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    has500 = os.path.isdir(os.path.join(a.root, "records500"))
    if a.fs is None:
        a.fs = 500 if has500 else 100
    col = "filename_hr" if has500 and a.fs >= 250 else "filename_lr"
    fs_in = 500 if col == "filename_hr" else 100
    out = a.out or os.path.join(a.root, f"cache_{a.fs}hz_{a.norm}.npy")

    df = pd.read_csv(os.path.join(a.root, "ptbxl_database.csv"))
    n, L = len(df), a.fs * 10
    print(f"{n} records -> {out}   ({n*12*L*4/1e9:.2f} GB, {a.fs} Hz)")

    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32,
                                    shape=(n, 12, L))
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    order = None
    for i, rel in enumerate(tqdm(df[col].tolist(), desc="preprocessing",
                                 ncols=90, ascii=True)):
        sig, fields = wfdb.rdsamp(os.path.join(a.root, rel))
        if order is None:
            names = [s.strip().upper() for s in fields["sig_name"]]
            order = [names.index(l.upper()) for l in LEAD_NAMES]
        sig12 = np.nan_to_num(sig.T)[order]
        proc, _ = preprocess(sig12, fs_in, a.fs, a.mains,
                             norm=a.norm, clip=a.clip)
        if proc.shape[1] < L:
            proc = np.pad(proc, ((0, 0), (0, L - proc.shape[1])))
        arr[i] = proc[:, :L]
    arr.flush()

    meta = out.replace(".npy", "_meta.npz")
    np.savez(meta, ecg_id=df.ecg_id.to_numpy(),
             strat_fold=df.strat_fold.to_numpy(),
             patient_id=df.patient_id.to_numpy(), fs=a.fs)
    print(f"\nwrote {out}\n      {meta}")
    print(f"\nNow train with --cache {out}")


if __name__ == "__main__":
    main()