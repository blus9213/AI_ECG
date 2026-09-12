"""
Data pipeline for 12-lead ECG reconstruction.

Primary source: PTB-XL (PhysioNet). 21,837 clinical 10-second 12-lead records,
open access, and it ships an official 10-fold split (`strat_fold`) so results
are comparable with published work: folds 1-8 train, 9 validation, 10 test.

    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

A synthetic generator is included so you can smoke-test the training loop
before the download finishes. Do not report numbers from it.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
INDEPENDENT_IDX = [0, 1, 6, 7, 8, 9, 10, 11]          # I, II, V1..V6
INDEPENDENT_NAMES = [LEAD_NAMES[i] for i in INDEPENDENT_IDX]


# --------------------------------------------------------------------------- #
# Signal preprocessing
# --------------------------------------------------------------------------- #

def bandpass(sig, fs, lo=0.5, hi=100.0, order=4):
    """0.5 Hz high-pass removes baseline wander; the low-pass is set at the
    diagnostic-bandwidth limit so QRS detail survives."""
    hi = min(hi, fs / 2.0 - 1.0)
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1).copy()


def notch(sig, fs, freq=50.0, q=30.0):
    """Mains interference. 50 Hz in Pakistan/Europe/Asia, 60 Hz in the Americas."""
    if freq >= fs / 2.0:
        return sig
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, sig, axis=-1).copy()


def normalise(sig12, mode="fixed", clip=10.0, eps=1e-6):
    """Scale all leads by ONE factor, so relative amplitude between leads is
    preserved (R-wave progression, voltage criteria).

    mode='fixed' (default): keep physical units. PTB-XL is already millivolts
        and a normal R wave is 0.5-2 mV, so the data is close to unit scale as
        it stands. Two reasons this is the right default:
          1. It is computable at inference. Any per-record scale must be derived
             from the VISIBLE leads only, since the hidden ones are what you are
             predicting. A fixed constant sidesteps that entirely.
          2. RMSE comes out in mV, directly comparable with published results.

    mode='robust': the old MAD-of-lead-II estimator. Do not use. An ECG is
        isoelectric most of the time, so its MAD measures baseline noise, not
        signal amplitude — clean records get a tiny denominator and blow up to a
        200x dynamic range. Kept only to reproduce earlier runs.

    mode='percentile': scale by the 99th percentile of |all leads|, which tracks
        R-wave amplitude. Removes body-habitus and electrode-placement variance,
        but is only deployable if you can compute it from the measured leads.

    clip: symmetric clamp in output units, applied last. PTB-XL contains
        saturation artifacts of tens of mV that would otherwise dominate the
        loss. None disables.
    """
    if mode == "fixed":
        scale = 1.0
    elif mode == "percentile":
        scale = max(float(np.percentile(np.abs(sig12), 99.0)), eps)
    elif mode == "robust":
        scale = np.median(np.abs(sig12[1] - np.median(sig12[1]))) * 1.4826
        scale = max(float(scale), eps)
    else:
        raise ValueError(f"unknown norm mode {mode!r}")

    out = sig12 / scale
    if clip is not None:
        out = np.clip(out, -clip, clip)
    return out, scale


def preprocess(sig12, fs_in, fs_out=500, mains=50.0, norm="fixed", clip=10.0):
    """(12, L) raw -> filtered, resampled, normalised."""
    sig12 = np.asarray(sig12, dtype=np.float64)
    sig12 = notch(bandpass(sig12, fs_in), fs_in, mains)
    if fs_in != fs_out:
        g = np.gcd(int(fs_in), int(fs_out))
        sig12 = resample_poly(sig12, fs_out // g, fs_in // g, axis=-1)
    sig12, scale = normalise(sig12, mode=norm, clip=clip)
    return sig12.astype(np.float32), scale


# --------------------------------------------------------------------------- #
# PTB-XL
# --------------------------------------------------------------------------- #

class PTBXLDataset(Dataset):
    """
    Args:
        root:      directory containing ptbxl_database.csv and records500/
        split:     'train' | 'val' | 'test' (official fold assignment)
        input_leads: which leads the model receives, e.g. ["I","II","V2","V5"].
                    Must match what your hardware actually records. Note that
                    I and II together already determine III/aVR/aVL/aVF, so
                    including those adds nothing.
        seq_len:   samples per window at fs_out
    """

    FOLDS = {"train": list(range(1, 9)), "val": [9], "test": [10]}

    def __init__(self, root, split="train", input_leads=("I", "II", "V2", "V5"),
                 fs_out=500, seq_len=5000, mains=50.0, augment=False,
                 mask_mode="random", count_weights=None, always_visible=(),
                 eval_seed=1234):
        """mask_mode:
            'random' -> a fresh random lead subset each time an item is drawn.
                        This is the masked-autoencoder objective: one model that
                        handles any electrode configuration.
            'fixed'  -> always exactly `input_leads`. Use to train or evaluate a
                        single deployment configuration.

        For val/test set augment=False; the mask is then drawn from a fixed seed
        per record, so evaluation is deterministic and comparable across epochs.
        """
        import pandas as pd

        self.root, self.fs_out, self.seq_len = root, fs_out, seq_len
        self.mains, self.augment = mains, augment
        self.mask_mode, self.eval_seed = mask_mode, eval_seed
        self.count_weights, self.always_visible = count_weights, always_visible

        self.fixed_mask = np.zeros(len(INDEPENDENT_IDX), dtype=np.float32)
        for l in input_leads:
            if l not in INDEPENDENT_NAMES:
                raise ValueError(
                    f"{l} is a derived lead. Measure from {INDEPENDENT_NAMES} — "
                    "III/aVR/aVL/aVF carry no information beyond I and II.")
            self.fixed_mask[INDEPENDENT_NAMES.index(l)] = 1.0

        df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
        df = df[df.strat_fold.isin(self.FOLDS[split])].reset_index(drop=True)
        col = "filename_hr" if fs_out >= 250 else "filename_lr"
        self.fs_in = 500 if fs_out >= 250 else 100

        # Fail with a useful message rather than a FileNotFoundError 200 frames
        # deep inside wfdb. Never silently upsample 100 Hz data to stand in for
        # 500 Hz — that would fabricate detail the recording does not contain.
        need = "records500" if self.fs_in == 500 else "records100"
        if not os.path.isdir(os.path.join(root, need)):
            have = [d for d in ("records100", "records500")
                    if os.path.isdir(os.path.join(root, d))]
            raise FileNotFoundError(
                f"fs_out={fs_out} needs {need}/, which is not in {root}. "
                f"Found: {have or 'nothing'}. "
                + ("Use fs_out=100 (--fs 100 --seq-len 1000 --patch 4) or "
                   "download records500." if "records100" in have else
                   "Re-run the download."))
        self.files = df[col].tolist()
        self._order = None      # header-to-canonical lead permutation, cached

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import wfdb
        sig, fields = wfdb.rdsamp(os.path.join(self.root, self.files[i]))

        # Resolve lead order from the WFDB header rather than assuming it.
        # PTB-XL's documentation lists aVL before aVR in one place; trusting
        # a hardcoded order would silently swap two leads throughout training.
        if self._order is None:
            names = [n.strip().upper() for n in fields["sig_name"]]
            self._order = [names.index(l.upper()) for l in LEAD_NAMES]

        sig12 = np.nan_to_num(sig.T)[self._order]          # (12, L), mV
        sig12, _ = preprocess(sig12, self.fs_in, self.fs_out, self.mains)

        L = sig12.shape[1]
        if L >= self.seq_len:
            s = np.random.randint(0, L - self.seq_len + 1) if self.augment else 0
            sig12 = sig12[:, s:s + self.seq_len]
        else:
            sig12 = np.pad(sig12, ((0, 0), (0, self.seq_len - L)))

        if self.augment:
            sig12 = sig12 * np.random.uniform(0.8, 1.25)   # global gain jitter
            if np.random.rand() < 0.3:                      # input-only noise:
                pass                                        # applied below

        y8 = sig12[INDEPENDENT_IDX].copy()                  # (8, L) target basis

        if self.mask_mode == "fixed":
            mask = self.fixed_mask.copy()
        else:
            # Deterministic per record when not training, so val/test masks do
            # not change between epochs and the curve is actually comparable.
            rng = (np.random.default_rng() if self.augment
                   else np.random.default_rng(self.eval_seed + i))
            mask = sample_lead_mask(rng, count_weights=self.count_weights,
                                    always_visible=self.always_visible)

        vis = y8.copy()
        if self.augment and np.random.rand() < 0.3:
            vis = vis + np.random.randn(*vis.shape).astype(np.float32) * 0.02
        x = apply_mask(vis, mask)                           # (16, L)

        return (torch.from_numpy(x), torch.from_numpy(y8),
                torch.from_numpy(sig12), torch.from_numpy(mask))


# --------------------------------------------------------------------------- #
# Lead masking
# --------------------------------------------------------------------------- #

def sample_lead_mask(rng, min_visible=1, max_visible=7, count_weights=None,
                     always_visible=()):
    """Sample which of the 8 INDEPENDENT leads [I, II, V1..V6] are visible.

    Masking over all 12 leads leaks. Hiding I while leaving II and III visible
    hands the model I = II - III for free, so a sizeable fraction of random
    masks would be partly trivial and validation scores would flatter the model.
    Sampling over the independent basis avoids that entirely.

    count_weights: optional length-7 weights over visible counts 1..7. Uniform
        sampling over count spends most of the budget on easy configurations;
        weighting toward few-lead cases matches what you actually deploy.
    always_visible: lead names forced visible, e.g. ("I",) if every device in
        your target hardware records lead I.

    Returns float32 mask of shape (8,), 1.0 = visible.
    """
    n = len(INDEPENDENT_IDX)
    counts = np.arange(min_visible, max_visible + 1)
    if count_weights is not None:
        w = np.asarray(count_weights, dtype=float)[:len(counts)]
        k = int(rng.choice(counts, p=w / w.sum()))
    else:
        k = int(rng.choice(counts))

    forced = [INDEPENDENT_NAMES.index(l) for l in always_visible]
    k = max(k, len(forced))
    pool = [i for i in range(n) if i not in forced]
    extra = rng.choice(pool, size=k - len(forced), replace=False) if k > len(forced) else []

    mask = np.zeros(n, dtype=np.float32)
    mask[forced] = 1.0
    mask[np.asarray(extra, dtype=int)] = 1.0
    return mask


def apply_mask(ind8, mask):
    """(8, L) independent leads + (8,) mask -> (16, L) model input.

    Channels 0-7 are the signals with hidden leads zeroed; channels 8-15 are the
    mask broadcast along time. The model needs the mask explicitly — a zeroed
    channel is otherwise indistinguishable from a genuinely flat lead.
    """
    m = mask[:, None]
    return np.concatenate([ind8 * m, np.broadcast_to(m, ind8.shape)], axis=0
                          ).astype(np.float32)


class CachedPTBXL(Dataset):
    """Reads the memmap built by prepare_cache.py. Same interface as
    PTBXLDataset, but __getitem__ does only a slice and a mask draw, so the
    dataloader stops being the bottleneck."""

    FOLDS = {"train": list(range(1, 9)), "val": [9], "test": [10]}

    def __init__(self, cache, split="train", input_leads=("I", "II", "V2", "V5"),
                 seq_len=None, augment=False, mask_mode="random",
                 count_weights=None, always_visible=(), eval_seed=1234,
                 force_count=None):
        """force_count: if set, every mask has exactly this many visible leads.
        Used at evaluation time — a single average over 1..7 visible leads mixes
        a hard problem with an easy one and tells you little about either."""
        meta = np.load(cache.replace(".npy", "_meta.npz"))
        self.cache_path = cache
        self._data = None          # opened lazily — see _arr()
        self.rows = np.flatnonzero(np.isin(meta["strat_fold"],
                                           self.FOLDS[split]))
        self.fs = int(meta["fs"])
        with open(cache, "rb") as f:
            shape = np.lib.format.read_magic(f) and \
                np.lib.format.read_array_header_1_0(f)[0]
        self.cache_len = shape[2]
        self.seq_len = seq_len or self.cache_len
        self.augment, self.mask_mode, self.eval_seed = augment, mask_mode, eval_seed
        self.count_weights, self.always_visible = count_weights, always_visible
        self.force_count = force_count

        self.fixed_mask = np.zeros(len(INDEPENDENT_IDX), dtype=np.float32)
        for l in input_leads:
            if l not in INDEPENDENT_NAMES:
                raise ValueError(f"{l} is a derived lead; use {INDEPENDENT_NAMES}")
            self.fixed_mask[INDEPENDENT_NAMES.index(l)] = 1.0

    def _arr(self):
        """Open the memmap in whichever process is asking.

        It must NOT be an attribute at pickle time. Windows uses spawn, so the
        DataLoader pickles this whole object to each worker — and pickling an
        np.memmap tries to push the entire array through a pipe, which fails
        with OSError 22 / truncated pickle. Each worker opens its own view
        instead; the OS page cache is shared, so this costs nothing.
        """
        if self._data is None:
            self._data = np.load(self.cache_path, mmap_mode="r")
        return self._data

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_data"] = None          # never pickle the memmap
        return d

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        # np.array (not asarray) forces a writable copy. A memmap slice is
        # read-only, and torch.from_numpy on read-only memory is documented as
        # undefined behaviour — a plausible contributor to the earlier illegal
        # memory access.
        sig12 = np.array(self._arr()[self.rows[i]], dtype=np.float32)

        L = sig12.shape[1]
        if L > self.seq_len:
            s = np.random.randint(0, L - self.seq_len + 1) if self.augment else 0
            sig12 = sig12[:, s:s + self.seq_len]

        y8 = sig12[INDEPENDENT_IDX].copy()
        if self.mask_mode == "fixed":
            mask = self.fixed_mask.copy()
        else:
            rng = (np.random.default_rng() if self.augment
                   else np.random.default_rng(self.eval_seed + i))
            lo, hi = ((self.force_count, self.force_count)
                      if self.force_count else (1, 7))
            mask = sample_lead_mask(rng, min_visible=lo, max_visible=hi,
                                    count_weights=None if self.force_count
                                    else self.count_weights,
                                    always_visible=self.always_visible)

        if self.augment:
            # Global gain jitter: scale input and target together, so the model
            # learns shape rather than absolute amplitude.
            sig12 = sig12 * np.float32(np.random.uniform(0.8, 1.25))
            y8 = sig12[INDEPENDENT_IDX].copy()
        vis = y8
        if self.augment and np.random.rand() < 0.3:
            vis = vis + np.random.randn(*vis.shape).astype(np.float32) * 0.02

        return (torch.from_numpy(apply_mask(vis, mask)), torch.from_numpy(y8),
                torch.from_numpy(sig12), torch.from_numpy(mask))


# --------------------------------------------------------------------------- #
# Synthetic fallback — smoke tests only
# --------------------------------------------------------------------------- #

class SyntheticECG(Dataset):
    """Crude beat model with fixed per-lead projection weights. Enough to verify
    shapes, loss curves and the lead algebra. Not clinically meaningful."""

    def __init__(self, n=512, seq_len=5000, fs=500,
                 input_leads=("I", "II", "V2", "V5"), seed=0,
                 mask_mode="random", count_weights=None, always_visible=()):
        self.n, self.seq_len, self.fs = n, seq_len, fs
        self.mask_mode, self.seed = mask_mode, seed
        self.count_weights, self.always_visible = count_weights, always_visible
        self.fixed_mask = np.zeros(len(INDEPENDENT_IDX), dtype=np.float32)
        for l in input_leads:
            self.fixed_mask[INDEPENDENT_NAMES.index(l)] = 1.0
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 1, (8, 3))       # 8 leads x 3 latent components

    def _wave(self, t, centre, width, amp):
        return amp * np.exp(-((t - centre) ** 2) / (2 * width ** 2))

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        t = np.arange(self.seq_len) / self.fs
        hr = rng.uniform(50, 110)
        comps = np.zeros((3, self.seq_len))
        for beat in np.arange(0, t[-1], 60.0 / hr):
            comps[0] += self._wave(t, beat + 0.00, 0.010, 1.0)    # QRS
            comps[1] += self._wave(t, beat + 0.25, 0.045, 0.30)   # T
            comps[2] += self._wave(t, beat - 0.15, 0.035, 0.15)   # P
        ind = self.w @ comps
        ind += rng.normal(0, 0.02, ind.shape)

        I, II = ind[0], ind[1]
        full = np.stack([I, II, II - I, -(I + II) / 2, I - II / 2, II - I / 2,
                         *ind[2:]]).astype(np.float32)
        y8 = full[INDEPENDENT_IDX].copy()
        mask = (self.fixed_mask.copy() if self.mask_mode == "fixed"
                else sample_lead_mask(rng, count_weights=self.count_weights,
                                      always_visible=self.always_visible))
        return (torch.from_numpy(apply_mask(y8, mask)), torch.from_numpy(y8),
                torch.from_numpy(full), torch.from_numpy(mask))

    def __len__(self):
        return self.n


# --------------------------------------------------------------------------- #
# Self-test:  python data.py            (synthetic, no download needed)
#             python data.py --root ... (real PTB-XL)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--fs", type=int, default=None,
                    help="Default: detected from which records folder exists.")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    if args.root:
        if args.fs is None:
            args.fs = 500 if os.path.isdir(
                os.path.join(args.root, "records500")) else 100
            print(f"detected {args.fs} Hz")
        ds = PTBXLDataset(args.root, split="train", augment=True,
                          fs_out=args.fs, seq_len=args.fs * 10)
        print(f"PTB-XL train split: {len(ds)} records at {args.fs} Hz")
    else:
        ds = SyntheticECG(n=args.n, seq_len=1000, fs=100)
        print(f"synthetic: {len(ds)} records")

    x, y8, y12, m = ds[0]
    print(f"\nx    {tuple(x.shape)}   8 signal + 8 mask channels")
    print(f"y8   {tuple(y8.shape)}   independent-lead target")
    print(f"y12  {tuple(y12.shape)}  full 12-lead, for metrics/plots")
    print(f"mask {tuple(m.shape)}    {m.numpy().astype(int)}")
    print(f"visible: {[n for n, v in zip(INDEPENDENT_NAMES, m) if v > 0]}")

    # Hidden channels must be exactly zero, visible ones must not be.
    sig, mm = x[:8], x[8:]
    for j in range(8):
        nz = float(sig[j].abs().max())
        assert (nz == 0.0) == (m[j] == 0), f"channel {j} zeroing disagrees with mask"
        assert float(mm[j].min()) == float(mm[j].max()) == float(m[j])
    print("\nmask/zeroing consistency  OK")

    # Mask distribution over many draws.
    counts = Counter()
    seen = np.zeros(8)
    N = min(len(ds), args.n)
    for i in range(N):
        _, _, _, mi = ds[i] if args.root is None else ds[i]
        counts[int(mi.sum())] += 1
        seen += mi.numpy()
    print(f"\nvisible-lead count over {N} draws:")
    for k in sorted(counts):
        print(f"  {k} leads: {counts[k]:5d}  ({100*counts[k]/N:5.1f}%)")
    print("\nper-lead visibility rate (should be roughly flat):")
    print("  " + "  ".join(f"{n}:{v/N:.2f}" for n, v in zip(INDEPENDENT_NAMES, seen)))
    assert min(counts) >= 1, "a fully-masked sample would have no input"
    print("\nall checks passed")