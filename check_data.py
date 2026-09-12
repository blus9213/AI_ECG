"""
Run this before training. It verifies the environment, confirms PTB-XL is
laid out where the loader expects it, and checks the assumptions the model
depends on against the real files.

    python check_data.py --root /path/to/ptbxl
"""

import argparse
import os
import sys

import numpy as np

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]


def check_env():
    print("=" * 62)
    print("ENVIRONMENT")
    print("=" * 62)
    print(f"python           {sys.version.split()[0]}")
    ok = True
    for mod in ["numpy", "scipy", "pandas", "wfdb", "torch"]:
        try:
            m = __import__(mod)
            print(f"{mod:16s} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod:16s} MISSING")
            ok = False
    # WSL detection — several failure modes are specific to it.
    is_wsl = "microsoft" in open("/proc/version").read().lower() \
        if os.path.exists("/proc/version") else False
    if is_wsl:
        print("platform         WSL2")

    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            vram = prop.total_memory / 1e9
            print(f"cuda             {torch.version.cuda} -> {prop.name}")
            print(f"vram             {vram:.1f} GB")
            print(f"bf16             {'yes' if torch.cuda.is_bf16_supported() else 'no'}")
            print(f"cudnn            {torch.backends.cudnn.version()}")
            if vram < 7:
                print("                 6 GB class card. Memory scales with "
                      "sequence length,")
                print("                 so it depends on --fs:")
                print("                   --fs 100 (seq 1000): --batch-size 64, "
                      "likely 128")
                print("                   --fs 500 (seq 5000): --batch-size 16 "
                      "--accum 2")
                print("                 Watch nvidia-smi on epoch 1 and raise it "
                      "if idle.")
        else:
            print("cuda             NOT AVAILABLE")
            if is_wsl:
                print("                 On WSL, do NOT install a Linux NVIDIA driver.")
                print("                 The Windows driver supplies libcuda via")
                print("                 /usr/lib/wsl/lib. Check `nvidia-smi` works,")
                print("                 then reinstall torch from the cu124 index.")
            ok = False
    except ImportError:
        pass
    return ok


def check_data(root):
    import pandas as pd
    import wfdb

    print("\n" + "=" * 62)
    print("DATASET")
    print("=" * 62)

    # In WSL2, reading from the Windows filesystem goes over the 9p protocol
    # and is roughly an order of magnitude slower. With ~87k small files this
    # dominates epoch time and starves the GPU.
    real = os.path.realpath(root)
    if real.startswith("/mnt/c") or real.startswith("/mnt/d"):
        print("WARNING: dataset is on the Windows filesystem.")
        print("         Move it into the Linux filesystem (e.g. ~/data/ptbxl)")
        print("         or your GPU will sit idle waiting on disk I/O.\n")

    csv = os.path.join(root, "ptbxl_database.csv")
    if not os.path.exists(csv):
        print(f"FAIL: {csv} not found.")
        print("      --root must point at the folder holding ptbxl_database.csv,")
        print("      records500/ and records100/.")
        return False

    # Which sampling rates were actually downloaded?
    has500 = os.path.isdir(os.path.join(root, "records500"))
    has100 = os.path.isdir(os.path.join(root, "records100"))
    print(f"records500       {'present' if has500 else 'MISSING'}  (500 Hz, _hr)")
    print(f"records100       {'present' if has100 else 'MISSING'}  (100 Hz, _lr)")
    if not (has500 or has100):
        print("FAIL: neither waveform folder found.")
        return False
    if not has500:
        print("                 -> train with --fs 100 --seq-len 1000 --patch 4")
        print("                    (the 500 Hz default would fail)")

    df = pd.read_csv(csv)
    print(f"\nrecords          {len(df)}")
    print(f"patients         {df.patient_id.nunique()}")
    counts = df.strat_fold.value_counts().sort_index()
    print("fold sizes       " + " ".join(f"{k}:{v}" for k, v in counts.items()))
    tr = df.strat_fold.isin(range(1, 9)).sum()
    print(f"split            train {tr} | val {counts.get(9,0)} | test {counts.get(10,0)}")

    col = "filename_hr" if has500 else "filename_lr"
    rec = os.path.join(root, df[col].iloc[0])
    if not os.path.exists(rec + ".hea"):
        print(f"FAIL: waveform {rec}.hea not found. The folder exists but the")
        print("      files inside do not — an interrupted sync? Re-run it; the")
        print("      S3 sync skips what is already complete.")
        return False

    sig, fields = wfdb.rdsamp(rec)
    print(f"\nsample record    {df[col].iloc[0]}")
    print(f"shape            {sig.shape}  (samples, leads)")
    print(f"sampling rate    {fields['fs']} Hz")
    print(f"units            {set(fields['units'])}")

    # Lead ORDER is the assumption most likely to silently break things.
    names = [n.strip() for n in fields["sig_name"]]
    print(f"lead order       {names}")
    if [n.upper() for n in names] == [n.upper() for n in LEAD_NAMES]:
        print("                 matches canonical order")
    else:
        print("                 DIFFERS from canonical — data.py reorders by name,")
        print("                 so this is handled, but worth knowing.")

    # Do the Einthoven/Goldberger identities actually hold in these files?
    # If they do, the four dependent leads are derived, not independently
    # measured, and there is nothing for a model to learn about them.
    idx = {n.upper(): i for i, n in enumerate(names)}
    s = np.nan_to_num(sig.T)
    I, II, III = s[idx["I"]], s[idx["II"]], s[idx["III"]]
    aVR, aVL, aVF = s[idx["AVR"]], s[idx["AVL"]], s[idx["AVF"]]
    # Judge in absolute microvolts against the storage quantisation, NOT as a
    # relative error. PTB-XL is 16-bit at 1 uV/LSB, and aVR/aVL/aVF are defined
    # with a halving, so their exact values sit on a 0.5 uV grid while the
    # stored values are rounded to 1 uV. A ~1 uV discrepancy is the file format,
    # not independently measured leads. A relative threshold flags this as a
    # violation on any record with a small R wave, which is misleading.
    unit_to_uv = 1000.0 if fields["units"][0].lower() == "mv" else 1.0
    lsb_uv = 1.0

    print("\nlead identities (max absolute error)")
    verdicts = []
    for label, resid in [
        ("III = II - I   ", III - (II - I)),
        ("aVR = -(I+II)/2", aVR + (I + II) / 2),
        ("aVL = I - II/2 ", aVL - (I - II / 2)),
        ("aVF = II - I/2 ", aVF - (II - I / 2)),
    ]:
        err_uv = float(np.abs(resid).max()) * unit_to_uv
        exact = err_uv <= 2.5 * lsb_uv
        verdicts.append(exact)
        note = "quantisation only" if exact else "GENUINELY INDEPENDENT"
        print(f"  {label}  {err_uv:7.3f} uV   {note}")

    peak_uv = float(np.abs(II).max()) * unit_to_uv
    print(f"\n  (lead II peak here is {peak_uv:.0f} uV, so 1 uV of rounding is"
          f" {1/max(peak_uv,1e-9):.1e} relative)")
    if all(verdicts):
        print("\n  -> Derived leads are computed, not measured. Use the DEFAULT")
        print("     8-lead head. Do NOT pass --direct-12; there is nothing for")
        print("     a model to learn about these four leads.")
    else:
        print("\n  -> Errors exceed quantisation. Use --direct-12 --w-consist 0.5")
    return True


def plot_sample(root, out="sample_12lead.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import wfdb
    from data import preprocess

    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    col = ("filename_hr" if os.path.isdir(os.path.join(root, "records500"))
           else "filename_lr")
    sig, fields = wfdb.rdsamp(os.path.join(root, df[col].iloc[0]))
    names = [n.strip().upper() for n in fields["sig_name"]]
    order = [names.index(l.upper()) for l in LEAD_NAMES]
    raw = np.nan_to_num(sig.T)[order]
    fs = int(fields["fs"])
    proc, scale = preprocess(raw, fs, fs)

    fig, ax = plt.subplots(12, 1, figsize=(12, 14), sharex=True)
    t = np.arange(proc.shape[1]) / float(fs)
    for i, name in enumerate(LEAD_NAMES):
        ax[i].plot(t, proc[i], lw=0.7)
        ax[i].set_ylabel(name, rotation=0, ha="right", va="center")
        ax[i].margins(x=0)
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"PTB-XL record after preprocessing ({fs} Hz, normalised units)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}  (eyeball this — filtering bugs are obvious visually)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    env_ok = check_env()
    if not env_ok:
        print("\nInstall the missing packages first:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    try:
        data_ok = check_data(a.root)
    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}")
        data_ok = False
    if data_ok and not a.no_plot:
        try:
            plot_sample(a.root)
        except Exception as e:
            print(f"\nplot skipped: {e}")

    print("\n" + "=" * 62)
    print("READY" if (env_ok and data_ok) else "NOT READY — fix the items above")
    print("=" * 62)
