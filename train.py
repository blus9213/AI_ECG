"""
Training loop for single-lead -> 12-lead ECG reconstruction.

Smoke test (no data needed):
    python train.py --synthetic --epochs 3

Real run:
    python train.py --root /path/to/ptb-xl-1.0.3 \
        --input-leads I II V2 V5 --epochs 60
"""

import argparse
import math
import os

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:                                    # pip install tqdm
    def tqdm(it, **kw):
        return it

from ecg_recon import (ECGReconstructor, ReconLoss, amplitude_ratio,
                       consistency_error, kernels_for_fs, pearson_matrix,
                       pearson_per_lead, rmse_per_lead, LEAD_NAMES,
                       INDEPENDENT_IDX, INDEPENDENT_NAMES)
from data import PTBXLDataset, CachedPTBXL, SyntheticECG


def build_loaders(a):
    if a.synthetic:
        tr = SyntheticECG(1024, a.seq_len, a.fs, a.input_leads, seed=0,
                          mask_mode=a.mask_mode)
        va = SyntheticECG(128, a.seq_len, a.fs, a.input_leads, seed=1,
                          mask_mode=a.mask_mode)
    elif a.cache:
        common = dict(cache=a.cache, input_leads=a.input_leads,
                      seq_len=a.seq_len, mask_mode=a.mask_mode)
        tr = CachedPTBXL(split="train", augment=True, **common)
        va = CachedPTBXL(split="val", augment=False, **common)
    else:
        common = dict(root=a.root, input_leads=a.input_leads,
                      fs_out=a.fs, seq_len=a.seq_len, mains=a.mains,
                      mask_mode=a.mask_mode)
        tr = PTBXLDataset(split="train", augment=True, **common)
        va = PTBXLDataset(split="val", augment=False, **common)
    kw = dict(num_workers=a.workers, pin_memory=not a.no_pin)
    if a.workers > 0:
        # Windows respawns workers every epoch otherwise, which for a dataset
        # this cheap costs more than the loading itself.
        kw.update(persistent_workers=True, prefetch_factor=4)
    return (DataLoader(tr, a.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(va, a.batch_size, shuffle=False, **kw))


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, desc="  validating"):
    model.eval()
    r_sum, e_sum, c_sum, n = 0.0, 0.0, 0.0, 0
    idx = torch.tensor(INDEPENDENT_IDX, device=device)
    # Accumulate r for the 8 independent leads, split by whether that lead was
    # given to the model. Averaging over all 12 mixes in leads the model was
    # handed as input — it scores ~1.0 on those and the mean looks great while
    # saying nothing about reconstruction.
    hid_num = torch.zeros(8); hid_den = torch.zeros(8)
    vis_num = torch.zeros(8); vis_den = torch.zeros(8)
    amp_sum = torch.zeros(8)

    for x, _, y12, mask in tqdm(loader, desc=desc, ncols=90, ascii=True,
                                leave=False):
        x, y12 = x.to(device), y12.to(device)
        pred12 = model(x).float()              # (B, 12, L) either way
        b = x.size(0)
        r_sum += pearson_per_lead(pred12, y12).cpu() * b
        e_sum += rmse_per_lead(pred12, y12).cpu() * b
        c_sum += consistency_error(pred12) * b
        n += b

        r8 = pearson_matrix(pred12[:, idx], y12[:, idx]).cpu()   # (B, 8)
        m = mask.cpu()
        hid_num += (r8 * (1 - m)).sum(0); hid_den += (1 - m).sum(0)
        vis_num += (r8 * m).sum(0);       vis_den += m.sum(0)
        amp_sum += amplitude_ratio(pred12[:, idx], y12[:, idx]).cpu() * b

    hidden = hid_num / hid_den.clamp(min=1)
    visible = vis_num / vis_den.clamp(min=1)
    return r_sum / n, e_sum / n, c_sum / n, hidden, visible, amp_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--cache", type=str, default=None,
                   help="Path to the .npy from prepare_cache.py. Strongly "
                        "recommended: per-item wfdb+scipy work starves the GPU.")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--input-leads", nargs="+", default=["I", "II", "V2", "V5"],
                   help="Leads the model receives. I+II already determine all "
                        "six frontal leads, so extra limb leads add nothing; "
                        "spend the budget on precordial leads instead.")
    p.add_argument("--fs", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=5000)
    p.add_argument("--mains", type=float, default=50.0)
    p.add_argument("--patch", type=int, default=20)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--stem-kernel", type=int, default=None,
                   help="Conv width in samples. Default scales with --fs so the "
                        "receptive field stays ~30 ms at any rate.")
    p.add_argument("--refine-kernel", type=int, default=None)
    p.add_argument("--mask-mode", choices=["random", "fixed"], default="random",
                   help="random = masked-autoencoder training, one model for "
                        "any electrode set. fixed = only --input-leads.")
    p.add_argument("--direct-12", action="store_true",
                   help="Head emits all 12 channels instead of 8 + algebra. "
                        "Needs --w-consist to keep the frontal leads coherent.")
    p.add_argument("--w-amp", type=float, default=0.0,
                   help="Weight on the amplitude-matching term. Try 0.3 if "
                        "reconstructions look flattened. 0 reproduces the "
                        "original objective.")
    p.add_argument("--w-consist", type=float, default=0.5,
                   help="Weight on the Einthoven/Goldberger consistency "
                        "penalty. Only active with --direct-12.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-pin", action="store_true",
                   help="Disable pinned memory. Try this if you hit CUDA "
                        "illegal-memory-access or pinned allocator errors.")
    p.add_argument("--accum", type=int, default=1,
                   help="Gradient accumulation steps. Effective batch = "
                        "batch_size * accum. Use this to keep a large effective "
                        "batch when VRAM forces --batch-size down.")
    p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16",
                   help="bf16 needs Ampere or newer (RTX 30/40/50, A100). It "
                        "needs no loss scaler and will not produce NaNs the way "
                        "fp16 can. Falls back to fp16 automatically.")
    p.add_argument("--out", type=str, default="checkpoints")
    a = p.parse_args()

    if not a.synthetic and a.root is None and a.cache is None:
        p.error("pass --cache <cache.npy>, --root <ptb-xl dir>, or --synthetic")

    sk, rk = kernels_for_fs(a.fs)
    a.stem_kernel = a.stem_kernel or sk
    a.refine_kernel = a.refine_kernel or rk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"
    if use_cuda:
        # Ada/Ampere tensor cores. benchmark=True lets cuDNN pick the fastest
        # conv algorithm; safe here because our input shape never varies.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    train_loader, val_loader = build_loaders(a)

    model = ECGReconstructor(seq_len=a.seq_len, patch=a.patch,
                             d_model=a.d_model, n_heads=a.heads,
                             n_layers=a.layers, dropout=a.dropout,
                             n_in=16,      # 8 signal channels + 8 mask channels
                             direct_12=a.direct_12,
                             stem_kernel=a.stem_kernel,
                             refine_kernel=a.refine_kernel).to(device)
    print(f"mask mode: {a.mask_mode}   "
          f"head: {'12 direct' if a.direct_12 else '8 + derived'}   "
          f"kernels: {a.stem_kernel}/{a.refine_kernel} @ {a.fs} Hz")
    print(f"device={device}  params={sum(q.numel() for q in model.parameters())/1e6:.2f}M")

    crit = ReconLoss(w_consist=a.w_consist if a.direct_12 else 0.0,
                     w_amp=a.w_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd,
                            betas=(0.9, 0.95))

    # Pick precision. bf16 has the same exponent range as fp32, so it needs no
    # GradScaler; fp16 does.
    if a.amp == "off" or not use_cuda:
        amp_dtype, amp_on = torch.float32, False
    elif a.amp == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype, amp_on = torch.bfloat16, True
    else:
        amp_dtype, amp_on = torch.float16, True
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype is torch.float16))
        autocast = lambda: torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_on)
    except AttributeError:                                  # torch < 2.4
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16))
        autocast = lambda: torch.cuda.amp.autocast(dtype=amp_dtype, enabled=amp_on)
    if use_cuda:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"vram {vram:.1f} GB   precision {amp_dtype}   "
              f"effective batch {a.batch_size * a.accum}")

    total = a.epochs * len(train_loader)
    warmup = min(1000, total // 20)
    step, best = 0, -1.0

    print(f"\n{len(train_loader)} steps/epoch, {a.epochs} epochs\n")

    for ep in range(1, a.epochs + 1):
        model.train()
        run = 0.0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"epoch {ep}/{a.epochs}", ncols=90,
                   ascii=True, leave=False)
        for micro, (x, y8, y12, mask) in enumerate(bar):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, warmup, a.lr)
            x = x.to(device, non_blocking=True)
            # direct_12: supervise all 12 channels. otherwise: supervise the 8
            # independent leads and let the algebra produce the rest.
            target = (y12 if a.direct_12 else y8).to(device, non_blocking=True)

            with autocast():
                pred = model(x, return_12=a.direct_12)
                loss, parts = crit(pred, target)

            scaler.scale(loss / a.accum).backward()
            if (micro + 1) % a.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            run += loss.item()
            step += 1

            if micro % 10 == 0:
                # Running mean, not the instantaneous value — single-batch loss
                # is noisy enough to hide a real trend.
                bar.set_postfix_str(
                    f"loss {run/(micro+1):.4f}  lr {opt.param_groups[0]['lr']:.2e}",
                    refresh=False)

        if use_cuda:
            torch.cuda.synchronize()      # localise async faults to this epoch
        train_s = time.time() - t0
        r, e, c, hidden, visible, amp = evaluate(model, val_loader, device)
        ips = len(train_loader.dataset) / max(train_s, 1e-9)
        print(f"epoch {ep:3d}  loss {run/len(train_loader):.4f}  "
              f"rmse {e.mean():.4f}  einthoven {c:.1e}  "
              f"[{train_s:.0f}s train, {ips:.0f} rec/s]")
        print(f"  HIDDEN  r = {hidden.mean():.4f}   <-- the number that matters")
        print(f"  visible r = {visible.mean():.4f}   (sanity check, should be ~0.99)")
        print(f"  amp ratio = {amp.mean():.3f}   (1.0 = correct scale, "
              f"<1 = flattened)")
        print("  hidden per lead: " + "  ".join(
            f"{n}:{v:.3f}" for n, v in zip(INDEPENDENT_NAMES, hidden.tolist())),
            flush=True)

        if hidden.mean() > best:
            best = hidden.mean().item()
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "hidden_r": best, "epoch": ep},
                       os.path.join(a.out, "best.pt"))
            print(f"  saved (best hidden r = {best:.4f})")


if __name__ == "__main__":
    main()