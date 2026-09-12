"""
Transformer for reconstructing a 12-lead ECG from a single lead.

Key design point: only 8 of the 12 standard leads are linearly independent.
III, aVR, aVL and aVF are exact algebraic functions of I and II:

    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2

So the network predicts only [I, II, V1..V6] (8 channels) and the remaining
four are derived. This shrinks the output space and makes the derived leads
exactly consistent instead of approximately consistent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PTB-XL / WFDB channel order
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Indices of the 8 independent leads inside the 12-lead array
INDEPENDENT_IDX = [0, 1, 6, 7, 8, 9, 10, 11]   # I, II, V1..V6
INDEPENDENT_NAMES = [LEAD_NAMES[i] for i in INDEPENDENT_IDX]


def derive_full_12(ind: torch.Tensor) -> torch.Tensor:
    """(B, 8, L) in order [I, II, V1..V6]  ->  (B, 12, L) in standard order."""
    I, II = ind[:, 0], ind[:, 1]
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    precordial = [ind[:, i] for i in range(2, 8)]
    return torch.stack([I, II, III, aVR, aVL, aVF, *precordial], dim=1)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class ConvStem(nn.Module):
    """Full-resolution conv front-end, then non-overlapping patch projection.

    The conv layers capture local morphology (QRS slopes, notches) at native
    sample rate; the patch projection hands fixed-length tokens to the
    transformer, which handles the long-range structure (rhythm, beat-to-beat
    context, T-wave relationships).
    """

    def __init__(self, d_model: int, patch: int = 20, width: int = 64,
                 n_in: int = 1, kernel: int = 15):
        super().__init__()
        pad = kernel // 2
        self.local = nn.Sequential(
            nn.Conv1d(n_in, width, kernel, padding=pad),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, kernel, padding=pad),
            nn.BatchNorm1d(width), nn.GELU(),
            nn.Conv1d(width, width, kernel, padding=pad),
            nn.BatchNorm1d(width), nn.GELU(),
        )
        self.proj = nn.Conv1d(width, d_model, kernel_size=patch, stride=patch)

    def forward(self, x):                 # x: (B, 1, L)
        h = self.local(x)                 # (B, W, L)
        return self.proj(h).transpose(1, 2)   # (B, N, d_model)


class ECGReconstructor(nn.Module):
    """Single-lead -> 8 independent leads -> 12 leads."""

    def __init__(
        self,
        seq_len: int = 5000,
        patch: int = 20,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_in: int = 1,
        direct_12: bool = False,
        stem_kernel: int = 15,
        refine_kernel: int = 9,
    ):
        """direct_12=False (default): the head emits the 8 independent leads and
        forward() returns all 12, with III/aVR/aVL/aVF computed exactly.

        direct_12=True: the head emits all 12 channels and the network must
        learn the frontal-lead relationships from data. Use this only if your
        leads are acquired on independent amplifier channels so the algebraic
        identities genuinely do not hold, or if you need it as a baseline for
        comparison. Pair it with consistency_loss() to stop the four dependent
        leads drifting out of physiological agreement.
        """
        super().__init__()
        assert seq_len % patch == 0, "seq_len must be divisible by patch"
        n_out = 12 if direct_12 else 8
        self.seq_len, self.patch = seq_len, patch
        self.n_in, self.n_out, self.direct_12 = n_in, n_out, direct_12
        n_tokens = seq_len // patch

        self.stem = ConvStem(d_model, patch, n_in=n_in, kernel=stem_kernel)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm: far more stable to train
        )
        # enable_nested_tensor=False: the nested-tensor fast path does not apply
        # to pre-norm layers and only emits a warning otherwise. We have no
        # padding anyway — every sequence is the same length.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        # Each token expands back into `patch` samples for each output lead.
        self.head = nn.Linear(d_model, patch * n_out)

        # Refinement at full resolution: smooths patch-boundary discontinuities
        # and lets the model copy fine detail straight from the input lead.
        rpad = refine_kernel // 2
        self.refine = nn.Sequential(
            nn.Conv1d(n_out + n_in, 64, refine_kernel, padding=rpad), nn.GELU(),
            nn.Conv1d(64, 64, refine_kernel, padding=rpad), nn.GELU(),
            nn.Conv1d(64, n_out, refine_kernel, padding=rpad),
        )

    def forward(self, x, return_12=True):
        """x: (B, n_in, L) measured leads. Returns (B, 12, L) or (B, 8, L)."""
        B, _, L = x.shape
        tok = self.stem(x) + self.pos
        tok = self.encoder(self.drop(tok))
        tok = self.norm(tok)

        out = self.head(tok)                                  # (B, N, patch*n_out)
        out = out.view(B, -1, self.n_out, self.patch)         # (B, N, n_out, patch)
        out = out.permute(0, 2, 1, 3).reshape(B, self.n_out, L)

        out = out + self.refine(torch.cat([out, x], dim=1))   # residual refinement
        if self.direct_12:
            return out                                        # already (B, 12, L)
        return derive_full_12(out) if return_12 else out


def kernels_for_fs(fs: int):
    """Conv receptive fields should be a fixed duration, not a fixed number of
    samples. 15 taps is 30 ms at 500 Hz but 150 ms at 100 Hz — wide enough to
    smear the entire QRS complex. Returns (stem_kernel, refine_kernel) giving
    roughly 30 ms and 18 ms at either rate."""
    if fs >= 400:
        return 15, 9
    if fs >= 200:
        return 7, 5
    return 3, 3


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def derivative_loss(pred, target):
    """Penalises slope error. Plain L1/L2 alone yields over-smoothed, rounded
    QRS complexes; matching the first difference keeps the sharp deflections."""
    return F.l1_loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))


def correlation_loss(pred, target, eps=1e-8):
    """1 - Pearson r, computed per (sample, lead) then averaged.
    Scale-invariant, so it forces correct *shape* even where amplitude is off."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(-1)
    den = p.norm(dim=-1) * t.norm(dim=-1) + eps
    return (1.0 - num / den).mean()


def consistency_loss(pred12):
    """Penalises violation of the Einthoven/Goldberger identities.

    Only meaningful with direct_12=True. In the default 8-lead parameterisation
    this is identically zero by construction, which is the whole argument for
    that design.
    """
    I, II, III = pred12[:, 0], pred12[:, 1], pred12[:, 2]
    aVR, aVL, aVF = pred12[:, 3], pred12[:, 4], pred12[:, 5]
    return (F.l1_loss(III, II - I)
            + F.l1_loss(aVR, -(I + II) / 2)
            + F.l1_loss(aVL, I - II / 2)
            + F.l1_loss(aVF, II - I / 2)) / 4.0


@torch.no_grad()
def consistency_error(pred12):
    """Same quantity as a reportable diagnostic (mean absolute violation)."""
    return consistency_loss(pred12).item()


def amplitude_loss(pred, target, eps=1e-8):
    """Penalise per-lead amplitude mismatch.

    L1 and L2 both reward hedging: for a lead the model cannot see, the
    loss-minimising output is a conservative signal near the conditional median,
    so predictions come out visibly flattened. Correlation does not catch this
    at all — it is scale-invariant, so a half-amplitude prediction still scores
    r = 1.0. This term compares standard deviation per lead, which is a direct
    proxy for deflection size, and is the only part of the objective that
    punishes shrinkage.
    """
    ps = pred.std(dim=-1)
    ts = target.std(dim=-1)
    return (torch.log1p(ps + eps) - torch.log1p(ts + eps)).abs().mean()


class ReconLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_deriv=0.5, w_corr=0.3, w_consist=0.0,
                 w_amp=0.0):
        super().__init__()
        self.w_l1, self.w_deriv, self.w_corr = w_l1, w_deriv, w_corr
        self.w_consist = w_consist          # only used when pred has 12 channels
        self.w_amp = w_amp                  # counteracts amplitude collapse

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        dv = derivative_loss(pred, target)
        cr = correlation_loss(pred, target)
        total = self.w_l1 * l1 + self.w_deriv * dv + self.w_corr * cr
        parts = {"l1": l1.item(), "deriv": dv.item(), "corr": cr.item()}
        if self.w_amp > 0:
            am = amplitude_loss(pred, target)
            total = total + self.w_amp * am
            parts["amp"] = am.item()
        if self.w_consist > 0 and pred.shape[1] == 12:
            cs = consistency_loss(pred)
            total = total + self.w_consist * cs
            parts["consist"] = cs.item()
        return total, parts


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@torch.no_grad()
def amplitude_ratio(pred, target, eps=1e-8):
    """(B, C, L) -> (C,) mean std(pred)/std(true). 1.0 is correct; below 1
    means the model is flattening. Report this alongside r, which cannot see
    it."""
    return (pred.std(dim=-1) / (target.std(dim=-1) + eps)).mean(dim=0)


@torch.no_grad()
def pearson_matrix(pred, target, eps=1e-8):
    """(B, C, L) -> (B, C) Pearson r, kept per sample so it can be split by
    which leads were visible."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    return (p * t).sum(-1) / (p.norm(dim=-1) * t.norm(dim=-1) + eps)


@torch.no_grad()
def pearson_per_lead(pred, target, eps=1e-8):
    """(B, C, L) -> (C,) mean Pearson r per lead."""
    return pearson_matrix(pred, target, eps).mean(dim=0)


@torch.no_grad()
def rmse_per_lead(pred, target):
    return torch.sqrt(((pred - target) ** 2).mean(dim=-1)).mean(dim=0)