# 12-Lead ECG Reconstruction from Reduced Lead Sets

A masked-lead transformer that reconstructs the full 12-lead ECG from any
subset of measured leads. Trained on PTB-XL, evaluated on the official held-out
fold with per-configuration and per-diagnosis breakdowns.

> **Research code.** Not a medical device, not validated for clinical use, and
> not evaluated prospectively. See [Limitations](#limitations).

---

## Results

Held-out test set (PTB-XL fold 10, n = 2,198). Correlation is computed on
**hidden leads only** — leads given to the model as input score ~1.0 and would
otherwise inflate the average.

| Visible leads | Hidden-lead *r* | RMSE (mV) | ST error (mV) |
|---:|---:|---:|---:|
| 1 | 0.813 | 0.117 | 0.041 |
| 2 | 0.874 | 0.097 | 0.030 |
| 3 | 0.901 | 0.085 | 0.023 |
| 4 | 0.915 | 0.078 | 0.018 |
| 5 | 0.924 | 0.074 | 0.014 |
| 6 | 0.928 | 0.069 | 0.009 |
| 7 | 0.934 | 0.066 | 0.006 |

### Named configurations

| Input leads | Electrodes | *r* | ST error (mV) | % over 0.1 mV |
|---|---:|---:|---:|---:|
| I | 4 | 0.796 | 0.047 | 6.3% |
| I + II | 4 | 0.844 | 0.040 | 4.3% |
| I + II + V3 | 5 | 0.925 | — | — |
| I + II + V2 | 5 | 0.914 | 0.023 | 1.2% |
| I + II + V2 + V5 | 6 | 0.937 | 0.017 | 0.5% |
| I + II + V1 + V3 + V5 | 7 | 0.954 | 0.013 | 0.2% |

Electrode counts include the three limb electrodes (RA, LA, LL) required for
the Wilson central terminal, plus one ground. ST error is mean absolute error
in the 80–120 ms window after each R peak; the clinical threshold for ST
elevation is 0.1 mV.

---

## Findings

### 1. Which leads matters more than how many

Two chest leads (V2 + V5, *r* = 0.910) outperform two limb leads
(I + II, *r* = 0.856). Leads I and II both lie in the frontal plane and span
only two spatial dimensions; the precordial leads sample the transverse plane.

An exhaustive search over all 28 two-lead and 56 three-lead subsets shows the
governing principle is **spatial diversity**, not lead count. The worst subsets
pair redundant directions (I + V6, V5 + V6, I + II); the best spread across
septal, anterior and lateral positions.

Hardware caveat: precordial leads are referenced to the Wilson central terminal,
so V2 cannot be recorded without the three limb electrodes — which then provide
I and II at no extra cost. Rank configurations by **electrode** count.

### 2. Precordial amplitude is biased toward the population mean

Binning test records by their true V4 amplitude:

| True amplitude quintile | Predicted / true |
|---|---:|
| lowest 20% | **1.36** |
| | 1.04 |
| middle | 0.88 |
| | 0.78 |
| highest 20% | **0.60** |

Correlation between true amplitude and ratio: **−0.71** (V4), **−0.57** (V3).

The dataset-average ratio is 0.94, which reads as a 6% error. The reality is
±40% depending on the patient, averaging out to 6%.

This is **not a fixable bug**. Precordial amplitude depends on body habitus,
heart position and electrode placement — none of which are present in the limb
leads. Predicting the conditional mean is the correct estimate under L1 loss.
An amplitude-matching loss term (`--w-amp`) lifts the mean ratio to 0.99 but
leaves the conditional bias unchanged (−0.705 vs −0.708): it rescales the
output without making it patient-specific.

**Consequence:** voltage-based criteria (left ventricular hypertrophy, R-wave
progression) are unreliable on reconstructed leads. Hypertrophy is the
worst-scoring diagnostic class in our evaluation.

### 3. Ectopic beats are flattened, and no aggregate metric catches it

On records containing ventricular ectopy, large precordial complexes (+8 mV in
V3/V4) reconstruct as near-flat — amplitude ratios of 0.14–0.31 — while
surrounding sinus beats reconstruct acceptably. Two beats in a 10-second record
barely move a correlation, so the error is invisible to *r*, RMSE and ST error
alike.

### Pathology costs accuracy

ST error by diagnostic superclass, at I + II:

| Class | n | ST mean (mV) | p95 (mV) |
|---|---:|---:|---:|
| Normal | 963 | 0.029 | 0.054 |
| Myocardial infarction | 550 | 0.048 | 0.117 |
| ST/T change | 521 | 0.044 | 0.104 |
| Conduction disturbance | 496 | 0.052 | 0.140 |
| Hypertrophy | 262 | 0.054 | 0.126 |

Abnormal records carry 1.5–1.9× the error of normal ones. PTB-XL is ~44% normal,
so any dataset-wide average understates error on the cases that matter.

---

## Quickstart

```bash
git clone <this-repo> && cd <this-repo>
python -m venv ecg_env && source ecg_env/bin/activate    # Windows: ecg_env\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /cpu
pip install -r requirements.txt

python train.py --synthetic --epochs 3        # smoke test, no data required
```

### Get the data

**PTB-XL v1.0.3** — 21,799 clinical 12-lead ECGs, 18,869 patients, 10 s each.
Open access under CC-BY 4.0; no credentialing required.

```bash
# Recommended: S3 mirror, resumable, 100 Hz only (~600 MB)
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptbxl \
    --exclude "records500/*"
```

<details>
<summary>Alternatives</summary>

```bash
# Full ZIP (1.7 GB, 3.0 GB extracted)
curl -L -C - -o ptbxl.zip https://physionet.org/content/ptb-xl/get-zip/1.0.3/
tar -xf ptbxl.zip -C data          # use tar, not Expand-Archive — 87k files

# Recursive wget (slow: ~87,000 small files)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

On Windows PowerShell, `wget` and `curl` are aliases for `Invoke-WebRequest`
and reject these flags. Use `curl.exe` explicitly.
</details>

`--root` must point at the folder containing `ptbxl_database.csv`.

### Run

```bash
python check_data.py --root data/ptbxl                  # verify environment + data
python prepare_cache.py --root data/ptbxl --fs 100      # one-time, ~1 GB
python train.py --cache data/ptbxl/cache_100hz_fixed.npy \
    --fs 100 --seq-len 1000 --patch 4 --batch-size 128 --amp bf16
python evaluate.py --ckpt checkpoints/best.pt \
    --cache data/ptbxl/cache_100hz_fixed.npy --root data/ptbxl
python visualize.py --ckpt checkpoints/best.pt --cache data/ptbxl/cache_100hz_fixed.npy
```

**Build the cache.** Per-item WFDB parsing plus scipy filtering starves the GPU
— 0% utilisation in our runs. Caching moved it to 92% and ~520 records/s.

---

## Repository

| File | Purpose |
|---|---|
| `ecg_recon.py` | Model, lead algebra, losses, metrics |
| `data.py` | Datasets, preprocessing, lead masking, cached loader |
| `prepare_cache.py` | One-time preprocessing into a memmap |
| `train.py` | Training loop |
| `evaluate.py` | Test-set report, lead-subset search, diagnostic stratification |
| `visualize.py` | 12-lead reconstruction plots |
| `compare_models.py` | Two-model overlay and the amplitude-bias test |
| `check_data.py` | Environment and dataset verification |

---

## Design decisions

**Predict 8 leads, derive 4.** III, aVR, aVL and aVF are exact linear functions
of I and II. The head emits only `[I, II, V1..V6]`; the rest follow by algebra,
so the Einthoven and Goldberger identities hold exactly (error 0.0) rather than
approximately. `--direct-12` enables a 12-channel head for comparison.

**Mask over the independent basis, not all 12 leads.** Masking uniformly across
12 leaks: hide I while II and III are visible and the model recovers
`I = II − III` exactly. Roughly a third of random masks would be partly free.

**Input is 16 channels** — 8 signal + 8 mask. The mask must be explicit; a
zeroed channel is otherwise indistinguishable from a genuinely flat lead, which
occurs in real recordings when an electrode detaches.

**Score hidden leads only.** Averaging over all 12 mixes in leads the model was
handed as input.

**Conv kernels scale with sampling rate.** 15 taps is 30 ms at 500 Hz but
150 ms at 100 Hz — wide enough to smear the QRS. `kernels_for_fs()` keeps the
receptive field at a fixed duration.

**Fixed physical units, not per-record normalisation.** Values stay in
millivolts, so RMSE is directly interpretable and comparable with published
work. A per-record scale would also be uncomputable at inference, since it would
have to come from the leads being predicted. (An earlier MAD-based normaliser
measured baseline noise rather than signal amplitude and produced a 200×
dynamic range; see `data.normalise`.)

**Loss.** L1 + first-derivative (keeps QRS edges sharp; plain L1 over-smooths)
+ correlation (scale-invariant, enforces shape). Optional amplitude and
consistency terms.

---

## Limitations

Read this section before citing any number above.

- **No comparison against published baselines.** PTB-XL is a standard benchmark
  with existing lead-reconstruction results. Without that comparison, the
  numbers here have no context.
- **Single dataset, single architecture, single seed.** No confidence intervals,
  no cross-dataset validation.
- **100 Hz only.** The 500 Hz variant was not tested. QRS detail above 50 Hz is
  absent by construction.
- **Metrics are QRS-dominated.** Pearson *r* is driven by variance, and almost
  all ECG variance is the QRS complex. A model can score *r* = 0.9 while
  reproducing ST deviation poorly. ST error is reported alongside for this
  reason, but it too is a mean over beats and misses rare events (Finding 3).
- **Reconstruction is visually indistinguishable from measurement.** A clinician
  reading the output cannot tell which leads were inferred. Any deployment
  should mark reconstructed leads distinctly, or use the output as classifier
  features rather than as a human-read ECG.
- **Not validated on pathology at deployment scale.** Abnormal records score
  materially worse than normal ones, and rare morphologies worse still.

---

## Future work

1. **Patient-specific calibration.** One recorded 12-lead snapshot per patient
   fixes a per-lead gain. Targets Finding 2 directly: the unrecoverable quantity
   is a per-patient scale, and one snapshot measures it.
2. **Generative output.** The conditional mean is the *optimal* point estimate
   under L1, so any regressor shrinks. A diffusion or flow-matching head over
   the hidden leads samples the conditional distribution instead. Caveat: it
   would generate plausible detail that is not there, which in a clinical
   context may be worse than visible under-confidence.
3. **Class-balanced sampling.** Training is 44% normal; pathological records
   score 1.5–1.9× worse.
4. **Downstream classification.** Three-way comparison — classifier on true
   12-lead, on reconstructed 12-lead, and on the raw measured leads — to
   determine whether reconstruction adds information or is an expensive
   identity function.
5. **Scale.** 4.9M parameters, 100 Hz. Pretraining on MIMIC-IV-ECG (~800k
   records) with masked-lead prediction is the largest available win.

---

## Hardware

Developed on a single RTX 4050 Laptop (6 GB). At 100 Hz with batch 128: ~34 s
per epoch, ~520 records/s, 2.8 GB VRAM. A 60-epoch run takes about 35 minutes.

<details>
<summary>Notes for 6 GB cards and WSL</summary>

- `--amp bf16` is correct on RTX 30/40/50: same exponent range as fp32, so no
  loss scaler and no NaN blowups mid-run.
- If VRAM-limited, halve `--batch-size` and double `--accum`; gradients are
  unchanged.
- On WSL, never install a Linux NVIDIA driver — the Windows driver supplies
  `libcuda.so` via `/usr/lib/wsl/lib`.
- Keep the dataset in the Linux filesystem. Reading `/mnt/c/...` goes over 9p
  and starves the GPU.
- A system cuDNN on `LD_LIBRARY_PATH` can shadow the one bundled with the torch
  wheels. If you hit cuDNN version errors, unset it.
</details>

---

## Data citation

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I.,
Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available
electrocardiography dataset. *Scientific Data*, 7, 154.

Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation*, 101(23), e215–e220.

PTB-XL is released under CC-BY 4.0.

---

## License

MIT for the code in this repository. PTB-XL has its own terms; see the citation
above.
