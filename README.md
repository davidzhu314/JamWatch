# Jam Watch

A small CNN that spots and classifies RF jamming from a spectrogram, and tells you when it's unsure. It is trained on four classes (`clean`, `tone`, `wideband`, `sweep`) of synthetic QPSK at 1 MHz complex baseband. A fifth jammer, `pulsed` (a tone gated on and off), is held out completely: it is never used for training, validation, temperature fitting or threshold tuning. The demo compares a plain classifier with a temperature-calibrated one plus an out-of-distribution (OOD) "unfamiliar interference" flag. The intended point is that the plain model is confidently wrong on a jammer it has never seen, and the flag should admit it is unsure. **Result: the first half holds strongly. The second half only partly holds (see below).**

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt     # app deps + torch (torch only needed for training)
python train.py                           # ~3 min on a laptop CPU; writes everything to artifacts/
streamlit run app.py                      # http://localhost:8501
python build_site.py                      # optional: static version in site/ (for Vercel)
```

The app loads `artifacts/` at startup and never trains. Inference runs in plain numpy (BatchNorm folded into the conv weights), so the deployed app does not need torch. `train.py` checks that the numpy and torch logits match (max diff ~1e-5).

## Files

| File | What it does |
|---|---|
| `signals.py` | QPSK + AWGN, the five jammers, 64×64 two-sided log spectrogram, `generate(kind, jsr_db, seed)` |
| `model.py` | CNN (3× conv-BN-ReLU-pool, linear head), BN folding, numpy forward pass, softmax/energy/Mahalanobis |
| `train.py` | Builds the dataset, trains, exports weights, runs calibration, prints every number below |
| `calibrate.py` | Temperature scaling, ECE + reliability (15 bins), OOD scores, thresholds, AUROC, FPR@95TPR |
| `app_core.py` | Artifact loading + single-sample prediction, shared by the app and the static site |
| `charts.py` | Altair chart builders, shared by the app and the static site |
| `app.py` | Streamlit app: live detector + calibration tab |
| `build_site.py`, `site_template.html` | Static site: 210 precomputed samples + the same charts as Vega-Lite |

## Setup details

- **Signal:** fs = 1 MHz, 4096 samples, QPSK at 4 samples/symbol with a windowed-sinc pulse-shaping filter, SNR uniform 5–20 dB.
- **Jammers:** JSR uniform −10 to +20 dB. `tone` is a random-offset complex sinusoid. `wideband` is noise band-limited to 30–300 kHz at a random center. `sweep` is a `scipy.signal.chirp` sawtooth with 1–4 sweeps per example. `pulsed` is a tone gated with period 256–1024 samples and duty cycle 10–60%. For pulsed, JSR is the power while the pulse is on. These ranges were fixed before any results were seen.
- **Spectrogram:** Hann window, 64-sample segments, no overlap, two-sided, giving exactly 64×64 with no resizing. Values are in dB and z-scored per example.
- **Data:** 3000 / 600 / 600 train/val/test per class, plus 1500 pulsed for evaluation only. Every split uses its own seed range.
- **Accuracy on pulsed:** pulsed has no correct label among the 4 classes, so **every prediction on a pulsed sample counts as incorrect** in the "mixed" accuracy, ECE and reliability diagrams.
- **OOD scores (higher = more familiar):**
  - max-softmax at the fitted T
  - energy score `logsumexp(logits)`, with T = 1 (the standard form)
  - Mahalanobis distance in the 64-d penultimate feature space, with class means and shared covariance fit on training data

  Each threshold is the 5th percentile of the in-distribution **validation** scores, so 95% of familiar samples pass. The app uses the score with the best AUROC.
  - Disclosure: Mahalanobis was added after the first run showed max-softmax and energy were weak. It was the only change made after seeing pulsed results, and it did not help.

## Results

| | Plain | Calibrated (T = 1.032) |
|---|---|---|
| Accuracy, seen jammers (test) | 97.8% | 97.8% |
| ECE, seen jammers | 0.007 | 0.008 |
| Mean confidence, seen + pulsed | 94.4% | 94.2% |
| Accuracy, seen + pulsed (pulsed = wrong) | 60.2% | 60.2% |
| ECE, seen + pulsed | 0.342 | 0.340 |
| Pulsed samples with >90% confidence | 67.5% | 66.5% |

Per-class test accuracy: clean 100%, tone 96.7%, wideband 96.2%, sweep 98.3%. Most errors are weak jammers called `clean`.

| OOD score | AUROC (pulsed vs seen) | FPR @ 95% TPR | Pulsed flagged | Seen flagged |
|---|---|---|---|---|
| **max-softmax (used)** | **0.701** | **0.687** | **31%** | **5%** |
| energy | 0.552 | 0.860 | 14% | 5% |
| Mahalanobis | 0.701 | 0.859 | 18% | 6% |

What the model calls pulsed: sweep 62%, wideband 24%, clean 14%, tone 1%.

Pulsed flag rate by JSR (dB): −10…−5: 39%, −5…0: 41%, 0…5: 39%, 5…10: 27%, 10…15: 19%, 15…20: 19%.

### What this means

- **The model is confidently wrong on the unseen jammer.** Two thirds of pulsed samples get >90% confidence, and ECE jumps from 0.007 to 0.34 once pulsed is included.
- **Temperature scaling does nothing here.** The model is already well calibrated on the classes it knows (T ≈ 1), and calibration can't fix confidence on a class the model has never seen.
- **The OOD flag only half-works.** It catches 31% of pulsed at a 5% false-alarm rate (AUROC 0.70). It is weakest exactly where you'd want it most: strong pulsed jammers are flagged only ~19% of the time. Their sharp on/off edges produce broadband vertical clicks, which the model treats as a sweep's wrap-around and labels `sweep` with near-100% confidence.
- **Pulsed is not called `tone`.** The interesting failure is `sweep`, driven by the gating transients.
- None of these numbers were tuned on pulsed data.

## Deploy

**Streamlit Community Cloud**
1. Push this folder to a GitHub repo, including `artifacts/` (~1.3 MB). `.venv/` and `site/` are gitignored.
2. At share.streamlit.io, click "New app", pick the repo, and set the main file to `app.py`. Under Advanced settings, choose Python 3.10 or newer.
3. Cloud installs `requirements.txt`, which has no torch, so the build is small and fast.

**Vercel (static version)**
```bash
python build_site.py
vercel deploy site --prod
```
The static site can't generate new signals. Instead it picks from 210 precomputed samples (5 scenarios × 7 JSR levels × 6 seeds), run through the same model.
