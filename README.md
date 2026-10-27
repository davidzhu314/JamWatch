# Jam Watch

**Calibration isn't enough.** The model is 97.8% accurate on known jammers but confidently wrong on an unseen pulsed jammer, and simple OOD scores flag only 31% of those cases (19% for strong pulsed).

**Live demo:** https://jamwatch.vercel.app · **Code:** https://github.com/davidzhu314/JamWatch

Jam Watch is a small CNN that classifies RF jamming from a spectrogram. It is trained on four classes (`clean`, `tone`, `wideband`, `sweep`) of synthetic QPSK at 1 MHz complex baseband. A fifth jammer, `pulsed` (a tone switched on and off), is held out completely: it is never used for training, validation, temperature fitting or threshold tuning. On pulsed samples, 67.5% get over 90% confidence. Temperature scaling (T = 1.03) changes almost nothing, because the model is already well calibrated on the classes it knows. An out-of-distribution (OOD) "unfamiliar interference" flag, set so 95% of familiar samples pass, catches 31% of pulsed samples overall and only 19% of strong ones (JSR ≥ 10 dB). Strong pulses fool it most: their sharp on/off edges look like a sweep wrapping around, so the model answers "sweep" at near-100% confidence.

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

### The finding: calibration isn't enough

- **Confidently wrong on the unseen jammer.** 67.5% of pulsed samples get over 90% confidence, and ECE jumps from 0.007 to 0.34 once pulsed is included.
- **Temperature scaling doesn't help.** The model is already well calibrated on the classes it knows (T = 1.03). Calibration can't fix confidence on a class the model has never seen.
- **Simple OOD scores catch only some of it.** The flag catches 31% of pulsed at a 5% false-alarm rate (AUROC 0.70), and only 19% of strong pulsed (JSR ≥ 10 dB), which is where you'd most want it.
- **Why strong pulses fool it.** Each on/off edge splashes a brief burst of energy across every frequency. The model reads these as a sweep wrapping around and answers `sweep` at near-100% confidence. Pulsed is almost never called `tone` (1%).
- **No tuning on pulsed.** None of these numbers were tuned on pulsed data. The one change made after seeing pulsed results was adding Mahalanobis as a third OOD score, and it did not help.

In the live demo, a pulsed sample with **no** warning badge is the expected, common case, not a bug. After "Surprise me" → "Reveal", the app shows a short callout explaining why.

## Next steps

- **Hold out a second unseen jammer** (for example, frequency hopping). Tune the OOD detector on one held-out jammer and report results on the other, so the reported number never comes from the jammer the detector was tuned on.
- With that tuning set in place, fairly try stronger OOD methods: outlier exposure during training, multi-layer Mahalanobis, and small ensembles.

## Deploy

**Streamlit Community Cloud**
1. Fork or clone https://github.com/davidzhu314/JamWatch. It already includes `artifacts/` (~1.2 MB); `.venv/` and `site/` are gitignored.
2. At share.streamlit.io, click "New app", pick the repo, and set the main file to `app.py`. Under Advanced settings, choose Python 3.10 or newer.
3. Cloud installs `requirements.txt`, which has no torch, so the build is small and fast.

**Vercel (static version)**
```bash
python build_site.py
cd site && vercel deploy --prod
```
The static site can't generate new signals. Instead it picks from 210 precomputed samples (5 scenarios × 7 JSR levels × 6 seeds), run through the same model.
