"""Build every artifact from scratch: dataset -> CNN -> numpy weights -> calibration + OOD metrics.

    python train.py
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

import calibrate
from model import build_torch_model, export_weights, numpy_features, numpy_logits, head, fit_mahalanobis
from signals import TRAIN_CLASSES, generate

SEED = 1234
N_PER_CLASS = {"train": 3000, "val": 600, "test": 600}
N_PULSED_TEST = 1500
EPOCHS = 12
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# Disjoint seed ranges per split/class so no example is ever reused.
SPLIT_OFFSET = {"train": 0, "val": 1_000_000, "test": 2_000_000, "pulsed": 3_000_000}


def _one(args):
    kind, seed = args
    return generate(kind, seed=seed)[1]


def _one_meta(args):
    kind, seed = args
    _, S, meta = generate(kind, seed=seed, return_meta=True)
    return S, meta["jsr_db"]


def make_split(split, pool):
    jobs, labels = [], []
    for ci, kind in enumerate(TRAIN_CLASSES):
        base = SEED * 10_000_000 + SPLIT_OFFSET[split] + ci * 100_000
        jobs += [(kind, base + i) for i in range(N_PER_CLASS[split])]
        labels += [ci] * N_PER_CLASS[split]
    return np.stack(pool.map(_one, jobs, chunksize=200)), np.array(labels)


def main():
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.time()

    with Pool() as pool:
        X = {s: make_split(s, pool) for s in ["train", "val", "test"]}
        base = SEED * 10_000_000 + SPLIT_OFFSET["pulsed"]
        ood = pool.map(_one_meta, [("pulsed", base + i) for i in range(N_PULSED_TEST)], chunksize=200)
    X_ood = np.stack([o[0] for o in ood])
    jsr_ood = np.array([o[1] for o in ood])
    print(f"data: {[(s, v[0].shape) for s, v in X.items()]}, pulsed {X_ood.shape}  ({time.time()-t0:.0f}s)")

    model = build_torch_model(len(TRAIN_CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(X["train"][0][:, None])
    yt = torch.from_numpy(X["train"][1])
    xv, yv = torch.from_numpy(X["val"][0][:, None]), X["val"][1]
    g = torch.Generator().manual_seed(SEED)

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = (model(xv).argmax(1).numpy() == yv).mean()
        print(f"epoch {ep+1:2d}  loss {loss.item():.3f}  val acc {va:.3f}  ({time.time()-t0:.0f}s)")

    weights = export_weights(model)
    np.savez(f"{OUT}/weights.npz", **weights)

    # Sanity check: numpy inference (what the app runs) matches torch.
    with torch.no_grad():
        ref = model(xv[:64]).numpy()
    diff = np.abs(numpy_logits(weights, X["val"][0][:64]) - ref).max()
    assert diff < 1e-3, f"numpy/torch mismatch {diff}"
    print(f"numpy vs torch max logit diff: {diff:.2e}")

    def feats(a):
        return np.concatenate([numpy_features(weights, a[i:i + 500]) for i in range(0, len(a), 500)])

    f_tr, f_val, f_te, f_ood = (feats(a) for a in (X["train"][0], X["val"][0], X["test"][0], X_ood))
    mu, prec = fit_mahalanobis(f_tr, X["train"][1], len(TRAIN_CLASSES))
    np.savez(f"{OUT}/mahalanobis.npz", mu=mu, prec=prec)

    r = calibrate.run(head(weights, f_val), X["val"][1], head(weights, f_te), X["test"][1],
                      head(weights, f_ood), TRAIN_CLASSES, OUT,
                      feats=(f_val, f_te, f_ood), maha=(mu, prec), jsr_ood=jsr_ood)

    rel = r["reliability"]
    print(f"\nT = {r['temperature']:.3f}   ID test acc = {r['test_accuracy']:.3f}")
    print("per-class:", {k: round(v, 3) for k, v in r["per_class_accuracy"].items()})
    print("confusion (rows=true):", r["confusion"])
    for k in ["id_plain", "id_calibrated", "mixed_plain", "mixed_calibrated"]:
        print(f"ECE {k:18s} {rel[k]['ece']:.3f}  acc {rel[k]['accuracy']:.3f}  mean conf {rel[k]['mean_conf']:.3f}")
    for k, v in r["ood"].items():
        print(f"OOD {k:7s} AUROC {v['auroc']:.3f}  FPR@95TPR {v['fpr_at_95tpr']:.3f}  "
              f"pulsed flagged {v['pulsed_flag_rate']:.3f}  ID flagged {v['id_test_flag_rate']:.3f}")
    print("chosen OOD score:", r["ood_score"])
    print("pulsed predicted as:", {k: round(v, 3) for k, v in r["pulsed_predicted_as"].items()})
    print(f"pulsed conf>0.9: plain {r['pulsed_frac_conf_over_90_plain']:.3f}, "
          f"calibrated {r['pulsed_frac_conf_over_90_calibrated']:.3f}")
    print("pulsed flag rate by JSR:", r["pulsed_flag_by_jsr"])
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
