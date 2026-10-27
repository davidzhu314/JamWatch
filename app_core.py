"""Artifact loading + single-sample inference (numpy only). Shared by app.py and build_site.py."""
import json
import os

import numpy as np

from model import energy, head, mahalanobis, numpy_features, softmax

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def load_artifacts():
    w = dict(np.load(f"{ART}/weights.npz"))
    m = np.load(f"{ART}/mahalanobis.npz")
    with open(f"{ART}/metrics.json") as f:
        metrics = json.load(f)
    arrays = {n[:-4]: np.load(f"{ART}/{n}") for n in os.listdir(ART) if n.endswith(".npy")}
    return w, (m["mu"], m["prec"]), metrics, arrays


def predict(W, maha, M, spec):
    """Returns (plain probs, calibrated probs, OOD score, flagged)."""
    T, score_name = M["temperature"], M["ood_score"]
    f = numpy_features(W, spec)
    z = head(W, f)
    score = {"msp": lambda: softmax(z, T).max(1), "energy": lambda: energy(z),
             "mahalanobis": lambda: mahalanobis(f, *maha)}[score_name]()[0]
    return softmax(z)[0], softmax(z, T)[0], float(score), bool(score < M["ood"][score_name]["threshold"])
