"""Temperature scaling, ECE, reliability diagrams, OOD scores and threshold.

Rule: pulsed data is never used to fit anything here. T and the OOD threshold come
from the in-distribution validation set only. Pulsed is only used to *report* metrics.
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar

from model import softmax, energy, mahalanobis

N_BINS = 15
ID_PASS_RATE = 0.95  # threshold lets 95% of in-distribution val samples through


def nll(logits, labels, T):
    p = softmax(logits, T)
    return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))


def fit_temperature(logits, labels):
    res = minimize_scalar(lambda lt: nll(logits, labels, np.exp(lt)), bounds=(-3, 3), method="bounded")
    return float(np.exp(res.x))


def reliability(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    acc, cbin, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, int)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b]:
            acc[b], cbin[b] = correct[m].mean(), conf[m].mean()
    ece = float(np.sum(cnt / len(conf) * np.abs(acc - cbin)))
    return {"edges": edges.tolist(), "acc": acc.tolist(), "conf": cbin.tolist(),
            "count": cnt.tolist(), "ece": ece}


def auroc(s_in, s_out):
    """Probability a random in-dist sample scores higher (more familiar) than a random OOD one."""
    s = np.concatenate([s_in, s_out])
    ranks = np.argsort(np.argsort(s)) + 1.0
    # average ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / counts)[inv]
    r_in = ranks[: len(s_in)].sum()
    return float((r_in - len(s_in) * (len(s_in) + 1) / 2) / (len(s_in) * len(s_out)))


def fpr_at_tpr(s_in, s_out, tpr=0.95):
    thr = np.quantile(s_in, 1 - tpr)
    return float(np.mean(s_out >= thr))


def run(logits_val, y_val, logits_test, y_test, logits_ood, class_names, out_dir, feats, maha, jsr_ood):
    T = fit_temperature(logits_val, y_val)
    results = {"temperature": T, "classes": class_names}

    # ---- accuracy / confusion on in-distribution test
    pred = logits_test.argmax(1)
    k = len(class_names)
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y_test, pred), 1)
    results["test_accuracy"] = float((pred == y_test).mean())
    results["per_class_accuracy"] = {c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(class_names)}
    results["confusion"] = cm.tolist()

    # ---- calibration: ID test, and ID test + pulsed (pulsed always counts as wrong)
    correct_id = (pred == y_test).astype(float)
    correct_mix = np.concatenate([correct_id, np.zeros(len(logits_ood))])
    logits_mix = np.concatenate([logits_test, logits_ood])
    rel = {}
    for name, (lg, corr) in {"id": (logits_test, correct_id), "mixed": (logits_mix, correct_mix)}.items():
        for model_name, temp in {"plain": 1.0, "calibrated": T}.items():
            conf = softmax(lg, temp).max(1)
            r = reliability(conf, corr)
            r["accuracy"] = float(corr.mean())
            r["mean_conf"] = float(conf.mean())
            rel[f"{name}_{model_name}"] = r
    results["reliability"] = rel

    # ---- OOD scores (higher = more familiar). MSP uses calibrated T; energy uses T=1 (standard form).
    # Mahalanobis uses penultimate features, with class means/covariance fit on ID train only.
    scores = {
        "msp": [softmax(z, T).max(1) for z in (logits_val, logits_test, logits_ood)],
        "energy": [energy(z, 1.0) for z in (logits_val, logits_test, logits_ood)],
        "mahalanobis": [mahalanobis(f, *maha) for f in feats],
    }
    ood = {}
    for name, (s_val, s_in, s_out) in scores.items():
        thr = float(np.quantile(s_val, 1 - ID_PASS_RATE))
        ood[name] = {
            "auroc": auroc(s_in, s_out),
            "fpr_at_95tpr": fpr_at_tpr(s_in, s_out),
            "threshold": thr,
            "id_test_flag_rate": float(np.mean(s_in < thr)),
            "pulsed_flag_rate": float(np.mean(s_out < thr)),
        }
        np.save(f"{out_dir}/score_{name}_id.npy", s_in.astype(np.float32))
        np.save(f"{out_dir}/score_{name}_pulsed.npy", s_out.astype(np.float32))
    chosen = max(ood, key=lambda n: ood[n]["auroc"])
    results["ood"] = ood
    results["ood_score"] = chosen

    # flag rate on pulsed, by JSR bucket, for the chosen score
    s_out, thr = scores[chosen][2], ood[chosen]["threshold"]
    buckets = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
    results["pulsed_flag_by_jsr"] = {
        f"{a} to {b}": round(float(np.mean(s_out[(jsr_ood >= a) & (jsr_ood < b)] < thr)), 3) for a, b in buckets}
    np.save(f"{out_dir}/pulsed_jsr.npy", jsr_ood.astype(np.float32))

    # ---- what does the model call pulsed?
    p_ood = logits_ood.argmax(1)
    conf_ood = softmax(logits_ood, T).max(1)
    results["pulsed_predicted_as"] = {c: float(np.mean(p_ood == i)) for i, c in enumerate(class_names)}
    results["pulsed_mean_conf_calibrated"] = float(conf_ood.mean())
    results["pulsed_frac_conf_over_90_calibrated"] = float(np.mean(conf_ood > 0.9))
    results["pulsed_frac_conf_over_90_plain"] = float(np.mean(softmax(logits_ood).max(1) > 0.9))

    np.save(f"{out_dir}/conf_plain_id.npy", softmax(logits_test).max(1).astype(np.float32))
    np.save(f"{out_dir}/conf_plain_pulsed.npy", softmax(logits_ood).max(1).astype(np.float32))
    np.save(f"{out_dir}/conf_cal_id.npy", softmax(logits_test, T).max(1).astype(np.float32))
    np.save(f"{out_dir}/conf_cal_pulsed.npy", conf_ood.astype(np.float32))

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    return results
