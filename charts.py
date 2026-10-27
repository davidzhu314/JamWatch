"""Chart builders shared by the Streamlit app and the static site (build_site.py)."""
import altair as alt
import numpy as np
import pandas as pd

from signals import TRAIN_CLASSES

# ---- palette (dark-mode steps of the reference categorical palette)
SURFACE, PANEL, BORDER = "#0f1115", "#171a21", "#262a33"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#8b8a84"
BLUE, ORANGE, AQUA, AMBER = "#3987e5", "#d95926", "#199e70", "#c98500"
GOOD, WARN = "#2fb47c", "#e5a50a"

SCORE_NAMES = {"msp": "max-softmax", "energy": "energy", "mahalanobis": "Mahalanobis distance"}


def nines(p):
    """-log10(1 - p): 1 = 90%, 2 = 99%, 3 = 99.9%. Spreads out the pile-up near 100%."""
    return np.clip(-np.log10(np.clip(1 - np.asarray(p, np.float64), 1e-6, 1)), 0, 6)


def base_chart(chart, height=260):
    return (chart.properties(height=height, background=SURFACE)
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=INK2, gridColor=BORDER, domainColor=BORDER,
                            tickColor=BORDER, labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=INK2, titleColor=INK2, orient="top", labelFontSize=12, title=None)
            .configure_title(color=INK, fontSize=14, anchor="start", fontWeight=600))


def reliability_chart(M, key, title):
    rows = []
    for model_name, color_name in (("plain", "Plain"), ("calibrated", "Calibrated")):
        r = M["reliability"][f"{key}_{model_name}"]
        e = r["edges"]
        for b in range(len(r["count"])):
            if r["count"][b] > 0:
                rows.append({"model": f'{color_name} (ECE {r["ece"]:.3f})', "conf": r["conf"][b],
                             "acc": r["acc"][b], "n": r["count"][b], "bin": f"{e[b]:.2f}–{e[b+1]:.2f}"})
    df = pd.DataFrame(rows)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color=MUTED, strokeDash=[4, 4], strokeWidth=1.5).encode(x="x", y="y")
    domain = sorted(df["model"].unique(), key=lambda s: not s.startswith("Plain"))
    color = alt.Color("model:N", scale=alt.Scale(domain=domain, range=[ORANGE, BLUE]))
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("conf:Q", title="Model confidence", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("acc:Q", title="Actual accuracy", scale=alt.Scale(domain=[0, 1])), color=color)
    pts = alt.Chart(df).mark_circle(size=70, stroke=SURFACE, strokeWidth=2, opacity=1).encode(
        x="conf:Q", y="acc:Q", color=color,
        tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("bin:N", title="Confidence bin"),
                 alt.Tooltip("conf:Q", title="Avg confidence", format=".2f"),
                 alt.Tooltip("acc:Q", title="Accuracy", format=".2f"), alt.Tooltip("n:Q", title="Samples")])
    return base_chart(alt.layer(diag, line, pts).properties(title=title), 300)


def hist_chart(a_id, a_ood, x_title, threshold=None, bins=30, domain=None):
    lo = float(min(a_id.min(), a_ood.min())) if domain is None else domain[0]
    hi = float(max(a_id.max(), a_ood.max())) if domain is None else domain[1]
    edges = np.linspace(lo, hi, bins + 1)
    rows = []
    for name, arr in (("Seen jammers + clean", a_id), ("Unseen: pulsed", a_ood)):
        h, _ = np.histogram(np.clip(arr, lo, hi), edges)
        for i in range(bins):
            rows.append({"group": name, "x0": edges[i], "x1": edges[i + 1], "share": h[i] / len(arr)})
    df = pd.DataFrame(rows)
    color = alt.Color("group:N", scale=alt.Scale(domain=["Seen jammers + clean", "Unseen: pulsed"],
                                                 range=[BLUE, ORANGE]))
    bars = alt.Chart(df).mark_rect(opacity=0.7, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("x0:Q", title=x_title, scale=alt.Scale(domain=[lo, hi], nice=False)), x2="x1:Q",
        y=alt.Y("share:Q", title="Share of samples", axis=alt.Axis(format="%"), stack=None), color=color,
        tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("x0:Q", title="From", format=".2f"),
                 alt.Tooltip("x1:Q", title="To", format=".2f"), alt.Tooltip("share:Q", title="Share", format=".1%")])
    layers = [bars]
    if threshold is not None:
        tdf = pd.DataFrame({"t": [threshold], "label": ["flag threshold"]})
        layers.append(alt.Chart(tdf).mark_rule(color=WARN, strokeWidth=2, strokeDash=[6, 3]).encode(x="t:Q"))
        layers.append(alt.Chart(tdf).mark_text(color=WARN, align="left", dx=6, dy=-6, baseline="top",
                                               fontSize=11).encode(x="t:Q", y=alt.value(0), text="label:N"))
    return base_chart(alt.layer(*layers))


def ood_hist(M, A):
    score, thr = M["ood_score"], M["ood"][M["ood_score"]]["threshold"]
    s_id, s_ood = A[f"score_{score}_id"], A[f"score_{score}_pulsed"]
    if score == "mahalanobis":
        s_id, s_ood, thr, xt = -s_id, -s_ood, -thr, "Distance from training data (lower = familiar)"
        dom = (float(min(s_id.min(), s_ood.min())), float(np.quantile(np.concatenate([s_id, s_ood]), 0.98)))
    elif score == "msp":
        s_id, s_ood, thr, dom = nines(s_id), nines(s_ood), float(nines(np.array([thr]))[0]), (0, 6)
        xt = "Calibrated confidence, in nines (1 = 90%, 2 = 99%, 3 = 99.9%; 6 = capped)"
    else:
        xt, dom = f"{SCORE_NAMES[score]} score", None
    return hist_chart(s_id, s_ood, xt, threshold=thr, domain=dom, bins=40)


def flag_jsr_chart(M):
    fj = pd.DataFrame([{"jsr": k.replace(" to ", "…"), "rate": v, "order": i}
                       for i, (k, v) in enumerate(M["pulsed_flag_by_jsr"].items())])
    ch = alt.Chart(fj).mark_bar(color=ORANGE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("jsr:N", title="JSR bucket (dB)", sort=alt.SortField("order"), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("rate:Q", title="Flagged unfamiliar", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("jsr:N", title="JSR (dB)"), alt.Tooltip("rate:Q", title="Flagged", format=".0%")])
    return base_chart(ch)


def pred_as_chart(M):
    pa = pd.DataFrame([{"label": k, "share": v} for k, v in M["pulsed_predicted_as"].items()])
    ch = alt.Chart(pa).mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("label:N", title=None, sort=TRAIN_CLASSES),
        x=alt.X("share:Q", title="Share of pulsed samples", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("label:N", title="Predicted"), alt.Tooltip("share:Q", title="Share", format=".1%")])
    return base_chart(ch)


NOTES = {
    "rel_id": "Points near the dashed line mean '90% confident' really is right about 90% of the time. "
              "On jammers it trained on, the model is already honest, so temperature scaling has little to fix.",
    "rel_mixed": "Add the unseen jammer (always counted as wrong, since none of the 4 labels fit) and the points drop "
                 "far below the line: high confidence, low accuracy. Temperature scaling doesn't fix this.",
    "ood": "The threshold is set on seen-jammer validation data so 95% of familiar samples pass; anything left of it "
           "gets the ⚠️ badge. The overlap is the problem: much of the pulsed mass sits at the same 99.9%+ "
           "confidence as familiar jammers, so no threshold can catch it.",
    "jsr": "Counterintuitive: <b>strong</b> pulsed jammers are flagged <b>less</b> often. Their sharp on/off "
           "edges look like a sweep's wrap-around, so the model locks onto 'sweep' with high confidence.",
}


REPO_URL = "https://github.com/davidzhu314/JamWatch"


def strong_flag_rate(M, A, min_jsr=10.0):
    """Share of pulsed samples with JSR >= min_jsr that the OOD flag catches."""
    s, thr = A[f"score_{M['ood_score']}_pulsed"], M["ood"][M["ood_score"]]["threshold"]
    strong = A["pulsed_jsr"] >= min_jsr
    return float(np.mean(s[strong] < thr))


def headline(M, A):
    """Numbers used in the intro, callout and next steps."""
    o = M["ood"][M["ood_score"]]
    return {"acc": f'{M["test_accuracy"]*100:.1f}%', "conf90": f'{M["pulsed_frac_conf_over_90_plain"]*100:.0f}%',
            "flag": f'{o["pulsed_flag_rate"]*100:.0f}%', "strong": f'{strong_flag_rate(M, A)*100:.0f}%',
            "false": f'{o["id_test_flag_rate"]*100:.0f}%'}


def intro_html(h):
    return (f"<b>Calibration isn't enough.</b> This small CNN classifies RF jamming from a spectrogram. It is "
            f"<b>{h['acc']}</b> accurate on the jammers it trained on (clean, tone, wideband, sweep), but confidently "
            f"wrong on an unseen <b>pulsed</b> jammer: {h['conf90']} of pulsed samples get over 90% confidence. "
            f"Temperature scaling doesn't change that, and simple OOD scores flag only <b>{h['flag']}</b> of those "
            f"cases ({h['strong']} when the pulsed jammer is strong).")


def callout_html(h, pred, flagged, conf):
    """Explains a prediction on the unseen pulsed jammer, so a missing badge reads as the finding."""
    why = {
        "sweep": "Each time the pulse switches on or off, it splashes a brief burst of energy across every frequency "
                 "(the faint vertical lines). Those edges look like a sweep wrapping around, so the model answers "
                 "'sweep', often at ~100%.",
        "clean": "A weak pulse is buried in the noise, so the spectrogram looks like a clean signal.",
        "wideband": "The gated on/off bursts smear energy across a band, which the model reads as band-limited noise.",
        "tone": "The pulse sits at one frequency like a tone, but its gaps rarely fool the model this way.",
    }[pred]
    tail = (f"This time the flag fired, which happens for only {h['flag']} of pulsed samples." if flagged else
            f"The flag catches only {h['flag']} of pulsed jammers ({h['strong']} of strong ones), so a missing "
            f"⚠️ badge here is the finding, not a bug.")
    head = "Why so sure?" if conf >= 0.9 else f"Why '{pred}'?"
    return (f"<b>{head}</b> There's no 'pulsed' label, so the model has to pick a class it knows. "
            f"{why} {tail}")


NEXT_STEPS = [
    "Hold out a second unseen jammer (for example, frequency-hopping). Tune the OOD detector on one held-out "
    "jammer, then report on the other, so the reported number is never one the detector was tuned for.",
    "With that tuning set in place, try stronger OOD methods fairly: outlier exposure during training, "
    "multi-layer Mahalanobis, and small ensembles.",
]
