"""Jam Watch: a jamming classifier that tells you when it's unsure.

    streamlit run app.py

Loads precomputed artifacts only. Never trains. Inference runs in numpy (no torch needed).
"""
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from app_core import load_artifacts, predict  # noqa: E402
from signals import ALL_KINDS, JSR_RANGE, TRAIN_CLASSES, UNSEEN_CLASSES, generate, spectrogram_axes  # noqa: E402

from charts import (BLUE, BORDER, INK, INK2, MUTED, NOTES, ORANGE, PANEL, SCORE_NAMES, SURFACE,  # noqa: E402
                    NEXT_STEPS, REPO_URL, callout_html, flag_jsr_chart, headline, intro_html, ood_hist,
                    pred_as_chart, reliability_chart)

st.set_page_config(page_title="Jam Watch", page_icon="📡", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.jw-hero h1 {{ font-size: 2.1rem; margin: 0 0 .2rem 0; }}
.jw-hero p {{ color: {INK2}; font-size: 1.02rem; margin: 0; max-width: 760px; }}
.jw-card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; height: 100%; }}
.jw-card .k {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }}
.jw-card .label {{ font-size: 1.7rem; font-weight: 650; color: {INK}; margin: 2px 0 0 0; }}
.jw-card .conf {{ font-size: .95rem; color: {INK2}; margin-bottom: 12px; }}
.jw-row {{ margin: 7px 0; font-size: .84rem; color: {INK2}; }}
.jw-top {{ display: flex; justify-content: space-between; margin-bottom: 3px; }}
.jw-bar {{ background: {BORDER}; border-radius: 4px; height: 6px; overflow: hidden; }}
.jw-bar > div {{ height: 100%; border-radius: 4px; }}
.jw-num {{ text-align: right; font-variant-numeric: tabular-nums; color: {INK}; }}
.jw-badge {{ border-radius: 12px; padding: 14px 18px; font-weight: 600; font-size: 1.05rem; display: flex; gap: 12px; align-items: center; }}
.jw-badge small {{ display: block; font-weight: 400; font-size: .85rem; opacity: .85; margin-top: 2px; }}
.jw-warn {{ background: rgba(229,165,10,.12); border: 1px solid rgba(229,165,10,.55); color: #f5c451; }}
.jw-ok {{ background: rgba(47,180,124,.10); border: 1px solid rgba(47,180,124,.45); color: #6fd3a5; }}
.jw-truth {{ color: {INK2}; font-size: .95rem; padding: 10px 0 0 2px; }}
.jw-truth b {{ color: {INK}; }}
.jw-stat {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; }}
.jw-stat .v {{ font-size: 1.6rem; font-weight: 650; color: {INK}; font-variant-numeric: tabular-nums; }}
.jw-stat .d {{ color: {MUTED}; font-size: .82rem; }}
.jw-links {{ margin-top: 8px !important; font-size: .9rem !important; }}
.jw-links a {{ color: {BLUE}; text-decoration: none; }}
.jw-callout {{ background: rgba(57,135,229,.08); border: 1px solid rgba(57,135,229,.35); border-left: 3px solid {BLUE};
  border-radius: 10px; padding: 12px 16px; margin-top: 12px; color: {INK2}; font-size: .9rem; line-height: 1.5; }}
.jw-callout b {{ color: {INK}; }}
.jw-note {{ color: {INK2}; font-size: .9rem; margin: -4px 0 18px 0; }}
div[data-testid="stTabs"] button p {{ font-size: 1rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- artifacts
@st.cache_resource
def load():
    return load_artifacts()


W, MAHA, M, A = load()
T = M["temperature"]
SCORE = M["ood_score"]
THR = M["ood"][SCORE]["threshold"]


@st.cache_data(max_entries=256)
def run_sample(kind, jsr, seed):
    _, spec, meta = generate(kind, jsr_db=jsr, seed=seed, return_meta=True)
    p_plain, p_cal, _, flagged = predict(W, MAHA, M, spec)
    return spec, p_plain, p_cal, flagged, meta


# ---------------------------------------------------------------- helpers
def spectrogram_fig(spec):
    t0, t1, f0, f1 = spectrogram_axes()
    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(spec, origin="lower", aspect="auto", extent=(t0, t1, f0, f1), cmap="magma",
                   vmin=-2.5, vmax=3.5, interpolation="nearest")
    ax.set_xlabel("Time (ms)", color=INK2, fontsize=9)
    ax.set_ylabel("Frequency offset (kHz)", color=INK2, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    cb.set_label("Power (normalized dB)", color=MUTED, fontsize=8)
    fig.tight_layout()
    return fig


def prob_card(title, probs, color):
    i = int(np.argmax(probs))
    rows = "".join(
        f'<div class="jw-row"><div class="jw-top"><span>{c}</span><span class="jw-num">{p*100:.0f}%</span></div>'
        f'<div class="jw-bar"><div style="width:{max(p*100, 1.5):.1f}%;'
        f'background:{color if j == i else MUTED}"></div></div></div>'
        for j, (c, p) in enumerate(zip(TRAIN_CLASSES, probs)))
    return (f'<div class="jw-card"><div class="k">{title}</div><div class="label">{TRAIN_CLASSES[i]}</div>'
            f'<div class="conf">{probs[i]*100:.1f}% confident</div>{rows}</div>')


def stat(value, label):
    return f'<div class="jw-stat"><div class="v">{value}</div><div class="d">{label}</div></div>'


def note(text):
    st.markdown(f'<div class="jw-note">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("kind", "tone")
ss.setdefault("seed", 7)
ss.setdefault("surprise", False)
ss.setdefault("revealed", False)
rng = np.random.default_rng()


def new_seed():
    ss.seed = int(rng.integers(0, 2**31))
    ss.revealed = False


def on_pick():
    if ss.picker:
        ss.kind = ss.picker
    ss.surprise = False
    new_seed()


def on_surprise():
    ss.kind = str(rng.choice(UNSEEN_CLASSES))
    ss.surprise = True
    ss.picker = None
    new_seed()


# ---------------------------------------------------------------- header
H = headline(M, A)
st.markdown(f"""<div class="jw-hero"><h1>📡 Jam Watch</h1><p>{intro_html(H)}</p>
<p class="jw-links"><a href="{REPO_URL}" target="_blank">Code on GitHub ↗</a></p></div>""", unsafe_allow_html=True)
st.write("")

tab1, tab2 = st.tabs(["Live detector", "Calibration & results"])

# ================================================================ TAB 1
with tab1:
    c1, c2, c3 = st.columns([3.2, 1.1, 1.1], vertical_alignment="bottom")
    with c1:
        st.segmented_control("Scenario", ALL_KINDS, key="picker", default="tone", on_change=on_pick,
                             format_func=lambda k: f"{k} (unseen)" if k in UNSEEN_CLASSES else k)
    with c2:
        st.button("🎲 Surprise me", on_click=on_surprise, width="stretch",
                  help="Picks a jammer type the model never trained on")
    with c3:
        st.button("↻ Regenerate", on_click=new_seed, width="stretch")
    jsr = st.slider("Jammer-to-signal ratio (dB)", *map(int, JSR_RANGE), 10, 1,
                    help="How strong the jammer is relative to the signal. Ignored for 'clean'.")

    spec, p_plain, p_cal, flagged, meta = run_sample(ss.kind, float(jsr), ss.seed)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.pyplot(spectrogram_fig(spec), width="stretch")
        st.caption("QPSK signal in the middle band. A jammer, if present, shows as extra energy: a line (tone), "
                   "a block (wideband), a diagonal (sweep), or dashes (pulsed).")
    with right:
        a, b = st.columns(2)
        a.markdown(prob_card("Plain model", p_plain, ORANGE), unsafe_allow_html=True)
        b.markdown(prob_card(f"Calibrated · T={T:.2f}", p_cal, BLUE), unsafe_allow_html=True)
        st.write("")
        if flagged:
            st.markdown(f'<div class="jw-badge jw-warn">⚠️<div>Unfamiliar interference<small>'
                        f'{SCORE_NAMES[SCORE]} score is outside what the model saw in training. '
                        f'Don\'t trust the label above.</small></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="jw-badge jw-ok">✓<div>Looks familiar<small>'
                        f'{SCORE_NAMES[SCORE]} score is within the training range.</small></div></div>',
                        unsafe_allow_html=True)

        def truth_line():
            unseen = ss.kind in UNSEEN_CLASSES
            tag = " <span style='color:#f5c451'>(never seen in training)</span>" if unseen else ""
            jtxt = "" if ss.kind == "clean" else f" · JSR {meta['jsr_db']:.0f} dB"
            st.markdown(f'<div class="jw-truth">Ground truth: <b>{ss.kind}</b>{tag}{jtxt} · '
                        f'SNR {meta["snr_db"]:.0f} dB</div>', unsafe_allow_html=True)

        if ss.surprise and not ss.revealed:
            st.write("")
            if st.button("Reveal what it really was", width="stretch"):
                ss.revealed = True
                st.rerun()
        else:
            truth_line()
            if ss.kind in UNSEEN_CLASSES:
                pred = TRAIN_CLASSES[int(np.argmax(p_cal))]
                st.markdown(f'<div class="jw-callout">{callout_html(H, pred, flagged, float(p_cal.max()))}</div>', unsafe_allow_html=True)

# ================================================================ TAB 2
with tab2:
    o = M["ood"]
    rel = M["reliability"]
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(stat(f'{M["test_accuracy"]*100:.1f}%', "accuracy on seen jammers"), unsafe_allow_html=True)
    s2.markdown(stat(f'{M["pulsed_frac_conf_over_90_plain"]*100:.0f}%',
                     "of pulsed samples get >90% confidence (plain)"), unsafe_allow_html=True)
    s3.markdown(stat(f'{o[SCORE]["pulsed_flag_rate"]*100:.0f}%', "of pulsed samples flagged unfamiliar"),
                unsafe_allow_html=True)
    s4.markdown(stat(f'{o[SCORE]["id_test_flag_rate"]*100:.0f}%', "of seen samples flagged (false alarms)"),
                unsafe_allow_html=True)
    st.write("")

    st.subheader("Is confidence honest?")
    r1, r2 = st.columns(2, gap="large")
    with r1:
        st.altair_chart(reliability_chart(M, "id", "Seen jammers only"), width="stretch")
        note(NOTES["rel_id"])
    with r2:
        st.altair_chart(reliability_chart(M, "mixed", "Including unseen pulsed jammer"), width="stretch")
        note(NOTES["rel_mixed"])

    st.subheader("Confidence vs. the unfamiliar flag")
    st.altair_chart(ood_hist(M, A), width="stretch")
    note(f"The flag uses the <b>{SCORE_NAMES[SCORE]}</b> score (it separated pulsed best of the three we tried). "
         + NOTES["ood"])

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.subheader("Flag rate on pulsed, by jammer strength")
        st.altair_chart(flag_jsr_chart(M), width="stretch")
        note(NOTES["jsr"])
    with g2:
        st.subheader("What the model calls the pulsed jammer")
        st.altair_chart(pred_as_chart(M), width="stretch")
        top = max(M["pulsed_predicted_as"], key=M["pulsed_predicted_as"].get)
        note(f"With no 'pulsed' option, the model forces it into a known class, most often <b>{top}</b>.")

    st.subheader("Numbers")
    rows = []
    for key, name in (("id", "Seen jammers"), ("mixed", "Seen + unseen pulsed")):
        for mdl in ("plain", "calibrated"):
            r = rel[f"{key}_{mdl}"]
            rows.append({"Test set": name, "Model": mdl.title(), "Accuracy": f'{r["accuracy"]*100:.1f}%',
                         "Mean confidence": f'{r["mean_conf"]*100:.1f}%', "ECE": f'{r["ece"]:.3f}'})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ood_rows = [{"OOD score": SCORE_NAMES[k] + (" ✓ used for flag" if k == SCORE else ""),
                 "AUROC (pulsed vs seen)": f'{v["auroc"]:.3f}', "FPR @ 95% TPR": f'{v["fpr_at_95tpr"]:.3f}',
                 "Pulsed flagged": f'{v["pulsed_flag_rate"]*100:.0f}%',
                 "Seen flagged": f'{v["id_test_flag_rate"]*100:.0f}%'} for k, v in o.items()]
    st.dataframe(pd.DataFrame(ood_rows), hide_index=True, width="stretch")
    st.subheader("Next steps")
    st.markdown("".join(f"- {x}\n" for x in NEXT_STEPS))
    st.caption(f"Temperature T = {T:.3f}, fit on seen-jammer validation data. Thresholds use seen-jammer validation "
               "data only. Pulsed data was never used to fit or tune anything. AUROC: 1.0 = perfect separation, "
               "0.5 = coin flip. FPR @ 95% TPR: share of pulsed samples that slip through when 95% of seen samples pass.")
