"""Build a static version of the demo in site/ (for Vercel or any static host).

Streamlit needs a live Python server, so the static site uses a precomputed bank of
samples run through the same model, plus the same charts as Vega-Lite specs.

    python build_site.py      # after train.py
"""
import base64
import json
import os

import numpy as np
from matplotlib import colormaps

import charts
from app_core import load_artifacts, predict
from signals import ALL_KINDS, TRAIN_CLASSES, UNSEEN_CLASSES, generate

JSRS = [-10, -5, 0, 5, 10, 15, 20]
SEEDS_PER_CELL = 6
SEED_BASE = 9_000_000_000   # far from the training/val/test seed ranges
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "site")


def q8(spec):
    """Quantize a normalized spectrogram to bytes over the same color range the app uses."""
    return np.clip((spec + 2.5) / 6.0 * 255, 0, 255).astype(np.uint8)


def main():
    W, MAHA, M, A = load_artifacts()
    H = charts.headline(M, A)
    bank = {}
    for ki, kind in enumerate(ALL_KINDS):
        for jsr in JSRS:
            cell = []
            for s in range(SEEDS_PER_CELL):
                seed = SEED_BASE + ki * 1000 + s if kind == "clean" else SEED_BASE + ki * 1000 + (jsr + 50) * 10 + s
                _, spec, meta = generate(kind, jsr_db=float(jsr), seed=seed, return_meta=True)
                p_plain, p_cal, score, flagged = predict(W, MAHA, M, spec)
                cell.append({"s": base64.b64encode(q8(spec).tobytes()).decode(),
                             "p": [round(float(x), 4) for x in p_plain],
                             "c": [round(float(x), 4) for x in p_cal],
                             "f": bool(flagged), "snr": round(meta["snr_db"], 1)})
                if kind in UNSEEN_CLASSES:
                    cell[-1]["co"] = charts.callout_html(H, TRAIN_CLASSES[int(np.argmax(p_cal))], flagged, float(p_cal.max()))
            bank[f"{kind}|{jsr}"] = cell

    specs = {
        "rel_id": charts.reliability_chart(M, "id", "Seen jammers only"),
        "rel_mixed": charts.reliability_chart(M, "mixed", "Including unseen pulsed jammer"),
        "ood": charts.ood_hist(M, A),
        "jsr": charts.flag_jsr_chart(M),
        "pred": charts.pred_as_chart(M),
    }
    specs = {k: {**v.to_dict(), "width": "container", "autosize": {"type": "fit", "contains": "padding"}}
             for k, v in specs.items()}

    lut = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    data = {"bank": bank, "jsrs": JSRS, "kinds": ALL_KINDS, "metrics": M, "specs": specs,
            "notes": charts.NOTES, "intro": charts.intro_html(H), "next": charts.NEXT_STEPS,
            "repo": charts.REPO_URL, "scoreName": charts.SCORE_NAMES[M["ood_score"]],
            "lut": base64.b64encode(lut.tobytes()).decode()}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, "site_template.html")) as f:
        html = f.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(os.path.join(OUT, "index.html"), "w") as f:
        f.write(html)
    print(f"wrote site/index.html ({os.path.getsize(os.path.join(OUT, 'index.html')) / 1e6:.2f} MB, "
          f"{sum(len(v) for v in bank.values())} samples)")


if __name__ == "__main__":
    main()
