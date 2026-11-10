"""Shared UI helpers: styling, cached artifacts, passenger state, cards."""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.features import FEATURE_LABELS  # noqa: E402
from src.models import BASELINES, MEMBER_MODELS, MODEL_INFO, predict_all  # noqa: E402
from src.evaluation import DISPLAY_METRICS as METRICS  # noqa: E402,F401


def _artifact_dir() -> Path:
    """Full-experiment artifacts when train_full.py has produced them, else the prototype.
    SKYPULSE_ARTIFACTS overrides (e.g. to inspect a smoke run)."""
    import os
    if os.environ.get("SKYPULSE_ARTIFACTS"):
        return Path(os.environ["SKYPULSE_ARTIFACTS"])
    return C.ARTIFACTS_FULL if (C.ARTIFACTS_FULL / "models.joblib").exists() else C.ARTIFACTS


ART = _artifact_dir()
EXP_DIR = ART / "experiments"

GOOD, BAD, ACCENT, MUTED = "#3ddc97", "#ff5d5d", "#ffb627", "#8a9bb4"
MODEL_COLORS = {"knn": "#5ab0ff", "svm": "#c792ea", "mlp": "#3ddc97", "xgb": "#ffb627",
                "team": "#ffffff", "rule": "#8a9bb4", "dummy": "#4a5a73", "xgb_mono": "#ff8c42"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@600;800&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --bg:#0b1220; --panel:#121c2e; --line:#22314a; --text:#e9eef6; --muted:#8a9bb4;
        --accent:#ffb627; --good:#3ddc97; --bad:#ff5d5d; }
html, body, .stMarkdown, p, label, input, textarea, button, [data-baseweb="select"] div { font-family:'Instrument Sans',sans-serif; }
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
h1, h2, h3, h4 { font-family:'Big Shoulders Display',sans-serif !important; letter-spacing:.02em; text-transform:uppercase; }
.stApp { background:
   radial-gradient(1200px 500px at 85% -10%, rgba(255,182,39,.07), transparent 60%),
   repeating-linear-gradient(90deg, rgba(255,255,255,.018) 0 1px, transparent 1px 80px), var(--bg); }
[data-testid="stSidebar"] { background:#0e1729; border-right:1px solid var(--line); }
.mono { font-family:'JetBrains Mono',monospace; }

/* Page header: departure-board strip */
.board { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; flex-wrap:wrap;
         border-bottom:1px solid var(--line); padding-bottom:.8rem; margin-bottom:1.2rem; }
.board .code { font-family:'JetBrains Mono',monospace; color:var(--accent); font-size:.8rem; letter-spacing:.25em; }
.board h1 { margin:0; padding:0; font-size:2.6rem; line-height:1; }
.board .sub { color:var(--muted); font-size:.92rem; max-width:44ch; }

/* Split-flap verdict */
.flaps { display:flex; gap:4px; flex-wrap:wrap; margin:.2rem 0 .6rem; }
.flap { width:2.1rem; height:2.9rem; display:grid; place-items:center; border-radius:4px;
        font-family:'JetBrains Mono',monospace; font-weight:600; font-size:1.6rem;
        background:linear-gradient(#18243a 49%, #0c1424 50%, #141f33); box-shadow:inset 0 0 0 1px #24344f;
        animation:flip .45s cubic-bezier(.3,1.4,.6,1) both; transform-origin:50% 50%; }
.flap.good { color:var(--good); } .flap.bad { color:var(--bad); }
@keyframes flip { from { transform:rotateX(90deg); opacity:0 } to { transform:rotateX(0); opacity:1 } }

/* Model cards (boarding-pass stubs) */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr)); gap:.7rem; }
.card { position:relative; background:var(--panel); border:1px solid var(--line); border-radius:10px;
        padding:.8rem .9rem .75rem; overflow:hidden; animation:rise .5s ease both; }
.card::after { content:""; position:absolute; right:-9px; top:50%; width:18px; height:18px; border-radius:50%;
               background:var(--bg); border:1px solid var(--line); transform:translateY(-50%); }
.card.base { border-style:dashed; opacity:.82; }
.card.team { border-color:var(--accent); box-shadow:0 0 0 1px rgba(255,182,39,.25), 0 10px 30px -12px rgba(255,182,39,.35); }
.card .role { font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:.18em; color:var(--muted); text-transform:uppercase; }
.card .name { font-family:'Big Shoulders Display',sans-serif; font-size:1.25rem; font-weight:800; text-transform:uppercase; }
.card .pct { font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:600; line-height:1.1; }
.card .bar { height:6px; border-radius:3px; background:#1d2a41; margin:.45rem 0 .4rem; overflow:hidden; }
.card .bar > i { display:block; height:100%; border-radius:3px; }
.chip { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.08em;
        padding:.12rem .45rem; border-radius:99px; border:1px solid; }
.chip.good { color:var(--good); border-color:rgba(61,220,151,.4); } .chip.bad { color:var(--bad); border-color:rgba(255,93,93,.4); }
@keyframes rise { from { transform:translateY(8px); opacity:0 } to { transform:none; opacity:1 } }

.note { border-left:3px solid var(--accent); background:rgba(255,182,39,.06); padding:.55rem .8rem;
        border-radius:0 6px 6px 0; color:#d8dfeb; font-size:.9rem; margin:.4rem 0 .8rem; }
.note.warn { border-color:var(--bad); background:rgba(255,93,93,.07); }
.step { display:flex; gap:.7rem; align-items:center; padding:.45rem 0; border-bottom:1px dashed var(--line); }
.step .n { font-family:'JetBrains Mono',monospace; color:var(--accent); width:1.6rem; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:.7rem .9rem;
       min-height:5.2rem; height:100%; display:flex; flex-direction:column; justify-content:space-between; min-width:0; }
.kpi .l { font-family:'JetBrains Mono',monospace; font-size:.68rem; letter-spacing:.15em; color:var(--muted);
          text-transform:uppercase; line-height:1.35; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-size:clamp(1rem, 1.45vw, 1.5rem); font-weight:600;
          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .s { font-size:.74rem; color:var(--muted); margin-top:.15rem; line-height:1.3; }

/* Tabs: make section switching obvious during a demo */
[data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
[data-baseweb="tab"] { padding:.35rem .7rem !important; border-radius:6px 6px 0 0; }
[data-baseweb="tab"] p { font-size:.95rem !important; font-weight:600; }
[data-baseweb="tab"][aria-selected="true"] { background:rgba(255,182,39,.08); }
[data-baseweb="tab"][aria-selected="true"] p { color:var(--accent); }

/* Chart / section headings: tighter, consistent rhythm */
h4 { font-size:1.15rem !important; margin-top:.6rem !important; color:#cfd8e6; }
.step-head { display:flex; gap:.7rem; padding:.2rem 0 .35rem; border-bottom:1px solid var(--line);
             font-family:'JetBrains Mono',monospace; font-size:.66rem; letter-spacing:.14em;
             color:var(--muted); text-transform:uppercase; }
.step .n, .step-head .n { min-width:2.4rem; white-space:nowrap; }

/* Buttons show their full label (wrap) instead of an ellipsis */
.stButton button .stMarkdownEllipsis, .stButton button .stMarkdownEllipsis p,
.stDownloadButton button .stMarkdownEllipsis, .stDownloadButton button .stMarkdownEllipsis p {
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important; line-height:1.25; }

/* Segmented controls wrap instead of clipping on narrow screens */
[data-testid="stButtonGroup"] > div { flex-wrap:wrap; }

/* Predict page: two-column layout stacks when there is not enough room */
.st-key-predict_top [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 21rem); }
.st-key-predict_main > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-predict_main > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 20rem); }

/* Wide fixed-size charts scroll inside their own box rather than widening the page */
.st-key-fair_chart { overflow-x:auto; max-width:100%; }

@media (max-width: 1100px) {
  .board h1 { font-size:2rem; }
  .flap { width:1.7rem; height:2.4rem; font-size:1.3rem; }
  .cards { grid-template-columns:repeat(auto-fill,minmax(145px,1fr)); }
}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def header(code: str, title: str, sub: str):
    st.markdown(f'<div class="board"><div><div class="code">{code}</div><h1>{title}</h1></div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


# ---------- artifacts ----------
@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    path = ART / "models.joblib" if (ART / "models.joblib").exists() else ART / "prototype_models.joblib"
    if not path.exists():
        st.error("No trained models found. Run `python prototype_train.py` first.")
        st.stop()
    return joblib.load(path)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(ART / "test_predictions.parquet")


@st.cache_data
def load_metrics() -> dict:
    return json.loads((ART / "metrics.json").read_text())


@st.cache_resource(show_spinner=False)
def load_variants():
    from src.store import load_variants as _load
    return _load(ART)


@st.cache_resource(show_spinner=False)
def load_trust():
    import joblib
    return joblib.load(ART / "trust.joblib")


def predict(X: pd.DataFrame) -> pd.DataFrame:
    return predict_all(load_models(), X)


def predict_stage(X: pd.DataFrame, stage: str) -> float:
    """P(satisfied) from the stage-specific model (Landed = team ensemble)."""
    if stage == "Landed":
        return float(predict(X)["team"].iloc[0])
    return float(load_variants()[f"stage::{stage}"].predict_proba(X)[0, 1])


def conformal_label(p: float, level: float, model: str = "team") -> tuple[str, bool]:
    """Conformal prediction set for one passenger -> (text, is_certain)."""
    from src.trust import conformal_qhat, conformal_sets
    cal = load_trust()["cal"]
    qhat = conformal_qhat(cal[model].to_numpy(), cal["y_true"].to_numpy(), 1 - level)
    has_s, has_d = conformal_sets(np.array([p]), qhat)
    labels = [n for n, h in [("Satisfied", has_s[0]), ("Dissatisfied", has_d[0])] if h]
    return "{" + ", ".join(labels) + "}" if labels else "{ } " + "(empty)", len(labels) == 1


def ood_percentile(X: pd.DataFrame) -> float:
    return float(load_trust()["ood"].percentile(X)[0])


@st.cache_data(show_spinner=False)
def test_shap(n: int = 6000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(SHAP values with friendly column names, matching raw rows) for a hold-out sample."""
    preds = load_test_predictions()
    raw = preds.sample(min(n, len(preds)), random_state=C.SEED)
    pipe = load_models()["xgb"]
    feats = pipe.named_steps["features"].transform(raw[C.FEATURE_COLS])
    contrib = pipe.named_steps["model"].get_booster().predict(
        xgb.DMatrix(feats, feature_names=list(feats.columns)), pred_contribs=True)[:, :-1]
    shap_df = pd.DataFrame(contrib, columns=[FEATURE_LABELS.get(c, c) for c in feats.columns], index=raw.index)
    return shap_df, raw


@st.cache_data
def cabin_rates() -> dict:
    df = load_test_predictions()
    return (df["team"] >= 0.5).groupby(df["Class"].astype(str)).mean().to_dict()


# ---------- passenger state ----------
PRESETS = {
    "Frustrated economy flyer": {
        "Gender": "Female", "Customer Type": "disloyal Customer", "Age": 24, "Type of Travel": "Personal Travel",
        "Class": "Eco", "Flight Distance": 620, "Departure Delay in Minutes": 45, "Arrival Delay in Minutes": 52,
        "ratings": [2, 4, 2, 3, 2, 2, 3, 2, 3, 2, 3, 3, 3, 2]},
    "Loyal business traveller": {
        "Gender": "Male", "Customer Type": "Loyal Customer", "Age": 42, "Type of Travel": "Business travel",
        "Class": "Business", "Flight Distance": 2450, "Departure Delay in Minutes": 0, "Arrival Delay in Minutes": 0,
        "ratings": [4, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4, 5, 4]},
    "Business trip, long delay": {
        "Gender": "Female", "Customer Type": "Loyal Customer", "Age": 36, "Type of Travel": "Business travel",
        "Class": "Eco Plus", "Flight Distance": 1300, "Departure Delay in Minutes": 120, "Arrival Delay in Minutes": 135,
        "ratings": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]},
    "Borderline case": {
        "Gender": "Male", "Customer Type": "Loyal Customer", "Age": 51, "Type of Travel": "Business travel",
        "Class": "Business", "Flight Distance": 900, "Departure Delay in Minutes": 15, "Arrival Delay in Minutes": 10,
        "ratings": [2, 3, 2, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 4]},
}


def key(col: str) -> str:
    return "p_" + col.replace(" ", "_").replace("/", "_")


def apply_preset(name: str):
    p = PRESETS[name]
    for col in C.CAT_COLS + [C.CLASS_COL] + C.NUM_COLS:
        st.session_state[key(col)] = p[col]
    for col, r in zip(C.RATING_COLS, p["ratings"]):
        st.session_state[key(col)] = r


def init_passenger():
    if key("Age") not in st.session_state:
        apply_preset("Borderline case")


def current_passenger() -> pd.DataFrame:
    init_passenger()
    return pd.DataFrame([{col: st.session_state[key(col)] for col in C.FEATURE_COLS}])


def rating_label(v: int) -> str:
    return "N/A" if v == 0 else "★" * v


# ---------- cards ----------
def flaps(word: str, good: bool) -> str:
    cls = "good" if good else "bad"
    tiles = "".join(f'<div class="flap {cls}" style="animation-delay:{i*45}ms">{"&nbsp;" if ch == " " else ch}</div>'
                    for i, ch in enumerate(word.upper()))
    return f'<div class="flaps">{tiles}</div>'


def model_card(key_: str, p: float, threshold: float = 0.5) -> str:
    info = MODEL_INFO[key_]
    good = p >= threshold
    kind = "base" if key_ in BASELINES else ("team" if key_ == "team" else "")
    color = GOOD if good else BAD
    return (f'<div class="card {kind}"><div class="role">{info["role"]}</div><div class="name">{info["name"]}</div>'
            f'<div class="pct" style="color:{color}">{p*100:.1f}%</div>'
            f'<div class="bar"><i style="width:{p*100:.1f}%;background:{color}"></i></div>'
            f'<span class="chip {"good" if good else "bad"}">{"SATISFIED" if good else "DISSATISFIED"}</span></div>')


def cards_html(probs: pd.Series, order=None) -> str:
    order = order or (["team"] + MEMBER_MODELS + BASELINES)
    return '<div class="cards">' + "".join(model_card(k, float(probs[k])) for k in order if k in probs) + "</div>"


def kpi(label: str, value: str, color: str = "#e9eef6", sub: str = "") -> str:
    """Stat tile. `sub` is an optional one-line plain-language explanation under the value."""
    s = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="l">{label}</div><div><div class="v" style="color:{color}" title="{value}">'
            f'{value}</div>{s}</div></div>')


def step_head(*cols: tuple[str, str]) -> str:
    """Column header row for a `.step` list: cols = (label, inline style) pairs."""
    return '<div class="step-head">' + "".join(f'<span style="{style}">{label}</span>' for label, style in cols) + "</div>"


# ---------- explanations ----------
def xgb_contributions(X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP values from XGBoost (log-odds units) for one row."""
    pipe = load_models()["xgb"]
    feats = pipe.named_steps["features"].transform(X)
    contrib = pipe.named_steps["model"].get_booster().predict(
        xgb.DMatrix(feats, feature_names=list(feats.columns)), pred_contribs=True)[0]
    df = pd.DataFrame({"feature": list(feats.columns), "value": feats.iloc[0].to_numpy(), "shap": contrib[:-1]})
    df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index)


# ---------- charts ----------
def themed(chart: alt.Chart) -> alt.Chart:
    """Dark theme + sizing. `height` set on a chart means the PLOT height: Streamlit's default
    autosize "fit" squeezes axis titles, chart titles and legends into that height, which squashed
    bars and hid labels. fit-x keeps the width responsive and lets the height grow for decorations."""
    enc = getattr(chart, "encoding", alt.Undefined)
    faceted = enc is not alt.Undefined and any(getattr(enc, ch, alt.Undefined) is not alt.Undefined
                                               for ch in ("row", "column", "facet"))
    if not faceted:
        chart = chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    return (chart.configure(background="transparent", font="Instrument Sans")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#aab7cb", titleColor=MUTED, gridColor="#1b2740", domainColor="#22314a",
                            labelFont="JetBrains Mono", titleFont="Instrument Sans", labelLimit=260,
                            labelFontSize=11, titleFontSize=12, titlePadding=10)
            .configure_axisBand(labelOverlap=False)
            .configure_axisX(labelAngle=0)
            .configure_title(color="#e9eef6", font="Instrument Sans", fontSize=14, fontWeight=600,
                             anchor="start", offset=12)
            .configure_header(labelColor="#e9eef6", titleColor=MUTED, labelFont="JetBrains Mono")
            .configure_legend(labelColor="#e9eef6", titleColor=MUTED, labelFont="Instrument Sans", labelFontSize=12,
                              labelLimit=320, symbolSize=110, columnPadding=18, rowPadding=6, padding=6))


def bottom_legend(columns: int | None = None, **kw) -> alt.Legend:
    """Untitled legend under the chart; `columns` wraps long legends into a grid instead of clipping them."""
    return alt.Legend(orient="bottom", title=None, **({"columns": columns} if columns else {}), **kw)


# Six display metrics -> six distinct colours (Accuracy, Recall, Precision, F1, ROC-AUC, MCC)
METRIC_COLORS = ["#ffb627", "#3ddc97", "#5ab0ff", "#c792ea", "#ff8c42", "#e9eef6"]


def color_scale(keys):
    return alt.Scale(domain=[MODEL_INFO[k]["name"] for k in keys], range=[MODEL_COLORS[k] for k in keys])


def mname(k: str) -> str:
    """Translated display name of a model key."""
    return MODEL_INFO.get(k, {"name": k})["name"]


def translate_condition(cond: str) -> str:
    """Translate one surrogate-rule condition, e.g. 'Online boarding ≥ 4' or 'Class is Eco / Eco Plus'."""
    import re
    if cond.startswith("Class is "):
        names = [C.CABIN_NAMES.get(c.strip(), c.strip()) for c in cond[len("Class is "):].split("/")]
        return "Class is" + " " + " / ".join(names)
    m = re.match(r"^(.*?) (≤|≥|>|=) (.*)$", cond)
    if m:
        subject, op, value = m.groups()
        value = value.replace(" mi", " " + "mi").replace(" min", " " + "min")
        return f"{subject} {op} {value}"
    return cond
