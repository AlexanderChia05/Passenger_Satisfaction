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
from src.features import validate_frame  # noqa: E402
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

GOOD, BAD, ACCENT, MUTED = "#4cc38a", "#e8675d", "#f0b43c", "#6f7987"
TEXT, TEXT2, BG = "#e4e8ee", "#a7b0bd", "#0e1218"
MODEL_COLORS = {"knn": "#5ab0ff", "svm": "#c792ea", "mlp": "#3ddc97", "xgb": "#f0b43c",
                "rule": "#8b95a3", "dummy": "#4a5a73", "xgb_mono": "#ff8c42"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
/* Night-shift operations room: graphite neutrals with a faint blue cast, one runway-amber accent.
   Spacing on a 4px grid; radius 8 for controls, 12 for cards. */
:root { --bg:#0e1218; --panel:#151a22; --raised:#1b212b; --line:rgba(214,222,235,.09); --line-strong:rgba(214,222,235,.16);
        --text:#e4e8ee; --text-2:#a7b0bd; --muted:#6f7987;
        --accent:#f0b43c; --good:#4cc38a; --bad:#e8675d; }
html, body, .stMarkdown, p, label, input, textarea, button, [data-baseweb="select"] div { font-family:'Instrument Sans',sans-serif; }
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
h1, h2, h3, h4 { font-family:'Instrument Sans',sans-serif !important; font-weight:500 !important; letter-spacing:0; text-transform:none; }
.stApp { background:var(--bg); }
[data-testid="stSidebar"] { background:#11161d; border-right:1px solid var(--line); }
.mono { font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; }

/* Page header */
.board { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap;
         border-bottom:1px solid var(--line); padding-bottom:16px; margin-bottom:24px; }
.board h1 { margin:0; padding:0; font-size:1.9rem; line-height:1.15; color:var(--text); }
.board .sub { color:var(--text-2); font-size:.92rem; line-height:1.5; max-width:52ch; }
.brand { padding:4px 0 0; }
.brand .name { font-size:1.05rem; font-weight:500; color:var(--text); }
.brand .meta { font-family:'JetBrains Mono',monospace; font-size:.72rem; color:var(--muted); margin-top:4px; line-height:1.5; }

/* Split-flap verdict: the one signature detail, used only for the verdict */
.flaps { display:flex; gap:4px; flex-wrap:wrap; margin:4px 0 12px; }
.flap { width:2rem; height:2.8rem; display:grid; place-items:center; border-radius:4px;
        font-family:'JetBrains Mono',monospace; font-weight:500; font-size:1.5rem;
        background:linear-gradient(#1d232d 49%, #161b23 50%); box-shadow:inset 0 0 0 1px var(--line-strong);
        animation:flip .32s cubic-bezier(.2,.7,.2,1) both; }
.flap.good { color:var(--good); } .flap.bad { color:var(--bad); }
@keyframes flip { from { transform:rotateX(70deg); opacity:0 } to { transform:none; opacity:1 } }

/* Model cards */
.cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(168px,1fr)); gap:12px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
.card.base { background:transparent; border-style:dashed; }
.card.lead { border-color:rgba(240,180,60,.55); }
.card .role { font-size:.75rem; color:var(--muted); }
.card .name { font-size:1rem; font-weight:500; color:var(--text); margin-top:2px; }
.card .pct { font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; font-size:1.6rem; font-weight:500;
             line-height:1.2; margin-top:8px; }
.card .bar { height:4px; border-radius:2px; background:var(--raised); margin:8px 0; overflow:hidden; }
.card .bar > i { display:block; height:100%; border-radius:2px; }
.chip { font-family:'JetBrains Mono',monospace; font-size:.75rem; color:var(--text-2); }
.chip.good { color:var(--good); } .chip.bad { color:var(--bad); }

.note { border:1px solid var(--line); border-left:2px solid var(--muted); background:var(--panel); padding:12px 16px;
        border-radius:8px; color:var(--text-2); font-size:.9rem; line-height:1.5; margin:4px 0 16px; }
.note b { color:var(--text); font-weight:500; }
.note.warn { border-left-color:var(--bad); }
.step { display:flex; gap:12px; align-items:center; padding:8px 0; border-bottom:1px solid var(--line); }
.step .n { font-family:'JetBrains Mono',monospace; color:var(--muted); width:1.6rem; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px;
       min-height:5.5rem; height:100%; display:flex; flex-direction:column; justify-content:space-between; gap:8px; min-width:0; }
.kpi .l { font-size:.8rem; color:var(--text-2); line-height:1.35; }
.kpi .v { font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; font-size:clamp(1.05rem, 1.5vw, 1.5rem);
          font-weight:500; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .s { font-size:.76rem; color:var(--muted); margin-top:4px; line-height:1.35; }

/* Tabs: quiet underline, accent only on the selected label */
[data-baseweb="tab-list"] { gap:16px; border-bottom:1px solid var(--line); }
[data-baseweb="tab"] { padding:8px 0 !important; background:transparent !important; }
[data-baseweb="tab"] p { font-size:.95rem !important; font-weight:500; color:var(--text-2); }
[data-baseweb="tab"][aria-selected="true"] p { color:var(--text); }
[data-baseweb="tab-highlight"] { background-color:var(--accent) !important; height:2px !important; }

/* Multiselect tags: neutral, not accent-filled */
[data-baseweb="tag"] { background:var(--raised) !important; border:1px solid var(--line-strong) !important; border-radius:6px !important; }
[data-baseweb="tag"] span { color:var(--text) !important; }

/* Section headings */
h4 { font-size:1.05rem !important; margin-top:8px !important; color:var(--text); }
h3 { font-size:1.2rem !important; }
.step-head { display:flex; gap:12px; padding:4px 0 8px; border-bottom:1px solid var(--line);
             font-size:.78rem; color:var(--muted); }
.step .n, .step-head .n { min-width:2.4rem; white-space:nowrap; }

/* Plain HTML tables (overview): wrap text instead of clipping */
.table-wrap table { width:100%; border-collapse:collapse; font-size:.88rem; }
.table-wrap th { text-align:left; font-weight:500; color:var(--text-2); padding:8px 12px; border-bottom:1px solid var(--line-strong); }
.table-wrap td { padding:8px 12px; border-bottom:1px solid var(--line); color:var(--text); vertical-align:top; line-height:1.45; }
.table-wrap td.num { text-align:right; font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; }
.table-wrap td.where { color:var(--text-2); }

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
  .board h1 { font-size:1.6rem; }
  .flap { width:1.6rem; height:2.3rem; font-size:1.2rem; }
  .cards { grid-template-columns:repeat(auto-fill,minmax(144px,1fr)); }
}
</style>
"""


def setup_page():
    """Inject the shared CSS theme into the page."""
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, sub: str):
    """Page header: title on the left, one-sentence purpose on the right."""
    st.markdown(f'<div class="board"><h1>{title}</h1><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False):
    """Highlighted note box; warn=True uses the warning colour."""
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


# ---------- artifacts ----------
def artifact_failure(what: str, exc: Exception):
    """Stop the page with a readable message when a saved artifact is missing or unreadable.

    The usual causes are an incomplete `git clone` / pull, or library versions that differ from the
    ones that pickled the models (scikit-learn / xgboost), so the message names both fixes.
    """
    st.error(f"Could not load {what} from `{ART}`: {type(exc).__name__}: {exc}\n\n"
             "Fix: re-run `git pull` (the model files are large), check `pip install -r requirements.txt`, "
             "or retrain with `python train_full.py` / `python prototype_train.py`.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_models():
    """All fitted models (members, baselines) as {key: sklearn Pipeline}."""
    import joblib
    path = ART / "models.joblib" if (ART / "models.joblib").exists() else ART / "prototype_models.joblib"
    if not path.exists():
        st.error("No trained models found. Run `python prototype_train.py` first.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as exc:  # corrupt / truncated pickle or incompatible library version
        artifact_failure("the trained models", exc)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Hold-out probabilities of every model plus the raw test features (one row per passenger)."""
    try:
        return pd.read_parquet(ART / "test_predictions.parquet")
    except Exception as exc:
        artifact_failure("the hold-out predictions", exc)


@st.cache_data
def load_oof() -> pd.DataFrame | None:
    """Out-of-fold P(satisfied) of every member and the rule baseline on the 103,904 training passengers
    (5-fold CV), plus y_true. Used to CHOOSE settings such as decision thresholds, so the hold-out set
    is only used to report the result. None when the full experiments were not run."""
    try:
        oof = pd.read_parquet(EXP_DIR / "oof_final.parquet")
        base = pd.read_parquet(EXP_DIR / "oof_baselines.parquet")
    except Exception:  # prototype artifacts have no OOF predictions
        return None
    rule = [c for c in base.columns if c.startswith("Rule")]
    if rule:
        oof["rule"] = base[rule[0]].to_numpy()
    return oof


@st.cache_data
def load_metrics() -> dict:
    """Hold-out metrics and run metadata written by the training pipeline."""
    try:
        return json.loads((ART / "metrics.json").read_text())
    except Exception as exc:  # missing file or invalid JSON
        artifact_failure("the hold-out metrics", exc)


@st.cache_resource(show_spinner=False)
def load_variants():
    """Advanced / stage / monotone model variants (merged from every variants*.joblib file)."""
    from src.store import load_variants as _load
    try:
        return _load(ART)
    except Exception as exc:
        artifact_failure("the model variants", exc)


@st.cache_resource(show_spinner=False)
def load_trust():
    """Trust artifacts: conformal calibration scores, OOD detector, surrogate rules, monotonic checks."""
    import joblib
    try:
        return joblib.load(ART / "trust.joblib")
    except Exception as exc:
        artifact_failure("the trust & fairness artifacts", exc)


HEADLINE = "xgb"  # the model behind every single-number answer (Predict, Simulator, Planner, Trust)


def predict(X: pd.DataFrame) -> pd.DataFrame:
    """P(satisfied) from every member model and baseline (no team ensemble)."""
    models = {k: m for k, m in load_models().items() if k != "team"}
    return predict_all(models, X).drop(columns="team", errors="ignore")


def p_headline(X: pd.DataFrame) -> np.ndarray:
    """P(satisfied) from the headline model only (fast path for searches)."""
    return load_models()[HEADLINE].predict_proba(X)[:, 1]


def predict_stage(X: pd.DataFrame, stage: str) -> float:
    """P(satisfied) from the stage-specific model (Landed = the full XGBoost)."""
    if stage == "Landed":
        return float(load_models()[HEADLINE].predict_proba(X)[0, 1])
    return float(load_variants()[f"stage::{stage}"].predict_proba(X)[0, 1])


def conformal_label(p: float, level: float, model: str = HEADLINE) -> tuple[str, bool]:
    """Conformal prediction set for one passenger -> (text, is_certain)."""
    from src.trust import conformal_qhat, conformal_sets
    cal = load_trust()["cal"]
    qhat = conformal_qhat(cal[model].to_numpy(), cal["y_true"].to_numpy(), 1 - level)
    has_s, has_d = conformal_sets(np.array([p]), qhat)
    labels = [n for n, h in [("Satisfied", has_s[0]), ("Dissatisfied", has_d[0])] if h]
    return "{" + ", ".join(labels) + "}" if labels else "{ } " + "(empty)", len(labels) == 1


def ood_percentile(X: pd.DataFrame) -> float:
    """How unusual a passenger is versus the calibration passengers (percentile)."""
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
    """Predicted satisfaction rate per cabin class on the hold-out set (3D cabin view)."""
    df = load_test_predictions()
    return (df[HEADLINE] >= 0.5).groupby(df["Class"].astype(str)).mean().to_dict()


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
    """Session-state key of a passenger field."""
    return "p_" + col.replace(" ", "_").replace("/", "_")


def apply_preset(name: str):
    """Load a preset passenger into the input widgets."""
    p = PRESETS[name]
    for col in C.CAT_COLS + [C.CLASS_COL] + C.NUM_COLS:
        st.session_state[key(col)] = p[col]
    for col, r in zip(C.RATING_COLS, p["ratings"]):
        st.session_state[key(col)] = r


def init_passenger():
    """Load the default preset on first visit."""
    if key("Age") not in st.session_state:
        apply_preset("Borderline case")


def current_passenger() -> pd.DataFrame:
    """The passenger being edited, as a one-row frame. Falls back to a preset if the stored values are
    invalid (e.g. a stale browser session), so no page ever predicts on malformed input."""
    init_passenger()
    try:
        X = pd.DataFrame([{col: st.session_state[key(col)] for col in C.FEATURE_COLS}])
        problems = validate_frame(X)
    except KeyError as exc:
        problems = [f"missing value {exc}"]
    if problems:
        st.warning("The stored passenger was invalid ({p}); a preset passenger was loaded instead."
                   .format(p="; ".join(problems)))
        apply_preset("Borderline case")
        X = pd.DataFrame([{col: st.session_state[key(col)] for col in C.FEATURE_COLS}])
    return X


def rating_label(v: int) -> str:
    """Star string for a rating; 0 shows as N/A."""
    return "N/A" if v == 0 else "★" * v


# ---------- cards ----------
def flaps(word: str, good: bool) -> str:
    """Split-flap board animation for the verdict word."""
    cls = "good" if good else "bad"
    tiles = "".join(f'<div class="flap {cls}" style="animation-delay:{i*45}ms">{"&nbsp;" if ch == " " else ch}</div>'
                    for i, ch in enumerate(word.upper()))
    return f'<div class="flaps">{tiles}</div>'


def model_card(key_: str, p: float, threshold: float = 0.5) -> str:
    """HTML card with one model's probability and verdict."""
    info = MODEL_INFO[key_]
    good = p >= threshold
    kind = "base" if key_ in BASELINES else ("lead" if key_ == HEADLINE else "")
    color = GOOD if good else BAD
    return (f'<div class="card {kind}"><div class="role">{info["role"]}</div><div class="name">{info["name"]}</div>'
            f'<div class="pct" style="color:{color}">{p*100:.1f}%</div>'
            f'<div class="bar"><i style="width:{p*100:.1f}%;background:{color}"></i></div>'
            f'<span class="chip {"good" if good else "bad"}">{"Satisfied" if good else "Dissatisfied"}</span></div>')


def cards_html(probs: pd.Series, order=None) -> str:
    """Row of model cards, headline model first."""
    order = order or ([HEADLINE] + [k for k in MEMBER_MODELS if k != HEADLINE] + BASELINES)
    return '<div class="cards">' + "".join(model_card(k, float(probs[k])) for k in order if k in probs) + "</div>"


def kpi(label: str, value: str, color: str = TEXT, sub: str = "") -> str:
    """Stat tile. `sub` is an optional one-line plain-language explanation under the value."""
    s = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div class="kpi"><div class="l">{label}</div><div><div class="v" style="color:{color}" title="{value}">'
            f'{value}</div>{s}</div></div>')


def html_table(df: pd.DataFrame, num=(), muted=(), fmt=None) -> str:
    """Plain HTML table whose cells wrap (st.dataframe clips long text). `num` columns align right in mono,
    `muted` columns use the secondary text colour, `fmt` maps a column to a format string."""
    import html as _h
    fmt = fmt or {}
    head = "".join(f"<th{' style=\"text-align:right\"' if c in num else ''}>{_h.escape(str(c))}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        cells = ""
        for c in df.columns:
            v = r[c]
            txt = fmt[c].format(v) if c in fmt and pd.notna(v) else ("-" if pd.isna(v) else str(v))
            cls = "num" if c in num else ("where" if c in muted else "")
            cells += f'<td class="{cls}">{_h.escape(txt)}</td>'
        body += f"<tr>{cells}</tr>"
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


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
            .configure_axis(labelColor=TEXT2, titleColor=TEXT2, gridColor="#1c222b", domainColor="#2a313c",
                            labelFont="JetBrains Mono", titleFont="Instrument Sans", labelLimit=260,
                            labelFontSize=11, titleFontSize=12, titlePadding=10)
            .configure_axisBand(labelOverlap=False)
            .configure_axisX(labelAngle=0)
            .configure_title(color=TEXT, font="Instrument Sans", fontSize=14, fontWeight=500,
                             anchor="start", offset=12)
            .configure_header(labelColor=TEXT, titleColor=TEXT2, labelFont="Instrument Sans")
            .configure_legend(labelColor=TEXT2, titleColor=TEXT2, labelFont="Instrument Sans", labelFontSize=12,
                              labelLimit=320, symbolSize=110, columnPadding=18, rowPadding=6, padding=6))


def bottom_legend(columns: int | None = None, **kw) -> alt.Legend:
    """Untitled legend under the chart; `columns` wraps long legends into a grid instead of clipping them."""
    return alt.Legend(orient="bottom", title=None, **({"columns": columns} if columns else {}), **kw)


# Six display metrics -> six distinct colours (Accuracy, Recall, Precision, F1, ROC-AUC, MCC)
METRIC_COLORS = ["#f0b43c", "#3ddc97", "#5ab0ff", "#c792ea", "#ff8c42", "#e4e8ee"]


def color_scale(keys):
    """Altair colour scale that gives every model its fixed colour."""
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
