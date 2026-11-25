"""Error analysis on the hold-out set (spec Part A.4: "examine types of errors").

Everything here is computed from `test_predictions.parquet`, i.e. predictions on the 25,976
hold-out passengers that no model saw during training or tuning, so the error rates are honest.

Sections:
1. Confusion matrix of one model, with the business meaning of each error type.
2. Where the errors are: error rate per passenger segment versus the model's overall error rate.
3. Error overlap: do the four algorithms fail on the same passengers? (justifies combining models)
4. Confidently wrong passengers: the errors the model was surest about, the most dangerous ones.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui
from ui import C

THRESHOLD = 0.5  # every model predicts "satisfied" when P(satisfied) >= 0.5

# Segments a business user would act on. Delay buckets follow the EDA (most flights have 0 delay,
# long delays > 60 min are rare but hurt satisfaction most).
DELAY_BINS = [-1, 0, 15, 60, 10_000]
DELAY_LABELS = ["On time", "1-15 min", "16-60 min", "> 60 min"]
AGE_BINS = [0, 24, 39, 59, 120]
AGE_LABELS = ["<25", "25-39", "40-59", "60+"]


def _segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Passenger segments as string Series aligned with `df` (one entry per grouping)."""
    return {
        "Class": df[C.CLASS_COL].astype(str).map(lambda v: C.CABIN_NAMES.get(v, v)),
        "Type of Travel": df["Type of Travel"].astype(str),
        "Customer Type": df["Customer Type"].astype(str),
        "Age group": pd.cut(df["Age"], AGE_BINS, labels=AGE_LABELS).astype(str),
        "Departure delay": pd.cut(df["Departure Delay in Minutes"], DELAY_BINS, labels=DELAY_LABELS).astype(str),
        "Inflight wifi": df["Inflight wifi service"].map(ui.rating_label).astype(str),
    }


def confusion(y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """2x2 confusion matrix as a long table (actual, predicted, count, share of all passengers)."""
    rows = []
    names = {1: "Satisfied", 0: "Dissatisfied"}
    for a in (1, 0):
        for p in (1, 0):
            n = int(((y == a) & (pred == p)).sum())
            rows.append({"Actual": names[a], "Predicted": names[p], "count": n, "share": n / len(y),
                         "kind": "Correct" if a == p else ("False alarm" if a == 1 else "Missed unhappy")})
    return pd.DataFrame(rows)


def render():
    """Draw the Error analysis tab for the selected model."""
    preds = ui.load_test_predictions()
    y = preds["y_true"].to_numpy()
    st.markdown("Every number on this tab comes from the **hold-out set** (25,976 passengers never used for training "
                "or tuning). An error is either a **missed unhappy passenger** (predicted satisfied, actually "
                "dissatisfied: no service recovery is offered) or a **false alarm** (predicted dissatisfied, actually "
                "satisfied: a voucher is wasted).")
    model = st.selectbox("Model", ui.MEMBER_MODELS + ["xgb_mono", "rule"], format_func=ui.mname, key="err_model",
                         index=ui.MEMBER_MODELS.index(ui.HEADLINE))
    p = preds[model].to_numpy()
    pred = (p >= THRESHOLD).astype(int)
    wrong = pred != y
    fn = (pred == 1) & (y == 0)  # missed unhappy
    fp = (pred == 0) & (y == 1)  # false alarm

    k = st.columns(4)
    k[0].markdown(ui.kpi("Errors", f"{int(wrong.sum()):,}", ui.TEXT, f"{wrong.mean() * 100:.2f}% of passengers"),
                  unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Missed unhappy", f"{int(fn.sum()):,}", ui.BAD,
                         f"{fn.sum() / max((y == 0).sum(), 1) * 100:.1f}% of dissatisfied"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("False alarms", f"{int(fp.sum()):,}", ui.TEXT,
                         f"{fp.sum() / max((y == 1).sum(), 1) * 100:.1f}% of satisfied"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Error balance", f"{fn.sum() / max(fp.sum(), 1):.2f} : 1", ui.TEXT,
                         "missed unhappy per false alarm"), unsafe_allow_html=True)

    # ---- 1) confusion matrix ----
    st.markdown("#### Confusion matrix (rows = actual, columns = predicted)")
    cm = confusion(y, pred)
    order = ["Satisfied", "Dissatisfied"]
    base = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=order, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
                                y=alt.Y("Actual:N", sort=order, title=None, axis=alt.Axis(labelPadding=8)))
    # Same visual language as the Decision policy page: colour = what the cell means, label inside the cell.
    cells = base.mark_rect(cornerRadius=8, stroke=ui.BG, strokeWidth=4).encode(
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["Correct", "False alarm", "Missed unhappy"],
                                        range=["#1f2630", "#4a3a1c", "#4a2422"])),
        tooltip=["Actual", "Predicted", "kind", alt.Tooltip("count:Q", format=","),
                 alt.Tooltip("share:Q", format=".2%")])
    num = base.mark_text(fontSize=20, fontWeight=500, color=ui.TEXT, dy=-12, font="JetBrains Mono").encode(
        text=alt.Text("count:Q", format=","))
    kind = base.mark_text(fontSize=11, color=ui.TEXT2, dy=10).encode(text="kind:N")
    pct = base.mark_text(fontSize=11, color=ui.MUTED, dy=26, font="JetBrains Mono").encode(
        text=alt.Text("share:Q", format=".1%"))
    a, b = st.columns([1, 1], gap="large")
    a.altair_chart(ui.themed((cells + num + kind + pct).properties(height=260)), width="stretch")
    b.markdown(
        f"- **Precision** {((pred == 1) & (y == 1)).sum() / max((pred == 1).sum(), 1):.3f}: of passengers predicted satisfied, "
        f"this share really is.\n"
        f"- **Recall** {((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1):.3f}: of truly satisfied passengers, "
        f"this share is found.\n"
        f"- **Specificity** {((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1):.3f}: of truly dissatisfied "
        f"passengers, this share is caught for service recovery.\n\n"
        "Missed unhappy passengers are the costlier error for an airline (a churn risk with no recovery offered). "
        "The **Decision policy** page moves the threshold to trade one error type for the other.")

    # ---- 2) where the errors are ----
    st.markdown("#### Where the errors are")
    seg = _segments(preds)
    group = st.segmented_control("Group passengers by", list(seg), default="Class", key="err_group") or "Class"
    g = seg[group]
    rows = []
    for name in g.unique():
        m = (g == name).to_numpy()
        rows.append({"segment": name, "passengers": int(m.sum()), "error rate": wrong[m].mean(),
                     "missed unhappy": fn[m].mean(), "false alarm": fp[m].mean()})
    sdf = pd.DataFrame(rows).sort_values("error rate", ascending=False)
    overall = float(wrong.mean())
    sdf["vs overall"] = np.where(sdf["error rate"] > overall * 1.25, "Harder than average", "Typical")
    long = sdf.melt(["segment", "passengers", "vs overall"], ["missed unhappy", "false alarm"],
                    var_name="error type", value_name="rate")
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)).encode(
        y=alt.Y("segment:N", sort=list(sdf["segment"]), title=None),
        x=alt.X("rate:Q", stack=True, axis=alt.Axis(format="%"), title="Error rate (share of the segment)"),
        color=alt.Color("error type:N", scale=alt.Scale(domain=["missed unhappy", "false alarm"],
                                                         range=[ui.BAD, ui.ACCENT]), legend=ui.bottom_legend()),
        tooltip=["segment", "error type", alt.Tooltip("rate:Q", format=".2%"), alt.Tooltip("passengers:Q", format=",")])
    rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(strokeDash=[5, 4], color=ui.TEXT).encode(x="x:Q")
    st.altair_chart(ui.themed((bars + rule).properties(height=34 * len(sdf) + 20)), width="stretch")
    worst = sdf.iloc[0]
    st.caption(f"Dashed line = overall error rate {overall:.2%}. Hardest segment: **{worst['segment']}** "
               f"({worst['error rate']:.2%} errors, {worst['error rate'] / max(overall, 1e-9):.1f}x the average). "
               "Segments far above the line are candidates for more data, extra features or a segment-specific threshold.")

    # ---- 3) error overlap across the four algorithms ----
    st.markdown("#### Do the four algorithms make the same mistakes?")
    member_wrong = pd.DataFrame({k: (preds[k].to_numpy() >= THRESHOLD).astype(int) != y for k in ui.MEMBER_MODELS})
    n_wrong = member_wrong.sum(axis=1)
    ov = n_wrong.value_counts().reindex(range(5), fill_value=0).rename_axis("models wrong").reset_index(name="passengers")
    ov["label"] = ov["models wrong"].map({0: "All 4 right", 1: "1 of 4 wrong", 2: "2 of 4 wrong",
                                          3: "3 of 4 wrong", 4: "All 4 wrong"})
    errs = ov[ov["models wrong"] > 0]
    c1, c2 = st.columns([3, 2], gap="large")
    c1.altair_chart(ui.themed(alt.Chart(errs).mark_bar(cornerRadiusEnd=3, color=ui.BAD).encode(
        y=alt.Y("label:N", sort=list(errs["label"]), title=None),
        x=alt.X("passengers:Q", title="Passengers"),
        tooltip=["label", alt.Tooltip("passengers:Q", format=",")]).properties(height=180)), width="stretch")
    any_wrong = int((n_wrong > 0).sum())
    all_wrong = int((n_wrong == 4).sum())
    fixable = int((n_wrong == 1).sum())  # 3 right vs 1 wrong: a majority vote corrects it
    c2.markdown(ui.kpi("Outvoted by the other 3", f"{fixable:,}", ui.GOOD,
                       f"{fixable / max(any_wrong, 1) * 100:.0f}% of passengers some model gets wrong"),
                unsafe_allow_html=True)
    c2.markdown(ui.kpi("Wrong for every model", f"{all_wrong:,}", ui.BAD,
                       f"{all_wrong / len(y) * 100:.2f}% of passengers: likely label noise or missing information"),
                unsafe_allow_html=True)
    corr = member_wrong.astype(float).corr()
    pairs = [(ui.mname(a), ui.mname(b), corr.loc[a, b]) for i, a in enumerate(corr) for b in list(corr)[i + 1:]]
    lo_pair = min(pairs, key=lambda t: t[2])
    st.caption(f"When only one model is wrong, the other three outvote it; that is the reason for "
               f"combining different algorithms. Least correlated errors: **{lo_pair[0]}** and **{lo_pair[1]}** "
               f"(error correlation {lo_pair[2]:.2f}), the most complementary pair.")

    # ---- 4) confidently wrong passengers ----
    st.markdown("#### Confidently wrong passengers")
    conf = np.abs(p - 0.5) * 2  # 0 = coin flip, 1 = fully certain
    err_df = preds.loc[wrong, C.FEATURE_COLS].copy()
    err_df.insert(0, "P(satisfied)", p[wrong])
    err_df.insert(1, "confidence", conf[wrong])
    err_df.insert(2, "actual", np.where(y[wrong] == 1, "satisfied", "dissatisfied"))
    err_df[C.CLASS_COL] = err_df[C.CLASS_COL].astype(str).map(lambda v: C.CABIN_NAMES.get(v, v))
    for c in C.CAT_COLS:
        err_df[c] = err_df[c].astype(str)
    hist = pd.DataFrame({"confidence": conf, "result": np.where(wrong, "Wrong", "Right")})
    st.altair_chart(ui.themed(alt.Chart(hist).mark_bar(opacity=.85).encode(
        x=alt.X("confidence:Q", bin=alt.Bin(maxbins=20), title="Model confidence |P - 0.5| x 2"),
        y=alt.Y("count():Q", stack=None, title="Passengers", scale=alt.Scale(type="symlog")),
        color=alt.Color("result:N", scale=alt.Scale(domain=["Right", "Wrong"], range=[ui.GOOD, ui.BAD]),
                        legend=ui.bottom_legend())).properties(height=220)), width="stretch")
    high = float((conf[wrong] > 0.8).mean())
    st.caption(f"Errors cluster at low confidence: only {high:.0%} of this model's errors are made with confidence "
               "above 0.8. Low-confidence passengers are exactly the ones the conformal sets on the Predict page "
               "flag for human review (log scale on the y-axis).")
    st.dataframe(err_df.sort_values("confidence", ascending=False).head(200), hide_index=True, width="stretch",
                 height=320, column_config={
                     "P(satisfied)": st.column_config.ProgressColumn("P(satisfied)", min_value=0, max_value=1,
                                                                     format="%.2f"),
                     "confidence": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Top 200 errors, most confident first. Many have contradictory answers (e.g. high service ratings but "
               "reported dissatisfied), a sign of survey label noise that no model can learn.")
