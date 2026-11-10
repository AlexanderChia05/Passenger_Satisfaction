import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score

import ui
from ui import C
from src.trust import conformal_qhat, conformal_sets

ui.header("GATE 05 · TRUST", "Trust & fairness",
          "Can we rely on the models? Statistically guaranteed uncertainty, fairness across passenger groups, "
            "sensible behaviour under constraints, and accuracy before the survey is complete.")

test = ui.load_test_predictions()
y = test["y_true"].to_numpy()
trust = ui.load_trust()
cal = trust["cal"]
keys = ["team"] + ui.MEMBER_MODELS

tab_conf, tab_fair, tab_mono, tab_stage = st.tabs(
    ["Conformal uncertainty", "Fairness audit", "Monotonic constraints", "Early-warning accuracy"])

with tab_conf:
    st.markdown("Split conformal prediction calibrated on held-back training passengers. At level 1-α the "
                  "prediction **set** should contain the true outcome at least (1-α) of the time.")
    levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    rows = []
    for k in keys:
        for lv in levels:
            q = conformal_qhat(cal[k].to_numpy(), cal["y_true"].to_numpy(), 1 - lv)
            s, d = conformal_sets(test[k].to_numpy(), q)
            covered = np.where(y == 1, s, d)
            rows.append({"model": ui.mname(k), "target": lv, "coverage": covered.mean(),
                         # not exactly one label: both labels OR empty set (Predict flags both as uncertain)
                         "uncertain share": (s == d).mean(),"avg set size": (s.astype(int) + d.astype(int)).mean()})
    cdf = pd.DataFrame(rows)
    a, b = st.columns(2)
    dom = alt.Scale(domain=[0.8, 1])
    diag = alt.Chart(pd.DataFrame({"t": [0.8, 1]})).mark_line(strokeDash=[4, 4], color=ui.MUTED).encode(
        x=alt.X("t:Q", scale=dom), y=alt.Y("t:Q", scale=dom))
    cov = alt.Chart(cdf).mark_line(point=True).encode(
        x=alt.X("target:Q", title="Target coverage", scale=alt.Scale(domain=[0.8, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("coverage:Q", title="Achieved coverage (hold-out)", scale=alt.Scale(domain=[0.8, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=ui.color_scale(keys), legend=ui.bottom_legend(3)),
        tooltip=["model", alt.Tooltip("target:Q", format=".0%"), alt.Tooltip("coverage:Q", format=".2%")])
    a.altair_chart(ui.themed((diag + cov).properties(height=280, title="Coverage is guaranteed")), width="stretch")
    unc = alt.Chart(cdf).mark_line(point=True).encode(
        x=alt.X("target:Q", title="Target coverage", scale=alt.Scale(domain=[0.8, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("uncertain share:Q", title="Passengers flagged 'uncertain'", axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=ui.color_scale(keys), legend=ui.bottom_legend(3)))
    b.altair_chart(ui.themed(unc.properties(height=280, title="Price of certainty: more flagged passengers")),
                   width="stretch")
    st.caption("A sharper model flags fewer passengers as uncertain at the same guaranteed coverage.")

with tab_fair:
    c1, c2, c3 = st.columns(3)
    attr = c1.selectbox("Group by", ["Gender", "Age group", "Class", "Type of Travel", "Customer Type"])
    model = c2.selectbox("Model", keys, format_func=ui.mname)
    tol = c3.slider("Flag gap above (pts)", 1, 10, 3)
    groups = (pd.cut(test["Age"], [0, 24, 39, 59, 100], labels=["<25", "25-39", "40-59", "60+"]).astype(str)
              if attr == "Age group" else test[attr].astype(str))
    pred = (test[model] >= 0.5).astype(int)
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        yt, yp = y[m], pred[m].to_numpy()
        rows.append({"group": C.CABIN_NAMES.get(g, g), "passengers": int(m.sum()), "accuracy": (yt == yp).mean(),
                     "F1": f1_score(yt, yp, zero_division=0),
                     "false alarm rate": ((yp == 0) & (yt == 1)).sum() / max((yt == 1).sum(), 1),
                     "missed unhappy rate": ((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1),
                     "actual satisfied": yt.mean(), "predicted satisfied": yp.mean()})
    fair = pd.DataFrame(rows)
    gap = (fair["accuracy"].max() - fair["accuracy"].min()) * 100
    k = st.columns(3)
    k[0].markdown(ui.kpi("Accuracy gap", "{g:.1f} pts".format(g=gap), ui.BAD if gap > tol else ui.GOOD), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Worst group", fair.loc[fair["accuracy"].idxmin(), "group"]), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Missed-unhappy gap",
                         "{g:.1f} pts".format(g=(fair['missed unhappy rate'].max() - fair['missed unhappy rate'].min()) * 100)),
                  unsafe_allow_html=True)
    if gap > tol:
        ui.note(("<b>Fairness flag:</b> accuracy differs by {g:.1f} pts across {a} groups (limit {tol}). "
                  "Investigate with group-specific thresholds or reweighting in Part B.").format(g=gap, a=attr, tol=tol),
                warn=True)
    st.write("")
    long = fair.melt("group", ["accuracy", "false alarm rate", "missed unhappy rate"], var_name="metric")
    n_groups = fair["group"].nunique()
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)).encode(
        x=alt.X("value:Q", axis=alt.Axis(format="%"), title=None), y=alt.Y("group:N", title=None),
        color=alt.Color("group:N", legend=None,
                        scale=alt.Scale(range=["#5ab0ff", ui.ACCENT, "#c792ea", ui.GOOD, "#ff8c42"])),
        column=alt.Column("metric:N", title=None, sort=["accuracy", "false alarm rate", "missed unhappy rate"],
                          header=alt.Header(labelColor="#e9eef6", labelFont="Instrument Sans", labelFontSize=13,
                                            labelFontWeight=600, labelAnchor="start", labelAlign="left")),
        tooltip=["group", "metric", alt.Tooltip("value:Q", format=".2%")]).properties(height=28 * n_groups + 10, width=250).resolve_scale(x="independent")
    # fixed-width facets (3 x 250px) scroll inside their own box on narrow screens instead of the page
    st.container(key="fair_chart").altair_chart(ui.themed(bars), width="content")
    st.dataframe(fair.style.format({c: "{:.2%}" for c in fair.columns if c not in ("group", "passengers")}),
                 hide_index=True, width="stretch")

with tab_mono:
    mono = trust["mono"]
    m = ui.load_metrics()["models"]
    st.markdown("**Rule:** a better service rating should never make a passenger *less* likely to be satisfied. "
                  "We test it by adding +1 star to each service for 1,500 hold-out passengers.")
    mdf = pd.DataFrame([{"model": ui.mname(k), "key": k, "violations": v, "accuracy": m.get(k, {}).get("Accuracy", np.nan)}
                        for k, v in mono.items()])
    a, b = st.columns(2)
    vbase = alt.Chart(mdf).encode(y=alt.Y("model:N", sort="-x", title=None))
    vbars = vbase.mark_bar(cornerRadiusEnd=3, height=18).encode(
        x=alt.X("violations:Q", axis=alt.Axis(format="%"), title="+1 star lowered P(satisfied) by >0.5 pts"),
        color=alt.condition("datum.violations == 0", alt.value(ui.GOOD), alt.value(ui.BAD)))
    vtext = vbase.mark_text(align="left", dx=5, font="JetBrains Mono", fontSize=11).encode(
        x="violations:Q", text=alt.Text("violations:Q", format=".1%"),
        color=alt.condition("datum.violations == 0", alt.value(ui.GOOD), alt.value("#e9eef6")))
    a.altair_chart(ui.themed((vbars + vtext).properties(height=260, title="Monotonicity violations")), width="stretch")
    lo = float(np.nanmin(mdf["accuracy"])) - 0.01 if mdf["accuracy"].notna().any() else 0.9
    acc = mdf.dropna(subset=["accuracy"]).assign(base=lo)  # bars start at the axis minimum, not at 0
    b.altair_chart(ui.themed(alt.Chart(acc).mark_bar(cornerRadiusEnd=3, height=18, color=ui.ACCENT).encode(
        x=alt.X("accuracy:Q", scale=alt.Scale(domain=[lo, 1]), axis=alt.Axis(format="%"), title="Hold-out accuracy"),
        x2="base:Q", y=alt.Y("model:N", sort="-x", title=None),
        tooltip=["model", alt.Tooltip("accuracy:Q", format=".2%")]).properties(
        height=260, title="Accuracy cost of the constraint")), width="stretch")
    if "xgb_mono" in m:
        ui.note(("Monotonic XGBoost removes every violation for a {d:.1f}-pt accuracy cost - a trade-off between raw "
                  "accuracy and trustworthy what-if advice.").format(d=(m['xgb']['Accuracy'] - m['xgb_mono']['Accuracy']) * 100))

with tab_stage:
    sm = trust["stage_metrics"]
    st.markdown("Each stage uses a model trained only on what is known by then (unknown ratings hidden).")
    k = st.columns(len(sm))
    for i, r in sm.iterrows():
        k[i].markdown(ui.kpi("{s} · {n} ratings".format(s=r["stage"], n=r["known ratings"]), f"{r['Accuracy']*100:.1f}%",
                             ui.GOOD if r["stage"] == "Landed" else ui.ACCENT), unsafe_allow_html=True)
    st.write("")
    long = sm.melt("stage", [m for m in ui.METRICS if m in sm.columns])
    st.altair_chart(ui.themed(alt.Chart(long).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("stage:N", sort=[s for s in C.STAGE_ORDER], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=None),
        color=alt.Color("variable:N", sort=[m for m in ui.METRICS],
                        scale=alt.Scale(domain=[m for m in ui.METRICS], range=ui.METRIC_COLORS),
                        legend=ui.bottom_legend())
    ).properties(height=300)), width="stretch")
