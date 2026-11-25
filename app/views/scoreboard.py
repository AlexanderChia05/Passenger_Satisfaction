import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import error_analysis
import ui
from ui import C

meta = ui.load_metrics()
ui.header("Model scoreboard",
          "Hold-out performance of every model versus the two baselines, error analysis, head-to-head model duels, "
          "and the feedback log."
          + " " + ("Tuned models trained on the full training set." if meta.get("mode") == "full"
                   else "Prototype numbers: untuned models trained on a 20k-row subsample."))

m = meta["models"]
preds = ui.load_test_predictions()
y = preds["y_true"].to_numpy()
tab_over, tab_err, tab_duel, tab_fb = st.tabs(["Overview", "Error analysis", "Model duel", "Feedback log"])

with tab_over:
    order = ui.MEMBER_MODELS + ["xgb_mono"] + ui.BASELINES
    metric_cols = ui.METRICS
    rows = [{"Model": ui.mname(k), "Owner": ui.MODEL_INFO[k]["role"],
             **{c: m[k].get(c) for c in metric_cols}, "Fit (s)": m[k].get("fit_seconds")}
            for k in order if k in m]
    raw = pd.DataFrame(rows)
    # pre-formatted text: the Streamlit grid shows nulls as "None" whatever na_rep says (e.g. models without a recorded fit time)
    raw["Fit (s)"] = pd.to_numeric(raw["Fit (s)"], errors="coerce").map(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    table = raw

    best = max(ui.MEMBER_MODELS, key=lambda k: m[k]["F1"])
    k = st.columns(4)
    k[0].markdown(ui.kpi("Best member model (F1)", ui.mname(best), ui.ACCENT), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Best accuracy", f"{m[best]['Accuracy']*100:.2f}%"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Accuracy gain vs dummy", "+{d:.1f} pts".format(d=(m[best]['Accuracy'] - m['dummy']['Accuracy']) * 100)),
                  unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Accuracy gain vs rule", "+{d:.1f} pts".format(d=(m[best]['Accuracy'] - m['rule']['Accuracy']) * 100)),
                  unsafe_allow_html=True)
    st.write("")

    tm = [c for c in metric_cols]
    st.dataframe(table.style.format({c: "{:.4f}" for c in tm}, na_rep="-")
                 .highlight_max(subset=tm, props="color:#f0b43c; font-weight:500"),
                 hide_index=True, width="stretch")
    st.caption("Recall: share of satisfied passengers found. Precision: share of predicted-satisfied that truly are. MCC: robust to class imbalance.")

    curve_keys = ui.MEMBER_MODELS + ["rule"]
    roc_rows, pr_rows = [], []
    for key in curve_keys:
        fpr, tpr, _ = roc_curve(y, preds[key])
        idx = slice(None, None, max(1, len(fpr) // 300))  # thin the curve for plotting speed
        roc_rows.append(pd.DataFrame({"x": fpr[idx], "y": tpr[idx], "model": ui.mname(key)}))
        prec, rec, _ = precision_recall_curve(y, preds[key])
        idx = slice(None, None, max(1, len(prec) // 300))
        pr_rows.append(pd.DataFrame({"x": rec[idx], "y": prec[idx], "model": ui.mname(key)}))

    left, right = st.columns(2, gap="large")
    # Every member sits in the top-left corner on a 0-1 axis; zoom there so the curves can be told apart.
    zoom = [(left, roc_rows, "ROC curve, zoomed to the top-left corner", "False positive rate", "True positive rate",
             [0, 0.3], [0.7, 1]),
            (right, pr_rows, "Precision-recall curve, zoomed to the top-right corner", "Recall", "Precision",
             [0.7, 1], [0.7, 1])]
    for col, data, title, xt, yt, xd, yd in zoom:
        with col:
            st.markdown(f"#### {title}")
            ch = alt.Chart(pd.concat(data)).mark_line(strokeWidth=2, clip=True).encode(
                x=alt.X("x:Q", title=xt, scale=alt.Scale(domain=xd)), y=alt.Y("y:Q", title=yt, scale=alt.Scale(domain=yd)),
                color=alt.Color("model:N", scale=ui.color_scale(curve_keys), legend=ui.bottom_legend(3)))
            st.altair_chart(ui.themed(ch.properties(height=300)), width="stretch")
    st.caption("ROC-AUC: " + " · ".join(f"{ui.mname(k)} {m[k]['ROC-AUC']:.4f}" for k in curve_keys if k in m)
               + ". The rule baseline sits far below the members; only its edge enters this window.")

with tab_err:
    error_analysis.render()

with tab_duel:
    duel_keys = [k for k in ui.MEMBER_MODELS + ["xgb_mono", "rule"] if k in preds]
    a, b = st.columns(2)
    m1 = a.selectbox("Model A", duel_keys, index=0, format_func=ui.mname)
    m2 = b.selectbox("Model B", duel_keys, index=3, format_func=ui.mname)
    pa, pb = (preds[m1] >= .5).astype(int).to_numpy(), (preds[m2] >= .5).astype(int).to_numpy()
    ra, rb = pa == y, pb == y
    only_a, only_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
    k = st.columns(4)
    k[0].markdown(ui.kpi("Disagree on", "{n:,} pax".format(n=int((pa != pb).sum())), ui.ACCENT), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Only {m} right".format(m=ui.mname(m1)), f"{only_a:,}", ui.MODEL_COLORS.get(m1, ui.ACCENT)),
                  unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Only {m} right".format(m=ui.mname(m2)), f"{only_b:,}", ui.MODEL_COLORS.get(m2, ui.ACCENT)),
                  unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Both wrong", f"{int((~ra & ~rb).sum()):,}", ui.BAD), unsafe_allow_html=True)
    # McNemar test (continuity-corrected) on the discordant pairs
    if only_a + only_b:
        from scipy.stats import chi2
        stat = (abs(only_a - only_b) - 1) ** 2 / (only_a + only_b)
        pval = float(chi2.sf(stat, 1))
        winner = ui.mname(m1 if only_a > only_b else m2)
        ui.note(f"McNemar χ² = {stat:.1f}, p = {pval:.2g}: " + ("<b>{w}</b> is significantly better (p < 0.05).".format(w=winner)
                if pval < 0.05 else "no significant difference between the two models."))
    st.markdown("#### Passengers they disagree on")
    # same model on both sides: keep one probability column (duplicate names crash st.dataframe)
    dis = preds.loc[pa != pb, C.FEATURE_COLS + list(dict.fromkeys([m1, m2])) + ["y_true"]].copy()
    dis["actual"] = dis.pop("y_true").map({1: "satisfied", 0: "dissatisfied"})
    for c in C.CAT_COLS:  # category values follow the app language like everywhere else
        dis[c] = dis[c].astype(str)
    dis[C.CLASS_COL] = dis[C.CLASS_COL].astype(str).map(lambda v: C.CABIN_NAMES.get(v, v))
    dis = dis.rename(columns={c: c for c in dis.columns if c not in (m1, m2)})
    st.dataframe(dis.head(300), hide_index=True, width="stretch", height=320,
                 column_config={m1: st.column_config.ProgressColumn(ui.mname(m1), min_value=0, max_value=1, format="%.2f"),
                                m2: st.column_config.ProgressColumn(ui.mname(m2), min_value=0, max_value=1, format="%.2f")})

with tab_fb:
    path = C.ARTIFACTS / "feedback.csv"
    if not path.exists():
        ui.note("No feedback yet. Use <b>Record actual outcome</b> on the Predict page to log real outcomes.")
    else:
        fb = pd.read_csv(path)
        right = (fb["predicted"] == fb[C.TARGET]).mean()
        k = st.columns(3)
        k[0].markdown(ui.kpi("Feedback records", f"{len(fb):,}"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Live accuracy", f"{right*100:.0f}%", ui.GOOD if right >= .9 else ui.ACCENT), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Errors to learn from", f"{int((fb['predicted'] != fb[C.TARGET]).sum()):,}", ui.BAD),
                      unsafe_allow_html=True)
        shown = fb.iloc[::-1].copy()
        for c in C.CAT_COLS + [C.TARGET, "predicted"]:
            if c in shown:
                shown[c] = shown[c].map(lambda v: v if isinstance(v, str) else v)
        if C.CLASS_COL in shown:
            shown[C.CLASS_COL] = shown[C.CLASS_COL].map(lambda v: C.CABIN_NAMES.get(v, v) if isinstance(v, str) else v)
        st.dataframe(shown, hide_index=True, width="stretch")
        st.caption("These labelled records are appended to the training data at the next retraining cycle.")
