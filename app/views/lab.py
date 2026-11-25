"""Experiment lab: every Part A / Part B result produced by train_full.py."""
import altair as alt
import pandas as pd
import streamlit as st

import ui

ui.header("Experiment lab",
          "Part A and Part B results from the automated pipeline: cross-validation, tuning, missing values, "
            "feature selection, statistical tests, learning curves and robustness.")

OWNERS = ["Member 1", "Member 2", "Member 3", "Member 4"]
OWNER_COLORS = {"Baseline": "#4a5a73", "Member 1": "#5ab0ff", "Member 2": "#c792ea", "Member 3": "#3ddc97",
                "Member 4": "#f0b43c", "Shared": "#8b95a3"}
KEY_OWNER = {"knn": "Member 1", "svm": "Member 2", "mlp": "Member 3", "xgb": "Member 4"}


@st.cache_data
def load(name):
    """Read one experiment result file; team-ensemble rows are dropped and unreadable files
    return None so the tab shows 'not run' instead of crashing.
    """
    path = ui.EXP_DIR / name
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    except Exception as exc:  # truncated / corrupt result file: show "not run" instead of crashing
        st.warning(f"Could not read {name}: {exc}")
        return None
    # the team ensemble is not shown in the app: drop its rows wherever a table lists models
    if "owner" in df:
        df = df[df["owner"] != "Team"]
    for c in ("model", "A", "B"):
        if c in df:
            df = df[df[c] != "team"]
    return df.reset_index(drop=True)


def missing(step):
    """Note telling the user which training step produces the missing results."""
    ui.note("Not run yet. Run <code>python train_full.py --steps {s}</code> to produce these results.".format(s=step), warn=True)


def owner_scale():
    """Fixed colour per member / baseline."""
    return alt.Scale(domain=[o for o in OWNER_COLORS], range=list(OWNER_COLORS.values()))


COLUMN_NAMES = {"model": "Model", "owner": "Owner", "stage": "Stage", "F1_std": "F1 std", "fit_s": "Fit (s)",
                "predict_ms_per_row": "Predict (ms/row)", "model_family": "Model family", "strategy": "Strategy",
                "rows_used": "Rows used", "chi2": "Chi-square", "p_value": "p-value", "mean_diff": "Mean difference",
                "verdict": "Verdict"}


def localise(df):
    """Readable column names for display tables."""
    return df.rename(columns=COLUMN_NAMES)


def metric_bars(df, metric="F1", height_per=26):
    """Horizontal bars of a CV metric per model with a +/- 1 std whisker.

    The Dummy baseline (F1 0, ROC-AUC 0.5) is left out of the chart: on a shared axis it would squash the
    0.92-0.96 range where the real differences are. It stays in the table under the chart."""
    d = df[~df["model"].astype(str).str.startswith("Dummy")].copy()
    std = f"{metric}_std"
    d["lo"] = d[metric] - (d[std] if std in d else 0)
    d["hi"] = d[metric] + (d[std] if std in d else 0)
    lo = max(0.0, float(d["lo"].min()) - 0.02)
    d["base"] = lo  # bars start at the axis minimum (the axis is truncated to show small differences)
    # explicit order: sort="-x" is dropped by Vega-Lite when layers encode x with different fields
    order = d.sort_values(metric, ascending=False)["model"].drop_duplicates().tolist()
    base = alt.Chart(d).encode(y=alt.Y("model:N", sort=order, title=None))
    bars = base.mark_bar(cornerRadiusEnd=3, height=alt.RelativeBandSize(0.7)).encode(
        x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[lo, 1]), title="{m} (mean of K folds)".format(m=metric)),
        x2="base:Q",
        color=alt.Color("owner:N", scale=owner_scale(), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["model", "owner", alt.Tooltip(f"{metric}:Q", format=".4f")])
    chart = bars + base.mark_rule(color="#e4e8ee").encode(x="lo:Q", x2="hi:Q") if std in d else bars
    return ui.themed(chart.properties(height=max(120, height_per * len(d))))


@st.fragment(run_every=5)
def training_monitor():
    """Live view of train_full.py progress (refreshes every 5 s while the page is open)."""
    import time
    from src.progress import fmt_secs, read_status
    overall, running = read_status(ui.C.ARTIFACTS_FULL)
    if overall is None:
        return
    now = time.time()
    status = overall["status"]
    with st.container(border=True):
        st.markdown("#### Training monitor")
        k = st.columns(3)
        k[0].markdown(ui.kpi("Status", status.capitalize(), ui.ACCENT if status == "running" else
                             (ui.GOOD if status == "done" else ui.BAD)), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Elapsed", fmt_secs((overall["finished"] or now) - overall["started"])), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Remaining (estimate)", fmt_secs(overall.get("eta_s")) if status == "running" else "-"),
                      unsafe_allow_html=True)
        steps_box = st.expander("Pipeline steps") if status == "done" else st.container()
        for name, s in overall["steps"].items():
            frac = s["tasks_done"] / s["tasks_total"] if s["tasks_total"] else (1.0 if s["status"] == "done" else 0.0)
            took = fmt_secs((s["end"] or now) - s["start"]) if s["start"] else "-"
            steps_box.progress(min(frac, 1.0), text=f"{name} · {s['status']} · {s['tasks_done']}/{s['tasks_total']} · "
                                             f"{took} ({'estimate'} {fmt_secs(s['estimate_s'])})")
        for task in running:
            if task.get("kind") == "tuning":
                best = f"{task['best']:.4f}" if task.get("best") is not None else "-"
                st.caption("Tuning {m}: trial {d}/{n} · best F1 {b} · pruned {p} · ETA {e}".format(m=task["member"], d=task["done"], n=task["total"], b=best, p=task.get("pruned", 0), e=task.get("eta", "-")))


training_monitor()

if not ui.EXP_DIR.exists():
    ui.note("No experiment results yet - the app is running on <b>prototype</b> models. Run the full pipeline:"
              "<br><code>python train_full.py</code> (all steps, several hours) or step by step, e.g. "
              "<code>python train_full.py --steps baselines partA</code>.", warn=True)
    st.stop()

view = st.segmented_control("Focus", ["All"] + OWNERS, default="All", key="lab_focus")


def focus(df):
    """Filter a results table to the member chosen in the Focus control."""
    if df is None or view in (None, "All") or "owner" not in df:
        return df
    return df[df["owner"].isin([view, "Baseline"])]


TUNE_NAMES = {"knn": "M1 · Bagging-KNN", "svm": "M2 · Bagging-SVM", "mlp": "M3 · Seed-bagged MLP",
              "xgb": "M4 · XGBoost"}
DEFAULT_ROW = {"knn": "1. Bagging-KNN (default)", "svm": "2. Bagging-SVM (default)",
               "mlp": "3. Seed-bagged MLP (default)", "xgb": "4. XGBoost (default)"}


def convergence_chart(t: pd.DataFrame, key: str) -> alt.Chart:
    """One member's Optuna run: every trial (hollow = pruned), best-so-far line, default (trial 0) and best trial."""
    color = ui.MODEL_COLORS[key]
    t = t.sort_values("number").copy()
    t["Trial"] = t["state"].map({"COMPLETE": "Completed", "PRUNED": "Pruned early"})
    done = t[t["state"] == "COMPLETE"].copy()
    done["best"] = done["value"].cummax()
    last = pd.DataFrame([{"number": int(t["number"].max()), "best": done["best"].iloc[-1]}])
    done = pd.concat([done[["number", "best"]], last], ignore_index=True)
    default = float(t.loc[t["number"] == 0, "value"].iloc[0])
    best = done.iloc[-1:].assign(label=lambda d: d["best"].map(lambda v: f"best {v:.4f}"))
    best["number"] = int(t.loc[t["state"] == "COMPLETE"].sort_values("value").iloc[-1]["number"])
    lo = float(t["value"].quantile(0.05))
    y = alt.Y("value:Q", title="Inner-CV F1 (3 folds)", scale=alt.Scale(zero=False, domain=[lo - 0.002, best["best"].iloc[0] + 0.004],
                                                                      clamp=True))
    x = alt.X("number:Q", title="Trial", scale=alt.Scale(domain=[0, int(t["number"].max())]))
    tip = [alt.Tooltip("number:Q", title="Trial"), alt.Tooltip("value:Q", title="F1", format=".4f"), "Trial:N"]
    dots = alt.Chart(t.dropna(subset=["value"])).mark_point(size=46, strokeWidth=1.5).encode(
        x=x, y=y, tooltip=tip,
        color=alt.value("#8b95a3"),
        fill=alt.Fill("Trial:N", scale=alt.Scale(domain=["Completed", "Pruned early"], range=["#8b95a3", "transparent"]),
                      legend=ui.bottom_legend()),
        opacity=alt.value(0.75))
    line = alt.Chart(done).mark_line(interpolate="step-after", strokeWidth=2.5, color=color).encode(
        x="number:Q", y=alt.Y("best:Q"))
    rule = alt.Chart(pd.DataFrame([{"v": default, "t": f"default {default:.4f}"}])).mark_rule(
        strokeDash=[5, 4], color="#e4e8ee", opacity=.7).encode(y="v:Q")
    rule_txt = alt.Chart(pd.DataFrame([{"v": default, "t": f"default (trial 0) {default:.4f}", "x": int(t["number"].max())}])
                         ).mark_text(align="right", dy=10, fontSize=11, color="#a7b0bd").encode(x="x:Q", y="v:Q", text="t:N")
    star = alt.Chart(best).mark_point(shape="diamond", size=170, filled=True, color=color, stroke="#0e1218",
                                      strokeWidth=1.5).encode(x="number:Q", y="best:Q")
    star_txt = alt.Chart(best).mark_text(dy=-14, fontSize=11, fontWeight=600, color="#e4e8ee").encode(
        x="number:Q", y="best:Q", text="label:N")
    return (dots + line + rule + rule_txt + star + star_txt).properties(height=250, title=TUNE_NAMES[key])


def before_after(cv: dict, metric: str) -> alt.Chart:
    """Grouped bars: each algorithm's default ('Before') vs tuned ('After') score, 5-fold CV on all training rows."""
    rows = [{"Algorithm": TUNE_NAMES[k], "Settings": s, "value": float(v[s][metric])}
            for k, v in cv.items() for s in ("Before", "After")]
    d = pd.DataFrame(rows)
    lo = float(d["value"].min()) - 0.012
    hi = float(d["value"].max()) + 0.008
    d["base"] = lo  # bars start at the axis minimum (axis truncated so small gains are visible)
    gain = pd.DataFrame([{"Algorithm": TUNE_NAMES[k],
                          "top": max(float(v["Before"][metric]), float(v["After"][metric])) + 0.0045,
                          "label": f"{(float(v['After'][metric]) - float(v['Before'][metric])) * 100:+.2f} pts"}
                         for k, v in cv.items()])
    order = list(gain["Algorithm"])
    x = alt.X("Algorithm:N", sort=order, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=200))
    off = alt.XOffset("Settings:N", sort=["Before", "After"])
    settings = alt.Scale(domain=["Before", "After"], range=["#8b95a3", ui.ACCENT])
    bars = alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x, xOffset=off,
        y=alt.Y("value:Q", title=f"{metric} (mean of 5 folds)", scale=alt.Scale(domain=[lo, hi], clamp=True)),
        y2="base:Q",
        color=alt.Color("Settings:N", scale=settings, sort=["Before", "After"],
                        legend=ui.bottom_legend(labelExpr="datum.label == 'Before' ? 'Before tuning (default settings)'"
                                                          " : 'After tuning (Optuna best)'")),
        tooltip=["Algorithm", "Settings", alt.Tooltip("value:Q", title=metric, format=".4f")])
    vals = alt.Chart(d).mark_text(dy=-7, fontSize=11, color="#a7b0bd", font="JetBrains Mono").encode(
        x=x, xOffset=off, y="value:Q", text=alt.Text("value:Q", format=".4f"))
    gains = alt.Chart(gain).mark_text(dy=-10, fontSize=13, fontWeight=700, color=ui.GOOD).encode(
        x=x, y=alt.Y("top:Q"), text="label:N")
    return (bars + vals + gains).properties(height=360)


def tuning_tab(trials: pd.DataFrame, params: pd.DataFrame | None):
    """Tuning evidence: before vs after per algorithm, convergence curves, search space."""
    st.markdown("Equal budget for every algorithm: the same stratified 30,000-row subsample, **60 Optuna (TPE) trials**, "
                "3-fold inner CV, F1 objective. **Before** = default settings, **After** = best settings found. "
                "Both are scored with the same 5-fold CV on all 103,904 training rows.")
    keys = [k for k in KEY_OWNER if view in (None, "All") or KEY_OWNER[k] == view]
    final, partA = load("cv_final.csv"), load("cv_partA.csv")
    cv = {}
    if final is not None and partA is not None:
        for k in keys:
            d = partA.loc[partA["model"] == DEFAULT_ROW[k]]
            f = final.loc[final["model"] == k]
            if len(d) and len(f):
                cv[k] = {"Before": d.iloc[0], "After": f.iloc[0]}

    # 1) Before vs After, all algorithms
    if cv:
        st.markdown("#### Before vs after tuning")
        metric = st.segmented_control("Metric", ui.METRICS, default="F1", key="tune_ba_metric") or "F1"
        st.altair_chart(ui.themed(before_after(cv, metric)), width="stretch")
        tbl = pd.DataFrame([{"Algorithm": TUNE_NAMES[k], "Metric": m, "Before": float(v["Before"][m]),
                             "After": float(v["After"][m]),
                             "Change (pts)": (float(v["After"][m]) - float(v["Before"][m])) * 100}
                            for k, v in cv.items() for m in ui.METRICS])
        wide = tbl.pivot(index="Algorithm", columns="Metric", values="Change (pts)")[ui.METRICS].reindex(
            [TUNE_NAMES[k] for k in cv])
        st.markdown("Change from tuning on every metric (percentage points, positive = better):")
        st.dataframe(wide.style.format("{:+.2f}").map(lambda v: f"color: {ui.GOOD if v > 0 else ui.BAD}"),
                     width="stretch")
        stds = ", ".join(f"{TUNE_NAMES[k].split(' · ')[1]} ±{float(v['After']['F1_std']):.4f}" for k, v in cv.items())
        st.caption("The settings were chosen on a 30k subsample, then re-tested here on all rows with different folds, "
                   "so the gain is not luck on the tuning folds. Fold-to-fold F1 std after tuning: " + stds + ".")
        with st.expander("Before / after values for every metric"):
            st.dataframe(tbl.style.format({"Before": "{:.4f}", "After": "{:.4f}", "Change (pts)": "{:+.2f}"}),
                         hide_index=True, width="stretch")

    # 2) how the search got there: convergence curves, one panel per algorithm
    st.markdown("#### How tuning found it: best F1 so far vs trial")
    for i in range(0, len(keys), 2):
        row = st.columns(2)
        for c, k in zip(row, keys[i:i + 2]):
            c.altair_chart(ui.themed(convergence_chart(trials[trials["member"] == k], k)), width="stretch")
    st.caption("Inner-CV F1 on the 30k tuning subsample. Line = best score found so far (only rises). Dashed line = "
               "default settings (trial 0). Diamond = selected settings. Hollow points were stopped early by the "
               "median pruner. The curves flatten in the last trials, so 60 trials were enough.")

    # 3) parameter sensitivity: what the search explored
    st.markdown("#### What the search explored")
    a, b = st.columns(2)
    k = a.selectbox("Algorithm", keys, format_func=lambda x: TUNE_NAMES[x], key="tune_sens_model")
    t = trials[(trials["member"] == k)].dropna(subset=["value"])
    pcols = [c for c in t.columns if c.startswith("params_") and t[c].notna().any()]
    p = b.selectbox("Hyperparameter", pcols, format_func=lambda x: x.removeprefix("params_"), key="tune_sens_param")
    best_n = int(t.loc[t["state"] == "COMPLETE"].sort_values("value").iloc[-1]["number"])
    s = t[[p, "value", "number", "state"]].dropna().assign(
        Selected=lambda x: x["number"].eq(best_n).map({True: "Selected", False: "Other trials"}))
    numeric = pd.api.types.is_numeric_dtype(s[p]) and s[p].nunique() > 3
    xe = alt.X(f"{p}:{'Q' if numeric else 'N'}", title=p.removeprefix("params_"),
               scale=alt.Scale(type="log") if numeric and s[p].min() > 0 and s[p].max() / s[p].min() > 50 else alt.Undefined)
    sc = alt.Chart(s).mark_point(filled=True, strokeWidth=0).encode(
        x=xe, y=alt.Y("value:Q", title="Inner-CV F1", scale=alt.Scale(zero=False)),
        size=alt.Size("Selected:N", scale=alt.Scale(domain=["Other trials", "Selected"], range=[60, 260]), legend=None),
        color=alt.Color("Selected:N", scale=alt.Scale(domain=["Other trials", "Selected"],
                                                      range=["#8b95a3", ui.MODEL_COLORS[k]]), legend=ui.bottom_legend()),
        tooltip=[alt.Tooltip("number:Q", title="Trial"), alt.Tooltip(f"{p}:{'Q' if numeric else 'N'}"),
                 alt.Tooltip("value:Q", title="F1", format=".4f"), "state"])
    st.altair_chart(ui.themed(sc.properties(height=280)), width="stretch")
    st.caption("Each point is one trial. A clear peak shows the selected value sits in the best region of the search space.")

    with st.expander("Trial counts and best hyperparameters"):
        st.dataframe(trials[trials["member"].isin(keys)].groupby("member")["state"].value_counts().unstack(fill_value=0)
                     .rename(index=TUNE_NAMES), width="stretch")
        if params is not None:
            st.dataframe(params[params["member"].isin(keys)].astype(str), hide_index=True, width="stretch")

tabs = st.tabs([x for x in ["Part A", "Missing values", "Feature selection", "Tuning", "Part B final",
                               "Statistics", "Learning curves", "Robustness"]])

with tabs[0]:
    parts = [d for d in [load("cv_baselines.csv"), load("cv_partA.csv")] if d is not None]
    if load("cv_partA.csv") is None:
        missing("baselines partA")
    else:
        df = focus(pd.concat(parts, ignore_index=True))
        st.markdown("Default settings, identical stratified K-fold CV on the full training set. "
                      "Bars = mean, whisker = ± 1 std across folds.")
        metric = st.selectbox("Metric", ui.METRICS, index=ui.METRICS.index("F1"), key="pa_metric")
        st.altair_chart(metric_bars(df, metric), width="stretch")
        cols = ["model", "owner", "stage", *ui.METRICS, "F1_std", "fit_s", "predict_ms_per_row"]
        st.dataframe(localise(df[cols].sort_values("F1", ascending=False)).style.format(
                     {**{c: "{:.4f}" for c in [*ui.METRICS, "F1 std"]}, "Fit (s)": "{:.1f}", "Predict (ms/row)": "{:.3f}"}),
                     hide_index=True, width="stretch")
        folds = load("folds_partA.csv")
        if folds is not None:
            folds = focus(folds)
            box = alt.Chart(folds).mark_boxplot(size=18).encode(
                x=alt.X("F1:Q", scale=alt.Scale(zero=False)), y=alt.Y("model:N", title=None),
                color=alt.Color("owner:N", scale=owner_scale(), legend=None))
            st.markdown("#### Fold-to-fold stability")
            st.altair_chart(ui.themed(box.properties(height=30 * folds["model"].nunique())), width="stretch")

with tabs[1]:
    df = load("cv_missing.csv")
    if df is None:
        missing("missing")
    else:
        st.markdown("**Owner: Member 1.** Same folds and models, different handling of missing arrival delay and "
                      "ratings recorded as 0 (N/A).")
        df = df.copy()
        df[["model_family", "strategy"]] = df["model"].str.split(" \\| ", expand=True)
        df = df
        f1_lo = float(df["F1"].min()) - 0.005
        df["base"] = f1_lo
        ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=14).encode(
            x=alt.X("F1:Q", title="F1", scale=alt.Scale(domain=[f1_lo, float(df["F1"].max()) + 0.002])), x2="base:Q", y=alt.Y("strategy:N", title=None, sort="-x"),
            color=alt.Color("model_family:N", scale=alt.Scale(range=[ui.ACCENT, "#5ab0ff"]), title=None),
            yOffset="model_family:N", tooltip=["model", alt.Tooltip("F1:Q", format=".4f"), "rows_used"])
        st.altair_chart(ui.themed(ch.properties(height=260)), width="stretch")
        st.dataframe(localise(df[["model_family", "strategy", "rows_used", *ui.METRICS, "F1_std"]])
                     .style.format({**{c: "{:.4f}" for c in [*ui.METRICS, "F1 std"]}, "Rows used": "{:,.0f}"}), hide_index=True,
                     width="stretch")

with tabs[2]:
    df, mi = load("cv_features.csv"), load("feature_mi.csv")
    if df is None:
        missing("features")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("#### Mutual information ranking")
            mi = mi.assign(feature=mi["feature"].map(lambda f: ui.FEATURE_LABELS.get(f, f)))
            st.altair_chart(ui.themed(alt.Chart(mi.head(25)).mark_bar(cornerRadiusEnd=3, height=10, color=ui.ACCENT).encode(
                x=alt.X("mutual_info:Q", title="Mutual information with satisfaction"),
                y=alt.Y("feature:N", sort="-x", title=None)).properties(height=420)), width="stretch")
        with b:
            st.markdown("#### Feature-set ablation (XGBoost)")
            st.altair_chart(metric_bars(df), width="stretch")

with tabs[3]:
    trials, params = load("tuning_trials.csv"), load("tuned_params_table.csv")
    if trials is None:
        missing("tune")
    else:
        tuning_tab(trials, params)

with tabs[4]:
    final, partA = load("cv_final.csv"), load("cv_partA.csv")
    if final is None:
        missing("final")
    else:
        names = {"knn": "1. Bagging-KNN (tuned)", "svm": "2. Bagging-SVM (tuned)", "mlp": "3. Seed-bagged MLP (tuned)",
                 "xgb": "4. XGBoost (tuned)"}
        final = final.assign(model=final["model"].replace(names))
        df = focus(pd.concat([d for d in [partA, final] if d is not None], ignore_index=True))
        st.markdown("Default → tuned → advanced technique, per member, all with identical K-fold CV on the full training set.")
        st.altair_chart(metric_bars(df), width="stretch")
        st.dataframe(localise(df[["model", "owner", "stage", *ui.METRICS, "F1_std", "fit_s"]]
                                  .sort_values(["owner", "F1"], ascending=[True, False])).style.format(precision=4),
                     hide_index=True, width="stretch")

with tabs[5]:
    mc, t52, ci = load("stat_mcnemar.csv"), load("stat_5x2cv.csv"), load("stat_bootstrap.csv")
    if mc is None:
        missing("analysis")
    else:
        st.markdown("#### McNemar test on the hold-out set")
        mc = mc.assign(A=mc["A"].map(ui.mname), B=mc["B"].map(ui.mname),
                       verdict=mc["p_value"].map(lambda p: "significant" if p < 0.05 else "not significant"))
        st.dataframe(localise(mc).style.format({"Chi-square": "{:.1f}", "p-value": "{:.2g}"}), hide_index=True,
                     width="stretch")
        if t52 is not None and len(t52):
            st.markdown("#### Dietterich 5x2cv paired t-test (F1)")
            st.dataframe(localise(t52.assign(A=t52["A"].map(ui.mname), B=t52["B"].map(ui.mname))).style.format(
                {"Mean difference": "{:+.4f}", "t": "{:.2f}", "p-value": "{:.3g}"}), hide_index=True, width="stretch")
        if ci is not None:
            st.markdown("#### Bootstrap 95% confidence intervals (hold-out)")
            metric = st.selectbox("Metric", [m for m in ui.METRICS if m in set(ci["metric"])], key="ci_metric")
            d = ci[ci["metric"] == metric].assign(model=lambda x: x["model"].map(ui.mname))
            order = d.sort_values("value", ascending=False)["model"].tolist()
            base = alt.Chart(d).encode(y=alt.Y("model:N", sort=order, title=None))
            st.altair_chart(ui.themed((base.mark_rule(strokeWidth=3, color=ui.ACCENT).encode(
                x=alt.X("lo:Q", scale=alt.Scale(zero=False), title=metric), x2="hi:Q")
                + base.mark_point(filled=True, size=80, color="#ffffff").encode(x="value:Q")).properties(height=260)),
                width="stretch")

with tabs[6]:
    lc = load("learning_curves.csv")
    if lc is None:
        missing("analysis")
    else:
        lc = lc.copy()
        lc["owner"] = lc["model"].map(KEY_OWNER)
        if view not in (None, "All"):
            lc = lc[lc["owner"] == view]
        long = lc.melt(["model", "owner", "train_size"], ["train_f1", "valid_f1"], var_name="set", value_name="F1")
        long = long
        ch = alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("train_size:Q", title="Training rows"), y=alt.Y("F1:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("owner:N", scale=owner_scale(), legend=alt.Legend(orient="bottom", title=None)),
            strokeDash=alt.StrokeDash("set:N", legend=alt.Legend(orient="bottom", title=None)))
        st.altair_chart(ui.themed(ch.properties(height=360)), width="stretch")
        st.caption("A large train-validation gap = variance (over-fitting); both curves low = bias (under-fitting).")

with tabs[7]:
    rb = load("robustness.csv")
    if rb is None:
        missing("analysis")
    else:
        rb = rb.assign(model=rb["model"].map(ui.mname))
        for test in rb["test"].unique():
            st.markdown(f"#### {test}")
            d = rb[rb["test"] == test]
            st.altair_chart(ui.themed(alt.Chart(d).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("level:Q", title="Severity"), y=alt.Y("F1:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=260)),
                width="stretch")
