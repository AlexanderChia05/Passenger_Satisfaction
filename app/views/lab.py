"""Experiment lab: every Part A / Part B result produced by train_full.py."""
import altair as alt
import pandas as pd
import streamlit as st

import ui

ui.header("GATE 08 · EVIDENCE", "Experiment lab",
          "Part A and Part B results from the automated pipeline: cross-validation, tuning, missing values, "
            "feature selection, ensembles, statistical tests, learning curves and robustness.")

OWNERS = ["Member 1", "Member 2", "Member 3", "Member 4"]
OWNER_COLORS = {"Baseline": "#4a5a73", "Member 1": "#5ab0ff", "Member 2": "#c792ea", "Member 3": "#3ddc97",
                "Member 4": "#ffb627", "Team": "#ffffff", "Shared": "#8a9bb4"}
KEY_OWNER = {"knn": "Member 1", "svm": "Member 2", "mlp": "Member 3", "xgb": "Member 4"}


@st.cache_data
def load(name):
    path = ui.EXP_DIR / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)


def missing(step):
    ui.note("Not run yet. Run <code>python train_full.py --steps {s}</code> to produce these results.".format(s=step), warn=True)


def owner_scale():
    return alt.Scale(domain=[o for o in OWNER_COLORS], range=list(OWNER_COLORS.values()))


def localise(df):
    """Translate owner / model / stage values for display."""
    return df


def metric_bars(df, metric="F1", height_per=26):
    d = localise(df)
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
    chart = bars + base.mark_rule(color="#e9eef6").encode(x="lo:Q", x2="hi:Q") if std in d else bars
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
        st.markdown(f"#### {'Training monitor'}")
        k = st.columns(3)
        k[0].markdown(ui.kpi("Status", status.upper(), ui.ACCENT if status == "running" else
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
    if df is None or view in (None, "All") or "owner" not in df:
        return df
    return df[df["owner"].isin([view, "Baseline", "Team"])]


tabs = st.tabs([x for x in ["Part A", "Missing values", "Feature selection", "Tuning", "Part B final",
                               "Team ensemble", "Statistics", "Learning curves", "Robustness"]])

with tabs[0]:
    parts = [d for d in [load("cv_baselines.csv"), load("cv_partA.csv")] if d is not None]
    if load("cv_partA.csv") is None:
        missing("baselines partA")
    else:
        df = focus(pd.concat(parts, ignore_index=True))
        st.markdown("Default settings, identical stratified K-fold CV on the full training set. "
                      "Bars = mean, white line = ± 1 std across folds.")
        metric = st.selectbox("Metric", ui.METRICS, index=ui.METRICS.index("F1"), key="pa_metric")
        st.altair_chart(metric_bars(df, metric), width="stretch")
        cols = ["model", "owner", "stage", *ui.METRICS, "F1_std", "fit_s", "predict_ms_per_row"]
        st.dataframe(localise(df[cols].sort_values("F1", ascending=False)).style.format(precision=4),
                     hide_index=True, width="stretch")
        folds = load("folds_partA.csv")
        if folds is not None:
            folds = localise(focus(folds))
            box = alt.Chart(folds).mark_boxplot(size=18).encode(
                x=alt.X("F1:Q", scale=alt.Scale(zero=False)), y=alt.Y("model:N", title=None),
                color=alt.Color("owner:N", scale=owner_scale(), legend=None))
            st.markdown(f"#### {'Fold-to-fold stability'}")
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
        st.dataframe(df[["model_family", "strategy", "rows_used", *ui.METRICS, "F1_std"]]
                     .style.format(precision=4), hide_index=True, width="stretch")

with tabs[2]:
    df, mi = load("cv_features.csv"), load("feature_mi.csv")
    if df is None:
        missing("features")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown(f"#### {'Mutual information ranking'}")
            mi = mi.assign(feature=mi["feature"].map(lambda f: ui.FEATURE_LABELS.get(f, f)))
            st.altair_chart(ui.themed(alt.Chart(mi.head(25)).mark_bar(cornerRadiusEnd=3, height=10, color=ui.ACCENT).encode(
                x=alt.X("mutual_info:Q", title="Mutual information with satisfaction"),
                y=alt.Y("feature:N", sort="-x", title=None)).properties(height=420)), width="stretch")
        with b:
            st.markdown(f"#### {'Feature-set ablation (XGBoost)'}")
            st.altair_chart(metric_bars(df), width="stretch")

with tabs[3]:
    trials, params = load("tuning_trials.csv"), load("tuned_params_table.csv")
    if trials is None:
        missing("tune")
    else:
        st.markdown("Equal budget for every member: same tuning subsample, same number of Optuna trials, same inner CV. "
                      "Pruned trials were stopped early by the median pruner.")
        trials = trials.copy()
        trials["owner"] = trials["member"].map(KEY_OWNER)
        if view not in (None, "All"):
            trials = trials[trials["owner"] == view]
        trials = trials
        done = trials[trials["state"] == "COMPLETE"].sort_values("number")
        done["best so far"] = done.groupby("member")["value"].cummax()
        pts = alt.Chart(trials.dropna(subset=["value"])).mark_circle(size=40, opacity=.5).encode(
            x=alt.X("number:Q", title="Trial"), y=alt.Y("value:Q", title="Inner-CV F1", scale=alt.Scale(zero=False)),
            color=alt.Color("owner:N", scale=owner_scale(), legend=alt.Legend(orient="bottom", title=None)),
            shape=alt.Shape("state:N", legend=None), tooltip=["member", "number", "state", alt.Tooltip("value:Q", format=".4f")])
        line = alt.Chart(done).mark_line(strokeWidth=2.5).encode(x="number:Q", y="best so far:Q",
                                                                color=alt.Color("owner:N", scale=owner_scale(), legend=None))
        st.altair_chart(ui.themed((pts + line).properties(height=340)), width="stretch")
        st.dataframe(trials.groupby("member")["state"].value_counts().unstack(fill_value=0), width="stretch")
        if params is not None:
            st.markdown(f"#### {'Best hyperparameters'}")
            st.dataframe(params.astype(str), hide_index=True, width="stretch")

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
    team, depth = load("cv_team.csv"), load("stack_depth.csv")
    if team is None:
        missing("team")
    else:
        final = load("cv_final.csv")
        best_single = final[final["stage"] == "Part B - tuned"].nlargest(1, "F1") if final is not None else None
        k = st.columns(3)
        chosen = team.loc[team["F1"].idxmax()]
        k[0].markdown(ui.kpi("Chosen team ensemble", chosen["model"], ui.ACCENT), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Team OOF F1", f"{chosen['F1']:.4f}", ui.GOOD), unsafe_allow_html=True)
        if best_single is not None and len(best_single):
            gain = (chosen["F1"] - best_single["F1"].iloc[0]) * 100
            k[2].markdown(ui.kpi("vs best single member", "{d:+.2f} pts".format(d=gain), ui.GOOD if gain > 0 else ui.BAD),
                          unsafe_allow_html=True)
        st.dataframe(localise(team[["model", *ui.METRICS]])
                     .style.format(precision=4), hide_index=True, width="stretch")
        if depth is not None:
            st.markdown(f"#### {'Stacking meta-tree depth (chosen by CV on out-of-fold predictions)'}")
            st.altair_chart(ui.themed(alt.Chart(depth).mark_line(point=True, color=ui.ACCENT).encode(
                x=alt.X("depth:O", title="depth"), y=alt.Y("F1:Q", scale=alt.Scale(zero=False))).properties(height=220)),
                width="stretch")

with tabs[6]:
    mc, t52, ci = load("stat_mcnemar.csv"), load("stat_5x2cv.csv"), load("stat_bootstrap.csv")
    if mc is None:
        missing("analysis")
    else:
        st.markdown(f"#### {'McNemar test on the hold-out set'}")
        mc = mc.assign(A=mc["A"].map(ui.mname), B=mc["B"].map(ui.mname),
                       verdict=mc["p_value"].map(lambda p: "significant" if p < 0.05 else "not significant"))
        st.dataframe(mc.style.format({"chi2": "{:.1f}", "p_value": "{:.2g}"}), hide_index=True, width="stretch")
        if t52 is not None and len(t52):
            st.markdown(f"#### {'Dietterich 5x2cv paired t-test (F1)'}")
            st.dataframe(t52.assign(A=t52["A"].map(ui.mname), B=t52["B"].map(ui.mname)).style.format(
                {"mean_diff": "{:+.4f}", "t": "{:.2f}", "p_value": "{:.3g}"}), hide_index=True, width="stretch")
        if ci is not None:
            st.markdown(f"#### {'Bootstrap 95% confidence intervals (hold-out)'}")
            metric = st.selectbox("Metric", [m for m in ui.METRICS if m in set(ci["metric"])], key="ci_metric")
            d = ci[ci["metric"] == metric].assign(model=lambda x: x["model"].map(ui.mname))
            order = d.sort_values("value", ascending=False)["model"].tolist()
            base = alt.Chart(d).encode(y=alt.Y("model:N", sort=order, title=None))
            st.altair_chart(ui.themed((base.mark_rule(strokeWidth=3, color=ui.ACCENT).encode(
                x=alt.X("lo:Q", scale=alt.Scale(zero=False), title=metric), x2="hi:Q")
                + base.mark_point(filled=True, size=80, color="#ffffff").encode(x="value:Q")).properties(height=260)),
                width="stretch")

with tabs[7]:
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

with tabs[8]:
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
