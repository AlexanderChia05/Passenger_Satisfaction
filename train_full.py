"""Full Part A + Part B experiment pipeline (automated, resumable, process-parallel).

    python train_full.py                      # everything
    python train_full.py --steps tune final   # run selected steps only
    python train_full.py --force              # redo tuning even if tuned params exist
    python train_full.py --smoke --out tmp/   # tiny run to check the code works (NOT real results)
    python train_status.py                    # live progress in another terminal (or Experiment lab page)

Steps (in order):
  baselines  Dummy + Rule, K-fold CV on the full training set
  partA      each member: default base model + default ensemble, K-fold CV
  missing    Part B: missing-value strategies (study subsample)
  features   Part B: feature selection (study subsample)
  tune       Part B: Optuna tuning, equal budget per member, 4 members tuned at the same time
  final      tuned ensembles + advanced techniques, K-fold CV -> out-of-fold predictions
  team       team ensemble from OOF: tuned-weight soft vote and stacking
  holdout    fit everything on the full training set, score test.csv ONCE
  analysis   McNemar, 5x2cv t-test, bootstrap CIs, learning curves, robustness, fairness
  trust      conformal calibration, OOD, drift reference, surrogate rules, monotonicity, stages

Parallelism: every (model x fold) is an independent task run in a pool of worker
processes; each model inside a worker gets a fixed thread budget so that
workers x threads ~= CPU threads. Outputs go to artifacts/full/.
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src import config as C
from src import parallel, tasks, trust
from src.ensemble import fit_stack_meta, optimise_weights
from src.estimators import PrefitStack, WeightedVote
from src.evaluation import (DISPLAY_METRICS, bootstrap_cis, make_cv, mcnemar, metrics_from_proba, robustness_df, subgroup_df,
                            summarize)
from src.features import AirlineFeatures, DropColumns, StageMask, load_raw, split_xy
from src.members import MEMBERS, m1_knn, m4_tree
from src.models import RuleBaseline
from src.progress import Tracker, fmt_secs
from src.store import load_variants, save_variants
from src.tuning import best_params_table

STEPS = ["baselines", "partA", "missing", "features", "tune", "final", "team", "holdout", "analysis", "trust"]

THREADS = os.cpu_count() or 4
POOLS = {  # (worker processes, threads per model inside a worker)
    "cv": (max(1, THREADS // 2), 2),
    "tune": (4, max(1, THREADS // 4)),
    "fit": (max(1, THREADS // 2), 2),
    "analysis": (4, max(1, THREADS // 4)),
}

# Estimated wall-clock seconds per step on the full data with the pools above
# (measured on an 8-core / 16-thread laptop; see the estimate table in the README).
ESTIMATES = {"baselines": 30, "partA": 300, "missing": 150, "features": 200, "tune": 5400, "final": 900,
             "team": 30, "holdout": 500, "analysis": 720, "trust": 480}

# Relative cost of one task per algorithm family (longest tasks are scheduled first).
FAMILY_COST = {"svm": 8, "SVM": 8, "Nystroem": 3, "mlp": 6, "MLP": 6, "knn": 5, "KNN": 5, "NCA": 6,
               "xgb": 1, "XGBoost": 1, "AdaBoost": 2, "Tree": 0.5, "Dummy": 0.1, "Rule": 0.1}


def _cost(name):
    return max([v for k, v in FAMILY_COST.items() if k in name] or [1])


class Run:
    """Holds data, index subsets and experiment outputs (so steps can resume)."""

    def __init__(self, out, smoke):
        self.out, self.exp = out, out / "experiments"
        self.exp.mkdir(parents=True, exist_ok=True)
        X, y = split_xy(load_raw("train"))
        Xt, yt = split_xy(load_raw("test"))
        if smoke:  # tiny data: only proves the code runs end to end
            X, _, y, _ = train_test_split(X, y, train_size=3_000, stratify=y, random_state=C.SEED)
            Xt, _, yt, _ = train_test_split(Xt, yt, train_size=2_000, stratify=yt, random_state=C.SEED)
        self.X, self.y = X.reset_index(drop=True), y.reset_index(drop=True)
        self.X_test, self.y_test = Xt.reset_index(drop=True), yt.reset_index(drop=True)
        self.idx_all = np.arange(len(self.X))
        self.idx_tune = self._subsample(C.EXP["tune_rows"])
        self.idx_study = self._subsample(C.EXP["study_rows"])
        self.tracker, self.step = None, None
        print(f"train {self.X.shape} | test {self.X_test.shape} | tune {len(self.idx_tune)} | "
              f"study {len(self.idx_study)} | CPU threads {THREADS} | pools {POOLS}")

    def _subsample(self, n):
        if n >= len(self.X):
            return self.idx_all
        idx, _ = train_test_split(self.idx_all, train_size=n, stratify=self.y, random_state=C.SEED)
        return np.sort(idx)

    def pool(self, kind, task_list):
        workers, inner = POOLS[kind]
        return parallel.run(task_list, min(workers, max(len(task_list), 1)), inner, self.X, self.y, self.out,
                            self.tracker, self.step)

    # ---- persistence helpers ----
    def path(self, name):
        return self.exp / name

    def has(self, name):
        return self.path(name).exists()

    def save_df(self, df, name):
        df.to_csv(self.path(name), index=False)

    def load_df(self, name):
        return pd.read_csv(self.path(name))

    def save_json(self, obj, name):
        self.path(name).write_text(json.dumps(obj, indent=2, default=float))

    def load_json(self, name):
        return json.loads(self.path(name).read_text())


def run_cv_block(run, entries, idx, name, keep_oof=False):
    """entries: list of (model_name, owner, stage, estimator). Every (model, fold) runs in parallel."""
    idx = np.asarray(idx)
    y_sub = run.y.to_numpy()[idx]
    splits = list(make_cv().split(np.zeros(len(idx)), y_sub))
    task_list = [{"name": f"{m} | fold {f}", "fn": tasks.fold_task, "args": (est, idx[tr], idx[va]), "cost": _cost(m)}
                 for m, _, _, est in entries for f, (tr, va) in enumerate(splits)]
    res = run.pool("cv", task_list)

    rows, folds, oof = [], [], {}
    for m, owner, stage, _ in entries:
        parts = [res.get(f"{m} | fold {f}") for f in range(len(splits))]
        if any(p is None for p in parts):
            print(f"  [{m}] skipped: a fold failed")
            continue
        p_all = np.full(len(idx), np.nan)
        fold_rows = []
        for f, ((_, va), part) in enumerate(zip(splits, parts)):
            p_all[va] = part["p"]
            fold_rows.append({"fold": f, **metrics_from_proba(y_sub[va], part["p"]), "fit_s": part["fit_s"],
                              "predict_ms_per_row": part["predict_ms_per_row"]})
        fold_df = pd.DataFrame(fold_rows)
        rows.append(summarize(m, owner, stage, fold_df))
        folds.append(fold_df.assign(model=m, owner=owner, stage=stage))
        oof[m] = p_all
        print(f"  {m:45s} F1 {rows[-1]['F1']:.4f} ± {rows[-1]['F1_std']:.4f}  AUC {rows[-1]['ROC-AUC']:.4f}")
    if rows:
        run.save_df(pd.DataFrame(rows), f"cv_{name}.csv")
        run.save_df(pd.concat(folds, ignore_index=True), f"folds_{name}.csv")
    if keep_oof:
        pd.DataFrame(oof).assign(y_true=y_sub).to_parquet(run.path(f"oof_{name}.parquet"))
    return rows


# ======================================================================= steps
def step_baselines(run):
    run_cv_block(run, [("Dummy (majority)", "Baseline", "Baseline", DummyClassifier(strategy="most_frequent")),
                       ("Rule (business trip + Business class)", "Baseline", "Baseline", RuleBaseline())],
                 run.idx_all, "baselines", keep_oof=True)


def step_partA(run):
    entries = []
    for m in MEMBERS:
        entries += [(f"{m.OWNER[-1]}. {m.BASE_NAME}", m.OWNER, "Part A - base", m.base()),
                    (f"{m.OWNER[-1]}. {m.ENSEMBLE_NAME} (default)", m.OWNER, "Part A - ensemble", m.ensemble())]
    run_cv_block(run, entries, run.idx_all, "partA")


def step_missing(run):
    """Part B: how to handle missing arrival delay and ratings recorded as 0 (= N/A).
    Each strategy is its own CV block because 'drop rows' changes the rows used."""
    X = run.X.iloc[run.idx_study]
    incomplete = (X["Arrival Delay in Minutes"].isna() | (X[C.RATING_COLS] == 0).any(axis=1)).to_numpy()
    strategies = {
        "Drop incomplete rows": ({}, run.idx_study[~incomplete]),
        "Ratings: keep 0 as a value": ({"rating_zero": "keep"}, run.idx_study),
        "Ratings: 0 -> median + count (default)": ({}, run.idx_study),
        "Arrival delay <- median": ({"arr_fill": "median"}, run.idx_study),
        "Arrival delay <- departure delay (default)": ({"arr_fill": "departure"}, run.idx_study),
    }
    all_rows = []
    for model_key, member in [("KNN", m1_knn), ("XGBoost", m4_tree)]:
        for strat, (feat_kwargs, idx) in strategies.items():
            est = member.ensemble() if model_key == "XGBoost" else member.base()
            est.set_params(**{f"features__{k}": v for k, v in feat_kwargs.items()})
            safe = strat.replace(" ", "_").replace(":", "").replace(">", "").replace("<", "").replace("-", "")
            rows = run_cv_block(run, [(f"{model_key} | {strat}", "Member 1", "Missing values", est)], idx,
                                f"missing_{model_key}_{safe}")
            for r in rows:
                r["rows_used"] = int(len(idx))
            all_rows += rows
    run.save_df(pd.DataFrame(all_rows), "cv_missing.csv")
    run.save_json({"incomplete_share": float(incomplete.mean())}, "missing_meta.json")


def _mi(X_, y_):
    return mutual_info_classif(X_, y_, random_state=C.SEED)


def step_features(run):
    """Part B: feature selection with mutual information and ablations (XGBoost, defaults)."""
    X, y = run.X.iloc[run.idx_study], run.y.iloc[run.idx_study]
    F = AirlineFeatures().fit(X).transform(X)
    run.save_df(pd.DataFrame({"feature": F.columns, "mutual_info": _mi(F, y)}).sort_values("mutual_info", ascending=False),
                "feature_mi.csv")
    xgb = m4_tree.ensemble().named_steps["model"]
    entries = [("All 25 engineered features", None)]
    entries += [(f"Top {k} by mutual information", SelectKBest(_mi, k=k)) for k in [8, 12, 16, 20]]
    entries += [("Drop arrival delay (redundant, r=0.965)", DropColumns(["log_arr_delay"])),
                ("Drop derived features", DropColumns(["service_mean", "n_unrated", "delay_recovered"])),
                ("Drop demographics (gender, age)", DropColumns(["is_male", "age"]))]
    blocks = [(name, "Shared", "Feature selection",
               Pipeline([("features", AirlineFeatures())] + ([("select", sel)] if sel is not None else []) + [("model", xgb)]))
              for name, sel in entries]
    run_cv_block(run, blocks, run.idx_study, "features")


def step_tune(run, force):
    params = run.load_json("tuned_params.json") if run.has("tuned_params.json") else {}
    trials = [run.load_df("tuning_trials.csv")] if run.has("tuning_trials.csv") else []
    todo = [m for m in MEMBERS if force or m.KEY not in params]
    for m in MEMBERS:
        if m not in todo:
            print(f"  {m.KEY}: already tuned (use --force to redo)")
    task_list = [{"name": f"tune {m.KEY}", "fn": tasks.tune_task, "args": (m.KEY, run.idx_tune, C.EXP["n_trials"],
                                                                           C.EXP["inner_folds"]), "cost": _cost(m.KEY)}
                 for m in todo]
    res = run.pool("tune", task_list) if task_list else {}
    for m in todo:
        r = res.get(f"tune {m.KEY}")
        if r is None:
            print(f"  {m.KEY}: tuning failed; keeping defaults")
            continue
        params[m.KEY] = r["best"]
        trials = [t[t["member"] != m.KEY] for t in trials] + [r["trials"]]
    run.save_json(params, "tuned_params.json")
    if trials:
        run.save_df(pd.concat(trials, ignore_index=True), "tuning_trials.csv")
    run.save_df(best_params_table(params), "tuned_params_table.csv")


def tuned_params(run):
    params = run.load_json("tuned_params.json") if run.has("tuned_params.json") else {}
    return {m.KEY: params.get(m.KEY, {}) for m in MEMBERS}


def step_final(run):
    """Tuned ensembles + advanced techniques on the full training set (K-fold CV, OOF kept)."""
    params = tuned_params(run)
    entries = [(m.KEY, m.OWNER, "Part B - tuned", m.ensemble(**params[m.KEY])) for m in MEMBERS]
    for m in MEMBERS:
        for name, est in m.advanced(params[m.KEY]).items():
            entries.append((f"{m.OWNER[-1]}. {name}", m.OWNER, "Part B - advanced", est))
    run_cv_block(run, entries, run.idx_all, "final", keep_oof=True)


def step_team(run):
    oof = pd.read_parquet(run.path("oof_final.parquet"))
    keys = [m.KEY for m in MEMBERS if m.KEY in oof]
    P, y = oof[keys].to_numpy(), oof["y_true"].to_numpy()
    w = optimise_weights(P, y, C.EXP["weight_trials"])
    meta, depth_scores = fit_stack_meta(P, y)
    meta_oof = cross_val_predict(meta, P, y, cv=make_cv(), method="predict_proba")[:, 1]
    candidates = {"Equal-weight soft vote": P.mean(axis=1), "Tuned-weight soft vote": P @ w,
                  "Stacking (decision-tree meta)": meta_oof}
    team = pd.DataFrame([{"model": n, "owner": "Team", "stage": "Team ensemble", **metrics_from_proba(y, p)}
                         for n, p in candidates.items()])
    run.save_df(team, "cv_team.csv")
    run.save_df(depth_scores, "stack_depth.csv")
    best = team.loc[team["F1"].idxmax(), "model"]
    run.save_json({"members": keys, "weights": w.tolist(), "chosen": best}, "team.json")
    joblib.dump(meta, run.path("stack_meta.joblib"))
    pd.DataFrame({"team": candidates[best], "y_true": y}).to_parquet(run.path("oof_team.parquet"))
    print(team[["model", "F1", "ROC-AUC", "Brier"]].to_string(index=False))
    print(f"  weights {dict(zip(keys, np.round(w, 3)))} -> chosen: {best}")


def step_holdout(run):
    """Fit every model on the full training set (in parallel) and score the untouched test set once."""
    params = tuned_params(run)
    team_cfg = run.load_json("team.json")
    to_fit = {"dummy": DummyClassifier(strategy="most_frequent"), "rule": RuleBaseline()}
    for m in MEMBERS:
        to_fit[m.KEY] = m.ensemble(**params[m.KEY])
        to_fit[f"{m.KEY}_base"] = m.base()
        for name, est in m.advanced(params[m.KEY]).items():
            to_fit[f"{m.KEY}::{name}"] = est
    for s in C.STAGE_ORDER[:-1]:
        est = m4_tree.ensemble(**params["xgb"])
        est.steps.insert(0, ("stage", StageMask(s)))
        to_fit[f"stage::{s}"] = est
    res = run.pool("fit", [{"name": k, "fn": tasks.fit_task, "args": (est, run.idx_all), "cost": _cost(k)}
                           for k, est in to_fit.items()])
    fitted = {k: r["model"] for k, r in res.items() if r is not None}
    fit_s = {k: r["fit_s"] for k, r in res.items() if r is not None}

    members = [(k, fitted[k]) for k in team_cfg["members"]]
    vote = WeightedVote(members, team_cfg["weights"]).fit(run.X, run.y)
    stack = PrefitStack(members, joblib.load(run.path("stack_meta.joblib"))).fit(run.X, run.y)
    team = stack if team_cfg["chosen"].startswith("Stacking") else vote
    fitted["xgb_mono"] = fitted.get("xgb::XGBoost (monotone constraints)")

    app_keys = ["dummy", "rule"] + [m.KEY for m in MEMBERS]
    models = {k: fitted[k] for k in app_keys if k in fitted}
    models["team"] = team
    variants = {k: v for k, v in fitted.items() if k not in models and v is not None}
    variants.update({"team_vote": vote, "team_stack": stack})

    scored = {**models, **{k: v for k, v in variants.items() if not k.startswith("stage::")}}
    probs = pd.DataFrame({k: m.predict_proba(run.X_test)[:, 1] for k, m in scored.items()})
    rows = [{"key": k, **metrics_from_proba(run.y_test, probs[k]), "fit_seconds": fit_s.get(k)} for k in probs]
    holdout = pd.DataFrame(rows).sort_values("F1", ascending=False)
    run.save_df(holdout, "holdout.csv")
    print(holdout[["key", *DISPLAY_METRICS]].to_string(index=False))

    stage_rows = []
    for s in C.STAGE_ORDER:
        p = probs["xgb"] if s == "Landed" else variants[f"stage::{s}"].predict_proba(run.X_test)[:, 1]
        mt = metrics_from_proba(run.y_test, p)
        stage_rows.append({"stage": s, "known ratings": len(C.stage_known_ratings(s)),
                           **{m: mt[m] for m in DISPLAY_METRICS}})
    run.save_df(pd.DataFrame(stage_rows), "stage_metrics.csv")

    joblib.dump(models, run.out / "models.joblib", compress=("xz", 9))  # xz: ~6x smaller (KNN bags hold many copies of the data), keeps files <100 MB for git
    save_variants(variants, run.out)
    probs.assign(y_true=run.y_test.to_numpy()).join(run.X_test).to_parquet(run.out / "test_predictions.parquet")
    sample = run.X_test.sample(min(300, len(run.X_test)), random_state=C.SEED)
    sample[C.TARGET] = np.where(run.y_test.loc[sample.index] == 1, C.POSITIVE_LABEL, "neutral or dissatisfied")
    sample.to_csv(run.out / "sample_batch.csv", index=False)
    pool = run.X_test[(run.X_test["Age"] > 50) & (run.X_test["Flight Distance"] > 2000)]
    shifted = pool.sample(min(300, len(pool)), random_state=C.SEED).copy()
    shifted["Departure Delay in Minutes"] += 90
    shifted["Arrival Delay in Minutes"] = shifted["Arrival Delay in Minutes"].fillna(0) + 90
    shifted.to_csv(run.out / "sample_batch_shifted.csv", index=False)
    (run.out / "metrics.json").write_text(json.dumps({
        "mode": "full", "trained_on_rows": len(run.X), "test_rows": len(run.X_test), "team": team_cfg["chosen"],
        "models": {r["key"]: {k: v for k, v in r.items() if k != "key"} for r in rows}}, indent=2, default=float))


def step_analysis(run):
    models = joblib.load(run.out / "models.joblib")
    probs = pd.read_parquet(run.out / "test_predictions.parquet")
    y = run.y_test.to_numpy()
    keys = [m.KEY for m in MEMBERS] + ["team"]
    best = max(keys, key=lambda k: metrics_from_proba(y, probs[k])["F1"])
    params = tuned_params(run)
    build = {m.KEY: m.ensemble(**params[m.KEY]) for m in MEMBERS}
    best_member = max([m.KEY for m in MEMBERS], key=lambda k: metrics_from_proba(y, probs[k])["F1"])

    # parallel part: 5x2cv paired t-tests + learning curves
    others = [k for k in build if k != best_member] + ["rule"]
    task_list = [{"name": f"5x2cv {best_member} vs {o}", "fn": tasks.t52_task, "cost": _cost(o) + _cost(best_member),
                  "args": (build[best_member], RuleBaseline() if o == "rule" else build[o], run.idx_study)} for o in others]
    task_list += [{"name": f"learning curve {m.KEY}", "fn": tasks.lc_task, "cost": _cost(m.KEY),
                   "args": (m.KEY, build[m.KEY], run.idx_study)} for m in MEMBERS]
    res = run.pool("analysis", task_list)
    run.save_df(pd.DataFrame([{"A": best_member, "B": o, **res[f"5x2cv {best_member} vs {o}"]}
                              for o in others if res.get(f"5x2cv {best_member} vs {o}")]), "stat_5x2cv.csv")
    run.save_df(pd.concat([res[f"learning curve {m.KEY}"] for m in MEMBERS if res.get(f"learning curve {m.KEY}") is not None]),
                "learning_curves.csv")

    # sequential part (fast): McNemar, bootstrap CIs, robustness, fairness
    run.tracker.add_tasks(run.step, 4)
    t0 = time.perf_counter()
    rows = [{"A": k, "B": b, **mcnemar(y, probs[k], probs[b])} for k in keys for b in ["rule", "dummy"]]
    rows += [{"A": best, "B": k, **mcnemar(y, probs[best], probs[k])} for k in keys if k != best]
    run.save_df(pd.DataFrame(rows), "stat_mcnemar.csv")
    run.tracker.task_done(run.step, "McNemar", time.perf_counter() - t0)

    t0 = time.perf_counter()
    ci = []
    for k in keys + ["rule", "dummy"]:
        point = metrics_from_proba(y, probs[k])
        for metric, (lo, hi) in bootstrap_cis(y, probs[k], DISPLAY_METRICS, n=300 if len(y) > 5000 else 200).items():
            ci.append({"model": k, "metric": metric, "value": point[metric], "lo": lo, "hi": hi})
    run.save_df(pd.DataFrame(ci), "stat_bootstrap.csv")
    run.tracker.task_done(run.step, "bootstrap CIs", time.perf_counter() - t0)

    t0 = time.perf_counter()
    sub = run.X_test.sample(min(8_000, len(run.X_test)), random_state=C.SEED)
    run.save_df(robustness_df({k: models[k] for k in keys}, sub, run.y_test.loc[sub.index]), "robustness.csv")
    run.tracker.task_done(run.step, "robustness", time.perf_counter() - t0)

    t0 = time.perf_counter()
    run.save_df(subgroup_df(run.X_test, y, {k: probs[k] for k in keys}), "fairness.csv")
    run.tracker.task_done(run.step, "fairness", time.perf_counter() - t0)


def step_trust(run):
    models = joblib.load(run.out / "models.joblib")
    variants = load_variants(run.out)
    oof = pd.read_parquet(run.path("oof_final.parquet"))
    team_oof = pd.read_parquet(run.path("oof_team.parquet"))
    cal = oof[[m.KEY for m in MEMBERS]].assign(team=team_oof["team"].to_numpy(), y_true=oof["y_true"].to_numpy())
    run.tracker.add_tasks(run.step, 3)

    t0 = time.perf_counter()
    X_fit, X_cal = train_test_split(run.X, test_size=0.1, random_state=C.SEED)
    ood = trust.NeighbourOOD(models["knn"][:-1]).fit(X_fit, X_cal)
    run.tracker.task_done(run.step, "OOD detector", time.perf_counter() - t0)

    t0 = time.perf_counter()
    feat = models["xgb"].named_steps["features"]
    F_train, F_test = feat.transform(run.X), feat.transform(run.X_test)
    teacher = (cal["team"].to_numpy() >= 0.5).astype(int)
    teacher_test = (models["team"].predict_proba(run.X_test)[:, 1] >= 0.5).astype(int)
    surrogates = {}
    for depth in range(2, 6):
        tree = trust.fit_surrogate(F_train, teacher, depth)
        pred = tree.predict(F_test)
        surrogates[depth] = {"tree": tree, "rules": trust.extract_rules(tree, list(F_train.columns)),
                             "fidelity": float((pred == teacher_test).mean()),
                             "accuracy": float((pred == run.y_test.to_numpy()).mean()), "n_rules": int(tree.get_n_leaves())}
    run.tracker.task_done(run.step, "surrogate rules", time.perf_counter() - t0)

    t0 = time.perf_counter()
    probe = run.X_test.sample(min(1_500, len(run.X_test)), random_state=C.SEED)
    mono = {k: trust.monotonic_violation_rate(lambda X, m=models[k]: m.predict_proba(X)[:, 1], probe)
            for k in [m.KEY for m in MEMBERS] + ["team"]}
    if variants.get("xgb_mono") is not None:
        mono["xgb_mono"] = trust.monotonic_violation_rate(lambda X: variants["xgb_mono"].predict_proba(X)[:, 1], probe)
    run.tracker.task_done(run.step, "monotonicity check", time.perf_counter() - t0)
    joblib.dump({"ood": ood, "psi_ref": trust.fit_reference(run.X), "cal": cal, "surrogates": surrogates,
                 "mono": mono, "stage_metrics": run.load_df("stage_metrics.csv")}, run.out / "trust.joblib", compress=("xz", 9))
    print("  monotonic violations:", {k: round(v, 4) for k, v in mono.items()})


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--force", action="store_true", help="redo tuning even if tuned params exist")
    ap.add_argument("--smoke", action="store_true", help="tiny data + tiny budgets to test the code")
    ap.add_argument("--out", default=None, help="output folder (default artifacts/full)")
    args = ap.parse_args()

    if args.smoke:
        C.EXP.update(C.SMOKE_EXP)
    out = Path(args.out) if args.out else C.ARTIFACTS_FULL
    run = Run(out, args.smoke)
    steps = [s for s in STEPS if s in args.steps]
    est = {s: (ESTIMATES[s] / 20 if args.smoke else ESTIMATES[s]) for s in steps}
    run.tracker = Tracker(out, steps, est)
    run.tracker.log(f"Run started: steps {steps} | estimated total {fmt_secs(sum(est.values()))}")
    run.save_json({"exp": C.EXP, "smoke": args.smoke, "steps": steps, "pools": POOLS,
                   "started": time.strftime("%Y-%m-%d %H:%M:%S")}, "run_config.json")

    fns = {"baselines": step_baselines, "partA": step_partA, "missing": step_missing, "features": step_features,
           "tune": lambda r: step_tune(r, args.force), "final": step_final, "team": step_team,
           "holdout": step_holdout, "analysis": step_analysis, "trust": step_trust}
    for step in steps:
        run.step = step
        run.tracker.start_step(step)
        try:
            fns[step](run)
        except Exception as exc:
            run.tracker.log(f"ERROR in {step}: {type(exc).__name__}: {exc}")
            run.tracker.end_step(step, failed=True)
            run.tracker.finish("failed")
            raise
        run.tracker.end_step(step)
    run.tracker.finish()


if __name__ == "__main__":
    main()
