"""Fit quick, UNTUNED prototype models so the Streamlit UI has real outputs.

Not the Part A/B experiment: no CV or tuning here.
- fit set:          stratified 20k rows of train.csv
- calibration set:  10k other rows of train.csv (conformal, OOD, surrogate rules)
- hold-out:         full test.csv, scored once

Usage: python prototype_train.py
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from src import config as C
from src import trust
from src.features import load_raw, split_xy
from src.models import MEMBER_MODELS, build_models, build_variants, predict_all

FIT_ROWS, CAL_ROWS = 20_000, 10_000


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, pred),
        "Brier": brier_score_loss(y, p),
    }


def fit_all(estimators: dict, X, y) -> tuple[dict, dict]:
    fitted, secs = {}, {}
    for key, model in estimators.items():
        t0 = time.perf_counter()
        try:
            model.fit(X, y)
        except Exception as exc:  # keep going so one failure doesn't block the UI
            print(f"[skip] {key}: {exc}")
            continue
        secs[key] = time.perf_counter() - t0
        fitted[key] = model
        print(f"fit {key:24s} {secs[key]:6.1f}s")
    return fitted, secs


def main():
    C.ARTIFACTS.mkdir(exist_ok=True)
    X_train, y_train = split_xy(load_raw("train"))
    X_test, y_test = split_xy(load_raw("test"))
    X_fit, X_rest, y_fit, y_rest = train_test_split(X_train, y_train, train_size=FIT_ROWS,
                                                    stratify=y_train, random_state=C.SEED)
    X_cal, _, y_cal, _ = train_test_split(X_rest, y_rest, train_size=CAL_ROWS,
                                          stratify=y_rest, random_state=C.SEED)

    models, fit_secs = fit_all(build_models(), X_fit, y_fit)
    variants, var_secs = fit_all(build_variants(), X_fit, y_fit)

    # ---- hold-out scoring ----
    t0 = time.perf_counter()
    probs = predict_all(models, X_test)
    probs["xgb_mono"] = variants["xgb_mono"].predict_proba(X_test)[:, 1]
    print(f"scored test set in {time.perf_counter() - t0:.1f}s")
    report = {k: {**metrics(y_test.to_numpy(), probs[k].to_numpy()),
                  "fit_seconds": fit_secs.get(k, var_secs.get(k))} for k in probs.columns}
    for k, r in report.items():
        print(f"{k:9s} acc={r['Accuracy']:.4f} f1={r['F1']:.4f} auc={r['ROC-AUC']:.4f}")

    # ---- progressive (journey-stage) accuracy ----
    stage_rows = []
    for stage in C.STAGE_ORDER:
        p = probs["xgb"].to_numpy() if stage == "Landed" else variants[f"stage::{stage}"].predict_proba(X_test)[:, 1]
        m = metrics(y_test.to_numpy(), p)
        stage_rows.append({"stage": stage, "known ratings": len(C.stage_known_ratings(stage)),
                           "Accuracy": m["Accuracy"], "F1": m["F1"], "ROC-AUC": m["ROC-AUC"]})
    stage_metrics = pd.DataFrame(stage_rows)
    print(stage_metrics.round(4).to_string(index=False))

    # ---- calibration-set artifacts: conformal, OOD, surrogate ----
    cal_probs = predict_all(models, X_cal)
    cal_probs["y_true"] = y_cal.to_numpy()

    knn_prep = models["knn"][:-1]  # fitted features + scaler
    ood = trust.NeighbourOOD(knn_prep).fit(X_fit, X_cal)

    feat = models["xgb"].named_steps["features"]
    F_cal, F_test = feat.transform(X_cal), feat.transform(X_test)
    teacher_cal = (cal_probs["team"] >= 0.5).astype(int).to_numpy()
    teacher_test = (probs["team"] >= 0.5).astype(int).to_numpy()
    surrogates = {}
    for depth in range(2, 6):
        tree = trust.fit_surrogate(F_cal, teacher_cal, depth)
        pred = tree.predict(F_test)
        surrogates[depth] = {"tree": tree, "rules": trust.extract_rules(tree, list(F_cal.columns)),
                             "fidelity": float((pred == teacher_test).mean()),
                             "accuracy": float(accuracy_score(y_test, pred)),
                             "n_rules": int(tree.get_n_leaves())}
        print(f"surrogate depth {depth}: fidelity {surrogates[depth]['fidelity']:.4f}")

    # ---- monotonicity check (does +1 star ever LOWER P(satisfied)?) ----
    probe = X_test.sample(1500, random_state=C.SEED)
    mono = {
        "xgb": trust.monotonic_violation_rate(lambda X: models["xgb"].predict_proba(X)[:, 1], probe),
        "xgb_mono": trust.monotonic_violation_rate(lambda X: variants["xgb_mono"].predict_proba(X)[:, 1], probe),
        "team": trust.monotonic_violation_rate(lambda X: predict_all(models, X)["team"].to_numpy(), probe),
    }
    for k in MEMBER_MODELS[:-1]:
        mono[k] = trust.monotonic_violation_rate(lambda X, k=k: models[k].predict_proba(X)[:, 1], probe)
    print("monotonic violations:", {k: round(v, 4) for k, v in mono.items()})

    # ---- save ----
    joblib.dump(models, C.ARTIFACTS / "prototype_models.joblib", compress=3)
    joblib.dump(variants, C.ARTIFACTS / "variants.joblib", compress=3)
    joblib.dump({"ood": ood, "psi_ref": trust.fit_reference(X_fit), "cal": cal_probs,
                 "surrogates": surrogates, "mono": mono, "stage_metrics": stage_metrics},
                C.ARTIFACTS / "trust.joblib", compress=3)
    probs.assign(y_true=y_test.to_numpy()).join(X_test).to_parquet(C.ARTIFACTS / "test_predictions.parquet")

    # Small labelled CSV for demoing the Batch Scoring page.
    sample = X_test.sample(300, random_state=C.SEED)
    sample[C.TARGET] = np.where(y_test.loc[sample.index] == 1, C.POSITIVE_LABEL, "neutral or dissatisfied")
    sample.to_csv(C.ARTIFACTS / "sample_batch.csv", index=False)
    # Shifted demo file (older, long-haul, delayed passengers) to show drift detection.
    shifted = X_test[(X_test["Age"] > 50) & (X_test["Flight Distance"] > 2000)].sample(300, random_state=C.SEED).copy()
    shifted["Departure Delay in Minutes"] += 90
    shifted["Arrival Delay in Minutes"] = shifted["Arrival Delay in Minutes"].fillna(0) + 90
    shifted.to_csv(C.ARTIFACTS / "sample_batch_shifted.csv", index=False)

    (C.ARTIFACTS / "metrics.json").write_text(json.dumps(
        {"trained_on_rows": FIT_ROWS, "calibration_rows": CAL_ROWS, "test_rows": len(X_test),
         "models": report}, indent=2))


if __name__ == "__main__":
    main()
