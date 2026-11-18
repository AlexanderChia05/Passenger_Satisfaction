"""Recompute derived tables from saved artifacts, without retraining: stage_metrics.csv,
stat_bootstrap.csv, and trust.joblib's stage metrics and surrogate rules (teacher = XGBoost).

    python refresh_metrics.py [--out artifacts/full]
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from src import config as C
from src.evaluation import DISPLAY_METRICS, bootstrap_cis, metrics_from_proba
from src.features import load_raw, split_xy
from src.trust import build_surrogates
from src.members import MEMBERS
from src.store import load_variants


def main():
    """Rebuild stage metrics, bootstrap CIs and surrogate rules from saved models."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ARTIFACTS_FULL))
    out = Path(ap.parse_args().out)
    exp = out / "experiments"

    X_test, y_test = split_xy(load_raw("test"))
    probs = pd.read_parquet(out / "test_predictions.parquet")
    y = probs["y_true"].to_numpy()

    variants = load_variants(out)
    rows = []
    for s in C.STAGE_ORDER:
        p = probs["xgb"].to_numpy() if s == "Landed" else variants[f"stage::{s}"].predict_proba(X_test)[:, 1]
        mt = metrics_from_proba(y_test.to_numpy(), p)
        rows.append({"stage": s, "known ratings": len(C.stage_known_ratings(s)), **{m: mt[m] for m in DISPLAY_METRICS}})
    stage = pd.DataFrame(rows)
    stage.to_csv(exp / "stage_metrics.csv", index=False)
    print(stage.round(4).to_string(index=False))

    ci = []
    for k in [m.KEY for m in MEMBERS] + ["team", "rule", "dummy"]:
        point = metrics_from_proba(y, probs[k].to_numpy())
        for metric, (lo, hi) in bootstrap_cis(y, probs[k].to_numpy(), DISPLAY_METRICS, n=300).items():
            ci.append({"model": k, "metric": metric, "value": point[metric], "lo": lo, "hi": hi})
    pd.DataFrame(ci).to_csv(exp / "stat_bootstrap.csv", index=False)
    print(f"bootstrap CIs: {len(ci)} rows")

    trust_path = out / "trust.joblib"
    if trust_path.exists():
        trust = joblib.load(trust_path)
        trust["stage_metrics"] = stage
        models = joblib.load(out / "models.joblib")
        feat = models["xgb"].named_steps["features"]
        X_train, y_train = split_xy(load_raw("train"))
        teacher = (trust["cal"]["xgb"].to_numpy() >= 0.5).astype(int)
        teacher_test = (probs["xgb"].to_numpy() >= 0.5).astype(int)
        trust["surrogates"] = build_surrogates(feat.transform(X_train), teacher, feat.transform(X_test),
                                               teacher_test, y_test)
        print("surrogates:", {d: round(s["fidelity"], 4) for d, s in trust["surrogates"].items()})
        joblib.dump(trust, trust_path, compress=("xz", 9))
        print("trust.joblib stage_metrics updated")


if __name__ == "__main__":
    main()
