"""Evaluation engine: identical CV, metrics and statistical tests for every model."""
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, learning_curve

from . import config as C

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "MCC", "Brier", "LogLoss"]
# The six headline metrics shown everywhere in the app and report (others stay computed for diagnostics).
DISPLAY_METRICS = ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]


def metrics_from_proba(y, p, threshold=0.5) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= threshold).astype(int)
    two_classes = len(np.unique(y)) > 1
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, p) if two_classes else np.nan,
        "PR-AUC": average_precision_score(y, p) if two_classes else np.nan,
        "MCC": matthews_corrcoef(y, pred),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, p, labels=[0, 1]),
    }


def make_cv(n_splits=None, seed=C.SEED):
    return StratifiedKFold(n_splits or C.EXP["cv_folds"], shuffle=True, random_state=seed)


def cv_oof(model, X, y, cv=None, name="model"):
    """K-fold CV returning out-of-fold probabilities AND per-fold metrics.

    One pass gives both the CV score and the OOF predictions later used by
    the team ensemble and conformal calibration. Returns None if the model fails.
    """
    cv = cv or make_cv()
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        t0 = time.perf_counter()
        try:
            m = clone(model).fit(X.iloc[tr], y[tr])
            fit_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            p = m.predict_proba(X.iloc[va])[:, 1]
        except Exception as exc:  # one broken model must not stop the whole experiment
            print(f"  [{name}] fold {fold} failed: {type(exc).__name__}: {exc}")
            return None
        oof[va] = p
        rows.append({"fold": fold, **metrics_from_proba(y[va], p), "fit_s": fit_s,
                     "predict_ms_per_row": (time.perf_counter() - t1) / len(va) * 1000})
    return oof, pd.DataFrame(rows)


def summarize(name, owner, stage, fold_df) -> dict:
    """Mean and std of every metric across folds -> one results row."""
    row = {"model": name, "owner": owner, "stage": stage}
    for m in METRIC_NAMES + ["fit_s", "predict_ms_per_row"]:
        row[m] = float(fold_df[m].mean())
        if m in METRIC_NAMES:
            row[f"{m}_std"] = float(fold_df[m].std(ddof=1))
    return row


# ---------------- statistical tests ----------------
def mcnemar(y, p_a, p_b, threshold=0.5) -> dict:
    """Continuity-corrected McNemar test on two models' correctness (same rows)."""
    y = np.asarray(y)
    ca, cb = (np.asarray(p_a) >= threshold) == y, (np.asarray(p_b) >= threshold) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    return {"only_A_right": n01, "only_B_right": n10, "chi2": stat, "p_value": float(stats.chi2.sf(stat, 1))}


def paired_5x2cv(model_a, model_b, X, y, scorer=f1_score, seed=C.SEED) -> dict:
    """Dietterich's 5x2cv paired t-test (more reliable than a plain paired t-test on K-fold scores)."""
    y = np.asarray(y)
    diffs = np.zeros((5, 2))
    for i in range(5):
        cv = StratifiedKFold(2, shuffle=True, random_state=seed + i)
        for j, (tr, te) in enumerate(cv.split(X, y)):
            sa = scorer(y[te], clone(model_a).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
            sb = scorer(y[te], clone(model_b).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
            diffs[i, j] = sa - sb
    var = ((diffs - diffs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t = diffs[0, 0] / np.sqrt(var.mean()) if var.mean() > 0 else 0.0
    return {"mean_diff": float(diffs.mean()), "t": float(t), "p_value": float(2 * stats.t.sf(abs(t), df=5))}


def bootstrap_ci(y, p, metric="F1", n=1000, seed=C.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = [metrics_from_proba(y[idx], p[idx])[metric] for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_cis(y, p, metrics=None, n=1000, seed=C.SEED) -> dict[str, tuple[float, float]]:
    """95% bootstrap CIs for several metrics from ONE set of resamples (same cost as a single metric)."""
    metrics = metrics or DISPLAY_METRICS
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    draws = [metrics_from_proba(y[idx], p[idx]) for idx in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {m: (float(np.percentile([d[m] for d in draws], 2.5)), float(np.percentile([d[m] for d in draws], 97.5)))
            for m in metrics}


# ---------------- diagnostics ----------------
def learning_curve_df(name, model, X, y, folds=3) -> pd.DataFrame:
    sizes, tr, va = learning_curve(model, X, y, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], cv=make_cv(folds),
                                   scoring="f1", n_jobs=1, shuffle=True, random_state=C.SEED)
    return pd.DataFrame({"model": name, "train_size": sizes, "train_f1": tr.mean(axis=1), "train_std": tr.std(axis=1),
                         "valid_f1": va.mean(axis=1), "valid_std": va.std(axis=1)})


def robustness_df(models: dict, X, y, seed=C.SEED) -> pd.DataFrame:
    """F1 under (a) Gaussian noise on numeric columns and (b) randomly hidden ratings."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    rows = []
    num_std = X[C.NUM_COLS].std()
    for level in [0.0, 0.1, 0.25, 0.5]:
        Xn = X.copy()
        noise = rng.normal(0, 1, (len(X), len(C.NUM_COLS))) * num_std.to_numpy() * level
        Xn[C.NUM_COLS] = (Xn[C.NUM_COLS].fillna(0) + noise).clip(lower=0)
        for name, m in models.items():
            rows.append({"test": "Gaussian noise on numeric (x std)", "level": level, "model": name,
                         "F1": f1_score(y, m.predict_proba(Xn)[:, 1] >= 0.5)})
    for rate in [0.0, 0.1, 0.2, 0.4]:
        Xm = X.copy()
        mask = rng.random((len(X), len(C.RATING_COLS))) < rate
        Xm[C.RATING_COLS] = Xm[C.RATING_COLS].mask(mask, 0)
        for name, m in models.items():
            rows.append({"test": "Ratings hidden (share set to N/A)", "level": rate, "model": name,
                         "F1": f1_score(y, m.predict_proba(Xm)[:, 1] >= 0.5)})
    return pd.DataFrame(rows)


def subgroup_df(X, y, probs: dict) -> pd.DataFrame:
    """Accuracy / F1 / error rates per passenger group for every model (fairness audit)."""
    y = np.asarray(y)
    groups = {"Gender": X["Gender"].astype(str), "Class": X["Class"].astype(str),
              "Type of Travel": X["Type of Travel"].astype(str), "Customer Type": X["Customer Type"].astype(str),
              "Age group": pd.cut(X["Age"], [0, 24, 39, 59, 120], labels=["<25", "25-39", "40-59", "60+"]).astype(str)}
    rows = []
    for name, p in probs.items():
        pred = (np.asarray(p) >= 0.5).astype(int)
        for attr, g in groups.items():
            for val in sorted(g.unique()):
                m = (g == val).to_numpy()
                yt, yp = y[m], pred[m]
                rows.append({"model": name, "attribute": attr, "group": val, "n": int(m.sum()),
                             "accuracy": float((yt == yp).mean()), "F1": f1_score(yt, yp, zero_division=0),
                             "false_alarm_rate": float(((yp == 0) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                             "missed_unhappy_rate": float(((yp == 1) & (yt == 0)).sum() / max((yt == 0).sum(), 1))})
    return pd.DataFrame(rows)
