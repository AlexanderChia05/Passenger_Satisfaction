"""Units of work executed inside parallel worker processes (see src/parallel.py).

Every function reads the training data from the worker's memory and receives
only row indices + an estimator, so tasks are cheap to send between processes.
"""
import time

import numpy as np
import optuna
from sklearn.base import clone

from . import parallel
from .evaluation import learning_curve_df, paired_5x2cv
from .progress import clear_task, fmt_secs, report_task


def fold_task(est, tr, va):
    """Fit on rows `tr`, predict probabilities for rows `va` (one CV fold)."""
    X, y = parallel.data()
    m = parallel.set_inner_jobs(clone(est))
    t0 = time.perf_counter()
    m.fit(X.iloc[tr], y[tr])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    p = m.predict_proba(X.iloc[va])[:, 1]
    return {"p": p, "fit_s": fit_s, "predict_ms_per_row": (time.perf_counter() - t1) / len(va) * 1000}


def fit_task(est, idx):
    """Fit an estimator on rows `idx` and return the fitted model."""
    X, y = parallel.data()
    m = parallel.set_inner_jobs(clone(est))
    t0 = time.perf_counter()
    m.fit(X.iloc[idx], y[idx])
    return {"model": m, "fit_s": time.perf_counter() - t0}


def tune_task(member_key, idx, n_trials, folds):
    """Optuna study for one member; writes live trial progress for the monitor."""
    from .members import BY_KEY
    from .tuning import tune_member
    X, y = parallel.data()
    out, name, t0 = parallel.out_dir(), f"tune_{member_key}", time.time()
    report_task(out, name, kind="tuning", member=member_key, done=0, total=n_trials, best=None)

    def progress(study, trial):
        done = len(study.trials)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        elapsed = time.time() - t0
        report_task(out, name, kind="tuning", member=member_key, done=done, total=n_trials,
                    best=max((t.value for t in complete), default=None),
                    pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                    elapsed=fmt_secs(elapsed), eta=fmt_secs(elapsed / done * (n_trials - done)))

    try:
        best, trials = tune_member(BY_KEY[member_key], X.iloc[idx].reset_index(drop=True), y[idx],
                                   n_trials, folds, callbacks=[progress])
    finally:
        clear_task(out, name)
    return {"best": best, "trials": trials}


def t52_task(est_a, est_b, idx):
    X, y = parallel.data()
    return paired_5x2cv(parallel.set_inner_jobs(clone(est_a)), parallel.set_inner_jobs(clone(est_b)),
                        X.iloc[idx].reset_index(drop=True), y[idx])


def lc_task(name, est, idx):
    X, y = parallel.data()
    return learning_curve_df(name, parallel.set_inner_jobs(clone(est)), X.iloc[idx].reset_index(drop=True),
                             np.asarray(y[idx]))
