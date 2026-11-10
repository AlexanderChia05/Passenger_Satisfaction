"""Shared tuning engine: the SAME budget for every member (fair workload).

Optuna TPE sampler (smarter than grid / random search for slow models) with
median pruning: a trial whose running mean F1 after a fold is below the median
of earlier trials is stopped early.
"""
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from . import config as C
from .evaluation import make_cv

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_member(member, X, y, n_trials, folds, timeout=None, callbacks=None):
    """Tune member.ensemble(**params) on (X, y). Returns (best_params, trials DataFrame)."""
    y = np.asarray(y)
    splits = list(make_cv(folds).split(X, y))

    def objective(trial):
        params = member.space(trial)
        scores = []
        for i, (tr, va) in enumerate(splits):
            model = member.ensemble(**params).fit(X.iloc[tr], y[tr])
            scores.append(f1_score(y[va], model.predict(X.iloc[va])))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", study_name=member.KEY,
                                sampler=optuna.samplers.TPESampler(seed=C.SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
    # Start from the hand-picked defaults so tuning can only improve on them.
    study.enqueue_trial(_defaults(member))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,), callbacks=callbacks)
    elapsed = time.perf_counter() - t0

    trials = study.trials_dataframe(attrs=("number", "value", "state", "duration", "params"))
    trials["member"] = member.KEY
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"No tuning trial completed for {member.KEY}")
    print(f"  {member.KEY}: best inner-CV F1 {study.best_value:.4f} after {len(study.trials)} trials "
          f"({sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)} pruned) in {elapsed:.0f}s")
    return study.best_params, trials


def _defaults(member):
    """Default ensemble params expressed in the search-space parameter names."""
    import inspect
    sig = inspect.signature(member.ensemble)
    space_names = _space_names(member)
    return {k: v.default for k, v in sig.parameters.items() if k in space_names}


def _space_names(member):
    """Parameter names requested by member.space(), found with a fake trial."""
    names = set()

    class Probe:
        def __getattr__(self, attr):
            def suggest(name, *args, **kwargs):
                names.add(name)
                if attr == "suggest_categorical":
                    return args[0][0] if args else kwargs["choices"][0]
                return args[0] if args else kwargs.get("low")
            return suggest

    member.space(Probe())
    return names


def best_params_table(params: dict) -> pd.DataFrame:
    return pd.DataFrame([{"member": k, "param": p, "value": v} for k, ps in params.items() for p, v in ps.items()])
