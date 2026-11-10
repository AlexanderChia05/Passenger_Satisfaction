"""Team ensemble built from the members' out-of-fold (OOF) probabilities.

Both combiners learn only from OOF predictions, so no member ever scores rows
it was trained on when the combination is learned:
- weighted soft vote: weights optimised with Optuna (maximise F1, tie-break log loss)
- stacking: shallow decision-tree meta-model (not a regression model)
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .evaluation import make_cv, metrics_from_proba


def optimise_weights(oof: np.ndarray, y, n_trials=200) -> np.ndarray:
    y = np.asarray(y)

    def objective(trial):
        w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(oof.shape[1])])
        if w.sum() == 0:
            return -1.0
        p = oof @ (w / w.sum())
        return f1_score(y, p >= 0.5) - 1e-3 * log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.enqueue_trial({f"w{i}": 1.0 for i in range(oof.shape[1])})  # equal weights as a start
    study.optimize(objective, n_trials=n_trials)
    w = np.array([study.best_params[f"w{i}"] for i in range(oof.shape[1])])
    return w / w.sum()


def fit_stack_meta(oof: np.ndarray, y) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """Choose the meta-tree depth by CV on the OOF matrix, then fit it on all OOF rows."""
    y = np.asarray(y)
    rows = []
    for depth in [2, 3, 4, 5, 6]:
        meta = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=C.SEED)
        p = cross_val_predict(meta, oof, y, cv=make_cv(), method="predict_proba")[:, 1]
        rows.append({"depth": depth, **metrics_from_proba(y, p)})
    scores = pd.DataFrame(rows)
    best = int(scores.loc[scores["F1"].idxmax(), "depth"])
    meta = DecisionTreeClassifier(max_depth=best, min_samples_leaf=50, random_state=C.SEED).fit(oof, y)
    return meta, scores
