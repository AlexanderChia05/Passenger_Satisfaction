"""Model registry: baselines + one ensemble model per team member.

Prototype settings only (untuned). Part A/B will replace the fixed
hyperparameters below with Optuna-tuned values from the same registry.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C
from .features import AirlineFeatures, StageMask


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """Domain heuristic: business travellers in Business class are satisfied.

    predict_proba returns the empirical satisfaction rate of the rule's
    "yes" and "no" groups on the training data, so it is comparable to
    the ML models on probability metrics.
    """

    def fit(self, X, y):
        hit = self._rule(X)
        y = np.asarray(y)
        self.p_yes_ = float(y[hit].mean())
        self.p_no_ = float(y[~hit].mean())
        self.classes_ = np.array([0, 1])
        return self

    @staticmethod
    def _rule(X):
        return ((X["Type of Travel"].astype(str) == "Business travel")
                & (X["Class"].astype(str) == "Business")).to_numpy()

    def predict_proba(self, X):
        p = np.where(self._rule(X), self.p_yes_, self.p_no_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _scaled(model):
    return Pipeline([("features", AirlineFeatures()), ("scale", StandardScaler()), ("model", model)])


def build_models() -> dict:
    """Return {key: unfitted estimator}. Order = display order."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "rule": RuleBaseline(),
        # M1 - distance family: bagged KNN with random feature subspaces
        "knn": _scaled(BaggingClassifier(
            KNeighborsClassifier(n_neighbors=15, weights="distance"),
            n_estimators=5, max_samples=0.8, max_features=0.7,
            random_state=C.SEED, n_jobs=-1)),
        # M2 - kernel family: bagged RBF-SVMs, each on 20% of rows (O(n^2) cost)
        "svm": _scaled(BaggingClassifier(
            # SVC has no native probabilities: Platt-scale it (sigmoid calibration)
            CalibratedClassifierCV(SVC(C=1.0, gamma="scale"), method="sigmoid", ensemble=False),
            n_estimators=5, max_samples=0.2, random_state=C.SEED, n_jobs=-1)),
        # M3 - neural family: soft-vote of 3 MLPs with different seeds
        "mlp": _scaled(VotingClassifier(
            [(f"mlp{s}", MLPClassifier((64, 32), early_stopping=True, max_iter=300, random_state=s))
             for s in range(3)], voting="soft")),
        # M4 - tree family: gradient boosting
        "xgb": _xgb(),
    }


# Monotone constraints: a better service rating may never lower P(satisfied).
MONOTONE = {**{c: 1 for c in C.RATING_COLS}, "service_mean": 1}


def _xgb(monotone: bool = False, stage: str | None = None):
    params = dict(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.9,
                  colsample_bytree=0.9, eval_metric="logloss", random_state=C.SEED, n_jobs=-1)
    if monotone:
        params["monotone_constraints"] = MONOTONE
    steps = [("features", AirlineFeatures()), ("model", XGBClassifier(**params))]
    if stage:
        steps.insert(0, ("stage", StageMask(stage)))
    return Pipeline(steps)


def build_variants() -> dict:
    """Extra models used by trust / progressive features (not in the team vote)."""
    variants = {"xgb_mono": _xgb(monotone=True)}
    for stage in C.STAGE_ORDER[:-1]:  # "Landed" = full models
        variants[f"stage::{stage}"] = _xgb(stage=stage)
    return variants


MODEL_INFO = {
    "dummy": {"name": "Dummy", "role": "Baseline", "detail": "Always predicts the majority class"},
    "rule": {"name": "Rule", "role": "Baseline", "detail": "Business trip + Business class = satisfied"},
    "knn": {"name": "Bagged KNN", "role": "Member 1", "detail": "Distance family"},
    "svm": {"name": "Bagged SVM", "role": "Member 2", "detail": "Kernel family"},
    "mlp": {"name": "MLP Ensemble", "role": "Member 3", "detail": "Neural family"},
    "xgb": {"name": "XGBoost", "role": "Member 4", "detail": "Tree family"},
    "team": {"name": "Team Ensemble", "role": "All members", "detail": "Soft vote of the 4 member models"},
    "xgb_mono": {"name": "XGBoost (monotonic)", "role": "Member 4 variant", "detail": "Ratings can only raise P(satisfied)"},
}

BASELINES = ["dummy", "rule"]
MEMBER_MODELS = ["knn", "svm", "mlp", "xgb"]


def predict_all(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """P(satisfied) for every model + the team ensemble, one column each.

    Full artifacts ship a fitted "team" estimator (tuned weights or stacking);
    the prototype has none, so the team is a plain average of the members.
    """
    out = pd.DataFrame(index=X.index)
    for key, model in models.items():
        out[key] = model.predict_proba(X)[:, 1]
    if "team" not in models:
        out["team"] = out[MEMBER_MODELS].mean(axis=1)
    return out
