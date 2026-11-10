"""Member 4 - tree family: Decision Tree -> gradient boosting (XGBoost).

Base: a single, fully grown decision tree (high variance).
Ensemble: XGBoost with early stopping on a validation split taken from the
training fold only.

Advanced technique: cost-complexity pruning of the single tree, monotone
constraints (more stars can never lower P(satisfied)) and AdaBoost as a
second boosting algorithm.
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ..estimators import PrunedTree, XGBEarlyStop
from ..models import MONOTONE
from .common import SEED, inner_jobs, unscaled

KEY, OWNER, FAMILY = "xgb", "Member 4", "Tree"
BASE_NAME, ENSEMBLE_NAME = "Decision Tree", "XGBoost"


def base():
    return unscaled(DecisionTreeClassifier(random_state=SEED))


def ensemble(learning_rate=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
             min_child_weight=1, gamma=0.0, monotone=False):
    return unscaled(XGBEarlyStop(learning_rate=learning_rate, max_depth=max_depth, subsample=subsample,
                                 colsample_bytree=colsample_bytree, reg_lambda=reg_lambda,
                                 min_child_weight=min_child_weight, gamma=gamma,
                                 monotone_constraints=MONOTONE if monotone else None, random_state=SEED,
                                 n_jobs=inner_jobs()))


def space(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )


def advanced(params):
    """Pruned single tree, monotone-constrained XGBoost, AdaBoost on stumps."""
    return {
        "Decision Tree (cost-complexity pruned)": unscaled(PrunedTree(random_state=SEED)),
        "XGBoost (monotone constraints)": ensemble(**{**params, "monotone": True}),
        "AdaBoost (depth-2 trees)": unscaled(AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                                                                n_estimators=300, learning_rate=0.5,
                                                                random_state=SEED)),
    }
