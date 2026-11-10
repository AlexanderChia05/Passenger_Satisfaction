"""Custom sklearn-compatible building blocks used by the member models.

All classes follow the sklearn API (get_params / fit / predict_proba) so they
work inside Pipeline, cross_validate, Bagging and Voting.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from xgboost import XGBClassifier


def _stratified_rows(X, y, max_rows, random_state):
    """Return (X, y) reduced to at most max_rows rows, keeping the class ratio."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=random_state)
    return Xs, ys


class SubsampleFit(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on at most `max_rows` stratified rows.

    Used for the exact RBF-SVM (O(n^2) training): the model still predicts on
    every row, only training is capped. The cap is a documented design choice.
    """

    def __init__(self, estimator=None, max_rows=20_000, random_state=42):
        self.estimator = estimator
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        Xs, ys = _stratified_rows(X, np.asarray(y), self.max_rows, self.random_state)
        self.estimator_ = clone(self.estimator).fit(Xs, ys)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


class SubsampledNCA(TransformerMixin, BaseEstimator):
    """Neighbourhood Components Analysis learned on a subsample.

    NCA learns a linear projection in which KNN classifies well (metric
    learning). Its cost is quadratic in rows, so it is fitted on `max_rows`
    stratified rows and then applied to all rows.
    """

    def __init__(self, n_components=None, max_rows=4_000, max_iter=50, random_state=42):
        self.n_components = n_components
        self.max_rows = max_rows
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        Xs, ys = _stratified_rows(X, np.asarray(y), self.max_rows, self.random_state)
        n_comp = None if self.n_components is None else min(self.n_components, np.asarray(X).shape[1])
        self.nca_ = NeighborhoodComponentsAnalysis(n_components=n_comp, max_iter=self.max_iter,
                                                   random_state=self.random_state).fit(Xs, ys)
        return self

    def transform(self, X):
        return self.nca_.transform(X)


class XGBEarlyStop(ClassifierMixin, BaseEstimator):
    """XGBoost with early stopping on an internal validation split.

    The split is taken from the data passed to fit(), i.e. from the training
    fold only, so early stopping never sees the CV validation fold (no leakage).
    """

    def __init__(self, n_estimators=1000, learning_rate=0.1, max_depth=6, subsample=0.9,
                 colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=1, gamma=0.0,
                 early_stopping_rounds=50, val_fraction=0.1, monotone_constraints=None,
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.early_stopping_rounds = early_stopping_rounds
        self.val_fraction = val_fraction
        self.monotone_constraints = monotone_constraints
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _xgb(self, **extra):
        return XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate, max_depth=self.max_depth,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight, gamma=self.gamma, eval_metric="logloss",
            monotone_constraints=self.monotone_constraints, random_state=self.random_state,
            n_jobs=self.n_jobs, **extra)

    def fit(self, X, y):
        y = np.asarray(y)
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                                  random_state=self.random_state)
        self.model_ = self._xgb(early_stopping_rounds=self.early_stopping_rounds)
        self.model_.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.best_iteration_ = int(self.model_.best_iteration)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)

    def get_booster(self):
        return self.model_.get_booster()


class PrunedTree(ClassifierMixin, BaseEstimator):
    """Decision tree whose cost-complexity pruning strength (ccp_alpha) is
    chosen automatically on an internal validation split.

    Walks the pruning path (sequence of ever-smaller subtrees), keeps the alpha
    with the best validation accuracy, then refits on all training rows.
    """

    def __init__(self, criterion="gini", min_samples_leaf=1, max_candidates=30, val_fraction=0.2,
                 random_state=42):
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.val_fraction = val_fraction
        self.random_state = random_state

    def fit(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        y = np.asarray(y)
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=self.val_fraction, stratify=y,
                                                  random_state=self.random_state)
        base = DecisionTreeClassifier(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf,
                                      random_state=self.random_state)
        alphas = base.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas
        alphas = np.unique(np.quantile(alphas[:-1], np.linspace(0, 1, self.max_candidates))) if len(alphas) > 1 else [0.0]
        scores = [clone(base).set_params(ccp_alpha=a).fit(X_tr, y_tr).score(X_va, y_va) for a in alphas]
        self.ccp_alpha_ = float(alphas[int(np.argmax(scores))])
        self.path_ = list(zip(map(float, alphas), map(float, scores)))
        self.tree_model_ = clone(base).set_params(ccp_alpha=self.ccp_alpha_).fit(X, y)
        self.classes_ = self.tree_model_.classes_
        return self

    def predict_proba(self, X):
        return self.tree_model_.predict_proba(X)

    def predict(self, X):
        return self.tree_model_.predict(X)


class PrefitStack(ClassifierMixin, BaseEstimator):
    """Stacking ensemble assembled from already-fitted members.

    The meta-model was trained on the members' out-of-fold probabilities, so
    no member ever predicted rows it was trained on when the meta-model learned.
    """

    def __init__(self, members=None, meta=None):
        self.members = members  # list of (name, fitted estimator)
        self.meta = meta        # fitted meta classifier

    def fit(self, X, y):  # members and meta are fitted by the training script
        self.classes_ = np.array([0, 1])
        return self

    def _member_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for _, m in self.members])

    def predict_proba(self, X):
        return self.meta.predict_proba(self._member_proba(X))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class WeightedVote(ClassifierMixin, BaseEstimator):
    """Soft vote of already-fitted members with tuned weights."""

    def __init__(self, members=None, weights=None):
        self.members = members
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        w = np.asarray(self.weights, dtype=float)
        p = np.column_stack([m.predict_proba(X)[:, 1] for _, m in self.members]) @ (w / w.sum())
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
