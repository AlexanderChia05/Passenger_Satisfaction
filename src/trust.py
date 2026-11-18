"""Trust tooling: conformal prediction, out-of-distribution score, drift (PSI),
surrogate rule extraction and monotonicity checks."""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .features import FEATURE_LABELS


# ---------------- conformal prediction (split / inductive) ----------------
def conformal_qhat(cal_p: np.ndarray, cal_y: np.ndarray, alpha: float) -> float:
    """Threshold on the nonconformity score 1 - p(true class)."""
    scores = 1 - np.where(cal_y == 1, cal_p, 1 - cal_p)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_sets(p: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean arrays (contains_satisfied, contains_dissatisfied)."""
    return (1 - p) <= qhat, p <= qhat


# ---------------- out-of-distribution score (KNN distance) ----------------
class NeighbourOOD:
    """Mean distance to the k nearest training passengers in the scaled
    feature space, converted to a percentile of calibration distances."""

    def __init__(self, transformer, k: int = 10):
        """transformer = the KNN pipeline's preprocessing, so distances use scaled features."""
        self.transformer, self.k = transformer, k

    def fit(self, X_train, X_cal):
        """Index the training passengers and record calibration-passenger distances as reference."""
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.transformer.transform(X_train))
        self.cal_dist_ = np.sort(self.distance(X_cal))
        return self

    def distance(self, X):
        """Mean distance to the k nearest training passengers (large = unusual passenger)."""
        d, _ = self.nn_.kneighbors(self.transformer.transform(X))
        return d.mean(axis=1)

    def percentile(self, X):
        """Where a passenger's distance falls among calibration passengers (0-100)."""
        return np.searchsorted(self.cal_dist_, self.distance(X)) / len(self.cal_dist_) * 100


# ---------------- data drift: population stability index ----------------
def fit_reference(X: pd.DataFrame) -> dict:
    """Reference histograms of the training data for the Population Stability Index.

    Numeric columns use 10 quantile bins, ratings one bin per star, categories their
    share; new data is later compared against these to detect drift.
    """
    ref = {}
    for col in C.NUM_COLS + C.RATING_COLS:
        v = pd.to_numeric(X[col], errors="coerce").dropna()
        if col in C.RATING_COLS:
            edges = np.arange(-0.5, 6.5, 1.0)
        else:
            edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
            edges[0], edges[-1] = -np.inf, np.inf
        ref[col] = {"edges": edges, "p": np.histogram(v, edges)[0] / len(v)}
    for col in C.CAT_COLS + [C.CLASS_COL]:
        ref[col] = {"cats": C.CATEGORY_VALUES[col],
                    "p": X[col].astype(str).value_counts(normalize=True).reindex(C.CATEGORY_VALUES[col]).fillna(0).to_numpy()}
    return ref


def psi_table(ref: dict, X: pd.DataFrame) -> pd.DataFrame:
    """PSI per feature. < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    eps = 1e-4
    rows = []
    for col, r in ref.items():
        if "edges" in r:
            v = pd.to_numeric(X[col], errors="coerce").dropna()
            q = np.histogram(v, r["edges"])[0] / max(len(v), 1)
        else:
            q = X[col].astype(str).value_counts(normalize=True).reindex(r["cats"]).fillna(0).to_numpy()
        p = np.clip(r["p"], eps, None)
        q = np.clip(q, eps, None)
        rows.append({"feature": col, "PSI": float(np.sum((q - p) * np.log(q / p)))})
    df = pd.DataFrame(rows).sort_values("PSI", ascending=False)
    df["status"] = pd.cut(df["PSI"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate shift", "major shift"])
    return df


# ---------------- surrogate rules ----------------
def fit_surrogate(F: pd.DataFrame, teacher_pred: np.ndarray, depth: int) -> DecisionTreeClassifier:
    """Shallow decision tree that imitates a teacher model's labels (global explanation)."""
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=150, random_state=C.SEED).fit(F, teacher_pred)


def build_surrogates(F_train, teacher, F_test, teacher_test, y_test, depths=range(2, 6)) -> dict:
    """Shallow trees that imitate a teacher model's labels, one per depth, with hold-out fidelity / accuracy."""
    out = {}
    for depth in depths:
        tree = fit_surrogate(F_train, teacher, depth)
        pred = tree.predict(F_test)
        out[depth] = {"tree": tree, "rules": extract_rules(tree, list(F_train.columns)),
                      "fidelity": float((pred == teacher_test).mean()),
                      "accuracy": float((pred == np.asarray(y_test)).mean()), "n_rules": int(tree.get_n_leaves())}
    return out


def _fmt_condition(feat: str, thr: float, left: bool) -> str:
    """Turn `feature <= thr` on engineered features into readable text."""
    label = FEATURE_LABELS.get(feat, feat)
    binary = {"is_male": ("Female", "Male"), "is_loyal": ("Disloyal customer", "Loyal customer"),
              "is_business_travel": ("Personal travel", "Business travel")}
    if feat in binary:
        return binary[feat][0 if left else 1]
    if feat == "class_level":
        names = C.CLASS_ORDER
        k = int(np.floor(thr))  # thr is 0.5 or 1.5
        return ("Class is " + " / ".join(names[: k + 1])) if left else ("Class is " + " / ".join(names[k + 1:]))
    if feat in C.RATING_COLS:
        return f"{feat} ≤ {int(np.floor(thr))}" if left else f"{feat} ≥ {int(np.floor(thr)) + 1}"
    if feat.startswith("log_"):
        val = np.expm1(thr)
        unit = "mi" if "distance" in feat else "min"
        return f"{label} ≤ {val:,.0f} {unit}" if left else f"{label} > {val:,.0f} {unit}"
    return f"{label} ≤ {thr:.2f}" if left else f"{label} > {thr:.2f}"


def extract_rules(tree: DecisionTreeClassifier, feature_names) -> pd.DataFrame:
    """Turn every root-to-leaf path of the surrogate tree into a readable IF ... THEN rule."""
    t = tree.tree_
    rows = []

    def walk(node, conds):
        """Depth-first walk collecting the conditions on the path to each leaf."""
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            share = counts / counts.sum()
            cls = int(np.argmax(share))
            rows.append({"rule": " AND ".join(conds) or "(all passengers)",
                         "prediction": "Satisfied" if cls == 1 else "Dissatisfied",
                         "confidence": float(share[cls]), "support": int(t.n_node_samples[node])})
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [_fmt_condition(f, thr, True)])
        walk(t.children_right[node], conds + [_fmt_condition(f, thr, False)])

    walk(0, [])
    df = pd.DataFrame(rows)
    df["support %"] = df["support"] / df["support"].sum() * 100
    return df.sort_values("support", ascending=False).reset_index(drop=True)


# ---------------- monotonicity check ----------------
def monotonic_violation_rate(predict_fn, X: pd.DataFrame, tol: float = 0.005) -> float:
    """Share of (passenger, service) pairs where +1 star LOWERS P(satisfied)
    by more than `tol` (ignores tiny numerical wiggles)."""
    base = predict_fn(X)
    viol, total = 0, 0
    for col in C.RATING_COLS:
        mask = X[col].between(1, 4).to_numpy()
        if not mask.any():
            continue
        up = X[mask].copy()
        up[col] = up[col] + 1
        diff = predict_fn(up) - base[mask]
        viol += int((diff < -tol).sum())
        total += int(mask.sum())
    return viol / max(total, 1)
