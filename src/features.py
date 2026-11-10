"""Data loading, validation and the feature-engineering transformer.

All cleaning lives inside `AirlineFeatures` so it is fitted per CV fold
(no leakage) and the exact same logic runs in the Streamlit app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as C


def load_raw(split: str) -> pd.DataFrame:
    """Load train/test CSV, drop index/id columns."""
    df = pd.read_csv(C.DATA_RAW / f"{split}.csv")
    return df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])


def split_xy(df: pd.DataFrame):
    y = (df[C.TARGET] == C.POSITIVE_LABEL).astype(int)
    return df[C.FEATURE_COLS].copy(), y


def validate_frame(df: pd.DataFrame) -> list[str]:
    """Return human-readable schema problems (empty list = valid)."""
    problems = []
    missing = [c for c in C.FEATURE_COLS if c not in df.columns]
    if missing:
        return [f"Missing columns: {', '.join(missing)}"]
    for col, allowed in C.CATEGORY_VALUES.items():
        bad = set(df[col].dropna().astype(str)) - set(allowed)
        if bad:
            problems.append(f"{col}: unexpected values {sorted(bad)[:5]}")
    for col in C.NUM_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(vals.isna().sum() - df[col].isna().sum())
        if n_bad:
            problems.append(f"{col}: {n_bad} non-numeric values")
        if (vals < 0).any():
            problems.append(f"{col}: negative values found")
    for col in C.RATING_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        if (~vals.dropna().isin([0, 1, 2, 3, 4, 5])).any():
            problems.append(f"{col}: ratings must be integers 0-5")
    return problems


class AirlineFeatures(TransformerMixin, BaseEstimator):
    """Clean + engineer features, returning a numeric DataFrame.

    - Arrival delay missing -> filled with departure delay (corr 0.965).
    - Rating 0 ("not applicable") -> NaN, counted in `n_unrated`,
      then filled with the training median of that rating.
    - log1p on skewed delay / distance columns.
    - Derived: delay_recovered, service_mean.

    Parameters (for the Part B missing-value experiment):
    rating_zero: "median" (default) treats 0 as missing and imputes it;
                 "keep" leaves 0 as its own value.
    arr_fill:    "departure" (default) fills missing arrival delay with the
                 departure delay; "median" uses the training median.
    """

    def __init__(self, rating_zero: str = "median", arr_fill: str = "departure"):
        self.rating_zero = rating_zero
        self.arr_fill = arr_fill

    def fit(self, X, y=None):
        ratings = X[C.RATING_COLS].replace(0, np.nan)
        self.rating_medians_ = ratings.median().fillna(3).to_dict()
        self.arr_median_ = float(pd.to_numeric(X["Arrival Delay in Minutes"], errors="coerce").median())
        self.feature_names_out_ = self._transform(X).columns.to_numpy()
        return self

    def _transform(self, X):
        X = X.copy()
        out = pd.DataFrame(index=X.index)
        out["is_male"] = (X["Gender"].astype(str) == "Male").astype(float)
        out["is_loyal"] = (X["Customer Type"].astype(str) == "Loyal Customer").astype(float)
        out["is_business_travel"] = (X["Type of Travel"].astype(str) == "Business travel").astype(float)
        out["class_level"] = X["Class"].astype(str).map({c: i for i, c in enumerate(C.CLASS_ORDER)}).astype(float)

        dep = pd.to_numeric(X["Departure Delay in Minutes"], errors="coerce").fillna(0).clip(lower=0)
        arr = pd.to_numeric(X["Arrival Delay in Minutes"], errors="coerce")
        # getattr defaults keep transformers pickled before these params existed working
        if getattr(self, "arr_fill", "departure") == "median":
            arr = arr.fillna(getattr(self, "arr_median_", 0.0)).clip(lower=0)
        else:
            arr = arr.fillna(dep).clip(lower=0)
        out["age"] = pd.to_numeric(X["Age"], errors="coerce").astype(float)
        out["log_distance"] = np.log1p(pd.to_numeric(X["Flight Distance"], errors="coerce").astype(float))
        out["log_dep_delay"] = np.log1p(dep.astype(float))
        out["log_arr_delay"] = np.log1p(arr.astype(float))
        out["delay_recovered"] = (dep - arr).astype(float)

        raw = X[C.RATING_COLS].apply(pd.to_numeric, errors="coerce")
        ratings = raw.replace(0, np.nan)
        out["n_unrated"] = ratings.isna().sum(axis=1).astype(float)
        if getattr(self, "rating_zero", "median") == "keep":
            filled = raw.fillna(0)
        else:
            filled = ratings.fillna(self.rating_medians_ if hasattr(self, "rating_medians_") else 3)
        for col in C.RATING_COLS:
            out[col] = filled[col].astype(float)
        out["service_mean"] = filled.mean(axis=1)
        return out

    def transform(self, X):
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class DropColumns(TransformerMixin, BaseEstimator):
    """Drop engineered columns (feature-ablation experiments). A class, not a lambda,
    so it can be sent to parallel worker processes."""

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=list(self.columns))


class StageMask(TransformerMixin, BaseEstimator):
    """Hide information not yet known at a journey stage.

    Unknown ratings become 0 ("not rated") and unknown delays become 0 /
    missing, exactly as at inference time, so the stage model is trained
    on the same distribution it will see.
    """

    def __init__(self, stage: str = "Landed"):
        self.stage = stage

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        known = set(C.stage_known_ratings(self.stage))
        for col in C.RATING_COLS:
            if col not in known:
                X[col] = 0
        cfg_dep = any(C.STAGES[s]["dep_delay"] for s in C.STAGE_ORDER[: C.STAGE_ORDER.index(self.stage) + 1])
        cfg_arr = any(C.STAGES[s]["arr_delay"] for s in C.STAGE_ORDER[: C.STAGE_ORDER.index(self.stage) + 1])
        if not cfg_dep:
            X["Departure Delay in Minutes"] = 0
        if not cfg_arr:
            X["Arrival Delay in Minutes"] = np.nan  # AirlineFeatures fills it from departure delay
        return X


# Friendly labels for engineered features (used in explanation charts).
FEATURE_LABELS = {
    "is_male": "Gender",
    "is_loyal": "Customer Type",
    "is_business_travel": "Type of Travel",
    "class_level": "Class",
    "age": "Age",
    "log_distance": "Flight Distance",
    "log_dep_delay": "Departure Delay",
    "log_arr_delay": "Arrival Delay",
    "delay_recovered": "Delay recovered in-flight",
    "n_unrated": "No. of unrated services",
    "service_mean": "Average service score",
}
