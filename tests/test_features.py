"""Data loading, validation and feature engineering (src/features.py)."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import AirlineFeatures, StageMask, load_raw, split_xy, validate_frame


def test_load_raw_drops_id_columns_and_keeps_schema():
    df = load_raw("test")
    assert len(df) == 25_976
    assert not set(C.DROP_COLS) & set(df.columns)
    assert set(C.FEATURE_COLS + [C.TARGET]) <= set(df.columns)


def test_load_raw_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        load_raw("validation")


def test_split_xy_encodes_target_as_0_1(test_slice):
    X, y = test_slice
    assert list(X.columns) == C.FEATURE_COLS
    assert set(np.unique(y)) <= {0, 1}


def test_split_xy_requires_target():
    with pytest.raises(KeyError):
        split_xy(pd.DataFrame({"Age": [30]}))


def test_validate_frame_accepts_real_data(test_slice):
    assert validate_frame(test_slice[0]) == []


def test_validate_frame_reports_bad_values(passenger):
    bad = passenger.copy()
    bad.loc[:, "Class"] = "First"
    bad.loc[:, "Inflight wifi service"] = 9
    bad.loc[:, "Age"] = -3
    problems = " ".join(validate_frame(bad))
    assert "Class" in problems and "Inflight wifi service" in problems and "negative" in problems


def test_validate_frame_reports_missing_columns(passenger):
    assert validate_frame(passenger.drop(columns=["Age"]))[0].startswith("Missing columns")


def test_features_are_numeric_complete_and_fold_safe(test_slice):
    X, _ = test_slice
    train, other = X.iloc[:300], X.iloc[300:]
    feat = AirlineFeatures().fit(train)
    out = feat.transform(other)
    assert len(feat.get_feature_names_out()) == 25
    assert out.shape == (len(other), 25)
    assert not out.isna().any().any(), "every missing value must be imputed"
    # imputation values come from the training fold only
    expected = train[C.RATING_COLS].replace(0, np.nan).median()
    assert feat.rating_medians_["Inflight wifi service"] == expected["Inflight wifi service"]


def test_rating_zero_is_treated_as_missing(passenger):
    p = passenger.copy()
    p.loc[:, C.RATING_COLS] = 0
    out = AirlineFeatures().fit(p.assign(**{c: 3 for c in C.RATING_COLS})).transform(p)
    assert out["n_unrated"].iloc[0] == 14
    assert (out[C.RATING_COLS] == 3).all(axis=None)


def test_missing_arrival_delay_filled_from_departure(passenger):
    p = passenger.copy()
    p.loc[:, "Departure Delay in Minutes"] = 40
    p.loc[:, "Arrival Delay in Minutes"] = np.nan
    out = AirlineFeatures().fit(p).transform(p)
    assert out["log_arr_delay"].iloc[0] == pytest.approx(np.log1p(40))


def test_stage_mask_hides_unknown_ratings(passenger):
    masked = StageMask("Booked").transform(passenger)
    known = set(C.stage_known_ratings("Booked"))
    hidden = [c for c in C.RATING_COLS if c not in known]
    assert (masked[hidden] == 0).all(axis=None)
    assert masked["Departure Delay in Minutes"].iloc[0] == 0
