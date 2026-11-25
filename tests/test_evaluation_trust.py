"""Metrics, statistical tests and trust tools (src/evaluation.py, src/trust.py)."""
import numpy as np
import pytest

from src.evaluation import DISPLAY_METRICS, bootstrap_cis, make_cv, mcnemar, metrics_from_proba
from src.trust import conformal_qhat, conformal_sets, monotonic_violation_rate


def test_perfect_predictions_score_one():
    y = np.array([0, 1, 1, 0, 1])
    m = metrics_from_proba(y, y.astype(float))
    for k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC"):
        assert m[k] == pytest.approx(1.0)


def test_majority_class_predictor_has_zero_f1_and_mcc():
    y = np.array([0] * 6 + [1] * 4)
    m = metrics_from_proba(y, np.zeros(10))
    assert m["Accuracy"] == pytest.approx(0.6)
    assert m["F1"] == 0 and m["MCC"] == 0


def test_single_class_labels_do_not_crash():
    m = metrics_from_proba(np.ones(5), np.full(5, 0.9))
    assert np.isnan(m["ROC-AUC"])


def test_display_metrics_are_the_six_reported():
    assert DISPLAY_METRICS == ["Accuracy", "Recall", "Precision", "F1", "ROC-AUC", "MCC"]


def test_stratified_cv_keeps_class_ratio(rng):
    y = (rng.random(1000) < 0.43).astype(int)
    for _, va in make_cv(5).split(np.zeros((1000, 1)), y):
        assert y[va].mean() == pytest.approx(y.mean(), abs=0.01)


def test_bootstrap_interval_contains_point_estimate(rng):
    y = (rng.random(500) < 0.5).astype(int)
    p = np.clip(y * 0.7 + rng.random(500) * 0.3, 0, 1)
    point = metrics_from_proba(y, p)
    for metric, (lo, hi) in bootstrap_cis(y, p, ["F1", "MCC"], n=200).items():
        assert lo <= point[metric] <= hi


def test_mcnemar_identical_models_not_significant(rng):
    y = (rng.random(300) < 0.5).astype(int)
    p = rng.random(300)
    res = mcnemar(y, p, p)
    assert res["p_value"] == pytest.approx(1.0)


def test_conformal_coverage_meets_target(rng):
    # calibrated probabilities: P(y=1) equals p, so the guarantee must hold on fresh data
    p_cal, p_new = rng.random(4000), rng.random(4000)
    y_cal, y_new = (rng.random(4000) < p_cal).astype(int), (rng.random(4000) < p_new).astype(int)
    for target in (0.8, 0.9, 0.95):
        q = conformal_qhat(p_cal, y_cal, 1 - target)
        has_s, has_d = conformal_sets(p_new, q)
        covered = np.where(y_new == 1, has_s, has_d).mean()
        assert covered >= target - 0.02


def test_monotonic_rate_is_zero_for_a_monotone_model(test_slice):
    X, _ = test_slice
    from src import config as C
    score = lambda d: d[C.RATING_COLS].astype(float).mean(axis=1).to_numpy() / 5  # noqa: E731
    assert monotonic_violation_rate(score, X.head(50)) == 0
