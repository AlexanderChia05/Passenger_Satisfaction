"""What-if searches (src/whatif.py) and the saved, deployed models (artifacts/full)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.whatif import DEFAULT_COSTS, counterfactual, plan_budget


def rating_model(d: pd.DataFrame) -> np.ndarray:
    """Toy model: P(satisfied) = mean rating / 5, so every +1 star helps."""
    return d[C.RATING_COLS].astype(float).mean(axis=1).to_numpy() / 5


def test_counterfactual_reaches_target_with_rising_probability(passenger):
    p = passenger.copy()
    p.loc[:, C.RATING_COLS] = 2
    steps, p_final = counterfactual(rating_model, p, DEFAULT_COSTS, target=0.5, max_steps=20)
    assert p_final >= 0.5
    assert all(after > before for *_, before, after in steps)


def test_counterfactual_respects_locked_services(passenger):
    p = passenger.copy()
    p.loc[:, C.RATING_COLS] = 2
    locked = ["Gate location", "Seat comfort"]
    steps, _ = counterfactual(rating_model, p, DEFAULT_COSTS, locked=locked, target=0.6, max_steps=20)
    assert not {s[0] for s in steps} & set(locked)


@pytest.mark.parametrize("kwargs, message", [({"target": 1.5}, "target"), ({"target": 0}, "target")])
def test_counterfactual_rejects_bad_target(passenger, kwargs, message):
    with pytest.raises(ValueError, match=message):
        counterfactual(rating_model, passenger, DEFAULT_COSTS, **kwargs)


def test_counterfactual_rejects_several_passengers(test_slice):
    with pytest.raises(ValueError, match="one passenger"):
        counterfactual(rating_model, test_slice[0].head(3), DEFAULT_COSTS)


def test_plan_budget_never_overspends(test_slice):
    X = test_slice[0].head(60)
    steps, _ = plan_budget(lambda d: (rating_model(d) > 0.6).astype(float) * 0.4 + rating_model(d) * 0.5,
                           X, DEFAULT_COSTS, budget=2_000)
    spent = steps["cost (RM)"].sum() if len(steps) else 0
    assert spent <= 2_000


def test_plan_budget_rejects_negative_budget(test_slice):
    with pytest.raises(ValueError, match="negative"):
        plan_budget(rating_model, test_slice[0].head(5), DEFAULT_COSTS, budget=-1)


# ---------------------------------------------------------------- deployed artifacts
@pytest.fixture(scope="module")
def deployed():
    if not (C.ARTIFACTS_FULL / "models.joblib").exists():
        pytest.skip("full artifacts not present")
    import joblib
    return joblib.load(C.ARTIFACTS_FULL / "models.joblib")


def test_saved_models_reproduce_saved_hold_out_predictions(deployed):
    preds = pd.read_parquet(C.ARTIFACTS_FULL / "test_predictions.parquet").head(300)
    X = preds[C.FEATURE_COLS]
    for key in ("knn", "svm", "mlp", "xgb"):
        np.testing.assert_allclose(deployed[key].predict_proba(X)[:, 1], preds[key], atol=1e-6, err_msg=key)


def test_reported_hold_out_metrics_match_predictions():
    preds = pd.read_parquet(C.ARTIFACTS_FULL / "test_predictions.parquet")
    reported = json.loads((C.ARTIFACTS_FULL / "metrics.json").read_text())["models"]
    from src.evaluation import metrics_from_proba
    for key in ("knn", "svm", "mlp", "xgb"):
        assert metrics_from_proba(preds["y_true"], preds[key])["F1"] == pytest.approx(reported[key]["F1"], abs=1e-9)


def test_installed_versions_match_the_saved_models():
    """The pickled models must be used with the library versions recorded when they were saved."""
    from src.store import environment_mismatches
    assert environment_mismatches(C.ARTIFACTS_FULL) == []
