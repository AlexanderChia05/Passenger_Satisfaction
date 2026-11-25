"""Shared fixtures: small slices of the real Kaggle data, so tests are fast but realistic."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import load_raw, split_xy  # noqa: E402


@pytest.fixture(scope="session")
def test_slice():
    """First 400 hold-out passengers as (X, y)."""
    X, y = split_xy(load_raw("test"))
    return X.head(400).reset_index(drop=True), y.head(400).to_numpy()


@pytest.fixture
def passenger(test_slice):
    """One valid passenger as a one-row frame."""
    return test_slice[0].head(1).copy()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
