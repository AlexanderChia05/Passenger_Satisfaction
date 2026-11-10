"""Pipeline helpers shared by the member modules."""
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..features import AirlineFeatures

SEED = C.SEED


def inner_jobs() -> int:
    """Threads/processes a model may use. Parallel training sets SKYPULSE_INNER_JOBS
    in each worker so workers x inner jobs matches the CPU; -1 (all cores) otherwise."""
    return int(os.environ.get("SKYPULSE_INNER_JOBS", "-1"))


def scaled(model, *extra_steps):
    """features -> StandardScaler -> [extra steps] -> model (for distance / kernel / neural models)."""
    steps = [("features", AirlineFeatures()), ("scale", StandardScaler())]
    steps += [(f"step{i}", s) for i, s in enumerate(extra_steps)]
    return Pipeline(steps + [("model", model)])


def unscaled(model):
    """features -> model (trees are scale-invariant)."""
    return Pipeline([("features", AirlineFeatures()), ("model", model)])
