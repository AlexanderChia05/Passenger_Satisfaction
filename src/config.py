"""Shared constants: paths, column groups, seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_FULL = ARTIFACTS / "full"

SEED = 42

# Full experiment settings (train_full.py). Equal-budget rule: every member is
# tuned on the same stratified subsample with the same number of trials/folds.
EXP = {
    "tune_rows": 30_000,     # stratified tuning subsample shared by all members
    "n_trials": 60,          # Optuna trials per member
    "inner_folds": 3,        # CV folds inside each tuning trial
    "cv_folds": 5,           # final K-fold CV on the full training set
    "study_rows": 30_000,    # rows for missing-value / feature / 5x2cv / learning-curve studies
    "weight_trials": 200,    # Optuna trials for the team soft-vote weights
}
SMOKE_EXP = {"tune_rows": 1_500, "n_trials": 3, "inner_folds": 2, "cv_folds": 2, "study_rows": 1_500,
             "weight_trials": 10}
TARGET = "satisfaction"
POSITIVE_LABEL = "satisfied"

# Identifier / index columns carry no signal and are dropped on load.
DROP_COLS = ["Unnamed: 0", "id"]

CAT_COLS = ["Gender", "Customer Type", "Type of Travel"]
CLASS_COL = "Class"
CLASS_ORDER = ["Eco", "Eco Plus", "Business"]

NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

# Survey ratings on a 1-5 scale; 0 means "not applicable / not answered".
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

FEATURE_COLS = CAT_COLS + [CLASS_COL] + NUM_COLS + RATING_COLS

CATEGORY_VALUES = {
    "Gender": ["Female", "Male"],
    "Customer Type": ["Loyal Customer", "disloyal Customer"],
    "Type of Travel": ["Business travel", "Personal Travel"],
    "Class": CLASS_ORDER,
}

# Journey stages for progressive prediction: what is known at each point.
# Each stage adds ratings / delays to the previous one.
STAGES = {
    "Booked": {"ratings": ["Ease of Online booking", "Departure/Arrival time convenient"],
               "dep_delay": False, "arr_delay": False},
    "At the airport": {"ratings": ["Online boarding", "Checkin service", "Gate location"],
                       "dep_delay": True, "arr_delay": False},
    "In flight": {"ratings": ["Inflight wifi service", "Food and drink", "Seat comfort", "Inflight entertainment",
                              "On-board service", "Leg room service", "Inflight service", "Cleanliness"],
                  "dep_delay": True, "arr_delay": False},
    "Landed": {"ratings": ["Baggage handling"], "dep_delay": True, "arr_delay": True},
}
STAGE_ORDER = list(STAGES)


def stage_known_ratings(stage: str) -> list[str]:
    """All ratings known at `stage` (cumulative)."""
    known = []
    for s in STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]:
        known += STAGES[s]["ratings"]
    return known


# Cabin names used in the UI (dataset uses Eco / Eco Plus / Business).
CABIN_NAMES = {"Business": "Business", "Eco Plus": "Premium Economy", "Eco": "Economy"}

NUM_RANGES = {
    "Age": (7, 85),
    "Flight Distance": (31, 4983),
    "Departure Delay in Minutes": (0, 1592),
    "Arrival Delay in Minutes": (0, 1584),
}
