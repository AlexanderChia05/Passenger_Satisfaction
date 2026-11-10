"""Member 1 - distance family: KNN -> Bagging-KNN (random subspace).

Advanced technique: NCA metric learning. Instead of a fixed Euclidean distance,
NCA learns a linear projection in which neighbours share the same label.
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from ..estimators import SubsampledNCA
from .common import SEED, inner_jobs, scaled

KEY, OWNER, FAMILY = "knn", "Member 1", "Distance"
BASE_NAME, ENSEMBLE_NAME = "KNN", "Bagging-KNN"


def base():
    return scaled(KNeighborsClassifier(n_neighbors=15, weights="distance"))


def ensemble(n_neighbors=15, weights="distance", p=2, n_estimators=10, max_samples=0.8,
             max_features=0.7, use_nca=False, nca_components=12):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    bag = BaggingClassifier(knn, n_estimators=n_estimators, max_samples=max_samples,
                            max_features=max_features, random_state=SEED, n_jobs=inner_jobs())
    extra = [SubsampledNCA(n_components=nca_components, random_state=SEED)] if use_nca else []
    return scaled(bag, *extra)


def space(trial):
    params = dict(
        n_neighbors=trial.suggest_int("n_neighbors", 3, 61, step=2),
        weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
        p=trial.suggest_categorical("p", [1, 2]),
        n_estimators=trial.suggest_int("n_estimators", 5, 20),
        max_samples=trial.suggest_float("max_samples", 0.5, 1.0),
        max_features=trial.suggest_float("max_features", 0.4, 1.0),
        use_nca=trial.suggest_categorical("use_nca", [False, True]),
    )
    if params["use_nca"]:
        params["nca_components"] = trial.suggest_int("nca_components", 6, 24)
    return params


def advanced(params):
    """Tuned ensemble with vs without NCA metric learning."""
    with_nca = {**params, "use_nca": True, "nca_components": params.get("nca_components", 12)}
    without = {k: v for k, v in params.items() if k != "nca_components"} | {"use_nca": False}
    return {"Bagging-KNN + NCA": ensemble(**with_nca), "Bagging-KNN (Euclidean)": ensemble(**without)}
