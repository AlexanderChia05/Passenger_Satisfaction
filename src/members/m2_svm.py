"""Member 2 - kernel family: SVM -> Bagging-SVM.

Exact kernel SVMs cost O(n^2) to train, so:
- the base SVM is trained on a capped stratified subsample (SubsampleFit),
- the ensemble bags several SVMs, each on `max_samples` rows (fast AND lower variance).
SVC has no native probabilities: they come from calibration (Platt / isotonic).

Advanced technique: Nystroem kernel approximation + linear SVM (scales to all
rows) and a comparison of the two calibration methods.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC, LinearSVC

from ..estimators import SubsampleFit
from .common import SEED, inner_jobs, scaled

KEY, OWNER, FAMILY = "svm", "Member 2", "Kernel"
BASE_NAME, ENSEMBLE_NAME = "SVM (RBF, 20k rows)", "Bagging-SVM"


def _svc(C=1.0, gamma="scale", kernel="rbf", degree=3, calibration="sigmoid"):
    """RBF/poly SVC wrapped in probability calibration.

    SVC outputs a margin, not a probability; ROC-AUC, soft voting, conformal sets and the
    decision policy all need probabilities, so Platt ('sigmoid') or isotonic calibration is added.
    """
    return CalibratedClassifierCV(SVC(C=C, gamma=gamma, kernel=kernel, degree=degree),
                                  method=calibration, ensemble=False)


def base():
    """Part A single model: one calibrated RBF-SVM trained on a stratified 20k-row subsample.

    Exact kernel SVM training grows ~O(n^2) in rows; 104k rows would take hours, so the
    training set is capped (prediction still covers every passenger).
    """
    return scaled(SubsampleFit(_svc(), max_rows=20_000, random_state=SEED))


def ensemble(C=1.0, gamma=0.05, kernel="rbf", degree=3, calibration="sigmoid",
             n_estimators=7, max_samples=6_000):
    """Bagging-SVM: n_estimators calibrated SVMs, each trained on max_samples random rows,
    probabilities averaged.

    Why: splits one O(n^2) problem into several small ones (fast, parallel) while the
    bags together still see far more than 20k rows; averaging also lowers variance.
    """
    return scaled(BaggingClassifier(_svc(C, gamma, kernel, degree, calibration),
                                    n_estimators=n_estimators, max_samples=max_samples,
                                    random_state=SEED, n_jobs=inner_jobs()))


def space(trial):
    """Optuna search space. C (softness of the margin) and gamma (kernel width) are searched on
    a log scale because their useful values span several orders of magnitude.
    """
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly"])
    return dict(
        C=trial.suggest_float("C", 0.05, 100, log=True),
        gamma=trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        kernel=kernel,
        degree=trial.suggest_int("degree", 2, 4) if kernel == "poly" else 3,
        calibration=trial.suggest_categorical("calibration", ["sigmoid", "isotonic"]),
        n_estimators=trial.suggest_int("n_estimators", 3, 12),
        max_samples=trial.suggest_int("max_samples", 2_000, 12_000, step=1_000),
    )


def nystroem(n_components=600, C=1.0, gamma=0.05, calibration="sigmoid"):
    """Advanced variant: Nystroem approximation of the RBF kernel + a linear SVM.

    Maps the data into an explicit 600-dimensional kernel feature space, so a linear model
    can train on ALL rows in seconds; tests whether full data beats an exact kernel on a subsample.
    """
    return scaled(CalibratedClassifierCV(LinearSVC(C=C, max_iter=5_000), method=calibration, ensemble=False),
                  Nystroem(kernel="rbf", gamma=gamma, n_components=n_components, random_state=SEED))


def advanced(params):
    """Kernel approximation on ALL rows vs bagged exact SVMs; Platt vs isotonic calibration."""
    other = "isotonic" if params.get("calibration", "sigmoid") == "sigmoid" else "sigmoid"
    return {
        "Nystroem + LinearSVC (all rows)": nystroem(C=params.get("C", 1.0), gamma=params.get("gamma", 0.05),
                                                    calibration=params.get("calibration", "sigmoid")),
        f"Bagging-SVM ({other} calibration)": ensemble(**{**params, "calibration": other}),
    }
