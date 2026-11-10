"""Member 3 - neural family: MLP -> seed-bagged MLP ensemble.

A single network is sensitive to its random initialisation. Averaging several
networks trained with different seeds (soft vote) reduces that variance.

Advanced technique: architecture search (depth, width, activation) with early
stopping, plus bootstrap-bagged networks vs seed-only ensembles.
"""
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

from .common import SEED, inner_jobs, scaled

KEY, OWNER, FAMILY = "mlp", "Member 3", "Neural"
BASE_NAME, ENSEMBLE_NAME = "MLP", "Seed-bagged MLP"


def _net(layers=(64, 32), activation="relu", alpha=1e-4, lr=1e-3, batch=256, seed=SEED):
    return MLPClassifier(hidden_layer_sizes=layers, activation=activation, alpha=alpha,
                         learning_rate_init=lr, batch_size=batch, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10, max_iter=300, random_state=seed)


def _layers(n_layers, width, shrink):
    # shrink=True halves the width at each deeper layer (funnel), else constant width
    return tuple(max(8, width // (2 ** i)) if shrink else width for i in range(n_layers))


def base():
    return scaled(_net())


def ensemble(n_layers=2, width=64, shrink=True, activation="relu", alpha=1e-4, lr=1e-3, batch=256, n_members=5):
    layers = _layers(n_layers, width, shrink)
    nets = [(f"net{s}", _net(layers, activation, alpha, lr, batch, seed=s)) for s in range(n_members)]
    return scaled(VotingClassifier(nets, voting="soft", n_jobs=inner_jobs()))


def space(trial):
    return dict(
        n_layers=trial.suggest_int("n_layers", 1, 3),
        width=trial.suggest_categorical("width", [32, 64, 128, 256]),
        shrink=trial.suggest_categorical("shrink", [True, False]),
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        batch=trial.suggest_categorical("batch", [64, 128, 256, 512]),
        n_members=trial.suggest_int("n_members", 3, 7),
    )


def advanced(params):
    """Seed-only ensemble vs bootstrap bagging of the same architecture."""
    layers = _layers(params.get("n_layers", 2), params.get("width", 64), params.get("shrink", True))
    net = _net(layers, params.get("activation", "relu"), params.get("alpha", 1e-4), params.get("lr", 1e-3),
               params.get("batch", 256))
    return {"Bootstrap-bagged MLP": scaled(BaggingClassifier(net, n_estimators=params.get("n_members", 5),
                                                            max_samples=0.8, random_state=SEED, n_jobs=inner_jobs()))}
