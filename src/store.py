"""Saving / loading the model-variant dictionary.

The variants are split across two files so each stays under GitHub's 100 MB file limit:
`variants.joblib` (members' variants) and `variants_team.joblib` (team ensembles, which embed
copies of every member model). `load_variants` merges every `variants*.joblib` in the folder,
so older single-file artifacts still load.
"""
from pathlib import Path

import joblib

XZ = ("xz", 9)  # ~6x smaller than zlib here: bagged KNNs hold many near-identical copies of the data


def save_variants(variants: dict, out: Path, compress=XZ) -> None:
    """Save the variants dict as two xz-compressed files (team ensembles separately),
    keeping each file under GitHub's 100 MB limit.
    """
    out = Path(out)
    team = {k: v for k, v in variants.items() if k.startswith("team")}
    rest = {k: v for k, v in variants.items() if k not in team}
    joblib.dump(rest, out / "variants.joblib", compress=compress)
    if team:
        joblib.dump(team, out / "variants_team.joblib", compress=compress)
    else:
        (out / "variants_team.joblib").unlink(missing_ok=True)


def load_variants(out: Path) -> dict:
    """Load and merge every variants*.joblib file in `out`."""
    merged = {}
    for f in sorted(Path(out).glob("variants*.joblib")):
        merged.update(joblib.load(f))
    return merged


# Libraries whose versions decide whether the pickled models load and predict identically.
PINNED = ("scikit-learn", "xgboost", "numpy", "pandas", "scipy", "joblib", "optuna")


def environment_info() -> dict:
    """Python and library versions of the current environment (saved next to the models)."""
    import platform
    from importlib.metadata import PackageNotFoundError, version
    libs = {}
    for name in PINNED:
        try:
            libs[name] = version(name)
        except PackageNotFoundError:
            libs[name] = None
    return {"python": platform.python_version(), "libraries": libs}


def save_environment(out: Path, note: str = "") -> None:
    """Write environment.json beside the trained models."""
    import json
    info = environment_info()
    if note:
        info["note"] = note
    (Path(out) / "environment.json").write_text(json.dumps(info, indent=2))


def environment_mismatches(out: Path) -> list[str]:
    """Libraries whose installed version differs from the one recorded with the models (empty = all match)."""
    import json
    path = Path(out) / "environment.json"
    if not path.exists():
        return []
    saved = json.loads(path.read_text()).get("libraries", {})
    now = environment_info()["libraries"]
    return [f"{k} {now.get(k)} installed, models saved with {v}" for k, v in saved.items() if v and now.get(k) != v]

