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
    out = Path(out)
    team = {k: v for k, v in variants.items() if k.startswith("team")}
    rest = {k: v for k, v in variants.items() if k not in team}
    joblib.dump(rest, out / "variants.joblib", compress=compress)
    if team:
        joblib.dump(team, out / "variants_team.joblib", compress=compress)
    else:
        (out / "variants_team.joblib").unlink(missing_ok=True)


def load_variants(out: Path) -> dict:
    merged = {}
    for f in sorted(Path(out).glob("variants*.joblib")):
        merged.update(joblib.load(f))
    return merged
