"""Training (learning) curves for the four members' ensembles, from saved results - no retraining.

    python plot_training_curves.py   ->  reports/training/learning_curves.png

Data: artifacts/full/experiments/learning_curves.csv, written by `train_full.py --steps analysis`
(sklearn learning_curve, 3 stratified folds on the 30k-row study sample, F1, tuned ensembles).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "artifacts" / "full" / "experiments" / "learning_curves.csv"
OUT = ROOT / "reports" / "training"

MODELS = [("knn", "M1  Bagging-KNN"), ("svm", "M2  Bagging-SVM"),
          ("mlp", "M3  Seed-bagged MLP"), ("xgb", "M4  XGBoost")]
VALID, TRAIN = "#2a78d6", "#7a7a76"  # validation = palette blue, training = muted ink
INK, MUTED, GRID, SURFACE = "#1f1f1d", "#6b6b66", "#e6e6e2", "#fcfcfb"


def main():
    """Draw train vs validation F1 against training-set size for the four member ensembles."""
    lc = pd.read_csv(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True, facecolor=SURFACE)
    lo = min((lc.valid_f1 - lc.valid_std).min(), (lc.train_f1 - lc.train_std).min()) - 0.01
    for ax, (key, title) in zip(axes.flat, MODELS):
        d = lc[lc.model == key].sort_values("train_size")
        x = d.train_size / 1000
        ax.set_facecolor(SURFACE)
        for col, color, ls, label in [("train_f1", TRAIN, "--", "Training"), ("valid_f1", VALID, "-", "Validation (CV)")]:
            sd = d[col.replace("f1", "std")]
            ax.fill_between(x, d[col] - sd, d[col] + sd, color=color, alpha=0.15, lw=0)
            ax.plot(x, d[col], ls, color=color, lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label=label)
        last = d.iloc[-1]
        gap = last.train_f1 - last.valid_f1
        ax.annotate(f"{last.valid_f1:.3f}", (x.iloc[-1], last.valid_f1), xytext=(0, -14),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.annotate(f"{last.train_f1:.3f}", (x.iloc[-1], last.train_f1), xytext=(0, 8),
                    textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.set_title(f"{title}\ngap at 20k rows = {gap:.3f}", loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(lo, 1.008)
        ax.set_xticks([2, 5, 10, 15, 20])
    for ax in axes[1]:
        ax.set_xlabel("Training rows (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel("F1 score")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Training curves - F1 vs training-set size (tuned ensembles)", x=0.06, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    fig.text(0.06, 0.905, "3-fold stratified CV on the 30k-row study sample; bands = ±1 std across folds.\n"
             "Validation still rising at 20k rows -> more data helps; the final models use all 103,904 rows.",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(OUT / "learning_curves.png", dpi=150, facecolor=SURFACE)
    print(f"saved {OUT / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
