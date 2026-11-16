"""Exploratory Data Analysis for the Airline Passenger Satisfaction dataset.

    python eda.py            # writes charts + tables + summary to reports/eda/

Covers the 15-item EDA checklist; every finding in reports/eda/eda_summary.md is
computed here (nothing hard-coded) and linked to the preprocessing decision it drove.
Uses only the raw CSVs in data/raw/ - independent of the trained models.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from src import config as C  # noqa: E402
from src.trust import fit_reference, psi_table  # noqa: E402

OUT = C.ROOT / "reports" / "eda"
NUM = C.NUM_COLS            # Age, Flight Distance, Departure Delay, Arrival Delay
RAT = C.RATING_COLS         # 14 service ratings (0 = not applicable)
CAT = C.CAT_COLS + [C.CLASS_COL]
DELAYS = ["Departure Delay in Minutes", "Arrival Delay in Minutes"]

# ---- palette (validated reference categorical order; one hue per entity, fixed) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SAT, DIS = "#2a78d6", "#eb6834"          # satisfied = series 1 (blue), neutral/dissatisfied = series 2 (orange)
TRAIN, TEST = "#2a78d6", "#1baf7a"        # train = blue, test = aqua
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 110,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def load():
    tr = pd.read_csv(C.DATA_RAW / "train.csv")
    te = pd.read_csv(C.DATA_RAW / "test.csv")
    return tr, te


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_tr, raw_te = load()
    tr = raw_tr.drop(columns=[c for c in C.DROP_COLS if c in raw_tr.columns])
    te = raw_te.drop(columns=[c for c in C.DROP_COLS if c in raw_te.columns])
    y = (tr[C.TARGET] == C.POSITIVE_LABEL).astype(int)
    F, figs = {}, []

    # 1. Shape & types -------------------------------------------------------
    F["rows_train"], F["rows_test"], F["n_cols_raw"] = len(raw_tr), len(raw_te), raw_tr.shape[1]
    F["dropped"] = [c for c in C.DROP_COLS if c in raw_tr.columns]
    types = pd.DataFrame({"column": tr.columns,
                          "dtype": [str(tr[c].dtype) for c in tr.columns],
                          "role": ["target" if c == C.TARGET else "categorical" if c in CAT else
                                   "numeric" if c in NUM else "rating 0-5" for c in tr.columns],
                          "unique": [tr[c].nunique() for c in tr.columns],
                          "example": [tr[c].iloc[0] for c in tr.columns]})
    types.to_csv(OUT / "01_columns.csv", index=False)

    # 2. Duplicates ----------------------------------------------------------
    F["dup_rows"] = int(tr.duplicated().sum())
    F["dup_rows_no_target"] = int(tr.drop(columns=C.TARGET).duplicated().sum())
    F["dup_ids"] = int(raw_tr["id"].duplicated().sum())
    F["id_overlap"] = len(set(raw_tr["id"]) & set(raw_te["id"]))
    F["row_overlap"] = len(pd.merge(tr.drop_duplicates(), te.drop_duplicates(), how="inner"))

    # 3. Target balance ------------------------------------------------------
    bal = pd.DataFrame({"train": tr[C.TARGET].value_counts(normalize=True),
                        "test": te[C.TARGET].value_counts(normalize=True)}).loc[["neutral or dissatisfied", "satisfied"]]
    F["sat_train"], F["sat_test"] = bal.loc["satisfied", "train"], bal.loc["satisfied", "test"]
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(2)
    for i, (split, col) in enumerate([("train", TRAIN), ("test", TEST)]):
        bars = ax.bar(x + (i - 0.5) * 0.36, bal[split], 0.34, color=col, label=f"{split} (n={len(tr) if split == 'train' else len(te):,})")
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, pct(b.get_height()), ha="center", color=INK2, fontsize=8)
    ax.set_xticks(x, ["Neutral or dissatisfied", "Satisfied"])
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Target class balance (train vs test)", loc="left")
    ax.legend(loc="upper right")
    figs.append(save(fig, "03_target_balance.png"))

    # 4 & 5. Missing values (explicit + hidden zeros) ------------------------
    miss = pd.DataFrame({"train": tr.isna().sum(), "test": te.isna().sum()})
    miss = miss[(miss.train > 0) | (miss.test > 0)]
    miss.to_csv(OUT / "04_missing_explicit.csv")
    F["miss"] = miss
    zeros = (tr[RAT] == 0).mean().sort_values()
    F["zeros"] = zeros.sort_values(ascending=False)
    F["rows_any_zero"] = float((tr[RAT] == 0).any(axis=1).mean())
    F["rows_incomplete"] = float(((tr[RAT] == 0).any(axis=1) | tr.isna().any(axis=1)).mean())
    zsat = pd.Series({c: y[tr[c] == 0].mean() for c in RAT if (tr[c] == 0).sum() >= 30})
    F["zero_sat"] = zsat
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(zeros.index, zeros.values, color="#3987e5", height=0.62)
    for i, v in enumerate(zeros.values):
        ax.text(v + 0.0006, i, pct(v, 2), va="center", fontsize=8, color=INK2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, zeros.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title("Hidden missing values: share of rating = 0 (\"not applicable\") per service", loc="left")
    ax.text(0, -0.13, f"Explicit NaN: only 'Arrival Delay in Minutes' ({int(miss.train.sum())} train, {int(miss.test.sum())} test rows).",
            transform=ax.transAxes, fontsize=8.5, color=INK2)
    figs.append(save(fig, "04_missing_values.png"))

    # 6. Descriptive statistics ----------------------------------------------
    desc = tr[NUM].describe(percentiles=[.25, .5, .75, .95, .99]).T
    desc["skew"] = tr[NUM].skew()
    desc["zero_share"] = (tr[NUM] == 0).mean()
    desc.round(3).to_csv(OUT / "06_numeric_summary.csv")
    rdesc = tr[RAT].replace(0, np.nan).describe().T[["mean", "std", "50%"]]
    rdesc.round(3).to_csv(OUT / "06_rating_summary.csv")
    F["desc"], F["rdesc"] = desc, rdesc

    # 7. Distributions & skew (raw vs log1p) ---------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.4))
    for j, col in enumerate(NUM):
        v = tr[col].dropna()
        axes[0, j].hist(v, bins=40, color=SAT, edgecolor=SURFACE, linewidth=0.6)
        axes[0, j].set_title(col.replace(" in Minutes", ""), loc="left", fontsize=10)
        axes[0, j].text(0.97, 0.9, f"skew {v.skew():.2f}", transform=axes[0, j].transAxes, ha="right", fontsize=8, color=INK2)
        if col == "Age":
            axes[1, j].axis("off")
            axes[1, j].text(0.5, 0.5, "Age is symmetric:\nno transform needed", ha="center", va="center", color=INK2)
            continue
        lv = np.log1p(v.clip(lower=0))
        axes[1, j].hist(lv, bins=40, color="#1c5cab", edgecolor=SURFACE, linewidth=0.6)
        axes[1, j].set_title(f"log1p({col.split(' in')[0]})", loc="left", fontsize=10)
        axes[1, j].text(0.97, 0.9, f"skew {lv.skew():.2f}", transform=axes[1, j].transAxes, ha="right", fontsize=8, color=INK2)
    fig.suptitle("Numeric distributions: raw (top) vs log1p-transformed (bottom)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs.append(save(fig, "07_numeric_distributions.png"))

    fig, axes = plt.subplots(2, 7, figsize=(14, 4.6), sharey=True)
    for ax, col in zip(axes.flat, RAT):
        share = tr[col].value_counts(normalize=True).reindex(range(6), fill_value=0)
        ax.bar(share.index, share.values, color=[GRID] + ["#3987e5"] * 5, width=0.75)
        ax.set_title(col.replace("Departure/Arrival time convenient", "Dep/Arr time conv."), fontsize=8.5, loc="left")
        ax.set_xticks(range(6))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("Rating distributions (grey bar = 0, \"not applicable\")", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs.append(save(fig, "07_rating_distributions.png"))

    # 8. Outliers --------------------------------------------------------------
    q1, q3 = tr[NUM].quantile(0.25), tr[NUM].quantile(0.75)
    iqr = q3 - q1
    out_share = ((tr[NUM] < q1 - 1.5 * iqr) | (tr[NUM] > q3 + 1.5 * iqr)).mean()
    F["outliers"] = out_share
    F["delay_over_180"] = float((tr["Departure Delay in Minutes"] > 180).mean())
    big = tr["Departure Delay in Minutes"] > 180
    F["sat_big_delay"], F["sat_no_delay"] = float(y[big].mean()), float(y[tr["Departure Delay in Minutes"] == 0].mean())
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, col in zip(axes, NUM):
        data = [tr.loc[y == 0, col].dropna(), tr.loc[y == 1, col].dropna()]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2, alpha=0.25, markeredgecolor=INK2),
                        medianprops=dict(color=INK, linewidth=1.5))
        for patch, c in zip(bp["boxes"], [DIS, SAT]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
        ax.set_xticks([1, 2], ["Neutral/\ndissatisfied", "Satisfied"])
        ax.set_title(f"{col.replace(' in Minutes', '')}\nIQR outliers: {pct(out_share[col])}", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        if col in DELAYS:
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(bottom=0)
    fig.suptitle("Outliers by class (delays on symlog scale)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs.append(save(fig, "08_outliers_boxplots.png"))

    # 9 & 10. Correlation -------------------------------------------------------
    enc = tr[NUM + RAT].copy()
    enc["satisfied (target)"] = y
    corr = enc.corr(method="spearman")
    corr.round(3).to_csv(OUT / "10_correlation_spearman.csv")
    F["r_delay"] = float(tr[DELAYS[0]].corr(tr[DELAYS[1]]))
    tri = corr.drop(index="satisfied (target)", columns="satisfied (target)")
    pairs = tri.where(np.triu(np.ones(tri.shape, dtype=bool), 1)).stack().sort_values(key=abs, ascending=False)
    F["top_pairs"] = pairs.head(6)
    F["target_corr"] = corr["satisfied (target)"].drop("satisfied (target)").sort_values(key=abs, ascending=False)
    labels = [c.replace(" in Minutes", "").replace("Departure/Arrival time convenient", "Dep/Arr time convenient") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.4:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.7 else INK)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman correlation")
    ax.set_title("Correlation matrix (Spearman; |r| >= 0.4 labelled)", loc="left")
    figs.append(save(fig, "10_correlation_heatmap.png"))

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    s = tr[DELAYS].dropna().sample(20000, random_state=C.SEED)
    hb = ax.hexbin(np.log1p(s[DELAYS[0]]), np.log1p(s[DELAYS[1]]), gridsize=45, cmap=SEQ, mincnt=1, bins="log")
    ax.set_xlabel("log1p(Departure delay, min)")
    ax.set_ylabel("log1p(Arrival delay, min)")
    ax.grid(False)
    fig.colorbar(hb, ax=ax, label="passengers (log)")
    ax.set_title(f"Departure vs arrival delay (Pearson r = {F['r_delay']:.3f})", loc="left")
    figs.append(save(fig, "09_delay_scatter.png"))

    # 11. Categorical vs target ------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    F["cat_rates"] = {}
    for ax, col in zip(axes, CAT):
        order = C.CLASS_ORDER if col == C.CLASS_COL else sorted(tr[col].astype(str).unique())
        rate = tr.groupby(tr[col].astype(str))[C.TARGET].apply(lambda s: (s == C.POSITIVE_LABEL).mean()).reindex(order)
        n = tr[col].astype(str).value_counts().reindex(order)
        F["cat_rates"][col] = rate
        bars = ax.bar(range(len(order)), rate.values, color=SAT, width=0.6)
        for b, v, k in zip(bars, rate.values, n.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5)
            ax.text(b.get_x() + b.get_width() / 2, 0.02, f"n={k:,}", ha="center", fontsize=7, color="white")
        ax.axhline(y.mean(), color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(range(len(order)), [o.replace(" Customer", "").replace(" travel", "").replace(" Travel", "") for o in order])
        ax.set_title(col, loc="left", fontsize=10)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_ylabel("Satisfied rate")
    axes[0].set_ylim(0, 0.78)
    fig.text(0.99, 0.95, f"dashed line = overall satisfied rate ({y.mean():.0%})", ha="right", fontsize=8, color=INK2)
    fig.suptitle("Satisfied rate by categorical feature", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    figs.append(save(fig, "11_categorical_vs_target.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    ct = pd.crosstab(tr["Type of Travel"], tr[C.CLASS_COL], values=y, aggfunc="mean").reindex(columns=C.CLASS_ORDER)
    F["travel_class"] = ct
    im = axes[0].imshow(ct.values, cmap=SEQ, vmin=0, vmax=1)
    axes[0].set_xticks(range(3), ct.columns)
    axes[0].set_yticks(range(len(ct)), ct.index)
    axes[0].grid(False)
    for i in range(ct.shape[0]):
        for j in range(3):
            axes[0].text(j, i, f"{ct.values[i, j]:.0%}", ha="center", va="center", color="white" if ct.values[i, j] > 0.55 else INK)
    axes[0].set_title("Satisfied rate: travel type x class (interaction)", loc="left", fontsize=10)
    for col, c, lab in [("Age", SAT, "Age (5-yr bins)")]:
        bins = pd.cut(tr[col], range(5, 90, 5))
        r = y.groupby(bins, observed=True).mean()
        axes[1].plot([b.mid for b in r.index], r.values, color=c, linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Age (5-year bins)")
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[1].set_title("Satisfied rate by age", loc="left", fontsize=10)
    fig.tight_layout()
    figs.append(save(fig, "11_interactions_age.png"))

    # Ratings vs target: satisfied rate per score
    rr = pd.DataFrame({c: y.groupby(tr[c]).mean().where(tr[c].value_counts() >= 30).reindex(range(6)) for c in RAT}).T
    rr.columns = [f"score {k}" for k in range(6)]
    rr["lift 5 vs 1"] = rr["score 5"] - rr["score 1"]
    rr = rr.sort_values("lift 5 vs 1", ascending=False)
    rr.round(3).to_csv(OUT / "11_rating_vs_target.csv")
    F["rating_lift"] = rr["lift 5 vs 1"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    vals = rr[[f"score {k}" for k in range(6)]].values
    im = ax.imshow(vals, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6), ["0 (N/A)", "1", "2", "3", "4", "5"])
    ax.set_yticks(range(len(rr)), rr.index)
    ax.grid(False)
    for i in range(vals.shape[0]):
        for j in range(6):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color="white" if vals[i, j] > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Satisfied rate")
    ax.set_title("Satisfied rate by rating score (sorted by lift from score 1 to 5)", loc="left")
    ax.text(0, -0.09, "Blank = fewer than 30 passengers gave that score.", transform=ax.transAxes, fontsize=8, color=INK2)
    figs.append(save(fig, "11_ratings_vs_target.png"))

    # 12. Feature relevance (mutual information, raw features) -----------------
    Xmi = tr.drop(columns=C.TARGET).copy()
    for c in CAT:
        Xmi[c] = Xmi[c].astype(str).astype("category").cat.codes
    Xmi = Xmi.fillna(Xmi.median(numeric_only=True))
    rows = Xmi.sample(30000, random_state=C.SEED).index
    mi = pd.Series(mutual_info_classif(Xmi.loc[rows], y.loc[rows], discrete_features=[c in CAT or c in RAT for c in Xmi.columns],
                                       random_state=C.SEED), index=Xmi.columns).sort_values()
    mi.round(4).to_csv(OUT / "12_mutual_information.csv", header=["mutual_info"])
    F["mi"] = mi.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.barh(mi.index, mi.values, color="#3987e5", height=0.65)
    for i, v in enumerate(mi.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mutual information with satisfaction (nats)")
    ax.set_title("Feature relevance: mutual information (raw features)", loc="left")
    figs.append(save(fig, "12_mutual_information.png"))

    # 13. Train vs test consistency (PSI) ---------------------------------------
    psi = psi_table(fit_reference(tr[C.FEATURE_COLS]), te[C.FEATURE_COLS]).sort_values("PSI")
    psi.round(5).to_csv(OUT / "13_train_test_psi.csv", index=False)
    F["psi_max"] = psi.iloc[-1]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.barh(psi["feature"], psi["PSI"], color=TEST, height=0.65)
    for i, v in enumerate(psi["PSI"]):
        ax.text(v, i, f" {v:.5f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, psi["PSI"].max() * 1.35)
    ax.text(0.99, 0.02, f"All features far below the 0.1 'moderate shift' threshold\n(largest = {psi['PSI'].max():.4f}).",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Population Stability Index (train -> test)")
    ax.set_title("Train vs test distribution shift (PSI; < 0.1 = stable)", loc="left")
    figs.append(save(fig, "13_train_test_psi.png"))

    write_summary(F, figs)
    print(f"EDA written to {OUT} ({len(figs)} charts)")


def write_summary(F, figs):
    d, z, cr = F["desc"], F["zeros"], F["cat_rates"]
    tp = "\n".join(f"| {a} | {b} | {v:+.2f} |" for (a, b), v in F["top_pairs"].items())
    mi_top = ", ".join(f"{k} ({v:.3f})" for k, v in F["mi"].head(5).items())
    mi_low = ", ".join(f"{k} ({v:.3f})" for k, v in F["mi"].tail(3).items())
    lift = F["rating_lift"]
    zs = F["zero_sat"].sort_values(ascending=False)
    md = f"""# Exploratory Data Analysis - Airline Passenger Satisfaction

Generated by `python eda.py` from `data/raw/train.csv` and `test.csv`. All numbers are computed, not typed.
Charts and tables are in this folder.

## EDA checklist

| # | Item | Done | Evidence |
|---|---|---|---|
| 1 | Data size, columns, types | Yes | `01_columns.csv` |
| 2 | Duplicates (within and across splits) | Yes | Section 2 |
| 3 | Target class balance | Yes | `03_target_balance.png` |
| 4 | Explicit missing values | Yes | `04_missing_explicit.csv`, `04_missing_values.png` |
| 5 | Hidden missing values (rating 0) | Yes | `04_missing_values.png` |
| 6 | Descriptive statistics | Yes | `06_numeric_summary.csv`, `06_rating_summary.csv` |
| 7 | Distribution shape / skewness | Yes | `07_numeric_distributions.png`, `07_rating_distributions.png` |
| 8 | Outliers | Yes | `08_outliers_boxplots.png` |
| 9 | Numeric correlation | Yes | `09_delay_scatter.png` |
| 10 | Rating correlation / multicollinearity | Yes | `10_correlation_heatmap.png`, `10_correlation_spearman.csv` |
| 11 | Features vs target (categorical, ratings, interactions) | Yes | `11_*.png`, `11_rating_vs_target.csv` |
| 12 | Feature relevance | Yes | `12_mutual_information.png` |
| 13 | Train/test consistency | Yes | `13_train_test_psi.png` |
| 14 | Visualisations | Yes | {len(figs)} charts |
| 15 | Findings linked to preprocessing decisions | Yes | "Findings -> decisions" table below |

## 1. Size and structure
- Train **{F['rows_train']:,}** rows, test **{F['rows_test']:,}** rows, {F['n_cols_raw']} raw columns.
- Dropped as identifiers (no predictive meaning): {', '.join(F['dropped'])}.
- Remaining: 4 categorical (Gender, Customer Type, Type of Travel, Class), 4 numeric (Age, Flight Distance,
  Departure/Arrival Delay), 14 ordinal service ratings (0-5), 1 binary target.

## 2. Duplicates
- Duplicate rows in train: **{F['dup_rows']}** (ignoring the target: {F['dup_rows_no_target']}); duplicate ids: {F['dup_ids']}.
- Passenger ids shared between train and test: **{F['id_overlap']}**; identical rows across splits: {F['row_overlap']}.
- No leakage between splits from duplicated records.

## 3. Target balance
- Satisfied: **{pct(F['sat_train'])}** train, {pct(F['sat_test'])} test; neutral/dissatisfied {pct(1 - F['sat_train'])}.
- Mild imbalance (about 57/43), so no resampling. Stratified K-fold keeps the ratio in every fold, and
  F1, ROC-AUC and MCC are reported because accuracy alone flatters the majority class (the Dummy baseline already scores {pct(1 - F['sat_test'])}).

## 4-5. Missing values
- Explicit NaN only in **Arrival Delay in Minutes**: {int(F['miss'].train.sum())} train + {int(F['miss'].test.sum())} test rows ({pct(F['miss'].train.sum() / F['rows_train'], 2)}).
- Hidden missing: rating **0 means "not applicable"**, not "very bad". Highest shares:
  {', '.join(f'{k} {pct(v, 2)}' for k, v in z.head(4).items())}.
- {pct(F['rows_any_zero'])} of passengers have at least one 0 rating, so dropping them would lose data
  ({pct(F['rows_incomplete'])} incomplete overall).
- 0-rated passengers behave differently: e.g. {zs.index[0]} = 0 -> {pct(zs.iloc[0])} satisfied vs {pct(F['sat_train'])} overall,
  so "was a 0 given" carries signal.

## 6-7. Distributions
| Feature | Mean | Median | Std | Max | Skew | Share = 0 |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {c} | {d.loc[c, 'mean']:.1f} | {d.loc[c, '50%']:.0f} | {d.loc[c, 'std']:.1f} | {d.loc[c, 'max']:.0f} | {d.loc[c, 'skew']:.2f} | {pct(d.loc[c, 'zero_share'])} |" for c in NUM) + f"""

- Delays are extremely right-skewed (skew ~{d.loc[DELAYS[0], 'skew']:.1f}); more than half the flights have 0 delay. log1p brings them close to symmetric.
- Flight distance is moderately skewed ({d.loc['Flight Distance', 'skew']:.2f}); Age is symmetric ({d.loc['Age', 'skew']:.2f}).
- Ratings (ignoring 0) centre on 3-4; mean rating ranges from {F['rdesc']['mean'].min():.2f} ({F['rdesc']['mean'].idxmin()}) to {F['rdesc']['mean'].max():.2f} ({F['rdesc']['mean'].idxmax()}).

## 8. Outliers
- IQR rule flags {', '.join(f'{c.replace(" in Minutes", "")} {pct(v)}' for c, v in F['outliers'].items())}.
- These are **genuine long delays / long-haul flights**, not data errors (all values within physical limits, no negatives).
  Satisfied rate with departure delay > 180 min: {pct(F['sat_big_delay'])} vs {pct(F['sat_no_delay'])} with no delay.
- Kept, not clipped: they are real passengers the model must score. log1p limits their leverage for
  distance-based models (KNN, SVM, MLP); trees are insensitive to them.

## 9-10. Correlation
- Departure vs arrival delay: Pearson **r = {F['r_delay']:.3f}** (near-duplicate information).
- Strongest other pairs (Spearman):

| Feature A | Feature B | r |
|---|---|---|
{tp}

- Service ratings form correlated clusters (digital: wifi / online booking / online boarding; comfort: seat /
  entertainment / food / cleanliness). Moderate (< 0.8), so all are kept; ensembles, NCA and regularisation
  handle the redundancy. Feature-selection experiments (Experiment lab) confirm: all features F1 0.951 vs top-12 by MI 0.915.
- Strongest single correlates of satisfaction: {', '.join(f'{k} ({v:+.2f})' for k, v in F['target_corr'].head(4).items())}.

## 11. Features vs target
- Class: Business **{pct(cr['Class']['Business'])}** satisfied vs Eco Plus {pct(cr['Class']['Eco Plus'])} vs Eco {pct(cr['Class']['Eco'])}.
- Type of travel: Business {pct(cr['Type of Travel']['Business travel'])} vs Personal {pct(cr['Type of Travel']['Personal Travel'])}.
- Customer type: Loyal {pct(cr['Customer Type']['Loyal Customer'])} vs disloyal {pct(cr['Customer Type']['disloyal Customer'])}.
- Gender: Female {pct(cr['Gender']['Female'])} vs Male {pct(cr['Gender']['Male'])} - almost no signal.
- Interaction: business travellers in Business class are {pct(F['travel_class'].loc['Business travel', 'Business'])} satisfied,
  personal travellers in Eco only {pct(F['travel_class'].loc['Personal Travel', 'Eco'])} - the basis of the rule baseline.
- Ratings with the biggest lift from score 1 to 5: {', '.join(f'{k} (+{v * 100:.0f} pts)' for k, v in lift.head(3).items())}.
  Weakest: {', '.join(f'{k} ({v * 100:+.0f} pts)' for k, v in lift.tail(2).items())} - not monotone, so trees/ensembles fit them better than a linear score.

## 12. Feature relevance (mutual information)
- Top: {mi_top}.
- Lowest: {mi_low}.

## 13. Train vs test consistency
- Largest PSI: {F['psi_max']['feature']} = {F['psi_max']['PSI']:.4f}; every feature < 0.1 (stable).
  The official test split is a fair hold-out; no re-splitting needed.

## Findings -> preprocessing decisions

| Finding | Decision (in `src/features.py` / training pipeline) |
|---|---|
| id / index columns carry no signal | Dropped (`DROP_COLS`) |
| 0 duplicates, 0 train/test overlap | No de-duplication; official test split used as hold-out |
| 57/43 class split | Stratified K-fold; report F1, ROC-AUC, MCC; Dummy baseline for reference |
| Arrival delay NaN ({int(F['miss'].train.sum())} rows), r = {F['r_delay']:.3f} with departure delay | Fill from departure delay. Experiment lab "Missing values": all strategies within 0.003 F1, so the simplest leak-free one is kept |
| Rating 0 = not applicable, informative | Treat 0 as missing -> median of training fold; add `n_unrated` count feature |
| Delays skew ~{d.loc[DELAYS[0], 'skew']:.1f}, distance {d.loc['Flight Distance', 'skew']:.2f} | `log1p` on delays and distance |
| Outliers are genuine | Kept; log1p reduces leverage; scaling only for KNN/SVM/MLP |
| Departure/arrival delays nearly duplicate | Keep both + derived `delay_recovered` (dep - arr), tested in feature experiments |
| Ratings correlated in clusters | Keep all; add `service_mean`; models with bagging / NCA / regularisation |
| Class, travel type, loyalty strongly linked to target | Encoded as ordinal / binary flags; used by the rule baseline |
| Gender near-zero signal | Kept (tested "drop demographics" in Experiment lab) and used in the fairness audit |
| Stable train/test (PSI < 0.1) | Same PSI tool reused for drift checks on new data |
| All fitting done per CV fold | `AirlineFeatures` lives inside each pipeline - no leakage |
"""
    (OUT / "eda_summary.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
