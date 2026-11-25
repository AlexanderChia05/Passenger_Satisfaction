# SkyPulse — Airline Passenger Satisfaction

BMDS2114 Machine Learning assignment. A Streamlit system that predicts whether an airline passenger
will be **satisfied** or **neutral / dissatisfied**, compares four members' algorithms (each with an
ensemble technique) against two baselines, and turns predictions into practical decisions
(service what-if, budget planning, decision policy). M4 XGBoost gives the single-passenger verdict
and every page also shows the other members' predictions. XGBoost is not the top scorer (MLP ensemble F1
0.9593 vs 0.9590; McNemar p = 0.70, 5x2cv t-test p = 0.28, not significant): it was chosen because it trains
about 18x faster, gives exact per-passenger TreeSHAP explanations and has a monotone variant.

Dataset: [Airline Passenger Satisfaction (Kaggle, teejmahal20)](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
— `train.csv` 103,904 rows, `test.csv` 25,976 rows (copies in `data/raw/`).

| Member | Base model | Ensemble technique | Advanced variant |
|---|---|---|---|
| M1 | K-Nearest Neighbours | Bagging-KNN (random subspace) | NCA metric learning |
| M2 | Support Vector Machine | Bagging-SVM (calibrated) | Nystroem + LinearSVC |
| M3 | Multi-layer Perceptron | Seed-bagged MLP soft vote | Bootstrap-bagged MLP |
| M4 | Decision Tree | XGBoost (early stopping) | Pruned tree, monotone XGBoost, AdaBoost |
| Baselines | Majority class (Dummy), Business-rule baseline | | |

## Quick start (no training needed)

The fully trained models are committed in `artifacts/full/` (compressed with xz), so the app works
straight after cloning.

```bash
git clone https://github.com/AlexanderChia05/Passenger_Satisfaction.git
cd Passenger_Satisfaction
pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Open http://localhost:8501. The sidebar should say **tuned models**.
Always run commands from the project root (not from `app/`), otherwise `No module named 'src'`.

> If the sidebar says **prototype models**, `artifacts/full/models.joblib` is missing; check the clone finished.
> Models are split into `models.joblib`, `variants.joblib`, `variants_team.joblib` and `trust.joblib`
> (each under GitHub's 100 MB limit); the first app start takes ~30 s to decompress them.

Use Python 3.13 and the exact versions pinned in `requirements.txt` (scikit-learn 1.9.1, xgboost 3.4.1, ...):
pickled models are only guaranteed to load and predict identically with the versions that saved them.
The versions used for training are recorded in `artifacts/full/environment.json`; the app shows a warning in
the sidebar if the installed versions differ. A virtual environment keeps them separate from other projects:

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## App pages

| Page | What it does |
|---|---|
| Project overview | The problem, what the system adds beyond a standard classifier, the four algorithm families and a demo route |
| Predict passenger | Enter a passenger (or a preset) and pick the cabin; XGBoost gives the verdict and the other members show whether they agree, with 3D seat map, uncertainty (conformal), out-of-distribution check, drivers and a PDF report |
| Service simulator | What-if: change service ratings, see the satisfaction probability move; cheapest upgrade path |
| Budget planner | Spend a fleet budget on service upgrades for the largest gain in satisfied passengers |
| Decision policy | Pick a probability threshold for an intervention, see costs and errors |
| Trust & fairness | Conformal coverage, fairness by passenger group, monotonic checks, early-warning accuracy by journey stage |
| Explain the models | Feature importance / SHAP, segment heatmap, surrogate decision rules |
| Model scoreboard | Hold-out metrics (Accuracy, Recall, Precision, F1, ROC-AUC, MCC), ROC/PR curves, error analysis (confusion matrix, errors by segment, error overlap between algorithms), McNemar model duel |
| Experiment lab | All training evidence: baselines, missing-value and feature experiments, tuning before vs after, CV, statistical tests, learning curves, robustness |

## Project layout

```
app/                Streamlit UI (streamlit_app.py = entry, views/ = pages, ui.py = shared helpers)
src/                ML code: features, member models (src/members/m1..m4), tuning, evaluation, trust, what-if
data/raw/           Kaggle train.csv / test.csv
artifacts/full/     Trained models + experiment results used by the app
artifacts/          Small prototype models (fallback)
train_full.py       Full training pipeline (parallel, with progress tracker)
train_status.py     Live training progress in the terminal
refresh_metrics.py  Recompute stage metrics, bootstrap CIs and surrogate rules from saved models (no retraining)
prototype_train.py  Quick prototype training (~1 min)
eda.py              Exploratory data analysis -> reports/eda/ (charts, tables, eda_summary.md)
plot_training_curves.py  Learning curves of the four ensembles -> reports/training/
tests/              pytest suite (features, metrics, conformal, what-if, saved models)
```

## Tests

```bash
python -m pytest
```

30 tests, about 15 s. They cover data loading and validation, leak-free feature engineering, the metrics
and statistical tests, the conformal coverage guarantee, the what-if searches, and that the committed
models still reproduce the saved hold-out predictions.

## Retraining (optional, ~3 hours on 8 cores)

```bash
python train_full.py            # all steps: baselines, partA, missing, features, tune, final, team, holdout, analysis, trust
python train_status.py          # watch progress in another terminal
python train_full.py --smoke    # 5-minute end-to-end check on a tiny sample
python train_full.py --steps holdout analysis trust   # resume / rerun selected steps
```

Equal budget for every member: 60 Optuna trials on a 30k-row subsample, 3 inner folds, then 5-fold CV
on the full training set and a final hold-out test. The pipeline's `team` step still builds a soft-vote /
stacking experiment across members; the app does not use it, since each member already has its own ensemble.

## Hold-out results (test set, 25,976 passengers)

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| M3 MLP ensemble | 0.9649 | 0.9435 | 0.9756 | 0.9593 | 0.9955 | 0.9288 |
| M4 XGBoost | 0.9645 | 0.9453 | 0.9732 | 0.9590 | 0.9955 | 0.9281 |
| M2 Bagging-SVM | 0.9521 | 0.9301 | 0.9596 | 0.9446 | 0.9902 | 0.9028 |
| M1 Bagging-KNN | 0.9449 | 0.9241 | 0.9489 | 0.9364 | 0.9878 | 0.8880 |
| Rule baseline | 0.7649 | 0.7574 | 0.7210 | 0.7388 | 0.7641 | 0.5257 |
| Dummy baseline | 0.5610 | 0 | 0 | 0 | 0.5 | 0 |
