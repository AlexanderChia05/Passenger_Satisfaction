"""Project overview: the problem, what SkyPulse adds beyond a standard classifier, and a demo route.

Numbers on this page are read from the saved artifacts (metrics.json), never typed by hand, so the
overview always matches the models that are actually deployed.
"""
import pandas as pd
import streamlit as st

import ui

ui.header("GATE 00 · OVERVIEW", "SkyPulse",
          "Predict which airline passengers will be dissatisfied, explain why, and choose the cheapest "
          "service fix, with an honest measure of how much each prediction can be trusted.")

m = ui.load_metrics()
models = m["models"]
best = max(ui.MEMBER_MODELS, key=lambda k: models[k]["F1"])

# ---- headline numbers ----
k = st.columns(4)
k[0].markdown(ui.kpi("Passengers", f"{m['trained_on_rows'] + m['test_rows']:,}", sub=
                     f"{m['trained_on_rows']:,} train · {m['test_rows']:,} hold-out"), unsafe_allow_html=True)
k[1].markdown(ui.kpi("Algorithms", "4 + 2", sub="member ensembles + two baselines"), unsafe_allow_html=True)
k[2].markdown(ui.kpi("Best hold-out F1", f"{models[best]['F1']:.4f}", ui.GOOD, ui.mname(best)), unsafe_allow_html=True)
k[3].markdown(ui.kpi("vs business rule", f"+{(models[best]['F1'] - models['rule']['F1']) * 100:.1f} pts", ui.ACCENT,
                     f"rule baseline F1 {models['rule']['F1']:.3f}"), unsafe_allow_html=True)

st.markdown("#### The problem")
st.markdown(
    "An airline collects post-flight satisfaction surveys but only learns a passenger was unhappy **after** the "
    "fact. The data (Kaggle *Airline Passenger Satisfaction*, 129,880 survey responses) has 4 passenger / trip "
    "attributes, 4 numeric features (age, distance, delays) and 14 service ratings; the target is *satisfied* vs "
    "*neutral or dissatisfied* (43% / 57%). The business questions are: **who** will be dissatisfied, **why**, "
    "**what is the cheapest fix**, and **how far can a prediction be trusted**.")

st.markdown("#### Beyond a standard classifier")
st.caption("Left: what a typical notebook on this dataset stops at. Right: what this system adds, and where to see it.")
rows = [
    ("One accuracy number", "Six metrics (Accuracy, Recall, Precision, F1, ROC-AUC, MCC), bootstrap 95% CIs, "
     "McNemar and 5x2cv significance tests", "Model scoreboard, Experiment lab → Statistics"),
    ("A point prediction", "Conformal prediction sets with a coverage guarantee, plus an out-of-distribution warning "
     "for unusual passengers", "Predict passenger, Trust & fairness"),
    ("'Which features matter'", "Per-passenger cheapest upgrade path (counterfactual search) and a fleet-level "
     "budget optimiser", "Service simulator, Budget planner"),
    ("Fixed 0.5 threshold", "Threshold chosen by the cost of a voucher vs a lost passenger", "Decision policy"),
    ("Black-box model", "TreeSHAP drivers per passenger, segment heatmaps and IF-THEN surrogate rules",
     "Predict passenger, Explain the models"),
    ("Unconstrained model", "Monotone XGBoost: a better service rating can never lower P(satisfied)",
     "Service simulator, Trust → Monotonic constraints"),
    ("Prediction only after the full survey", "Journey-stage early-warning models (booked → at the airport → "
     "in flight → landed)", "Trust → Early-warning accuracy"),
    ("Accuracy for everyone", "Fairness audit by gender, age, class and travel type; error analysis by segment",
     "Trust → Fairness audit, Scoreboard → Error analysis"),
    ("Manual, one-off experiments", "Automated, resumable pipeline: equal-budget Optuna tuning, parallel K-fold CV, "
     "one untouched hold-out evaluation and a feedback log for retraining", "Experiment lab, Scoreboard → Feedback log"),
]
st.dataframe(pd.DataFrame(rows, columns=["Typical approach", "SkyPulse", "Where in the app"]),
             hide_index=True, width="stretch")

st.markdown("#### Four algorithm families, one ensemble technique each")
team = pd.DataFrame([
    ("Member 1", "K-Nearest Neighbours", "Bagging (random subspace)", "NCA metric learning", "knn_base", "knn"),
    ("Member 2", "Support Vector Machine", "Bagging + calibration", "Nystroem kernel approximation", "svm_base", "svm"),
    ("Member 3", "Multi-layer Perceptron", "Seed-bagged soft vote", "Bootstrap-bagged MLP", "mlp_base", "mlp"),
    ("Member 4", "Decision Tree", "XGBoost (boosting)", "Pruning, monotone constraints, AdaBoost", "xgb_base", "xgb"),
], columns=["Owner", "Base algorithm", "Ensemble technique", "Advanced technique", "_b", "_e"])
team["Base F1"] = team["_b"].map(lambda k: models.get(k, {}).get("F1"))
team["Tuned ensemble F1"] = team["_e"].map(lambda k: models.get(k, {}).get("F1"))
st.dataframe(team.drop(columns=["_b", "_e"]).style.format({"Base F1": "{:.4f}", "Tuned ensemble F1": "{:.4f}"}),
             hide_index=True, width="stretch")
st.caption("Hold-out F1 on 25,976 unseen passengers. Every member had the same tuning budget (60 Optuna trials on "
           "the same 30k-row subsample) and the same 5-fold CV, so the comparison is fair.")

st.markdown("#### Suggested demo route")
st.markdown(
    "1. **Predict passenger** - load *Frustrated economy flyer*: verdict, conformal set, model votes, SHAP drivers, PDF.\n"
    "2. **Service simulator** - raise one rating, find the cheapest path to *satisfied*.\n"
    "3. **Budget planner / Decision policy** - turn predictions into a spending and voucher decision.\n"
    "4. **Model scoreboard** - six metrics, error analysis, McNemar duel.\n"
    "5. **Experiment lab** - Part A vs Part B evidence: tuning before vs after, statistics, learning curves.\n"
    "6. **Trust & fairness / Explain** - can we rely on it, and is it fair?")
