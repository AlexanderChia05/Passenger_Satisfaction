"""One module per team member. Every module exposes the same interface so the
training pipeline treats all four members identically (equal workload):

    KEY, OWNER, FAMILY, BASE_NAME, ENSEMBLE_NAME
    base()               -> default single model          (Part A)
    ensemble(**params)   -> ensemble version              (Part A, tuned in Part B)
    space(trial)         -> Optuna search space for ensemble()
    advanced(params)     -> {name: estimator} advanced-technique comparisons (Part B)
"""
from . import m1_knn, m2_svm, m3_mlp, m4_tree

MEMBERS = [m1_knn, m2_svm, m3_mlp, m4_tree]
BY_KEY = {m.KEY: m for m in MEMBERS}
