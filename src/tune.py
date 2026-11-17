"""Étapes 6 et 7 : tuning Optuna (CV sur le train) + run log.

Lancer depuis la racine du projet : python -m src.tune
"""
import joblib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.data import make_cv
from src.features import prepare_features
from src.paths import DATA_DIR, FIG_DIR, LOG_DIR, MODEL_DIR
from src.train import RESULTS_DIR, evaluate_model, make_pipeline

N_TRIALS = 20  # mettre 3 pour un test rapide
SEED = 42


# Mêmes modèles que train.py (HW5), avec les hyperparamètres à tuner.
# class_weight="balanced" reste fixe (étape 3).
def make_model(name, trial):
    if name == "Logistic Regression":
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 10, log=True),
                                  max_iter=1000, class_weight="balanced", random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200,
max_depth=trial.suggest_int("max_depth", 4, 30),
                                      min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50),
                                      class_weight="balanced", random_state=SEED, n_jobs=-1)
    return LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 600),
                          learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                          num_leaves=trial.suggest_int("num_leaves", 15, 127),
                          class_weight="balanced", random_state=SEED, verbosity=-1)
