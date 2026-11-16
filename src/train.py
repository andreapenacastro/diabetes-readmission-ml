"""Étapes 3, 4 et 5 : déséquilibre, 3 modèles, Pipeline.

Lancer depuis la racine du projet, après `python -m src.data` :
    python -m src.train
"""
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import load_config
from src.features import build_preprocessor, prepare_features
from src.paths import DATA_DIR, MODEL_DIR

RESULTS_DIR = MODEL_DIR.parent / "reports"


def load_splits():
    """Charge train et val (le test n'est pas touché ici)."""
    train = pd.read_pickle(DATA_DIR / "train.pkl")
    val = pd.read_pickle(DATA_DIR / "val.pkl")
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    return X_train, y_train, X_val, y_val


# Repris de evaluate_model du HW5, + PR-AUC (S7 / HW7).
# On évalue sur le validation set, pas sur le test.
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    y_pred = model.predict(X_val)  # seuil 0.5 par défaut, le vrai seuil sera choisi plus tard
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),  # baseline = prévalence (S7)
        "training_time_seconds": training_time,
    }