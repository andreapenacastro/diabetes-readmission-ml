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


# pour info (repris de evaluate_model du HW5, + PR-AUC (qu'on a fait dans la S7))
# On évalue sur validation uniquement 

def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    y_pred = model.predict(X_val)  # seuil 0.5 par défaut dans predict, on recup le vrai après 
    y_proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": name,
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),  
        "training_time_seconds": training_time,
    }


# model préprocesseur + estimateur dans une pipeline (cf. S6, consigne étape 5)
def make_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])

# Nos 3 modèles, du plus simple au plus complexe (S2 -> S5)
# Hyperparamètres qu'on a pris des cours mais seront tunés avec Optuna (tune.py)
def make_models(seed=42): 
    return {
        # baseline linéaire
        "Logistic Regression": make_pipeline(LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed, 
            # class_weight = "balanced" aussi sur RF et LightGBM c
        )),
        # bagging d'arbres 
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )),
        # boosting 
        "LightGBM": make_pipeline(LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=seed, verbosity=-1,
        )),
    }

# Étape 3 : on test avec et sans class weight pour prouver que ça aide (j'ai repris la méthode du Homework 7)
def compare_imbalance_remedy(X_train, y_train, X_val, y_val, seed=42):
    rows = []
    for weight in [None, "balanced"]:
        model = make_pipeline(LogisticRegression(max_iter=1000, class_weight=weight, random_state=seed))
        res = evaluate_model(f"LogReg class_weight={weight}", model, X_train, y_train, X_val, y_val)
        rows.append(res)
    return pd.DataFrame(rows).set_index("model")