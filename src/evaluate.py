"""Étape 9 : calibration du modèle final + score final sur le test.
commandes à faire:  `python -m src.tune` puis  python -m src.evaluate
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator  # comme dans le HW8 (scikit-learn >= 1.6)
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from src.features import prepare_features
from src.paths import DATA_DIR, FIG_DIR, MODEL_DIR
from src.train import RESULTS_DIR

TARGET_RECALL = 0.70  # on veut repérer 70 % des patients qui seront réadmis (notre choix)


# vient du HW7 (exo 4) : le plus grand seuil qui garde recall >= cible
def threshold_for_recall(y_true, y_proba, target_recall=0.8):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    eligible = np.where(recall >= target_recall)[0]
    best_idx = eligible[np.argmax(thresholds[eligible])]
    return thresholds[best_idx]


# source: summarize du HW7 (exo 7), + Brier (HW8)
def summarize(name, y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "mean_proba": y_proba.mean(),
    }


# Calibration sur des données que le modèle n'a pas vues (HW8, exo 5)
def calibrate(model, method, X, y):
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X, y)
