"""Étape 10 : audit fairness sur race + une mitigation, avant / après sur le test.

Lancer depuis la racine du projet, après `python -m src.evaluate` : python -m src.fairness
"""
import json

import joblib
import pandas as pd

from src.evaluate import TARGET_RECALL, summarize, threshold_for_recall
from src.features import prepare_features
from src.paths import DATA_DIR, MODEL_DIR
from src.train import RESULTS_DIR

MIN_POSITIVES = 20  # groupe trop petit pour avoir son propre seuil -> seuil global


def load(split, model):
    df = pd.read_pickle(DATA_DIR / f"{split}.pkl")
    X, y = prepare_features(df)
    race = df["race"].astype(object).fillna("Unknown").replace("?", "Unknown")  # manquants gardés (EDA)
    return pd.DataFrame({"y": y.to_numpy(), "proba": model.predict_proba(X)[:, 1], "race": race.to_numpy()})


# Définition d'équité : equal opportunity = même recall dans chaque groupe (aide IA)
# (un patient qui sera réadmis doit avoir la même chance d'être repéré, quel que soit son groupe)
def audit(df, thresholds):
    rows = []
    for group, sub in df.groupby("race"):
        res = summarize(group, sub["y"], sub["proba"], thresholds[group])  # summarize du HW7
        rows.append({**res, "n": len(sub), "n_readmitted": int(sub["y"].sum())})
    return pd.DataFrame(rows).set_index("model")[["n", "n_readmitted", "threshold", "recall", "precision"]]