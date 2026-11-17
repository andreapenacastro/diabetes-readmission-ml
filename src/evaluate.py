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

def main():
    X_val, y_val = prepare_features(pd.read_pickle(DATA_DIR / "val.pkl"))
    X_test, y_test = prepare_features(pd.read_pickle(DATA_DIR / "test.pkl"))

    # Modèle final = le modèle tuné avec la meilleure PR-AUC sur le validation set
    tuned = pd.read_csv(RESULTS_DIR / "tuned_comparison.csv", index_col="model")
    name = tuned["pr_auc"].idxmax().replace(" (tuned)", "")
    model = joblib.load(MODEL_DIR / f"{name.lower().replace(' ', '_')}_tuned.joblib")

    # Calibrateurs appris sur le validation set : ni train, ni test (HW8)
    platt = calibrate(model, "sigmoid", X_val, y_val)
    isotonic = calibrate(model, "isotonic", X_val, y_val)

    # Pour chaque version : seuil choisi sur le validation set puis figé (S7, slide 36),
    # et évaluation sur le test, utilisé une seule fois à la fin
    versions = {"Brut": model, "Platt": platt, "Isotonic": isotonic}
    rows, probas = [], {}
    for n, m in versions.items():
        t = float(threshold_for_recall(y_val, m.predict_proba(X_val)[:, 1], TARGET_RECALL))
        probas[n] = m.predict_proba(X_test)[:, 1]
        rows.append(summarize(n, y_test, probas[n], t))
    results = pd.DataFrame(rows).set_index("model")
    results.to_csv(RESULTS_DIR / "test_final_metrics.csv")
    threshold = results.loc["Platt", "threshold"]
    print(f"Modèle final : {name} + Platt | taux réel de réadmission (test) : {y_test.mean():.3f}")
    print(results.round(4).to_string())

    # Courbe de fiabilité (HW8) : sous la diagonale = le modèle sur-estime le risque
    plt.plot([0, 1], [0, 1], "k--", label="calibration parfaite")
    for n, p in probas.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10)
        plt.plot(mean_pred, frac_pos, "o-", label=n)
    plt.xlabel("probabilité prédite")
    plt.ylabel("taux de réadmission observé")
    plt.legend()
    plt.savefig(FIG_DIR / "reliability_test.png", bbox_inches="tight")
    plt.close()

    # Modèle final (Platt) + seuil, utilisés par explain.py et fairness.py
    joblib.dump(platt, MODEL_DIR / "final_model.joblib")
    (MODEL_DIR / "final_config.json").write_text(json.dumps({"model": name, "threshold": threshold}))


if __name__ == "__main__":
    main()


