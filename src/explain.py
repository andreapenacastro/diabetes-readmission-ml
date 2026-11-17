"""Étape 8 : explicabilité avec SHAP (global + quelques patients).

Lancer depuis la racine du projet, après `python -m src.evaluate` : python -m src.explain
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.features import prepare_features
from src.paths import DATA_DIR, FIG_DIR, MODEL_DIR

# Consigne étape 8 : importance globale + explications locales avec SHAP.
# SHAP n'a pas encore été vu en code dans les HW -> tout le code SHAP : aide IA


def main():
    config = json.loads((MODEL_DIR / "final_config.json").read_text())
    name, threshold = config["model"], config["threshold"]
    # on explique le modèle tuné non calibré : Platt change l'échelle, pas l'ordre des patients (HW8)
    pipe = joblib.load(MODEL_DIR / f"{name.lower().replace(' ', '_')}_tuned.joblib")
    final = joblib.load(MODEL_DIR / "final_model.joblib")

    # 1000 patients du validation set (SHAP est lent)
    sample = pd.read_pickle(DATA_DIR / "val.pkl").sample(1000, random_state=42)
    X, y = prepare_features(sample)
    proba = final.predict_proba(X)[:, 1]

    # SHAP travaille sur les colonnes après le préprocesseur de la Pipeline (S6)
    pre, model = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    Xt = pre.transform(X)
    Xt = pd.DataFrame(Xt.toarray() if hasattr(Xt, "toarray") else Xt,
                      columns=pre.get_feature_names_out())
    if hasattr(model, "feature_importances_"):  # LightGBM ou Random Forest
        explainer = shap.TreeExplainer(model)
    else:  # régression logistique
        explainer = shap.Explainer(model, Xt)
    sv = explainer(Xt)
    if sv.values.ndim == 3:  # Random Forest : une colonne par classe -> on garde la classe 1
        sv = sv[:, :, 1]

    # Global : les features qui comptent le plus en moyenne
    shap.plots.bar(sv, max_display=15, show=False)
    plt.savefig(FIG_DIR / "shap_global.png", bbox_inches="tight")
    plt.close()

    # Local : un patient réadmis bien repéré et un patient réadmis raté (faux négatif)
    y = y.to_numpy()
    cases = {
        "repere": np.where((y == 1) & (proba >= threshold))[0][0],
        "rate": np.where((y == 1) & (proba < threshold))[0][0],
    }
    for label, i in cases.items():
        shap.plots.waterfall(sv[int(i)], max_display=10, show=False)
        plt.title(f"Patient {label} : proba {proba[i]:.2f} (seuil {threshold:.2f})")
        plt.savefig(FIG_DIR / f"shap_patient_{label}.png", bbox_inches="tight")
        plt.close()
    print(f"Figures SHAP sauvegardées dans {FIG_DIR}")


if __name__ == "__main__":
    main()