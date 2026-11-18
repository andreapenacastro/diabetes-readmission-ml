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

def main():
    config = json.loads((MODEL_DIR / "final_config.json").read_text())
    final = joblib.load(MODEL_DIR / "final_model.joblib")
    val, test = load("val", final), load("test", final)
    groups = sorted(test["race"].unique())

    # Avant : le même seuil pour tout le monde (evaluate.py)
    before = audit(test, {g: config["threshold"] for g in groups})

    # Mitigation : un seuil par groupe, choisi sur le validation set pour que chaque groupe
    # atteigne le même recall cible -> threshold_for_recall du HW7, groupe par groupe (idée : aide IA)
    thresholds = {}
    for g in groups:
        sub = val[val["race"] == g]
        if sub["y"].sum() >= MIN_POSITIVES:
            thresholds[g] = float(threshold_for_recall(sub["y"], sub["proba"], TARGET_RECALL))
        else:
            thresholds[g] = config["threshold"]
    after = audit(test, thresholds)

    table = before.join(after.drop(columns=["n", "n_readmitted"]), lsuffix="_avant", rsuffix="_apres")
    table.to_csv(RESULTS_DIR / "fairness_test.csv")
    big = table["n_readmitted"] >= MIN_POSITIVES  # écart calculé sur les groupes assez grands
    print(table.round(3).to_string())
    for v in ["avant", "apres"]:
        gap = table.loc[big, f"recall_{v}"].max() - table.loc[big, f"recall_{v}"].min()
        print(f"Écart de recall entre groupes ({v}) : {gap:.3f}")


if __name__ == "__main__":
    main()