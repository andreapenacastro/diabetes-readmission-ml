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

N_TRIALS = 30  # mettre 3 pour un test rapide
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

def main():
    train = pd.read_pickle(DATA_DIR / "train.pkl")
    val = pd.read_pickle(DATA_DIR / "val.pkl")
    X_train, y_train = prepare_features(train)
    X_val, y_val = prepare_features(val)
    cv = make_cv(5, SEED)  # CV 5 folds groupée par patient + stratifiée (data.py)

    log_rows, results = [], []
    for name in ["Logistic Regression", "Random Forest", "LightGBM"]:
        file = name.lower().replace(" ", "_")

        # Un essai = on fixe les hyperparamètres, on évalue la Pipeline complète en CV 5 folds
        # et on renvoie le score moyen (S6, slide 44). Syntaxe Optuna et cross_val_score : aide IA
        def objective(trial):
            pipe = make_pipeline(make_model(name, trial))
            scores = cross_val_score(pipe, X_train, y_train, groups=train["patient_nbr"],
                                     cv=cv, scoring="average_precision")  # PR-AUC (S7)
            log_rows.append({"model": name, "trial": trial.number, **trial.params,
                             "cv_pr_auc_mean": scores.mean(), "cv_pr_auc_std": scores.std(),
                             "artifact": f"optuna_{file}.png"})
            return scores.mean()

        # Lance la recherche des meilleurs hyperparamètres
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )

        # Exécute les essais Optuna
        study.optimize(objective, n_trials=N_TRIALS)

        # Crée la pipeline avec les meilleurs paramètres
        best = make_pipeline(make_model(name, study.best_trial))

        # Évalue le modèle sur les données train et validation
        results.append(
            evaluate_model(
                f"{name} (tuned)",
                best,
                X_train,
                y_train,
                X_val,
                y_val
            )
        )

        # Sauvegarde le modèle entraîné
        joblib.dump(best, MODEL_DIR / f"{file}_tuned.joblib")

        for row in log_rows:  # dans le run log, le meilleur essai pointe aussi vers le modèle sauvegardé
            if row["model"] == name and row["trial"] == study.best_trial.number:
                row["artifact"] += f" + {file}_tuned.joblib"

        # Graphique : score CV de chaque essai
        scores = [row["cv_pr_auc_mean"] for row in log_rows if row["model"] == name]
        plt.plot(scores, "o-")
        plt.xlabel("essai Optuna")
        plt.ylabel("PR-AUC moyenne (CV 5 folds)")
        plt.title(f"Optuna – {name}")
        plt.savefig(FIG_DIR / f"optuna_{file}.png", bbox_inches="tight")
        plt.close()

    # Run log (étape 7) : un tableau, une ligne par essai (comme le tableau du HW5)
    pd.DataFrame(log_rows).to_csv(LOG_DIR / "optuna_trials.csv", index=False)
    tuned = pd.DataFrame(results).set_index("model").sort_values("pr_auc", ascending=False)
    tuned.to_csv(RESULTS_DIR / "tuned_comparison.csv")
    print(tuned.round(4).to_string())


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
