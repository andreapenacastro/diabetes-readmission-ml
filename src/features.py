"""Feature engineering helpers."""
import pandas as pd

# ajout des bibliothèques et autres pour features engineering (grace EDA d'andréa)
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import MED_COLS


def icd9_family(code) -> str:
    """Map an ICD-9 diagnosis code to a broad disease family.

    Groups follow the scheme used in Strack et al. (2014), the paper that
    introduced the dataset.
    """
    code = str(code)
    if code in ("?", "nan"):
        return "Missing"
    if code.startswith(("V", "E")):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if int(value) == 250:
        return "Diabetes"
    if 390 <= value <= 459 or int(value) == 785:
        return "Circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "Respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or int(value) == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def add_diag_families(df: pd.DataFrame) -> pd.DataFrame:
    """Add diag_1_family, diag_2_family and diag_3_family columns."""
    df = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_family"] = df[col].map(icd9_family)
    return df


# Étape 2 : feature engineering 

# médicaments jamais prescrits =9 => on enlève (décidé avec EDA d Andrea)
NEVER_USED_MEDS = [
    "citoglipton", "examide", "metformin.pioglitazone", "glimepiride.pioglitazone",
    "acetohexamide", "metformin.rosiglitazone", "troglitazone",
]
USED_MEDS = [m for m in MED_COLS if m not in NEVER_USED_MEDS]


# features trouvés grace à l'EDA : total_prior_visits et age_num proposées avec aide IA
def add_engineered_features(df):
    """Nos features créées (calculs ligne par ligne, pas de fuite)."""
    df = add_diag_families(df)
    # nb de traitements modifiés = 8
    df["n_meds_changed"] = df[USED_MEDS].astype(str).isin(["Up", "Down"]).sum(axis=1)
    # total des visites passées = 3
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    # "[70-80)" -> 75 pour garder l'ordre des âges 
    df["age_num"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
    return df 


# liste explicite des features : patient_nbr, encounter_id et la cible n'y sont pas
NUM_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "total_prior_visits", "n_meds_changed",
]
# les *_id sont des codes sans ordre -> catégories, pas de scaling (cf. S6)
CAT_FEATURES = [
    "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "payer_code", "diag_1_family", "diag_2_family", "diag_3_family",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + USED_MEDS
# pas gardés : weight (97 % manquant), diag bruts (-> familles), age (-> age_num),
# race : c'est l'attribut sensible du dataset ("All three include a genuine sensitive attribute", consigne étape 10) -> gardé pour l'audit fairness, pas mis dans le modèle
# (notre choix avec aide IA)


def prepare_features(df):
    """Renvoie X, y pour un split (train, val ou test)."""
    df = add_engineered_features(df)
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    # catégorielles en texte pour le OneHotEncoder, les NaN restent NaN (aide IA)
    for col in CAT_FEATURES:
        s = X[col].astype(object).replace("?", np.nan)
        X[col] = s.where(s.isna(), s.astype(str))
    y = df["readmit_30"] if "readmit_30" in df.columns else None
    return X, y

# Repris du cours S6 (Pipeline + ColumnTransformer) = adapté à nos colonnes
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # médiane apprise sur le train only
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        # notre choix vs S6 (most_frequent) : un manquant devient "Missing" (EDA = 6)
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES),
    ])

