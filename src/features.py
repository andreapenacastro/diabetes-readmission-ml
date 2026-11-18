"""Feature engineering helpers (guidelines, step 2)."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import MED_COLS


def icd9_family(code) -> str:
    """Map an ICD-9 diagnosis code to a broad disease family (AI help).
    Groups taken from Strack et al. (2014), the paper that introduced the dataset."""
    code = str(code)
    # '?' or NaN = no diagnosis written down (EDA, finding 6)
    if code in ("?", "nan"):
        return "Missing"
    # V / E codes are not diseases (health contacts, external causes)
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
    """Add diag_1/2/3_family: hundreds of raw codes would give way too many one-hot columns."""
    df = df.copy()
    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_family"] = df[col].map(icd9_family)
    return df

# ICD-9 grouping suggested by an AI assistant, based on Strack et al. (2014).

# Étape 2 : feature engineering

# médicaments jamais prescrits (EDA d'Andrea = 9) = on supp
NEVER_USED_MEDS = [
    "citoglipton", "examide", "metformin.pioglitazone", "glimepiride.pioglitazone",
    "acetohexamide", "metformin.rosiglitazone", "troglitazone",
]
USED_MEDS = [m for m in MED_COLS if m not in NEVER_USED_MEDS]


# features trouvées grâce à l'EDA (aide IA)
def add_engineered_features(df):
    """Nos features créées (calculs ligne par ligne, pas de fuite)."""
    df = add_diag_families(df)
    # nb de traitements modifiés
    df["n_meds_changed"] = df[USED_MEDS].astype(str).isin(["Up", "Down"]).sum(axis=1)
    # total des visites passées
    df["total_prior_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    # "[70-80)" = 75 pour garder l'ordre des âges (aide IA)
    df["age_num"] = df["age"].astype(str).str.extract(r"\[(\d+)-")[0].astype(float) + 5
    return df


# list of features : patient_nbr, encounter_id et la cible ne sont pas la
NUM_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_num", "total_prior_visits", "n_meds_changed",
]
# les *_id sont des codes sans ordre donc pas de scaling 
CAT_FEATURES = [
    "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "medical_specialty", "payer_code", "diag_1_family", "diag_2_family", "diag_3_family",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
] + USED_MEDS
# pas gardés => weight (97 % manquant), diag bruts (=> familles), age (=> age_num)
# race : attribut sensible (consigne étape 10) => gardé pour l'audit fairness, pas dans le modèle (aide IA)


def prepare_features(df):
    """Renvoie X, y pour un split (train, val ou test)."""
    df = add_engineered_features(df)
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    # codes *_id : toujours "1" (pas "1.0"), quel que soit l'ordi, pour que l'API colle à l'entraînement (aide IA)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        X[col] = X[col].map(lambda v: str(int(float(v))) if pd.notna(v) else np.nan)
    # catégorielles en texte pour le OneHotEncoder, les NaN restent NaN (aide IA)
    for col in CAT_FEATURES:
        s = X[col].astype(object).replace("?", np.nan)
        X[col] = s.where(s.isna(), s.astype(str))
    y = df["readmit_30"] if "readmit_30" in df.columns else None
    return X, y


# repris du cours S6 (Pipeline + ColumnTransformer)
def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        # un manquant devient "Missing" (EDA = 6)
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES),
    ])