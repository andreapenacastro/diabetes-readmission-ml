"""Endpoint FastAPI : risque de réadmission à 30 jours pour un patient.

Lancer depuis la racine du projet, après `python -m src.evaluate` :
    uvicorn src.serve:app --reload
puis tester sur http://127.0.0.1:8000/docs
"""
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import USED_MEDS, prepare_features
from src.paths import LOG_DIR, MODEL_DIR

# chaque prédiction est écrite dans logs/serve.log (aide IA)
logging.basicConfig(filename=LOG_DIR / "serve.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")

# modèle final (tuné + Platt) et seuil choisi sur le validation set (evaluate.py, aide IA)
model = joblib.load(MODEL_DIR / "final_model.joblib")
threshold = json.loads((MODEL_DIR / "final_config.json").read_text())["threshold"]

app = FastAPI(title="Risque de réadmission à 30 jours")


# Pydantic vérifie les données reçues (types, valeurs >= 0, aide IA)
class Patient(BaseModel):
    time_in_hospital: int = Field(5, ge=1, le=14)
    num_lab_procedures: int = Field(45, ge=0)
    num_procedures: int = Field(1, ge=0)
    num_medications: int = Field(16, ge=0)
    number_outpatient: int = Field(0, ge=0)
    number_emergency: int = Field(0, ge=0)
    number_inpatient: int = Field(1, ge=0)
    number_diagnoses: int = Field(8, ge=0)
    age: str = "[70-80)"
    gender: str = "Female"
    admission_type_id: int | None = 1
    discharge_disposition_id: int | None = 1
    admission_source_id: int | None = 7
    medical_specialty: str | None = None
    payer_code: str | None = None
    diag_1: str | None = "428"
    diag_2: str | None = "250"
    diag_3: str | None = "401"
    max_glu_serum: str | None = None
    A1Cresult: str | None = None
    change: str = "Ch"
    diabetesMed: str = "Yes"
    meds: dict[str, str] = {"insulin": "Up"}  # médicament non donné = "No"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    data = patient.model_dump()
    meds = data.pop("meds")
    for med in USED_MEDS:
        data[med] = meds.get(med, "No")
    # même préparation que pour l'entraînement (features.py, aide IA)
    X, _ = prepare_features(pd.DataFrame([data]))
    proba = float(model.predict_proba(X)[0, 1])
    alerte = proba >= threshold
    logging.info("proba=%.3f alerte=%s", proba, alerte)
    return {"proba_readmission": round(proba, 3), "seuil": round(threshold, 3), "alerte": alerte}