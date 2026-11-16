"""Loading the Diabetes 130-US Hospitals dataset (OpenML 4541) and building the target."""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

OPENML_ID = 4541

# The 23 diabetes medication columns (values: No / Steady / Up / Down)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide.metformin",
    "glipizide.metformin", "glimepiride.pioglitazone", "metformin.rosiglitazone",
    "metformin.pioglitazone",
]


def load_raw() -> pd.DataFrame:
    """Download (or read from the local sklearn cache) the raw dataset.

    Returns one row per hospital encounter, with the original 3-class
    target ("NO", ">30", "<30") in the column `target`.
    """
    ds = fetch_openml(data_id=OPENML_ID, as_frame=True)
    df = ds.data.copy()
    df["target"] = ds.target
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target: 1 if readmitted within 30 days, else 0."""
    df = df.copy()
    df["readmit_30"] = (df["target"] == "<30").astype(int)
    return df


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """The dataset writes missing values as '?'. Replace them with NaN."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if "?" in df[col].cat.categories:
                df[col] = df[col].cat.remove_categories("?")
        elif df[col].dtype == object:
            df[col] = df[col].replace("?", np.nan)
    return df


def load_dataset() -> pd.DataFrame:
    """Raw data + binary target. Missing values are still written as '?'."""
    return add_target(load_raw())
