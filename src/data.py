"""Data loading, cleaning and splitting for Diabetes 130-US Hospitals (OpenML 4541).

Run from the project root to build and save the train / validation / test sets:

    python -m src.data
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedGroupKFold

from src.paths import CONFIG_PATH, DATA_DIR

OPENML_ID = 4541
TARGET = "readmit_30"
GROUP_COL = "patient_nbr"

# The 23 diabetes medication columns (values: No / Steady / Up / Down)
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide.metformin",
    "glipizide.metformin", "glimepiride.pioglitazone", "metformin.rosiglitazone",
    "metformin.pioglitazone",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path=CONFIG_PATH) -> dict:
    """Read configs/config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
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
    df[TARGET] = (df["target"] == "<30").astype(int)
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


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def remove_impossible_readmissions(df: pd.DataFrame, exclude_ids) -> pd.DataFrame:
    """Drop encounters whose discharge code is in `exclude_ids` (EDA finding 5).

    Used for deaths (a readmission is impossible) and hospice (end-of-life
    care, where preventing a readmission is not the goal).
    """
    keep = ~df["discharge_disposition_id"].isin(exclude_ids)
    return df.loc[keep].reset_index(drop=True)


def excluded_discharge_ids(cfg: dict) -> list:
    """Death codes, plus hospice codes if the config says to exclude them."""
    ids = list(cfg["data"]["death_discharge_ids"])
    if cfg["data"].get("exclude_hospice", True):
        ids += list(cfg["data"]["hospice_discharge_ids"])
    return ids


def clean_dataset(df: pd.DataFrame, cfg: dict, verbose: bool = False) -> pd.DataFrame:
    """Apply every row filter decided in the EDA.

    1. Remove deaths (and hospice, see config) -> cannot / should not be readmitted.
    2. Remove the 3 rows with gender 'Unknown/Invalid'.
    """
    n0 = len(df)
    df = remove_impossible_readmissions(df, excluded_discharge_ids(cfg))
    n1 = len(df)
    bad_gender = df["gender"].isin(cfg["data"]["drop_gender_values"])
    df = df.loc[~bad_gender].reset_index(drop=True)
    if verbose:
        print(f"Removed {n0 - n1:,} death / hospice encounters")
        print(f"Removed {bad_gender.sum():,} rows with invalid gender")
        print(f"Rows left: {len(df):,} (from {n0:,})")
    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_cv(n_splits: int, seed: int) -> StratifiedGroupKFold:
    """Cross-validator that is grouped by patient AND stratified on the target.

    Use the same object for Optuna's cross-validation inside the training
    set, passing groups=train[GROUP_COL].
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _one_fold(df: pd.DataFrame, n_splits: int, seed: int):
    """Hold out one fold of a StratifiedGroupKFold: (rest, held_out)."""
    cv = make_cv(n_splits, seed)
    rest_idx, held_idx = next(cv.split(df, df[TARGET], groups=df[GROUP_COL]))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def split_train_val_test(df: pd.DataFrame, test_n_splits: int = 5,
                         val_n_splits: int = 5, seed: int = 42):
    """Split into train / validation / test, grouped by patient and stratified.

    - grouped: the same patient never appears in two different sets;
    - stratified: each set keeps about the same share of positives (~11%).

    GroupShuffleSplit is grouped but NOT stratified, which is why we use
    StratifiedGroupKFold and keep a single fold each time.
    """
    train_val, test = _one_fold(df, test_n_splits, seed)
    train, val = _one_fold(train_val, val_n_splits, seed)
    return train, val, test


def check_splits(splits: dict) -> pd.DataFrame:
    """Check that no patient is shared between splits and report class balance."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a][GROUP_COL]) & set(splits[b][GROUP_COL])
            assert not shared, f"{len(shared)} patients appear in both {a} and {b}"

    total = sum(len(s) for s in splits.values())
    return pd.DataFrame({
        "rows": {k: len(s) for k, s in splits.items()},
        "share_of_rows": {k: round(len(s) / total, 3) for k, s in splits.items()},
        "patients": {k: s[GROUP_COL].nunique() for k, s in splits.items()},
        "positive_rate": {k: round(s[TARGET].mean(), 4) for k, s in splits.items()},
    })


def build_splits(cfg: dict | None = None):
    """Load, clean and split the data using the settings in config.yaml."""
    cfg = cfg or load_config()
    df = clean_dataset(load_dataset(), cfg, verbose=True)
    return split_train_val_test(
        df,
        test_n_splits=cfg["split"]["test_n_splits"],
        val_n_splits=cfg["split"]["val_n_splits"],
        seed=cfg["seed"],
    )


def main():
    cfg = load_config()
    train, val, test = build_splits(cfg)
    splits = {"train": train, "val": val, "test": test}
    print(check_splits(splits))
    for name, part in splits.items():
        part.to_pickle(DATA_DIR / f"{name}.pkl")
    print(f"Saved train / val / test to {DATA_DIR}")


if __name__ == "__main__":
    main()
