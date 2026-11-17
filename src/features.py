"""Feature engineering helpers (guidelines, step 2)."""
import pandas as pd


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
