"""Stat tests + confidence intervals to back our EDA findings with numbers (AI help).
Not seen in class: an AI assistant suggested them so we could analyse the data properly."""
import numpy as np
import pandas as pd
from scipy import stats


def lump_rare(s: pd.Series, min_count: int = 100, other: str = "Other") -> pd.Series:
    """Group categories with < `min_count` rows into 'Other' (AI help).
    Chi-square isn't reliable below ~5 expected per cell, and some codes have only 2-6 rows."""
    counts = s.value_counts()
    rare = counts[counts < min_count].index
    return s.astype(object).where(~s.isin(rare), other)


def wilson_ci(successes, n, alpha: float = 0.05):
    """Wilson interval for a proportion, better than the normal one for small groups (AI help)."""
    successes, n = np.asarray(successes, float), np.asarray(n, float)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(df: pd.DataFrame, col: str, target: str = "readmit_30",
               alpha: float = 0.05) -> pd.DataFrame:
    """Readmission rate per group + its Wilson CI (EDA, findings 4, 7, 8, 10)."""
    g = df.groupby(col, observed=True)[target].agg(positives="sum", n="count")
    g["rate"] = g["positives"] / g["n"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["positives"], g["n"], alpha)
    g["ci_width"] = g["ci_high"] - g["ci_low"]
    return g


def chi2_test(df: pd.DataFrame, col: str, target: str = "readmit_30",
              min_count: int = 100) -> dict:
    """Chi-square test, H0: same readmission rate in every category (AI help).
    Also gives Cramér's V (0 to 1), since with ~100k rows even tiny gaps get a tiny p-value."""
    x = lump_rare(df[col], min_count)
    ct = pd.crosstab(x, df[target])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    k = min(ct.shape) - 1
    return {
        "feature": col,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": np.sqrt(chi2 / (n * k)),
        "min_expected": expected.min(),
    }


def mannwhitney_test(df: pd.DataFrame, col: str, target: str = "readmit_30") -> dict:
    """Mann-Whitney U, H0: same distribution for readmitted and not readmitted (AI help).
    Not a t-test because visits / days are skewed; effect size = rank-biserial (0 = no effect)."""
    pos = df.loc[df[target] == 1, col]
    neg = df.loc[df[target] == 0, col]
    u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "feature": col,
        "U": u,
        "p_value": p,
        "rank_biserial": 2 * u / (len(pos) * len(neg)) - 1,
        "mean_readmitted": pos.mean(),
        "mean_not_readmitted": neg.mean(),
    }


def add_bonferroni(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Flag what stays significant after Bonferroni, since we run many tests at once (AI help)."""
    table = table.copy()
    table["significant_bonferroni"] = table["p_value"] < alpha / len(table)
    return table

# Tests suggested by an AI assistant (not seen in class), used to analyse the data in the EDA.
