from pathlib import Path
import pandas as pd

from .config import FEATURES, GROUP_COL, OUTCOME_COLUMNS


def load_logged_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(FEATURES + [GROUP_COL, "position"] + OUTCOME_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Logged impression data is empty.")

    df = df.copy()
    for col in FEATURES + OUTCOME_COLUMNS + ["position"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["relevance"] = (
        df["clicked"] * 1
        + df["applied"] * 2
        + df["hired"] * 3
    )
    return df
