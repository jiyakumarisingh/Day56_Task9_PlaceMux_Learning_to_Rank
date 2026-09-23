import pandas as pd


def heuristic_score(df: pd.DataFrame) -> pd.Series:
    # Current interpretable heuristic: skill + profile + salary + experience,
    # with distance as a negative signal.
    experience = (df["experience_years"] / 10.0).clip(0, 1)
    distance = (1.0 - df["distance_km"] / 35.0).clip(0, 1)
    return (
        0.35 * df["skill_match"]
        + 0.20 * df["profile_score"]
        + 0.20 * df["salary_match"]
        + 0.15 * experience
        + 0.10 * distance
    )
