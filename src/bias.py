import pandas as pd
from .config import PROPENSITY_ALPHA, MIN_PROPENSITY, MAX_IPW


def estimate_position_propensity(df: pd.DataFrame) -> dict[int, float]:
    stats = df.groupby("position")["clicked"].agg(["sum", "count"])
    # Beta(1,1) smoothing prevents zero propensities on sparse positions.
    propensity = (
        (stats["sum"] + PROPENSITY_ALPHA)
        / (stats["count"] + 2 * PROPENSITY_ALPHA)
    )
    propensity = propensity.clip(lower=MIN_PROPENSITY)
    return {int(k): float(v) for k, v in propensity.items()}


def add_ipw(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    out = df.copy()
    prop = estimate_position_propensity(out)
    out["propensity"] = out["position"].map(prop)
    out["ipw"] = (1.0 / out["propensity"]).clip(upper=MAX_IPW)
    return out, prop
