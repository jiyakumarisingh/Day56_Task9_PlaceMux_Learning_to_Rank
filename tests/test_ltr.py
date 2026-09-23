from pathlib import Path
import pandas as pd

from src.bias import add_ipw
from src.data import load_logged_data
from src.ltr import PairwiseLTR
from src.metrics import ndcg_at_k, average_precision
from src.pipeline import safe_rank


DATA = Path("data/logged_impressions.csv")


def test_logged_data_has_outcome_label():
    df = load_logged_data(DATA)
    assert len(df) == 50
    assert df["relevance"].max() > 0


def test_position_bias_correction_creates_ipw():
    df = load_logged_data(DATA)
    corrected, prop = add_ipw(df)
    assert "propensity" in corrected
    assert "ipw" in corrected
    assert min(prop.values()) >= 0.10
    assert corrected["ipw"].max() <= 10


def test_pairwise_ltr_trains_on_informative_pairs():
    df = load_logged_data(DATA)
    df, _ = add_ipw(df)
    model = PairwiseLTR().fit(df.iloc[:35])
    assert model.n_pairs_ > 0


def test_ndcg_perfect_order():
    assert ndcg_at_k([3, 2, 1, 0]) == 1.0


def test_map_perfect_order():
    assert average_precision([3, 2, 0, 0]) == 1.0


def test_safe_rank_uses_fallback_when_model_missing():
    df = load_logged_data(DATA)
    group = df[df.query_id == 1]
    ranked, source = safe_rank(group, "models/does_not_exist.joblib")
    assert source == "heuristic_fallback"
    assert len(ranked) == 5


def test_safe_rank_uses_model_after_training(tmp_path):
    df = load_logged_data(DATA)
    df, _ = add_ipw(df)
    model = PairwiseLTR().fit(df)
    path = tmp_path / "model.joblib"
    model.save(path)
    ranked, source = safe_rank(df[df.query_id == 1], path)
    assert source == "pairwise_ltr"
    assert len(ranked) == 5
