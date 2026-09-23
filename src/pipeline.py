from pathlib import Path
import json
import pandas as pd

from .bias import add_ipw
from .baseline import heuristic_score
from .config import GROUP_COL, TEST_QUERIES, TOP_K
from .data import load_logged_data
from .ltr import PairwiseLTR
from .metrics import evaluate_ranker


def split_by_query(df):
    queries = sorted(df[GROUP_COL].unique())
    if len(queries) <= TEST_QUERIES:
        raise ValueError("Not enough query groups for a held-out evaluation.")
    train_queries = queries[:-TEST_QUERIES]
    test_queries = queries[-TEST_QUERIES:]
    train = df[df[GROUP_COL].isin(train_queries)].copy()
    test = df[df[GROUP_COL].isin(test_queries)].copy()
    return train, test, train_queries, test_queries


def run(data_path="data/logged_impressions.csv", output_dir="outputs", model_path="models/pairwise_ltr.joblib"):
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    raw = load_logged_data(data_path)
    biased, propensities = add_ipw(raw)
    train, test, train_queries, test_queries = split_by_query(biased)

    model = PairwiseLTR().fit(train)
    model.save(model_path)

    baseline = evaluate_ranker(test, lambda g: heuristic_score(g).to_numpy(), TOP_K)
    ltr = evaluate_ranker(test, lambda g: model.score(g), TOP_K)

    comparison = {
        "baseline": baseline,
        "pairwise_ltr_ipw": ltr,
        "lift": {
            "ndcg": ltr[f"ndcg@{TOP_K}"] - baseline[f"ndcg@{TOP_K}"],
            "map": ltr["map"] - baseline["map"],
        },
        "quality_gate": {
            "strictly_beats_heuristic": (
                ltr[f"ndcg@{TOP_K}"] > baseline[f"ndcg@{TOP_K}"]
                and ltr["map"] > baseline["map"]
            )
        },
        "position_propensity": propensities,
        "train_queries": [int(x) for x in train_queries],
        "test_queries": [int(x) for x in test_queries],
        "train_rows": len(train),
        "test_rows": len(test),
        "pair_count": model.n_pairs_,
        "position_bias_method": "smoothed click propensity + inverse propensity weighting",
    }

    (outdir / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    ranked_rows = []
    for q, group in test.groupby("query_id"):
        g = group.copy()
        g["ltr_score"] = model.score(g)
        g["heuristic_score"] = heuristic_score(g)
        g["ltr_rank"] = g["ltr_score"].rank(method="first", ascending=False).astype(int)
        g["heuristic_rank"] = g["heuristic_score"].rank(method="first", ascending=False).astype(int)
        ranked_rows.append(g)
    pd.concat(ranked_rows).sort_values(["query_id", "ltr_rank"]).to_csv(
        outdir / "heldout_rankings.csv", index=False
    )

    # Worked example + safe fallback.
    example = test.sort_values(["query_id", "position"]).iloc[0].copy()
    example_payload = {
        "input": {k: example[k].item() if hasattr(example[k], "item") else example[k] for k in
                  ["query_id", "candidate_id"] + list(model.feature_names)},
        "ltr_score": float(model.score(pd.DataFrame([example]))[0]),
        "reason": model.explain(example),
        "fallback": {
            "when": "LTR model unavailable or scoring raises an exception",
            "action": "use the current heuristic score and preserve existing ordering safeguards",
        },
    }
    (outdir / "worked_example.json").write_text(
        json.dumps(example_payload, indent=2), encoding="utf-8"
    )

    return comparison


def safe_rank(group, model_path="models/pairwise_ltr.joblib"):
    try:
        model = PairwiseLTR.load(model_path)
        g = group.copy()
        g["score"] = model.score(g)
        g["rank_source"] = "pairwise_ltr"
        return g.sort_values("score", ascending=False), "pairwise_ltr"
    except Exception:
        g = group.copy()
        g["score"] = heuristic_score(g)
        g["rank_source"] = "heuristic_fallback"
        return g.sort_values("score", ascending=False), "heuristic_fallback"
