import numpy as np


def ndcg_at_k(relevance, k=5):
    rel = np.asarray(relevance, dtype=float)[:k]
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rel) + 2))
    dcg = np.sum((2**rel - 1) / discounts)

    ideal = np.sort(rel)[::-1]
    idcg = np.sum((2**ideal - 1) / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def average_precision(relevance):
    rel = np.asarray(relevance) > 0
    positives = int(rel.sum())
    if positives == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, is_rel in enumerate(rel, start=1):
        if is_rel:
            hits += 1
            total += hits / rank
    return float(total / positives)


def evaluate_ranker(df, score_fn, k=5):
    ndcgs, aps = [], []
    for _, group in df.groupby("query_id"):
        scores = np.asarray(score_fn(group), dtype=float)
        order = np.argsort(-scores)
        rel = group["relevance"].to_numpy()[order]
        ndcgs.append(ndcg_at_k(rel, k))
        aps.append(average_precision(rel))
    return {
        "n_queries": len(ndcgs),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "map": float(np.mean(aps)),
    }
