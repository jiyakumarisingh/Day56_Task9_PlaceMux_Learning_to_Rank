from itertools import combinations
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, RANDOM_SEED


class PairwiseLTR:
    """Pairwise RankSVM-style learner implemented with logistic loss.

    Training creates both directions for every informative within-query pair.
    IPW weights are attached to each pair so highly exposed observations do
    not dominate the learned preference relation.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        self.feature_names = FEATURES

    def _pairs(self, df):
        X, y, w = [], [], []
        for _, group in df.groupby("query_id"):
            rows = list(group.iterrows())
            for (_, a), (_, b) in combinations(rows, 2):
                delta = a["relevance"] - b["relevance"]
                if delta == 0:
                    continue
                if delta > 0:
                    diff = a[FEATURES].to_numpy(float) - b[FEATURES].to_numpy(float)
                else:
                    diff = b[FEATURES].to_numpy(float) - a[FEATURES].to_numpy(float)

                weight = float((a["ipw"] + b["ipw"]) / 2.0)
                X.extend([diff, -diff])
                y.extend([1, 0])
                w.extend([weight, weight])
        if not X:
            raise ValueError("No informative preference pairs were generated.")
        return np.asarray(X), np.asarray(y), np.asarray(w)

    def fit(self, df):
        X, y, weights = self._pairs(df)
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y, sample_weight=weights)
        self.n_pairs_ = int(len(y) // 2)
        return self

    def score(self, df):
        return self.model.decision_function(
            self.scaler.transform(df[FEATURES].to_numpy(float))
        )

    def explain(self, row):
        coef = self.model.coef_[0] / self.scaler.scale_
        contributions = {
            feature: float(value * row[feature])
            for feature, value in zip(FEATURES, coef)
        }
        ordered = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top = ordered[:2]
        reasons = []
        for feature, contribution in top:
            direction = "increases" if contribution >= 0 else "reduces"
            reasons.append(f"{feature} {direction} the learned score")
        return "; ".join(reasons) + "."

    def save(self, path: str | Path):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path):
        return joblib.load(path)
