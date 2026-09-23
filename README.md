# PlaceMux — Day 56 / Task 9
## Learning-to-Rank, Offline Evaluation & Position-Bias Correction

### Objective
Rebuild marketplace matching as a proper pairwise Learning-to-Rank system trained from logged impressions and outcomes.

### What is implemented
- Pairwise LTR using within-query preference pairs.
- Training from logged `clicked`, `applied`, and `hired` outcomes.
- Position-bias correction using smoothed click propensity and inverse propensity weighting (IPW).
- Held-out query-group evaluation.
- nDCG@5 and MAP.
- Direct comparison with the current heuristic ranker.
- Reproducible experiment output.
- Plain-English worked explanation.
- Safe heuristic fallback when the LTR model is unavailable.
- Automated tests.

### Data
`data/logged_impressions.csv` is the available PlaceMux logged-impression dataset used for this implementation. It contains 50 impressions across 10 query groups with candidate features, position, click, apply and hire outcomes.

### Relevance
The training relevance label is:

```text
relevance = clicked + 2*applied + 3*hired
```

This gives downstream outcomes greater weight while retaining click signal.

### Position-bias correction
Observed clicks are not treated as unbiased relevance labels. Position propensity is estimated from logged click rates with Beta-style smoothing:

```text
propensity(position) = (clicks + 1) / (impressions + 2)
IPW = 1 / propensity
```

Weights are clipped at 10 to prevent unstable estimates.

The pairwise training examples use the average IPW of the two documents in the preference pair.

### Evaluation protocol
Queries are split by group, not individual rows. The final 3 query groups are held out and never used for training.

Metrics:
- nDCG@5
- MAP

The baseline is the current interpretable heuristic using:
- skill match
- profile score
- salary match
- experience
- distance

### Quality gate
The production candidate must **strictly improve both nDCG@5 and MAP** on held-out queries.

The supplied dataset produces a tie between the heuristic and LTR on these metrics. Therefore the run is intentionally reported as **HOLD**, not falsely marked as a win. A larger/diverse production log with counterfactual exposure variation is required before claiming an offline improvement.

### Run

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py
pytest -v
```

### Outputs
After `python main.py`:

```text
outputs/comparison.json
outputs/heldout_rankings.csv
outputs/worked_example.json
models/pairwise_ltr.joblib
```

### Failure path
If the LTR model cannot be loaded or scoring fails, `safe_rank()` falls back to the existing heuristic ranking. This keeps the matching flow available while clearly marking the source as `heuristic_fallback`.

### Online-test hand-off
Before an online experiment:
1. Log impression IDs, query IDs, candidate IDs, model version and served position.
2. Preserve click/apply/hire outcomes.
3. Randomize/control exposure enough to estimate position propensity reliably.
4. Run the LTR model behind a controlled experiment.
5. Monitor ranking quality, conversion outcomes, latency, fallback rate and segment-level behavior.
6. Promote only after online evidence confirms the offline result.

### Important limitation
The available logged dataset is small and strongly ordered by position. Several positions have zero clicks, so even smoothed IPW cannot create genuine counterfactual information. The implementation is therefore complete as an engineering pipeline, but the evidence is not sufficient to claim an offline win or production readiness.

### Repository
Recommended GitHub repository name:

`Day56_Task9_PlaceMux_Learning_to_Rank`
