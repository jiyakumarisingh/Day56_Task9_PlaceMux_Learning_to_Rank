# Day 56 Handoff

## Candidate
`models/pairwise_ltr.joblib`

## Training
Pairwise logistic ranking trained on within-query outcome preferences with IPW weights.

## Bias correction
Smoothed position click propensity + inverse propensity weighting.

## Offline gate
Compare against the heuristic on held-out query groups using nDCG@5 and MAP.

## Current evidence
The supplied 50-row log yields a metric tie, so the quality gate is HOLD.

## Online experiment prerequisites
Use a larger real log with exposure variation and continue logging:
`impression_id`, `query_id`, `candidate_id`, `position`, `model_version`, `clicked`, `applied`, `hired`.

## Fallback
`safe_rank()` uses the heuristic if the LTR model is unavailable.
