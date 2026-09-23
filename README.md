# PlaceMux — Day 56 / Task 9

## Learning-to-Rank, Offline Evaluation & Position-Bias Correction

---

## 1. Task Overview

**Project:** PlaceMux
**Phase:** Phase 3 — Scale, Intelligence & Enterprise Readiness
**Day:** 56
**Task:** Task 9 — Learning-to-Rank
**Focus:** Matching Intelligence / Ranking / Experimentation

### Objective

Rebuild the PlaceMux marketplace matching system as a proper **Learning-to-Rank (LTR)** solution trained from logged impressions and real interaction outcomes.

The system must:

* Learn ranking preferences from historical interactions.
* Correct for position bias in logged observations.
* Compare the learned ranker against the current heuristic.
* Evaluate ranking quality using **nDCG@5 and MAP**.
* Use held-out query groups for honest evaluation.
* Provide an explainable ranking example.
* Gracefully fall back to the existing heuristic if the LTR model is unavailable.
* Produce a candidate model that can be taken forward into an online experiment.

---

# 2. Definition of Good

The success criterion for this task is:

> **A ranker that beats the current heuristic on held-out offline ranking metrics and is technically prepared for a controlled online experiment.**

The required offline quality gate is:

```text
LTR nDCG@5 > Heuristic nDCG@5
AND
LTR MAP > Heuristic MAP
```

A tie is **not** considered an offline win.

---

# 3. Task Requirements

The Day 56 task requires three major components.

### A. Learning-to-Rank Model

Build an LTR model trained from:

* Logged impressions
* Clicks
* Applications
* Hires
* Candidate/job matching features
* Query-level ranking groups

### B. Offline Ranking Evaluation

Compare:

```text
Current Heuristic
        VS
Pairwise LTR
```

using:

* nDCG@5
* MAP

The evaluation must use held-out query groups.

### C. Position-Bias Correction

Logged clicks are affected by the position at which a candidate was displayed.

Therefore, raw clicks are not treated as unbiased relevance labels.

The implementation uses:

```text
Position propensity
        ↓
Inverse Propensity Weighting
        ↓
Pairwise LTR training
```

---

# 4. Project Architecture

```text
                    Logged Impressions
                           |
                           v
                 data/logged_impressions.csv
                           |
                           v
                    Data Validation
                           |
                           v
                Outcome / Relevance Label
                           |
                           v
                 Position Bias Estimation
                           |
                           v
               Inverse Propensity Weighting
                           |
                           v
                  Query Group Split
                    /           \
                   /             \
              Training          Held-out
                 |                |
                 v                v
          Pairwise LTR       Offline Evaluation
                 |                |
                 v                |
          LTR Ranking Score       |
                 |                |
                 +-------+--------+
                         |
                         v
                Heuristic Comparison
                         |
                         v
                  nDCG@5 + MAP
                         |
                         v
                   Quality Gate
                    /       \
                  PASS      HOLD
                         |
                         v
                 Online Experiment
```

---

# 5. Dataset

The project uses:

```text
data/logged_impressions.csv
```

This is the available PlaceMux logged interaction dataset.

The dataset contains:

```text
50 impressions
10 query groups
5 candidates per query
```

### Dataset columns

| Column             | Description                               |
| ------------------ | ----------------------------------------- |
| `query_id`         | Ranking/request group                     |
| `user_id`          | User identifier                           |
| `candidate_id`     | Candidate identifier                      |
| `skill_match`      | Skill compatibility score                 |
| `experience_years` | Candidate experience                      |
| `distance_km`      | Candidate distance                        |
| `salary_match`     | Salary compatibility                      |
| `profile_score`    | Existing profile score                    |
| `position`         | Logged serving position                   |
| `clicked`          | Whether candidate was clicked             |
| `applied`          | Whether candidate received an application |
| `hired`            | Whether candidate was hired               |

The dataset is grouped by `query_id`, which allows ranking evaluation at the query/list level.

---

# 6. Relevance Construction

The LTR model needs a graded relevance signal.

The implementation combines downstream interaction outcomes:

```text
relevance =
    clicked × 1
  + applied × 2
  + hired × 3
```

Therefore:

| Outcome         | Contribution |
| --------------- | -----------: |
| Impression only |            0 |
| Click           |           +1 |
| Apply           |           +2 |
| Hire            |           +3 |

Example:

```text
clicked = 1
applied = 1
hired   = 1
```

produces:

```text
relevance = 1 + 2 + 3
          = 6
```

A hired candidate therefore receives stronger relevance than a candidate that was only clicked.

---

# 7. Features

The LTR model uses the following ranking features:

```text
skill_match
experience_years
distance_km
salary_match
profile_score
```

### Feature interpretation

#### skill_match

Measures compatibility between candidate skills and job requirements.

Higher values indicate stronger skill compatibility.

#### experience_years

Represents candidate experience.

Higher experience can contribute positively to ranking.

#### distance_km

Represents candidate distance.

Lower distance is generally preferred by the current heuristic.

#### salary_match

Measures salary compatibility.

Higher values indicate stronger salary alignment.

#### profile_score

Represents the existing profile/matching score.

---

# 8. Current Heuristic Baseline

The existing heuristic is retained as the production comparison baseline.

Its score is calculated from:

```text
35% skill_match
20% profile_score
20% salary_match
15% normalized experience
10% normalized distance
```

Conceptually:

```text
heuristic_score =
    0.35 × skill_match
  + 0.20 × profile_score
  + 0.20 × salary_match
  + 0.15 × experience
  + 0.10 × distance_score
```

This provides an interpretable baseline before introducing learned ranking.

---

# 9. Pairwise Learning-to-Rank

The implementation uses a **pairwise LTR approach**.

Instead of directly predicting whether an individual candidate is relevant, the model learns:

> Which candidate should rank above another candidate for the same query?

For candidates A and B:

```text
Candidate A relevance > Candidate B relevance
```

creates a preference:

```text
A > B
```

The training process generates within-query preference pairs.

Pairs with identical relevance are ignored because they do not provide a useful ranking preference.

---

# 10. Pairwise Training

For each query group:

```text
Query
 |
 +-- Candidate A
 +-- Candidate B
 +-- Candidate C
 +-- Candidate D
 +-- Candidate E
```

the system compares candidates pairwise.

Example:

```text
A relevance = 6
B relevance = 3
```

creates:

```text
A > B
```

The model receives the feature difference:

```text
features(A) - features(B)
```

and learns the preferred ordering.

Both preference directions are included during training.

---

# 11. Position-Bias Problem

Clicks cannot automatically be interpreted as unbiased relevance.

A candidate shown in a highly visible position can receive more attention simply because of its position.

Therefore:

```text
Observed click
       ≠
Pure candidate relevance
```

The system explicitly models serving position.

---

# 12. Position Propensity Estimation

Position click propensity is estimated using smoothed click rates.

The implementation uses:

```text
propensity(position)
=
(clicks + 1)
----------------
(impressions + 2)
```

The smoothing prevents zero-propensity estimates when the dataset is small.

A minimum propensity of:

```text
0.10
```

is applied.

---

# 13. Inverse Propensity Weighting

After estimating propensity:

```text
IPW = 1 / propensity
```

The resulting weight is clipped at:

```text
10.0
```

This prevents extremely large weights from destabilizing training.

The pairwise training weight is calculated from the average IPW of the two candidates participating in a preference pair.

Therefore:

```text
Logged interaction
        |
        v
Position propensity
        |
        v
Inverse propensity weight
        |
        v
Weighted preference pair
        |
        v
Pairwise LTR
```

---

# 14. Train / Test Split

The system performs a **query-level split**.

Individual rows from the same query are never randomly scattered between training and test sets.

The final:

```text
3 query groups
```

are held out for evaluation.

The remaining query groups are used for training.

This prevents candidates from the same ranking request leaking between training and evaluation.

---

# 15. Offline Evaluation

The model is evaluated against the existing heuristic.

The evaluation uses:

### nDCG@5

Normalized Discounted Cumulative Gain measures whether highly relevant candidates appear near the top of the ranking.

Higher values are better.

```text
1.0 = ideal ranking
```

### MAP

Mean Average Precision measures how effectively relevant candidates are ranked across the evaluated query lists.

Higher values are better.

---

# 16. Evaluation Protocol

The evaluation compares:

```text
                    Held-out queries
                          |
             +------------+------------+
             |                         |
             v                         v
      Current Heuristic          Pairwise LTR
             |                         |
             v                         v
         nDCG@5                      nDCG@5
         MAP                          MAP
             |                         |
             +------------+------------+
                          |
                          v
                    Quality Gate
```

The LTR model must strictly outperform the heuristic on both metrics.

---

# 17. Actual Evaluation Result

The implementation was executed successfully.

### Offline result

| Metric | Heuristic | Pairwise LTR |    Lift |
| ------ | --------: | -----------: | ------: |
| nDCG@5 |    1.0000 |       1.0000 | +0.0000 |
| MAP    |    1.0000 |       1.0000 | +0.0000 |

### Quality Gate

```text
HOLD
```

This is intentional.

The model did not demonstrate an offline improvement over the heuristic.

The implementation therefore does **not** claim a false ranking improvement.

---

# 18. Why the Result Is HOLD

The currently available dataset is small:

```text
50 logged impressions
10 query groups
```

and has a strong existing ordering.

The observed interaction data does not provide enough counterfactual exposure variation to establish that the learned ranker is genuinely better than the existing ranking logic.

Consequently:

```text
LTR = 1.0000 nDCG@5
Heuristic = 1.0000 nDCG@5
```

and:

```text
LTR = 1.0000 MAP
Heuristic = 1.0000 MAP
```

means:

```text
No measured offline improvement
```

rather than:

```text
LTR failure
```

---

# 19. Why We Do Not Force PASS

A ranking experiment should not be marked successful simply because the model trains successfully.

The actual requirement is:

```text
Model trained
+
Bias corrected
+
Held-out evaluation
+
Improvement over baseline
```

The current run satisfies the engineering requirements but does not demonstrate the final improvement criterion.

Therefore the correct status is:

```text
ENGINEERING IMPLEMENTATION: COMPLETE

OFFLINE QUALITY GATE: HOLD
```

---

# 20. Explainability

The system produces a worked example containing:

```text
Input
  ↓
Candidate features
  ↓
LTR score
  ↓
Top contributing features
  ↓
Plain-English explanation
```

Example explanation format:

```text
skill_match increases the learned score;
profile_score increases the learned score.
```

This gives the user a human-readable explanation rather than exposing only a numerical ranking score.

---

# 21. Safe Failure Path

The ranking system includes a fallback mechanism.

Normal path:

```text
LTR model
   ↓
LTR score
   ↓
Rank candidates
```

Failure path:

```text
LTR unavailable / scoring error
              ↓
       Existing heuristic
              ↓
       Rank candidates
```

The output explicitly identifies the ranking source:

```text
pairwise_ltr
```

or:

```text
heuristic_fallback
```

This prevents an LTR model failure from taking down the matching flow.

---

# 22. Worked Demo

A demo can be executed through:

```powershell
python main.py
```

The pipeline:

1. Loads logged interaction data.
2. Validates the dataset.
3. Generates relevance labels.
4. Estimates position propensity.
5. Adds IPW weights.
6. Splits query groups.
7. Trains the pairwise LTR model.
8. Saves the trained model.
9. Evaluates the heuristic.
10. Evaluates the LTR model.
11. Calculates nDCG@5.
12. Calculates MAP.
13. Calculates metric lift.
14. Applies the quality gate.
15. Saves the evaluation results.
16. Saves held-out rankings.
17. Saves a worked explanation example.

---

# 23. Generated Outputs

Running:

```powershell
python main.py
```

generates:

```text
outputs/
│
├── comparison.json
├── heldout_rankings.csv
└── worked_example.json
```

The trained model is stored at:

```text
models/
└── pairwise_ltr.joblib
```

---

# 24. Experiment Log

`outputs/comparison.json` records:

* Baseline metrics
* LTR metrics
* nDCG lift
* MAP lift
* Quality-gate result
* Position propensity
* Training query groups
* Test query groups
* Training row count
* Test row count
* Pair count
* Bias-correction method

This makes the experiment reproducible.

---

# 25. Automated Tests

The project contains tests for:

* Logged-data validation
* Relevance construction
* Position-bias correction
* IPW generation
* Pairwise LTR training
* nDCG calculation
* MAP calculation
* Missing-model fallback
* Successful LTR ranking

Run:

```powershell
python -m pytest -v
```

### Actual verification

```text
7 passed
```

All automated tests passed successfully.

---

# 26. End-to-End Verification

The complete pipeline was successfully executed:

```powershell
python main.py
```

Output:

```text
========================================================================
PLACEMUX - DAY 56 / TASK 9
LEARNING-TO-RANK + OFFLINE EVALUATION + POSITION-BIAS CORRECTION
========================================================================

Baseline nDCG@5 : 1.0000
LTR nDCG@5      : 1.0000
Baseline MAP    : 1.0000
LTR MAP         : 1.0000

nDCG lift       : +0.0000
MAP lift        : +0.0000

Quality gate    : HOLD

Position IPW    : smoothed click propensity + inverse propensity weighting
```

---

# 27. Project Structure

```text
Day56_Task9_PlaceMux_Learning_to_Rank/
│
├── data/
│   └── logged_impressions.csv
│
├── models/
│   └── pairwise_ltr.joblib
│
├── outputs/
│   ├── comparison.json
│   ├── heldout_rankings.csv
│   └── worked_example.json
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── bias.py
│   ├── baseline.py
│   ├── ltr.py
│   ├── metrics.py
│   └── pipeline.py
│
├── tests/
│   └── test_ltr.py
│
├── main.py
├── requirements.txt
├── README.md
├── HANDOFF.md
└── .gitignore
```

---

# 28. Installation

Create the virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 29. Run the Pipeline

Execute:

```powershell
python main.py
```

The complete LTR training and evaluation pipeline will run.

---

# 30. Run Tests

Execute:

```powershell
python -m pytest -v
```

Expected current result:

```text
7 passed
```

---

# 31. Inspect Evaluation Results

View:

```powershell
type outputs\comparison.json
```

View the worked example:

```powershell
type outputs\worked_example.json
```

Open ranking results:

```powershell
notepad outputs\heldout_rankings.csv
```

---

# 32. Online Experiment Readiness

The implementation is structured for the next online experimentation stage.

Before starting an online experiment, the production logging layer should preserve:

```text
impression_id
query_id
candidate_id
position
model_version
timestamp
clicked
applied
hired
```

The experiment should also provide sufficient exposure variation to estimate position effects reliably.

---

# 33. Recommended Online Experiment Flow

```text
Existing Production Ranker
          |
          +------------------+
          |                  |
          v                  v
     Control Group       Treatment Group
          |                  |
          v                  v
     Heuristic             LTR
          |                  |
          +--------+---------+
                   |
                   v
             Interaction Logs
                   |
                   v
           Experiment Metrics
```

Monitor:

* nDCG
* MAP
* Click-through rate
* Apply rate
* Hire rate
* Ranking latency
* Model error rate
* Fallback rate
* Segment-level performance

---

# 34. Production Safety

The LTR model should not replace the existing ranker without controlled validation.

Recommended rollout:

```text
Offline evaluation
       ↓
Shadow mode
       ↓
Small controlled experiment
       ↓
Online metric validation
       ↓
Expanded rollout
```

The existing heuristic remains available as the fallback.

---

# 35. Limitations

### 1. Small dataset

The current dataset contains only:

```text
50 impressions
```

This is insufficient for strong production-level conclusions.

### 2. Limited exposure variation

Historical positions are strongly structured.

This limits the ability to estimate true counterfactual relevance.

### 3. Click bias

Clicks may reflect both:

```text
Candidate relevance
+
Position exposure
```

The implementation corrects for observed position propensity, but propensity correction cannot manufacture information that was never logged.

### 4. No online validation yet

The current evaluation is offline.

No claim of online business impact is made.

### 5. No causal guarantee

IPW reduces known position-exposure bias under its assumptions, but it does not guarantee unbiased causal estimates when important confounders are missing.

---

# 36. Future Improvements

For a stronger production LTR system:

* Larger production interaction dataset
* Randomized exploration traffic
* Better propensity estimation
* Doubly robust evaluation
* LambdaMART / LightGBM ranking
* XGBoost ranking objective
* Listwise LTR
* Temporal train/test split
* Feature freshness signals
* Candidate/job embeddings
* Search/retrieval integration
* MLflow experiment tracking
* Model registry
* Online A/B experimentation
* Ranking latency monitoring
* Drift monitoring
* Segment-level evaluation
* Fairness evaluation
* Automated retraining

---

# 37. Definition of Done

### Learning-to-Rank

* [x] Real logged interaction data loaded
* [x] Query-level ranking groups created
* [x] Graded relevance constructed
* [x] Pairwise LTR implemented
* [x] IPW weights integrated into training
* [x] Model persisted
* [x] Explainability implemented
* [x] Failure fallback implemented

### Offline Evaluation

* [x] Existing heuristic implemented
* [x] Held-out query groups used
* [x] nDCG@5 calculated
* [x] MAP calculated
* [x] LTR compared against heuristic
* [x] Metric lift calculated
* [x] Experiment output persisted

### Position Bias

* [x] Position recorded
* [x] Position propensity estimated
* [x] Smoothing applied
* [x] IPW calculated
* [x] IPW clipping applied
* [x] IPW integrated into pairwise training

### Verification

* [x] End-to-end pipeline executed
* [x] 7 automated tests passed
* [x] Model artifact generated
* [x] Evaluation outputs generated
* [x] Failure path tested

### Quality Gate

* [ ] LTR strictly beats heuristic on nDCG@5
* [ ] LTR strictly beats heuristic on MAP
* [ ] Online experiment validated

**Current status:**

```text
Engineering implementation: COMPLETE
Offline quality gate: HOLD
Online experiment: NOT YET VALIDATED
```

---

# 38. Final Task Status

The Day 56 implementation successfully demonstrates a complete Learning-to-Rank engineering pipeline:

```text
Logged Interaction Data
        ↓
Outcome-Based Relevance
        ↓
Position-Bias Correction
        ↓
IPW Weighted Pairwise Training
        ↓
Pairwise LTR Model
        ↓
Held-Out Evaluation
        ↓
nDCG@5 + MAP
        ↓
Heuristic Comparison
        ↓
Quality Gate
        ↓
Online Experiment Candidate
```

The current dataset does not demonstrate an offline improvement because both the heuristic and LTR achieve:

```text
nDCG@5 = 1.0000
MAP     = 1.0000
```

Therefore the system correctly reports:

```text
QUALITY GATE: HOLD
```

rather than claiming an unsupported ranking improvement.

This makes the experiment reproducible, explainable and ready for additional real interaction data and controlled online validation.

---

# 39. GitHub Repository

Recommended repository name:

```text
Day56_Task9_PlaceMux_Learning_to_Rank
```

Commit:

```powershell
git add .
git commit -m "Complete Day 56 learning to rank with position bias correction"
git push
```

---

# 40. Handoff

The next engineer can start from:

```text
models/pairwise_ltr.joblib
```

and review:

```text
outputs/comparison.json
outputs/heldout_rankings.csv
outputs/worked_example.json
```

The next required step is to obtain a larger and more varied production interaction log, preferably with controlled exposure variation, and rerun the offline evaluation before proceeding to an online experiment.
