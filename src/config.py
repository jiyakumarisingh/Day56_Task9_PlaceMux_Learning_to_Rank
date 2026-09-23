FEATURES = [
    "skill_match",
    "experience_years",
    "distance_km",
    "salary_match",
    "profile_score",
]

GROUP_COL = "query_id"
OUTCOME_COLUMNS = ["clicked", "applied", "hired"]

# Higher downstream outcomes carry more relevance.
CLICK_WEIGHT = 1
APPLY_WEIGHT = 2
HIRE_WEIGHT = 3

# Pairwise LTR training / evaluation settings.
TOP_K = 5
TEST_QUERIES = 3
RANDOM_SEED = 42

# Smoothed position propensity and IPW are clipped to avoid unstable weights.
PROPENSITY_ALPHA = 1.0
MIN_PROPENSITY = 0.10
MAX_IPW = 10.0

# Quality gate: a strict improvement is required for a production candidate.
MIN_NDCG_LIFT = 0.0
MIN_MAP_LIFT = 0.0
