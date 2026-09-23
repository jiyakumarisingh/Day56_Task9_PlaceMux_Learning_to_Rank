import json
from src.pipeline import run

if __name__ == "__main__":
    result = run()
    print("=" * 72)
    print("PLACEMUX - DAY 56 / TASK 9")
    print("LEARNING-TO-RANK + OFFLINE EVALUATION + POSITION-BIAS CORRECTION")
    print("=" * 72)
    print(f"Baseline nDCG@5 : {result['baseline']['ndcg@5']:.4f}")
    print(f"LTR nDCG@5      : {result['pairwise_ltr_ipw']['ndcg@5']:.4f}")
    print(f"Baseline MAP    : {result['baseline']['map']:.4f}")
    print(f"LTR MAP         : {result['pairwise_ltr_ipw']['map']:.4f}")
    print(f"nDCG lift       : {result['lift']['ndcg']:+.4f}")
    print(f"MAP lift        : {result['lift']['map']:+.4f}")
    print(f"Quality gate    : {'PASS' if result['quality_gate']['strictly_beats_heuristic'] else 'HOLD'}")
    print(f"Position IPW    : {result['position_bias_method']}")
    print("Outputs         : outputs/comparison.json, outputs/heldout_rankings.csv,")
    print("                  outputs/worked_example.json")
