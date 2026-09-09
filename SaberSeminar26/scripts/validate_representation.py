"""Validate canonical results; optionally regenerate the complete analysis."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseball2vec import matching, prediction, scaling_audit, stability  # noqa: E402


def check_artifacts() -> None:
    results = ROOT / "results"
    paper = (ROOT / "paper" / "baseball2vector_en.tex").read_text()
    pred = pd.read_csv(results / "prediction_results.csv")
    b2v = pred[(pred.model == "B2V Ridge") & (pred.metric == "mae")].iloc[0]
    assert f"{b2v.estimate:.3f}" in paper
    robust = pd.read_csv(results / "robustness_results.csv")
    assert set(robust.n_pairs) == {340, 507, 567, 1414}
    headline = pd.read_csv(results / "headline_results.csv")
    assert headline.claim_id.is_unique
    for row in headline.itertuples(index=False):
        assert str(row.manuscript_value) in paper, row.claim_id
    print(f"verified {len(headline)} manuscript headline values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if args.regenerate:
        prediction.main()
        scaling_audit.main()
        matching.main()
        stability.main()
    check_artifacts()


if __name__ == "__main__":
    main()
