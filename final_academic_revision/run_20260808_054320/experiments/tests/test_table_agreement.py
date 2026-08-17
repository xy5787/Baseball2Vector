from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from common import RUN_ROOT


def test_prediction_table_values_match_csv():
    results = pd.read_csv(RUN_ROOT / "results" / "calibrated_prediction.csv")
    table = (RUN_ROOT / "tables" / "prospective_prediction_revised.tex").read_text()
    for row in results.itertuples(index=False):
        if row.metric == "mae":
            assert f"{row.estimate:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]" in table
        elif row.metric in {"r2", "spearman"} and pd.notna(row.estimate):
            assert f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]" in table


def test_stability_table_values_match_csv():
    results = pd.read_csv(RUN_ROOT / "results" / "stability_by_pa.csv")
    table = (RUN_ROOT / "tables" / "stability_by_pa.tex").read_text()
    for row in results.itertuples(index=False):
        assert f"{row.pearson:.2f} [{row.pearson_ci_low:.2f}, {row.pearson_ci_high:.2f}]; {row.n_pairs}" in table


def test_figure_caption_content_matches_visible_panels():
    tex = (RUN_ROOT / "manuscript_preview" / "baseball2vector_en_revision.tex").read_text()
    assert "Left: standardized profile-distance distributions" in tex
    assert "Right: an illustrative rule-selected matched pair" in tex
    caption = tex.split("Concurrent dimension-change associations with $\\Delta$wRC+ from", 1)[1].split("}", 1)[0]
    assert "ROI" not in caption
    assert "random-forest feature importance" not in caption.lower()


def test_core_manuscript_numbers_trace_to_results():
    tex = (RUN_ROOT / "manuscript_preview" / "baseball2vector_en_revision.tex").read_text()
    paired = pd.read_csv(RUN_ROOT / "results" / "paired_model_differences.csv")
    row = paired[(paired["candidate"].eq("B2V Ridge")) & (paired["baseline"].eq("Calibrated wRC+ Ridge")) & paired["metric"].eq("mae")].iloc[0]
    assert f"{row.improvement:.2f}" in tex
    assert f"[{row.ci_low:.2f}, {row.ci_high:.2f}]" in tex
    matching = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distance_differences.csv")
    row = matching[(matching["left"].eq("scalar_matched")) & matching["right"].eq("same_player_adjacent")].iloc[0]
    assert f"{row.median_difference:.3f}" in tex
    assert f"[{row.ci_low:.3f}, {row.ci_high:.3f}]" in tex
