"""Reproducibility regression test for Baseball2Vec.

Runs the ROI pipeline on the committed v3_results.csv and asserts that
key abstract numbers are within documented tolerances.

Run with: pytest tests/test_reproducibility.py -v
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from baseball2vec.roi import build_delta_dataset, compute_individual_roi, run_regression

SEED = 42
V3_CSV = Path(__file__).parents[1] / "data" / "processed" / "v3_results.csv"

# Tolerances from the Saberseminar 2026 abstract
EXPECTED_ROWS = 2309
# Full precision from v3_results.csv (abstract reports as 0.81, rounded to 4dp = 0.8061)
EXPECTED_WAR_CORR = 0.8060958222864365
EXPECTED_LR_R2 = 0.7068  # ΔwRC+ LR 5-fold CV R²
EXPECTED_BUXTON_ROI = 14.02  # Byron Buxton Contact ROI (ΔwRC+)

WAR_CORR_TOL = 1e-6
LR_R2_TOL = 1e-4
ROI_TOL = 0.1


@pytest.fixture(scope="module")
def player_df() -> pd.DataFrame:
    assert V3_CSV.exists(), f"v3_results.csv not found at {V3_CSV}"
    return pd.read_csv(V3_CSV)


@pytest.fixture(scope="module")
def roi_results(player_df):
    random.seed(SEED)
    np.random.seed(SEED)
    delta_clean = build_delta_dataset(player_df)
    results = run_regression(delta_clean, random_state=SEED)
    wrc_res = next(r for r in results if "wRC" in r.target)
    prospect_df = compute_individual_roi(player_df, wrc_res.rf10_model)
    return wrc_res, prospect_df


def test_row_count(player_df):
    assert len(player_df) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(player_df)}"
    )


def test_pentagon_war_correlation(player_df):
    r = player_df["ZScore_Area"].corr(player_df["WAR"])
    assert abs(r - EXPECTED_WAR_CORR) < WAR_CORR_TOL, (
        f"Pentagon-WAR r = {r:.6f}, expected {EXPECTED_WAR_CORR} ± {WAR_CORR_TOL}"
    )


def test_lr_r2(roi_results):
    wrc_res, _ = roi_results
    assert abs(wrc_res.lr_r2_cv - EXPECTED_LR_R2) < LR_R2_TOL, (
        f"LR R² (CV) = {wrc_res.lr_r2_cv:.4f}, expected {EXPECTED_LR_R2} ± {LR_R2_TOL}"
    )


def test_buxton_roi(roi_results):
    _, prospect_df = roi_results
    buxton = prospect_df[prospect_df["Name"].str.contains("Buxton", na=False)]
    assert not buxton.empty, "Byron Buxton not found in prospect_df"
    roi = buxton.iloc[0]["ROI_Contact"]
    assert abs(roi - EXPECTED_BUXTON_ROI) < ROI_TOL, (
        f"Buxton Contact ROI = {roi:.2f}, expected {EXPECTED_BUXTON_ROI} ± {ROI_TOL}"
    )
