"""ΔTool → ΔwRC+ regression and per-player development ROI.

Inputs
------
data/processed/v3_results.csv

Outputs
-------
outputs/figures/roi_coefficients.png
outputs/figures/roi_individual_best_tool.png
outputs/figures/roi_partial_dependence.png
outputs/figures/roi_delta_scatter.png
outputs/tables/lr_results.csv
outputs/tables/individual_roi.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import random
import numpy as np
import pandas as pd

from baseball2vec.roi import (
    TOOL_NAMES,
    build_delta_dataset,
    compute_individual_roi,
    run_regression,
)
from baseball2vec.viz import (
    plot_delta_scatter,
    plot_partial_dependence,
    plot_roi_coefficients,
    plot_roi_individual,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
TABLES_DIR = Path(__file__).parents[1] / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 4: ROI analysis")
print("=" * 60)

player_df = pd.read_csv(PROCESSED_DIR / "v3_results.csv")
print(
    f"  Loaded: {len(player_df)} player-seasons, {player_df['Name'].nunique()} players"
)

delta_clean = build_delta_dataset(player_df)
print(
    f"  Consecutive-season pairs: {len(delta_clean)}, unique players: {delta_clean['Name'].nunique()}"
)

results = run_regression(delta_clean, random_state=SEED)

print("\n  Model comparison (5-fold CV R²):")
print(f"  {'Target':<10} {'LR(5f)':>9} {'Ridge':>9} {'RF(5f)':>9} {'RF(10f)':>9}")
for r in results:
    print(
        f"  {r.target:<10} {r.lr_r2_cv:>9.4f} {r.ridge_r2_cv:>9.4f} {r.rf5_r2_cv:>9.4f} {r.rf10_r2_cv:>9.4f}"
    )

# ΔwRC+ is the primary target
wrc_res = next((r for r in results if "wRC" in r.target), None)
if wrc_res is None:
    raise RuntimeError("ΔwRC+ target not found in regression results.")

plot_roi_coefficients(wrc_res)

# Partial dependence: needs original X_full array
delta_cols = [f"D_{t}" for t in TOOL_NAMES]
from_cols = [f"{t}_From" for t in TOOL_NAMES]
X_full = delta_clean[from_cols + delta_cols].values
plot_partial_dependence(wrc_res.rf10_model, X_full, TOOL_NAMES)

plot_delta_scatter(delta_clean, TOOL_NAMES)

prospect_df = compute_individual_roi(player_df, wrc_res.rf10_model)
latest_season = int(player_df["Season"].max())
plot_roi_individual(prospect_df, TOOL_NAMES, latest_season)

print("\n  Best development tool distribution:")
for tool, count in prospect_df["Best_Tool"].value_counts().items():
    print(f"    {tool}: {count} ({count / len(prospect_df) * 100:.1f}%)")

# Save tables
lr_rows = []
for r in results:
    row = {"target": r.target, "lr_r2_cv": r.lr_r2_cv, "rf10_r2_cv": r.rf10_r2_cv}
    row.update({f"lr_coef_{k}": v for k, v in r.lr_coefs.items()})
    lr_rows.append(row)
pd.DataFrame(lr_rows).to_csv(TABLES_DIR / "lr_results.csv", index=False)
prospect_df.to_csv(TABLES_DIR / "individual_roi.csv", index=False)
delta_clean.to_csv(TABLES_DIR / "delta_dataset.csv", index=False)

print(f"\n  Saved tables -> {TABLES_DIR}")

# Key numbers
print("\n  --- Key abstract numbers ---")
print(f"  ΔwRC+ LR R² (CV): {wrc_res.lr_r2_cv:.4f}")
buxton = prospect_df[prospect_df["Name"].str.contains("Buxton", na=False)]
if not buxton.empty:
    print(f"  Buxton Contact ROI: {buxton.iloc[0]['ROI_Contact']:+.2f}")
