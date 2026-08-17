"""Re-run of scripts/02_baseline_comparison.py comparing the old (MinMax)
and new (mean=50, SD=10) 20-80 scaling on freshly-fetched data.

All outputs are written under ScalingRevision/outputs/ — the original
pipeline's data/ and outputs/ directories are untouched.

Inputs
------
data/raw/batting_stats_2021_2025.csv (fetched via pybaseball if absent)

Outputs
-------
ScalingRevision/outputs/figures/old/v2_correlation_heatmap.png
ScalingRevision/outputs/figures/new/v2_correlation_heatmap.png
ScalingRevision/outputs/tables/correlations_old.csv
ScalingRevision/outputs/tables/correlations_new.csv
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import random
import numpy as np
import pandas as pd

from baseball2vec.baselines import (
    build_scaled_results,
    correlation_table,
    pca_tool_scores,
    zscore_tool_scores,
)
from baseball2vec.data import load_raw, preprocess
from baseball2vec.tools import apply_direction, build_final_groups, season_zscore
from baseball2vec.viz import plot_correlation_heatmap
from scaling import build_scaled_results_sd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
TABLES_DIR = OUT_DIR / "tables"
FIG_OLD_DIR = OUT_DIR / "figures" / "old"
FIG_NEW_DIR = OUT_DIR / "figures" / "new"
for d in (TABLES_DIR, FIG_OLD_DIR, FIG_NEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ScalingRevision Step 2: Baseline comparison (Z-Score / PCA)")
print("  MinMax (old) vs mean=50/SD=10 (new)")
print("=" * 60)

raw = load_raw()
df = preprocess(raw)
final_groups = build_final_groups(df)
all_features = [c for cols in final_groups.values() for c in cols]

X_df = season_zscore(df, all_features)
X_aligned = apply_direction(X_df)

zscore_tools = zscore_tool_scores(X_aligned, final_groups)
pca_tools, pca_explained = pca_tool_scores(X_aligned, final_groups)

print("\n  PCA explained variance (PC1):")
for tool, ev in pca_explained.items():
    print(f"    {tool}: {ev:.3f}")

methods = {"ZScore": zscore_tools, "PCA": pca_tools}
method_names = ["ZScore", "PCA"]
validation_metrics = ["WAR", "wRC+", "OPS"]

scaled_old = build_scaled_results(methods)
from baseball2vec.tools import TOOL_NAMES

scaled_new = build_scaled_results_sd(methods, TOOL_NAMES)

base_cols = df[["UniqueName", "Name", "Season", "PA"]].copy().reset_index(drop=True)
for metric in validation_metrics:
    if metric in df.columns:
        base_cols[metric] = df[metric].values

result_old = base_cols.copy()
for name, sdf in scaled_old.items():
    result_old = pd.concat([result_old, sdf.reset_index(drop=True)], axis=1)

result_new = base_cols.copy()
for name, sdf in scaled_new.items():
    result_new = pd.concat([result_new, sdf.reset_index(drop=True)], axis=1)

corr_old = correlation_table(result_old, method_names, validation_metrics)
corr_new = correlation_table(result_new, method_names, validation_metrics)

print("\n  Pentagon Area vs Performance (Pearson r)")
print(f"  {'Method':<10} {'Scale':<16} {'WAR':>8} {'wRC+':>8} {'OPS':>8}")
for m in method_names:
    print(
        f"  {m:<10} {'MinMax(old)':<16} "
        f"{corr_old[m]['WAR']:>8.4f} {corr_old[m]['wRC+']:>8.4f} {corr_old[m]['OPS']:>8.4f}"
    )
    print(
        f"  {m:<10} {'SD10(new)':<16} "
        f"{corr_new[m]['WAR']:>8.4f} {corr_new[m]['wRC+']:>8.4f} {corr_new[m]['OPS']:>8.4f}"
    )

plot_correlation_heatmap(corr_old, method_names, validation_metrics, figures_dir=FIG_OLD_DIR)
plot_correlation_heatmap(corr_new, method_names, validation_metrics, figures_dir=FIG_NEW_DIR)

pd.DataFrame(corr_old).T.to_csv(TABLES_DIR / "correlations_old.csv")
pd.DataFrame(corr_new).T.to_csv(TABLES_DIR / "correlations_new.csv")
print(f"\n  Saved: {TABLES_DIR / 'correlations_old.csv'}")
print(f"  Saved: {TABLES_DIR / 'correlations_new.csv'}")
