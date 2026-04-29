"""Compute Z-Score and PCA baselines, output correlation table and heatmap.

Inputs
------
data/raw/batting_stats_2021_2025.csv

Outputs
-------
outputs/figures/v2_correlation_heatmap.png
outputs/tables/correlations.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import random
import numpy as np
import pandas as pd

from baseball2vec.baselines import build_scaled_results, correlation_table, pca_tool_scores, zscore_tool_scores
from baseball2vec.data import load_raw, preprocess
from baseball2vec.tools import TOOL_NAMES, apply_direction, build_final_groups, season_zscore
from baseball2vec.viz import plot_correlation_heatmap

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TABLES_DIR = Path(__file__).parents[1] / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 2: Baseline comparison (Z-Score / PCA)")
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
scaled = build_scaled_results(methods)

result_df = df[["UniqueName", "Name", "Season", "PA"]].copy().reset_index(drop=True)
for metric in ["WAR", "wRC+", "OPS"]:
    if metric in df.columns:
        result_df[metric] = df[metric].values
for name, sdf in scaled.items():
    result_df = pd.concat([result_df, sdf.reset_index(drop=True)], axis=1)

method_names = ["ZScore", "PCA"]
validation_metrics = ["WAR", "wRC+", "OPS"]
corr = correlation_table(result_df, method_names, validation_metrics)

print("\n  Pentagon Area vs Performance (Pearson r):")
print(f"  {'Method':<10} {'WAR':>8} {'wRC+':>8} {'OPS':>8}")
for m in method_names:
    print(f"  {m:<10} {corr[m]['WAR']:>8.4f} {corr[m]['wRC+']:>8.4f} {corr[m]['OPS']:>8.4f}")

plot_correlation_heatmap(corr, method_names, validation_metrics)

corr_df = pd.DataFrame(corr).T
corr_df.to_csv(TABLES_DIR / "correlations.csv")
print(f"\n  Saved: {TABLES_DIR / 'correlations.csv'}")
