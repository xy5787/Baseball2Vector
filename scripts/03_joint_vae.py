"""Train the Joint Encoder VAE and produce all v3 outputs.

Inputs
------
data/raw/batting_stats_2021_2025.csv

Outputs
-------
data/processed/v3_results.csv
outputs/figures/v3_correlation_heatmap.png
outputs/figures/v3_covariance_heatmap.png
outputs/figures/v3_scatter_grid.png
outputs/figures/v3_radar_*.png
outputs/figures/v3_uncertainty.png
outputs/tables/v3_summary.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import random
import numpy as np
import pandas as pd
import torch

from baseball2vec.baselines import (
    build_scaled_results,
    correlation_table,
    pca_tool_scores,
    zscore_tool_scores,
)
from baseball2vec.data import load_raw, preprocess
from baseball2vec.joint_vae import train as train_vae
from baseball2vec.tools import (
    TOOL_NAMES,
    apply_direction,
    build_final_groups,
    season_zscore,
)
from baseball2vec.viz import (
    plot_covariance_heatmap,
    plot_correlation_heatmap,
    plot_radar,
    plot_scatter_grid,
    plot_uncertainty,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
TABLES_DIR = Path(__file__).parents[1] / "outputs" / "tables"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 3: Joint Encoder VAE")
print("=" * 60)

raw = load_raw()
df = preprocess(raw)
final_groups = build_final_groups(df)
all_features = [c for cols in final_groups.values() for c in cols]
group_dims = {g: len(cols) for g, cols in final_groups.items()}

X_df = season_zscore(df, all_features)
X_aligned = apply_direction(X_df)

print("\nBaselines...")
zscore_tools = zscore_tool_scores(X_aligned, final_groups)
pca_tools, pca_explained = pca_tool_scores(X_aligned, final_groups)

print("\nVAE...")
model, vae_mu, vae_sigma, mean_corr = train_vae(
    X_aligned, all_features, final_groups, group_dims, df["PA"]
)

vae_tools = pd.DataFrame(vae_mu, columns=TOOL_NAMES)
sigma_df = pd.DataFrame(vae_sigma, columns=[f"{t}_sigma" for t in TOOL_NAMES])

methods = {"ZScore": zscore_tools, "PCA": pca_tools, "JointVAE": vae_tools}
scaled = build_scaled_results(methods)

result_df = df[["UniqueName", "Name", "Season", "PA"]].copy().reset_index(drop=True)
for metric in ["WAR", "wRC+", "OPS"]:
    if metric in df.columns:
        result_df[metric] = df[metric].values
for name, sdf in scaled.items():
    result_df = pd.concat([result_df, sdf.reset_index(drop=True)], axis=1)
result_df = pd.concat([result_df, sigma_df.reset_index(drop=True)], axis=1)

method_names = ["ZScore", "PCA", "JointVAE"]
validation_metrics = ["WAR", "wRC+", "OPS"]
corr = correlation_table(result_df, method_names, validation_metrics)

print("\n  Pentagon Area vs Performance (Pearson r):")
print(f"  {'Method':<12} {'WAR':>8} {'wRC+':>8} {'OPS':>8}")
for m in method_names:
    print(
        f"  {m:<12} {corr[m]['WAR']:>8.4f} {corr[m]['wRC+']:>8.4f} {corr[m]['OPS']:>8.4f}"
    )

plot_correlation_heatmap(corr, method_names, validation_metrics)
plot_scatter_grid(result_df, method_names, validation_metrics)
plot_covariance_heatmap(mean_corr, TOOL_NAMES)
for p in [
    "Aaron Judge (2024)",
    "Juan Soto (2024)",
    "Bobby Witt Jr. (2024)",
    "Shohei Ohtani (2024)",
]:
    plot_radar(p, result_df, TOOL_NAMES)
plot_uncertainty(result_df)

result_df.to_csv(PROCESSED_DIR / "v3_results.csv", index=False)
print(f"\n  Saved: {PROCESSED_DIR / 'v3_results.csv'} ({len(result_df)} rows)")

with open(TABLES_DIR / "v3_summary.txt", "w") as f:
    f.write("Baseball2Vec — Joint Encoder VAE (v3) Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(
        f"Data: {len(df)} player-seasons ({min(df['Season'])}-{max(df['Season'])})\n"
    )
    f.write(f"Tools: {', '.join(TOOL_NAMES)}\n\n")
    f.write("Pentagon Area vs Performance (Pearson r)\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Method':<12} {'WAR':>8} {'wRC+':>8} {'OPS':>8}\n")
    f.write("-" * 50 + "\n")
    for m in method_names:
        f.write(
            f"{m:<12} {corr[m]['WAR']:>8.4f} {corr[m]['wRC+']:>8.4f} {corr[m]['OPS']:>8.4f}\n"
        )
    f.write("\nTool Tradeoff Matrix (league-avg corr from VAE Σ)\n")
    import pandas as _pd

    corr_matrix_df = _pd.DataFrame(mean_corr, index=TOOL_NAMES, columns=TOOL_NAMES)
    f.write(corr_matrix_df.round(3).to_string())
print(f"  Saved: {TABLES_DIR / 'v3_summary.txt'}")
