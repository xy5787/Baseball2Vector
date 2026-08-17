"""Re-run of scripts/03_joint_vae.py: train the Joint Encoder VAE from scratch
and compare the old (MinMax) vs new (mean=50, SD=10) 20-80 scaling on its
output, alongside the Z-Score / PCA baselines.

The VAE is trained ONCE (same seed/hyperparameters as the original script);
both scalings are then applied post-hoc to the same trained latent means, so
this is an apples-to-apples comparison of the scaling step alone, on a
freshly-retrained model (not the archived data/v3_results_clean.csv).

All outputs are written under ScalingRevision/outputs/ — the original
pipeline's data/ and outputs/ directories are untouched.

Outputs
-------
ScalingRevision/outputs/data/v3_results_revision.csv   -- per-player old+new grades/areas
ScalingRevision/outputs/tables/v3_summary_old.txt
ScalingRevision/outputs/tables/v3_summary_new.txt
ScalingRevision/outputs/tables/correlation_comparison_retrained.csv
ScalingRevision/outputs/figures/old/v3_*.png
ScalingRevision/outputs/figures/new/v3_*.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    pentagon_area,
    season_zscore,
)
from baseball2vec.viz import (
    plot_covariance_heatmap,
    plot_correlation_heatmap,
    plot_radar,
    plot_scatter_grid,
    plot_uncertainty,
)
from scaling import build_scaled_results_sd, clip_fraction

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DATA_DIR = OUT_DIR / "data"
TABLES_DIR = OUT_DIR / "tables"
FIG_OLD_DIR = OUT_DIR / "figures" / "old"
FIG_NEW_DIR = OUT_DIR / "figures" / "new"
for d in (DATA_DIR, TABLES_DIR, FIG_OLD_DIR, FIG_NEW_DIR):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ScalingRevision Step 3: Joint Encoder VAE (retrained from scratch)")
print("  MinMax (old) vs mean=50/SD=10 (new)")
print("=" * 60)

raw = load_raw()
df = preprocess(raw)
final_groups = build_final_groups(df)
all_features = [c for cols in final_groups.values() for c in cols]
group_dims = {g: len(cols) for g, cols in final_groups.items()}

X_df = season_zscore(df, all_features)
X_aligned = apply_direction(X_df)

print(f"\n  Player-seasons: {len(df)} ({sorted(df['Season'].unique())})")

print("\nBaselines...")
zscore_tools = zscore_tool_scores(X_aligned, final_groups)
pca_tools, pca_explained = pca_tool_scores(X_aligned, final_groups)

print("\nVAE (single training run, reused for both scalings)...")
model, vae_mu, vae_sigma, mean_corr = train_vae(
    X_aligned, all_features, final_groups, group_dims, df["PA"]
)

vae_tools = pd.DataFrame(vae_mu, columns=TOOL_NAMES)
sigma_df = pd.DataFrame(vae_sigma, columns=[f"{t}_sigma" for t in TOOL_NAMES])

methods = {"ZScore": zscore_tools, "PCA": pca_tools, "JointVAE": vae_tools}
method_names = ["ZScore", "PCA", "JointVAE"]
validation_metrics = ["WAR", "wRC+", "OPS"]

scaled_old = build_scaled_results(methods)
scaled_new = build_scaled_results_sd(methods, TOOL_NAMES)

base_cols = df[["UniqueName", "Name", "Season", "PA"]].copy().reset_index(drop=True)
for metric in validation_metrics:
    if metric in df.columns:
        base_cols[metric] = df[metric].values

result_old = base_cols.copy()
for name, sdf in scaled_old.items():
    result_old = pd.concat([result_old, sdf.reset_index(drop=True)], axis=1)
result_old = pd.concat([result_old, sigma_df.reset_index(drop=True)], axis=1)

result_new = base_cols.copy()
for name, sdf in scaled_new.items():
    result_new = pd.concat([result_new, sdf.reset_index(drop=True)], axis=1)
result_new = pd.concat([result_new, sigma_df.reset_index(drop=True)], axis=1)

corr_old = correlation_table(result_old, method_names, validation_metrics)
corr_new = correlation_table(result_new, method_names, validation_metrics)

print("\n  Pentagon Area vs Performance (Pearson r)")
print(f"  {'Method':<12} {'Scale':<16} {'WAR':>8} {'wRC+':>8} {'OPS':>8}")
for m in method_names:
    print(
        f"  {m:<12} {'MinMax(old)':<16} "
        f"{corr_old[m]['WAR']:>8.4f} {corr_old[m]['wRC+']:>8.4f} {corr_old[m]['OPS']:>8.4f}"
    )
    print(
        f"  {m:<12} {'SD10(new)':<16} "
        f"{corr_new[m]['WAR']:>8.4f} {corr_new[m]['wRC+']:>8.4f} {corr_new[m]['OPS']:>8.4f}"
    )

# --- Figures: full v3 figure set, once per scale ---
for corr, result_df, fig_dir in [
    (corr_old, result_old, FIG_OLD_DIR),
    (corr_new, result_new, FIG_NEW_DIR),
]:
    plot_correlation_heatmap(corr, method_names, validation_metrics, figures_dir=fig_dir)
    plot_scatter_grid(result_df, method_names, validation_metrics, figures_dir=fig_dir)
    plot_covariance_heatmap(mean_corr, TOOL_NAMES, figures_dir=fig_dir)
    for p in [
        "Aaron Judge (2024)",
        "Juan Soto (2024)",
        "Bobby Witt Jr. (2024)",
        "Shohei Ohtani (2024)",
    ]:
        plot_radar(p, result_df, TOOL_NAMES, figures_dir=fig_dir)
    plot_uncertainty(result_df, figures_dir=fig_dir)

# --- Combined per-player CSV (old + new grades/areas side by side) ---
combined = base_cols.copy()
for m in method_names:
    for t in TOOL_NAMES:
        combined[f"{m}_{t}_old"] = result_old[f"{m}_{t}"].values
        combined[f"{m}_{t}_new"] = result_new[f"{m}_{t}"].values
    combined[f"{m}_Area_old"] = result_old[f"{m}_Area"].values
    combined[f"{m}_Area_new"] = result_new[f"{m}_Area"].values
combined = pd.concat([combined, sigma_df.reset_index(drop=True)], axis=1)
combined.to_csv(DATA_DIR / "v3_results_revision.csv", index=False)
print(f"\n  Saved: {DATA_DIR / 'v3_results_revision.csv'} ({len(combined)} rows)")

# --- Comparison correlation CSV ---
rows = []
for m in method_names:
    row_old = {"Method": m, "Scale": "MinMax (old)"}
    row_new = {"Method": m, "Scale": "Mean50/SD10 (new)"}
    for metric in validation_metrics:
        row_old[metric] = corr_old[m][metric]
        row_new[metric] = corr_new[m][metric]
    rows.append(row_old)
    rows.append(row_new)
comp_df = pd.DataFrame(rows)
comp_df.to_csv(TABLES_DIR / "correlation_comparison_retrained.csv", index=False)
print(f"  Saved: {TABLES_DIR / 'correlation_comparison_retrained.csv'}")

# --- Clip-fraction diagnostic (new scale only) ---
clip_rows = {}
for m in method_names:
    raw_cols = pd.DataFrame(
        {t: result_old[f"{m}_{t}"] for t in TOOL_NAMES}
    )  # any positive-affine-equivalent input works; reuse old-scale grades
    clip_rows[m] = clip_fraction(raw_cols, TOOL_NAMES)
clip_df = pd.DataFrame(clip_rows).T
clip_df.to_csv(TABLES_DIR / "clip_fractions_retrained.csv")
print(f"  Saved: {TABLES_DIR / 'clip_fractions_retrained.csv'}")

# --- Summary text files ---
for label, corr, fname in [
    ("old (MinMax)", corr_old, "v3_summary_old.txt"),
    ("new (mean=50, SD=10)", corr_new, "v3_summary_new.txt"),
]:
    with open(TABLES_DIR / fname, "w") as f:
        f.write(f"Baseball2Vec — ScalingRevision Joint Encoder VAE ({label})\n")
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
    print(f"  Saved: {TABLES_DIR / fname}")

print("\nDone.")
