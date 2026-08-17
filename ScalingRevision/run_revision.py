"""Re-scale ZScore/PCA/JointVAE tool grades to a proper mean=50, SD=10 20-80
scale (per https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/)
and validate alignment with WAR / wRC+ / OPS, alongside the original
MinMax-based scaling.

Why this script reads from data/v3_results_clean.csv instead of retraining
------------------------------------------------------------------------
The stored ZScore_*/PCA_*/JointVAE_* columns in v3_results_clean.csv are
already MinMax-scaled to [20, 80] by the original pipeline
(``baseball2vec.tools.scale_to_2080``). MinMax scaling is a strictly
increasing affine map: scaled = a*raw + b (a = 60/(max-min) > 0).

Z-scoring is invariant to any positive affine transform of its input:
    z((a*x + b)) = (a*x+b - mean(a*x+b)) / std(a*x+b)
                 = a*(x - mean(x)) / (a*std(x))
                 = z(x)

So re-standardizing the stored (already-scaled) columns gives *exactly* the
same result as re-standardizing the model's true raw latent scores would —
no VAE retraining needed, and no risk of the retrained VAE landing in a
different local optimum than the one used for the paper's results.

Outputs
-------
ScalingRevision/outputs/rescaled_results.csv   -- per-player old + new grades/areas
ScalingRevision/outputs/correlation_comparison.csv -- Pearson r, old vs new
ScalingRevision/outputs/clip_fractions.csv     -- % of grades clipped at 20/80, new scale
ScalingRevision/outputs/report.md              -- written explanation of findings
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from baseball2vec.tools import TOOL_NAMES, pentagon_area
from baseball2vec.viz import (
    plot_correlation_heatmap,
    plot_radar,
    plot_scatter_grid,
    plot_uncertainty,
)
from scaling import clip_fraction, scale_to_2080_sd

DATA_PATH = REPO_ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["ZScore", "PCA", "JointVAE"]
VALIDATION_METRICS = ["WAR", "wRC+", "OPS"]

print("=" * 60)
print("ScalingRevision: mean=50, SD=10 20-80 scale (Fangraphs convention)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} player-seasons from {DATA_PATH}")

# ---------------------------------------------------------------------------
# Sanity check: recomputing pentagon_area from the stored per-tool grades
# must reproduce the stored *_Area column exactly (validates pentagon_area
# usage below before trusting it on the new scale).
# ---------------------------------------------------------------------------
print("\n[sanity check] recomputed vs stored Pentagon Area (old MinMax scale)")
for m in METHODS:
    cols = [f"{m}_{t}" for t in TOOL_NAMES]
    old_grades = df[cols].copy()
    old_grades.columns = TOOL_NAMES
    recomputed = old_grades.apply(pentagon_area, axes=TOOL_NAMES, axis=1)
    max_diff = (recomputed - df[f"{m}_Area"]).abs().max()
    print(f"  {m:<10} max |diff| = {max_diff:.2e}")
    # Tolerance accounts for CSV round-trip precision loss (Area values are
    # O(1e3-1e4), so 1e-2 absolute is a relative error of ~1e-6).
    assert max_diff < 1e-2, f"pentagon_area mismatch for {m}"

# ---------------------------------------------------------------------------
# Apply the new SD-based scaling per method, recompute Pentagon Area
# ---------------------------------------------------------------------------
result_df = df[["UniqueName", "Name", "Season", "PA"] + VALIDATION_METRICS].copy()

clip_report = {}
for m in METHODS:
    cols = [f"{m}_{t}" for t in TOOL_NAMES]
    raw = df[cols].copy()
    raw.columns = TOOL_NAMES  # rename to plain tool names for scaling/pentagon_area

    new_grades = scale_to_2080_sd(raw, TOOL_NAMES)
    clip_report[m] = clip_fraction(raw, TOOL_NAMES)

    for t in TOOL_NAMES:
        result_df[f"{m}_{t}_old"] = df[f"{m}_{t}"].values
        result_df[f"{m}_{t}_new"] = new_grades[t].values

    result_df[f"{m}_Area_old"] = df[f"{m}_Area"].values
    result_df[f"{m}_Area_new"] = new_grades.apply(
        pentagon_area, axes=TOOL_NAMES, axis=1
    ).values

result_df.to_csv(OUT_DIR / "rescaled_results.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'rescaled_results.csv'}")

clip_df = pd.DataFrame(clip_report).T
clip_df.columns = [f"{t}_clip_frac" for t in TOOL_NAMES]
clip_df.to_csv(OUT_DIR / "clip_fractions.csv")
print(f"Saved: {OUT_DIR / 'clip_fractions.csv'}")

# ---------------------------------------------------------------------------
# Correlation comparison: Pentagon Area vs WAR / wRC+ / OPS, old vs new scale
# ---------------------------------------------------------------------------
rows = []
for m in METHODS:
    row_old = {"Method": m, "Scale": "MinMax (old)"}
    row_new = {"Method": m, "Scale": "Mean50/SD10 (new)"}
    for metric in VALIDATION_METRICS:
        row_old[metric] = result_df[f"{m}_Area_old"].corr(result_df[metric])
        row_new[metric] = result_df[f"{m}_Area_new"].corr(result_df[metric])
    rows.append(row_old)
    rows.append(row_new)

corr_df = pd.DataFrame(rows)
corr_df.to_csv(OUT_DIR / "correlation_comparison.csv", index=False)
print(f"Saved: {OUT_DIR / 'correlation_comparison.csv'}")

print("\nPentagon Area vs Performance (Pearson r) — old vs new scale")
print(f"  {'Method':<10} {'Scale':<20} {'WAR':>8} {'wRC+':>8} {'OPS':>8}")
for _, r in corr_df.iterrows():
    print(
        f"  {r['Method']:<10} {r['Scale']:<20} "
        f"{r['WAR']:>8.4f} {r['wRC+']:>8.4f} {r['OPS']:>8.4f}"
    )

print("\nClip fraction at 20/80 under new (mean=50, SD=10) scale:")
print(clip_df.round(4).to_string())

# ---------------------------------------------------------------------------
# Per-tool grade shift diagnostics (mean absolute change old -> new)
# ---------------------------------------------------------------------------
print("\nMean |old - new| grade shift per tool:")
shift_rows = []
for m in METHODS:
    shifts = {
        t: (result_df[f"{m}_{t}_old"] - result_df[f"{m}_{t}_new"]).abs().mean()
        for t in TOOL_NAMES
    }
    shifts["Method"] = m
    shift_rows.append(shifts)
shift_df = pd.DataFrame(shift_rows).set_index("Method")[TOOL_NAMES]
print(shift_df.round(2).to_string())
shift_df.to_csv(OUT_DIR / "grade_shift.csv")

# ---------------------------------------------------------------------------
# Figures — mirrors the 02/03 script figure set, for old and new scale.
#
# NOTE: v3_covariance_heatmap (tool tradeoffs from the VAE's full covariance
# Sigma = LL^T) is NOT regenerated here — that matrix is only produced during
# JointVAE training and isn't stored in v3_results_clean.csv. Regenerating it
# requires retraining the VAE, which requires data/raw/batting_stats_2021_2025.csv
# (fetch blocked: FanGraphs returned 403 in this environment). See report_ko.md.
# ---------------------------------------------------------------------------
SIGMA_COLS = [f"{t}_sigma" for t in TOOL_NAMES]
BASE_COLS = ["UniqueName", "Name", "Season", "PA"] + VALIDATION_METRICS

result_old = df[BASE_COLS + SIGMA_COLS].copy()
result_new = df[BASE_COLS + SIGMA_COLS].copy()
corr_old: dict[str, dict[str, float]] = {}
corr_new: dict[str, dict[str, float]] = {}

for m in METHODS:
    for t in TOOL_NAMES:
        result_old[f"{m}_{t}"] = result_df[f"{m}_{t}_old"].values
        result_new[f"{m}_{t}"] = result_df[f"{m}_{t}_new"].values
    result_old[f"{m}_Area"] = result_df[f"{m}_Area_old"].values
    result_new[f"{m}_Area"] = result_df[f"{m}_Area_new"].values
    corr_old[m] = {
        metric: float(result_old[f"{m}_Area"].corr(result_old[metric]))
        for metric in VALIDATION_METRICS
    }
    corr_new[m] = {
        metric: float(result_new[f"{m}_Area"].corr(result_new[metric]))
        for metric in VALIDATION_METRICS
    }

FIG_OLD_DIR = OUT_DIR / "figures" / "old"
FIG_NEW_DIR = OUT_DIR / "figures" / "new"

print("\nGenerating figures...")
for corr, result, fig_dir, label in [
    (corr_old, result_old, FIG_OLD_DIR, "old (MinMax)"),
    (corr_new, result_new, FIG_NEW_DIR, "new (mean=50, SD=10)"),
]:
    print(f"  [{label}] -> {fig_dir}")
    plot_correlation_heatmap(corr, METHODS, VALIDATION_METRICS, figures_dir=fig_dir)
    plot_scatter_grid(result, METHODS, VALIDATION_METRICS, figures_dir=fig_dir)
    for p in [
        "Aaron Judge (2024)",
        "Juan Soto (2024)",
        "Bobby Witt Jr. (2024)",
        "Shohei Ohtani (2024)",
    ]:
        plot_radar(p, result, TOOL_NAMES, figures_dir=fig_dir)
    plot_uncertainty(result, figures_dir=fig_dir)

print("\nDone.")
