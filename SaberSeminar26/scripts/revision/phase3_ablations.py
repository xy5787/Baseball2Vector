"""Phase 3: Ablations on two heuristic choices flagged in the paper as
"heuristic hyperparameters, not estimated" (shrinkage prior strength) and
"population-level, axis-order-sensitive for individuals" (pentagon-area permutation, Sec. 4.4).

3a. Shrinkage prior strength sensitivity (Z-score encoding).
3b. Axis-permutation sensitivity for all three encodings (paper's Sec. 4.4 currently reports
    this only for Z-score; here we compute it for PCA and JointVAE too).

IMPORTANT constraint on 3a
---------------------------
Raw FanGraphs batting stats are not available in this environment: Phase 0 found data/raw/ empty,
and a live fetch attempt in this session (see REVISIONS.md, Phase 3 entry) confirms FanGraphs
returns HTTP 403 to this environment's IP, matching the README's documented "may be blocked from
server environments" warning. The Bayesian stabilization step (src/baseball2vec/data.py) that the
paper's shrinkage-strength hyperparameter controls operates on *raw* per-stat columns (K%, BB%,
AVG, ...) before Z-scoring -- those raw values are not retained anywhere in
data/v3_results_clean.csv, only the final post-Z-score, post-20-80-scaling composite Tool scores.

Given this, a literal re-run of the pipeline under alternative PA-equivalent thresholds is not
possible here. This script instead runs a clearly-labeled PROXY ablation that tests the same
underlying question -- "how much does Table 1's r move if the shrinkage-toward-league-average is
stronger or weaker?" -- by directly blending each tool's already-computed composite Z-score
average toward its own population mean by an extra fraction lambda:

    Tool_shrunk(lambda) = (1 - lambda) * Tool_zscore + lambda * mean(Tool_zscore)

This can only add shrinkage (lambda > 0), not remove it: the original stabilization already
collapsed information (raw values were shrunk toward the league mean before being saved), so
there is no way to recover a "less-shrunk" version of the composite score from the processed
dataset alone. This asymmetry is a real limitation of the proxy and is reported as such --the
result should be read as "how sensitive is Table 1's r to additional shrinkage," not literally
"+/-50% of the current thresholds."
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baseball2vec.tools import TOOL_NAMES, pentagon_area  # noqa: E402

DATA = ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase3_ablations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["WAR", "wRC+", "OPS"]
METHODS = ["ZScore", "PCA", "JointVAE"]
SHRINK_LAMBDAS = [0.0, 0.10, 0.25, 0.50]  # 0.0 = current pipeline (no extra shrinkage)


def run_shrinkage_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Blend each tool's composite Z-score toward its own population mean.

    Deliberately does NOT re-apply `scale_to_2080` after blending: a MinMaxScaler fit fresh on
    each lambda's output would always re-stretch the shrunk distribution back out to fill
    exactly [20, 80], which is a no-op (this was verified empirically -- an earlier version of
    this script did exactly that and every result was identical to the lambda=0 baseline to
    floating-point precision, because independently re-normalizing each shrunk column to a fixed
    [min, max] target range exactly cancels the shrinkage). Instead, all lambda values share one
    fixed reference frame (the lambda=0 / current-pipeline ZScore_{Tool} columns, already on the
    20-80 scale), and shrinkage genuinely compresses each tool's spread within that fixed frame.
    Pentagon area is a nonlinear (product) combination of the five tool values, so this
    compression does change area's correlation with WAR/wRC+/OPS, unlike a simple pairwise
    Pearson r which is invariant to per-variable affine rescaling.
    """
    rows = []
    for lam in SHRINK_LAMBDAS:
        shrunk = pd.DataFrame(index=df.index)
        for t in TOOL_NAMES:
            col = f"ZScore_{t}"
            mean_val = df[col].mean()
            shrunk[t] = (1 - lam) * df[col] + lam * mean_val

        area = shrunk.apply(pentagon_area, axes=TOOL_NAMES, axis=1)

        row = {"shrink_lambda": lam}
        for metric in METRICS:
            row[f"r_{metric}"] = float(area.corr(df[metric]))
        rows.append(row)

    result = pd.DataFrame(rows)
    baseline = result.loc[result["shrink_lambda"] == 0.0].iloc[0]
    for metric in METRICS:
        result[f"delta_r_{metric}"] = result[f"r_{metric}"] - baseline[f"r_{metric}"]
    return result


def run_permutation_ablation(df: pd.DataFrame, n_perms: int | None = None) -> pd.DataFrame:
    """Recompute pentagon area under every axis permutation, for all three encodings."""
    all_perms = list(itertools.permutations(TOOL_NAMES))
    if n_perms is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_perms), size=n_perms, replace=False)
        perms = [all_perms[i] for i in idx]
    else:
        perms = all_perms

    rows = []
    for method in METHODS:
        cols = [f"{method}_{t}" for t in TOOL_NAMES]
        sub = df[cols].copy()
        sub.columns = TOOL_NAMES  # pentagon_area expects axes named after tools

        r_by_metric = {m: [] for m in METRICS}
        for perm in perms:
            area = sub.apply(pentagon_area, axes=list(perm), axis=1)
            for metric in METRICS:
                r_by_metric[metric].append(float(area.corr(df[metric])))

        row = {"encoding": method, "n_permutations": len(perms)}
        for metric in METRICS:
            vals = np.array(r_by_metric[metric])
            row[f"{metric}_r_min"] = round(float(vals.min()), 4)
            row[f"{metric}_r_max"] = round(float(vals.max()), 4)
            row[f"{metric}_r_band"] = round(float(vals.max() - vals.min()), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(DATA)

    print("=== 3a. Shrinkage-strength sensitivity (Z-score encoding) ===")
    shrink_df = run_shrinkage_ablation(df)
    print(shrink_df.to_string(index=False))
    shrink_df.to_csv(OUT_DIR / "shrinkage_sensitivity.csv", index=False)
    print(f"Saved: {OUT_DIR / 'shrinkage_sensitivity.csv'}")

    print("\n=== 3b. Axis-permutation sensitivity, all three encodings (all 120 permutations) ===")
    perm_df = run_permutation_ablation(df, n_perms=None)
    print(perm_df.to_string(index=False))
    perm_df.to_csv(OUT_DIR / "permutation_sensitivity_all_encodings.csv", index=False)
    print(f"Saved: {OUT_DIR / 'permutation_sensitivity_all_encodings.csv'}")


if __name__ == "__main__":
    main()
