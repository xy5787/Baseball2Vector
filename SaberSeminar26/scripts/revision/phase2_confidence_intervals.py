"""Phase 2: Player-level cluster bootstrap CIs for Table 1's pentagon-area correlations.

Table 1 of the paper reports pooled Pearson correlations between pentagon area and
WAR/wRC+/OPS across all 2,309 season-rows, for three encodings (Z-score, PCA, JointVAE). Because
a given player contributes up to 5 season-rows that are not independent draws (a good player
tends to have several good seasons), the naive row-level standard error understates the true
uncertainty. This script instead resamples *players* with replacement (a cluster/block
bootstrap), taking every season-row belonging to each resampled player, and recomputes the
pooled correlation each time.

Output: outputs/revision/phase2_confidence_intervals/table1_with_ci.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase2_confidence_intervals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 2000
METHODS = ["ZScore", "PCA", "JointVAE"]
METRICS = ["WAR", "wRC+", "OPS"]


def run_joint_bootstrap(
    df: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """One shared set of player resamples per iteration, correlations for all 9 combos.

    Drawing all (method, metric) correlations from the *same* resampled players each
    iteration (rather than resampling independently per combo) lets us also report paired
    differences between encodings (e.g. ZScore - PCA) with the correct joint sampling
    distribution, instead of just eyeballing overlap between two marginal CIs.
    """
    players = df["Name"].unique()
    n_players = len(players)
    rows_by_player = {p: df.index[df["Name"] == p].to_numpy() for p in players}

    combos = [(m, met) for m in METHODS for met in METRICS]
    boot = {f"{m}_{met}": np.empty(n_boot) for m, met in combos}

    for b in range(n_boot):
        sampled_players = rng.choice(players, size=n_players, replace=True)
        idx = np.concatenate([rows_by_player[p] for p in sampled_players])
        sample = df.loc[idx]
        for m, met in combos:
            boot[f"{m}_{met}"][b] = sample[f"{m}_Area"].corr(sample[met])

    return boot


def main() -> None:
    df = pd.read_csv(DATA)
    rng = np.random.default_rng(SEED)

    boot_dists = run_joint_bootstrap(df, N_BOOT, rng)

    rows = []
    for method in METHODS:
        area_col = f"{method}_Area"
        for metric in METRICS:
            point = float(df[area_col].corr(df[metric]))
            dist = boot_dists[f"{method}_{metric}"]
            lo, hi = np.percentile(dist, [2.5, 97.5])
            rows.append(
                {
                    "encoding": method,
                    "metric": metric,
                    "r": round(point, 4),
                    "ci_low": round(float(lo), 4),
                    "ci_high": round(float(hi), 4),
                    "ci_width": round(float(hi - lo), 4),
                    "n_boot": N_BOOT,
                    "n_players": df["Name"].nunique(),
                    "n_rows": len(df),
                }
            )
            print(f"{method:9s} vs {metric:5s}: r={point:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "table1_with_ci.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'table1_with_ci.csv'}")

    np.savez(OUT_DIR / "bootstrap_distributions.npz", **boot_dists)
    print(f"Saved: {OUT_DIR / 'bootstrap_distributions.npz'}")

    # Paired encoding-vs-encoding comparisons (same bootstrap draws -> valid difference CI).
    print("\nPaired encoding comparisons (95% CI on r_A - r_B; excludes 0 => significant):")
    pair_rows = []
    pairs = [("ZScore", "PCA"), ("ZScore", "JointVAE"), ("PCA", "JointVAE")]
    for metric in METRICS:
        for a, b_ in pairs:
            diff = boot_dists[f"{a}_{metric}"] - boot_dists[f"{b_}_{metric}"]
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            point_diff = float(df[f"{a}_Area"].corr(df[metric])) - float(
                df[f"{b_}_Area"].corr(df[metric])
            )
            significant = not (d_lo <= 0 <= d_hi)
            pair_rows.append(
                {
                    "metric": metric,
                    "encoding_a": a,
                    "encoding_b": b_,
                    "r_diff": round(point_diff, 4),
                    "diff_ci_low": round(float(d_lo), 4),
                    "diff_ci_high": round(float(d_hi), 4),
                    "significant_95": significant,
                }
            )
            print(
                f"  {metric:5s} {a:9s} - {b_:9s} = {point_diff:+.3f}  "
                f"95% CI=[{d_lo:+.3f}, {d_hi:+.3f}]  "
                f"{'SIGNIFICANT' if significant else 'not significant'}"
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT_DIR / "encoding_pairwise_diffs.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'encoding_pairwise_diffs.csv'}")


if __name__ == "__main__":
    main()
