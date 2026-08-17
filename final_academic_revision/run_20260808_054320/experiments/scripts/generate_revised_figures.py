"""Generate the three corrected manuscript figures from saved/verified data."""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import N_BOOT, RUN_ROOT, SEED, TOOL_COLS, TOOL_NAMES, build_transitions, expanded_cluster_indices, load_player_seasons, metric, percentile_ci

COLORS = ["#3B6EA8", "#E58B2A", "#8A5A83"]


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(RUN_ROOT / "figures" / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RUN_ROOT / "figures" / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure1(players: pd.DataFrame) -> None:
    x = players["ZScore_Area"].to_numpy(float)
    y = players["WAR"].to_numpy(float)
    ids = players["player_id"].to_numpy()
    unique = np.unique(ids)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = expanded_cluster_indices(ids, sampled)
        values.append(metric(x[idx], y[idx], "pearson"))
    estimate = metric(x, y, "pearson")
    low, high = percentile_ci(values)
    pd.DataFrame([{
        "n_player_seasons": len(players), "n_players": len(unique), "pearson_r": estimate,
        "ci_low": low, "ci_high": high, "bootstrap_unit": "stable player ID", "n_boot": N_BOOT,
    }]).to_csv(RUN_ROOT / "results" / "figure1_alignment_result.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    ax.scatter(x, y, s=9, alpha=0.30, color=COLORS[0], linewidths=0)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, slope * grid + intercept, color="#A33A3A", linewidth=1.8,
            label=f"r = {estimate:.2f}; player-clustered 95% CI [{low:.2f}, {high:.2f}]")
    ax.set_xlabel("Pentagon area (Z-score encoding)")
    ax.set_ylabel("WAR")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    save_both(fig, "figure1_alignment_revised")


def figure2() -> None:
    pairs = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distances.csv")
    example = pd.read_csv(RUN_ROOT / "results" / "illustrative_matched_pair.csv").iloc[0]
    groups = ["same_player_adjacent", "scalar_matched", "random_same_season_pa_stratum"]
    labels = ["Same player\nadjacent", "Scalar-matched\ndifferent players", "Random same-season\nPA-stratified"]
    distributions = [pairs.loc[pairs["pair_type"].eq(group), "profile_distance"].to_numpy() for group in groups]

    fig = plt.figure(figsize=(10.2, 4.15))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    violins = ax.violinplot(distributions, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(violins["bodies"], COLORS):
        body.set_facecolor(color); body.set_edgecolor("black"); body.set_alpha(0.68)
    ax.boxplot(distributions, widths=0.15, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "white", "linewidth": 0.8},
               medianprops={"color": "black", "linewidth": 1.4})
    ax.set_xticks([1, 2, 3], labels, fontsize=8.5)
    ax.set_ylabel("Standardized five-dimensional distance")
    ax.set_title("(a) Pair distributions", fontsize=11)
    ax.grid(axis="y", alpha=0.20)

    radar = fig.add_subplot(grid[0, 1], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(TOOL_COLS), endpoint=False)
    closed = np.r_[angles, angles[0]]
    for prefix, color, label in [
        ("focal", COLORS[0], f"{example['focal_name']} ({int(example['focal_season'])})"),
        ("comparison", COLORS[1], f"{example['comparison_name']} ({int(example['comparison_season'])})"),
    ]:
        values = np.array([example[f"{prefix}_{col}"] for col in TOOL_COLS], dtype=float)
        values = np.r_[values, values[0]]
        radar.plot(closed, values, color=color, linewidth=2, label=label)
        radar.fill(closed, values, color=color, alpha=0.10)
    radar.set_theta_offset(np.pi / 2); radar.set_theta_direction(-1)
    radar.set_xticks(angles, TOOL_NAMES, fontsize=8)
    radar.set_ylim(20, 80); radar.set_yticks([20, 40, 60, 80]); radar.set_yticklabels(["20", "40", "60", "80"], fontsize=7)
    radar.set_title("(b) Illustrative matched pair", y=1.12, fontsize=11)
    radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=7.5, frameon=False)
    fig.subplots_adjust(bottom=0.18)
    save_both(fig, "figure2_profile_diversity_revised")


def figure3(players: pd.DataFrame) -> None:
    transitions = build_transitions(players)
    xcols = [f"{col}_t" for col in TOOL_COLS]
    ycols = [f"{col}_t1" for col in TOOL_COLS]
    x = transitions[ycols].to_numpy(float) - transitions[xcols].to_numpy(float)
    y = transitions["wRC+_t1"].to_numpy(float) - transitions["wRC+_t"].to_numpy(float)
    ids = transitions["player_id"].to_numpy()
    model = LinearRegression().fit(x, y)
    estimate = model.coef_
    rng = np.random.default_rng(SEED)
    unique = np.unique(ids)
    boot = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = expanded_cluster_indices(ids, sampled)
        boot.append(LinearRegression().fit(x[idx], y[idx]).coef_)
    boot = np.asarray(boot)
    rows = []
    for index, dimension in enumerate(TOOL_NAMES):
        low, high = percentile_ci(boot[:, index])
        rows.append({
            "dimension": dimension, "coefficient": estimate[index], "ci_low": low, "ci_high": high,
            "n_transitions": len(transitions), "n_players": len(unique),
            "bootstrap_unit": "stable player ID", "n_boot": N_BOOT,
        })
    results = pd.DataFrame(rows)
    results.to_csv(RUN_ROOT / "results" / "delta_association_coefficients.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    positions = np.arange(len(results))
    errors = np.vstack([results["coefficient"] - results["ci_low"], results["ci_high"] - results["coefficient"]])
    ax.errorbar(results["coefficient"], positions, xerr=errors, fmt="o", color=COLORS[0],
                ecolor="#4C4C4C", capsize=3, linewidth=1.3)
    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(positions, results["dimension"])
    ax.invert_yaxis()
    ax.set_xlabel("Partial association: change in wRC+ per one-point dimension change")
    ax.set_title("Concurrent Dimension-Change Associations with ΔwRC+")
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    save_both(fig, "figure3_delta_association_revised")


def main() -> None:
    players = load_player_seasons()
    figure1(players)
    figure2()
    figure3(players)
    print("Generated revised Figures 1-3 and machine-readable figure results.")


if __name__ == "__main__":
    main()
