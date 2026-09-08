"""Generate audited figures, LaTeX tables, and reproduction metadata."""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import N_BOOT, REPO_ROOT, RUN_ROOT, SEED, TOOL_COLS, percentile_ci, sha256_file


RESULTS = RUN_ROOT / "results"
FIGURES = RUN_ROOT / "figures"
TABLES = RUN_ROOT / "tables"


def cluster_bootstrap_correlation(frame: pd.DataFrame) -> tuple[float, float, float]:
    x = frame["ZScore_Area"].to_numpy(float)
    y = frame["WAR"].to_numpy(float)
    ids = frame["cluster_id"].to_numpy(str)
    unique = np.unique(ids)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(N_BOOT):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(ids == player_id) for player_id in sampled])
        values.append(np.corrcoef(x[idx], y[idx])[0, 1])
    low, high = percentile_ci(values)
    return float(np.corrcoef(x, y)[0, 1]), low, high


def figure1() -> dict[str, float]:
    frame = pd.read_csv(RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    frame["cluster_id"] = frame["player_id"].where(
        frame["id_status"].eq("matched"), "unmatched_" + frame["source_row"].astype(str)
    )
    estimate, low, high = cluster_bootstrap_correlation(frame)
    x = frame["ZScore_Area"].to_numpy(float)
    y = frame["WAR"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    order = np.argsort(x)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=12, alpha=0.32, color="#3569a8", edgecolors="none")
    ax.plot(x[order], intercept + slope * x[order], color="#b23a48", linewidth=2)
    ax.text(
        0.03, 0.95, f"r = {estimate:.3f}; player-clustered 95% CI [{low:.3f}, {high:.3f}]",
        transform=ax.transAxes, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.75"},
    )
    ax.set_xlabel("Z-score pentagon area")
    ax.set_ylabel("WAR")
    ax.set_title("Descriptive alignment of pentagon area with WAR")
    ax.grid(alpha=0.2)
    fig.savefig(FIGURES / "figure1_alignment_revised.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure1_alignment_revised.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    result = {"pearson_r": estimate, "ci_low": low, "ci_high": high, "n_rows": len(frame), "n_boot": N_BOOT}
    (RESULTS / "figure1_alignment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def figure2() -> None:
    pairs = pd.read_csv(RESULTS / "matched_profile_distances.csv")
    example = pd.read_csv(RESULTS / "illustrative_matched_pair.csv").iloc[0]
    categories = ["same_player_adjacent", "scalar_matched", "random_same_season_pa_stratum"]
    labels = ["Same player,\nadjacent seasons", "Scalar-matched,\ndifferent players", "Random same season,\nPA-stratified"]
    colors = ["#4c78a8", "#f58518", "#9b59b6"]
    distributions = [pairs.loc[pairs["pair_type"].eq(c), "profile_distance"].to_numpy() for c in categories]
    fig = plt.figure(figsize=(11.2, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    violins = ax.violinplot(distributions, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    violins["cmedians"].set_color("black")
    violins["cmedians"].set_linewidth(1.6)
    ax.set_xticks([1, 2, 3], labels)
    ax.set_ylabel("Standardized five-dimensional distance")
    ax.set_title("(a) Profile-distance distributions")
    ax.grid(axis="y", alpha=0.2)

    radar = fig.add_subplot(grid[0, 1], polar=True)
    order = ["ZScore_Contact", "ZScore_Power", "ZScore_Speed", "ZScore_Defense", "ZScore_Discipline"]
    axis_labels = ["Contact", "Power", "Speed", "Defense", "Discipline"]
    angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False).tolist()
    angles += angles[:1]
    for prefix, color, name_key in [
        ("focal_", "#3569a8", "focal_name"),
        ("comparison_", "#d95f02", "comparison_name"),
    ]:
        values = [float(example[prefix + col]) for col in order]
        values += values[:1]
        label = f"{example[name_key]} ({int(example['focal_season'])})"
        radar.plot(angles, values, linewidth=2, color=color, label=label)
        radar.fill(angles, values, alpha=0.12, color=color)
    radar.set_xticks(angles[:-1], axis_labels)
    radar.tick_params(axis="x", labelsize=9, pad=4)
    radar.set_ylim(20, 80)
    radar.set_yticks([20, 40, 60, 80])
    radar.set_yticklabels(["20", "40", "60", "80"], fontsize=8)
    radar.set_title("(b) Illustrative scalar-matched pair", pad=18)
    radar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), fontsize=8, frameon=False)
    detail = (
        f"wRC+ {example['focal_wrc']:.0f}/{example['comparison_wrc']:.0f}; "
        f"WAR {example['focal_war']:.1f}/{example['comparison_war']:.1f}; "
        f"PA {example['focal_pa']:.0f}/{example['comparison_pa']:.0f}; d={example['profile_distance']:.3f}"
    )
    radar.text(0.5, -0.33, detail, transform=radar.transAxes, ha="center", va="top", fontsize=8)
    fig.savefig(FIGURES / "figure2_profile_diversity_revised.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure2_profile_diversity_revised.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def delta_coefficients() -> pd.DataFrame:
    transitions = pd.read_csv(RUN_ROOT / "data_intermediate" / "stable_id_transitions.csv")
    names = ["Contact", "Power", "Plate Discipline", "Defense", "Speed"]
    features = []
    for name, col in zip(names, TOOL_COLS):
        delta = f"delta_{name.replace(' ', '_')}"
        transitions[delta] = transitions[f"{col}_t1"] - transitions[f"{col}_t"]
        features.append(delta)
    transitions["delta_wRC+"] = transitions["wRC+_t1"] - transitions["wRC+_t"]
    model = LinearRegression().fit(transitions[features], transitions["delta_wRC+"])
    ids = transitions["player_id"].to_numpy(str)
    unique = np.unique(ids)
    rng = np.random.default_rng(SEED)
    samples = np.empty((N_BOOT, len(features)))
    for replicate in range(N_BOOT):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(ids == player_id) for player_id in sampled])
        fitted = LinearRegression().fit(transitions.iloc[idx][features], transitions.iloc[idx]["delta_wRC+"])
        samples[replicate] = fitted.coef_
    rows = []
    for i, dimension in enumerate(names):
        low, high = percentile_ci(samples[:, i])
        rows.append({
            "dimension": dimension, "coefficient": model.coef_[i], "ci_low": low, "ci_high": high,
            "n_transitions": len(transitions), "n_players": len(unique),
            "bootstrap_unit": "stable player ID", "n_boot": N_BOOT,
        })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "delta_association_coefficients.csv", index=False)
    return out


def figure3(coefficients: pd.DataFrame) -> None:
    plot = coefficients.iloc[::-1].reset_index(drop=True)
    x = plot["coefficient"].to_numpy()
    errors = np.vstack([x - plot["ci_low"].to_numpy(), plot["ci_high"].to_numpy() - x])
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    ax.errorbar(x, np.arange(len(plot)), xerr=errors, fmt="o", markersize=6, capsize=3, color="#3569a8", ecolor="#555555")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(np.arange(len(plot)), plot["dimension"])
    ax.set_xlabel(r"Coefficient for concurrent $\Delta$wRC+ (per one grade point)")
    ax.set_title(r"Concurrent Dimension-Change Associations with $\Delta$wRC+")
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(FIGURES / "figure3_delta_association_revised.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "figure3_delta_association_revised.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_cell(row: pd.Series, metric: str, digits: int = 3) -> str:
    return f"{row[metric]:.{digits}f} [{row[f'{metric}_ci_low']:.{digits}f}, {row[f'{metric}_ci_high']:.{digits}f}]"


def write_tables() -> None:
    prediction = pd.read_csv(RESULTS / "calibrated_prediction.csv")
    order = [
        "League mean", "Carry-forward wRC+", "Calibrated wRC+ Ridge", "wRC+ + age + PA Ridge",
        "B2V Ridge", "B2V + age + PA Ridge", "wRC+ + B2V + age + PA Ridge",
    ]
    rows = []
    for name in order:
        subset = prediction[prediction["model"].eq(name)].set_index("metric")
        cells = []
        for metric_name in ["mae", "r2", "spearman"]:
            row = subset.loc[metric_name]
            cells.append("---" if not np.isfinite(row["estimate"]) else f"{row['estimate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]")
        rows.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{tabular}{lccc}", r"\toprule", r"Model & MAE $\downarrow$ & $R^2$ $\uparrow$ & Spearman $\rho$ $\uparrow$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "",
    ])
    (TABLES / "prospective_prediction_revised.tex").write_text(tex)

    stability = pd.read_csv(RESULTS / "stability_by_pa.csv")
    subset_order = ["Overall", "100-249", "250-499", "At least 500"]
    dimension_order = ["Contact", "Power", "Plate Discipline", "Defense", "Speed"]
    rows = []
    for dimension in dimension_order:
        cells = []
        for subset_name in subset_order:
            row = stability[(stability["dimension"].eq(dimension)) & (stability["subset"].eq(subset_name))].iloc[0]
            cells.append(f"{int(row['n_pairs'])}; {row['pearson']:.2f} [{row['pearson_ci_low']:.2f}, {row['pearson_ci_high']:.2f}]")
        rows.append(f"{dimension} & " + " & ".join(cells) + r" \\")
    tex = "\n".join([
        r"\begin{tabular}{lcccc}", r"\toprule", r"Dimension & Overall & 100--249 PA & 250--499 PA & $\geq$500 PA \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", "",
    ])
    (TABLES / "stability_by_pa.tex").write_text(tex)


def reproduction_check(alignment: dict[str, float]) -> None:
    prediction = pd.read_csv(RESULTS / "calibrated_prediction.csv")
    matched = pd.read_csv(RESULTS / "matched_profile_distance_summary.csv").set_index("pair_type")
    def value(model: str, metric_name: str) -> float:
        return float(prediction[(prediction["model"].eq(model)) & (prediction["metric"].eq(metric_name))]["estimate"].iloc[0])
    checks = {
        "original_manuscript_sha256": sha256_file(REPO_ROOT / "paper" / "baseball2vector_en.tex"),
        "processed_data_sha256": sha256_file(REPO_ROOT / "data" / "processed" / "v3_results.csv"),
        "python_version": platform.python_version(), "random_seed": SEED, "bootstrap_replicates": N_BOOT,
        "reproduced": {
            "alignment_r": alignment["pearson_r"],
            "league_mean_mae": value("League mean", "mae"),
            "carry_forward_mae": value("Carry-forward wRC+", "mae"),
            "forecast_safe_b2v_mae": value("B2V Ridge", "mae"),
            "forecast_safe_b2v_r2": value("B2V Ridge", "r2"),
            "forecast_safe_b2v_spearman": value("B2V Ridge", "spearman"),
            "same_player_median": float(matched.loc["same_player_adjacent", "median"]),
            "scalar_matched_median": float(matched.loc["scalar_matched", "median"]),
            "corrected_random_median": float(matched.loc["random_same_season_pa_stratum", "median"]),
        },
        "discrepancies": [
            "Forecast-safe within-input-season scaling changes B2V MAE from the archived approximately 18.023 to 18.027.",
            "The random-pair median changes from approximately 1.197 to 1.188 because random pairs now use exactly the 2,080 successfully matched focal rows.",
            "The same-player median reproduces the prior approximately 0.714 value; the scalar-matched median changes from approximately 1.001 to 1.004 after floating-point ties are resolved by a rounded-gap stable-ID rule.",
        ],
    }
    (RESULTS / "reproduction_check.json").write_text(json.dumps(checks, indent=2) + "\n")


def main() -> None:
    alignment = figure1()
    figure2()
    coefficients = delta_coefficients()
    figure3(coefficients)
    write_tables()
    reproduction_check(alignment)
    print(json.dumps({"alignment": alignment, "figures": 3, "tables": 2}, indent=2))


if __name__ == "__main__":
    main()
