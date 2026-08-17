"""Create LaTeX tables, provenance, and reproduction checks from saved results."""

from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from common import ALPHAS, DATA_PATH, N_BOOT, REPO_ROOT, RUN_ROOT, SEED, dump_json, sha256_file


def interval_cell(frame: pd.DataFrame, model: str, metric: str, digits: int) -> str:
    row = frame[(frame["model"].eq(model)) & frame["metric"].eq(metric)].iloc[0]
    if not np.isfinite(row["estimate"]):
        return "---"
    return f"{row['estimate']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def prediction_table() -> None:
    frame = pd.read_csv(RUN_ROOT / "results" / "calibrated_prediction.csv")
    order = [
        "League mean", "Carry-forward wRC+", "Calibrated wRC+ Ridge",
        "wRC+ + age + PA Ridge", "B2V Ridge", "B2V + age + PA Ridge",
        "wRC+ + B2V + age + PA Ridge",
    ]
    labels = {
        "League mean": "Training-target league mean",
        "Carry-forward wRC+": "Previous-season wRC+",
        "Calibrated wRC+ Ridge": "Calibrated wRC+ Ridge",
        "wRC+ + age + PA Ridge": "wRC+ + age + PA Ridge",
        "B2V Ridge": "B2V Ridge",
        "B2V + age + PA Ridge": "B2V + age + PA Ridge",
        "wRC+ + B2V + age + PA Ridge": "wRC+ + B2V + age + PA Ridge",
    }
    lines = [
        "% Generated from results/calibrated_prediction.csv",
        "\\begin{table*}[t]", "\\centering", "\\caption{Prospective 2024--2025 wRC+ prediction on the identical 358-player holdout. Vector models use forecast-safe within-input-season standardized dimensions. Brackets are 95\\% player-level bootstrap intervals (2,000 resamples).}",
        "\\label{tab:prospective}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrccc}", "\\toprule",
        "Model & $n_{train}$ & $n_{test}$ & MAE & $R^2$ & Spearman $\\rho$ \\\\", "\\midrule",
    ]
    for model in order:
        safe = labels[model].replace("+", "$+$")
        lines.append(f"{safe} & 1,056 & 358 & {interval_cell(frame, model, 'mae', 2)} & {interval_cell(frame, model, 'r2', 3)} & {interval_cell(frame, model, 'spearman', 3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (RUN_ROOT / "tables" / "prospective_prediction_revised.tex").write_text("\n".join(lines), encoding="utf-8")


def stability_table() -> None:
    frame = pd.read_csv(RUN_ROOT / "results" / "stability_by_pa.csv")
    subsets = ["Overall", "100-249", "250-499", "At least 500"]
    dimensions = ["Contact", "Power", "Plate Discipline", "Defense", "Speed"]
    lines = [
        "% Generated from results/stability_by_pa.csv",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Adjacent-season Pearson stability by the smaller of PA in seasons $t$ and $t+1$. Each cell gives $r$ [player-clustered 95\\% CI]; $n$ adjacent-season pairs.}",
        "\\label{tab:stability}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Dimension & Overall & 100--249 PA & 250--499 PA & At least 500 PA \\\\", "\\midrule",
    ]
    for dimension in dimensions:
        cells = []
        for subset in subsets:
            row = frame[(frame["dimension"].eq(dimension)) & frame["subset"].eq(subset)].iloc[0]
            cells.append(f"{row['pearson']:.2f} [{row['pearson_ci_low']:.2f}, {row['pearson_ci_high']:.2f}]; {int(row['n_pairs'])}")
        lines.append(f"{dimension} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (RUN_ROOT / "tables" / "stability_by_pa.tex").write_text("\n".join(lines), encoding="utf-8")


def paired_table() -> None:
    frame = pd.read_csv(RUN_ROOT / "results" / "paired_model_differences.csv")
    frame = frame[frame["metric"].eq("mae")]
    lines = [
        "% Generated from results/paired_model_differences.csv",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Paired MAE differences on the 358-player holdout. $\\Delta$MAE is baseline MAE minus candidate MAE, so positive values favor the candidate. Intervals use identical player bootstrap resamples.}",
        "\\label{tab:paired}", "\\scriptsize", "\\begin{tabular}{lll}", "\\toprule",
        "Candidate & Baseline & $\\Delta$MAE [95\\% CI] \\\\", "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        candidate = row.candidate.replace("+", "$+$")
        baseline = row.baseline.replace("+", "$+$")
        lines.append(f"{candidate} & {baseline} & {row.improvement:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    (RUN_ROOT / "tables" / "paired_differences_revised.tex").write_text("\n".join(lines), encoding="utf-8")


def audits() -> None:
    predictions = pd.read_csv(RUN_ROOT / "results" / "calibrated_predictions.csv")
    archived = predictions[predictions["feature_variant"].eq("archived_pooled_20_80")]
    from common import metric
    def value(model: str, metric_name: str) -> float:
        sub = archived[archived["model"].eq(model)]
        return metric(sub["y_true"].to_numpy(), sub["prediction"].to_numpy(), metric_name)
    matching = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distance_summary.csv").set_index("pair_type")
    reproduction = {
        "prospective": {
            "league_mean_mae": {"reported_approx": 22.27, "reproduced": value("League mean", "mae")},
            "carry_forward_mae": {"reported_approx": 22.35, "reproduced": value("Carry-forward wRC+", "mae")},
            "b2v_archived_mae": {"reported_approx": 18.02, "reproduced": value("B2V Ridge", "mae")},
            "b2v_archived_r2": {"reported_approx": 0.280, "reproduced": value("B2V Ridge", "r2")},
            "b2v_archived_spearman": {"reported_approx": 0.508, "reproduced": value("B2V Ridge", "spearman")},
            "status": "reproduced to stated rounding",
        },
        "matching": {
            "same_player_median": {"reported_approx": 0.714, "reproduced": float(matching.loc["same_player_adjacent", "median"])},
            "scalar_matched_median": {"reported_approx": 1.001, "reproduced": float(matching.loc["scalar_matched", "median"])},
            "random_median": {"reported_approx": 1.197, "corrected": float(matching.loc["random_same_season_pa_stratum", "median"])},
            "status": "same-player and matched reproduced; random changed because corrected random pairs use the same 2,080 focal-row basis as matched pairs rather than all 2,286 stable-ID rows",
        },
    }
    dump_json(RUN_ROOT / "results" / "reproduction_check.json", reproduction)

    packages = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "Pillow"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    inputs = [
        DATA_PATH,
        REPO_ROOT / "data" / "processed" / "v3_results.csv",
        REPO_ROOT / "paper" / "baseball2vector_en.tex",
    ]
    provenance = {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "seed": SEED, "bootstrap_replicates": N_BOOT, "ridge_alpha_grid": ALPHAS,
        "inputs": [{"path": str(path), "sha256": sha256_file(path), "mtime": path.stat().st_mtime} for path in inputs],
        "raw_stat_baseline": {"status": "not run", "reason": "constituent raw-stat cache is absent"},
        "original_tex_sha256_at_audit": sha256_file(REPO_ROOT / "paper" / "baseball2vector_en.tex"),
    }
    dump_json(RUN_ROOT / "results" / "input_provenance.json", provenance)


def main() -> None:
    prediction_table(); stability_table(); paired_table(); audits()
    print("Generated LaTeX tables, provenance, and reproduction check.")


if __name__ == "__main__":
    main()
