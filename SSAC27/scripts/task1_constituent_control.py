"""Task 1 -- constituent-statistics control for the Finding 2 prediction claim.

Question: does compressing 22 constituent statistics into 5 tool scores predict
next season better, worse, or the same as feeding the raw statistics in?

Six Ridge models share one time split (train: 2021->22, 2022->23, 2023->24;
test: 2024->25), one inner-CV protocol (player-grouped 5-fold on the training
window only) and one alpha grid.

    M0  scalar baseline (this target's own current-season value)
    M1  B2V 5 tool scores
    M2  all 22 constituent statistics
    M3  M0 + age + PA
    M4  M1 + age + PA
    M5  M2 + age + PA

M1's features are the five group means of M2's features, so M1 is a fixed linear
projection of M2's feature space and the contrast isolates the tool grouping.

Three targets are run. wRC+ is the published one, but it is a batting-only index
that prices neither Defense nor Speed -- two of the five tools. WAR prices both,
so it is the target on which the tool structure has something to gain that wRC+
cannot show. WAR is also reported per 600 PA, because WAR correlates with plate
appearances at r = 0.674 (against 0.449 for wRC+) and predicting the counting
version is partly predicting playing time rather than profile quality.

Run:  python SSAC27/scripts/task1_constituent_control.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as sc

GRIDS = {
    "wide_logspace_-3_4_50": sc.ALPHA_GRID_WIDE,
    "archived_6_value": sc.ALPHA_GRID_ARCHIVED,
}
PAIRS = [
    ("M1 B2V 5 tools", "M2 constituent stats"),
    ("M1 B2V 5 tools", "M0 scalar baseline"),
    ("M4 B2V + age + PA", "M5 constituents + age + PA"),
    ("M2 constituent stats", "M0 scalar baseline"),
    ("M4 B2V + age + PA", "M3 scalar + age + PA"),
    ("M5 constituents + age + PA", "M3 scalar + age + PA"),
]


def ubr_robustness(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: sc.Target,
    samples: list[np.ndarray],
) -> pd.DataFrame:
    """Re-run the constituent models without UBR.

    UBR is 100% missing for 2025 and 21% missing for 2024 in the recovered
    FanGraphs export (an export artifact, not a real data gap). The tool
    pipeline fills those with the season cohort mean. BsR already contains
    almost the same baserunning signal, so dropping UBR should barely move
    anything; this check confirms that rather than assuming it.
    """
    dropped = "cs_UBR_t"
    stats = [c for c in sc.constituent_features() if c != dropped]
    specs = {
        "M2b constituents (no UBR)": stats,
        "M5b constituents + age + PA (no UBR)": [*stats, *sc.CONTEXT_FEATURES],
    }
    results, _ = sc.run_models(
        train, test, specs, sc.ALPHA_GRID_WIDE, fold_label="C", target=target.column
    )
    rows = []
    for result in results:
        y = result.predictions["y_true"].to_numpy(float)
        pred = result.predictions["prediction"].to_numpy(float)
        low, high = sc.percentile_ci(np.array([sc.mae(y[i], pred[i]) for i in samples]))
        rows.append(
            {
                "target": target.key,
                "model": result.model,
                "n_features": result.n_features,
                "selected_alpha": result.selected_alpha,
                "mae": sc.mae(y, pred),
                "mae_ci_low": low,
                "mae_ci_high": high,
                "rmse": sc.rmse(y, pred),
                "dropped_feature": dropped,
            }
        )
    return pd.DataFrame(rows)


def run_target(
    transitions: pd.DataFrame, target: sc.Target
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = sc.prepare_for_target(transitions, target)
    train, test = sc.split_fold(frame, "C")
    if target.key == sc.PRIMARY_TARGET:
        assert len(train) == sc.EXPECTED_N_TRAIN_FOLD_C, len(train)
        assert len(test) == sc.EXPECTED_N_TEST_FOLD_C, len(test)
    print(f"\n  [{target.label}] Fold C  train n={len(train)}  test n={len(test)}")

    specs = sc.model_specs("zscore", target)
    samples = sc.cluster_bootstrap_indices(test["player_id"].to_numpy())

    summaries, paired_frames, prediction_frames, tuning_frames = [], [], [], []
    for grid_name, grid in GRIDS.items():
        results, tuning = sc.run_models(
            train, test, specs, grid, fold_label="C", target=target.column
        )
        tuning.insert(0, "target", target.key)
        tuning.insert(1, "alpha_grid", grid_name)
        tuning_frames.append(tuning)

        predictions = sc.attach_fit_metadata(
            pd.concat([r.predictions for r in results], ignore_index=True), results
        )
        predictions["target"] = target.key
        predictions["alpha_grid"] = grid_name
        prediction_frames.append(predictions)

        summary, paired = sc.evaluate_predictions(
            predictions, "C", PAIRS,
            extra={"target": target.key, "alpha_grid": grid_name},
        )
        summaries.append(summary)
        paired_frames.append(paired)

    robustness = ubr_robustness(train, test, target, samples)
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
        robustness,
    )


def main() -> None:
    (sc.RESULTS_DIR / "task1").mkdir(parents=True, exist_ok=True)
    players = sc.load_player_seasons()
    transitions = sc.build_transitions(players)

    blocks = [run_target(transitions, target) for target in sc.TARGETS.values()]
    summary = pd.concat([b[0] for b in blocks], ignore_index=True)
    paired = pd.concat([b[1] for b in blocks], ignore_index=True)

    # Reproduction anchor: under the published alpha grid and the published
    # target, M0 and M1 must land on the published Finding 2 test MAEs.
    anchor = summary[
        summary["alpha_grid"].eq("archived_6_value")
        & summary["target"].eq(sc.PRIMARY_TARGET)
    ].set_index("model")
    for model, expected in [
        ("M0 scalar baseline", sc.EXPECTED_M0_MAE),
        ("M1 B2V 5 tools", sc.EXPECTED_M1_MAE),
    ]:
        observed = float(anchor.loc[model, "mae"])
        assert abs(observed - expected) < 0.01, (model, observed, expected)
        print(f"  reproduction check {model:<20s} MAE {observed:.4f} (published {expected})")

    summary.to_csv(sc.RESULTS_DIR / "task1" / "model_comparison.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task1" / "paired_differences.csv", index=False)
    pd.concat([b[2] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task1" / "predictions.csv", index=False
    )
    pd.concat([b[3] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task1" / "alpha_tuning.csv", index=False
    )
    pd.concat([b[4] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task1" / "ubr_robustness.csv", index=False
    )
    (sc.RESULTS_DIR / "task1" / "environment.json").write_text(
        json.dumps(sc.environment_stamp(), indent=2) + "\n", encoding="utf-8"
    )

    for target in sc.TARGETS.values():
        for grid_name in GRIDS:
            block = summary[
                summary["target"].eq(target.key) & summary["alpha_grid"].eq(grid_name)
            ]
            print(f"\n=== {target.label} | alpha grid: {grid_name} ===")
            print(
                block[["model", "n_features", "selected_alpha", "mae", "mae_ci_low",
                       "mae_ci_high", "rmse"]].to_string(index=False)
            )
            print()
            print(
                paired[
                    paired["target"].eq(target.key)
                    & paired["alpha_grid"].eq(grid_name)
                ][["candidate", "baseline", "improvement_mae", "ci_low", "ci_high",
                   "excludes_zero"]].to_string(index=False)
            )


if __name__ == "__main__":
    main()
