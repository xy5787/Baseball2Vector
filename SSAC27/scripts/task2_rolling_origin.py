"""Task 2 -- rolling-origin evaluation across three temporal holdouts.

Question: is the Finding 2 improvement a property of the 2025 season, or does it
repeat when the origin moves?

Three folds. Training uses only transitions whose *input* season is strictly
earlier than the test fold's input season, so no fold ever trains on an outcome
it later has to predict.

    Fold A   train 2021->22                                test 2022->23
    Fold B   train 2021->22, 2022->23                      test 2023->24
    Fold C   train 2021->22, 2022->23, 2023->24            test 2024->25   (published)

All six Task 1 models are run in every fold, and alpha is re-selected
independently inside each fold's own training window -- no information crosses
folds. The headline output is the *sign consistency* of M1-M0 and M1-M2 across
the three folds, not any single fold's MAE.

Run:  python SSAC27/scripts/task2_rolling_origin.py
"""

from __future__ import annotations

import json

import pandas as pd

import common as sc

GRIDS = {
    "wide_logspace_-3_4_50": sc.ALPHA_GRID_WIDE,
    "archived_6_value": sc.ALPHA_GRID_ARCHIVED,
}
PRIMARY_GRID = "wide_logspace_-3_4_50"

#: Contrasts whose sign is tracked across folds. (candidate, baseline);
#: positive means the candidate has the lower MAE.
PAIRS = [
    ("M1 B2V 5 tools", "M0 scalar baseline"),
    ("M1 B2V 5 tools", "M2 constituent stats"),
    ("M2 constituent stats", "M0 scalar baseline"),
    ("M4 B2V + age + PA", "M3 scalar + age + PA"),
    ("M4 B2V + age + PA", "M5 constituents + age + PA"),
]


def add_relative_column(summary: pd.DataFrame) -> pd.DataFrame:
    """MAE as a percentage of the scalar baseline's, so targets are comparable.

    wRC+ MAE is on a ~19 scale and WAR MAE on a ~1.1 scale, so raw differences
    between targets mean nothing. Everything is therefore also read as skill
    relative to that target's own M0.
    """
    out = summary.copy()
    keys = ["target", "alpha_grid", "fold"] if "alpha_grid" in out else ["target", "fold"]
    baseline = (
        out[out["model"].eq("M0 scalar baseline")]
        .set_index(keys)["mae"]
        .rename("baseline_mae")
    )
    out = out.join(baseline, on=keys)
    out["mae_pct_of_scalar_baseline"] = 100 * out["mae"] / out["baseline_mae"]
    return out.drop(columns=["baseline_mae"])


def run_target(
    transitions: pd.DataFrame, target: sc.Target
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = sc.prepare_for_target(transitions, target)
    specs = sc.model_specs("zscore", target)

    fold_frames, paired_frames, prediction_frames, tuning_frames = [], [], [], []
    for grid_name, grid in GRIDS.items():
        per_fold_predictions: list[pd.DataFrame] = []
        for fold in sc.ROLLING_FOLDS:
            train, test = sc.split_fold(frame, fold)
            print(
                f"  [{target.key} | {grid_name}] fold {fold}: "
                f"train n={len(train)} ({train['Season_t'].min()}-{train['Season_t'].max()} in) "
                f"test n={len(test)} ({test['Season_t'].iloc[0]}->{test['Season_t1'].iloc[0]})"
                + ("  [THIN TRAINING SET]" if len(train) < sc.THIN_TRAIN_THRESHOLD else "")
            )
            results, tuning = sc.run_models(
                train, test, specs, grid, fold_label=fold, target=target.column
            )
            tuning.insert(0, "target", target.key)
            tuning.insert(1, "alpha_grid", grid_name)
            tuning_frames.append(tuning)

            predictions = sc.attach_fit_metadata(
                pd.concat([r.predictions for r in results], ignore_index=True), results
            )
            predictions["target"] = target.key
            predictions["alpha_grid"] = grid_name
            per_fold_predictions.append(predictions)

            fold_summary, fold_paired = sc.evaluate_predictions(
                predictions, fold, PAIRS,
                extra={"target": target.key, "alpha_grid": grid_name},
            )
            fold_frames.append(fold_summary)
            paired_frames.append(fold_paired)

        pooled_predictions = pd.concat(per_fold_predictions, ignore_index=True)
        prediction_frames.append(pooled_predictions)
        pooled_summary, pooled_paired = sc.evaluate_predictions(
            pooled_predictions, "pooled", PAIRS, pooled=True,
            extra={"target": target.key, "alpha_grid": grid_name},
        )
        fold_frames.append(pooled_summary)
        paired_frames.append(pooled_paired)

    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
    )


def main() -> None:
    (sc.RESULTS_DIR / "task2").mkdir(parents=True, exist_ok=True)
    players = sc.load_player_seasons()
    transitions = sc.build_transitions(players)

    blocks = [run_target(transitions, target) for target in sc.TARGETS.values()]
    fold_performance = add_relative_column(
        pd.concat([b[0] for b in blocks], ignore_index=True)
    )
    paired = pd.concat([b[1] for b in blocks], ignore_index=True)
    # Sign consistency is read off the primary grid; the archived grid produced
    # an identical sign pattern everywhere it was checked.
    signs = sc.sign_consistency(
        paired[paired["alpha_grid"].eq(PRIMARY_GRID)], PAIRS, by="target"
    )

    # The published Fold C result must survive being recomputed inside Task 2.
    anchor = fold_performance[
        fold_performance["alpha_grid"].eq("archived_6_value")
        & fold_performance["fold"].eq("C")
        & fold_performance["target"].eq(sc.PRIMARY_TARGET)
    ].set_index("model")
    for model, expected in [
        ("M0 scalar baseline", sc.EXPECTED_M0_MAE),
        ("M1 B2V 5 tools", sc.EXPECTED_M1_MAE),
    ]:
        observed = float(anchor.loc[model, "mae"])
        assert abs(observed - expected) < 0.01, (model, observed, expected)

    fold_performance.to_csv(sc.RESULTS_DIR / "task2" / "fold_performance.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task2" / "paired_differences.csv", index=False)
    signs.to_csv(sc.RESULTS_DIR / "task2" / "sign_consistency.csv", index=False)
    pd.concat([b[2] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task2" / "predictions.csv", index=False
    )
    pd.concat([b[3] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task2" / "alpha_tuning.csv", index=False
    )
    (sc.RESULTS_DIR / "task2" / "environment.json").write_text(
        json.dumps(sc.environment_stamp(), indent=2) + "\n", encoding="utf-8"
    )

    primary = fold_performance[fold_performance["alpha_grid"].eq(PRIMARY_GRID)]
    for target in sc.TARGETS.values():
        block = primary[primary["target"].eq(target.key)]
        print(f"\n=== {target.label} -- MAE by fold x model ({PRIMARY_GRID}) ===")
        print(block.pivot(index="model", columns="fold", values="mae").round(4).to_string())
        print("  as % of the scalar baseline's MAE:")
        print(
            block.pivot(
                index="model", columns="fold", values="mae_pct_of_scalar_baseline"
            ).round(1).to_string()
        )
        print(f"\n  paired differences ({target.label}):")
        print(
            paired[
                paired["target"].eq(target.key) & paired["alpha_grid"].eq(PRIMARY_GRID)
            ][["fold", "candidate", "baseline", "improvement_mae", "ci_low", "ci_high",
               "excludes_zero"]].to_string(index=False)
        )

    print("\n=== Sign consistency across folds A, B, C ===")
    print(
        signs[signs["target"].isin(sc.TARGETS)][
            ["target", "candidate", "baseline", "delta_mae_fold_A", "delta_mae_fold_B",
             "delta_mae_fold_C", "all_three_same_sign", "n_folds_ci_excludes_zero"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
