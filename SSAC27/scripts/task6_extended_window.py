"""Task 6 -- the study widened to 2019-2025, with five rolling origins.

Tasks 1-3 are anchored to the archived 2021-2025 tool scores so that the
published Finding 2 numbers keep reproducing exactly. That anchoring costs
coverage: three rolling origins, the thinnest trained on 351 transitions.

This task drops the anchor and rebuilds the whole cohort from the raw FanGraphs
exports for 2019-2025, which the 2019-2020 export makes possible. It is a
**separate extension**, not a replacement: Tasks 1-3 remain the primary results
and this one says whether their conclusions survive on 40% more data and nearly
twice as many origins.

    transitions  1,414 -> 1,981
    origins          3 -> 5

Fold construction is unchanged in kind -- expanding window, training only on
transitions whose input season is strictly earlier than the test input season,
alpha re-selected inside each fold.

Two things to keep in mind when reading the result, both flagged in the tables:

* **Tool scores are recomputed, not archived.** They agree with the archived ones
  to Pearson r >= 0.9999 on the overlapping seasons; the agreement is asserted on
  every run. But this is a rebuild, so it is not expected to land on 18.0267 to
  four decimals.
* **2020 is a 60-game season.** Its qualified cohort is 310 players with a
  maximum of 267 PA, against ~460 players and ~730 PA in a full season. The
  PA >= 100 qualifier therefore selects a different kind of player in 2020. Every
  fold that touches it is marked, and a 2020-free variant is reported.

Run:  python SSAC27/scripts/task6_extended_window.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as sc
from baseball2vec import marcel
import data as sd

GRID = sc.ALPHA_GRID_WIDE
GRID_NAME = "wide_logspace_-3_4_50"
MARCEL_FEATURE = "marcel_pred_t"
MARCEL_MODEL = "M6 Marcel (no fit)"

COVID_SEASON = 2020

PAIRS = [
    ("M1 B2V 5 tools", "M0 scalar baseline"),
    ("M1 B2V 5 tools", "M2 constituent stats"),
    ("M2 constituent stats", "M0 scalar baseline"),
    ("M4 B2V + age + PA", "M3 scalar + age + PA"),
    ("M7 Marcel + B2V", "M6c Marcel calibrated"),
    ("M6c Marcel calibrated", "M1 B2V 5 tools"),
]

#: Target -> the Marcel metric that projects it.
TARGET_METRIC = {"wrc_plus": "wRC+", "war": "WAR", "war_rate": "WAR_per_600"}


def extended_specs(target: sc.Target) -> dict[str, list[str]]:
    tools = [f"safe_ZScore_{tool}_t" for tool in sc.TOOL_NAMES]
    stats = sc.constituent_features()
    return {
        "M0 scalar baseline": [target.baseline],
        "M1 B2V 5 tools": tools,
        "M2 constituent stats": stats,
        "M3 scalar + age + PA": [target.baseline, *sc.CONTEXT_FEATURES],
        "M4 B2V + age + PA": [*tools, *sc.CONTEXT_FEATURES],
        "M5 constituents + age + PA": [*stats, *sc.CONTEXT_FEATURES],
        "M6c Marcel calibrated": [MARCEL_FEATURE],
        "M7 Marcel + B2V": [MARCEL_FEATURE, *tools],
    }


def assert_matches_archived(cohort: pd.DataFrame) -> pd.DataFrame:
    """The recomputed tool scores must track the archived ones where they overlap."""
    archived = sc.load_player_seasons(verbose=False)
    merged = archived[["player_id", "Season"] + [
        f"safe_ZScore_{tool}" for tool in sc.TOOL_NAMES
    ]].merge(
        cohort[["player_id", "Season"] + [
            f"safe_ZScore_{tool}" for tool in sc.TOOL_NAMES
        ]],
        on=["player_id", "Season"],
        how="inner",
        suffixes=("_archived", "_recomputed"),
    )
    rows: list[dict] = []
    for tool in sc.TOOL_NAMES:
        left = merged[f"safe_ZScore_{tool}_archived"].to_numpy(float)
        right = merged[f"safe_ZScore_{tool}_recomputed"].to_numpy(float)
        correlation = float(np.corrcoef(left, right)[0, 1])
        assert correlation > 0.999, f"{tool} archived vs recomputed r={correlation:.5f}"
        rows.append(
            {
                "tool": tool,
                "n_overlapping_rows": len(merged),
                "pearson_r": correlation,
                "mean_abs_difference": float(np.abs(left - right).mean()),
            }
        )
    table = pd.DataFrame(rows)
    print("\n  Recomputed vs archived tool scores on the 2021-2025 overlap:")
    print(table.to_string(index=False))
    return table


def run_window(
    transitions: pd.DataFrame,
    folds: dict[str, tuple[list[int], int]],
    label: str,
    target: sc.Target,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = extended_specs(target)
    fold_frames, paired_frames, tuning_frames = [], [], []
    per_fold: list[pd.DataFrame] = []

    for fold, (train_inputs, test_input) in folds.items():
        train, test = sc.split_by_input_season(transitions, train_inputs, test_input)
        touches_covid = COVID_SEASON in train_inputs or test_input == COVID_SEASON or (
            test_input + 1 == COVID_SEASON
        )
        print(
            f"  [{target.key} | {label}] fold {fold}: train n={len(train)} test n={len(test)}"
            + ("  [includes the 60-game 2020 season]" if touches_covid else "")
        )
        results, tuning = sc.run_models(
            train, test, specs, GRID, fold_label=fold, target=target.column
        )
        tuning.insert(0, "target", target.key)
        tuning.insert(1, "window", label)
        tuning_frames.append(tuning)

        predictions = sc.attach_fit_metadata(
            pd.concat([r.predictions for r in results], ignore_index=True), results
        )
        raw = test[["player_id", "Name", "Season_t", "Season_t1"]].copy()
        raw["y_true"] = test[target.column].to_numpy(float)
        raw["prediction"] = test[MARCEL_FEATURE].to_numpy(float)
        raw["fold"] = fold
        raw["model"] = MARCEL_MODEL
        raw["n_features"] = 1
        predictions = pd.concat([predictions, raw], ignore_index=True)
        predictions["window"] = label
        predictions["target"] = target.key
        per_fold.append(predictions)

        summary, paired = sc.evaluate_predictions(
            predictions, fold, PAIRS,
            extra={"target": target.key, "window": label, "touches_2020": touches_covid},
        )
        fold_frames.append(summary)
        paired_frames.append(paired)

    pooled = pd.concat(per_fold, ignore_index=True)
    pooled_summary, pooled_paired = sc.evaluate_predictions(
        pooled, "pooled", PAIRS, pooled=True,
        extra={"target": target.key, "window": label,
               "touches_2020": COVID_SEASON in
               {s for inputs, _ in folds.values() for s in inputs}},
    )
    fold_frames.append(pooled_summary)
    paired_frames.append(pooled_paired)
    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
        pooled,
        pd.concat(tuning_frames, ignore_index=True),
    )


def main() -> None:
    (sc.RESULTS_DIR / "task6").mkdir(parents=True, exist_ok=True)
    cohort = sd.load_extended_cohort()
    agreement = assert_matches_archived(cohort)
    agreement.to_csv(sc.RESULTS_DIR / "task6" / "encoding_agreement.csv", index=False)

    pool = sd.load_history_pool()
    base = sd.build_extended_transitions(cohort)
    base = sc.add_rate_columns(base.rename(columns={"WAR_t": "WAR", "PA_t": "PA"})).rename(
        columns={"WAR": "WAR_t", "PA": "PA_t", "WAR_per_600": "WAR_per_600_t"}
    )
    base = sc.add_rate_columns(
        base.rename(columns={"WAR_t1": "WAR", "PA_t1": "PA"})
    ).rename(columns={"WAR": "WAR_t1", "PA": "PA_t1", "WAR_per_600": "WAR_per_600_t1"})

    flow = (
        base.groupby(["Season_t", "Season_t1"]).size().rename("n_transitions").reset_index()
    )
    flow["n_cohort_input_season"] = flow["Season_t"].map(cohort.groupby("Season").size())
    flow["max_PA_input_season"] = flow["Season_t"].map(cohort.groupby("Season")["PA"].max())
    flow.to_csv(sc.RESULTS_DIR / "task6" / "sample_flow.csv", index=False)
    print("\n  Extended transitions:")
    print(flow.to_string(index=False))

    fold_frames, paired_frames, prediction_frames, tuning_frames = [], [], [], []
    for target in sc.TARGETS.values():
        marcel_metric = TARGET_METRIC[target.key]
        projections = marcel.project(
            pool, league_baseline="definition", recenter=False, metric=marcel_metric
        )
        transitions = base.merge(
            projections.rename(columns={"marcel_pred": MARCEL_FEATURE}),
            left_on=["player_id", "Season_t"],
            right_on=["player_id", "origin_season"],
            how="left",
            validate="one_to_one",
        )
        assert transitions[MARCEL_FEATURE].notna().all()
        for features in extended_specs(target).values():
            sc.assert_no_future_columns(features)

        # Counting targets cannot use 2020 as an outcome season, so for them the
        # extended window collapses to the 2020-free one. Reported as such rather
        # than duplicating a table that carries no extra origins.
        windows = [] if target.excludes_2020 else [
            (transitions, "2019-2025 (all)")
        ]
        windows.append((sc.drop_2020(transitions), "2021-2025 (2020 excluded)"))

        for frame, label in windows:
            folds = sd.extended_rolling_folds(frame)
            block = run_window(frame, folds, label, target)
            fold_frames.append(block[0])
            paired_frames.append(block[1])
            prediction_frames.append(block[2])
            tuning_frames.append(block[3])

    performance = pd.concat(fold_frames, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    primary = paired[
        paired["target"].eq(sc.PRIMARY_TARGET) & paired["window"].eq("2019-2025 (all)")
    ]
    signs = sc.sign_consistency(
        primary, PAIRS, by="window",
        folds=[f for f in primary["fold"].unique() if f != "pooled"],
    )

    performance.to_csv(sc.RESULTS_DIR / "task6" / "fold_performance.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task6" / "paired_differences.csv", index=False)
    signs.to_csv(sc.RESULTS_DIR / "task6" / "sign_consistency.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task6" / "predictions.csv", index=False
    )
    pd.concat(tuning_frames, ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task6" / "alpha_tuning.csv", index=False
    )
    (sc.RESULTS_DIR / "task6" / "environment.json").write_text(
        json.dumps(
            {
                **sc.environment_stamp(),
                "export_2019_2020_sha256": sc.sha256_file(sd.EXPORT_2019_2020),
                "export_2021_2025_sha256": sc.sha256_file(sd.EXPORT_2021_2025),
                "cohort_rows": int(len(cohort)),
                "transitions": int(len(base)),
                "encodings": list(sd.extended_encodings()),
                "counting_targets_exclude_2020": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for target in sc.TARGETS.values():
        for window in performance[performance["target"].eq(target.key)]["window"].unique():
            block = performance[
                performance["target"].eq(target.key) & performance["window"].eq(window)
            ]
            print(f"\n=== {target.label} -- {window} ===")
            print(
                block.pivot(index="model", columns="fold", values="mae")
                .round(4).to_string()
            )
            print("  pooled paired differences:")
            print(
                paired[
                    paired["target"].eq(target.key)
                    & paired["window"].eq(window)
                    & paired["fold"].eq("pooled")
                ][["candidate", "baseline", "improvement_mae", "ci_low", "ci_high",
                   "excludes_zero"]].to_string(index=False)
            )

    print("\n=== Sign consistency across the five wRC+ origins ===")
    print(
        signs[
            ["candidate", "baseline", "n_folds", "n_folds_candidate_better",
             "all_folds_same_sign", "n_folds_ci_excludes_zero"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
