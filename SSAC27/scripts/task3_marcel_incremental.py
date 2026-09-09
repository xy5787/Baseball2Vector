"""Task 3 -- Marcel baseline and the incremental test.

Question: on top of what a standard projection system already squeezes out of a
player's recent record, does the five-tool profile carry residual signal?

The claim being tested is **not** "B2V beats Marcel". It is "adding B2V on top of
Marcel improves on Marcel alone". Marcel alone beating B2V alone would be an
entirely normal outcome and is reported as such if it happens.

Models, all on the Task 2 rolling-origin folds so the answer is not a single
season's accident:

    M0   current wRC+ Ridge                         (context, from Task 1)
    M1   B2V 5 tool scores Ridge                    (context, from Task 1)
    M6   Marcel projection, used directly           (no fitting at all)
    M6c  Ridge on [marcel]                          (Marcel + affine recalibration)
    M7   Ridge on [marcel, 5 tool scores]           (the incremental model)
    M7x  Ridge on [marcel, 5 tool scores, age, PA]

**M6c exists because M7 - M6 conflates two things.** M7 gets both the B2V
features *and* a fitted intercept/slope on Marcel, so part of any M7 - M6 gain is
simply recalibration that has nothing to do with B2V. M7 - M6c is the honest
incremental test; M7 - M6 is reported too because the brief asks for it.

Run:  python SSAC27/scripts/task3_marcel_incremental.py
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

PAIRS = [
    # The incremental test, in both its lenient and its honest form.
    ("M7 Marcel + B2V", MARCEL_MODEL),
    ("M7 Marcel + B2V", "M6c Marcel calibrated"),
    ("M7x Marcel + B2V + age + PA", "M6c Marcel calibrated"),
    # Where Marcel itself stands.
    ("M6c Marcel calibrated", "M0 scalar baseline"),
    ("M6c Marcel calibrated", "M1 B2V 5 tools"),
    (MARCEL_MODEL, "M1 B2V 5 tools"),
    ("M7 Marcel + B2V", "M1 B2V 5 tools"),
]

#: Target -> the Marcel metric that projects it.
TARGET_METRIC = {"wrc_plus": "wRC+", "war": "WAR", "war_rate": "WAR_per_600"}


def model_specs(target: sc.Target) -> dict[str, list[str]]:
    tools = sc.tool_features("zscore")
    return {
        "M0 scalar baseline": [target.baseline],
        "M1 B2V 5 tools": tools,
        "M6c Marcel calibrated": [MARCEL_FEATURE],
        "M7 Marcel + B2V": [MARCEL_FEATURE, *tools],
        "M7x Marcel + B2V + age + PA": [MARCEL_FEATURE, *tools, *sc.CONTEXT_FEATURES],
    }


def unfitted_marcel_predictions(
    test: pd.DataFrame, fold: str, target: sc.Target
) -> pd.DataFrame:
    """M6: the Marcel projection used as-is. Nothing is estimated from any data."""
    out = test[["player_id", "Name", "Season_t", "Season_t1"]].copy()
    out["y_true"] = test[target.column].to_numpy(float)
    out["prediction"] = test[MARCEL_FEATURE].to_numpy(float)
    out["fold"] = fold
    out["model"] = MARCEL_MODEL
    out["n_features"] = 1
    out["selected_alpha"] = np.nan
    out["n_train"] = np.nan
    out["train_period"] = "none (deterministic projection)"
    out["test_period"] = f"{test['Season_t'].iloc[0]}->{test['Season_t1'].iloc[0]}"
    return out


def qualified_only_history(players: pd.DataFrame) -> pd.DataFrame:
    """The lookback pool this package used before the 2019-2025 augmentation.

    Kept only so the variant table can quantify what widening the pool bought.
    """
    pool = players[["player_id", "Season", "PA", "wRC+", "age_t"]].copy()
    return pool.rename(columns={"age_t": "Age"})


def marcel_variant_table(
    transitions: pd.DataFrame, players: pd.DataFrame, pool: pd.DataFrame
) -> pd.DataFrame:
    """Marcel accuracy across history-pool x league-baseline choices.

    Reported both raw and after the training-only affine recalibration (M6c),
    because the two answer different questions. Raw Marcel is sensitive to the
    baseline constant: our test rows are players who qualified (PA >= 100) in the
    *outcome* season, a survivor-selected group averaging about 102 wRC+, so a
    projector correctly regressing toward the league's 100 will sit low on them.
    That is population selection, not a defect in Marcel. M6c absorbs the offset
    using training folds only, which is the leakage-free way to correct it -- so
    the M6c column is the one that should be read when choosing a variant.
    """
    pools = {
        "qualified only (PA>=100, 2021-2025)": qualified_only_history(players),
        "full history (all PA, 2019-2025)": pool,
    }
    rows: list[dict] = []
    for pool_name, history in pools.items():
        for baseline in ("definition", "cohort"):
            projected = marcel.project(
                history, league_baseline=baseline, recenter=(baseline == "cohort")
            )
            merged = transitions.drop(
                columns=[c for c in transitions.columns if c.startswith("marcel_")],
                errors="ignore",
            ).merge(
                projected.rename(columns={"marcel_pred": MARCEL_FEATURE}),
                left_on=["player_id", "Season_t"],
                right_on=["player_id", "origin_season"],
                how="left",
            )
            assert merged[MARCEL_FEATURE].notna().all()

            raw_parts, calibrated_parts = [], []
            for fold in sc.ROLLING_FOLDS:
                train, test = sc.split_fold(merged, fold)
                alpha, _ = sc.select_alpha(train, [MARCEL_FEATURE], sc.ALPHA_GRID_WIDE)
                model = sc.ridge_pipeline([MARCEL_FEATURE], alpha)
                model.fit(train[[MARCEL_FEATURE]], train[sc.TARGET])
                truth = test[sc.TARGET].to_numpy(float)
                raw_parts.append((truth, test[MARCEL_FEATURE].to_numpy(float)))
                calibrated_parts.append(
                    (truth, model.predict(test[[MARCEL_FEATURE]]))
                )

            def pooled_mae(parts):
                y = np.concatenate([a for a, _ in parts])
                p = np.concatenate([b for _, b in parts])
                return sc.mae(y, p)

            test_rows = merged[merged["Season_t"].isin([2022, 2023, 2024])]
            rows.append(
                {
                    "history_pool": pool_name,
                    "league_baseline": baseline,
                    "recentered": baseline == "cohort",
                    "n_test": sum(len(a) for a, _ in raw_parts),
                    "mean_seasons_used": float(test_rows["marcel_seasons_used"].mean()),
                    "mean_reliability": float(test_rows["marcel_reliability"].mean()),
                    "mae_M6_raw": pooled_mae(raw_parts),
                    "mae_M6c_calibrated": pooled_mae(calibrated_parts),
                }
            )
    return pd.DataFrame(rows)


def build_transitions_with_marcel(
    players: pd.DataFrame, metric: str = "wRC+"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Marcel reads the widened 2019-2025 no-PA-minimum pool; the focal cohort
    # and every model feature are unchanged, so the Finding 2 anchors hold.
    pool = sd.load_history_pool()
    projections = marcel.project(
        pool, league_baseline="definition", recenter=False, metric=metric
    )
    transitions = sc.build_transitions(players)
    merged = transitions.merge(
        projections.rename(columns={"marcel_pred": MARCEL_FEATURE}),
        left_on=["player_id", "Season_t"],
        right_on=["player_id", "origin_season"],
        how="left",
        validate="one_to_one",
    )
    assert merged[MARCEL_FEATURE].notna().all()
    sc.assert_no_future_columns([MARCEL_FEATURE])
    return merged, projections


def coverage_table(transitions: pd.DataFrame) -> pd.DataFrame:
    """How much history each origin season actually had available."""
    rows: list[dict] = []
    for season, block in transitions.groupby("Season_t"):
        counts = block["marcel_seasons_used"].value_counts()
        rows.append(
            {
                "origin_season": int(season),
                "n": len(block),
                "max_lookback_seasons_possible": min(3, int(season) - 2020),
                "players_with_1_season": int(counts.get(1, 0)),
                "players_with_2_seasons": int(counts.get(2, 0)),
                "players_with_3_seasons": int(counts.get(3, 0)),
                "mean_seasons_used": float(block["marcel_seasons_used"].mean()),
                "mean_reliability": float(block["marcel_reliability"].mean()),
                "league_baseline_wrcplus": float(
                    block["marcel_league_baseline"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def stratify_by_history_depth(
    pooled_predictions: pd.DataFrame, transitions: pd.DataFrame
) -> pd.DataFrame:
    """Split the pooled test cases by how many lookback seasons Marcel actually had.

    Added as a diagnostic *after* observing that this Marcel trails B2V, which the
    2021-2025 data window makes a live suspicion (see `marcel.py` and
    decisions_log D9). It changes no model, no feature and no hyper-parameter --
    it only asks whether Marcel's deficit shrinks as its history fills in. If the
    deficit is flat in history depth, the truncation excuse does not hold.
    """
    keyed = transitions.set_index(["player_id", "Season_t"])
    frame = pooled_predictions.copy()
    key = pd.MultiIndex.from_arrays([frame["player_id"], frame["Season_t"]])
    frame["marcel_seasons_used"] = key.map(keyed["marcel_seasons_used"])
    frame["marcel_reliability"] = key.map(keyed["marcel_reliability"])

    rows: list[dict] = []
    for seasons, block in frame.groupby("marcel_seasons_used"):
        pivot = block.pivot_table(
            index=["player_id", "Season_t"], columns="model", values="prediction"
        ).sort_index()
        truth = (
            block.drop_duplicates(["player_id", "Season_t"])
            .set_index(["player_id", "Season_t"])
            .loc[pivot.index, "y_true"]
            .to_numpy(float)
        )
        focal = np.asarray([player_id for player_id, _ in pivot.index])
        samples = sc.cluster_bootstrap_indices(focal)
        record = {
            "marcel_seasons_used": int(seasons),
            "n_test": len(truth),
            "n_focal_players": int(pd.unique(focal).size),
            "mean_marcel_reliability": float(block["marcel_reliability"].mean()),
        }
        for model in ["M0 scalar baseline", "M1 B2V 5 tools", MARCEL_MODEL,
                      "M6c Marcel calibrated", "M7 Marcel + B2V"]:
            record[f"mae_{model}"] = sc.mae(truth, pivot[model].to_numpy(float))
        contrast = sc.paired_bootstrap_difference(
            truth,
            pivot["M6c Marcel calibrated"].to_numpy(float),
            pivot["M1 B2V 5 tools"].to_numpy(float),
            focal,
            samples=samples,
        )
        record["marcel_minus_b2v_advantage"] = contrast["improvement_mae"]
        record["ci_low"] = contrast["ci_low"]
        record["ci_high"] = contrast["ci_high"]
        record["excludes_zero"] = contrast["excludes_zero"]
        rows.append(record)
    return pd.DataFrame(rows)


def run_target(
    players: pd.DataFrame, target: sc.Target
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transitions, _ = build_transitions_with_marcel(players, TARGET_METRIC[target.key])
    frame = sc.prepare_for_target(transitions, target)
    specs = model_specs(target)

    fold_frames, paired_frames, tuning_frames = [], [], []
    per_fold: list[pd.DataFrame] = []
    for fold in sc.ROLLING_FOLDS:
        train, test = sc.split_fold(frame, fold)
        results, tuning = sc.run_models(
            train, test, specs, GRID, fold_label=fold, target=target.column
        )
        tuning.insert(0, "target", target.key)
        tuning.insert(1, "alpha_grid", GRID_NAME)
        tuning_frames.append(tuning)

        predictions = pd.concat(
            [
                sc.attach_fit_metadata(
                    pd.concat([r.predictions for r in results], ignore_index=True),
                    results,
                ),
                unfitted_marcel_predictions(test, fold, target),
            ],
            ignore_index=True,
        )
        predictions["target"] = target.key
        per_fold.append(predictions)

        summary, paired = sc.evaluate_predictions(
            predictions, fold, PAIRS, extra={"target": target.key}
        )
        fold_frames.append(summary)
        paired_frames.append(paired)

    pooled = pd.concat(per_fold, ignore_index=True)
    pooled_summary, pooled_paired = sc.evaluate_predictions(
        pooled, "pooled", PAIRS, pooled=True, extra={"target": target.key}
    )
    fold_frames.append(pooled_summary)
    paired_frames.append(pooled_paired)

    depth = (
        stratify_by_history_depth(pooled, frame).assign(target=target.key)
        if target.key == sc.PRIMARY_TARGET
        else pd.DataFrame()
    )
    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
        pooled,
        pd.concat(tuning_frames, ignore_index=True),
        depth,
    )


def main() -> None:
    (sc.RESULTS_DIR / "task3").mkdir(parents=True, exist_ok=True)
    players = sc.load_player_seasons()
    transitions, _ = build_transitions_with_marcel(players)

    variants = marcel_variant_table(transitions, players, sd.load_history_pool())
    variants.to_csv(sc.RESULTS_DIR / "task3" / "marcel_variants.csv", index=False)
    print("\n  Marcel history pool x league baseline (wRC+):")
    print(variants.to_string(index=False))

    coverage = coverage_table(transitions)
    coverage.to_csv(sc.RESULTS_DIR / "task3" / "marcel_coverage.csv", index=False)
    print("\n  Marcel history coverage by origin season:")
    print(coverage.to_string(index=False))

    blocks = [run_target(players, target) for target in sc.TARGETS.values()]
    performance = pd.concat([b[0] for b in blocks], ignore_index=True)
    paired = pd.concat([b[1] for b in blocks], ignore_index=True)
    signs = sc.sign_consistency(paired, PAIRS, by="target")
    depth = pd.concat([b[4] for b in blocks if not b[4].empty], ignore_index=True)

    performance.to_csv(sc.RESULTS_DIR / "task3" / "fold_performance.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task3" / "paired_differences.csv", index=False)
    signs.to_csv(sc.RESULTS_DIR / "task3" / "sign_consistency.csv", index=False)
    depth.to_csv(sc.RESULTS_DIR / "task3" / "history_depth_strata.csv", index=False)
    pd.concat([b[2] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task3" / "predictions.csv", index=False
    )
    pd.concat([b[3] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task3" / "alpha_tuning.csv", index=False
    )
    (sc.RESULTS_DIR / "task3" / "environment.json").write_text(
        json.dumps(
            {
                **sc.environment_stamp(),
                "marcel_season_weights": list(marcel.SEASON_WEIGHTS),
                "marcel_regression_pa": marcel.REGRESSION_PA_PER_WEIGHT
                * sum(marcel.SEASON_WEIGHTS),
                "marcel_age_pivot": marcel.AGE_PIVOT,
                "marcel_age_slope_over": marcel.AGE_SLOPE_OVER,
                "marcel_age_slope_under": marcel.AGE_SLOPE_UNDER,
                "marcel_metrics": {k: marcel.METRICS[v] for k, v in TARGET_METRIC.items()},
                "marcel_playing_time": "200 + 0.5*PA_t + 0.1*PA_{t-1}"
                " (multiplies the rate only for counting WAR)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for target in sc.TARGETS.values():
        block = performance[performance["target"].eq(target.key)]
        print(f"\n=== {target.label} -- MAE by fold x model ===")
        print(block.pivot(index="model", columns="fold", values="mae").round(4).to_string())
        print(f"\n  paired differences ({target.label}):")
        print(
            paired[paired["target"].eq(target.key) & paired["fold"].eq("pooled")][
                ["candidate", "baseline", "improvement_mae", "ci_low", "ci_high",
                 "excludes_zero"]
            ].to_string(index=False)
        )

    print("\n=== Pooled wRC+ test cases split by Marcel history depth ===")
    print(depth.to_string(index=False))
    print("\n=== Sign consistency across folds A, B, C ===")
    print(
        signs[["target", "candidate", "baseline", "delta_mae_fold_A",
               "delta_mae_fold_B", "delta_mae_fold_C", "all_three_same_sign",
               "n_folds_ci_excludes_zero"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
