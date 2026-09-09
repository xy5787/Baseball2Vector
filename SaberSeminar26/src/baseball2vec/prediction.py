"""Prospective calibrated scalar, vector, context, and combined Ridge models."""

from __future__ import annotations

import platform
import sys

import numpy as np
import pandas as pd

from .repro_common import (
    ALPHAS,
    N_BOOT,
    RUN_ROOT,
    SAFE_TOOL_COLS,
    SEED,
    TOOL_COLS,
    build_transitions,
    load_player_seasons,
    metric,
    percentile_ci,
    ridge_pipeline,
    select_alpha,
)


TARGET = "wRC+_t1"
TRAIN_END_YEARS = [2022, 2023, 2024]
TEST_START, TEST_END = 2024, 2025


def feature_names(cols: list[str]) -> list[str]:
    return [f"{col}_t" for col in cols]


def fit_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_variant: str,
    vector_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {
        "League mean": None,
        "Carry-forward wRC+": None,
        "Calibrated wRC+ Ridge": ["wRC+_t"],
        "wRC+ + age + PA Ridge": ["wRC+_t", "age_t_t", "PA_t"],
        "B2V Ridge": feature_names(vector_cols),
        "B2V + age + PA Ridge": feature_names(vector_cols) + ["age_t_t", "PA_t"],
        "wRC+ + B2V + age + PA Ridge": ["wRC+_t", *feature_names(vector_cols), "age_t_t", "PA_t"],
    }
    base = test[["player_id", "Name", "Season_t", "Season_t1", TARGET]].copy()
    base = base.rename(columns={TARGET: "y_true"})
    predictions: list[pd.DataFrame] = []
    tuning_frames: list[pd.DataFrame] = []
    for model_name, features in models.items():
        out = base.copy()
        out["feature_variant"] = feature_variant
        out["model"] = model_name
        out["features"] = "training-target mean" if model_name == "League mean" else (
            "season-t wRC+" if model_name == "Carry-forward wRC+" else ",".join(features or [])
        )
        if model_name == "League mean":
            out["prediction"] = float(train[TARGET].mean())
            out["selected_alpha"] = np.nan
        elif model_name == "Carry-forward wRC+":
            out["prediction"] = test["wRC+_t"].to_numpy(float)
            out["selected_alpha"] = np.nan
        else:
            assert features is not None
            alpha, tuning = select_alpha(train, features, TARGET)
            tuning.insert(0, "feature_variant", feature_variant)
            tuning.insert(1, "model", model_name)
            tuning_frames.append(tuning)
            fitted = ridge_pipeline(features, alpha)
            fitted.fit(train[features], train[TARGET])
            out["prediction"] = fitted.predict(test[features])
            out["selected_alpha"] = alpha
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True), pd.concat(tuning_frames, ignore_index=True)


def bootstrap_results(primary: pd.DataFrame, n_train: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_order = [
        "League mean", "Carry-forward wRC+", "Calibrated wRC+ Ridge",
        "wRC+ + age + PA Ridge", "B2V Ridge", "B2V + age + PA Ridge",
        "wRC+ + B2V + age + PA Ridge",
    ]
    pivot = primary.pivot(index="player_id", columns="model", values="prediction").sort_index()
    truth = primary.drop_duplicates("player_id").set_index("player_id").loc[pivot.index, "y_true"]
    assert pivot.notna().all().all()
    assert len(truth) == len(pivot) == 358
    rng = np.random.default_rng(SEED)
    samples = [rng.integers(0, len(truth), len(truth)) for _ in range(N_BOOT)]
    metrics = ["mae", "r2", "spearman"]
    boot_values: dict[tuple[str, str], np.ndarray] = {}
    boot_rows: list[dict] = []
    summary_rows: list[dict] = []
    y = truth.to_numpy(float)
    for model_name in model_order:
        pred = pivot[model_name].to_numpy(float)
        model_row = primary[primary["model"].eq(model_name)].iloc[0]
        for metric_name in metrics:
            values = np.asarray([metric(y[idx], pred[idx], metric_name) for idx in samples])
            boot_values[(model_name, metric_name)] = values
            low, high = percentile_ci(values)
            summary_rows.append({
                "feature_variant": "within_input_season_standardized",
                "model": model_name,
                "features": model_row["features"],
                "n_train": n_train,
                "n_test": len(y),
                "train_period": "2021->2022; 2022->2023; 2023->2024",
                "test_period": "2024->2025",
                "selected_alpha": model_row["selected_alpha"],
                "metric": metric_name,
                "estimate": metric(y, pred, metric_name),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_unit": "held-out stable player ID",
                "n_boot": N_BOOT,
            })
            boot_rows.extend({
                "replicate": i,
                "model": model_name,
                "metric": metric_name,
                "value": value,
            } for i, value in enumerate(values))
    comparisons = [
        ("B2V Ridge", "Carry-forward wRC+"),
        ("B2V Ridge", "Calibrated wRC+ Ridge"),
        ("wRC+ + age + PA Ridge", "Calibrated wRC+ Ridge"),
        ("B2V + age + PA Ridge", "B2V Ridge"),
        ("B2V + age + PA Ridge", "wRC+ + age + PA Ridge"),
        ("wRC+ + B2V + age + PA Ridge", "wRC+ + age + PA Ridge"),
        ("wRC+ + B2V + age + PA Ridge", "B2V + age + PA Ridge"),
    ]
    pair_rows: list[dict] = []
    for candidate, baseline in comparisons:
        for metric_name in metrics:
            candidate_values = boot_values[(candidate, metric_name)]
            baseline_values = boot_values[(baseline, metric_name)]
            differences = baseline_values - candidate_values if metric_name == "mae" else candidate_values - baseline_values
            low, high = percentile_ci(differences)
            candidate_point = next(r["estimate"] for r in summary_rows if r["model"] == candidate and r["metric"] == metric_name)
            baseline_point = next(r["estimate"] for r in summary_rows if r["model"] == baseline and r["metric"] == metric_name)
            point = baseline_point - candidate_point if metric_name == "mae" else candidate_point - baseline_point
            pair_rows.append({
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric_name,
                "improvement": point,
                "ci_low": low,
                "ci_high": high,
                "direction": "baseline minus candidate" if metric_name == "mae" else "candidate minus baseline",
                "positive_means_candidate_better": True,
                "bootstrap_unit": "held-out stable player ID",
                "n_boot": N_BOOT,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(boot_rows), pd.DataFrame(pair_rows)


def main() -> None:
    players = load_player_seasons()
    transitions = build_transitions(players)
    transitions.to_csv(RUN_ROOT / "data" / "processed" / "transitions.generated.csv", index=False)
    train = transitions[transitions["Season_t1"].isin(TRAIN_END_YEARS)].copy()
    test = transitions[(transitions["Season_t"].eq(TEST_START)) & (transitions["Season_t1"].eq(TEST_END))].copy()

    # Explicit temporal, identity, and common-test-set assertions.
    assert len(train) == 1056 and len(test) == 358
    assert set(train["Season_t1"]) == set(TRAIN_END_YEARS)
    assert train["Season_t1"].max() < test["Season_t1"].min()
    assert (train["Season_t1"] == train["Season_t"] + 1).all()
    assert (test["Season_t1"] == test["Season_t"] + 1).all()
    assert not test.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert test["player_id"].nunique() == len(test)
    predictor_cols = ["wRC+_t", "age_t_t", "PA_t", *feature_names(TOOL_COLS), *feature_names(SAFE_TOOL_COLS)]
    assert all(name.endswith("_t") or name == "age_t_t" for name in predictor_cols)
    assert not any("2025" in name or name.endswith("_t1") for name in predictor_cols)
    assert (test["age_t_t"] == test["Season_t"] - test["birth_year_t"]).all()

    primary_pred, primary_tuning = fit_predictions(
        train, test, "within_input_season_standardized", SAFE_TOOL_COLS
    )
    archived_pred, archived_tuning = fit_predictions(
        train, test, "archived_pooled_20_80", TOOL_COLS
    )
    all_predictions = pd.concat([primary_pred, archived_pred], ignore_index=True)
    all_predictions.to_csv(RUN_ROOT / "results" / "calibrated_predictions.csv", index=False)
    pd.concat([primary_tuning, archived_tuning], ignore_index=True).to_csv(
        RUN_ROOT / "results" / "ridge_tuning.csv", index=False
    )
    summary, boot, paired = bootstrap_results(primary_pred, len(train))
    summary.to_csv(RUN_ROOT / "results" / "calibrated_prediction.csv", index=False)
    boot.to_csv(RUN_ROOT / "results" / "calibrated_prediction_bootstrap.csv", index=False)
    paired.to_csv(RUN_ROOT / "results" / "paired_model_differences.csv", index=False)

    vector_models = ["B2V Ridge", "B2V + age + PA Ridge", "wRC+ + B2V + age + PA Ridge"]
    comparison = all_predictions[all_predictions["model"].isin(vector_models)].pivot(
        index=["player_id", "model"], columns="feature_variant", values="prediction"
    ).reset_index()
    comparison["safe_minus_archived"] = (
        comparison["within_input_season_standardized"] - comparison["archived_pooled_20_80"]
    )
    comparison.to_csv(RUN_ROOT / "results" / "scaling_prediction_differences.csv", index=False)

    flow = pd.DataFrame([
        {"stage": "source player-seasons", "n_rows": 2309, "n_players": players["Name"].nunique(), "exclusion": "none"},
        {"stage": "stable-ID matched player-seasons", "n_rows": len(players), "n_players": players["player_id"].nunique(), "exclusion": "15 ambiguous and 8 unmatched rows"},
        {"stage": "all adjacent transitions", "n_rows": len(transitions), "n_players": transitions["player_id"].nunique(), "exclusion": "non-adjacent or single-season records"},
        {"stage": "training transitions", "n_rows": len(train), "n_players": train["player_id"].nunique(), "exclusion": "held-out 2024->2025"},
        {"stage": "held-out test transitions", "n_rows": len(test), "n_players": test["player_id"].nunique(), "exclusion": "none after common-set construction"},
    ])
    flow.to_csv(RUN_ROOT / "results" / "prediction_sample_flow.csv", index=False)
    print(summary.pivot(index="model", columns="metric", values="estimate").to_string())
    print("\nPaired improvements:\n", paired.to_string(index=False))
    print(f"\nPython {platform.python_version()}; alpha grid {ALPHAS}; executable {sys.executable}")


if __name__ == "__main__":
    main()
