"""Leakage-checked final-year prospective prediction using season-t features only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import (  # noqa: E402
    EXP_ROOT,
    TOOL_COLS,
    build_transitions,
    cluster_bootstrap_indices,
    ensure_dirs,
    load_config,
    metric_value,
    percentile_ci,
)


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def select_alpha(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    n_splits: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict] = []
    for alpha in alphas:
        fold_mae = []
        for train_idx, valid_idx in splitter.split(X, y, groups):
            assert set(groups[train_idx]).isdisjoint(groups[valid_idx])
            model = ridge_pipeline(alpha)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            fold_mae.append(metric_value(y[valid_idx], pred, "mae"))
        rows.append({"alpha": alpha, "mean_inner_mae": np.mean(fold_mae), "sd_inner_mae": np.std(fold_mae, ddof=1)})
    scores = pd.DataFrame(rows).sort_values(["mean_inner_mae", "alpha"], kind="mergesort")
    return float(scores.iloc[0]["alpha"]), scores


def assert_no_future_features(feature_sets: dict[str, list[str]], transitions: pd.DataFrame) -> None:
    forbidden_fragments = ["_t1", "Season_t1", "PA_t1", "WAR_t1", "wRC+_t1", "OPS_t1"]
    for model, features in feature_sets.items():
        assert features, f"No features for {model}"
        assert set(features).issubset(transitions.columns)
        for feature in features:
            assert not any(fragment in feature for fragment in forbidden_fragments), (
                f"Future-season leakage in {model}: {feature}"
            )
        assert all(feature.endswith("_t") for feature in features), features


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    player_df = pd.read_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv")
    transitions = build_transitions(player_df)
    transitions["WAR600_t"] = 600.0 * transitions["WAR_t"] / transitions["PA_t"]
    transitions["WAR600_t1"] = 600.0 * transitions["WAR_t1"] / transitions["PA_t1"]
    transitions["Age_t"] = transitions["Season_t"] - transitions["birth_year_t"]

    assert (transitions["Season_t1"] == transitions["Season_t"] + 1).all()
    assert not transitions.duplicated(["player_id", "Season_t", "Season_t1"]).any()

    train = transitions[
        transitions["Season_t1"].isin(cfg["train_transition_end_years"])
    ].copy()
    test = transitions[
        transitions["Season_t"].eq(cfg["test_transition"][0])
        & transitions["Season_t1"].eq(cfg["test_transition"][1])
    ].copy()
    assert set(train["Season_t1"]).isdisjoint({cfg["test_transition"][1]})
    assert (train["Season_t1"] <= 2024).all()
    assert (test["Season_t"] == 2024).all() and (test["Season_t1"] == 2025).all()
    assert not set(train.index).intersection(test.index)

    b2v_features = [f"{col}_t" for col in TOOL_COLS]
    feature_sets = {
        "b2v_ridge": b2v_features,
        "b2v_age_pa_ridge": b2v_features + ["Age_t", "PA_t"],
    }
    # The derived Age_t is explicitly season-t metadata; check it separately before the suffix rule.
    assert_no_future_features(
        {name: [f for f in features if f != "Age_t"] for name, features in feature_sets.items()},
        transitions,
    )
    assert "Age_t" not in test.columns[test.columns.str.endswith("_t1")]

    predictions: list[pd.DataFrame] = []
    tuning_rows: list[pd.DataFrame] = []
    target_defs = {
        "wRC+_t1": ("wRC+_t", "wRC+_t1"),
        "WAR600_t1": ("WAR600_t", "WAR600_t1"),
    }
    for target_name, (carry_col, target_col) in target_defs.items():
        y_train = train[target_col].to_numpy(float)
        y_test = test[target_col].to_numpy(float)
        base = test[["player_id", "Name", "Season_t", "Season_t1"]].copy()
        base["target"] = target_name
        base["y_true"] = y_test

        for model_name, pred in [
            ("league_mean", np.repeat(y_train.mean(), len(test))),
            ("carry_forward", test[carry_col].to_numpy(float)),
        ]:
            out = base.copy()
            out["model"] = model_name
            out["prediction"] = pred
            predictions.append(out)

        for model_name, features in feature_sets.items():
            best_alpha, tuning = select_alpha(
                train[features],
                y_train,
                train["player_id"].to_numpy(),
                cfg["ridge_alphas"],
                cfg["inner_group_folds"],
                cfg["seed"],
            )
            tuning.insert(0, "target", target_name)
            tuning.insert(1, "model", model_name)
            tuning["selected"] = tuning["alpha"].eq(best_alpha)
            tuning_rows.append(tuning)
            model = ridge_pipeline(best_alpha)
            model.fit(train[features], y_train)
            out = base.copy()
            out["model"] = model_name
            out["prediction"] = model.predict(test[features])
            out["selected_alpha"] = best_alpha
            predictions.append(out)

    prediction_df = pd.concat(predictions, ignore_index=True)
    prediction_df.to_csv(EXP_ROOT / "results" / "prospective_predictions.csv", index=False)
    pd.concat(tuning_rows, ignore_index=True).to_csv(
        EXP_ROOT / "results" / "prospective_ridge_tuning.csv", index=False
    )

    n_boot = cfg["bootstrap_replicates"]
    metrics = ["mae", "r2", "spearman"]
    summary_rows: list[dict] = []
    boot_rows: list[dict] = []
    pair_rows: list[dict] = []
    comparison_models = ["league_mean", "carry_forward", "b2v_ridge", "b2v_age_pa_ridge"]
    for target_name in target_defs:
        target_pred = prediction_df[prediction_df["target"].eq(target_name)].copy()
        pivot = target_pred.pivot(index="player_id", columns="model", values="prediction")
        truth = target_pred.drop_duplicates("player_id").set_index("player_id")["y_true"].loc[pivot.index]
        bootstrap = cluster_bootstrap_indices(pivot.index.to_numpy(), n_boot, cfg["seed"] + len(boot_rows))
        model_boot: dict[str, dict[str, list[float]]] = {
            model: {metric: [] for metric in metrics} for model in comparison_models
        }
        for replicate, idx in enumerate(bootstrap):
            y_b = truth.to_numpy()[idx]
            for model_name in comparison_models:
                pred_b = pivot[model_name].to_numpy()[idx]
                for metric in metrics:
                    value = metric_value(y_b, pred_b, metric)
                    model_boot[model_name][metric].append(value)
                    boot_rows.append(
                        {
                            "target": target_name,
                            "replicate": replicate,
                            "model": model_name,
                            "metric": metric,
                            "value": value,
                        }
                    )
        for model_name in comparison_models:
            actual_y = truth.to_numpy()
            actual_pred = pivot[model_name].to_numpy()
            for metric in metrics:
                estimate = metric_value(actual_y, actual_pred, metric)
                ci_low, ci_high = percentile_ci(model_boot[model_name][metric])
                summary_rows.append(
                    {
                        "status": "ok",
                        "target": target_name,
                        "model": model_name,
                        "features": (
                            "training-target mean" if model_name == "league_mean" else
                            "season-t target" if model_name == "carry_forward" else
                            ",".join(feature_sets[model_name])
                        ),
                        "n_train": len(train),
                        "n_test": len(test),
                        "train_period": "2021->2022;2022->2023;2023->2024",
                        "test_period": "2024->2025",
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
        comparison_specs = [
            ("league_mean", ["b2v_ridge", "b2v_age_pa_ridge"]),
            ("carry_forward", ["b2v_ridge", "b2v_age_pa_ridge"]),
            ("b2v_ridge", ["b2v_age_pa_ridge"]),
        ]
        for baseline, candidates in comparison_specs:
            for candidate in candidates:
                for metric in metrics:
                    a = np.asarray(model_boot[candidate][metric])
                    b = np.asarray(model_boot[baseline][metric])
                    diff = b - a if metric == "mae" else a - b
                    lo, hi = percentile_ci(diff)
                    point_a = metric_value(truth.to_numpy(), pivot[candidate].to_numpy(), metric)
                    point_b = metric_value(truth.to_numpy(), pivot[baseline].to_numpy(), metric)
                    pair_rows.append(
                        {
                            "target": target_name,
                            "candidate": candidate,
                            "baseline": baseline,
                            "metric": metric,
                            "improvement": point_b - point_a if metric == "mae" else point_a - point_b,
                            "ci_low": lo,
                            "ci_high": hi,
                            "positive_means_candidate_better": True,
                        }
                    )
        for metric in metrics:
            summary_rows.append(
                {
                    "status": "not_run_missing_raw",
                    "target": target_name,
                    "model": "raw_stat_age_pa_ridge",
                    "features": "season-t constituent statistics + age + PA",
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_period": "2021->2022;2022->2023;2023->2024",
                    "test_period": "2024->2025",
                    "metric": metric,
                    "estimate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "reason": "The committed processed file omits all constituent raw statistics.",
                }
            )

    pd.DataFrame(summary_rows).to_csv(EXP_ROOT / "results" / "prospective_prediction.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(EXP_ROOT / "results" / "prospective_prediction_bootstrap.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(EXP_ROOT / "results" / "prospective_prediction_pairwise.csv", index=False)
    flow = pd.DataFrame(
        [
            {"stage": "stable_id_adjacent_transitions", "n_rows": len(transitions), "n_players": transitions["player_id"].nunique()},
            {"stage": "prospective_training_transitions", "n_rows": len(train), "n_players": train["player_id"].nunique()},
            {"stage": "prospective_test_transitions", "n_rows": len(test), "n_players": test["player_id"].nunique()},
        ]
    )
    existing = pd.read_csv(EXP_ROOT / "results" / "sample_flow.csv")
    existing = existing[~existing["stage"].isin(flow["stage"])]
    pd.concat([existing, flow], ignore_index=True).to_csv(EXP_ROOT / "results" / "sample_flow.csv", index=False)
    print(pd.DataFrame(summary_rows).query("status == 'ok'").to_string(index=False))


if __name__ == "__main__":
    main()
