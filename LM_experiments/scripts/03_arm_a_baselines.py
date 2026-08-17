"""Phase 3, Arm A: statistical baselines (LR basic/extended + KNN k=10).

Time split: train on all windows except the last (21->22, 22->23, 23->24),
test on the last window (24->25). No random split: future transition
labels remain unavailable during training, while past player history may recur.

Inputs
------
outputs/tables/transitions.csv (Phase 1)
../data/processed/v3_results.csv (KNN candidate pool)

Outputs
-------
outputs/tables/arm_a_predictions.csv   (long format: record_id, target, model, y_true, y_pred)
outputs/tables/arm_a_metrics.csv       (model x target: train_r2, test_mae, test_r2)
outputs/tables/lr_coefficients.csv     (model x target x feature: coef, se, ci_lo, ci_hi)
outputs/logs/03_arm_a_baselines.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.baseline_models import (  # noqa: E402
    TARGET_COLS,
    build_knn_pool,
    fit_lr,
    knn_predict,
    time_split,
)
from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.transitions import drop_ambiguous_keys, load_v3_results  # noqa: E402

cfg = load_config()
np.random.seed(cfg["seed"])

print("=" * 60)
print("Phase 3, Arm A: statistical baselines")
print("=" * 60)

transitions = pd.read_csv(cfg["output"]["tables_dir"] / "transitions.csv")
tool_order = cfg["transitions"]["tool_order"]
min_pa = cfg["transitions"]["min_pa"]

knn_k = cfg["baselines"]["knn_k"]
ci_alpha = cfg["baselines"]["ci_alpha"]
knn_model_name = f"knn_k{knn_k}"
windows = [tuple(w) for w in cfg["transitions"]["windows"]]
test_window = windows[-1]
train_windows = windows[:-1]
train_seasons = sorted({s for w in train_windows for s in w})

train_df, test_df = time_split(transitions, test_window)
print(
    f"  Time split: train={train_windows} (n={len(train_df)}), "
    f"test={test_window} (n={len(test_df)})"
)

raw = load_v3_results(cfg["data"]["v3_results"])
clean, _ = drop_ambiguous_keys(raw)
knn_pool = build_knn_pool(clean, train_seasons, tool_order, min_pa)
print(f"  KNN candidate pool: {len(knn_pool)} player-seasons from {train_seasons}")

pred_rows: list[dict] = []
metric_rows: list[dict] = []
coef_rows: list[dict] = []

for target in ["WAR", "wRC+", "OPS"]:
    y_col = TARGET_COLS[target]
    y_test = test_df[y_col].to_numpy(dtype=float)

    for extended in (False, True):
        model_name = "lr_extended" if extended else "lr_basic"
        fit = fit_lr(
            train_df, tool_order, target, extended=extended, alpha=ci_alpha
        )
        y_pred = fit.predict(test_df)

        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        metric_rows.append(
            {
                "model": model_name,
                "target": target,
                "train_r2": fit.r2_train,
                "test_mae": test_mae,
                "test_r2": test_r2,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )
        for _, r in fit.coef_table.iterrows():
            coef_rows.append({"model": model_name, "target": target, **r.to_dict()})
        for rid, yt, yp in zip(test_df["record_id"], y_test, y_pred):
            pred_rows.append(
                {"record_id": rid, "target": target, "model": model_name, "y_true": yt, "y_pred": yp}
            )
        print(f"  {model_name:12s} {target:5s}: train R²={fit.r2_train:.3f}  test MAE={test_mae:.3f}  test R²={test_r2:.3f}")

    y_pred_knn = knn_predict(test_df, knn_pool, tool_order, target, k=knn_k)
    test_mae_knn = mean_absolute_error(y_test, y_pred_knn)
    test_r2_knn = r2_score(y_test, y_pred_knn)
    metric_rows.append(
        {
            "model": knn_model_name,
            "target": target,
            "train_r2": float("nan"),
            "test_mae": test_mae_knn,
            "test_r2": test_r2_knn,
            "n_train": len(knn_pool),
            "n_test": len(test_df),
        }
    )
    for rid, yt, yp in zip(test_df["record_id"], y_test, y_pred_knn):
        pred_rows.append(
            {"record_id": rid, "target": target, "model": knn_model_name, "y_true": yt, "y_pred": yp}
        )
    print(f"  {knn_model_name:12s} {target:5s}: {'':>16} test MAE={test_mae_knn:.3f}  test R²={test_r2_knn:.3f}")

predictions_df = pd.DataFrame(pred_rows)
metrics_df = pd.DataFrame(metric_rows)
coef_df = pd.DataFrame(coef_rows)

out_tables = cfg["output"]["tables_dir"]
predictions_df.to_csv(out_tables / "arm_a_predictions.csv", index=False)
metrics_df.to_csv(out_tables / "arm_a_metrics.csv", index=False)
coef_df.to_csv(out_tables / "lr_coefficients.csv", index=False)
print(f"\n  Saved -> {out_tables}/arm_a_{{predictions,metrics}}.csv, lr_coefficients.csv")

print("\n  --- LR basic coefficients (ΔwRC+) ---")
print(
    coef_df[(coef_df["model"] == "lr_basic") & (coef_df["target"] == "wRC+")]
    .round(3)
    .to_string(index=False)
)

log = {
    "test_window": test_window,
    "train_windows": train_windows,
    "train_seasons": train_seasons,
    "n_train": len(train_df),
    "n_test": len(test_df),
    "knn_pool_size": len(knn_pool),
    "knn_k": knn_k,
    "ci_alpha": ci_alpha,
    "age_feature": "omitted_missing_source_user_approved",
    "extended_features": "delta_vec + base_vec + same-tool interactions",
    "ci_method": "classical_ols_homoskedastic",
    "metrics": metrics_df.astype(object)
    .where(pd.notna(metrics_df), None)
    .to_dict("records"),
}
out_log = cfg["output"]["logs_dir"] / "03_arm_a_baselines.json"
with open(out_log, "w") as f:
    json.dump(log, f, indent=2, default=str)
print(f"  Log -> {out_log}")
