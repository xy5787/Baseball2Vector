"""Phase 4: stratified evaluation set and arm-agnostic scoring harness."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.baseline_models import (  # noqa: E402
    build_knn_pool,
    knn_comparables,
    time_split,
)
from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.scoring import STRATUM_FLAGS, score  # noqa: E402
from lm_experiments.strata import assign_strata, stratum_overlap_report  # noqa: E402
from lm_experiments.transitions import drop_ambiguous_keys, load_v3_results  # noqa: E402


def membership_sizes(df: pd.DataFrame) -> dict[str, int]:
    """Return overlapping special-stratum counts and exclusive baseline count."""
    counts = {name: int(df[flag].sum()) for name, flag in STRATUM_FLAGS.items()}
    any_special = df[list(STRATUM_FLAGS.values())].any(axis=1)
    counts["baseline"] = int((~any_special).sum())
    return counts


cfg = load_config()
rng = np.random.default_rng(cfg["seed"])
eval_cfg = cfg["evaluation"]
knn_k = cfg["baselines"]["knn_k"]
knn_model_name = f"knn_k{knn_k}"

print("=" * 60)
print("Phase 4: stratified evaluation set + scoring harness")
print("=" * 60)

transitions = pd.read_csv(cfg["output"]["tables_dir"] / "transitions.csv")
tool_order = cfg["transitions"]["tool_order"]
stratified = assign_strata(
    transitions,
    tool_order,
    elite_percentile=eval_cfg["elite_percentile"],
    strength_threshold=eval_cfg["strength_threshold"],
    weakness_threshold=eval_cfg["weakness_threshold"],
)
print("\n  Overlapping evaluation-stratum membership sizes:")
print(pd.Series(membership_sizes(stratified)).to_string())
print("\n  Exclusive priority labels retained for qualitative sampling:")
print(stratified["stratum"].value_counts().to_string())

overlap = stratum_overlap_report(stratified)
print("\n  Pairwise is_* overlap (diagnostic):")
print(overlap.to_string())

out_tables = cfg["output"]["tables_dir"]
stratified.to_csv(out_tables / "transitions_stratified.csv", index=False)
overlap.to_csv(out_tables / "stratum_overlap.csv", index_label="stratum_flag")

windows = [tuple(window) for window in cfg["transitions"]["windows"]]
test_window = windows[-1]
_, gt_test = time_split(stratified, test_window)
pred_long = pd.read_csv(out_tables / "arm_a_predictions.csv")
target_rename = {"WAR": "delta_war", "wRC+": "delta_wrc", "OPS": "delta_ops"}

raw = load_v3_results(cfg["data"]["v3_results"])
clean, _ = drop_ambiguous_keys(raw)
train_seasons = sorted({season for window in windows[:-1] for season in window})
pool = build_knn_pool(
    clean, train_seasons, tool_order, cfg["transitions"]["min_pa"]
)
_, test_df = time_split(transitions, test_window)
comparables = knn_comparables(test_df, pool, tool_order, k=knn_k)

overall_rows: list[pd.DataFrame] = []
stratum_rows: list[pd.DataFrame] = []
window_rows: list[pd.DataFrame] = []
jaccard_selfcheck = pd.DataFrame()
jaccard_mismatches = pd.DataFrame()

for model in pred_long["model"].unique():
    sub = pred_long[pred_long["model"] == model]
    wide = (
        sub.pivot(index="record_id", columns="target", values="y_pred")
        .reset_index()
        .rename(columns=target_rename)
    )
    if model == knn_model_name:
        wide["comparables"] = wide["record_id"].map(comparables)
        report = score(wide, gt_test, reference_comparables=comparables)
        jaccard_selfcheck = report.get("jaccard", pd.DataFrame())
        jaccard_mismatches = report.get("jaccard_mismatches", pd.DataFrame())
    else:
        report = score(wide, gt_test)

    for table, sink in (
        (report["overall"], overall_rows),
        (report["by_stratum"], stratum_rows),
        (report["by_window"], window_rows),
    ):
        table = table.copy()
        table.insert(0, "model", model)
        sink.append(table)

    print(
        f"\n  [{model}] overall "
        f"(n={report['n_scored']}/{report['n_ground_truth']}):"
    )
    print(report["overall"].round(3).to_string(index=False))

overall_df = pd.concat(overall_rows, ignore_index=True)
stratum_df = pd.concat(stratum_rows, ignore_index=True)
window_df = pd.concat(window_rows, ignore_index=True)
overall_df.to_csv(out_tables / "scoring_overall.csv", index=False)
stratum_df.to_csv(out_tables / "scoring_by_stratum.csv", index=False)
window_df.to_csv(out_tables / "scoring_by_window.csv", index=False)

print("\n  --- sign_accuracy by overlapping stratum (ΔWAR) ---")
print(
    stratum_df[stratum_df["target"] == "war"]
    .pivot(index="stratum", columns="model", values="sign_accuracy")
    .round(3)
    .to_string()
)
print("\n  --- n by overlapping stratum (ΔWAR) ---")
print(
    stratum_df[stratum_df["target"] == "war"]
    .pivot(index="stratum", columns="model", values="n")
    .to_string()
)

jaccard_selfcheck.to_csv(out_tables / "jaccard_knn_selfcheck.csv", index=False)
jaccard_mismatches.to_csv(out_tables / "jaccard_mismatches.csv", index=False)
if not jaccard_selfcheck.empty:
    print(
        f"\n  jaccard@10 self-check ({knn_model_name} vs own list): "
        f"mean={jaccard_selfcheck['jaccard_at_10'].mean():.3f}"
    )

partial_n = min(int(eval_cfg["partial_eval_n"]), len(gt_test))
sample_ids = rng.choice(gt_test["record_id"].to_numpy(), size=partial_n, replace=False)
lr_basic_wide = (
    pred_long[pred_long["model"] == "lr_basic"]
    .pivot(index="record_id", columns="target", values="y_pred")
    .reset_index()
    .rename(columns=target_rename)
)
partial_preds = lr_basic_wide[lr_basic_wide["record_id"].isin(sample_ids)]
partial_report = score(partial_preds, gt_test)
partial_report["overall"].to_csv(
    out_tables / "scoring_partial_eval_demo.csv", index=False
)
print(
    f"\n  Partial evaluation verified: "
    f"{partial_report['n_scored']}/{partial_report['n_ground_truth']} records"
)

# Verify that the contamination hook emits every supplied transition window.
perfect_shape_predictions = stratified[
    ["record_id", "delta_war", "delta_wrc", "delta_ops"]
].copy()
window_hook = score(perfect_shape_predictions, stratified)["by_window"]
verified_windows = int(window_hook[["season_t", "season_t1"]].drop_duplicates().shape[0])
assert verified_windows == len(windows)

log = {
    "thresholds": eval_cfg,
    "membership_semantics": "overlapping special strata; baseline has no special flag",
    "full_membership_sizes": membership_sizes(stratified),
    "test_membership_sizes": membership_sizes(gt_test),
    "exclusive_priority_sizes": stratified["stratum"].value_counts().to_dict(),
    "test_window": test_window,
    "n_test": len(gt_test),
    "partial_eval_n": len(partial_preds),
    "contamination_window_hook_verified": verified_windows,
    "jaccard_mismatch_rows": len(jaccard_mismatches),
}
out_log = cfg["output"]["logs_dir"] / "04_stratified_scoring.json"
with open(out_log, "w") as file:
    json.dump(log, file, indent=2, default=str)

print(f"\n  Saved Phase 4 tables -> {out_tables}")
print(f"  Log -> {out_log}")
