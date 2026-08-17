"""Phase 1: build the observed transition dataset.

Inputs
------
../data/processed/v3_results.csv (main baseball2vec repo, read-only)

Outputs
-------
outputs/tables/transitions.csv
outputs/logs/01_build_transitions.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.transitions import (  # noqa: E402
    build_transitions,
    drop_ambiguous_keys,
    load_v3_results,
)

cfg = load_config()
np.random.seed(cfg["seed"])

cfg["output"]["tables_dir"].mkdir(parents=True, exist_ok=True)
cfg["output"]["logs_dir"].mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Phase 1: build transition dataset")
print("=" * 60)

raw = load_v3_results(cfg["data"]["v3_results"])
print(f"  Loaded: {len(raw)} player-seasons, {raw['Name'].nunique()} unique names")

clean, dropped = drop_ambiguous_keys(raw)
if not dropped.empty:
    print(f"  Dropped {len(dropped)} rows with ambiguous (Name, Season) key:")
    print(dropped[["Name", "Season", "PA", "WAR"]].to_string(index=False))

windows = [tuple(w) for w in cfg["transitions"]["windows"]]
tool_order = cfg["transitions"]["tool_order"]

transitions, log = build_transitions(
    clean,
    windows=windows,
    tool_order=tool_order,
    min_pa=cfg["transitions"]["min_pa"],
    source_col_prefix=cfg["transitions"]["source_col_prefix"],
)
log["dropped_ambiguous_rows"] = dropped[["Name", "Season", "PA", "WAR"]].to_dict(
    "records"
)
log["min_pa"] = cfg["transitions"]["min_pa"]
log["identity"] = {
    "source_key": cfg["transitions"]["identity_key"],
    "player_id_is_surrogate": True,
    "duplicate_policy": cfg["transitions"]["duplicate_policy"],
    "warning": (
        "The source has no stable numeric player ID or team column. Name is a "
        "surrogate key; every ambiguous (Name, Season) group is excluded."
    ),
}
log["age"] = {
    "mode": cfg["transitions"]["age_mode"],
    "included_in_output": False,
}
log["total_records"] = len(transitions)

print("\n  Pairs per window (PA >= {}):".format(cfg["transitions"]["min_pa"]))
for k, v in log["windows"].items():
    print(f"    {k}: {v}")
print(f"  Total transition records: {len(transitions)}")
print(f"  Unique players: {transitions['player_id'].nunique()}")

if log["duplicate_record_ids"]:
    print(f"  WARNING: {len(log['duplicate_record_ids'])} duplicate record_id(s) found")

# Sanity check: recomputed pentagon area vs. the ZScore_Area column already
# present in v3_results.csv (must be built with the same axis order).
check = clean.merge(
    transitions[["player_id", "season_t", "pentagon_area_t"]],
    left_on=["Name", "Season"],
    right_on=["player_id", "season_t"],
    how="inner",
)
max_abs_diff = float((check["ZScore_Area"] - check["pentagon_area_t"]).abs().max())
print(f"\n  Pentagon area cross-check vs. ZScore_Area: max|diff| = {max_abs_diff:.2e}")
log["area_crosscheck_max_abs_diff"] = max_abs_diff

# Decliner count preview (delta_vec sums positive, delta_war negative) — this
# stratum matters most in Phase 4, worth a first look here.
delta_vec_cols = [c for c in transitions.columns if c.startswith("delta_vec_")]
vec_sum = transitions[delta_vec_cols].sum(axis=1)
decliner_mask = (vec_sum > 0) & (transitions["delta_war"] < 0)
log["decliner_preview_count"] = int(decliner_mask.sum())
print(f"  Preview: {decliner_mask.sum()} records with Δvector sum > 0 but ΔWAR < 0")

out_csv = cfg["output"]["tables_dir"] / "transitions.csv"
transitions.to_csv(out_csv, index=False)
print(f"\n  Saved -> {out_csv}")

out_log = cfg["output"]["logs_dir"] / "01_build_transitions.json"
with open(out_log, "w") as f:
    json.dump(log, f, indent=2, default=str)
print(f"  Log -> {out_log}")
