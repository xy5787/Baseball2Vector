"""Phase 2: empirical Δtool feasibility distributions.

Inputs
------
outputs/tables/transitions.csv (Phase 1)

Outputs
-------
outputs/tables/feasibility_overall.csv
outputs/tables/feasibility_conditional.csv
outputs/tables/feasibility_lookup.json
outputs/logs/02_feasibility.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.feasibility import (  # noqa: E402
    build_lookup,
    check_delta_feasibility,
    compute_conditional_distribution,
    compute_overall_distribution,
)

cfg = load_config()
np.random.seed(cfg["seed"])

print("=" * 60)
print("Phase 2: feasibility distributions")
print("=" * 60)

transitions = pd.read_csv(cfg["output"]["tables_dir"] / "transitions.csv")
print(f"  Loaded {len(transitions)} transition records")

tool_order = cfg["transitions"]["tool_order"]
bin_edges = cfg["feasibility"]["bin_edges"]
bin_labels = cfg["feasibility"]["bin_labels"]
quantiles = cfg["feasibility"]["quantiles"]

overall = compute_overall_distribution(transitions, tool_order, quantiles)
conditional = compute_conditional_distribution(
    transitions, tool_order, bin_edges, bin_labels, quantiles
)
lookup = build_lookup(transitions, tool_order, bin_edges, bin_labels)

print("\n  Overall Δtool distribution:")
print(overall.round(2).to_string(index=False))

print("\n  Conditional (tool x t-season value bin):")
print(conditional.round(2).to_string(index=False))

# Sparse-bin check: report any (tool, bin) combos with n < 20 (thin evidence)
sparse = conditional[conditional["n"] < 20]
if not sparse.empty:
    print("\n  WARNING: sparse bins (n < 20) — treat their percentiles cautiously:")
    print(sparse[["tool", "current_bin", "n"]].to_string(index=False))

# Illustration 1: instruction's own framing — "already-65 Power rising further"
# vs. "40 Power rising further".
print("\n  Illustration — P(Δ >= +10) by starting bin, for each tool:")
illustration_rows = []
for tool in tool_order:
    for label in bin_labels:
        deltas = lookup["tools"][tool][label]
        if not deltas:
            continue
        p_ge_10 = 100 - np.searchsorted(deltas, 10.0, side="left") / len(deltas) * 100
        illustration_rows.append(
            {"tool": tool, "current_bin": label, "n": len(deltas), "pct_delta_ge_10": p_ge_10}
        )
illustration_df = pd.DataFrame(illustration_rows)
print(illustration_df.round(1).to_string(index=False))

# Illustration 2: the motivating Soto case — 2024 Power +1SD (+10).
soto = transitions[
    (transitions["player_name"] == "Juan Soto") & (transitions["season_t"] == 2024)
]
soto_case = None
if not soto.empty:
    current = float(soto.iloc[0]["vec_t_power"])
    pct = check_delta_feasibility(lookup, "Power", current, 10.0)
    soto_case = {
        "player": "Juan Soto",
        "season_t": 2024,
        "tool": "Power",
        "current_value": current,
        "hypothetical_delta": 10.0,
        "percentile_of_delta_10_in_bin": pct,
    }
    print(
        f"\n  Soto check: Power={current:.1f} (2024) -> P(observed Δ <= +10 "
        f"in this bin) = {pct}"
        if pct is not None
        else "\n  Soto check: bin has no observations"
    )

out_tables = cfg["output"]["tables_dir"]
overall.to_csv(out_tables / "feasibility_overall.csv", index=False)
conditional.to_csv(out_tables / "feasibility_conditional.csv", index=False)
with open(out_tables / "feasibility_lookup.json", "w") as f:
    json.dump(lookup, f)
print(f"\n  Saved -> {out_tables}/feasibility_{{overall,conditional}}.csv, feasibility_lookup.json")

log = {
    "bin_edges": bin_edges,
    "bin_labels": bin_labels,
    "sparse_bins_n_lt_20": sparse[["tool", "current_bin", "n"]].to_dict("records"),
    "age_dimension": "omitted_missing_source_user_approved",
    "soto_illustration": soto_case,
}
out_log = cfg["output"]["logs_dir"] / "02_feasibility.json"
with open(out_log, "w") as f:
    json.dump(log, f, indent=2, default=str)
print(f"  Log -> {out_log}")
