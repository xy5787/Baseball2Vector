"""Phase 1: build the observed season-to-season transition dataset.

A transition record pairs a player's tool vector in season t with the same
player's vector in season t+1, and carries the *actually observed* deltas in
WAR/wRC+/OPS as labels (as opposed to a hypothetical +1SD shift).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Keep the axis order used to produce ZScore_Area in the main pipeline. The
# formula is local so Phase 1 does not import scikit-learn through tools.py.
AREA_TOOL_ORDER = ["Contact", "Power", "Speed", "Defense", "Discipline"]


def pentagon_area(row: pd.Series, axes: list[str]) -> float:
    """Compute 1/2 * sum(r_i * r_(i+1) * sin(72 degrees))."""
    if len(axes) != 5:
        raise ValueError(f"Expected five axes, got {len(axes)}")
    values = [float(row[axis]) for axis in axes]
    sine_72 = np.sin(2 * np.pi / 5)
    return sum(
        0.5 * values[i] * values[(i + 1) % 5] * sine_72 for i in range(5)
    )

REQUIRED_COLUMNS = ["Name", "Season", "PA", "WAR", "wRC+", "OPS"] + [
    f"ZScore_{t}" for t in AREA_TOOL_ORDER
]


def load_v3_results(path: str | Path) -> pd.DataFrame:
    """Load v3_results.csv and assert the columns this module depends on exist.

    Args:
        path: Path to v3_results.csv.

    Returns:
        Raw DataFrame, unmodified.

    Raises:
        ValueError: If any required column is missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"v3_results.csv missing required columns: {missing}")
    return df


def drop_ambiguous_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop (Name, Season) rows that appear more than once in *df*.

    There is no Team or numeric player-id column in v3_results.csv, so a
    duplicate (Name, Season) cannot be disambiguated as a same-name collision
    vs. a mid-season trade split. Both rows are dropped rather than guessed at.

    Args:
        df: Raw v3_results.csv rows.

    Returns:
        (clean_df, dropped_rows_df) — dropped_rows_df is empty if none found.
    """
    dup_mask = df.duplicated(subset=["Name", "Season"], keep=False)
    return df[~dup_mask].copy(), df[dup_mask].copy()


def build_transitions(
    df: pd.DataFrame,
    windows: list[tuple[int, int]],
    tool_order: list[str],
    min_pa: int = 200,
    source_col_prefix: str = "ZScore_",
) -> tuple[pd.DataFrame, dict]:
    """Build one row per (player, season_t -> season_t1) transition.

    Args:
        df: Deduplicated v3_results.csv rows (see drop_ambiguous_keys).
        windows: (season_t, season_t1) pairs to include, e.g. [(2021, 2022), ...].
        tool_order: Axis order for the vec_t_* / delta_vec_* output columns.
        min_pa: Minimum PA required in *both* seasons of a transition.
        source_col_prefix: Column prefix of the tool scores in *df*.

    Returns:
        (transitions_df, log) — log records per-window pair counts and any
        anomalies found while building the dataset.
    """
    log: dict = {"windows": {}, "duplicate_record_ids": []}
    by_season = {s: g for s, g in df.groupby("Season")}
    records: list[dict] = []

    for season_t, season_t1 in windows:
        key = f"{season_t}->{season_t1}"
        if season_t not in by_season or season_t1 not in by_season:
            log["windows"][key] = "season missing from data"
            continue

        gt_q = by_season[season_t][by_season[season_t]["PA"] >= min_pa]
        gt1_q = by_season[season_t1][by_season[season_t1]["PA"] >= min_pa]

        merged = gt_q.merge(
            gt1_q,
            on="Name",
            suffixes=("_t", "_t1"),
            how="inner",
            validate="one_to_one",
        )
        log["windows"][key] = len(merged)

        for _, row in merged.iterrows():
            vec_t = np.array(
                [row[f"{source_col_prefix}{t}_t"] for t in tool_order], dtype=float
            )
            vec_t1 = np.array(
                [row[f"{source_col_prefix}{t}_t1"] for t in tool_order], dtype=float
            )
            delta_vec = vec_t1 - vec_t

            area_t = pentagon_area(
                pd.Series({t: row[f"{source_col_prefix}{t}_t"] for t in AREA_TOOL_ORDER}),
                AREA_TOOL_ORDER,
            )
            area_t1 = pentagon_area(
                pd.Series({t: row[f"{source_col_prefix}{t}_t1"] for t in AREA_TOOL_ORDER}),
                AREA_TOOL_ORDER,
            )

            record = {
                "record_id": f"{row['Name']}__{season_t}_{season_t1}",
                "player_id": row["Name"],
                "player_name": row["Name"],
                "season_t": season_t,
                "season_t1": season_t1,
                "pentagon_area_t": area_t,
                "pentagon_area_t1": area_t1,
                "delta_area": area_t1 - area_t,
                "war_t": row["WAR_t"],
                "wrc_t": row["wRC+_t"],
                "ops_t": row["OPS_t"],
                "delta_war": row["WAR_t1"] - row["WAR_t"],
                "delta_wrc": row["wRC+_t1"] - row["wRC+_t"],
                "delta_ops": row["OPS_t1"] - row["OPS_t"],
            }
            for t, v in zip(tool_order, vec_t):
                record[f"vec_t_{t.lower()}"] = v
            for t, v in zip(tool_order, delta_vec):
                record[f"delta_vec_{t.lower()}"] = v
            records.append(record)

    transitions = pd.DataFrame(records)
    dup = transitions[transitions.duplicated(subset=["record_id"], keep=False)]
    if not dup.empty:
        log["duplicate_record_ids"] = dup["record_id"].tolist()

    return transitions, log
