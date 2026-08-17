"""Phase 4: stratify transitions into evaluation cohorts.

Random sampling over the full transition set is mostly "tool rose -> value
rose" easy cases where every arm scores similarly. These strata isolate the
cases that actually discriminate between arms.

Assignment priority (highest wins when a record matches more than one):
decliner > elite > strength_up > weakness_up > baseline. is_* boolean
columns are also kept so overlap between strata is never silently lost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STRATA_ORDER = ["decliner", "elite", "strength_up", "weakness_up", "baseline"]


def assign_strata(
    transitions: pd.DataFrame,
    tool_order: list[str],
    elite_percentile: float = 0.90,
    strength_threshold: float = 60.0,
    weakness_threshold: float = 45.0,
) -> pd.DataFrame:
    """Add is_elite / is_strength_up / is_weakness_up / is_decliner / stratum columns.

    Args:
        transitions: Output of build_transitions (Phase 1).
        tool_order: Tool names.
        elite_percentile: Pentagon-area quantile (global, across all windows)
            above which a t-season is "elite".
        strength_threshold: Tool value at/above which a tool counts as a
            "strength" for strength_up.
        weakness_threshold: Tool value at/below which a tool counts as a
            "weakness" for weakness_up.

    Returns:
        Copy of transitions with the new columns appended.
    """
    df = transitions.copy()
    vec_t_cols = [f"vec_t_{t.lower()}" for t in tool_order]
    delta_cols = [f"delta_vec_{t.lower()}" for t in tool_order]

    area_cutoff = df["pentagon_area_t"].quantile(elite_percentile)
    df["is_elite"] = df["pentagon_area_t"] >= area_cutoff

    strength_up = pd.Series(False, index=df.index)
    weakness_up = pd.Series(False, index=df.index)
    for vc, dc in zip(vec_t_cols, delta_cols):
        strength_up |= (df[vc] >= strength_threshold) & (df[dc] > 0)
        weakness_up |= (df[vc] <= weakness_threshold) & (df[dc] > 0)
    df["is_strength_up"] = strength_up
    df["is_weakness_up"] = weakness_up

    vec_sum_delta = df[delta_cols].sum(axis=1)
    df["is_decliner"] = (vec_sum_delta > 0) & (df["delta_war"] < 0)

    stratum = np.full(len(df), "baseline", dtype=object)
    stratum[df["is_weakness_up"].to_numpy()] = "weakness_up"
    stratum[df["is_strength_up"].to_numpy()] = "strength_up"
    stratum[df["is_elite"].to_numpy()] = "elite"
    stratum[df["is_decliner"].to_numpy()] = "decliner"
    df["stratum"] = stratum

    return df


def stratum_overlap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each is_* flag pair co-occurs (diagnostic only).

    Args:
        df: Output of assign_strata.

    Returns:
        Square DataFrame of pairwise co-occurrence counts.
    """
    flags = ["is_elite", "is_strength_up", "is_weakness_up", "is_decliner"]
    mat = pd.DataFrame(index=flags, columns=flags, dtype=int)
    for a in flags:
        for b in flags:
            mat.loc[a, b] = int((df[a] & df[b]).sum())
    return mat
