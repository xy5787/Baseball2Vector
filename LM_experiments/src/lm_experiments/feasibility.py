"""Phase 2: empirical feasibility distributions for Δtool magnitudes.

Answers "how often does a Δtool of this size actually happen?" — the
evidence needed to explain why pushing an already-elite player (e.g. Soto)
further out via a hypothetical +1SD shift lands in a near-empty region of
observed transitions.

Age is not available in the source data (see Phase 0 report), so the only
conditioning axis is the player's t-season tool value, bucketed on the
20-80 scouting scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_bin(
    values: pd.Series, bin_edges: list[float], bin_labels: list[str]
) -> pd.Series:
    """Bucket t-season tool values into scouting-scale bins.

    Args:
        values: Tool values (20-80 scale).
        bin_edges: Increasing bin boundaries, e.g. [20, 40, 50, 60, 70, 80].
        bin_labels: len(bin_edges) - 1 labels for the resulting bins.

    Returns:
        Series of bin labels (categorical), right-open except the last bin.
    """
    # pd.cut(..., right=False) excludes the final edge. Move values exactly at
    # the scouting-scale maximum infinitesimally inward so 80 belongs to 70+.
    adjusted = values.mask(
        values == bin_edges[-1], np.nextafter(float(bin_edges[-1]), -np.inf)
    )
    return pd.cut(
        adjusted, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True
    )


def _quantile_row(deltas: np.ndarray, quantiles: list[float]) -> dict:
    row = {
        "n": int(len(deltas)),
        "mean": float(np.mean(deltas)),
        "sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else float("nan"),
    }
    for q in quantiles:
        row[f"p{int(q * 100)}"] = float(np.quantile(deltas, q))
    return row


def compute_overall_distribution(
    transitions: pd.DataFrame, tool_order: list[str], quantiles: list[float]
) -> pd.DataFrame:
    """Unconditional Δtool distribution, one row per tool.

    Args:
        transitions: Output of build_transitions (Phase 1).
        tool_order: Tool names, lowercase-matching delta_vec_{tool} columns.
        quantiles: Quantile levels to report, e.g. [0.05, 0.25, 0.5, 0.75, 0.95].

    Returns:
        DataFrame with one row per tool: n, mean, sd, p5/p25/p50/p75/p95.
    """
    rows = []
    for tool in tool_order:
        deltas = transitions[f"delta_vec_{tool.lower()}"].to_numpy()
        rows.append({"tool": tool, **_quantile_row(deltas, quantiles)})
    return pd.DataFrame(rows)


def compute_conditional_distribution(
    transitions: pd.DataFrame,
    tool_order: list[str],
    bin_edges: list[float],
    bin_labels: list[str],
    quantiles: list[float],
) -> pd.DataFrame:
    """Δtool distribution conditioned on the t-season value bucket.

    Args:
        transitions: Output of build_transitions (Phase 1).
        tool_order: Tool names.
        bin_edges: Scouting-scale bin boundaries.
        bin_labels: Bin labels matching bin_edges.
        quantiles: Quantile levels to report.

    Returns:
        DataFrame with one row per (tool, bin): n, mean, sd, p5/.../p95.
        Bins with zero observations are omitted (logged separately by caller).
    """
    rows = []
    for tool in tool_order:
        current_col = f"vec_t_{tool.lower()}"
        delta_col = f"delta_vec_{tool.lower()}"
        binned = assign_bin(transitions[current_col], bin_edges, bin_labels)
        for label in bin_labels:
            deltas = transitions.loc[binned == label, delta_col].to_numpy()
            if len(deltas) == 0:
                continue
            rows.append({"tool": tool, "current_bin": label, **_quantile_row(deltas, quantiles)})
    return pd.DataFrame(rows)


def build_lookup(
    transitions: pd.DataFrame,
    tool_order: list[str],
    bin_edges: list[float],
    bin_labels: list[str],
) -> dict:
    """Build the {tool: {bin_label: sorted Δ array}} lookup used at query time.

    Args:
        transitions: Output of build_transitions (Phase 1).
        tool_order: Tool names.
        bin_edges: Scouting-scale bin boundaries.
        bin_labels: Bin labels matching bin_edges.

    Returns:
        Dict with keys "bin_edges", "bin_labels", and "tools" (nested dict of
        tool -> bin_label -> sorted list of observed Δ values). JSON-serializable.
    """
    lookup: dict = {"bin_edges": bin_edges, "bin_labels": bin_labels, "tools": {}}
    for tool in tool_order:
        current_col = f"vec_t_{tool.lower()}"
        delta_col = f"delta_vec_{tool.lower()}"
        binned = assign_bin(transitions[current_col], bin_edges, bin_labels)
        lookup["tools"][tool] = {
            label: sorted(transitions.loc[binned == label, delta_col].tolist())
            for label in bin_labels
        }
    return lookup


def check_delta_feasibility(
    lookup: dict, tool: str, current_value: float, magnitude: float
) -> float | None:
    """Percentile rank of a Δtool magnitude within its empirical bin distribution.

    Args:
        lookup: Output of build_lookup (or the loaded feasibility_lookup.json).
        tool: Tool name, must be a key of lookup["tools"].
        current_value: t-season tool value (20-80 scale) used to pick the bin.
        magnitude: Δtool value to evaluate, e.g. +10.0 for "Power +1SD".

    Returns:
        Percentile in [0, 100]: the fraction of observed Δtool values in the
        same current-value bin that are <= magnitude. None if the bin has no
        observed transitions (empty lookup entry).
    """
    bin_edges, bin_labels = lookup["bin_edges"], lookup["bin_labels"]
    if not bin_edges[0] <= current_value <= bin_edges[-1]:
        raise ValueError(
            f"current_value must be in [{bin_edges[0]}, {bin_edges[-1]}]"
        )
    idx = np.digitize([current_value], bin_edges[1:-1], right=False)[0]
    idx = min(idx, len(bin_labels) - 1)
    label = bin_labels[idx]

    deltas = lookup["tools"].get(tool, {}).get(label, [])
    if not deltas:
        return None
    return float(np.searchsorted(deltas, magnitude, side="right") / len(deltas) * 100)
