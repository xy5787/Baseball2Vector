"""The corrected 20-80 scaling (mean=50, SD=10), copied from ScalingRevision/scaling.py
so this folder is self-contained. See ../ScalingRevision/outputs/report_ko.md for the
full derivation and validation against MinMax."""

from __future__ import annotations

import pandas as pd

TARGET_MEAN = 50.0
TARGET_SD = 10.0


def scale_to_2080_sd(
    scores_df: pd.DataFrame,
    cols: list[str],
    target_mean: float = TARGET_MEAN,
    target_sd: float = TARGET_SD,
) -> pd.DataFrame:
    out = {}
    for c in cols:
        x = scores_df[c].astype(float)
        z = (x - x.mean()) / (x.std() + 1e-8)
        grade = target_mean + target_sd * z
        out[c] = grade.clip(20, 80)
    return pd.DataFrame(out, index=scores_df.index)


def build_scaled_results_sd(
    methods: dict[str, pd.DataFrame],
    tool_names: list[str],
    target_mean: float = TARGET_MEAN,
    target_sd: float = TARGET_SD,
) -> dict[str, pd.DataFrame]:
    from baseball2vec.tools import pentagon_area  # requires src/ on sys.path

    results: dict[str, pd.DataFrame] = {}
    for name, raw in methods.items():
        scaled = scale_to_2080_sd(raw, tool_names, target_mean, target_sd)
        out = scaled.copy()
        out.columns = [f"{name}_{c}" for c in tool_names]
        out[f"{name}_Area"] = scaled.apply(pentagon_area, axes=tool_names, axis=1)
        results[name] = out
    return results
