"""Phase 4: arm-agnostic scoring harness.

A Prediction has record_id, delta_war, delta_wrc, and delta_ops, plus optional
confidence, comparables, and rationale fields. The scorer is a pure function:
it validates and joins submitted predictions, then returns metric tables.
Partial evaluation works by submitting any unique subset of record IDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_PRED_COLS = {"record_id", "delta_war", "delta_wrc", "delta_ops"}
STRATUM_FLAGS = {
    "elite": "is_elite",
    "strength_up": "is_strength_up",
    "weakness_up": "is_weakness_up",
    "decliner": "is_decliner",
}
GT_COLS = [
    "record_id",
    "stratum",
    *STRATUM_FLAGS.values(),
    "season_t",
    "season_t1",
    "delta_war",
    "delta_wrc",
    "delta_ops",
]
TARGETS = ["war", "wrc", "ops"]
PredictionInput = pd.DataFrame | dict | list[dict]


def _prediction_frame(predictions: PredictionInput) -> pd.DataFrame:
    """Normalize one Prediction, a list of Predictions, or a DataFrame."""
    if isinstance(predictions, pd.DataFrame):
        return predictions.copy()
    if isinstance(predictions, dict):
        return pd.DataFrame([predictions])
    if isinstance(predictions, list):
        return pd.DataFrame(predictions)
    raise TypeError("predictions must be a Prediction dict, list[Prediction], or DataFrame")


def merge_predictions(
    predictions: PredictionInput, ground_truth: pd.DataFrame
) -> pd.DataFrame:
    """Validate and inner-join predictions to ground truth by record_id."""
    predictions = _prediction_frame(predictions)
    missing = REQUIRED_PRED_COLS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    missing_gt = set(GT_COLS) - set(ground_truth.columns)
    if missing_gt:
        raise ValueError(f"ground_truth missing required columns: {sorted(missing_gt)}")
    if predictions["record_id"].isna().any():
        raise ValueError("predictions contains null record_id")
    if predictions["record_id"].duplicated().any():
        duplicate_ids = predictions.loc[
            predictions["record_id"].duplicated(keep=False), "record_id"
        ].unique()
        raise ValueError(f"predictions contains duplicate record_id: {duplicate_ids[:5].tolist()}")
    if ground_truth["record_id"].duplicated().any():
        raise ValueError("ground_truth contains duplicate record_id")

    value_cols = ["delta_war", "delta_wrc", "delta_ops"]
    try:
        values = predictions[value_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction delta columns must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("prediction delta columns must contain only finite values")

    if "confidence" in predictions.columns:
        confidence = pd.to_numeric(predictions["confidence"], errors="coerce")
        supplied = predictions["confidence"].notna()
        if confidence[supplied].isna().any() or not confidence[supplied].between(0, 1).all():
            raise ValueError("confidence must be null or a numeric value in [0, 1]")

    merged = predictions.merge(
        ground_truth[GT_COLS], on="record_id", how="inner", suffixes=("_pred", "_true")
    )
    if merged.empty:
        raise ValueError("no overlapping record_id between predictions and ground_truth")
    return merged


def _mae_and_sign_by_group(
    merged: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    groups = merged.groupby(group_cols, sort=False) if group_cols else [((), merged)]
    for key, group in groups:
        keys = key if isinstance(key, tuple) else (key,)
        for target in TARGETS:
            true = group[f"delta_{target}_true"].to_numpy(dtype=float)
            pred = group[f"delta_{target}_pred"].to_numpy(dtype=float)
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "target": target,
                    "n": len(group),
                    "mae": float(np.mean(np.abs(true - pred))),
                    "sign_accuracy": float(np.mean(np.sign(true) == np.sign(pred))),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics_by_stratum_membership(merged: pd.DataFrame) -> pd.DataFrame:
    """Score overlapping strata independently; baseline means no special flag."""
    parts: list[pd.DataFrame] = []
    any_special = pd.Series(False, index=merged.index)
    for stratum, flag in STRATUM_FLAGS.items():
        mask = merged[flag].fillna(False).astype(bool)
        any_special |= mask
        if mask.any():
            metrics = _mae_and_sign_by_group(merged.loc[mask], [])
            metrics.insert(0, "stratum", stratum)
            parts.append(metrics)
    baseline_mask = ~any_special
    if baseline_mask.any():
        metrics = _mae_and_sign_by_group(merged.loc[baseline_mask], [])
        metrics.insert(0, "stratum", "baseline")
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _jaccard_table(
    merged: pd.DataFrame,
    reference_comparables: dict[str, list[str]],
    k: int = 10,
) -> pd.DataFrame:
    """Compute set Jaccard on each side's first k comparable IDs."""
    rows: list[dict] = []
    for _, row in merged.iterrows():
        pred_comp = row.get("comparables")
        ref_comp = reference_comparables.get(row["record_id"])
        if pred_comp is None or ref_comp is None:
            continue
        if isinstance(pred_comp, float) and np.isnan(pred_comp):
            continue
        if not isinstance(pred_comp, (list, tuple)):
            raise ValueError("comparables must be a list/tuple of IDs or null")
        if not isinstance(ref_comp, (list, tuple)):
            raise ValueError("reference comparables must be list/tuple values")
        pred_set = set(pred_comp[:k])
        ref_set = set(ref_comp[:k])
        union = pred_set | ref_set
        jaccard = len(pred_set & ref_set) / len(union) if union else float("nan")
        rows.append(
            {
                "record_id": row["record_id"],
                "stratum": row["stratum"],
                "jaccard_at_10": jaccard,
                "overlap_n": len(pred_set & ref_set),
                "pred_comparables": sorted(pred_set),
                "ref_comparables": sorted(ref_set),
            }
        )
    return pd.DataFrame(rows)


def score(
    predictions: PredictionInput,
    ground_truth: pd.DataFrame,
    reference_comparables: dict[str, list[str]] | None = None,
) -> dict:
    """Score any arm, including a partial unique subset, against ground truth."""
    predictions = _prediction_frame(predictions)
    merged = merge_predictions(predictions, ground_truth)
    known_ids = set(ground_truth["record_id"])
    result: dict = {
        "overall": _mae_and_sign_by_group(merged, []),
        "by_stratum": _metrics_by_stratum_membership(merged),
        "by_window": _mae_and_sign_by_group(merged, ["season_t", "season_t1"]),
        "n_submitted": len(predictions),
        "n_scored": len(merged),
        "n_unknown_record_ids": int((~predictions["record_id"].isin(known_ids)).sum()),
        "n_ground_truth": len(ground_truth),
    }

    if reference_comparables is not None and "comparables" in merged.columns:
        jaccard = _jaccard_table(merged, reference_comparables, k=10)
        result["jaccard"] = jaccard
        if not jaccard.empty:
            result["jaccard_mismatches"] = jaccard[
                jaccard["jaccard_at_10"] < 1.0
            ].sort_values("jaccard_at_10")

    return result
