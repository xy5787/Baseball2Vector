"""Shared utilities for the isolated academic-validation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXP_ROOT.parent
CONFIG_PATH = EXP_ROOT / "config" / "experiment_config.json"

TOOL_NAMES = ["Contact", "Power", "Speed", "Defense", "Discipline"]
TOOL_COLS = [f"ZScore_{name}" for name in TOOL_NAMES]


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def ensure_dirs() -> None:
    for name in [
        "data_intermediate",
        "results",
        "tables",
        "figures",
        "logs",
        "manuscript_preview",
    ]:
        (EXP_ROOT / name).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", value)
    return re.sub(r"[^a-z0-9]", "", value)


def metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "r2":
        denom = float(np.sum((y_true - y_true.mean()) ** 2))
        return float("nan") if denom == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denom)
    if metric == "spearman":
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            return float("nan")
        return float(spearmanr(y_true, y_pred).statistic)
    raise ValueError(metric)


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float("nan"), float("nan")
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_bootstrap_indices(
    groups: pd.Series | np.ndarray,
    n_boot: int,
    seed: int,
) -> list[np.ndarray]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(n_boot):
        selected = rng.choice(unique, size=len(unique), replace=True)
        samples.append(np.concatenate([by_group[g] for g in selected]))
    return samples


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return float(np.interp(q, cumulative, values))


def bootstrap_cluster_stat(
    df: pd.DataFrame,
    group_col: str,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int,
    seed: int,
) -> np.ndarray:
    groups = df[group_col].dropna().unique()
    rng = np.random.default_rng(seed)
    grouped = {g: x for g, x in df.groupby(group_col, sort=False)}
    values = []
    for _ in range(n_boot):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([grouped[g] for g in selected], ignore_index=True)
        values.append(stat_fn(sample))
    return np.asarray(values, dtype=float)


def pa_stratum(pa: float) -> str:
    if pa < 250:
        return "100-249"
    if pa < 500:
        return "250-499"
    return "500+"


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Build consecutive-season transitions using stable player IDs only."""
    eligible = df[df["id_status"].eq("matched")].copy()
    assert eligible["player_id"].notna().all()
    assert not eligible.duplicated(["player_id", "Season"]).any()
    rows: list[dict] = []
    for player_id, group in eligible.groupby("player_id", sort=False):
        group = group.sort_values("Season")
        records = list(group.to_dict("records"))
        for left, right in zip(records, records[1:]):
            if int(right["Season"]) != int(left["Season"]) + 1:
                continue
            row = {
                "player_id": player_id,
                "Name": left["Name"],
                "Season_t": int(left["Season"]),
                "Season_t1": int(right["Season"]),
            }
            for col in df.columns:
                if col in {"player_id", "Name", "Season"}:
                    continue
                row[f"{col}_t"] = left[col]
                row[f"{col}_t1"] = right[col]
            rows.append(row)
    out = pd.DataFrame(rows)
    assert not out.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert (out["Season_t1"] == out["Season_t"] + 1).all()
    return out


def standardize_vectors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = df[TOOL_COLS].mean()
    stds = df[TOOL_COLS].std(ddof=1).replace(0, np.nan)
    return (df[TOOL_COLS] - means) / stds, means, stds


def vector_distance(a: np.ndarray, b: np.ndarray, stds: np.ndarray) -> float:
    return float(np.linalg.norm((np.asarray(a) - np.asarray(b)) / stds) / math.sqrt(len(stds)))

