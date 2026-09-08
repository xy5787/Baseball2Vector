"""Shared deterministic utilities for the final academic revision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RUN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RUN_ROOT.parents[1]
DATA_PATH = RUN_ROOT / "data_intermediate" / "player_seasons_with_ids.csv"
SEED = 42
N_BOOT = 2000
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TOOL_NAMES = ["Contact", "Power", "Plate Discipline", "Defense", "Speed"]
TOOL_COLS = [
    "ZScore_Contact",
    "ZScore_Power",
    "ZScore_Discipline",
    "ZScore_Defense",
    "ZScore_Speed",
]
SAFE_TOOL_COLS = [f"safe_{c}" for c in TOOL_COLS]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_player_seasons() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    matched = df[df["id_status"].eq("matched")].copy()
    assert matched["player_id"].notna().all()
    assert not matched.duplicated(["player_id", "Season"]).any()
    # Forecast-safe transform uses only the input season's contemporaneous cohort.
    for source, target in zip(TOOL_COLS, SAFE_TOOL_COLS):
        matched[target] = matched.groupby("Season")[source].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )
    return matched.sort_values(["player_id", "Season", "source_row"], kind="mergesort")


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keep = [
        "Name", "Season", "PA", "WAR", "wRC+", "OPS", "age_t", "birth_year",
        "source_row", *TOOL_COLS, *SAFE_TOOL_COLS,
    ]
    for player_id, player in df.groupby("player_id", sort=True):
        player = player.sort_values(["Season", "source_row"], kind="mergesort")
        records = player[keep].to_dict("records")
        for left, right in zip(records, records[1:]):
            if int(right["Season"]) != int(left["Season"]) + 1:
                continue
            row = {"player_id": player_id, "Name": left["Name"]}
            for key, value in left.items():
                row[f"{key}_t"] = value
            for key, value in right.items():
                row[f"{key}_t1"] = value
            rows.append(row)
    out = pd.DataFrame(rows)
    assert not out.duplicated(["player_id", "Season_t", "Season_t1"]).any()
    assert (out["Season_t1"] == out["Season_t"] + 1).all()
    assert (out["birth_year_t"] == out["birth_year_t1"]).all()
    return out.sort_values(["Season_t", "player_id"], kind="mergesort").reset_index(drop=True)


def ridge_pipeline(features: list[str], alpha: float) -> Pipeline:
    preprocess = ColumnTransformer(
        [("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), features)],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("ridge", Ridge(alpha=alpha))])


def select_alpha(
    train: pd.DataFrame, features: list[str], target: str, groups: str = "player_id"
) -> tuple[float, pd.DataFrame]:
    splitter = GroupKFold(n_splits=5)
    tuning_rows: list[dict] = []
    for alpha in ALPHAS:
        fold_mae: list[float] = []
        for fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(train, train[target], train[groups])
        ):
            model = ridge_pipeline(features, alpha)
            model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx][target])
            pred = model.predict(train.iloc[valid_idx][features])
            fold_mae.append(mean_absolute_error(train.iloc[valid_idx][target], pred))
            tuning_rows.append(
                {"alpha": alpha, "fold": fold, "mae": fold_mae[-1]}
            )
        tuning_rows.append(
            {"alpha": alpha, "fold": "mean", "mae": float(np.mean(fold_mae))}
        )
    tuning = pd.DataFrame(tuning_rows)
    means = tuning[tuning["fold"].eq("mean")].copy()
    # Deterministic: smallest alpha wins an exact tie.
    best = float(means.sort_values(["mae", "alpha"], kind="mergesort").iloc[0]["alpha"])
    tuning["selected"] = tuning["alpha"].eq(best) & tuning["fold"].eq("mean")
    return best, tuning


def metric(y: np.ndarray, pred: np.ndarray, name: str) -> float:
    if name == "mae":
        return float(mean_absolute_error(y, pred))
    if name == "r2":
        return float(r2_score(y, pred))
    if name == "spearman":
        return float(spearmanr(y, pred).statistic)
    if name == "pearson":
        return float(pearsonr(y, pred).statistic)
    raise ValueError(name)


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan
    return tuple(np.quantile(clean, [0.025, 0.975]))


def unique_cluster_bootstrap(ids: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    unique = np.unique(ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.choice(unique, size=len(unique), replace=True)


def expanded_cluster_indices(ids: np.ndarray, sampled_ids: np.ndarray) -> np.ndarray:
    parts = [np.flatnonzero(ids == player_id) for player_id in sampled_ids]
    return np.concatenate(parts)


def dump_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
