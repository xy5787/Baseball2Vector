"""Phase 3, Arm A: statistical baselines (linear regression + KNN).

Both models are evaluated under a time split — train on 21->22, 22->23,
23->24; test on 24->25, so future transition labels never enter training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

TARGET_COLS = {"WAR": "delta_war", "wRC+": "delta_wrc", "OPS": "delta_ops"}
LEVEL_T_COLS = {"WAR": "war_t", "wRC+": "wrc_t", "OPS": "ops_t"}


def time_split(
    transitions: pd.DataFrame, test_window: tuple[int, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season window to enforce chronological evaluation.

    Args:
        transitions: Output of build_transitions (Phase 1).
        test_window: (season_t, season_t1) held out as the test set; all other
            windows are used for training.

    Returns:
        (train_df, test_df).
    """
    test_mask = (transitions["season_t"] == test_window[0]) & (
        transitions["season_t1"] == test_window[1]
    )
    return transitions[~test_mask].copy(), transitions[test_mask].copy()


def build_feature_matrix(
    df: pd.DataFrame, tool_order: list[str], extended: bool
) -> tuple[np.ndarray, list[str]]:
    """Build the LR feature matrix.

    Basic model: ΔTool_1..5 (5 features).
    Extended model: ΔTool_1..5 + Tool_t_1..5 + (ΔTool_i * Tool_t_i)_1..5 (15
    features) — lets the marginal return on a Δ vary with the player's
    starting level in that tool.

    Args:
        df: transitions rows (or a subset).
        tool_order: Tool names.
        extended: Whether to include base-level and interaction terms.

    Returns:
        (X, feature_names).
    """
    delta_cols = [f"delta_vec_{t.lower()}" for t in tool_order]
    delta_names = [f"Δ{t}" for t in tool_order]
    X = df[delta_cols].to_numpy(dtype=float)
    names = list(delta_names)

    if extended:
        base_cols = [f"vec_t_{t.lower()}" for t in tool_order]
        base = df[base_cols].to_numpy(dtype=float)
        interaction = df[delta_cols].to_numpy(dtype=float) * base
        X = np.hstack([X, base, interaction])
        names += [f"{t}_t" for t in tool_order] + [f"Δ{t}×{t}_t" for t in tool_order]

    return X, names


def _ols_with_ci(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], alpha: float = 0.05
) -> tuple[LinearRegression, pd.DataFrame, float]:
    """Fit OLS via sklearn for predictions, and manually for coefficient CIs.

    statsmodels is not a project dependency, so the classic OLS covariance
    formula (sigma^2 * (X'X)^-1) is used directly instead.

    Args:
        X: Feature matrix (no intercept column).
        y: Target vector.
        feature_names: Names matching X's columns.
        alpha: CI level (0.05 -> 95% CI).

    Returns:
        (fitted sklearn model, coefficient table, R^2 on training data).
    """
    model = LinearRegression()
    model.fit(X, y)
    r2 = float(model.score(X, y))

    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta = np.concatenate([[model.intercept_], model.coef_])
    resid = y - Xd @ beta
    dof = n - Xd.shape[1]
    sigma2 = np.sum(resid**2) / dof
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    tval = stats.t.ppf(1 - alpha / 2, dof)

    names = ["intercept"] + feature_names
    coef_table = pd.DataFrame(
        {
            "feature": names,
            "coef": beta,
            "se": se,
            "ci_lo": beta - tval * se,
            "ci_hi": beta + tval * se,
        }
    )
    return model, coef_table, r2


@dataclass
class LRFit:
    """A fitted linear regression, its coefficient table, and a predict fn."""

    target: str
    model_type: str  # "basic" | "extended"
    feature_names: list[str]
    coef_table: pd.DataFrame
    r2_train: float
    predict: Callable[[pd.DataFrame], np.ndarray]


def fit_lr(
    train_df: pd.DataFrame,
    tool_order: list[str],
    target: str,
    extended: bool,
    alpha: float = 0.05,
) -> LRFit:
    """Fit one LR model (basic or extended) for one target metric.

    Args:
        train_df: Training-split transitions.
        tool_order: Tool names.
        target: One of "WAR", "wRC+", "OPS".
        extended: Whether to fit the extended (base-level + interaction) model.
        alpha: Two-sided coefficient CI alpha (0.05 gives 95% intervals).

    Returns:
        LRFit with a predict(df) closure over the fitted feature builder.
    """
    y_col = TARGET_COLS[target]
    X, names = build_feature_matrix(train_df, tool_order, extended)
    y = train_df[y_col].to_numpy(dtype=float)
    model, coef_table, r2 = _ols_with_ci(X, y, names, alpha=alpha)

    def predict(df: pd.DataFrame) -> np.ndarray:
        Xq, _ = build_feature_matrix(df, tool_order, extended)
        return model.predict(Xq)

    return LRFit(
        target=target,
        model_type="extended" if extended else "basic",
        feature_names=names,
        coef_table=coef_table,
        r2_train=r2,
        predict=predict,
    )


def build_knn_pool(
    v3_results: pd.DataFrame,
    train_seasons: list[int],
    tool_order: list[str],
    min_pa: int,
    source_col_prefix: str = "ZScore_",
) -> pd.DataFrame:
    """Build the KNN candidate pool: single-season player vectors from train seasons only.

    Args:
        v3_results: Raw (deduplicated) v3_results.csv rows.
        train_seasons: Seasons available at train time (e.g. [2021..2024] for
            a 24->25 test split) — excludes the test season to avoid leakage.
        tool_order: Tool names.
        min_pa: Minimum PA filter, matching the transitions dataset.
        source_col_prefix: Column prefix of the tool scores.

    Returns:
        Filtered DataFrame with one row per qualifying player-season.
    """
    pool = v3_results[
        v3_results["Season"].isin(train_seasons) & (v3_results["PA"] >= min_pa)
    ].copy()
    cols = [f"{source_col_prefix}{t}" for t in tool_order]
    return pool.dropna(subset=cols + ["WAR", "wRC+", "OPS"]).reset_index(drop=True)


def knn_predict(
    test_df: pd.DataFrame,
    pool: pd.DataFrame,
    tool_order: list[str],
    target: str,
    k: int = 10,
    source_col_prefix: str = "ZScore_",
) -> np.ndarray:
    """Reproduce the original KNN methodology inside the delta-prediction benchmark.

    Original approach: shift a player's vector by a tool delta, find the k
    nearest real players to the *resulting* vector, and read off their WAR
    level as the implied outcome. Here the resulting vector is simply
    vec_t + delta_vec (the observed vec_t1, since delta_vec is a known input
    in this benchmark) — so this is an exact translation, not a variant.

    Args:
        test_df: Test-split transitions (has vec_t_*, delta_vec_*, and the
            known t-season level via war_t/wrc_t/ops_t).
        pool: Output of build_knn_pool — train-season candidate snapshots.
        tool_order: Tool names.
        target: One of "WAR", "wRC+", "OPS".
        k: Number of neighbors.
        source_col_prefix: Column prefix of pool's tool score columns.

    Returns:
        Array of predicted Δtarget values, one per test_df row.
    """
    pool_cols = [f"{source_col_prefix}{t}" for t in tool_order]
    pool_X = pool[pool_cols].to_numpy(dtype=float)
    pool_y = pool[target].to_numpy(dtype=float)

    vec_t = test_df[[f"vec_t_{t.lower()}" for t in tool_order]].to_numpy(dtype=float)
    delta_vec = test_df[[f"delta_vec_{t.lower()}" for t in tool_order]].to_numpy(
        dtype=float
    )
    query = vec_t + delta_vec  # == observed vec_t1

    nn = NearestNeighbors(n_neighbors=k).fit(pool_X)
    _, idx = nn.kneighbors(query)
    predicted_level_t1 = pool_y[idx].mean(axis=1)

    level_t = test_df[LEVEL_T_COLS[target]].to_numpy(dtype=float)
    return predicted_level_t1 - level_t


def knn_comparables(
    test_df: pd.DataFrame,
    pool: pd.DataFrame,
    tool_order: list[str],
    k: int = 10,
    source_col_prefix: str = "ZScore_",
    id_col: str = "UniqueName",
) -> dict[str, list[str]]:
    """The k=10 neighbor ids behind knn_predict — used as the jaccard@10 reference set.

    Args:
        test_df: Test-split transitions.
        pool: Output of build_knn_pool.
        tool_order: Tool names.
        k: Number of neighbors.
        source_col_prefix: Column prefix of pool's tool score columns.
        id_col: Pool column identifying each candidate player-season.

    Returns:
        Dict mapping record_id -> list of k neighbor ids (same neighbors
        knn_predict averaged over).
    """
    pool_cols = [f"{source_col_prefix}{t}" for t in tool_order]
    pool_X = pool[pool_cols].to_numpy(dtype=float)
    ids = pool[id_col].to_numpy()

    vec_t = test_df[[f"vec_t_{t.lower()}" for t in tool_order]].to_numpy(dtype=float)
    delta_vec = test_df[[f"delta_vec_{t.lower()}" for t in tool_order]].to_numpy(
        dtype=float
    )
    query = vec_t + delta_vec

    nn = NearestNeighbors(n_neighbors=k).fit(pool_X)
    _, idx = nn.kneighbors(query)
    return {
        rid: ids[row].tolist() for rid, row in zip(test_df["record_id"], idx)
    }
