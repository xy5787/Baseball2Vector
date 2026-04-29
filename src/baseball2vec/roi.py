"""ΔTool → ΔwRC+ regression analysis (LR + Random Forest, 10-feature)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score

TOOL_NAMES = ["Contact", "Power", "Speed", "Defense", "Discipline"]
IMPROVEMENT_STEP = 5.0  # synthetic Δ injected per tool for counterfactual ROI
MIN_PA = 200  # minimum PA for individual ROI analysis


@dataclass
class RegressionResult:
    """Holds fitted models and CV metrics for one target variable."""

    target: str
    lr_model: LinearRegression
    rf10_model: RandomForestRegressor
    lr_coefs: dict[str, float]
    lr_r2_train: float
    lr_r2_cv: float
    lr_r2_cv_std: float
    ridge_r2_cv: float
    rf5_r2_cv: float
    rf10_r2_train: float
    rf10_r2_cv: float
    rf10_r2_cv_std: float
    rf10_importance: dict[str, float]


def build_delta_dataset(
    player_df: pd.DataFrame,
    tool_names: list[str] = TOOL_NAMES,
    targets: list[str] = ("D_wRC+", "D_WAR", "D_OPS"),
) -> pd.DataFrame:
    """Build year-over-year ΔTool / Δperformance dataset.

    Only consecutive seasons (gap == 1) are included.

    Args:
        player_df: Output of the full pipeline (v3_results.csv).
        tool_names: Tool names; expects columns ``ZScore_{tool}`` in *player_df*.
        targets: Delta target column names to include.

    Returns:
        Clean DataFrame of player-season pairs with delta columns.
    """
    tool_cols = [f"ZScore_{t}" for t in tool_names]
    from_cols = [f"{t}_From" for t in tool_names]
    delta_cols = [f"D_{t}" for t in tool_names]

    records: list[dict] = []
    for name, group in player_df.sort_values(["Name", "Season"]).groupby("Name"):
        group = group.sort_values("Season")
        for i in range(len(group) - 1):
            s1, s2 = group.iloc[i], group.iloc[i + 1]
            if s2["Season"] - s1["Season"] != 1:
                continue

            record: dict = {
                "Name": name,
                "Season_From": int(s1["Season"]),
                "Season_To": int(s2["Season"]),
                "PA_From": s1["PA"],
                "PA_To": s2["PA"],
            }
            for t, col in zip(tool_names, tool_cols):
                record[f"{t}_From"] = s1[col]
                record[f"{t}_To"] = s2[col]
                record[f"D_{t}"] = s2[col] - s1[col]
            for metric in ["WAR", "wRC+", "OPS"]:
                record[f"{metric}_From"] = s1[metric]
                record[f"{metric}_To"] = s2[metric]
                record[f"D_{metric}"] = s2[metric] - s1[metric]
            record["Area_From"] = s1["ZScore_Area"]
            record["Area_To"] = s2["ZScore_Area"]
            record["D_Area"] = s2["ZScore_Area"] - s1["ZScore_Area"]
            records.append(record)

    delta_df = pd.DataFrame(records)
    available = [c for c in targets if c in delta_df.columns]
    return delta_df.dropna(subset=delta_cols + from_cols + available).reset_index(
        drop=True
    )


def run_regression(
    delta_clean: pd.DataFrame,
    tool_names: list[str] = TOOL_NAMES,
    random_state: int = 42,
) -> list[RegressionResult]:
    """Fit LR (5-feature) and RF (10-feature) for each available delta target.

    Args:
        delta_clean: Output of :func:`build_delta_dataset`.
        tool_names: Tool names in the desired order.
        random_state: Random seed for RF and KFold shuffle.

    Returns:
        List of :class:`RegressionResult` (one per target).
    """
    delta_cols = [f"D_{t}" for t in tool_names]
    from_cols = [f"{t}_From" for t in tool_names]
    feature_names_delta = [f"Δ{t}" for t in tool_names]
    feature_names_full = [f"{t}_From" for t in tool_names] + feature_names_delta

    X_delta = delta_clean[delta_cols].values
    X_full = delta_clean[from_cols + delta_cols].values

    available_targets = [
        c for c in ["D_wRC+", "D_WAR", "D_OPS"] if c in delta_clean.columns
    ]
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    results: list[RegressionResult] = []
    for tgt in available_targets:
        y = delta_clean[tgt].values
        nice = tgt.replace("D_", "Δ")

        lr = LinearRegression()
        lr.fit(X_delta, y)
        lr_cv = cross_val_score(lr, X_delta, y, cv=kf, scoring="r2")

        ridge = Ridge(alpha=1.0)
        ridge_cv = cross_val_score(ridge, X_delta, y, cv=kf, scoring="r2")

        rf5 = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=1,
        )
        rf5.fit(X_delta, y)
        rf5_cv = cross_val_score(rf5, X_delta, y, cv=kf, scoring="r2")

        rf10 = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=1,
        )
        rf10.fit(X_full, y)
        rf10_cv = cross_val_score(rf10, X_full, y, cv=kf, scoring="r2")

        results.append(
            RegressionResult(
                target=nice,
                lr_model=lr,
                rf10_model=rf10,
                lr_coefs=dict(zip(feature_names_delta, lr.coef_)),
                lr_r2_train=float(lr.score(X_delta, y)),
                lr_r2_cv=float(lr_cv.mean()),
                lr_r2_cv_std=float(lr_cv.std()),
                ridge_r2_cv=float(ridge_cv.mean()),
                rf5_r2_cv=float(rf5_cv.mean()),
                rf10_r2_train=float(rf10.score(X_full, y)),
                rf10_r2_cv=float(rf10_cv.mean()),
                rf10_r2_cv_std=float(rf10_cv.std()),
                rf10_importance=dict(
                    zip(feature_names_full, rf10.feature_importances_)
                ),
            )
        )

    return results


def compute_individual_roi(
    player_df: pd.DataFrame,
    rf10_model: RandomForestRegressor,
    tool_names: list[str] = TOOL_NAMES,
    improvement_step: float = IMPROVEMENT_STEP,
    min_pa: int = MIN_PA,
) -> pd.DataFrame:
    """Counterfactual ROI: predicted ΔwRC+ for a +5-unit improvement in each tool.

    Uses the RF 10-feature model (current level + Δ) so ROI is personalized
    to each player's current tool profile.

    Args:
        player_df: Full pipeline result DataFrame.
        rf10_model: Fitted 10-feature RF model for ΔwRC+.
        tool_names: Ordered tool names.
        improvement_step: Synthetic delta applied to each tool in turn.
        min_pa: Minimum PA threshold for inclusion.

    Returns:
        DataFrame with one row per qualifying player, sorted by WAR descending.
    """
    latest = player_df["Season"].max()
    prospects = player_df[
        (player_df["Season"] == latest) & (player_df["PA"] >= min_pa)
    ].copy()

    rows: list[dict] = []
    for _, row in prospects.iterrows():
        current = np.array([row[f"ZScore_{t}"] for t in tool_names])
        roi: dict[str, float] = {}
        for i, tool in enumerate(tool_names):
            delta = np.zeros(len(tool_names))
            delta[i] = improvement_step
            full_input = np.concatenate([current, delta]).reshape(1, -1)
            roi[tool] = float(rf10_model.predict(full_input)[0])

        best_tool = max(roi, key=roi.get)
        rows.append(
            {
                "Name": row["Name"],
                "UniqueName": row["UniqueName"],
                "WAR": row.get("WAR", np.nan),
                "wRC+": row.get("wRC+", np.nan),
                **{f"ROI_{t}": v for t, v in roi.items()},
                "Best_Tool": best_tool,
                "Best_ROI": roi[best_tool],
                **{t: row[f"ZScore_{t}"] for t in tool_names},
            }
        )

    return pd.DataFrame(rows).sort_values("WAR", ascending=False).reset_index(drop=True)
