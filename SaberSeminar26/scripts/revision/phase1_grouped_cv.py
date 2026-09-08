"""Phase 1: Player-grouped, leakage-resistant CV for the ΔTool -> ΔWAR/ΔwRC+/ΔOPS regression.

Replaces the pooled, shuffled 5-fold CV behind the paper's R^2=0.71 claim (Sec. 4.3) with
GroupKFold keyed on player identity, so no player's transitions appear in both train and test
within a fold. Also refits the 20-80 min-max scaling within each training fold (see the
"Note on preprocessing refitting" docstring below for why this is the only preprocessing step
that needs refitting, and how it is done without raw FanGraphs data). Adds a forward-time
holdout (train on transitions ending <=2023, test on 2024->2025) as a secondary check.

Note on preprocessing refitting
--------------------------------
Phase 0 (outputs/revision/phase0_data_audit/audit_report.md) established that:
  - Per-season Z-scoring and Bayesian stabilization are already computed strictly *within
    season* by the existing pipeline (src/baseball2vec/tools.py, data.py) -- season assignment
    doesn't depend on the CV fold, so there is nothing to refit there.
  - The only population-pooled step that feeds the ROI regression is the 20-80 min-max scaling
    (`scale_to_2080`), which is a per-tool AFFINE map v_scaled = a*z + b fit once on all 2,309
    season-rows.

Raw FanGraphs batting stats are not available in this environment (data/raw/ is empty; FanGraphs
blocks scraping from server IPs -- see README's "Data availability" section), so the pre-scaling
per-tool Z-score averages cannot be recomputed from scratch. However, re-applying a fresh
MinMaxScaler(20, 80) to the *already pooled-scaled* ZScore_{Tool} column is itself a valid affine
function of the true underlying (unobserved) per-tool Z-score average, because a MinMaxScaler
applied to an affine function of z is still an affine function of z. So fitting a fresh
MinMaxScaler on ZScore_{Tool} restricted to a fold's training-fold player-seasons, and applying
it (transform only, not fit) to the held-out fold's player-seasons, exactly reproduces
"refit the 20-80 bounds on the training fold only" without needing the raw stats.

A second, independent point (verified empirically below, not just asserted): linear regression
R^2 is invariant to any per-feature affine rescaling of X, and random-forest predictions are
invariant to any per-feature monotonic rescaling of X (tree splits/leaf means depend only on
rank order and the unchanged targets y). So whether the min-max bounds are fit on the full
pooled population or on the training fold only should make ~zero difference to LR/RF R^2 in
this specific case -- this is reported as an explicit finding below, not assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baseball2vec.roi import TOOL_NAMES, build_delta_dataset  # noqa: E402

SEED = 42
DATA = ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase1_grouped_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELTA_COLS = [f"D_{t}" for t in TOOL_NAMES]
FROM_COLS = [f"{t}_From" for t in TOOL_NAMES]
TARGETS = ["D_wRC+", "D_WAR", "D_OPS"]


def refit_2080_on_train(
    player_df: pd.DataFrame, train_names: set[str]
) -> dict[str, MinMaxScaler]:
    """Fit a fresh MinMaxScaler(20, 80) per tool on training-fold players only."""
    train_rows = player_df[player_df["Name"].isin(train_names)]
    scalers: dict[str, MinMaxScaler] = {}
    for tool in TOOL_NAMES:
        col = f"ZScore_{tool}"
        scaler = MinMaxScaler(feature_range=(20, 80))
        scaler.fit(train_rows[[col]])
        scalers[tool] = scaler
    return scalers


def rescale_delta_rows(
    delta_df: pd.DataFrame, scalers: dict[str, MinMaxScaler]
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute Tool_From and D_Tool under a fold-specific 20-80 scaler.

    Because the scaler is affine (v_new = a*v_old + b), D_Tool_new = a * D_Tool_old exactly
    (the intercept cancels), while Tool_From_new = a*Tool_From_old + b.
    """
    from_new = np.zeros((len(delta_df), len(TOOL_NAMES)))
    delta_new = np.zeros((len(delta_df), len(TOOL_NAMES)))
    for j, tool in enumerate(TOOL_NAMES):
        scaler = scalers[tool]
        a = float(scaler.scale_[0])
        b = float(scaler.min_[0])
        tool_from_old = delta_df[f"{tool}_From"].values
        tool_to_old = delta_df[f"{tool}_To"].values
        from_new[:, j] = tool_from_old * a + b
        delta_new[:, j] = (tool_to_old - tool_from_old) * a
    return from_new, delta_new


def pooled_kfold_r2(delta_clean: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Original-style pooled, shuffled 5-fold CV (no player grouping), on the audited data."""
    X_delta = delta_clean[DELTA_COLS].values
    X_full = delta_clean[FROM_COLS + DELTA_COLS].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    out: dict[str, dict[str, float]] = {}
    for tgt in TARGETS:
        y = delta_clean[tgt].values
        lr = LinearRegression()
        lr_cv = cross_val_score(lr, X_delta, y, cv=kf, scoring="r2")
        rf10 = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=SEED, n_jobs=1
        )
        rf10_cv = cross_val_score(rf10, X_full, y, cv=kf, scoring="r2")
        out[tgt] = {"lr_r2": float(lr_cv.mean()), "rf10_r2": float(rf10_cv.mean())}
    return out


def grouped_kfold_r2(
    delta_clean: pd.DataFrame, player_df: pd.DataFrame, refit_scaling: bool
) -> dict[str, dict[str, float]]:
    """Player-grouped 5-fold CV, optionally refitting the 20-80 scaler per training fold."""
    groups = delta_clean["Name"].values
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(gkf.split(delta_clean, groups=groups))

    lr_scores = {tgt: [] for tgt in TARGETS}
    rf_scores = {tgt: [] for tgt in TARGETS}

    for train_idx, test_idx in splits:
        train_names = set(delta_clean.iloc[train_idx]["Name"])
        test_names = set(delta_clean.iloc[test_idx]["Name"])
        assert train_names.isdisjoint(test_names), "GroupKFold leaked a player across folds"

        if refit_scaling:
            scalers = refit_2080_on_train(player_df, train_names)
            from_all, delta_all = rescale_delta_rows(delta_clean, scalers)
        else:
            from_all = delta_clean[FROM_COLS].values
            delta_all = delta_clean[DELTA_COLS].values

        X_delta_train, X_delta_test = delta_all[train_idx], delta_all[test_idx]
        X_full_train = np.hstack([from_all[train_idx], delta_all[train_idx]])
        X_full_test = np.hstack([from_all[test_idx], delta_all[test_idx]])

        for tgt in TARGETS:
            y = delta_clean[tgt].values
            y_train, y_test = y[train_idx], y[test_idx]

            lr = LinearRegression().fit(X_delta_train, y_train)
            lr_r2 = lr.score(X_delta_test, y_test)
            lr_scores[tgt].append(lr_r2)

            rf10 = RandomForestRegressor(
                n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=SEED, n_jobs=1
            )
            rf10.fit(X_full_train, y_train)
            rf_r2 = rf10.score(X_full_test, y_test)
            rf_scores[tgt].append(rf_r2)

    return {
        tgt: {
            "lr_r2": float(np.mean(lr_scores[tgt])),
            "rf10_r2": float(np.mean(rf_scores[tgt])),
        }
        for tgt in TARGETS
    }


def forward_time_holdout(
    delta_clean: pd.DataFrame, player_df: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Train on transitions ending <=2023, test on 2024->2025 transitions.

    The 20-80 scaler is fit on season-rows from seasons <=2023 only, then applied to 2024/2025
    season-rows -- a genuine prospective (not just player-disjoint) holdout.
    """
    train_mask = delta_clean["Season_To"] <= 2023
    test_mask = (delta_clean["Season_From"] == 2024) & (delta_clean["Season_To"] == 2025)
    train_df = delta_clean[train_mask]
    test_df = delta_clean[test_mask]

    train_season_rows = player_df[player_df["Season"] <= 2023]
    scalers: dict[str, MinMaxScaler] = {}
    for tool in TOOL_NAMES:
        col = f"ZScore_{tool}"
        scaler = MinMaxScaler(feature_range=(20, 80))
        scaler.fit(train_season_rows[[col]])
        scalers[tool] = scaler

    from_train, delta_train = rescale_delta_rows(train_df, scalers)
    from_test, delta_test = rescale_delta_rows(test_df, scalers)

    out: dict[str, dict[str, float]] = {}
    for tgt in TARGETS:
        y_train = train_df[tgt].values
        y_test = test_df[tgt].values

        lr = LinearRegression().fit(delta_train, y_train)
        lr_r2 = lr.score(delta_test, y_test)

        X_full_train = np.hstack([from_train, delta_train])
        X_full_test = np.hstack([from_test, delta_test])
        rf10 = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=8, random_state=SEED, n_jobs=1
        )
        rf10.fit(X_full_train, y_train)
        rf_r2 = rf10.score(X_full_test, y_test)

        out[tgt] = {"lr_r2": float(lr_r2), "rf10_r2": float(rf_r2), "n_train": len(train_df), "n_test": len(test_df)}
    return out


def main() -> None:
    player_df = pd.read_csv(DATA)
    delta_clean = build_delta_dataset(player_df)
    print(f"Transitions: {len(delta_clean)}, unique players: {delta_clean['Name'].nunique()}")

    pooled = pooled_kfold_r2(delta_clean)
    grouped_no_refit = grouped_kfold_r2(delta_clean, player_df, refit_scaling=False)
    grouped_refit = grouped_kfold_r2(delta_clean, player_df, refit_scaling=True)
    forward = forward_time_holdout(delta_clean, player_df)

    rows = []
    for tgt in TARGETS:
        rows.append(
            {
                "target": tgt.replace("D_", "Δ"),
                "pooled_kfold_lr_r2": pooled[tgt]["lr_r2"],
                "pooled_kfold_rf10_r2": pooled[tgt]["rf10_r2"],
                "grouped_kfold_lr_r2": grouped_no_refit[tgt]["lr_r2"],
                "grouped_kfold_rf10_r2": grouped_no_refit[tgt]["rf10_r2"],
                "grouped_kfold_refit_scaling_lr_r2": grouped_refit[tgt]["lr_r2"],
                "grouped_kfold_refit_scaling_rf10_r2": grouped_refit[tgt]["rf10_r2"],
                "forward_holdout_lr_r2": forward[tgt]["lr_r2"],
                "forward_holdout_rf10_r2": forward[tgt]["rf10_r2"],
                "forward_holdout_n_train": forward[tgt]["n_train"],
                "forward_holdout_n_test": forward[tgt]["n_test"],
            }
        )
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "results_table.csv", index=False)
    print(results_df.to_string(index=False))

    # Comparison plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    x = np.arange(len(TARGETS))
    width = 0.2
    for ax, model_key, title in [
        (axes[0], "lr_r2", "Linear Regression (5-feature Δ)"),
        (axes[1], "rf10_r2", "Random Forest (10-feature)"),
    ]:
        pooled_vals = [pooled[t][model_key] for t in TARGETS]
        grouped_vals = [grouped_no_refit[t][model_key] for t in TARGETS]
        refit_vals = [grouped_refit[t][model_key] for t in TARGETS]
        forward_vals = [forward[t][model_key] for t in TARGETS]
        ax.bar(x - 1.5 * width, pooled_vals, width, label="Pooled KFold (original)")
        ax.bar(x - 0.5 * width, grouped_vals, width, label="Player-grouped")
        ax.bar(x + 0.5 * width, refit_vals, width, label="Player-grouped + refit scaling")
        ax.bar(x + 1.5 * width, forward_vals, width, label="Forward-time holdout (≤2023→24-25)")
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("D_", "Δ") for t in TARGETS])
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("R²" if ax is axes[0] else "")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Phase 1: CV scheme comparison for ΔTool regression")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_plot.png", dpi=150)
    print(f"\nSaved: {OUT_DIR / 'comparison_plot.png'}")
    print(f"Saved: {OUT_DIR / 'results_table.csv'}")


if __name__ == "__main__":
    main()
