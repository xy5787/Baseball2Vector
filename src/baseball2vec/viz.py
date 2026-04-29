"""All plotting functions for Baseball2Vec."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"


def _save(fig: plt.Figure, filename: str, figures_dir: Path = FIGURES_DIR) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_correlation_heatmap(
    corr_table: dict[str, dict[str, float]],
    method_names: list[str],
    validation_metrics: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Pentagon-Area vs performance heatmap (rows = methods, cols = metrics).

    Args:
        corr_table: Nested dict from :func:`baselines.correlation_table`.
        method_names: Row order for the heatmap.
        validation_metrics: Column order.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    matrix = pd.DataFrame(
        {
            m: {meth: corr_table[meth][m] for meth in method_names}
            for m in validation_metrics
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        vmin=0.4,
        vmax=0.85,
        linewidths=1,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(
        "Pentagon Area vs Performance (Z-Score / PCA / Joint VAE)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    ax.set_ylabel("Method")
    plt.tight_layout()
    return _save(fig, "v2_correlation_heatmap.png", figures_dir)


def plot_scatter_grid(
    result_df: pd.DataFrame,
    method_names: list[str],
    validation_metrics: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """3×3 scatter grid of Pentagon Area vs performance metrics.

    Args:
        result_df: Combined result DataFrame with ``{method}_Area`` columns.
        method_names: Methods for rows (length 3).
        validation_metrics: Metrics for columns (length 3).
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    fig, axes_grid = plt.subplots(3, 3, figsize=(18, 16))
    for i, method in enumerate(method_names):
        area_col = f"{method}_Area"
        for j, metric in enumerate(validation_metrics):
            ax = axes_grid[i, j]
            ax.scatter(
                result_df[area_col],
                result_df[metric],
                c=result_df["Season"],
                cmap="viridis",
                alpha=0.4,
                s=20,
                edgecolors="none",
            )
            x, y = result_df[area_col], result_df[metric]
            mask = x.notna() & y.notna()
            if mask.sum() > 2:
                z = np.polyfit(x[mask], y[mask], 1)
                xl = np.linspace(x[mask].min(), x[mask].max(), 100)
                ax.plot(xl, np.poly1d(z)(xl), "r-", lw=1.5, alpha=0.8)
            r = x[mask].corr(y[mask])
            ax.set_title(
                f"{method} vs {metric} (r={r:.4f})", fontsize=11, weight="bold"
            )
            if i == 2:
                ax.set_xlabel(f"{method} Area")
            if j == 0:
                ax.set_ylabel(metric)
            ax.grid(True, alpha=0.2)
    plt.suptitle(
        "Baseline Comparison: 5-Tool Pentagon Area", fontsize=16, weight="bold", y=1.01
    )
    plt.tight_layout()
    return _save(fig, "v3_scatter_grid.png", figures_dir)


def plot_covariance_heatmap(
    mean_corr: np.ndarray,
    tool_names: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """League-average tool tradeoff matrix from the VAE covariance Σ.

    Args:
        mean_corr: [5, 5] correlation matrix derived from Σ = LL^T.
        tool_names: Tool labels for axes.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    annot = np.array([[f"{v:.3f}" for v in row] for row in mean_corr])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        mean_corr,
        xticklabels=tool_names,
        yticklabels=tool_names,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title(
        "Tool tradeoffs (league-avg covariance from VAE)",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.tight_layout()
    return _save(fig, "v3_covariance_heatmap.png", figures_dir)


def plot_radar(
    player_name: str,
    result_df: pd.DataFrame,
    tool_names: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """Radar chart comparing Z-Score / PCA / JointVAE for one player-season.

    Args:
        player_name: UniqueName string (e.g. ``"Aaron Judge (2024)"``).
        result_df: Combined result DataFrame.
        tool_names: Ordered tool names.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure, or None if the player is not found.
    """
    rows = result_df[result_df["UniqueName"] == player_name]
    if rows.empty:
        return None
    row = rows.iloc[0]

    angles = np.linspace(0, 2 * np.pi, len(tool_names), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for method, color in [("JointVAE", "red"), ("ZScore", "blue"), ("PCA", "green")]:
        vals = [row.get(f"{method}_{t}", 50) for t in tool_names] + [
            row.get(f"{method}_{tool_names[0]}", 50)
        ]
        ax.plot(angles, vals, color=color, lw=2, label=method)
        ax.fill(angles, vals, color=color, alpha=0.06)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tool_names, fontsize=12, weight="bold")
    ax.set_ylim(20, 80)
    ax.set_title(
        f"{player_name}\nWAR={row.get('WAR', '?')}, wRC+={row.get('wRC+', '?')}",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    safe = (
        player_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
    )
    return _save(fig, f"v3_radar_{safe}.png", figures_dir)


def plot_uncertainty(
    result_df: pd.DataFrame,
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """VAE Power score vs uncertainty scatter, annotated with notable players.

    Args:
        result_df: Combined result DataFrame.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure, or None if sigma columns are absent.
    """
    if "Power_sigma" not in result_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        result_df["JointVAE_Power"],
        result_df["Power_sigma"],
        c=result_df.get("WAR", 0),
        cmap="coolwarm",
        alpha=0.5,
        s=30,
    )
    for name in [
        "Aaron Judge (2024)",
        "Juan Soto (2024)",
        "Shohei Ohtani (2024)",
        "Luis Arraez (2023)",
        "Bobby Witt Jr. (2024)",
    ]:
        try:
            p = result_df[result_df["UniqueName"] == name].iloc[0]
            ax.annotate(
                name,
                (p["JointVAE_Power"], p["Power_sigma"]),
                fontsize=8,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.5},
                textcoords="offset points",
                xytext=(10, 5),
            )
        except IndexError:
            pass
    plt.colorbar(sc, label="WAR")
    ax.set_xlabel("Power (20-80)", fontsize=12)
    ax.set_ylabel("Power σ", fontsize=12)
    ax.set_title("Joint VAE: Power Score vs Uncertainty", fontsize=13, weight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return _save(fig, "v3_uncertainty.png", figures_dir)


def plot_roi_coefficients(
    wrc_res: object,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """LR coefficients + RF10 feature importance side-by-side for ΔwRC+.

    Args:
        wrc_res: :class:`roi.RegressionResult` for the ΔwRC+ target.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    coefs = wrc_res.lr_coefs
    t_sorted = sorted(coefs, key=coefs.get, reverse=True)
    vals = [coefs[t] for t in t_sorted]
    colors = ["#E24B4A" if v > 0 else "#378ADD" for v in vals]
    bars = ax.barh(t_sorted, vals, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("ΔwRC+ per unit ΔTool")
    ax.set_title(
        f"Linear Regression (5 feat)\nR²={wrc_res.lr_r2_cv:.3f} CV",
        fontsize=12,
        weight="bold",
    )
    ax.axvline(x=0, color="gray", lw=0.5)
    ax.grid(axis="x", alpha=0.2)
    for bar, val in zip(bars, vals):
        ax.text(
            val + (0.02 if val > 0 else -0.02),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=10,
        )

    ax = axes[1]
    imps = wrc_res.rf10_importance
    f_sorted = sorted(imps, key=imps.get, reverse=True)
    vals_rf = [imps[f] for f in f_sorted]
    colors_rf = ["#1D9E75" if "Δ" in f else "#EF9F27" for f in f_sorted]
    ax.barh(f_sorted, vals_rf, color=colors_rf, edgecolor="none", height=0.6)
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"Random Forest (10 feat: Current + Δ)\nR²={wrc_res.rf10_r2_cv:.3f} CV",
        fontsize=12,
        weight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(color="#1D9E75", label="ΔTool (change)"),
            Patch(color="#EF9F27", label="Current level"),
        ],
        fontsize=9,
        loc="lower right",
    )

    plt.suptitle(
        "Tool Improvement ROI: ΔTool → ΔwRC+", fontsize=14, weight="bold", y=1.02
    )
    plt.tight_layout()
    return _save(fig, "roi_coefficients.png", figures_dir)


def plot_roi_individual(
    prospect_df: pd.DataFrame,
    tool_names: list[str],
    latest_season: int,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Pie chart of best-tool distribution + top-10 WAR player table.

    Args:
        prospect_df: Output of :func:`roi.compute_individual_roi`.
        tool_names: Ordered tool names.
        latest_season: Season label for the chart title.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    ax = axes[0]
    best_counts = prospect_df["Best_Tool"].value_counts()
    palette = {
        "Power": "#E24B4A",
        "Contact": "#378ADD",
        "Speed": "#1D9E75",
        "Defense": "#EF9F27",
        "Discipline": "#7F77DD",
    }
    ax.pie(
        best_counts.values,
        labels=[f"{k}\n({v})" for k, v in best_counts.items()],
        autopct="%1.1f%%",
        colors=[palette.get(k, "#888") for k in best_counts.index],
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(
        f"Best Development Tool\n({latest_season}, n={len(prospect_df)})",
        fontsize=12,
        weight="bold",
    )

    ax = axes[1]
    ax.axis("off")
    top10 = prospect_df.nlargest(10, "WAR")
    table_data = []
    for _, row in top10.iterrows():
        second = sorted(
            [(t, row[f"ROI_{t}"]) for t in tool_names], key=lambda x: -x[1]
        )[1]
        table_data.append(
            [
                row["Name"],
                f"{row['WAR']:.1f}",
                f"{row['wRC+']:.0f}",
                row["Best_Tool"],
                f"{row['Best_ROI']:+.1f}",
                f"{second[0]}({second[1]:+.1f})",
            ]
        )
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Player", "WAR", "wRC+", "Best Tool", "ΔwRC+", "2nd Best"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(6):
        tbl[0, j].set_facecolor("#2C2C2A")
        tbl[0, j].set_text_props(color="white", weight="bold")
    ax.set_title(
        "Top 10 by WAR — Individualized ROI", fontsize=12, weight="bold", pad=20
    )

    plt.tight_layout()
    return _save(fig, "roi_individual_best_tool.png", figures_dir)


def plot_partial_dependence(
    rf10_model,
    X_full: np.ndarray,
    tool_names: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Partial dependence plots for all 10 RF features.

    Args:
        rf10_model: Fitted 10-feature RF model.
        X_full: Training array [n_samples, 10].
        tool_names: Tool names (used for axis labels).
        figures_dir: Output directory.

    Returns:
        Path to the saved figure.
    """
    from sklearn.inspection import partial_dependence

    labels = [f"{t} (current)" for t in tool_names] + [f"Δ{t}" for t in tool_names]
    fig, axes_pd = plt.subplots(2, 5, figsize=(22, 8))
    for i in range(10):
        ax = axes_pd[i // 5, i % 5]
        pd_res = partial_dependence(
            rf10_model, X_full, features=[i], kind="average", grid_resolution=50
        )
        # sklearn >= 1.2 uses "grid_values"; older versions use "values"
        grid = pd_res.get("grid_values", pd_res.get("values", [None]))[0]
        avg = pd_res["average"][0]
        ax.plot(grid, avg, color="#E24B4A", lw=2)
        ax.fill_between(grid, avg, alpha=0.1, color="#E24B4A")
        ax.set_xlabel(labels[i], fontsize=9)
        if i % 5 == 0:
            ax.set_ylabel("Predicted ΔwRC+", fontsize=10)
        ax.set_title(labels[i], fontsize=10, weight="bold")
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="gray", lw=0.5, ls="--")
        ax.axvline(x=0, color="gray", lw=0.5, ls="--")
    plt.suptitle(
        "Partial Dependence (RF 10-feature): Current Level + ΔTool → ΔwRC+",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return _save(fig, "roi_partial_dependence.png", figures_dir)


def plot_delta_scatter(
    delta_clean: pd.DataFrame,
    tool_names: list[str],
    figures_dir: Path = FIGURES_DIR,
) -> Path | None:
    """ΔTool vs ΔwRC+ scatter for each tool, with regression line.

    Args:
        delta_clean: Output of :func:`roi.build_delta_dataset`.
        tool_names: Tool names.
        figures_dir: Output directory.

    Returns:
        Path to the saved figure, or None if ΔwRC+ is absent.
    """
    if "D_wRC+" not in delta_clean.columns:
        return None

    fig, axes_sc = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for i, tool in enumerate(tool_names):
        ax = axes_sc[i]
        x, y = delta_clean[f"D_{tool}"], delta_clean["D_wRC+"]
        ax.scatter(x, y, alpha=0.3, s=15, edgecolors="none", color="#378ADD")
        mask = x.notna() & y.notna()
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            xl = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "r-", lw=1.5)
        r = x.corr(y)
        ax.set_title(f"Δ{tool} (r={r:.3f})", fontsize=11, weight="bold")
        ax.set_xlabel(f"Δ{tool}")
        if i == 0:
            ax.set_ylabel("ΔwRC+")
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="gray", lw=0.5, ls="--")
        ax.axvline(x=0, color="gray", lw=0.5, ls="--")
    plt.suptitle("ΔTool vs ΔwRC+ (Year-over-Year)", fontsize=14, weight="bold", y=1.05)
    plt.tight_layout()
    return _save(fig, "roi_delta_scatter.png", figures_dir)
