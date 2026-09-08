"""Figure 1: Z-Score Pentagon Area vs WAR scatter plot
Figure 2: Bobby Witt Jr. (2024) vs Adley Rutschman (2022) radar plot
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FIGURES_DIR = Path(__file__).parents[1] / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "v3_results.csv"
df = pd.read_csv(DATA_PATH)

TOOL_NAMES = ["Contact", "Power", "Speed", "Defense", "Discipline"]

# ── Figure 1: Z-Score Pentagon Area vs WAR scatter ──────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

seasons = sorted(df["Season"].unique())
cmap = plt.colormaps["viridis"]
season_colors = {s: cmap(i / (len(seasons) - 1)) for i, s in enumerate(seasons)}

for season, group in df.groupby("Season"):
    ax.scatter(
        group["ZScore_Area"],
        group["WAR"],
        color=season_colors[season],
        alpha=0.45,
        s=22,
        edgecolors="none",
        label=str(season),
        zorder=2,
    )

# regression line
mask = df["ZScore_Area"].notna() & df["WAR"].notna()
slope, intercept, r_val, p_val, _ = stats.linregress(
    df.loc[mask, "ZScore_Area"], df.loc[mask, "WAR"]
)
xl = np.linspace(df.loc[mask, "ZScore_Area"].min(), df.loc[mask, "ZScore_Area"].max(), 200)
ax.plot(xl, slope * xl + intercept, color="#E24B4A", lw=2, zorder=3,
        label=f"Trend (r={r_val:.3f}, p<0.001)")

# annotate two highlight players
'''
highlights = {
    "Bobby Witt Jr. (2024)": ("#FF6B35", "below"),
    "Adley Rutschman (2022)": ("#4ECDC4", "above"),
}
for uname, (color, pos) in highlights.items():
    row = df[df["UniqueName"] == uname]
    if row.empty:
        continue
    row = row.iloc[0]
    ax.scatter(row["ZScore_Area"], row["WAR"], color=color, s=120,
               edgecolors="white", lw=1.5, zorder=5)
    yoff = -18 if pos == "below" else 10
    ax.annotate(
        uname,
        (row["ZScore_Area"], row["WAR"]),
        xytext=(0, yoff),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color=color,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1},
        zorder=6,
    )
'''
ax.set_xlabel("Z-Score Pentagon Area (20–80 scale)", fontsize=12)
ax.set_ylabel("WAR", fontsize=12)
ax.set_title(
    "Z-Score Pentagon Area vs WAR  (2021–2025, n=2,309)",
    fontsize=14, fontweight="bold", pad=14,
)
ax.grid(True, alpha=0.2, zorder=1)

# season legend
handles_seasons = [
    plt.scatter([], [], color=season_colors[s], s=30, label=str(s))
    for s in seasons
]
legend_seasons = ax.legend(
    handles=handles_seasons,
    title="Season",
    loc="upper left",
    fontsize=9,
    title_fontsize=9,
    framealpha=0.85,
)
ax.add_artist(legend_seasons)

# trend line legend (separate from season legend)
trend_handle = plt.Line2D([0], [0], color="#E24B4A", lw=2,
                           label=f"Trend (r={r_val:.3f}, p<0.001)")
ax.legend(handles=[trend_handle], loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout()
out1 = FIGURES_DIR / "fig1_zscore_area_vs_war.png"
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")


# ── Figure 2: Bobby Witt Jr. (2024) vs Adley Rutschman (2022) radar ─────────
players = {
    "Bobby Witt Jr. (2024)": "#FF6B35",
    "Adley Rutschman (2022)": "#4ECDC4",
}

angles = np.linspace(0, 2 * np.pi, len(TOOL_NAMES), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 9), subplot_kw={"polar": True})
fig.subplots_adjust(bottom=0.2)

for uname, color in players.items():
    row = df[df["UniqueName"] == uname]
    if row.empty:
        print(f"  Not found: {uname}")
        continue
    row = row.iloc[0]
    vals = [row[f"ZScore_{t}"] for t in TOOL_NAMES] + [row[f"ZScore_{TOOL_NAMES[0]}"]]

    war_val = row["WAR"]
    wrc_val = int(row["wRC+"])
    area_val = row["ZScore_Area"]
    label = f"{uname}\nWAR={war_val:.1f}  wRC+={wrc_val}  Area={area_val:.0f}"

    ax.plot(angles, vals, color=color, lw=2.5, label=label, zorder=3)
    ax.fill(angles, vals, color=color, alpha=0.12, zorder=2)
    # dots at each vertex
    ax.scatter(angles[:-1], vals[:-1], color=color, s=60, zorder=4, edgecolors="white", lw=1)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(TOOL_NAMES, fontsize=13, fontweight="bold")

# y-axis: 20-80 scale
ax.set_ylim(20, 80)
ax.set_yticks([20, 30, 40, 50, 60, 70, 80])
ax.set_yticklabels(["20", "30", "40", "50", "60", "70", "80"], fontsize=8, color="gray")
ax.yaxis.set_tick_params(pad=6)

# reference ring at 50 (league average)
ax.plot(angles, [50] * len(angles), color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)
ax.text(np.pi / 2, 50, " avg", fontsize=8, color="gray", ha="left", va="center")

ax.set_title(
    "5-Tool Radar: Bobby Witt Jr. (2024) vs Adley Rutschman (2022)\n(Z-Score method, 20–80 scale)",
    fontsize=13, fontweight="bold", pad=18,
)
ax.grid(True, alpha=0.25)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right",
           bbox_to_anchor=(0.97, 0.02),
           fontsize=8, framealpha=0.9, borderaxespad=0)
out2 = FIGURES_DIR / "fig2_radar_witt_vs_rutschman.png"
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")
