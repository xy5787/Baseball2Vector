"""Diagnostic figure: does the actual player-grade distribution under the new
mean=50/SD=10 20-80 scale actually look like the theoretical N(50, 10) it's
built to approximate? One panel per tool: theoretical bell curve + a jittered
scatter of every player-season's grade for that tool.

Purely exploratory / curiosity-driven -- not part of the paper pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseball2vec.tools import TOOL_NAMES

CURVE_COLOR = "#2a78d6"   # theoretical N(50,10) -- dataviz palette slot 1
SCATTER_COLOR = "#eb6834"  # actual player grades -- dataviz palette slot 2
GRID_COLOR = "#c3c2b7"

df = pd.read_csv(Path(__file__).parent / "data" / "processed" / "v3_results.csv")

fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=False)
x = np.linspace(20, 80, 400)
theoretical = norm.pdf(x, loc=50, scale=10)

rng = np.random.default_rng(42)

for ax, tool in zip(axes, TOOL_NAMES):
    col = f"JointVAE_{tool}"
    grades = df[col].dropna().values

    ax.plot(x, theoretical, color=CURVE_COLOR, lw=2, label="Theoretical N(50, 10)", zorder=3)
    ax.fill_between(x, theoretical, color=CURVE_COLOR, alpha=0.06, zorder=1)

    rug_top = theoretical.max() * 0.22
    jitter = rng.uniform(0, rug_top, size=len(grades))
    ax.scatter(
        grades, jitter,
        s=8, color=SCATTER_COLOR, alpha=0.3, edgecolors="none",
        label="Actual player-seasons", zorder=2,
    )
    ax.axhline(rug_top, color=GRID_COLOR, lw=0.6, zorder=0)

    for gx in [20, 30, 40, 50, 60, 70, 80]:
        ax.axvline(gx, color=GRID_COLOR, lw=0.6, zorder=0)

    n_clip_lo = (grades <= 20.05).sum()
    n_clip_hi = (grades >= 79.95).sum()
    ax.set_title(
        f"{tool}\n(n={len(grades)}, clipped: {n_clip_lo} lo / {n_clip_hi} hi)",
        fontsize=11, weight="bold",
    )
    ax.set_xlim(15, 85)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors="#52514e")

axes[0].set_ylabel("density / player-seasons", color="#52514e", fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.suptitle(
    "20-80 scale vs. theoretical N(50, 10) — JointVAE grades, 2021-2025 (n=2309)",
    fontsize=13, weight="bold", y=1.14,
)

out = Path(__file__).parent / "outputs" / "figures" / "scale_normality_check.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
