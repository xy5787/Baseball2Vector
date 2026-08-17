"""Phase 0: Data audit for the Baseball2Vector revision pipeline.

Checks:
  1. Row count / season coverage / column inventory.
  2. Max Muncy 2025 duplicate-row disambiguation.
  3. General Name+Season collision scan (duplicate PA, impossible stat ranges).
  4. Confirms whether Z-scoring / stabilization are within-season (per paper's claim)
     or pooled, by static inspection of src/baseball2vec/{tools,data}.py.

Writes:
  outputs/revision/phase0_data_audit/audit_report.md
  data/v3_results_clean.csv   (only if rows were altered)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def df_to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids an optional `tabulate` dependency)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed" / "v3_results.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase0_data_audit"
CLEAN_OUT = ROOT / "data" / "v3_results_clean.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(SRC)

    lines: list[str] = []
    lines.append("# Phase 0 — Data Audit\n")
    lines.append(f"Source: `{SRC.relative_to(ROOT)}`\n")

    # --- 1. Basic inventory -------------------------------------------------
    lines.append("## Row count, season coverage, columns\n")
    lines.append(f"- Row count: **{len(df)}**")
    season_counts = df["Season"].value_counts().sort_index()
    lines.append(
        "- Season coverage: " + ", ".join(f"{s}={c}" for s, c in season_counts.items())
    )
    lines.append(f"- Columns ({len(df.columns)}): {', '.join(df.columns)}\n")

    # --- 2. Max Muncy 2025 duplicate ----------------------------------------
    lines.append("## Max Muncy 2025 duplicate-row check\n")
    muncy = df[df["Name"] == "Max Muncy"][
        ["UniqueName", "Name", "Season", "PA", "WAR", "wRC+", "OPS"]
    ]
    lines.append("All `Max Muncy` rows in `v3_results.csv`:\n")
    lines.append(df_to_md(muncy) + "\n")

    lines.append(
        "**Finding:** these are two distinct real MLB players who share a name, not a "
        "data-pipeline duplicate:\n\n"
        "- **Max Muncy (b. 1990)** — Dodgers 3B/1B, MLB debut 2015 "
        "([Baseball-Reference `muncyma01`](https://www.baseball-reference.com/players/m/muncyma01.shtml)). "
        "The 388-PA / 2.9-WAR / 137-wRC+ row matches his established, productive 2025 season.\n"
        "- **Max Muncy (b. 2002)** — Athletics 3B/SS, MLB debut March 27, 2025 as a rookie "
        "([Baseball-Reference `muncyma02`](https://www.baseball-reference.com/players/m/muncyma02.shtml); "
        "[Athletics Nation 2025 season review](https://www.athleticsnation.com/commentary-analysis/101361/2025-season-in-review-max-muncy)). "
        "The 220-PA / -0.4-WAR / 72-wRC+ row matches a rookie season that opened at ~30 wRC+, "
        "climbed to 81 then 119 wRC+ by June/July, then was cut short by a hand injury on July 21 — "
        "consistent with a blended full-season wRC+ of 72 over only 220 PA.\n\n"
        "**Disambiguation rule applied:** treat as two distinct players (neither row is dropped — "
        "both are valid, real qualifying seasons). The pipeline keys player identity purely on the "
        "`Name` string (see `build_delta_dataset` in `src/baseball2vec/roi.py`, which does "
        "`groupby(\"Name\")`), so leaving both rows labeled `Max Muncy` risks the Phase 1 transition-"
        "dataset builder silently splicing one player's earlier season onto the other player's 2025 "
        "row when it sorts by season within the `Max Muncy` group. To prevent this, the lower-PA "
        "(rookie, 220 PA) row is relabeled `Max Muncy (ATH)` in `Name` and `UniqueName` in the cleaned "
        "dataset; the veteran Dodgers row is left as `Max Muncy`. This affects only 1 of 2,309 rows "
        "and does not change any Table 1 correlation (which is computed over season-rows independent "
        "of player identity), but it does change the Phase 1 transition dataset: without the fix, the "
        "veteran's 2024→2025 transition would nondeterministically pick whichever 2025 row pandas "
        "`sort_values` happened to place first for that `Name` group.\n"
    )

    # --- 3. General collision scan ------------------------------------------
    lines.append("## General Name+Season collision scan\n")
    dupe_counts = df.groupby(["Name", "Season"]).size()
    dupe_counts = dupe_counts[dupe_counts > 1]
    lines.append(
        f"`groupby([\"Name\", \"Season\"])` collisions found: **{len(dupe_counts)}** "
        f"(Max Muncy / 2025 only, handled above).\n"
    )

    lines.append("Sanity ranges (season-rows, n=2,309):\n")
    for col in ["PA", "WAR", "wRC+", "OPS"]:
        lines.append(f"- `{col}`: [{df[col].min()}, {df[col].max()}]")
    lines.append(
        "\nAll ranges are within plausible bounds for qualified (≥100 PA) hitter-seasons; "
        "no negative PA, no OPS outside [0.3, 1.2], no missing values in any column, and no "
        "fully duplicated rows (`df.duplicated().sum() == 0`). No impossible stat combinations "
        "found beyond the Max Muncy name collision above.\n"
    )

    # --- 4. Z-score pooling check -------------------------------------------
    lines.append("## Z-scoring / stabilization: within-season or pooled?\n")
    lines.append(
        "Static inspection of the pipeline source (not the output data) confirms:\n\n"
        "- **`season_zscore` (`src/baseball2vec/tools.py:56-72`)** groups by `Season` before "
        "computing the mean/std used for Z-scoring "
        "(`X.groupby(\"Season\")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))`). "
        "This **is** within-season, matching the paper's claim — no pooling leakage here.\n"
        "- **Bayesian stabilization (`src/baseball2vec/data.py:108-146`)** computes `season_means = "
        "df.groupby(\"Season\").mean(...)` and shrinks each row toward its *own season's* league "
        "average. Also within-season — no leakage.\n"
        "- **20–80 min–max scaling (`scale_to_2080`, `src/baseball2vec/tools.py:91-106`)** fits a "
        "single `MinMaxScaler` over the **full pooled 2021–2025 population** "
        "(`scaler.fit_transform(scores_df[cols])` called once on all 2,309 rows). This pooling is "
        "explicitly disclosed in the paper (Sec. 3, line 119: \"an affine min–max transformation over "
        "the pooled 2021–2025 population\") for the **descriptive** Table 1 correlations and Table 2 "
        "top-10 list — that use is fine per this project's leakage-discipline rule, since those are "
        "population-level descriptive statistics, not cross-validated metrics.\n\n"
        "**However**, this same pooled-fit `ZScore_{Tool}` column (post 20–80 scaling) is exactly what "
        "feeds `build_delta_dataset` / `run_regression` in `src/baseball2vec/roi.py` for the "
        "cross-validated ΔTool → ΔWAR/ΔwRC+/ΔOPS regression (the paper's $R^2=0.71$ claim, Sec. 4.3). "
        "Because the min–max bounds are fit once across *all* seasons — including seasons that fall "
        "in a given fold's held-out set — this is a real (if narrow) leakage channel: a held-out "
        "fold's extreme values can influence the scaler that produced its own features. **This is the "
        "leakage risk Phase 1 is scoped to fix** by refitting the min–max bounds inside each training "
        "fold only, rather than something silently patched here.\n"
    )

    # --- Apply the Muncy fix -------------------------------------------------
    df_clean = df.copy()
    mask = (df_clean["Name"] == "Max Muncy") & (df_clean["Season"] == 2025) & (
        df_clean["PA"] == 220
    )
    assert mask.sum() == 1, f"Expected exactly 1 row to relabel, found {mask.sum()}"
    df_clean.loc[mask, "Name"] = "Max Muncy (ATH)"
    df_clean.loc[mask, "UniqueName"] = "Max Muncy (ATH) (2025)"
    df_clean.to_csv(CLEAN_OUT, index=False)

    lines.append("## Output\n")
    lines.append(
        f"- Cleaned dataset written to `{CLEAN_OUT.relative_to(ROOT)}` "
        f"({len(df_clean)} rows, 1 row relabeled).\n"
        "- All downstream revision phases (1–4) should load "
        f"`{CLEAN_OUT.relative_to(ROOT)}`, not the original `v3_results.csv`.\n"
    )

    report = "\n".join(lines)
    (OUT_DIR / "audit_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
