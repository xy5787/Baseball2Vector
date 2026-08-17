# Phase 0 — Data Audit

Source: `data/processed/v3_results.csv`

## Row count, season coverage, columns

- Row count: **2309**
- Season coverage: 2021=463, 2022=469, 2023=461, 2024=455, 2025=461
- Columns (30): UniqueName, Name, Season, PA, WAR, wRC+, OPS, ZScore_Contact, ZScore_Power, ZScore_Speed, ZScore_Defense, ZScore_Discipline, ZScore_Area, PCA_Contact, PCA_Power, PCA_Speed, PCA_Defense, PCA_Discipline, PCA_Area, JointVAE_Contact, JointVAE_Power, JointVAE_Speed, JointVAE_Defense, JointVAE_Discipline, JointVAE_Area, Contact_sigma, Power_sigma, Speed_sigma, Defense_sigma, Discipline_sigma

## Max Muncy 2025 duplicate-row check

All `Max Muncy` rows in `v3_results.csv`:

| UniqueName | Name | Season | PA | WAR | wRC+ | OPS |
| --- | --- | --- | --- | --- | --- | --- |
| Max Muncy (2021) | Max Muncy | 2021 | 592 | 4.5 | 139 | 0.895 |
| Max Muncy (2022) | Max Muncy | 2022 | 565 | 2.0 | 106 | 0.713 |
| Max Muncy (2023) | Max Muncy | 2023 | 579 | 2.5 | 117 | 0.808 |
| Max Muncy (2024) | Max Muncy | 2024 | 293 | 2.3 | 133 | 0.852 |
| Max Muncy (2025) | Max Muncy | 2025 | 388 | 2.9 | 137 | 0.846 |
| Max Muncy (2025) | Max Muncy | 2025 | 220 | -0.4 | 72 | 0.638 |

**Finding:** these are two distinct real MLB players who share a name, not a data-pipeline duplicate:

- **Max Muncy (b. 1990)** — Dodgers 3B/1B, MLB debut 2015 ([Baseball-Reference `muncyma01`](https://www.baseball-reference.com/players/m/muncyma01.shtml)). The 388-PA / 2.9-WAR / 137-wRC+ row matches his established, productive 2025 season.
- **Max Muncy (b. 2002)** — Athletics 3B/SS, MLB debut March 27, 2025 as a rookie ([Baseball-Reference `muncyma02`](https://www.baseball-reference.com/players/m/muncyma02.shtml); [Athletics Nation 2025 season review](https://www.athleticsnation.com/commentary-analysis/101361/2025-season-in-review-max-muncy)). The 220-PA / -0.4-WAR / 72-wRC+ row matches a rookie season that opened at ~30 wRC+, climbed to 81 then 119 wRC+ by June/July, then was cut short by a hand injury on July 21 — consistent with a blended full-season wRC+ of 72 over only 220 PA.

**Disambiguation rule applied:** treat as two distinct players (neither row is dropped — both are valid, real qualifying seasons). The pipeline keys player identity purely on the `Name` string (see `build_delta_dataset` in `src/baseball2vec/roi.py`, which does `groupby("Name")`), so leaving both rows labeled `Max Muncy` risks the Phase 1 transition-dataset builder silently splicing one player's earlier season onto the other player's 2025 row when it sorts by season within the `Max Muncy` group. To prevent this, the lower-PA (rookie, 220 PA) row is relabeled `Max Muncy (ATH)` in `Name` and `UniqueName` in the cleaned dataset; the veteran Dodgers row is left as `Max Muncy`. This affects only 1 of 2,309 rows and does not change any Table 1 correlation (which is computed over season-rows independent of player identity), but it does change the Phase 1 transition dataset: without the fix, the veteran's 2024→2025 transition would nondeterministically pick whichever 2025 row pandas `sort_values` happened to place first for that `Name` group.

## General Name+Season collision scan

`groupby(["Name", "Season"])` collisions found: **1** (Max Muncy / 2025 only, handled above).

Sanity ranges (season-rows, n=2,309):

- `PA`: [100, 753]
- `WAR`: [-2.2, 11.3]
- `wRC+`: [-18, 220]
- `OPS`: [0.322, 1.159]

All ranges are within plausible bounds for qualified (≥100 PA) hitter-seasons; no negative PA, no OPS outside [0.3, 1.2], no missing values in any column, and no fully duplicated rows (`df.duplicated().sum() == 0`). No impossible stat combinations found beyond the Max Muncy name collision above.

## Z-scoring / stabilization: within-season or pooled?

Static inspection of the pipeline source (not the output data) confirms:

- **`season_zscore` (`src/baseball2vec/tools.py:56-72`)** groups by `Season` before computing the mean/std used for Z-scoring (`X.groupby("Season")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))`). This **is** within-season, matching the paper's claim — no pooling leakage here.
- **Bayesian stabilization (`src/baseball2vec/data.py:108-146`)** computes `season_means = df.groupby("Season").mean(...)` and shrinks each row toward its *own season's* league average. Also within-season — no leakage.
- **20–80 min–max scaling (`scale_to_2080`, `src/baseball2vec/tools.py:91-106`)** fits a single `MinMaxScaler` over the **full pooled 2021–2025 population** (`scaler.fit_transform(scores_df[cols])` called once on all 2,309 rows). This pooling is explicitly disclosed in the paper (Sec. 3, line 119: "an affine min–max transformation over the pooled 2021–2025 population") for the **descriptive** Table 1 correlations and Table 2 top-10 list — that use is fine per this project's leakage-discipline rule, since those are population-level descriptive statistics, not cross-validated metrics.

**However**, this same pooled-fit `ZScore_{Tool}` column (post 20–80 scaling) is exactly what feeds `build_delta_dataset` / `run_regression` in `src/baseball2vec/roi.py` for the cross-validated ΔTool → ΔWAR/ΔwRC+/ΔOPS regression (the paper's $R^2=0.71$ claim, Sec. 4.3). Because the min–max bounds are fit once across *all* seasons — including seasons that fall in a given fold's held-out set — this is a real (if narrow) leakage channel: a held-out fold's extreme values can influence the scaler that produced its own features. **This is the leakage risk Phase 1 is scoped to fix** by refitting the min–max bounds inside each training fold only, rather than something silently patched here.

## Output

- Cleaned dataset written to `data/v3_results_clean.csv` (2309 rows, 1 row relabeled).
- All downstream revision phases (1–4) should load `data/v3_results_clean.csv`, not the original `v3_results.csv`.
