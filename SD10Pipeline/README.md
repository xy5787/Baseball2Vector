# SD10Pipeline

The "production" run of the full Baseball2Vec pipeline (`scripts/01-04` /
`run_all.py` equivalent) under the corrected 20-80 scale (mean=50, SD=10,
[Fangraphs convention](https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/))
instead of `MinMaxScaler`. Unlike `../ScalingRevision/`, which computes old
vs new side by side for comparison/validation, this folder runs the pipeline
**once, under the new scale only** — i.e. what the repo's real outputs would
look like if `src/baseball2vec/tools.py::scale_to_2080` were replaced.
Nothing here touches the main `src/`, `scripts/`, `data/`, or `outputs/`.

Data: `data/raw/batting_stats_2021_2025.csv` (shared cache, validated against
the archived `data/v3_results_clean.csv` — see `../ScalingRevision/`).
VAE: retrained from scratch here (seed=42, 4000 epochs, unmodified
`joint_vae.py`) — a separate training run from ScalingRevision's, same code
and data, so results match to the precision expected from PyTorch's
non-bit-exact reproducibility.

Run order: `02_baseline_comparison.py` → `03_joint_vae.py` → `04_roi_analysis.py`.

## Key results (see `outputs/report_ko.md` for full detail)

- Pentagon Area vs WAR/wRC+/OPS correlations: same modest decrease vs the
  archived MinMax pipeline as found in ScalingRevision (clipping at the
  tails; not a sign the new scale is worse — see that folder's report for
  why).
- ROI analysis (`04`): the linear-regression tool-importance ranking for
  ΔwRC+ is unchanged (Power > Contact > Discipline > Defense > Speed in both
  scales). But the **personalized RF10 "which tool should this player work
  on" recommendation flips in aggregate**: 213/348 (61%) of qualifying
  players were Contact-best under the old scale vs 98/348 (28%) under the
  new one — Power becomes the majority recommendation (250/348, 72%) instead.
  This is a real, retrained result, not a CSV-reuse artifact.

## Outputs

- `data/processed/v3_results.csv` — the new-scale-only per-player results
  (drop-in replacement for the main pipeline's `data/processed/v3_results.csv`).
- `outputs/figures/` — v2/v3 correlation heatmap, scatter grid, covariance
  heatmap, 4 player radars, uncertainty plot, ROI coefficients/partial
  dependence/delta scatter/individual-best-tool charts.
- `outputs/tables/` — `correlations.csv`, `v3_summary.txt`, `lr_results.csv`,
  `individual_roi.csv`, `delta_dataset.csv`.
