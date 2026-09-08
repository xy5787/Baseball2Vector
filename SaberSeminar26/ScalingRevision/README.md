# ScalingRevision

Isolated revision of the 20-80 tool-grade scale: `MinMaxScaler(20,80)` →
mean=50, SD=10 (per the [Fangraphs scouting-scale convention](https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/)).
Nothing here touches the main `src/baseball2vec/` package, `scripts/`, `data/`,
or `outputs/` — this is a self-contained experiment folder.

Full writeup: `outputs/report_ko.md`.

## Environment

A dedicated conda env, `baseball2vector` (python 3.12, exact versions from
`../requirements.txt`), was created for this revision:

```
conda activate baseball2vector
```

## Files

- `scaling.py` — the new `scale_to_2080_sd()` (and `build_scaled_results_sd()`,
  `clip_fraction()`).
- `run_revision.py` — **the script that actually ran.** Reads the archived
  `data/v3_results_clean.csv`, re-derives both old (MinMax) and new (SD=10)
  grades (mathematically identical to re-deriving them from the model's raw
  latents — see report for why), and writes correlation tables + the full
  02/03-style figure set to `outputs/`.
- `02_baseline_revision.py`, `03_joint_vae_revision.py` — **run successfully.**
  Mirror `scripts/02_baseline_comparison.py` / `03_joint_vae.py` but do a
  genuine from-scratch retrain and write to `outputs/` instead of the main
  pipeline's output dirs. `pybaseball.batting_stats()` (FanGraphs) is blocked
  in this environment (Cloudflare challenge, confirmed even via a headless
  Playwright browser — not just a UA check), so the raw data instead came
  from a manual FanGraphs leaderboard export, validated against the archived
  `data/v3_results_clean.csv` (see `build_raw_from_fangraphs_export.py` and
  `outputs/report_ko.md`) and saved to `data/raw/batting_stats_2021_2025.csv`
  (the shared cache path — the original `scripts/01-04` pipeline can also use
  it directly). To re-run:
  `conda activate baseball2vector && python 02_baseline_revision.py && python 03_joint_vae_revision.py`
- `build_raw_from_fangraphs_export.py` — converts the manual FanGraphs export
  into `data/raw/batting_stats_2021_2025.csv` (dedupes repeated ID columns,
  filters to 2021-2025 + PA>=100, resolves the one known Name+Season
  collision). Already run; re-run only if the export is refreshed.

## Outputs

- `outputs/rescaled_results.csv`, `correlation_comparison.csv`,
  `clip_fractions.csv`, `grade_shift.csv` — from `run_revision.py`.
- `outputs/figures/old/`, `outputs/figures/new/` — correlation heatmap,
  scatter grid, 4 player radar charts, uncertainty plot, for each scale.
  (VAE covariance heatmap requires the retrain above — not included.)
- `outputs/report_ko.md` — full explanation and results (Korean).
