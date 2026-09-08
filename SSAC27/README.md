# Baseball2Vector — SSAC 2027 Experiment Package

This directory contains the forecasting and comparable-player experiments prepared for an SSAC 2027 submission based on **Baseball2Vector (B2V)**, a five-dimensional representation of MLB hitters: Contact, Power, Speed, Defense, and Plate Discipline.

The central question is deliberately narrower than “Can five numbers predict a player?” It is whether a compact baseball-native profile adds useful next-season information beyond a calibrated scalar summary, how much information is lost relative to the 22 constituent statistics, and whether the profile remains useful when combined with a history-based Marcel projection.

The full, table-by-table analysis is in [`RESULTS.md`](RESULTS.md). Machine-readable predictions, fold summaries, paired comparisons, tuning records, and environment metadata are in [`results/`](results/).

## Headline results

- Across three anchored rolling origins (1,063 held-out player-seasons), B2V improves next-season wRC+ MAE over a calibrated current-wRC+ Ridge by **1.171 points** (95% player-cluster bootstrap CI **[0.706, 1.631]**). The sign is favorable in all three folds.
- In the extended 2019–2025 study (five origins; **1,706** held-out cases), the corresponding improvement is **0.981** (**[0.619, 1.343]**), favorable in all five folds and individually distinguishable from zero in four.
- The 22 constituent statistics outperform the five-tool compression by **0.180 wRC+ MAE** in the extended analysis (**[0.008, 0.350]**). The five tools retain most, but not all, of the predictive information.
- B2V and a training-window-calibrated Marcel projection are not clearly different on their own. Adding B2V to calibrated Marcel improves pooled wRC+ MAE by **0.404** (**[0.193, 0.608]**) across the five-origin study.
- B2V-based past-season neighbors improve next-season wRC+ retrieval MAE over neighbors selected by WAR and wRC+ by **0.784** (**[0.109, 1.406]**) at *k*=10. The retrieval model remains less accurate than fitted B2V Ridge, so its value is interpretability rather than state-of-the-art prediction.
- Z-score, PCA, and JointVAE versions give nearly the same forecasting result. Their pooled wRC+ improvements over the scalar baseline are 1.171, 1.188, and 1.106 points, respectively; the total spread is under 0.08 MAE.
- B2V is less effective for next-season counting WAR but improves WAR per 600 PA. This split indicates that much of counting-WAR error concerns future playing time rather than the five-dimensional quality profile itself.

Positive paired differences in this package mean **baseline MAE minus candidate MAE**; positive values therefore favor the candidate.

## Study populations

The analyses use three related but non-interchangeable datasets:

| Population | Seasons | Rows | PA rule | Purpose |
|---|---:|---:|---|---|
| Anchored cohort | 2021–2025 | 2,286 player-seasons | PA ≥ 100 | Archived stable-ID tools and outcomes for Tasks 1–5 |
| Marcel history pool | 2019–2025 | 5,293 player-seasons | No minimum | Prior-season history only; never a focal modeling cohort |
| Extended cohort | 2019–2025 | 3,070 player-seasons | PA ≥ 100 | Recomputed tools and five-origin evaluation in Task 6 |

The anchored cohort produces 1,414 adjacent-season transitions and three rolling-origin test folds. The extended cohort produces 1,981 transitions and five test folds. Every forecast case requires PA ≥ 100 in both the input and outcome season, so the evaluation is conditional on a player remaining in the qualified MLB sample.

## Experiments

| Task | Question | Main comparison |
|---:|---|---|
| 1 | Does five-tool compression discard predictive information? | B2V five tools vs 22 constituents on 2024→2025 |
| 2 | Does B2V beat a scalar baseline across time? | Rolling-origin B2V vs calibrated current-season value |
| 3 | Is B2V useful beyond player history? | B2V, Marcel, calibrated Marcel, and Marcel + B2V |
| 4 | Does B2V retrieve better historical comparables? | Past-season B2V neighbors vs WAR+wRC+ neighbors |
| 5 | Does the result depend on the encoder? | Z-score vs PCA vs JointVAE tool profiles |
| 6 | Does the conclusion hold over a longer window? | Recomputed 2019–2025 cohort and five origins |

Three targets are evaluated throughout: next-season **wRC+**, **WAR**, and **WAR per 600 PA**. Transitions touching the 60-game 2020 season are excluded from counting-WAR analyses. Task 6 also reports a 2020-free sensitivity analysis for wRC+.

## Models

- **M0:** calibrated Ridge using the current value of the target; this is stronger than naive carry-forward.
- **M1:** five B2V tools.
- **M2:** 22 constituent statistics.
- **M3–M5:** the corresponding scalar, B2V, and constituent models with age and PA.
- **M6:** raw Marcel projection.
- **M6c:** Marcel recalibrated within the available training window.
- **M7:** calibrated Marcel plus B2V.
- **M7x:** calibrated Marcel plus B2V, age, and PA.

All learned models use Ridge regression. Preprocessing and hyperparameter selection are fitted on training observations only. The default alpha grid is `numpy.logspace(-3, 4, 50)`, selected with five-fold player-grouped cross-validation inside each training window.

## Repository layout

```text
SSAC27/
├── README.md
├── RESULTS.md                     # narrative report and all main tables
├── requirements.txt
├── scripts/
│   ├── run_ssac_experiments.py   # task orchestrator
│   ├── ssac_common.py            # shared models, evaluation, and bootstrap
│   ├── ssac_data.py              # extended-cohort construction
│   ├── marcel.py
│   └── task1_*.py ... task6_*.py
├── data_intermediate/
│   └── transitions.csv           # committed derived transition table
└── results/                       # predictions, comparisons, tuning, provenance
```

The SSAC folder intentionally excludes the language-model pilot and unrelated validation branches. The original representation-learning pipeline and its manuscript live in [`../SaberSeminar26/`](../SaberSeminar26/).

## Environment

The archived SSAC results were generated with Python 3.14.6 and the exact package versions in `requirements.txt`:

```bash
cd SSAC27
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch is not required for Tasks 1–5 because the archived Z-score, PCA, and JointVAE tool scores are loaded rather than retrained.

## Reproduction status and required inputs

The committed `RESULTS.md`, `data_intermediate/transitions.csv`, and `results/*.csv` files are sufficient to inspect every reported estimate and audit the saved predictions. A complete rerun is more demanding because raw FanGraphs exports and the stable-ID player-season table are excluded from Git.

The scripts currently expect these local inputs relative to the repository root:

```text
data/raw/batting_stats_2021_2025.csv
data/raw/fangraphs-leaderboards (1).csv
data/raw/batting_stats_2019_2020.csv
final_academic_revision/run_20260808_055335/
└── data_intermediate/player_seasons_with_ids.csv
```

They also import the shared `baseball2vec` modules from a root-level `src/` directory. In the newly split repository, those modules and the corresponding data lineage live under `SaberSeminar26/`. Until the SSAC path constants are migrated to the split layout, either update `SSAC27/scripts/ssac_common.py` and `ssac_data.py` to point to `SaberSeminar26/`, or stage compatibility links at the legacy root paths. This path migration does not affect the committed results.

After the inputs and paths are available, run all tasks from the repository root:

```bash
python SSAC27/scripts/run_ssac_experiments.py
```

Run one or more tasks with repeated `--task` arguments:

```bash
python SSAC27/scripts/run_ssac_experiments.py --task 2
python SSAC27/scripts/run_ssac_experiments.py --task 3 --task 6
```

Tasks are independent. Each run overwrites the corresponding task files under `SSAC27/results/`.

## Evaluation safeguards

- Test seasons are strictly later than their training seasons.
- Imputation, standardization, and Ridge tuning use training rows only.
- Inner validation is grouped by stable player ID.
- Tool and constituent features are standardized within their input-season cohort; future seasons never determine a past feature.
- Every principal uncertainty interval uses 2,000 bootstrap replications clustered by focal player.
- Comparable-player candidates come only from seasons available before the forecast target; same-season retrieval is not treated as deployable.
- Seed 42, package versions, model grids, row counts, and input hashes are recorded in the task environment and decision files.

## Reading the results responsibly

- Improvements are modest in absolute scale. B2V adds context; it does not turn year-ahead player forecasting into a low-error problem.
- The five-tool compression is interpretable but is not information-free: the 22-stat model has a small predictive advantage in the longer study.
- The Marcel history begins in 2019. Early origins therefore cannot use three full prior seasons, and history-depth comparisons should be read with that boundary in mind.
- The 2020 season has an unusual schedule and qualified-player population. Results that include it are paired with explicit sensitivity checks.
- Counting WAR mixes player quality with future playing time, health, and roster opportunity. WAR per 600 PA is reported to separate those questions.
- Neighbor retrieval is a communication and case-comparison tool, not the best-performing forecast model in this package.
- These are retrospective observational experiments, not causal estimates or a production decision system.

## Result map

- Overall narrative and tables: [`RESULTS.md`](RESULTS.md)
- Analytic decisions and deviations: [`results/decisions_log.md`](results/decisions_log.md)
- Held-out constituent control: `results/task1_*`
- Rolling-origin estimates: `results/task2_*`
- Marcel and history depth: `results/task3_*`
- Comparable-player retrieval: `results/task4_*`
- Encoder comparison: `results/task5_*`
- Extended-window analysis: `results/task6_*`

## Citation

Citation details will be added after the SSAC submission is finalized. Until then, please cite the repository and the specific commit used for analysis.

## Contact

Jaeseok Choi — `jaeseok.choi@skku.edu`

I2SLAB, Sungkyunkwan University
