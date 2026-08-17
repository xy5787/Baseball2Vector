# Baseball2Vec

**Baseball2Vec** embeds MLB hitters into a 5-dimensional vector space aligned with the traditional scouting *5-Tool* framework — Contact, Power, Speed, Defense, and Plate Discipline. Each dimension is learned by a Joint Encoder VAE with Cholesky full covariance, so the off-diagonal entries of the learned Σ directly quantify tool tradeoffs. We validate the embedding by regressing the area of each player's *Pentagon* (a 20–80-scaled radar polygon) against WAR, wRC+, and OPS, and identify personalized development directions by regressing year-over-year tool changes (ΔTool) against ΔwRC+ with a 10-feature Random Forest.

> **Submitted to:** Saberseminar 2026 (August 29–30, Chicago)
> **Authors:** Jaeseok Choi · I2SLAB, Sungkyunkwan University · Advisor: Prof. Jangwon Lee

---

## What's in this repository

The original submission pipeline (`src/`, `scripts/01`–`04`, `run_all.py`) is unchanged. Everything
added since is **post-submission analysis kept in isolated, self-contained folders** — none of them
write to `src/`, `scripts/01-04`, `data/processed/`, or the top-level `outputs/figures|tables/`.

| Folder | What it is | Writeup |
|---|---|---|
| `scripts/revision/` + `outputs/revision/` | Methodological revision, Phases 0–5: data audit, player-grouped CV, bootstrap CIs, ablations, external pilot, repro packaging | [`REVISIONS.md`](REVISIONS.md) |
| `academic_validation_experiments/` | Stage A validation: stable-player-ID audit, prospective prediction, stability/ablation, profile diversity | `academic_validation_experiments/report_ko.md` |
| `final_academic_revision/run_*/` | Two dated, hash-pinned reproduction runs behind the revised manuscript (forecast-safe prediction, matching, figures, LaTeX tables) | each run's `README.md` and `report_ko.md` |
| `ScalingRevision/` | Old (MinMax) vs new (mean=50, SD=10) 20–80 scale, computed side by side | `ScalingRevision/outputs/report_ko.md` |
| `SD10Pipeline/` | The full 01–04 pipeline re-run *once*, under the SD=10 scale only | `SD10Pipeline/outputs/report_ko.md` |
| `DeltaAssociationExtension/` | Concurrent dimension-change association extended from ΔwRC+ to ΔWAR and ΔOPS | `DeltaAssociationExtension/outputs/` |
| `LM_experiments/` | RQ2: LM/agent arms vs regression and KNN anchors on the 2024→2025 holdout | `LM_experiments/outputs/agent_pilot_v2/PILOT_V2_REPORT.md` |
| `paper/` | Manuscript sources (`.tex`, Korean `.md`) and figures | — |

Headline post-submission findings are summarized in [Post-submission findings](#post-submission-findings) below.

---

## Citation

```bibtex
@misc{choi2026baseball2vec,
  title   = {Baseball2Vec: Learning the Five-Tool Embedding Space for MLB Hitters},
  author  = {Jaeseok Choi},
  year    = {2026},
  note    = {Presented at Saberseminar 2026, Chicago, IL}
}
```

---

## Reproducing the Results

```bash
git clone <repo-url>
cd baseball2vec
pip install -r requirements.txt
python run_all.py
```

All figures are written to `outputs/figures/` and all tables to `outputs/tables/`.

### Data availability

Raw batting stats are fetched from FanGraphs via [pybaseball](https://github.com/jldbc/pybaseball). FanGraphs applies anti-scraping measures that may block server environments — in this project's environment the block was confirmed to be a Cloudflare challenge (HTTP 403), reproducible even through a headless Playwright browser, so it is not a user-agent issue. If step 01 fails:

1. Run step 01 on a local machine that can reach FanGraphs, which caches the raw data to `data/raw/batting_stats_2021_2025.csv`.
2. Copy that CSV into the repo and re-run from step 02: `python run_all.py --from 02`.

Alternatively, use a **manual FanGraphs leaderboard export** and convert it with
`ScalingRevision/build_raw_from_fangraphs_export.py`, which writes the same
`data/raw/batting_stats_2021_2025.csv` cache path (deduping repeated ID columns, filtering to
2021–2025 with PA ≥ 100, and resolving the one known Name+Season collision). The required export
columns are listed in `ScalingRevision/manual_export_columns.md`.

Two processed datasets are committed and can be used without any re-fetch or re-training:

- `data/processed/v3_results.csv` — 2,309 player-seasons, 2021–2025, the original submission dataset. Step 04 runs directly against it: `python run_all.py --from 04`.
- `data/v3_results_clean.csv` — the Phase 0 audited version (same 2,309 rows; the one Name+Season collision relabeled). **Every revision phase after Phase 0 reads this file, not the original.**

### What is *not* committed

To keep the repo a reasonable size, the following are gitignored. All of them are regenerable from
the committed scripts:

| Excluded | Size | Regenerate with |
|---|---|---|
| `data/raw/` FanGraphs exports | ~36 MB | `scripts/01_build_dataset.py` or the manual-export route above |
| `final_academic_revision/**/tools/`, `**/.conda-tectonic/` (vendored Tectonic + Python toolchain) | ~400 MB | that run's `run_all.sh` re-downloads it |
| Full-resolution figure grids (`v3_scatter_grid.png`, `v3_uncertainty.png`, per-player `v3_radar_*.png`) | ~38 MB | the owning folder's plotting script |
| Per-row bootstrap and matching tables (>1 MB CSVs) | ~30 MB | the owning folder's analysis script |
| Rendered manuscript page images (`page_renders/`, `rendered_pages/`) | — | `pdftoppm` on the committed PDF |

Summary tables, small figures, and every script are committed.

---

## Data Sources

| Source | Access | Notes |
|--------|--------|-------|
| FanGraphs batting leaderboards | via `pybaseball.batting_stats()` | qual ≥ 100 PA, seasons 2021–2025 |
| Statcast (EV, Barrel%, HardHit%) | included in FanGraphs export | no separate fetch needed |
| Chadwick Register | `academic_validation_experiments/` only | stable player IDs and birth year, for identity disambiguation |

FanGraphs data is used for non-commercial research. See [FanGraphs Terms of Service](https://www.fangraphs.com/about/terms-of-service).

---

## Repository Structure

```
baseball2vec/
├── README.md
├── REVISIONS.md                    # dated log of the Phase 0–5 revision pass
├── requirements.txt
├── run_all.py                      # single-command orchestrator (original pipeline)
├── data/
│   ├── raw/                        # gitignored; generated by step 01
│   ├── processed/v3_results.csv    # 2,309 player-seasons, original submission dataset
│   └── v3_results_clean.csv        # Phase 0 audited dataset (input to all later phases)
├── src/baseball2vec/
│   ├── data.py                     # fetch, cache, Bayesian stabilization
│   ├── tools.py                    # feature groups, Z-Score, 20-80 scaling, pentagon area
│   ├── baselines.py                # Z-Score avg and PCA tool scores, correlation table
│   ├── joint_vae.py                # JointVAE model + losses + train()
│   ├── roi.py                      # ΔTool dataset, LR/RF regression, counterfactual ROI
│   └── viz.py                      # all plotting functions
├── scripts/
│   ├── 01_build_dataset.py         # data fetch / load → preprocess
│   ├── 02_baseline_comparison.py
│   ├── 03_joint_vae.py             # train VAE → v3_results.csv
│   ├── 04_roi_analysis.py          # ΔTool regression → figures + tables
│   ├── 05_scatter_and_radar.py
│   └── revision/                   # phase0..phase5 revision scripts
├── outputs/
│   ├── figures/ tables/            # original pipeline outputs (gitignored)
│   └── revision/<phase>/           # committed revision results
├── paper/                          # manuscript .tex / .md + figures
├── tests/test_reproducibility.py
│
│   # ── isolated post-submission analysis folders ──
├── academic_validation_experiments/
├── final_academic_revision/run_20260808_054320/
├── final_academic_revision/run_20260808_055335/
├── ScalingRevision/
├── SD10Pipeline/
├── DeltaAssociationExtension/
└── LM_experiments/
```

---

## Abstract Claims → Scripts & Outputs

| Abstract claim | Script | Output |
|---|---|---|
| Pentagon-WAR correlation = 0.81 (Z-Score) | `scripts/02_baseline_comparison.py` | `outputs/tables/correlations.csv` |
| Pentagon-WAR correlation = 0.81 (Z-Score, full 3-way) | `scripts/03_joint_vae.py` | `outputs/tables/v3_summary.txt` |
| ΔTool → ΔwRC+ LR R² = 0.71 (CV) | `scripts/04_roi_analysis.py` | `outputs/tables/lr_results.csv` |
| Buxton Contact ROI = +14.0 wRC+ | `scripts/04_roi_analysis.py` | `outputs/tables/individual_roi.csv` |
| Tool tradeoff matrix (Σ off-diagonals) | `scripts/03_joint_vae.py` | `outputs/figures/v3_covariance_heatmap.png` |
| Z-Score vs PCA vs JointVAE comparison | `scripts/03_joint_vae.py` | `outputs/figures/v3_correlation_heatmap.png` |

---

## Reproducing the Revision Pipeline (Phases 0–5)

A methodological/reproducibility revision pass (data audit, leakage-resistant CV, confidence
intervals, ablations, external pilot, repro packaging) lives in `scripts/revision/`, with a running
dated log in [`REVISIONS.md`](REVISIONS.md). It runs entirely against the committed, already-processed
`data/processed/v3_results.csv` — it does **not** require re-fetching raw FanGraphs data — except
where a phase explicitly notes that raw data is required and unavailable (Phase 3a; see
`REVISIONS.md`).

```bash
python scripts/revision/phase0_data_audit.py        # -> data/v3_results_clean.csv (audited data)
python scripts/revision/phase1_grouped_cv.py         # player-grouped CV for the ROI regression
python scripts/revision/phase2_confidence_intervals.py  # cluster-bootstrap CIs for Table 1
python scripts/revision/phase3_ablations.py          # shrinkage + permutation ablations
python scripts/revision/phase4_external_pilot.py     # scouting-grade pilot comparison
python scripts/revision/phase5_repro_package.py      # environment + fold-assignment audit trail
```

Every phase after Phase 0 reads `data/v3_results_clean.csv` (not the original `v3_results.csv`).
All outputs are written to `outputs/revision/<phase_name>/`. Random seed 42 is used throughout
(`GroupKFold`/`KFold` shuffling, `RandomForestRegressor`, the Phase 2 bootstrap RNG); Phase 1's
exact player→fold assignments are saved to
`outputs/revision/phase5_reproducibility/phase1_fold_assignments.csv` for audit. The exact package
versions used to generate these outputs are recorded in
`outputs/revision/phase5_reproducibility/environment_used.txt` — note that this sandbox's installed
`pandas`/`numpy` versions drift slightly from `requirements.txt`'s pins (see that file for the
exact diff); `matplotlib` and `pybaseball` match the pinned versions exactly.

---

## Post-submission findings

Each item links to the folder holding the code and full writeup. Nothing here has been folded back
into `paper/baseball2vector_en.tex`; proposed integration language lives in the respective
`manuscript_preview/` directories.

### The headline numbers hold up (Phases 0–3)

- **No player-level leakage.** Player-grouped `GroupKFold` R² is statistically indistinguishable from the paper's pooled estimate (ΔwRC+ 0.708 vs 0.707 LR; ~0.68 both ways for the RF). Refitting the 20–80 scaling inside each training fold changes nothing, as affine invariance predicts. The one real gap is a **forward-time holdout** (train ≤2023, test 2024→2025), where the RF drops 0.679 → 0.636 on ΔwRC+ — a more honest estimate for genuine forecasting.
- **Table 1 now has confidence intervals.** Player-level cluster bootstrap, 2,000 resamples: Z-score 0.81 [0.78, 0.83] / 0.65 [0.61, 0.68] / 0.63 [0.59, 0.66] vs WAR/wRC+/OPS. The *paired* bootstrap shows Z-score's edge over PCA is small (+0.01 to +0.02) but **significant on all three metrics**, even though the marginal CIs overlap — so the paper's "ZScore > PCA > JointVAE" ordering survives a proper uncertainty analysis. Ready-to-paste table and caption are in `REVISIONS.md`.
- **Shrinkage strength is not load-bearing.** Even halfway-to-the-mean extra shrinkage moves Table 1 correlations by ≤ 0.003.
- **Axis-permutation sensitivity extended to all three encodings.** PCA is the most permutation-sensitive; JointVAE is comparably or more stable than Z-score. Z-score's advantage should therefore be attributed to its higher correlation with value metrics, not to robustness against axis ordering.
- **One data-quality fix.** The two distinct MLB players named Max Muncy (b. 1990 LAD; b. 2002 ATH) collided in 2025. Both are kept; the rookie row is relabeled `Max Muncy (ATH)` so the `Name`-keyed transition builder in `src/baseball2vec/roi.py` cannot splice their seasons together. 1 of 2,309 rows affected; headline numbers unchanged.

### Forecast-safe prediction beats a calibrated scalar baseline — modestly (`final_academic_revision/`)

On a fixed 2024→2025 holdout of 358 stable-ID players, a forecast-safe B2V Ridge reached **MAE 18.027** vs **19.220** for a calibrated current-wRC+ Ridge. The paired player-bootstrap 95% CI on the 1.193 improvement is **[0.403, 2.012]** — distinguishable in this sample, but only ~6.2% of the baseline MAE. Adding current wRC+ on top of the B2V context gives no clear further gain. Both runs pin input SHA-256 hashes and leave `paper/` byte-identical.

### Changing the 20–80 scale flips the personalized recommendation (`ScalingRevision/`, `SD10Pipeline/`)

Replacing `MinMaxScaler(20,80)` with the FanGraphs convention (mean=50, SD=10) leaves the *aggregate* tool-importance ranking for ΔwRC+ unchanged (Power > Contact > Discipline > Defense > Speed). But the per-player RF10 "which tool should this player work on" answer moves substantially: **Contact-best drops from 213/348 (61%) to 98/348 (28%)**, and Power becomes the majority recommendation at 250/348 (72%). This is a genuine retrained result, not a CSV-reuse artifact. Pentagon-area correlations decrease modestly under the new scale due to tail clipping — see `ScalingRevision/outputs/report_ko.md` for why that is not evidence the new scale is worse.

### Dimension-change association differs by target metric (`DeltaAssociationExtension/`)

Extending the paper's ΔTool → ΔwRC+ association to ΔWAR and ΔOPS (1,414 transitions, 567 players, 2,000-resample player-cluster bootstrap, R² ≈ 0.71 for all three targets):

- **Contact, Power, and Plate Discipline** have CIs excluding zero for **all three** targets.
- **Defense and Speed** register only on **ΔWAR** — their CIs include zero for both ΔwRC+ and ΔOPS.

That split is what one would expect if the embedding is behaving sensibly: wRC+ and OPS are batting-only, while WAR is the only target that prices in defense and baserunning.

### LM agents do not yet beat the regression anchor (`LM_experiments/`)

A **six-record diagnostic pilot** (not publication-grade; one agent instance handled all six records per arm, so within-arm observations are not independent). Agent D v2.1, given comparable-lookup tools, lands at wRC+ MAE 9.774 / OPS MAE 0.0281 — numerically almost identical to the extended linear-regression anchor (9.760 / 0.0281), so it does not demonstrate incremental LM value. Leakage audit: 0/6 direct subject leakage in returned comparables, 0 ID-format violations. The clearest failure mode is decliners — all arms miss both WAR directions on those cases.

### Independent scouting comparison is directional only (Phase 4)

Six recently-graduated top prospects, rookie-season B2V vectors vs pre-MLB Baseball America grades on the three mappable axes (Contact↔Hit, Power↔Power, Speed↔Run). Speed agreed most consistently; Power disagreed most consistently — plausibly a real distinction between scouted *raw* power and B2V's *in-game-production* Power tool. With six players and two non-commensurable 20–80 scales this is hypothesis-generating only.

---

## Known Limitations

- This is research code written for a conference presentation, not a production system.
- Raw data re-fetch requires a machine with FanGraphs access (see [Data availability](#data-availability) above).
- The JointVAE training (~4,000 epochs on CPU) takes approximately 10–20 minutes. Results are deterministic given the fixed seed (42), but may differ slightly across hardware due to floating-point non-associativity in PyTorch.
- The 2025 season data (included in the 5-year window) is a partial season as of the submission date; results may shift slightly once the full season is available.
- **Raw constituent statistics and pre-min-max dimension scores are not archived**, so several checks could not be run and are reported as `not_run_missing_raw` rather than approximated: raw-statistic Ridge, the literal shrinkage-threshold ablation, leave-one-statistic-out, Defense reconstruction, and JointVAE multi-seed analysis. This is unresolved scope, not a negative result.
- **FanGraphs redistribution is an open item.** `data/processed/v3_results.csv` and `data/v3_results_clean.csv` are *derived* statistics (Z-scores, PCA components, pentagon areas), not a republication of FanGraphs' own tables — but whether a derived non-commercial research dataset at this transformation level may be committed to a public repo has not been verified against FanGraphs' Terms of Service. Confirm before making the repo public.
- The post-submission folders were run in different environments (`requirements.txt` pins, a dedicated `baseball2vector` conda env, and a Python 3.12 env with NumPy 2.5 / pandas 3.0 / scikit-learn 1.9). Exact versions per run are recorded in each folder's `environment_used.txt` or `logs/`.

---

## Contact

**Jaeseok Choi** — `jaeseok.choi@skku.edu`
Advisor: **Prof. Jangwon Lee**, I2SLAB, Sungkyunkwan University
