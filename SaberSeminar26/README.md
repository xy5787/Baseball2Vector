# Baseball2Vector — Saberseminar 2026

Baseball2Vector represents an MLB hitter as a five-dimensional profile aligned with the traditional scouting vocabulary: **Contact, Power, Speed, Defense, and Plate Discipline**. The project asks whether a compact, interpretable representation can preserve information about player value while distinguishing players who arrive at similar overall production through different skill combinations.

This directory contains the original Saberseminar 2026 analysis, manuscript materials, and the post-submission validation work that directly tests or extends its claims. The separate [`../SSAC27/`](../SSAC27/) package evaluates next-season forecasting and comparable-player retrieval over rolling-origin test seasons.

> **Presented at:** Saberseminar 2026, August 29–30, Chicago
>
> **Author:** Jaeseok Choi, I2SLAB, Sungkyunkwan University
>
> **Advisor:** Prof. Jangwon Lee

## Research questions

1. Can 22 batting, baserunning, and fielding statistics be summarized as five recognizable baseball tools?
2. Does the resulting profile align with established value metrics while retaining information that a single aggregate statistic hides?
3. Are year-to-year changes in the five dimensions associated with changes in player outcomes?
4. How sensitive are the conclusions to encoder choice, scaling, player identity, and validation design?

## Data and representation

The original analysis covers **2,309 MLB player-seasons from 2021–2025**, with a minimum of 100 plate appearances. Twenty-two direction-adjusted statistics are assigned to the five tool groups, stabilized toward their season means, and standardized within season.

Three encodings are compared:

- **Z-score grouping:** the mean standardized value within each tool group.
- **PCA:** a one-component linear representation for each tool group.
- **JointVAE:** a variational representation with a learned full covariance structure.

Tool profiles are displayed on a 20–80 scale. The original submission used min–max scaling; the validation package also reports the conventional mean-50, SD-10 transformation. Forecasting analyses use within-input-season standardized features rather than pooled display grades.

## Main findings

- The Z-score pentagon area correlates **0.81 with WAR**, **0.65 with wRC+**, and **0.63 with OPS**. Player-cluster bootstrap intervals are reported in [`REVISIONS.md`](REVISIONS.md).
- The original concurrent-change model explains approximately **71% of pooled variation in ΔwRC+**. This is an association result, not a causal estimate or a deployment-ready forecast.
- Contact, Power, and Plate Discipline changes are associated with ΔwRC+, ΔWAR, and ΔOPS; Defense and Speed changes register clearly only against ΔWAR, the target that includes fielding and baserunning value.
- On a fixed 2024→2025 holdout of 358 stable-ID players, forecast-safe B2V Ridge records **18.027 MAE**, versus **19.220** for calibrated current-wRC+ Ridge. The paired improvement is **1.193 wRC+ points** with a 95% player-bootstrap interval of **[0.403, 2.012]**.
- Z-score, PCA, and JointVAE produce similar predictive conclusions. The practical signal is primarily in the five-tool grouping; the more complex encoder does not establish a meaningful forecasting advantage.
- Changing the display scale can materially change individual “best tool to improve” recommendations even when aggregate rankings remain stable. Personalized recommendations should therefore be treated as scale-dependent descriptive outputs.

## Repository layout

```text
SaberSeminar26/
├── README.md
├── REVISIONS.md
├── requirements.txt
├── run_all.py
├── data/
│   ├── processed/v3_results.csv
│   └── v3_results_clean.csv
├── src/baseball2vec/            # data, tools, baselines, VAE, ROI, visualization
├── scripts/                     # original pipeline and revision phases
├── tests/test_reproducibility.py
├── paper/                       # English/Korean manuscript sources and figures
├── outputs/                     # generated figures and tables
├── ScalingRevision/             # min–max vs mean-50/SD-10 comparison
├── SD10Pipeline/                # pipeline rerun under the SD-10 display scale
├── DeltaAssociationExtension/   # ΔwRC+, ΔWAR, and ΔOPS associations
└── final_academic_revision/
    └── run_20260808_055335/      # forecast-safe validation and audit artifacts
```

## Quick start

From the repository root:

```bash
cd SaberSeminar26
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

The pipeline runs four stages:

1. fetch or load the raw FanGraphs data and preprocess it;
2. compute the Z-score and PCA baselines;
3. train the JointVAE and assemble the player-season results;
4. run the year-to-year tool-change analysis.

Figures and tables are written to `outputs/figures/` and `outputs/tables/`.

To use the committed processed data without fetching or retraining the encoders:

```bash
python run_all.py --from 04
```

Run the reproducibility checks with:

```bash
python -m pytest tests/test_reproducibility.py -v
```

## Data availability

The committed files `data/processed/v3_results.csv` and `data/v3_results_clean.csv` contain the original and identity-audited derived player-season tables. The audit retains both MLB players named Max Muncy and disambiguates their 2025 records.

Raw FanGraphs exports are intentionally not committed. A full run expects:

```text
data/raw/batting_stats_2021_2025.csv
```

Step 01 attempts to obtain the data through `pybaseball`. If FanGraphs blocks an automated request, export the leaderboard manually and use `ScalingRevision/build_raw_from_fangraphs_export.py`; required fields are listed in `ScalingRevision/manual_export_columns.md`. Users are responsible for complying with the source's terms of use and redistribution rules.

Some large intermediate tables and local toolchains are also excluded from Git. The final revision folder preserves small result tables, figures, logs, configuration, and checksums, but its complete rerun additionally requires the gitignored stable-ID player-season input documented in that folder.

## Validation and extensions

### Methodological revision

[`REVISIONS.md`](REVISIONS.md) records the player-identity audit, player-grouped cross-validation, clustered bootstrap intervals, scaling checks, ablations, and reproducibility metadata. Revision scripts live under `scripts/revision/` and write to `outputs/revision/`.

### Display-scale revision

[`ScalingRevision/`](ScalingRevision/) compares the original min–max grades with a mean-50, SD-10 display scale. [`SD10Pipeline/`](SD10Pipeline/) reruns the analysis using only the revised scale. These folders show why a display transformation that leaves rank information intact can still alter pentagon geometry and individual recommendations.

### Multi-target change associations

[`DeltaAssociationExtension/`](DeltaAssociationExtension/) applies the same player-clustered bootstrap framework to ΔwRC+, ΔWAR, and ΔOPS. It is a concurrent-change analysis and should not be interpreted as evidence that changing a tool causes a future outcome.

### Forecast-safe academic revision

[`final_academic_revision/run_20260808_055335/`](final_academic_revision/run_20260808_055335/) contains the held-out prediction, temporal-scaling audit, deterministic profile matching, PA-stratified stability analysis, and manuscript-ready artifacts. The run uses stable player IDs, training-only preprocessing, player-grouped inner validation, and 2,000 player-cluster bootstrap replications.

## Interpretation and limitations

- The five tools are statistical summaries of major-league outcomes, not traditional scouting grades of raw physical ability.
- Pentagon area is useful for visualization but depends on axis ordering and the chosen display transformation.
- The year-to-year change analysis is associative and can reflect playing time, injury, aging, role, and measurement error.
- Forecast samples require at least 100 PA in both the input and outcome seasons, so results are conditional on continued major-league participation at that threshold.
- The JointVAE is the most computationally expensive component and may show small hardware-dependent floating-point differences despite fixed random seeds.
- This is conference research code, not a production player-evaluation system.

## Citation

```bibtex
@misc{choi2026baseball2vector,
  title  = {Baseball2Vector: Learning the Five-Tool Embedding Space for MLB Hitters},
  author = {Choi, Jaeseok},
  year   = {2026},
  note   = {Presented at Saberseminar 2026, Chicago, Illinois}
}
```

## Contact

Jaeseok Choi — `jaeseok.choi@skku.edu`

I2SLAB, Sungkyunkwan University
