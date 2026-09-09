# Baseball2Vector — SSAC27 experiment package

Every number below is produced by a script in `SSAC27/scripts/`, listed under
each table with its exact run command. Nothing here was computed by hand.

**Three targets are run everywhere.** wRC+ is the published one, but it is a
batting-only index that prices neither Defense nor Speed — two of the five tools —
so evaluating the tool vector only on wRC+ is a structurally unfavourable test.
WAR prices both. WAR is also reported per 600 PA, because `corr(WAR_t, PA_t)`
is 0.674 against 0.449 for wRC+, so predicting counting WAR is substantially a
playing-time forecast. Because wRC+ MAE lives on a ~19 scale and WAR MAE on a
~1.1 scale, tables also carry each model's MAE **as a percentage of that target's
own scalar baseline**.

Configuration adopted before any test score was inspected, and unchanged since:
seed 42; 2,000 bootstrap replicates clustered on the focal player; Ridge with
train-only median imputation and train-only standardization; alpha chosen by
player-grouped `GroupKFold(n_splits=5)` inside the training window only, over
`np.logspace(-3, 4, 50)`, ties broken toward the smaller alpha. Rationale and
rejected alternatives: [`decisions_log.md`](decisions_log.md).

Reproduce everything:

```bash
python SSAC27/scripts/run_all.py
```

---

## Data

Three populations, kept deliberately apart. Conflating them is what produced the
two data defects this package had to fix.

| | rows | what it is | used by |
|---|---|---|---|
| **Anchored cohort** | 2,286 | 2021–2025, PA ≥ 100, archived tool scores, Chadwick stable IDs | Tasks 1–3 (focal players, features, targets) |
| **History pool** | 5,293 | 2019–2025, **every season played, no PA minimum** | Marcel's lookback only |
| **Extended cohort** | 3,070 | 2019–2025, PA ≥ 100, tool scores **recomputed** from raw | Task 6 |

| | |
|---|---|
| Targets | next-season `wRC+`, `WAR`, `WAR per 600 PA` |
| Transitions | 1,414 anchored (3 origins) · 1,981 extended (5 origins) |
| Tool scores | archived `ZScore_*` 20–80 columns re-standardized inside the input season (Tasks 1–3); recomputed ZScore + PCA (Task 6) |
| Join key | Chadwick `player_id` (Tasks 1–3) · `"fg:" + PlayerId` (Task 6, verified to agree on all 2,286 rows) |

### Source files

| File | Contents |
|---|---|
| `data/raw/batting_stats_2021_2025.csv` | 2021–2025, PA ≥ 100, 462 columns — the qualified cache |
| `data/raw/fangraphs-leaderboards (1).csv` | 2021–2026, **no PA minimum**, 465 columns |
| `data/raw/batting_stats_2019_2020.csv` | 2019–2020, **no PA minimum**, 465 columns |
| `data/processed/transitions.csv` | public Tasks 1–5 transition table, including archived scores, stable IDs, and constituent features |

2026 is dropped everywhere: it is later than every outcome season in the study,
so it could only ever be a forward reference. SHA-256 hashes of every input are
recorded in each task's `environment.json` and consolidated under `results/provenance/`.

**Resolved provenance limitation.** Earlier documentation (preserved in Git history)
reported that constituent statistics were unavailable. The recovered export contained
all 22 statistics, so Task 1 ran the raw-statistic control. Those forecast-safe
constituent features are now embedded in `data/processed/transitions.csv`; the source
export remains local under `data/raw/`. See `results/provenance/decisions_log.md` D1.

### The 22 constituent statistics

| Tool | Statistics | Sign-flipped |
|---|---|---|
| Contact | `Contact%`, `K%`, `AVG`, `xBA`, `SwStr%` | `K%`, `SwStr%` |
| Power | `ISO`, `SLG`, `HardHit%`, `Barrel%`, `maxEV`, `EV`, `HR/FB` | — |
| Speed | `Spd`, `BsR`, `UBR`, `wSB` | — |
| Defense | `Def`, `Fld` | — |
| Discipline | `BB%`, `O-Swing%`, `Swing%`, `BB/K` | `O-Swing%`, `Swing%` |

Each is Bayesian-stabilized toward its **own season's** league mean, z-scored
**inside the input season**, then direction-corrected — the identical chain the
tool scores are built from. The five tool scores are therefore exactly the five
group means of these 22 columns, so **M1's feature space is a fixed linear
projection of M2's** and the M1-vs-M2 contrast isolates the grouping alone.

### Integrity checks asserted on every run

| Check | Result |
|---|---|
| Archived rows joined to raw export | 2,286 / 2,286, zero loss |
| Archived rows joined on `"fg:" + PlayerId` | 2,286 / 2,286, PA identical on every row |
| Marcel reliability, 600-PA three-season regular | 0.850 measured vs 0.857 theoretical (n = 268) |
| Recomputed (Task 6) vs archived tool scores | r ≥ 0.99984 on all five tools |
| Recomputed vs archived tool scores (within-season Pearson r) | Contact 0.99999 · Power 1.00000 · Discipline 0.99998 · Defense 0.99987 · Speed 0.99994 |
| Fold C sample sizes | n_train 1,056 · n_test 358 |
| Published Finding 2 reproduction (published alpha grid) | M0 = 19.2195, M1 = 18.0267 — matches the manuscript to 4 dp |
| Future-season predictors | none (`assert_no_future_columns`) |
| NaN in the modelling matrix | 0 |

---

## The three targets compared

wRC+ is batting-only. It prices neither Defense nor Speed, two of the five tools,
so the published evaluation was a structurally unfavourable test of the tool
structure. WAR prices both. The SaberSeminar result `../SaberSeminar26/results/delta_association_coefficients.csv` reports the matching asymmetry: Defense and Speed have CIs excluding zero
for ΔWAR but not for ΔwRC+ or ΔOPS.

Adding WAR was expected to help the five-tool vector. **It does not.** The result
is worth stating up front because it runs against the motivating intuition.

### Table 0.1 — Pooled MAE as a percentage of each target's own scalar baseline

Three rolling-origin folds, wide alpha grid. Lower is better; 100 = no better
than predicting from the current-season value of the same quantity.

| Model | wRC+ | WAR (counting) | WAR / 600 PA |
|---|---|---|---|
| M0 scalar baseline | 100.0 | 100.0 | 100.0 |
| **M1 B2V 5 tools** | **94.1** | **104.0** | **97.2** |
| M2 constituent stats | 93.3 | 100.2 | 95.3 |
| M3 scalar + age + PA | 99.0 | 99.0 | 97.7 |
| M4 B2V + age + PA | 93.4 | 100.7 | 95.9 |
| M5 constituents + age + PA | 92.6 | 98.5 | 95.1 |

Absolute pooled MAE: wRC+ M0 19.704 / M1 18.533; WAR M0 1.176 / M1 1.223;
WAR per 600 M0 1.614 / M1 1.569.

### Table 0.2 — Pooled paired differences (positive = candidate better)

| Contrast | wRC+ | WAR (counting) | WAR / 600 PA |
|---|---|---|---|
| M1 vs M0 | **+1.171 [+0.706, +1.631]\*** | **−0.048 [−0.080, −0.016]\*** | **+0.046 [+0.013, +0.077]\*** |
| M2 vs M0 | +1.316 [+0.772, +1.836]\* | −0.003 [−0.036, +0.032] | +0.076 [+0.032, +0.119]\* |
| M1 vs M2 | −0.145 [−0.379, +0.085] | **−0.045 [−0.070, −0.021]\*** | **−0.030 [−0.060, −0.002]\*** |
| M4 vs M3 | +1.103 [+0.685, +1.514]\* | −0.020 [−0.043, +0.003] | +0.030 [+0.001, +0.058]\* |

> Source: `SSAC27/scripts/task2_rolling_origin.py` → `results/task2/fold_performance.csv`, `results/task2/paired_differences.csv`

### What the WAR targets say

**The motivating hypothesis is not supported.** If the five-tool structure paid
off on a target that prices Defense and Speed, B2V's edge over the scalar should
be *larger* on WAR than on wRC+. It is smaller: −5.9% of baseline on wRC+, −2.8%
on WAR per 600, and **+4.0% — worse than the baseline — on counting WAR**. The
tools' advantage is largest on the target that ignores two of them.

**On counting WAR the five tools lose to the scalar in all three folds**
(sign-consistent, pooled −0.048, CI excludes zero). The mechanism is not
mysterious: the tool scores are within-season z-scores carrying no playing-time
information at all, and counting WAR is substantially a playing-time forecast
(`corr(WAR_t, PA_t) = 0.674`). Current WAR encodes both quality and opportunity;
the tool vector encodes only quality. Adding PA back (M4 vs M1) recovers most of
the gap, which confirms the diagnosis.

**On WAR per 600 the tools do beat the scalar**, +0.046 [+0.013, +0.077], and in
all three folds. So the five-tool profile genuinely predicts future *rate* of
value better than last season's rate of value does. That is the honest positive
version of the WAR claim, and it is weaker than the wRC+ version.

**The constituent-statistics gap widens on WAR, and is now significant on both
WAR targets.** M1 vs M2 is −0.145 (not significant) on wRC+ but −0.045\* on
counting WAR and −0.030\* on WAR per 600. In relative terms: −0.7% of baseline on
wRC+, −1.9% on WAR per 600, −3.8% on counting WAR.

**A structural caveat that makes this less damaging than it looks.** WAR is
literally assembled from batting, baserunning, fielding and positional
components, and `Def` and `BsR` are constituent statistics in M2. So for a WAR
target M2 contains near-components of the outcome, while M1 averages `Def` with
`Fld`, and `BsR` with `Spd`, `UBR` and `wSB`, throwing away exactly the
decomposition WAR consumes directly. The M1 < M2 gap on WAR is therefore partly
mechanical and should not be read as evidence that the tool axes are poorly
constructed. It *is* fair to say the compression costs more when the target
prices the compressed components separately. (These are all input-season values
predicting a later season, so there is no leakage — only a structural advantage.)

**What the paper should say.** Not "the five-tool vector also predicts WAR". The
supportable statements are: (i) the profile beats a same-quantity scalar baseline
on wRC+ and on WAR per 600, but not on counting WAR; (ii) counting-WAR prediction
needs playing-time information the profile does not contain and was never meant
to; (iii) the tool compression costs more on WAR than on wRC+, partly because
WAR consumes the compressed components directly.

---

## Task 1 — Constituent-statistics control

**Question.** Is compressing 22 constituent statistics into 5 tool scores better
than feeding the statistics in directly?

**Design.** One time split (train 2021→22, 2022→23, 2023→24, n = 1,056; test
2024→25, n = 358), one target, one CV protocol, one alpha grid, six models.

| Model | Features |
|---|---|
| M0 | current wRC+ |
| M1 | B2V 5 tool scores |
| M2 | all 22 constituent statistics |
| M3 | M0 + age + PA |
| M4 | M1 + age + PA |
| M5 | M2 + age + PA |

### Table 1.1 — Held-out test performance (2024→25, n = 358)

Primary, wide alpha grid `np.logspace(-3, 4, 50)`:

| Model | k | α | MAE | MAE 95% CI | RMSE | RMSE 95% CI |
|---|---|---|---|---|---|---|
| M0 current wRC+ | 1 | 26.83 | 19.233 | [17.641, 20.896] | 24.899 | [22.713, 27.212] |
| M1 B2V 5 tools | 5 | 0.001 | 18.027 | [16.514, 19.681] | 23.705 | [21.609, 25.980] |
| **M2 constituent stats** | **22** | **1.389** | **17.746** | **[16.278, 19.257]** | **22.961** | **[21.157, 24.911]** |
| M3 wRC+ + age + PA | 3 | 100 | 19.182 | [17.576, 20.756] | 24.668 | [22.447, 26.813] |
| M4 B2V + age + PA | 7 | 37.28 | 18.048 | [16.551, 19.653] | 23.567 | [21.477, 25.690] |
| M5 constituents + age + PA | 24 | 100 | 17.762 | [16.325, 19.259] | 22.955 | [21.026, 24.943] |

Secondary, the published 6-value alpha grid `[0.01, 0.1, 1, 10, 100, 1000]`
(this variant is what pins the reproduction anchor):

| Model | k | α | MAE | MAE 95% CI | RMSE |
|---|---|---|---|---|---|
| M0 current wRC+ | 1 | 10 | 19.220 | [17.618, 20.892] | 24.896 |
| M1 B2V 5 tools | 5 | 0.01 | 18.027 | [16.514, 19.681] | 23.705 |
| M2 constituent stats | 22 | 1 | 17.775 | [16.301, 19.287] | 22.978 |
| M3 wRC+ + age + PA | 3 | 100 | 19.182 | [17.576, 20.756] | 24.668 |
| M4 B2V + age + PA | 7 | 10 | 18.028 | [16.528, 19.634] | 23.563 |
| M5 constituents + age + PA | 24 | 100 | 17.762 | [16.325, 19.259] | 22.955 |

> Source: `SSAC27/scripts/task1_constituent_control.py` → `results/task1/model_comparison.csv`
> ```bash
> python SSAC27/scripts/task1_constituent_control.py
> ```

### Table 1.2 — Paired differences, focal-player-clustered bootstrap (2,000 reps)

Positive = candidate better. Wide alpha grid:

| Candidate | Baseline | ΔMAE | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| M1 B2V 5 tools | M2 constituent stats | −0.281 | [−0.877, +0.288] | **no** |
| M1 B2V 5 tools | M0 current wRC+ | +1.206 | [+0.412, +2.019] | yes |
| M4 B2V + age + PA | M5 constituents + age + PA | −0.286 | [−0.658, +0.099] | **no** |
| M2 constituent stats | M0 current wRC+ | +1.487 | [+0.464, +2.469] | yes |
| M4 B2V + age + PA | M3 wRC+ + age + PA | +1.134 | [+0.404, +1.842] | yes |
| M5 constituents + age + PA | M3 wRC+ + age + PA | +1.420 | [+0.566, +2.261] | yes |

Published 6-value alpha grid:

| Candidate | Baseline | ΔMAE | 95% CI | CI excludes 0 |
|---|---|---|---|---|
| M1 B2V 5 tools | M2 constituent stats | −0.252 | [−0.844, +0.321] | **no** |
| M1 B2V 5 tools | M0 current wRC+ | +1.193 | [+0.403, +2.012] | yes |
| M4 B2V + age + PA | M5 constituents + age + PA | −0.266 | [−0.646, +0.117] | **no** |
| M2 constituent stats | M0 current wRC+ | +1.445 | [+0.423, +2.439] | yes |
| M4 B2V + age + PA | M3 wRC+ + age + PA | +1.154 | [+0.404, +1.880] | yes |
| M5 constituents + age + PA | M3 wRC+ + age + PA | +1.420 | [+0.566, +2.261] | yes |

> Source: `SSAC27/scripts/task1_constituent_control.py` → `results/task1/paired_differences.csv`

### Table 1.3 — UBR robustness

`UBR` is missing for 97 / 455 of the 2024 input season and 461 / 461 of 2025 in
the recovered export — an export artifact, not a real data gap. It is filled
with the input-season cohort mean in the primary run. Dropping it entirely:

| Model | k | α | MAE | MAE 95% CI | RMSE |
|---|---|---|---|---|---|
| M2b constituents, no UBR | 21 | 1.931 | 17.707 | [16.250, 19.228] | 22.941 |
| M5b constituents + age + PA, no UBR | 23 | 100 | 17.761 | [16.325, 19.252] | 22.954 |

A 0.04 wRC+ move. The conclusion does not depend on UBR.

> Source: `SSAC27/scripts/task1_constituent_control.py` → `results/task1/ubr_robustness.csv`

### What Task 1 means for the paper

**The interpretation that applies is the middle one: M1 ≈ M2.** The
five-dimensional tool vector and the full 22-statistic feature set are not
distinguishable in this sample. The point estimate favours the raw statistics by
0.25–0.29 wRC+ MAE — roughly 1.5% of the baseline error — and the confidence
interval comfortably contains zero under both alpha grids and with or without
age and PA.

**The favourable reading.** A 5-dimensional, scout-legible representation buys
the interpretability of the five-tool frame at **no measurable predictive cost**
against its own 22-dimensional source data, while both beat the calibrated
current-wRC+ baseline by a margin whose CI excludes zero. That is the stronger
version of the Finding 2 claim, not a weaker one: the compression is nearly
lossless for this task, so the tool axes are not throwing away signal.

**The unfavourable reading, stated plainly.** The tool grouping is *not*
demonstrated to be a useful regularizer. If the grouping were adding structure
beyond dimensionality reduction, M1 should have beaten M2 at n_train = 1,056
where 22 features is a real variance cost — and it does not; it trails slightly.
The honest claim is "no loss", not "a gain". Any manuscript sentence implying
that the five-tool structure *improves* prediction over its constituents is not
supported.

**A second thing Task 1 settles.** M2 also beats M0 (+1.487 [+0.464, +2.469]).
So the published Finding 2 result — that a profile beats a calibrated scalar —
is **not** an artifact of the tool encoding. Any reasonably rich description of
the input season does it. That reframes Finding 2's contribution from "our
encoding predicts better" to "profile-level information predicts better than a
scalar, and 5 interpretable dimensions capture essentially all of it."

**The WAR targets are reported in "The three targets compared" above** rather
than repeated here; on a single fold nothing on either WAR target is significant.

**Task 6 sharpens this, unfavourably.** On five rolling origins and 1,706 pooled
test cases the M1 − M2 interval no longer contains zero: −0.180 [−0.350, −0.008].
The tie reported here does not survive the wider evidence. See Task 6.

**One suspicious number, checked.** M1 selected α = 0.001, the floor of the wide
grid, and α = 0.01, the floor of the published grid. A hyper-parameter pinned at
a grid boundary usually means the grid is too narrow. Here it is benign: the
wide grid extends three decades below the published one and M1's test MAE is
identical to 4 dp at both floors (18.0267), so the ridge penalty is simply
inactive for 5 well-conditioned features. No action taken, and none needed.

**Leakage audit for this task.** Standardization and stabilization read only the
row's own season; imputation and scaling are fitted on training rows only; alpha
comes from player-grouped CV inside the training window; no predictor column
ends in `_t1`; the reproduction anchor lands on the published values in a
different Python/sklearn version. The result is not too good — M1 lost to M2 on
the point estimate — which is itself weak evidence against a leak.

---

## Task 2 — Rolling-origin multi-year evaluation

**Question.** Is the ~6% Finding 2 improvement a property of the 2025 season, or
does it repeat when the origin moves?

**Design.** Three folds. Training uses only transitions whose **input** season is
strictly earlier than the test fold's input season, so no fold trains on an
outcome it later predicts. Alpha is re-selected independently inside each fold's
own training window — no information crosses folds. All six Task 1 models run in
every fold.

| Fold | Training transitions | n_train | Test | n_test |
|---|---|---|---|---|
| A | 2021→22 | 351 | 2022→23 | 350 |
| B | 2021→22, 2022→23 | 701 | 2023→24 | 355 |
| C | 2021→22, 2022→23, 2023→24 | 1,056 | 2024→25 | 358 |
| pooled | re-fitted per fold | — | all three test seasons | 1,063 |

No fold falls below the n_train < 300 flag threshold, but **Fold A at 351 is the
thinnest and its intervals are correspondingly the widest**; the
`thin_training_set` column in the result CSV records the check.

The 1,063 pooled test rows come from only **499 distinct focal players** — 222 of
them contribute a row to all three folds. Row-level resampling would badly
understate the pooled intervals, so the bootstrap clusters on the focal player
(see `decisions_log.md` D7).

### Table 2.1 — Test MAE by fold × model

Primary, wide alpha grid:

| Model | A (2022→23) | B (2023→24) | C (2024→25) | pooled | α selected per fold (A / B / C) |
|---|---|---|---|---|---|
| M0 current wRC+ | 19.327 | 20.552 | 19.233 | 19.704 | 0.001 / 0.001 / 26.83 |
| M1 B2V 5 tools | 18.602 | 18.977 | 18.027 | 18.533 | 13.89 / 0.001 / 0.001 |
| M2 constituent stats | 18.747 | 18.682 | 17.746 | 18.388 | 193.1 / 100 / 1.389 |
| M3 wRC+ + age + PA | 18.676 | 20.675 | 19.182 | 19.514 | 5.179 / 7.197 / 100 |
| M4 B2V + age + PA | 18.058 | 19.125 | 18.048 | 18.411 | 10 / 13.89 / 37.28 |
| M5 constituents + age + PA | 18.317 | 18.681 | 17.762 | 18.252 | 138.9 / 100 / 100 |

Secondary, published 6-value alpha grid:

| Model | A | B | C | pooled |
|---|---|---|---|---|
| M0 current wRC+ | 19.327 | 20.552 | 19.220 | 19.700 |
| M1 B2V 5 tools | 18.600 | 18.977 | 18.027 | 18.533 |
| M2 constituent stats | 18.754 | 18.682 | 17.775 | 18.400 |
| M3 wRC+ + age + PA | 18.674 | 20.669 | 19.182 | 19.511 |
| M4 B2V + age + PA | 18.058 | 19.130 | 18.028 | 18.406 |
| M5 constituents + age + PA | 18.307 | 18.681 | 17.762 | 18.248 |

> Source: `SSAC27/scripts/task2_rolling_origin.py` → `results/task2/fold_performance.csv`
> ```bash
> python SSAC27/scripts/task2_rolling_origin.py
> ```

### Table 2.2 — Paired differences by fold (positive = candidate better)

Wide alpha grid; `*` marks a 95% CI excluding zero. Focal-player-clustered
bootstrap, 2,000 replicates.

| Contrast | A | B | C | pooled |
|---|---|---|---|---|
| **M1 vs M0** | **+0.725 [+0.024, +1.425]\*** | **+1.575 [+0.788, +2.413]\*** | **+1.206 [+0.412, +2.019]\*** | **+1.171 [+0.706, +1.631]\*** |
| M1 vs M2 | +0.145 [−0.154, +0.440] | −0.295 [−0.592, −0.025]\* | −0.281 [−0.877, +0.288] | −0.145 [−0.379, +0.085] |
| M2 vs M0 | +0.580 [−0.195, +1.377] | +1.870 [+1.056, +2.722]\* | +1.487 [+0.464, +2.469]\* | +1.316 [+0.772, +1.836]\* |
| M4 vs M3 | +0.618 [−0.008, +1.251] | +1.551 [+0.887, +2.249]\* | +1.134 [+0.404, +1.842]\* | +1.103 [+0.685, +1.514]\* |
| M4 vs M5 | +0.259 [−0.092, +0.616] | −0.444 [−0.701, −0.198]\* | −0.286 [−0.658, +0.099] | −0.159 [−0.351, +0.026] |

> Source: `SSAC27/scripts/task2_rolling_origin.py` → `results/task2/paired_differences.csv`

### Table 2.3 — Sign consistency across folds A, B, C

The headline output of Task 2.

| Contrast | Same sign in all 3 folds? | Folds where CI excludes 0 |
|---|---|---|
| **M1 vs M0** | **yes (candidate better ×3)** | **3 / 3** |
| M1 vs M2 | **no** (A favours M1, B and C favour M2) | 1 / 3 |
| M2 vs M0 | yes (candidate better ×3) | 2 / 3 |
| M4 vs M3 | yes (candidate better ×3) | 2 / 3 |
| M4 vs M5 | **no** (A favours M4, B and C favour M5) | 1 / 3 |

Identical pattern under the published 6-value alpha grid.

> Source: `SSAC27/scripts/task2_rolling_origin.py` → `results/task2/sign_consistency.csv`

### What Task 2 means for the paper

**The abstract may say "across three temporal holdouts" — for the B2V-vs-scalar
claim only.** M1 beats M0 in every fold, with the 95% CI excluding zero in every
fold, and pooled over 1,063 test cases from 499 players at
+1.171 wRC+ MAE [+0.706, +1.631]. The published 2024→25 result is not a 2025
artifact. Pooling also tightens the interval considerably (width 0.92 vs 1.61 in
Fold C alone), which is the main statistical gain from this task.

**The same sentence may *not* be written about the tool encoding versus its
constituents.** M1-vs-M2 flips sign: Fold A favours the 5 tools (+0.145, not
distinguishable), Folds B and C favour the 22 statistics (−0.295, CI excludes
zero; −0.281, does not). Pooled, −0.145 [−0.379, +0.085] — no measurable
difference. Task 1's "M1 ≈ M2" conclusion survives pooling, but it is a *tie
with a flip*, not a stable equivalence, and the one fold with a significant
result goes against the tool encoding. Reporting only the pooled tie would
overstate the stability.

**Favourable reading.** The profile-beats-scalar effect is the robust part of
this paper and it is now supported by three independent origins rather than one.
The effect is also not confined to the B2V encoding: M2-vs-M0 is sign-consistent
across all three folds too. So the finding being defended is real and repeatable.

**Unfavourable reading.** The effect size is origin-dependent — +0.725 in Fold A
versus +1.575 in Fold B, more than a factor of two. Fold A's interval only
barely clears zero ([+0.024, +1.425]) on 351 training transitions. A reader can
reasonably say the magnitude of the published 6.2% is not itself replicated;
what replicates is the direction.

**Fold B is the interesting one and should be flagged, not buried.** The
calibrated scalar baseline degrades sharply for 2023→24 (M0 20.552, M3 20.675,
both ~1.3 worse than in the other folds) while every profile model stays flat
(M2 18.682, M5 18.681). The profile advantage is therefore largest exactly where
last-season wRC+ carries forward worst. That is a substantive, checkable claim
about *when* profile information helps, and it is a better paper sentence than
an averaged effect size. It is a single fold, so it is hypothesis-generating.

**One thing that looks wrong and is not.** M0 selects α = 0.001 — the wide grid's
floor — in Folds A and B. For a one-feature Ridge that is simply unregularized
OLS, which is the correct answer when a single predictor is well conditioned;
the archived grid's floor of 0.01 gives the same MAE to 3 dp in both folds. No
model was re-tuned after seeing a test score.

**Leakage audit for this task.** `split_fold` asserts
`max(train Season_t1) <= test Season_t` in every fold, so a training row's
*outcome* season never reaches into the test fold's input season. Alpha comes
from grouped CV inside each fold's own training window. The Fold C row under the
published alpha grid is asserted to reproduce M0 = 19.2195 and M1 = 18.0267 on
every run. Fold A's advantage is the *smallest* of the three, which is what a
thin, honestly-split training set should produce.

---

## Task 3 — Marcel baseline and incremental test

**Question.** On top of what a standard projection system already squeezes out of
a player's recent record, does the five-tool profile carry residual signal?

The claim under test is **not** "B2V beats Marcel". It is "Marcel + B2V improves
on Marcel alone".

### The Marcel implementation

Parameters taken from Tango's original text, not from memory. Full sourcing and
quotations are in the module docstring of
[`SSAC27/scripts/marcel.py`](../scripts/marcel.py).

| Component | Adopted value | Source |
|---|---|---|
| Season weights | 5 / 4 / 3 on t, t−1, t−2, used raw | Tango: "Weight each season as 5/4/3" |
| Regression | league average forced in at **1200 PA = 100 × Σweights**, on the raw-weighted PA scale | Tango: "1200 PA for each player (2 weights × 600 PA)"; scale disambiguated by marcelR `num = x_av*100*sw + x_metric` |
| League baseline | recency- **and** PA-weighted mean of the season cohort's PA-weighted wRC+ | marcelR `x_av_num += lgAv_i * w_i * (pa_i + pebble)` |
| Age adjustment | at projected-season age: `1/(1+0.003(age−29))` if >29, `1+0.006(29−age)` if <29, `1` at 29; multiplicative on the rate | Tango gives the two slopes and the pivot; marcelR's `age_adjustment()` supplies the sign convention |
| Projected PA | `200 + 0.5·PA_t + 0.1·PA_{t−1}` | Tango. Computed for the record; **not** used in the rate projection and **not** used as a model feature |
| League re-centering | marcelR's per-origin-year rescale to the weight-averaged league rate | marcelR; reads only prior-season league rates, so forecast-safe |
| < 3 seasons of history | missing seasons contribute zero to both numerator and weighted PA, so regression does more work; no history at all → league average | Tango: "Any 2004 rookie with no MLB experience will project at the league average" |

wRC+ is already a league- and park-adjusted rate indexed to 100, so the
"counting stat" Marcel weights is taken to be `wRC+ × PA` and the projection is
the ratio.

### The history pool Marcel reads

Marcel treats a season absent from its lookback table as **zero PA**. That is
right for "did not play" and wrong for "played 45 times" — and the study's
PA ≥ 100 qualifier was silently turning the second into the first. The lookback
pool is therefore built separately from the modelling cohort:

| | rows | definition |
|---|---|---|
| **History pool** — what Marcel reads | 5,293 | every season actually played, 2019–2025, **no PA minimum** |
| **Modelling cohort** — focal players, tool standardization | 2,286 | 2021–2025, PA ≥ 100, archived, **unchanged** |

Because only the history pool widened, every model feature and every focal row
is untouched and the published Finding 2 anchors still reproduce exactly. See
`decisions_log.md` D12.

**Validation.** A 600-PA regular with three full seasons should have reliability
7200 / 8400 = **0.857**. Measured: **0.850** (n = 268). Before the pool was
widened it was 0.737 — the 1200-PA constant now behaves as Tango's arithmetic
says it should.

| Origin season | n | 1 season | 2 seasons | 3 seasons | mean depth | mean reliability |
|---|---|---|---|---|---|---|
| 2021 (train only) | 351 | 26 | 46 | 279 | 2.721 | 0.708 |
| 2022 (Fold A test) | 350 | 47 | 48 | 255 | 2.594 | 0.717 |
| 2023 (Fold B test) | 355 | 39 | 71 | 245 | 2.580 | 0.734 |
| 2024 (Fold C test) | 358 | 30 | 66 | 262 | 2.648 | 0.742 |

> Source: `SSAC27/scripts/task3_marcel_incremental.py` → `results/task3/marcel_coverage.csv`

### Table 3.0 — What widening the pool actually bought

| History pool | League baseline | M6 raw | **M6c calibrated** | mean depth | reliability |
|---|---|---|---|---|---|
| qualified only (PA ≥ 100, 2021–25) | definition | 19.732 | 19.110 | 2.221 | 0.722 |
| qualified only | cohort + re-centered | 19.681 | 19.125 | 2.221 | 0.722 |
| **full history (all PA, 2019–25)** | **definition** | 19.693 | **19.063** | 2.608 | 0.731 |
| full history | cohort + re-centered | 20.158 | 19.196 | 2.608 | 0.731 |

**Read the calibrated column.** Raw Marcel is sensitive to the baseline constant
because the test rows are players who qualified in the *outcome* season — a
survivor-selected group averaging about 102 wRC+ — so a projector correctly
regressing toward the league's 100 sits low on them. That is population
selection, not a defect. M6c absorbs the offset using training folds only, which
is the leakage-free correction. All four calibrated values fall within 0.13 of
each other, so **no conclusion here depends on the baseline choice.**

**Widening the pool bought 0.047 wRC+ (19.110 → 19.063), not the ~0.6 a scratch
diagnostic first suggested.** That earlier estimate was inflated: it re-centered
projections onto the focal population, which is calibration, not information. The
pool genuinely is better fed — depth 2.22 → 2.61, reliability 0.722 → 0.731 — but
the added seasons are low-PA ones that carry little weight, and the 2019–2020
backfill only reaches origins 2021 and 2022. Reported as measured.

> Source: `results/task3/marcel_variants.csv`

### Table 3.1 — Test MAE by fold × model

| Model | A (2022→23) | B (2023→24) | C (2024→25) | pooled |
|---|---|---|---|---|
| M0 current wRC+ | 19.327 | 20.552 | 19.233 | 19.704 |
| M1 B2V 5 tools | 18.602 | 18.977 | 18.027 | 18.533 |
| M6 Marcel (no fit) | 19.610 | 20.014 | 19.457 | 19.693 |
| M6c Marcel calibrated | 18.898 | 19.355 | 18.933 | 19.063 |
| **M7 Marcel + B2V** | 18.584 | 18.761 | 18.111 | **18.484** |
| M7x Marcel + B2V + age + PA | 18.191 | 18.978 | 18.148 | 18.439 |

**M6c is not decoration.** M7 − M6 gives M7 both the B2V features *and* a fitted
intercept and slope on Marcel, so part of that gap is recalibration unrelated to
B2V. M6c isolates it — worth 0.630 wRC+ pooled on its own. **M7 − M6c is the
honest incremental test.**

> Source: `SSAC27/scripts/task3_marcel_incremental.py` → `results/task3/fold_performance.csv`
> ```bash
> python SSAC27/scripts/task3_marcel_incremental.py
> ```

### Table 3.2 — Paired differences (positive = candidate better)

`*` marks a 95% CI excluding zero. Focal-player-clustered bootstrap, 2,000 reps.

| Contrast | A | B | C | pooled |
|---|---|---|---|---|
| M7 vs M6 (lenient) | +1.026\* | +1.253\* | +1.346\* | **+1.210 [+0.765, +1.680]\*** |
| **M7 vs M6c (honest incremental)** | +0.314 | +0.595\* | +0.823\* | **+0.579 [+0.288, +0.859]\*** |
| M7x vs M6c | +0.707\* | +0.377 | +0.785\* | +0.623 [+0.289, +0.968]\* |
| M6c vs M0 | +0.429 | +1.197\* | +0.299 | +0.642 [+0.189, +1.110]\* |
| **M6c vs M1** | −0.296 | −0.378 | −0.907\* | **−0.529 [−1.051, +0.017]** |
| M6 vs M1 | −1.008\* | −1.037 | −1.430\* | −1.160 [−1.790, −0.532]\* |
| M7 vs M1 | +0.017 | +0.217 | −0.084 | +0.050 [−0.231, +0.326] |

All seven contrasts keep the same sign in all three folds except M7 vs M1.

> Source: `results/task3/paired_differences.csv`, `results/task3/sign_consistency.csv`

### Table 3.3 — Pooled test cases split by Marcel's actual history depth

| Marcel seasons used | n | focal players | reliability | M1 B2V | M6c Marcel | M7 | Marcel − B2V | 95% CI |
|---|---|---|---|---|---|---|---|---|
| 1 | 116 | 116 | 0.495 | 20.208 | 22.054 | 21.087 | −1.846 | [−3.190, −0.479]\* |
| 2 | 185 | 180 | 0.650 | 20.178 | 20.630 | 20.275 | −0.452 | [−1.583, +0.675] |
| 3 | 762 | 368 | 0.787 | 17.879 | 18.227 | 17.652 | **−0.348** | [−0.992, +0.351] |

762 of 1,063 test cases now have full three-season history, against 437 before
the pool was widened. Marcel's deficit is confined to genuinely new players and
is indistinguishable from zero at full depth.

> Source: `results/task3/history_depth_strata.csv`

### Table 3.4 — Marcel on the WAR targets

Marcel projects a rate and multiplies by projected playing time, so counting WAR
is the one place Tango's `proj_pa = 200 + 0.5·PA_t + 0.1·PA_{t−1}` does real
work. The league baseline is the empirical per-PA league rate (1.874 WAR per 600
PA in 2024, running 1.833–1.900 across 2019–2025) — WAR has no definitional
centre the way wRC+ has 100. See `decisions_log.md` D19.

Pooled MAE over the three rolling-origin folds:

| Model | wRC+ | WAR (counting) | WAR / 600 |
|---|---|---|---|
| M0 scalar baseline | 19.704 | 1.176 | 1.614 |
| M1 B2V 5 tools | 18.533 | 1.223 | 1.569 |
| M6 Marcel (no fit) | 19.693 | 1.217 | 1.640 |
| M6c Marcel calibrated | 19.063 | 1.188 | 1.571 |
| **M7 Marcel + B2V** | **18.484** | **1.171** | **1.550** |
| M7x Marcel + B2V + age + PA | 18.439 | **1.147** | **1.524** |

Pooled paired differences:

| Contrast | wRC+ | WAR (counting) | WAR / 600 |
|---|---|---|---|
| **M7 vs M6c** (incremental) | **+0.579 [+0.288, +0.859]\*** | **+0.017 [+0.002, +0.030]\*** | **+0.020 [+0.002, +0.039]\*** |
| M7x vs M6c | +0.623 [+0.289, +0.968]\* | +0.041 [+0.015, +0.067]\* | +0.046 [+0.014, +0.078]\* |
| M6c vs M0 | +0.642 [+0.189, +1.110]\* | −0.012 [−0.039, +0.017] | +0.044 [+0.012, +0.079]\* |
| M6c vs M1 | −0.529 [−1.051, +0.017] | +0.035 [−0.003, +0.076] | −0.002 [−0.040, +0.037] |
| M7 vs M1 | +0.050 [−0.231, +0.326] | **+0.052 [+0.022, +0.083]\*** | +0.018 [−0.007, +0.043] |

Every one of these contrasts keeps the same sign in all three folds except
M7 vs M1 on wRC+.

> Source: `results/task3/fold_performance.csv`, `results/task3/paired_differences.csv`

### What Task 3 means for the paper

**The incremental claim survives on all three targets, which is the strongest
form it could take.** M7 − M6c is positive in 3/3 folds and the pooled CI
excludes zero for wRC+ (+0.579), counting WAR (+0.017) and WAR per 600 (+0.020).
The profile adds to a standard projection system regardless of what is being
projected.

**Marcel and B2V are equivalent on every target.** M6c − M1 pooled: −0.529
[−1.051, +0.017] on wRC+, +0.035 [−0.003, +0.076] on WAR, −0.002 [−0.040, +0.037]
on WAR per 600. Three targets, three intervals containing zero, and the sign is
not even consistent between them. **Do not write "B2V outperforms Marcel."**

**On counting WAR, Marcel adds to B2V — the reverse increment that is null
everywhere else.** M7 − M1 is +0.052 [+0.022, +0.083] on counting WAR, against
+0.050 (n.s.) on wRC+ and +0.018 (n.s.) on WAR per 600. The mechanism is clean:
Marcel's projected-PA term supplies playing-time information the tool vector
structurally cannot contain, and counting WAR is the only target that needs it.
This is a genuine, mechanistic finding about what each representation carries.

**The requested claim holds and is sign-consistent.** Adding the five tool scores
on top of a Marcel projection improves on properly recalibrated Marcel in all
three folds, pooled at **+0.579 wRC+ MAE [+0.288, +0.859]** (and +1.210
[+0.765, +1.680] against raw Marcel). Profile information is not redundant with
what a weighted three-year record encodes.

**Marcel and B2V are now statistically indistinguishable, and that is the honest
result.** M6c − M1 pooled is −0.529 **[−1.051, +0.017]** — the interval includes
zero. Before the history pool was widened this contrast excluded zero
(−0.591 [−1.088, −0.078]). Feeding Marcel properly removed the apparent B2V
advantage. **Do not write "B2V outperforms Marcel."** The supportable statement is
that the two are comparable, and that combining them beats either.

**Where B2V genuinely wins is players without history.** Table 3.3 localizes the
whole remaining difference: at one season of history B2V leads by 1.846
(CI excludes zero), at two by 0.452, at three by 0.348 (both include zero). This
is a real and defensible claim — **a single rich season of profile data
substitutes for the multi-year record Marcel needs** — and it is a more
interesting paper sentence than an averaged win.

**The reverse increment is null.** M7 vs M1 is +0.050 [−0.231, +0.326] pooled and
flips sign across folds: adding Marcel on top of B2V buys nothing measurable.
Favourably read, the five tool scores already contain what the weighted
three-year record contains. Unfavourably, the two are largely measuring the same
thing.

**Remaining honest limitation.** Even widened, this Marcel starts at 2019. A
full-history Marcel would carry a player's whole career, so origins 2021 and 2022
are still short for veterans. The direction of that residual handicap is known
and it favours B2V, so the Marcel-vs-B2V comparison should be reported as a tie
at best rather than a B2V win.

**Sanity checks that Marcel is functioning.** It beats naive carry-forward
comfortably (19.46 vs 22.35 in Fold C), beats the calibrated one-season scalar M0
(+0.642 pooled, CI excludes zero), and its reliability for a three-season
600-PA regular is 0.850 against the theoretical 0.857.

**Leakage audit.** Every Marcel input is a season ≤ t. The league baseline is the
constant 100 — no population, hence nothing to leak. Re-centering is off. The age
term is arithmetic on birth year. `assert_no_future_columns` runs on the Marcel
feature. M6 involves no fitting at all. The one population definition that would
have been most stable — "players who ever reached 100 PA in 2019–2025" — was
**rejected specifically because it references seasons later than the origin**
(`decisions_log.md` D13).

---

## Task 4 — Comparable-player retrieval

**Question.** Is a comparable found in the five-dimensional tool space more
useful than one found by matching WAR and wRC+?

For each focal player-season, retrieve k comparables, average *their* next-season
outcome, use that as the prediction. Nothing is fitted, so the two methods differ
only in the distance they search under. k ∈ {3, 5, 10}; same rolling-origin folds.

| Method | Coordinates |
|---|---|
| A scalar | WAR and wRC+, standardized within season, 2-D Euclidean |
| B B2V | the five tool scores, 5-D Euclidean |

The scalar coordinates are standardized within season before the distance is
taken — otherwise a raw 2-D distance over WAR and wRC+ is almost entirely a wRC+
distance, which would handicap method A rather than test it.

**Two candidate pools, and the distinction is load-bearing.**

| Pool | Definition | Status |
|---|---|---|
| `same_season` | the focal player's own season, as the brief specifies | **Not a forecast** — it reads the comparables' next-season outcomes, which at real forecast time have not happened. A fair A-vs-B comparison (both methods get identical information), but no number from it may be quoted as forecasting accuracy. |
| `past_seasons` | seasons strictly earlier than the focal season, whose following season is already observed | **Deployable.** This is the variant to quote for any practical claim. |

### Table 4.1 — Retrieval MAE, next-season wRC+, pooled over three folds

| Method | k=3 | k=5 | k=10 |
|---|---|---|---|
| **same_season** | | | |
| A scalar | 22.768 | 21.561 | 20.552 |
| B B2V | 22.072 | 20.894 | 20.117 |
| B − A | +0.696 [−0.350, +1.698] | +0.666 [−0.238, +1.521] | +0.435 [−0.308, +1.171] |
| **past_seasons (deployable)** | | | |
| A scalar | 22.220 | 21.239 | 20.193 |
| B B2V | 21.523 | 20.385 | **19.408** |
| B − A | +0.696 [−0.445, +1.798] | +0.853 [−0.066, +1.735] | **+0.784 [+0.109, +1.406]\*** |

### Table 4.2 — Pooled B − A by target and pool (positive = B2V better)

| Target | Pool | k=3 | k=5 | k=10 |
|---|---|---|---|---|
| wRC+ | same_season | +0.696 | +0.666 | +0.435 |
| wRC+ | past_seasons | +0.696 | +0.853 | **+0.784\*** |
| WAR (counting) | same_season | −0.064 | −0.062\* | −0.072\* |
| WAR (counting) | past_seasons | −0.056 | −0.048 | −0.030 |
| WAR / 600 | same_season | −0.029 | −0.032 | −0.058\* |
| WAR / 600 | past_seasons | −0.011 | −0.012 | −0.020 |

> Source: `SSAC27/scripts/task4_comparable_players.py` → `results/task4/retrieval_performance.csv`, `results/task4/paired_differences.csv`
> ```bash
> python SSAC27/scripts/task4_comparable_players.py
> ```

### What Task 4 means for the paper

**The one quotable result is the deployable one, and it is positive.** With
comparables drawn from past seasons and k = 10, profile retrieval beats scalar
retrieval by **+0.784 wRC+ MAE [+0.109, +1.406]**. Every wRC+ cell in the table
favours B2V, in both pools and at every k; k = 10 past-seasons is the only one
whose interval excludes zero. This is the cleanest support in the package for the
practical claim the paper wants to make — *"find me players like this one"* works
better in tool space than in WAR-and-wRC+ space.

**k matters more than the method.** Going from k = 3 to k = 10 buys 2.1–2.7 MAE
for both methods, roughly three times the B-vs-A gap. Any comp tool built on this
should average ten neighbours, not three.

**On the WAR targets B2V retrieval loses, and that is close to mechanical.**
Method A's coordinates *are* WAR and wRC+, so when the target is next-season WAR,
method A is searching on the target's own current-season value and method B is
not. The negative cells should not be read as evidence about profile similarity.
On WAR per 600, where that advantage is weaker, the gap shrinks to −0.011 to
−0.020 and no deployable cell is significant.

**Honest limitation.** Even the best retrieval MAE (19.41) is worse than the
fitted Ridge on the same folds (M1 = 18.53). Retrieval is an interpretability and
communication tool, not a better predictor.

---

## Task 5 — Encoder cross-check: structure or encoder?

**Question.** Does the result come from the five-tool *structure*, or from one
particular way of collapsing each group to a number?

All three encodings share the same grouping — the same 22 statistics assigned to
the same 5 tools — and differ only in the collapse:

| Encoding | Collapse |
|---|---|
| ZScore | group mean of the within-season z-scored constituents |
| PCA | first principal component of each group |
| JointVAE | latent means of the Cholesky full-covariance joint VAE |

All three are already archived in `v3_results.csv`, so nothing was retrained and
no encoder got a tuning advantage. Alpha is re-selected per fold per encoding.

**Reading fixed in advance:** if all three beat the scalar baseline, the result
belongs to the structure; if only one does, it belongs to that encoder and the
paper's framing has to change.

### Table 5.1 — Pooled MAE by encoding (three rolling-origin folds)

| Model | wRC+ | WAR (counting) | WAR / 600 |
|---|---|---|---|
| M0 scalar baseline | 19.704 | 1.176 | 1.614 |
| M1 ZScore 5 tools | 18.533 | 1.223 | 1.569 |
| M1 PCA 5 tools | **18.516** | 1.223 | **1.567** |
| M1 JointVAE 5 tools | 18.598 | **1.214** | 1.570 |
| M1 ZScore + age + PA | 18.411 | 1.184 | 1.548 |
| M1 PCA + age + PA | 18.392 | 1.184 | 1.546 |
| M1 JointVAE + age + PA | **18.319** | **1.167** | **1.536** |

### Table 5.2 — Does every encoding beat the scalar baseline?

| Target | Encoding | folds beating scalar | same sign | CI excludes 0 | pooled improvement |
|---|---|---|---|---|---|
| wRC+ | ZScore | **3 / 3** | yes | 3 / 3 | +1.171 [+0.706, +1.631]\* |
| wRC+ | PCA | **3 / 3** | yes | 2 / 3 | +1.188 [+0.728, +1.650]\* |
| wRC+ | JointVAE | **3 / 3** | yes | 3 / 3 | +1.106 [+0.648, +1.552]\* |
| WAR / 600 | ZScore | **3 / 3** | yes | 1 / 3 | +0.046 [+0.013, +0.077]\* |
| WAR / 600 | PCA | **3 / 3** | yes | 1 / 3 | +0.047 [+0.015, +0.078]\* |
| WAR / 600 | JointVAE | **3 / 3** | yes | 0 / 3 | +0.044 [+0.010, +0.077]\* |
| WAR (counting) | ZScore | 0 / 3 | yes | 1 / 3 | −0.047 [−0.080, −0.016]\* |
| WAR (counting) | PCA | 0 / 3 | yes | 1 / 3 | −0.047 [−0.080, −0.015]\* |
| WAR (counting) | JointVAE | 0 / 3 | yes | 0 / 3 | −0.038 [−0.069, −0.007]\* |

> Source: `SSAC27/scripts/task5_encoder_crosscheck.py` → `results/task5/encoder_verdicts.csv`
> ```bash
> python SSAC27/scripts/task5_encoder_crosscheck.py
> ```

### What Task 5 means for the paper

**This is the cleanest result in the package.** The three encodings are
interchangeable to within 0.08 wRC+ pooled — a 0.4% spread — and they agree on
every target, in every fold, including where they all *lose*. All three beat the
scalar baseline on wRC+ (3/3 folds each) and on WAR per 600 (3/3 each); all three
lose to it on counting WAR (0/3 each).

**The claim the paper may now make: the result belongs to the five-tool
structure, not to any encoder.** That is a genuinely strong statement and it was
falsifiable — a single encoder carrying the effect would have forced a different
framing. It also retires a live reviewer objection: nobody can attribute the
finding to the VAE, or to a lucky choice of group aggregator.

**A secondary, slightly awkward implication.** If a group mean, a first principal
component and a VAE latent are interchangeable, then the *encoder* contributes
essentially nothing beyond the grouping. The JointVAE's contribution to this
paper is its covariance structure and its uncertainty estimates, not predictive
accuracy — and the paper should say so rather than implying the VAE earns its
complexity on prediction. JointVAE is nominally best with age and PA (18.319) but
by a margin far inside the confidence intervals.

---

---

## Task 6 — The study widened to 2019–2025, five rolling origins

**Question.** Do Tasks 1–3's conclusions survive on 40% more data and nearly
twice as many origins?

Tasks 1–3 are anchored to the archived 2021–2025 tool scores so the published
numbers reproduce exactly. That anchoring costs coverage. With the 2019–2020
export the whole cohort can be rebuilt from raw, dropping the anchor:

| | Tasks 1–3 | Task 6 |
|---|---|---|
| Seasons | 2021–2025 | 2019–2025 |
| Cohort rows (PA ≥ 100) | 2,286 | 3,070 |
| Transitions | 1,414 | **1,981** |
| Rolling origins | 3 | **5** |
| Tool scores | archived | recomputed (ZScore, PCA) |

**This is an extension, not a replacement.** Tasks 1–3 remain the primary
results.

**Recomputed vs archived tool scores** on the 2,286-row overlap — asserted at
r > 0.999 on every run:

| Tool | Pearson r | mean abs. difference |
|---|---|---|
| Contact | 0.999981 | 0.0051 |
| Power | 0.999993 | 0.0036 |
| Discipline | 0.999980 | 0.0052 |
| Defense | 0.999843 | 0.0094 |
| Speed | 0.999934 | 0.0096 |

**2020 is kept and flagged.** Its qualified cohort is 310 players with a maximum
of 267 PA, against ~460 and ~730 in a full season, so the PA ≥ 100 qualifier
selects a different kind of player. Dropping it was considered and rejected as
primary: 2019 is reachable only through the 2019→2020 and 2020→2021 transitions,
so excluding 2020 discards 2019 too and collapses the window back to three
origins. A 2020-free variant is reported instead (`decisions_log.md` D16).

| Fold | train n | test | test n |
|---|---|---|---|
| 2020→21 | 275 | 2020→2021 | 279 |
| 2021→22 | 554 | 2021→2022 | 354 |
| 2022→23 | 908 | 2022→2023 | 353 |
| 2023→24 | 1,261 | 2023→2024 | 358 |
| 2024→25 | 1,619 | 2024→2025 | 362 |
| pooled | — | all five | **1,706** |

### Table 6.1 — Test MAE by fold × model, 2019–2025

| Model | 2020→21 | 2021→22 | 2022→23 | 2023→24 | 2024→25 | pooled |
|---|---|---|---|---|---|---|
| M0 current wRC+ | 20.782 | 20.127 | 19.370 | 20.502 | 19.312 | 19.983 |
| M1 B2V 5 tools | 19.961 | 19.315 | 18.688 | 19.056 | 18.210 | 19.002 |
| **M2 constituent stats** | 20.577 | 19.269 | 18.344 | 18.512 | 17.806 | **18.822** |
| M3 wRC+ + age + PA | 20.664 | 19.796 | 18.857 | 20.546 | 19.244 | 19.784 |
| M4 B2V + age + PA | 19.941 | 19.085 | 18.351 | 19.210 | 18.097 | 18.890 |
| M5 constituents + age + PA | 20.490 | 19.033 | 18.127 | 18.586 | 17.780 | 18.724 |
| M6 Marcel (no fit) | 19.256 | 19.297 | 19.540 | 20.021 | 19.474 | 19.530 |
| M6c Marcel calibrated | 20.117 | 18.987 | 18.819 | 19.303 | 19.013 | 19.209 |
| M7 Marcel + B2V | 19.842 | 18.854 | 18.527 | 18.831 | 18.203 | 18.805 |

2020-free variant (three origins), which independently reproduces the
archived-anchored Task 2 closely:

| Model | 2022→23 | 2023→24 | 2024→25 | pooled | Task 2 pooled |
|---|---|---|---|---|---|
| M0 current wRC+ | 19.237 | 20.576 | 19.254 | 19.689 | 19.704 |
| M1 B2V 5 tools | 18.518 | 19.011 | 18.076 | **18.533** | **18.533** |
| M2 constituent stats | 18.664 | 18.631 | 17.740 | 18.341 | 18.388 |

> Source: `SSAC27/scripts/task6_extended_window.py` → `results/task6/fold_performance.csv`
> ```bash
> python SSAC27/scripts/task6_extended_window.py
> ```

### Table 6.2 — Sign consistency across five origins

| Contrast | folds candidate better | same sign | CI excludes 0 | pooled |
|---|---|---|---|---|
| **M1 vs M0** | **5 / 5** | **yes** | **4 / 5** | **+0.981 [+0.619, +1.343]\*** |
| M2 vs M0 | 5 / 5 | yes | 4 / 5 | +1.161 [+0.733, +1.595]\* |
| M4 vs M3 | 5 / 5 | yes | 4 / 5 | +0.894 [+0.563, +1.236]\* |
| **M7 vs M6c** | **5 / 5** | **yes** | 2 / 5 | **+0.404 [+0.193, +0.608]\*** |
| **M1 vs M2** | 1 / 5 | **no** | 3 / 5 | **−0.180 [−0.350, −0.008]\*** |
| M6c vs M1 | 1 / 5 | no | 0 / 5 | −0.207 [−0.580, +0.167] |

> Source: `results/task6/sign_consistency.csv`, `results/task6/paired_differences.csv`

### What Task 6 means for the paper

**The headline claim strengthens.** B2V beats the calibrated scalar in **5 of 5
origins**, with the CI excluding zero in 4 of them and pooled over 1,706 test
cases at +0.981 [+0.619, +1.343]. "Across five temporal holdouts spanning
2020–2025" is now defensible, and the interval is far tighter than any single
fold's. The one fold that misses significance (2022→23, +0.683 [−0.001, +1.384])
misses by a hair.

**The Marcel result becomes cleanly null, which is the right outcome.** M6c vs
M1 is 1/5, sign-flipping, **0 / 5 significant**, pooled −0.207 [−0.580, +0.167].
On the widest evidence available, a properly fed Marcel and the five-tool vector
are simply equivalent. Meanwhile M7 vs M6c stays positive in **5 / 5** origins,
pooled +0.404 [+0.193, +0.608] — the incremental claim is the robust one.

**The unfavourable finding sharpens, and must be reported.** M1 vs M2 was a tie
in Task 1 (CI containing zero). On five origins and 1,706 cases the pooled
estimate is **−0.180 [−0.350, −0.008]**, now excluding zero: the 22 constituent
statistics are *significantly* better than the 5 tool scores. The magnitude is
about 1% of MAE, so the practical reading is still "compression is nearly
lossless" — but the paper can no longer claim even a tie. The defensible sentence
is: **"the five-tool compression costs about 0.18 wRC+ of accuracy relative to
its own 22 constituent statistics, roughly 1% of baseline error, in exchange for
a scout-legible representation."**

**One striking exception, worth a sentence in the paper.** The only origin where
the 5 tools *beat* the 22 statistics is **2020→21 (+0.616, CI excludes zero)** —
the 60-game COVID season. That is exactly where a short, noisy input season makes
group averaging pay: aggregating 22 noisy statistics into 5 means is a variance
reduction that earns its keep when each statistic is measured on 250 PA instead
of 600. This is a mechanistic, checkable claim about *when* the tool structure
helps, and it is the strongest positive evidence in the package for the grouping
being more than dimensionality reduction. It rests on one fold, so it is
hypothesis-generating.

### Table 6.3 — The WAR targets in the extended window

Counting targets cannot use 2020 as an outcome season (a 60-game total is on a
different scale and a model cannot know the schedule in advance), and with 2020
excluded on both sides the earliest usable transition is 2021→2022. **The
extended window therefore collapses to exactly the three origins Task 2 already
has for the WAR targets**, so Task 6 reports five origins for wRC+ and three for
WAR, rather than duplicating a table with no extra origins (`decisions_log.md`
D18).

Pooled over the 2020-free window, on the slightly larger PlayerId-keyed cohort:

| Contrast | wRC+ | WAR (counting) | WAR / 600 |
|---|---|---|---|
| M1 vs M0 | +1.156 [+0.699, +1.606]\* | −0.048 [−0.077, −0.016]\* | +0.045 [+0.013, +0.076]\* |
| M2 vs M0 | +1.348 [+0.832, +1.862]\* | −0.002 [−0.035, +0.032] | +0.071 [+0.032, +0.112]\* |
| M1 vs M2 | −0.192 [−0.394, +0.020] | −0.046 [−0.071, −0.021]\* | −0.026 [−0.055, −0.000]\* |
| M7 vs M6c | +0.538 [+0.265, +0.814]\* | +0.015 [+0.001, +0.029]\* | +0.018 [−0.001, +0.037] |
| M6c vs M1 | −0.507 [−1.013, +0.014] | +0.035 [−0.004, +0.076] | −0.001 [−0.041, +0.037] |

Every sign and every verdict matches Task 2 on the same three origins, computed
on an independently rebuilt cohort. That is the strongest available check that
the rebuild is sound.

> Source: `results/task6/fold_performance.csv`, `results/task6/paired_differences.csv`

**Caveats.** Tool scores are recomputed rather than archived, so Task 6 is not
expected to land on 18.0267 to four decimals — though the 2020-free variant's
pooled M1 of 18.533 matches Task 2's 18.533 exactly, which is a strong
independent check. Every fold's training window includes the 2019→2020
transition, whose outcome season is the 60-game year; the 2020-free variant
exists precisely so a reader can see what that is worth.

---

## Environment

Recorded per run in `results/task*_environment.json`, including SHA-256 hashes
of every input file and, for Task 3, the exact Marcel parameters. Pinned in `SSAC27/requirements.txt`.

| | |
|---|---|
| Python | 3.14.6 |
| numpy | 2.5.1 |
| pandas | 3.0.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| Seed | 42 |
| Bootstrap replicates | 2,000 |

The published Finding 2 run used Python 3.12.3 / numpy 2.5.1 / pandas 3.0.5 /
scikit-learn 1.9.0. M0 and M1 reproduce to 4 decimal places across that gap.
