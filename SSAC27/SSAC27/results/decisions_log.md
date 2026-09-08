# SSAC27 experiment package — decisions log

Every configuration that was considered, adopted or rejected, with the reason.
Written so a reviewer can tell what was decided *before* looking at a test score
from what was decided after (nothing in the second category, so far).

Seed 42 throughout. Bootstrap 2,000 replicates. All scripts live in
`SSAC27/scripts/`; all numbers in `SSAC27/results/`.

---

## D0 — Package isolation

**Adopted.** New folder `SSAC27/`, self-contained, writing only to
`SSAC27/results/` and `SSAC27/data_intermediate/`.

**Why.** The repository convention (see the root `README.md`) is that every
post-submission analysis lives in its own isolated folder and never overwrites
`src/`, `scripts/01-04`, `data/processed/`, or a previous run's outputs. Nothing
under `final_academic_revision/`, `academic_validation_experiments/`,
`outputs/`, or `paper/` is modified by this package.

---

## D1 — The raw constituent statistics exist; the stated limitation was stale

Both `README.md` ("Raw constituent statistics ... are not archived, so several
checks could not be run and are reported as `not_run_missing_raw`") and
`final_academic_revision/run_20260808_055335/README.md` say the raw-statistic
Ridge control could not be run.

**Finding.** `data/raw/batting_stats_2021_2025.csv` (2,309 rows x 462 columns,
written 2026-08-12 by `ScalingRevision/build_raw_from_fangraphs_export.py`)
contains all 22 constituent statistics plus `PlayerId` / `MLBAMID`. It is the
*pre-`preprocess()`* qualified export, so it is genuine raw input.

**Consequence.** Task 1 (the constituent-statistics control) is executable. Both
READMEs need that limitation retracted.

---

## D2 — Data source for M0/M1 (archived) vs M2 (recovered raw)

Two candidate constructions were on the table:

| Option | Description | Verdict |
|---|---|---|
| **(a)** Archived tool scores + archived targets; join raw export in for the constituent features only | keeps M0/M1 numerically identical to published Finding 2 | **adopted** |
| (b) Rebuild every model from the raw export | fully single-vintage, but moves the published 19.220 / 18.027 anchors | run as a robustness variant only |
| (c) Report both as co-primary | doubles every table, defers the judgement to the reader | rejected |

**Why (a).** Finding 2's headline numbers are already cited in the manuscript;
keeping M0/M1 byte-comparable makes the new control an *addition* to the
existing result rather than a replacement of it. Option (a) was confirmed by the
project owner before Task 1 was run.

**Known cost, stated up front.** The recovered export is a different FanGraphs
vintage than the April pybaseball fetch that produced `v3_results.csv`. On the
2,286 stable-ID rows the vintages differ by a median of -0.004 wRC+ (SD 1.45)
and -0.001 WAR (SD 0.08) — post-hoc FanGraphs restatements. So M2's features
carry a hair more measurement noise than M1's. **This biases against M2, not
against B2V**, i.e. it cannot manufacture the "M1 is no better than M2" result;
if anything it works the other way.

**Join integrity check, run on every execution** (`ssac_common.load_player_seasons`,
assertion threshold r > 0.99). Tool scores recomputed from the raw export vs the
archived scores, both standardized within season:

| Tool | Pearson r |
|---|---|
| Contact | 0.99999 |
| Power | 1.00000 |
| Discipline | 0.99998 |
| Defense | 0.99987 |
| Speed | 0.99994 |

**Reproduction anchor, asserted on every execution.** Under the published alpha
grid, M0 = 19.2195 and M1 = 18.0267 — the published Finding 2 values, to four
decimals, in a different environment (Python 3.14.6 / sklearn 1.8.0 vs the
archived run's 3.12 / 1.9.0).

---

## D3 — Six display-name variants between the two vintages

The archived table and the raw export disagree on the spelling of six
player-seasons. Each pair was verified to be the same player-season by an exact
PA match, and is mapped in `ssac_common.NAME_ALIASES`:

| Archived | Raw export | Season | PA |
|---|---|---|---|
| Andy Young | Andrew Young | 2021 | 104 |
| Jackson Frazier | Clint Frazier | 2021 | 218 |
| Jose Miranda | Jose F Miranda | 2022 | 483 |
| Calvin Mitchell | Cal Mitchell | 2022 | 232 |
| Jose Miranda | Jose F Miranda | 2023 | 152 |
| Jose Miranda | Jose F Miranda | 2024 | 429 |

**Rejected alternative:** dropping the six rows. With the alias map the join is
2,286 / 2,286 with zero loss, so dropping would have discarded real data for no
reason.

---

## D4 — Alpha grid: one wide grid for every model

The brief asks for a wide grid **for M2** because it has more features. Giving
only M2 a wide grid would have made the search budget a confound.

**Adopted.** `np.logspace(-3, 4, 50)` for **every** model, primary.
**Also reported.** The published 6-value grid `[0.01, 0.1, 1, 10, 100, 1000]` as
a secondary variant, which is what pins the reproduction anchor.

Selection rule in both cases: player-grouped `GroupKFold(n_splits=5)` on the
training window only, minimum mean fold MAE, exact ties broken toward the
smaller alpha. This was fixed before any test MAE was computed and has not been
changed since.

---

## D5 — M2's preprocessing is the tool pipeline's own preprocessing

**Adopted.** M2's 22 features are the constituent statistics after exactly the
transform chain the five tool scores are built from:

1. `data.preprocess()` — Bayesian stabilization toward the **same season's**
   league mean (`STABILIZATION_THRESHOLDS`)
2. `tools.season_zscore()` — z-score **inside the input season**, NaN to the
   cohort mean
3. `tools.apply_direction()` — sign flip on `K%`, `SwStr%`, `O-Swing%`, `Swing%`

**Why.** Under this choice the five tool scores are *exactly* the five group
means of M2's 22 columns, so M1's feature space is a fixed linear projection of
M2's. The M1-vs-M2 contrast then isolates the tool grouping and nothing else —
not a preprocessing difference, not a scaling difference. Any other choice would
have confounded the comparison.

**Leakage status.** Steps 1 and 2 read only the row's own season cohort; step 3
is a constant. No future season is visible. This is the same forecast-safe
construction the published `safe_*` tool features use.

**Rejected alternative:** feeding unstabilized, unscaled raw statistics. That
would have tested "tools vs raw numbers on different preprocessing", which is
not the question asked, and would have handed M1 an unearned advantage.

**Note.** After step 2 the feature matrix has zero NaN, so the pipeline's
train-only `SimpleImputer(median)` is a no-op here. It is kept in place anyway so
the pipeline is identical across all models and stays correct if a future data
vintage does have gaps.

---

## D6 — UBR is largely missing in the recovered export

| Season | UBR missing / n |
|---|---|
| 2021 | 0 / 463 |
| 2022 | 0 / 469 |
| 2023 | 1 / 461 |
| 2024 | **97 / 455** |
| 2025 | **461 / 461** |

An export artifact, not a real data gap (`BsR`, `wSB`, `Spd` are complete in
every season). It matters for Fold C, whose test *input* season is 2024.

**Adopted.** Keep UBR in M2, filled with the input-season cohort mean by step 2
of D5, **and** report a drop-UBR robustness run
(`results/task1_ubr_robustness.csv`). Since `BsR = UBR + wSB + wGDP`, most of the
signal survives regardless; the robustness run confirms it rather than assuming
it (17.707 without UBR vs 17.746 with, a 0.04 wRC+ move).

**Rejected alternative:** silently dropping UBR from the tool definition. That
would have changed the Speed tool relative to the published B2V and broken
comparability with Finding 1 and Finding 2.

---

## D7 — Bootstrap unit

**Adopted.** Cluster bootstrap on the **focal player**: resample the unique
focal `player_id`s with replacement, then take every test row belonging to each
drawn player. 2,000 replicates, seed 42, the *same* replicate index sets shared
across all models in a fold so the model differences are genuinely paired.

**Why clustering matters here.** In Fold C each test player contributes exactly
one row, so clustering is equivalent to a row bootstrap (this is what the
published run did). In the Task 2 rolling-origin pooled analysis a player can
contribute a test row to more than one fold, and there clustering is required
for the resampling unit to be independent. One implementation is used for both.

---

## D8 — Rolling-origin fold construction (Task 2)

**Adopted.** Expanding-window origins, defined by the *input* season:

| Fold | Train input seasons | n_train | Test | n_test |
|---|---|---|---|---|
| A | 2021 | 351 | 2022→23 | 350 |
| B | 2021, 2022 | 701 | 2023→24 | 355 |
| C | 2021, 2022, 2023 | 1,056 | 2024→25 | 358 |

`split_fold` asserts `max(train Season_t1) <= test Season_t` in every fold, so a
training row's outcome season never reaches into the test fold's input season.
Alpha is re-selected inside each fold's own training window; nothing is shared
between folds.

**Rejected alternative: a sliding (fixed-width) window.** An expanding window is
what a practitioner would actually have available at each origin, and a sliding
window would have thrown away training data in Folds B and C for no benefit at
this sample size.

**n_train reporting.** The brief asks for a flag below n_train = 300. No fold
falls below it (351 is the minimum), but the `thin_training_set` column is
emitted in `task2_fold_performance.csv` anyway so the check is visible rather
than asserted in prose, and Fold A's thinness is called out in the writeup.

**Pooled-row metadata.** The pooled row aggregates predictions from three
separately fitted models, so `n_train`, `selected_alpha` and `train_period` have
no single value. They are emitted as `NA` / a descriptive string rather than
silently inheriting Fold A's values — an earlier draft of the script did inherit
them, which was wrong and was fixed before any pooled number was reported.

**Pooled bootstrap unit.** The 1,063 pooled test rows come from 499 distinct
focal players (157 contribute one row, 120 two, 222 all three). A row-level
bootstrap would treat those as independent and understate the pooled interval,
so the focal-player clustering of D7 is load-bearing here, not decorative.

---

## D9 — Marcel implementation (Task 3)

**Parameters were sourced, not recalled.** Tango's original page
(https://www.tangotiger.net/archives/stud0346.shtml) supplies the 5/4/3 weights,
the 1200-PA regression, the projected-PA formula, the two age slopes and the
rookie rule. Two things his prose leaves ambiguous were resolved against
`bdilday/marcelR`, the widely used reference implementation:

1. **What scale the 1200 lives on.** marcelR: `num = x_av*100*sw + x_metric`,
   `denom = x_pa + 100*sw` with `sw = 12` and `x_pa` accumulated with the *raw*
   5/4/3 weights. So 1200 sits on the raw-weighted PA scale, and a 600-PA
   regular has reliability 7200/8400 = 0.857. The alternative reading —
   normalizing the weights to sum to 1, giving reliability 600/1800 = 0.33 —
   would have regressed roughly three times as hard and is **not** what Tango's
   "2 weights x 600 PA" arithmetic produces. Rejected.
2. **The sign of the age term.** Tango gives `AgeAdj = (age-29)*.003` above 29
   and `*.006` below, but not how it is applied. marcelR's `age_adjustment()`
   settles it: `1/(1+0.003(age-29))` above, `1+0.006(29-age)` below, multiplied
   into the rate at the *projected* season's age. This is the only reading under
   which young players project upward. Every full sourcing quotation is in
   `scripts/marcel.py`'s module docstring.

**Adaptation to wRC+.** wRC+ is already a league- and park-adjusted rate indexed
to 100, so the "counting stat" is taken to be `wRC+ x PA` and the projection is
the ratio. The league baseline is each season cohort's own PA-weighted mean wRC+
(about 102.1, not 100, because the PA >= 100 qualifier selects above-average
hitters). **Rejected alternative:** regressing toward a hard-coded 100. That
would inject a systematic downward bias for this cohort and is not what Tango's
"league average" means operationally.

**Handling of seasons outside 2021-2025.** Skipped from every accumulator rather
than entered as a zero league average, which would bias the baseline downward.

**Stated cost, not hidden.** This is a *history-truncated* Marcel: no pre-2021
seasons exist, and seasons under 100 PA are absent from the source table
entirely. Both handicap Marcel and neither handicaps B2V. The writeup states
this before reporting any Marcel-vs-B2V comparison, and the paper language is
constrained accordingly.

---

## D10 — M6c, the recalibration control (Task 3)

**Adopted.** A fourth model, `M6c` = Ridge on `[marcel_pred]` alone, in addition
to the brief's M6 (raw Marcel) and M7 (Marcel + B2V).

**Why.** M7 - M6 is not a clean incremental test: M7 receives the B2V features
*and* a fitted intercept and slope on the Marcel projection, so part of any gap
is affine recalibration unrelated to B2V. M6c isolates that term — it is worth
0.556 wRC+ pooled on its own. M7 - M6c is reported as the honest incremental
test; M7 - M6 is reported as well because the brief asks for it. Adding this
control **shrinks** the headline gain (+1.231 to +0.675), which is the direction
that matters for whether it was added in good faith.

---

## D11 — The history-depth stratification (Task 3), a post-hoc diagnostic

**Disclosed as post-hoc.** `task3_history_depth_strata.csv` was added *after*
observing that this Marcel trails B2V. It splits the pooled test cases by how
many lookback seasons Marcel actually had and asks whether the deficit closes as
history fills in.

**Why it is not test-set-driven tuning.** It changes no model, no feature, no
hyper-parameter and no split; nothing was refit. It is a falsifiable check on an
explanation that was already written down in `scripts/marcel.py` *before* the
result was known — the truncation limitation is stated in that docstring as a
design consequence, not as an excuse constructed afterwards. Had the deficit been
flat in history depth, the truncation explanation would have been reported as
failing.

**Outcome.** The deficit closes monotonically (-0.948 -> -0.697 -> -0.324) and is
indistinguishable from zero at three-season depth. The explanation survived.

---

## D12 — Separating the history pool from the modelling cohort

Two new inputs arrived after Tasks 1-3 were first run:

| File | Contents |
|---|---|
| `data/raw/fangraphs-leaderboards (1).csv` | 2021-2026, **no PA minimum**, 465 columns (was already on disk; only its PA >= 100 subset had been used) |
| `data/raw/batting_stats_2019_2020.csv` | 2019-2020, **no PA minimum**, 465 columns |

**Adopted.** `scripts/ssac_data.py` splits what had been one population into two:

* **History pool** (`load_history_pool`) — every season a player actually
  played, no PA minimum, 2019-2025, 5,293 rows. Marcel reads this.
* **Modelling cohort** (`load_extended_cohort`) — PA >= 100, the population the
  tool scores are standardized inside and the only rows that can be a focal
  player. The qualifier Findings 1 and 2 were built on is unchanged.

**Why the split is the whole point.** Marcel treats a season absent from its
lookback table as zero PA. That is correct for "did not play" and wrong for
"played 45 times", and the PA >= 100 qualifier was silently turning the second
into the first. Widening only the *history* pool fixes Marcel while leaving every
model feature and every focal row untouched, so the published anchors
(19.2195 / 18.0267) keep reproducing — asserted on every run.

**2026 is dropped** from both files: it is later than every outcome season in the
study, so it could only ever be a forward reference.

**Validation.** A 600-PA regular with three full seasons should have reliability
7200/8400 = 0.857. Measured on the widened pool: **0.850** (n = 268). On the
qualified-only pool it was 0.737. The 1200-PA constant now behaves as Tango's
arithmetic says it should.

---

## D13 — The league baseline: wRC+'s definition, not a population mean

D9 originally regressed toward the qualified cohort's PA-weighted mean wRC+
(about 102.1). The widened pool makes that choice untenable and supplies a better
one. Per-season PA-weighted mean wRC+ under three candidate populations:

| Season | all PA>0 | ever reached 100 PA in 2019-2025 | PA >= 100 only |
|---|---|---|---|
| 2019 | 96.71 | 101.55 | 102.21 |
| 2020 | 100.01 | 101.35 | **106.38** |
| 2021 | 96.73 | 100.89 | 101.98 |
| 2022 | 100.07 | 100.72 | 102.11 |
| 2023 | 100.18 | 100.75 | 102.04 |
| 2024 | 100.18 | 100.98 | 102.04 |
| 2025 | 99.81 | 100.86 | 102.04 |

- **`all PA>0`** swings 3.5 points between 2019/2021 and the rest. Cause: those
  are the last seasons before the universal designated hitter, so pitcher plate
  appearances drag the mean down. FanGraphs' export has no position label
  (`Pos` is a positional *run value*, not a position), so pitchers cannot be
  filtered out of the file we have.
- **`PA >= 100 only`** reads 106.38 in 2020, because a 100-PA floor in a 60-game
  season selects only near-everyday players. A selection rule that means
  something different each season is not a league average.
- **`ever reached 100 PA in 2019-2025`** is the most stable series of all — a
  0.83 range across seven seasons, immune to both problems. **It was rejected
  anyway**, because "ever reached" is evaluated over the whole window and
  therefore references seasons later than the origin. Rule 1 does not have an
  exception for well-behaved forward references.

**Adopted: 100.0 in every season**, because that is what wRC+ *is* — a league-
and park-adjusted index centred on 100. It requires no population definition, so
it cannot leak. It is confirmed by the five designated-hitter seasons, where the
full-population mean lands at 99.81-100.18.

**marcelR's league re-centering is switched off** as a consequence. That step
rescales an origin year's projections so their PA-weighted mean equals the league
rate, which presumes the projected set *is* the league. Our focal players are
survivor-selected (they qualified in the outcome season) and genuinely average
about 102, so re-centering them onto 100 would inject a systematic downward bias.

**This costs raw Marcel and does not cost calibrated Marcel**, which is the
column that matters. Pooled rolling-origin test MAE:

| History pool | Baseline | M6 raw | M6c calibrated |
|---|---|---|---|
| qualified only (PA>=100, 2021-2025) | definition | 19.732 | 19.110 |
| qualified only | cohort + re-centered | 19.681 | 19.125 |
| **full history (all PA, 2019-2025)** | **definition** | 19.693 | **19.063** |
| full history | cohort + re-centered | 20.158 | 19.196 |

All four calibrated values sit within 0.13 of each other, so the conclusion does
not depend on this choice. M6c is a Ridge fitted on training folds only, which is
the leakage-free way to absorb the survivor-selection offset — which is why the
control was introduced in D10 in the first place.

**A deflating finding, reported rather than buried.** Widening the history pool
buys only **0.047 wRC+** on calibrated Marcel (19.110 → 19.063). An earlier
scratch diagnostic suggested about 0.6; that estimate was inflated because it
re-centered onto the focal population, which is calibration, not information.
Mean lookback depth did rise from 2.22 to 2.61 seasons and reliability from 0.722
to 0.731, so the pool really is better fed — the added seasons are simply
low-PA ones that carry little weight, and the 2019-2020 backfill only reaches
origins 2021 and 2022.

---

## D14 — Age: two definitions, used deliberately in two places

FanGraphs' `Age` column is conventional baseball age (as of mid-season). The
archived pipeline's `age_t` is `Season - birth_year` from the Chadwick register.
They agree on **47%** of rows and differ by one year on the rest — a player born
after roughly 1 July is a year younger by the FanGraphs convention.

**Adopted.**

- **Marcel uses FanGraphs `Age`.** Tango's 0.003 / 0.006 slopes were fitted
  against conventional baseball age, so that is the input the constants expect.
- **The M3/M4/M5 context feature keeps `age_t`.** Changing it would alter the
  published Finding 2 model and break the reproduction anchors.
- The extended-window cohort (Task 6) uses FanGraphs `Age` throughout, since it
  is a self-contained rebuild with no anchor to preserve.

The discrepancy matters little numerically — one year moves the age multiplier
by 0.3-0.6% — but it is a real definitional difference and is recorded rather
than smoothed over.

---

## D15 — Joining on FanGraphs `PlayerId`

**Adopted** for the history pool and the extended cohort:
`player_id = "fg:" + PlayerId`.

**Verified.** All 2,286 archived stable-ID rows join to the export on this key,
with PA identical on every row. The key agrees exactly with the Chadwick-derived
`player_id` the published analysis used.

**Consequence.** The four display-name aliases of D3 (`Andy/Andrew Young`,
`Jackson/Clint Frazier`, `Jose/Jose F Miranda`, `Calvin/Cal Mitchell`) are not
needed on this path, and the Chadwick register download is not needed either.
They remain in `ssac_common.py` because Tasks 1-3 still read the archived
name-keyed table; new code should use `ssac_data.py`.

---

## D16 — The extended 2019-2025 window (Task 6)

**Adopted as an extension, not a replacement.** Tasks 1-3 stay anchored to the
archived tool scores and remain the primary results. Task 6 rebuilds the cohort
from raw for 2019-2025:

| | Tasks 1-3 | Task 6 |
|---|---|---|
| Seasons | 2021-2025 | 2019-2025 |
| Cohort rows | 2,286 | 3,070 |
| Transitions | 1,414 | 1,981 |
| Rolling origins | 3 | 5 |
| Tool scores | archived | recomputed |

Recomputed vs archived tool scores on the 2,286-row overlap: Pearson r =
0.99998 (Contact), 0.99999 (Power), 0.99998 (Discipline), 0.99984 (Defense),
0.99993 (Speed). Asserted at r > 0.999 on every run.

**JointVAE is excluded from the extended cohort.** Producing 2019-2020 latents
would require retraining the VAE on the widened data, which changes the encoder
rather than extending it. ZScore and PCA are recomputed; JointVAE is
cross-checked on the archived seasons in Task 5.

**2020 is kept, and flagged.** Its qualified cohort is 310 players with a maximum
of 267 PA, against ~460 players and ~730 PA in a full season, so the PA >= 100
qualifier selects a different kind of player. Excluding it was considered and
**rejected as the primary**: 2019 is only reachable through the 2019->2020 and
2020->2021 transitions, so dropping 2020 discards 2019 as well and collapses the
window straight back to three origins, defeating the purpose. A 2020-free variant
is reported alongside every table instead. That variant reproduces the
archived-anchored Task 2 closely (pooled M1 18.533 against 18.533, M0 19.689
against 19.704), which is a useful independent check that the rebuild is sound.

---

## D17 — Three targets, not one

**Adopted.** Every task runs three targets:

| Key | Column | Scalar baseline (M0) | Kind |
|---|---|---|---|
| `wrc_plus` | `wRC+_t1` | `wRC+_t` | rate, the published target |
| `war` | `WAR_t1` | `WAR_t` | **counting** |
| `war_rate` | `WAR_per_600_t1` | `WAR_per_600_t` | rate |

**Why.** wRC+ is a batting-only index. It prices neither Defense nor Speed —
two of the five tools — so evaluating the five-tool vector only on wRC+ is a
structurally unfavourable test of the tool structure. WAR prices both. The
repository's own `DeltaAssociationExtension` already reports the matching
asymmetry: Defense and Speed have CIs excluding zero for ΔWAR but not for ΔwRC+
or ΔOPS.

**Why WAR appears twice.** WAR is a season total, so predicting it is partly
predicting playing time. `corr(WAR_t, PA_t) = 0.674` against `0.449` for wRC+.
The counting version is the decision-relevant quantity; the per-600 rate isolates
profile quality from opportunity, and it is the version that actually tests the
hypothesis about tool structure. Both are reported everywhere.

**Model names became target-agnostic** (`M0 scalar baseline` rather than
`M0 current wRC+`) so the same row labels line up across targets.

**Reporting.** Because wRC+ MAE lives on a ~19 scale and WAR MAE on a ~1.1 scale,
raw differences are not comparable across targets. Tables therefore also carry
`mae_pct_of_scalar_baseline` — each model's MAE as a percentage of that target's
own M0.

---

## D18 — 2020 cannot be the outcome season of a counting target

The 2020 season was 60 games. A counting outcome for 2020 (mean WAR 0.71 against
1.29 in full seasons) lives on a different scale from every other season, and a
model cannot know the schedule in advance.

**Adopted.** `Target.excludes_2020` drops every transition touching 2020 for
`war` and `war_rate`. Rate targets are unaffected in principle — a rate is a rate
whatever the schedule — but `war_rate` inherits the exclusion anyway because its
*input* season would otherwise be a 267-PA maximum cohort, which is a different
selection, not a different scale.

**Consequence for Task 6, stated rather than papered over.** With 2020 excluded
on both sides, the earliest usable transition is 2021→2022, so the extended
window collapses to exactly the three origins Task 2 already has. Task 6
therefore reports five origins for wRC+ and three for the WAR targets, and does
not duplicate a table that carries no extra origins.

---

## D19 — Marcel for a counting stat

Tango's Marcel always projects a *rate* and then multiplies by projected playing
time (`proj_value = proj_pa * proj_rate`). For wRC+ that last step is a no-op,
because wRC+ is already a rate. For WAR it is the whole point, and it is the only
place the projected-PA formula does real work in this package.

`marcel.METRICS` carries a `source_is_rate` flag per metric, because the weighted
numerator differs:

| Metric | Numerator | League baseline |
|---|---|---|
| `wRC+` (a rate) | `w x PA x value` | 100.0 by definition (D13) |
| `WAR` (a total) | `w x value` | `sum(WAR) / sum(PA)` over the full pool |

Getting that flag wrong multiplies WAR by PA twice; an early draft did exactly
that and was caught by the league baseline printing 1.743 "per PA" instead of
0.00312. The corrected league rate is **1.874 WAR per 600 PA for 2024**, matching
a direct calculation, and runs 1.833–1.900 across 2019–2025 — a 3.5% spread,
stable enough to use empirically and, unlike a wRC+ population mean, undistorted
by the pitcher plate appearances of 2019 and 2021.

There is no definitional centre for WAR the way there is for wRC+, so the
empirical per-season league rate is used for both WAR variants. It reads only
seasons at or before the origin.

---

## D20 — Task 4: two candidate pools, only one of which is a forecast

The brief specifies retrieving comparables from the focal player's **own season**
and averaging their next-season outcome. That design reads the comparables'
next-season values, which at real forecast time have not happened yet.

**Adopted: report both pools, and label them.**

* `same_season` — the brief's design. A fair A-vs-B comparison, because both
  distance metrics receive identical information. **Not a forecast**, and no
  number from it may be quoted as forecasting accuracy.
* `past_seasons` — candidates restricted to seasons strictly earlier than the
  focal season, whose following season has already been observed. This *is*
  deployable and is the variant to quote for any practical claim.

**Scalar features are standardized within season before the distance is taken.**
WAR and wRC+ have very different numeric ranges, and a raw 2-D Euclidean distance
over them would be almost entirely a wRC+ distance. The tool features are already
within-season standardized, so this puts the two methods on equal footing rather
than handicapping the scalar one.

**A structural caveat on the WAR targets.** Method A's coordinates are WAR and
wRC+. When the target is next-season WAR, method A is searching on the target's
own current-season value and method B is not. B losing there is close to
mechanical and should not be read as evidence about profile similarity.

---

## D21 — Task 5: what would falsify the tool-structure claim

All three encodings (ZScore, PCA, JointVAE) share the same *grouping* — the same
22 statistics assigned to the same 5 tools — and differ only in how each group is
collapsed to one number. The pre-stated reading:

* all three beat the scalar baseline → the result belongs to the **structure**;
* only one does → the result belongs to that **encoder**, and the paper's framing
  would have to change.

The archived table already carries all three encodings, so nothing was retrained
and no encoder got a tuning advantage. Alpha is re-selected per fold per
encoding.

---

## Runs performed, including ones not adopted as primary

Nothing has been re-tuned after seeing a test score. The full list of what was
executed for Task 1:

| Run | Status |
|---|---|
| M0-M5, wide alpha grid | **primary** |
| M0-M5, published 6-value alpha grid | secondary, reported (pins the reproduction anchor) |
| M2b / M5b without UBR, wide grid | robustness, reported |
| Tool-score recomputation from the raw export vs archived | integrity check, reported in D2 |

And for Task 2:

| Run | Status |
|---|---|
| M0-M5 x folds A/B/C/pooled, wide alpha grid | **primary** |
| M0-M5 x folds A/B/C/pooled, published 6-value alpha grid | secondary, reported; same sign pattern throughout |

And for Task 3:

| Run | Status |
|---|---|
| M0, M1, M6, M6c, M7, M7x x folds A/B/C/pooled x 3 targets, wide alpha grid | **primary** |
| Marcel with the 1200 constant on a normalized-weight scale | rejected on sourcing grounds (D9), not run |
| Marcel regressing toward a hard-coded wRC+ 100 | rejected on sourcing grounds (D9), not run |
| History-depth stratification | post-hoc diagnostic, disclosed as such (D11), reported |

The published 6-value alpha grid was not re-run for Tasks 3-6: it made no
difference to any sign or conclusion in Tasks 1 and 2, and running it again
would add tables without adding information. Those tasks report the wide grid
only.

And for Tasks 4-6:

| Run | Status |
|---|---|
| Task 4, methods A and B x k in {3,5,10} x 2 candidate pools x 3 targets | **primary**, all reported |
| Task 5, 3 encodings x 2 feature sets x 3 folds x 3 targets | **primary**, all reported |
| Task 6, wRC+ on 5 origins; WAR targets on the 2020-free 3 origins | **primary** |
| Task 6, WAR targets on the 5-origin window | not run -- 2020 cannot be a counting outcome (D18), so the window collapses to the same 3 origins |

Nothing anywhere in this package was re-tuned after a test score was seen. Where
several settings were tried, every one of them is in a table above.

---

## Open items

- Both `README.md` files still state the raw-statistic control is
  `not_run_missing_raw`. That text is now wrong and should be retracted (see D1).
- Option (b) from D2 — a fully single-vintage rebuild — has not been run yet.
