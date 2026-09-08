# Revisions Log

Running record of the methodological/reproducibility revision pass on the Baseball2Vector
paper draft. Each dated section documents what was run, what changed, the resulting numbers,
and a plain-language paragraph suitable for dropping into the paper's Results/Limitations
sections. Generating scripts live in `scripts/revision/`; artifacts in `outputs/revision/`.

---

## 2026-08-08 — Phase 0: Data Audit

**Script:** `scripts/revision/phase0_data_audit.py`
**Full report:** `outputs/revision/phase0_data_audit/audit_report.md`
**Output data:** `data/v3_results_clean.csv` (2,309 rows, 1 relabeled)

### What was run
Loaded `data/processed/v3_results.csv` (2,309 rows, 2021–2025, 30 columns). Checked the
Max Muncy 2025 duplicate, scanned for other `Name`+`Season` collisions and implausible stat
ranges, and statically audited the pipeline source for pooled-vs-within-season Z-scoring.

### What changed
- **Max Muncy 2025 duplicate resolved.** Confirmed via web search that these are two distinct,
  real MLB players who happen to share a name: Max Muncy (b. 1990, Dodgers, MLB debut 2015,
  Baseball-Reference `muncyma01`) and Max Muncy (b. 2002, Athletics, MLB debut March 27, 2025,
  `muncyma02`). The 388-PA/2.9-WAR/137-wRC+ row matches the veteran Dodgers player's 2025
  season; the 220-PA/-0.4-WAR/72-wRC+ row matches the Athletics rookie's injury-shortened,
  early-struggles-then-improvement debut season. **Disambiguation rule:** treat as two distinct
  players (neither dropped); the lower-PA (rookie) row's `Name`/`UniqueName` is relabeled
  `Max Muncy (ATH)` in the cleaned dataset so the `Name`-keyed transition-dataset builder in
  `src/baseball2vec/roi.py` (`groupby("Name")`) cannot splice one player's season onto the
  other's. Only 1 of 2,309 rows is affected.
- **No other name collisions found.** `groupby(["Name","Season"])` produces exactly one group
  with >1 row (Max Muncy 2025, handled above). No duplicate full rows, no nulls, no PA/WAR/wRC+/OPS
  values outside plausible bounds for qualified hitter-seasons.
- **Z-scoring confirmed within-season, as claimed.** `season_zscore()`
  (`src/baseball2vec/tools.py:56-72`) and the Bayesian stabilization step
  (`src/baseball2vec/data.py:108-146`) both group by `Season` before computing means/stds — no
  pooling leakage in either step.
- **Pooling risk identified (not fixed here — scoped to Phase 1).** The 20–80 min–max scaling
  (`scale_to_2080`, `src/baseball2vec/tools.py:91-106`) fits one `MinMaxScaler` across the full
  pooled 2021–2025 population. This is fine and already disclosed for the *descriptive* Table 1
  correlations and Table 2 top-10 list (paper Sec. 3 already states the scaling is pooled). But
  the same pooled-fit `ZScore_{Tool}` columns feed the cross-validated ΔTool regression
  (`src/baseball2vec/roi.py`) behind the paper's $R^2=0.71$ claim — there, pooling across the
  fold boundary is a genuine (if narrow) leakage channel. Phase 1 fixes this by refitting the
  min–max bounds inside each training fold only.

### Numbers
| Check | Result |
|---|---|
| Row count | 2,309 |
| Season coverage | 2021=463, 2022=469, 2023=461, 2024=455, 2025=461 |
| Name+Season collisions | 1 (Max Muncy, 2025) |
| Duplicate full rows | 0 |
| Null values | 0 |
| PA range | [100, 753] |
| WAR range | [-2.2, 11.3] |
| wRC+ range | [-18, 220] |
| OPS range | [0.322, 1.159] |

### Plain-language summary (for paper)
A data audit of the 2,309-row processed dataset found one name collision: two distinct,
real MLB players both named Max Muncy (a Dodgers veteran and a 2025 Athletics rookie) shared
a `Max Muncy` label in the 2025 season. Both rows are genuine qualifying seasons and are kept;
the rookie's row is now distinguishably labeled to prevent the two players' career trajectories
from being merged in the player-identity-keyed transition dataset used for the ROI regression.
No other data quality issues were found — no duplicate rows, missing values, or implausible
statistic ranges. Separately, the audit confirms the paper's claim that Z-scoring and Bayesian
stabilization are computed within-season (not pooled), so that claim requires no correction.
It also surfaces a narrower leakage channel than previously discussed: the pooled-population
20–80 min–max scaling, while appropriately disclosed for the paper's descriptive correlations,
is applied before the cross-validated ΔTool regression and therefore lets held-out-fold values
influence their own features' scaling — addressed in Phase 1.

---

## 2026-08-08 — Phase 1: Player-Grouped, Leakage-Resistant Cross-Validation

**Script:** `scripts/revision/phase1_grouped_cv.py`
**Outputs:** `outputs/revision/phase1_grouped_cv/results_table.csv`,
`outputs/revision/phase1_grouped_cv/comparison_plot.png`

### What was run
Rebuilt the transition dataset from the Phase 0-audited `data/v3_results_clean.csv`: 1,427
consecutive-season pairs across 570 players (identical counts to the original pipeline — the
Max Muncy relabeling does not change the transition dataset because the rookie's single 2025
row has no prior season to pair with). Re-ran the linear regression (5-feature Δ) and random
forest (10-feature: 5 starting levels + 5 deltas) under four CV schemes:
1. **Pooled KFold (original)** — shuffled 5-fold `KFold`, same as the paper/`roi.py`, recomputed
   here only so the "before" number is measured on the identical audited dataset as the "after"
   numbers.
2. **Player-grouped** — `GroupKFold(n_splits=5)` keyed on player `Name`, so no player's
   transitions appear in both train and test within a fold.
3. **Player-grouped + refit scaling** — same grouping, plus the 20–80 min–max scaler refit on
   training-fold players only (see the script's docstring for why this is done via an affine
   re-scaling of the already-scaled `ZScore_{Tool}` column rather than from raw stats: raw
   FanGraphs data isn't available in this environment, per Phase 0, but per-tool 20–80 scaling
   is a per-feature affine map, and linear/random-forest R² is provably invariant to per-feature
   affine/monotonic rescaling — so this refit is expected, and confirmed, to change nothing).
4. **Forward-time holdout** — train on all transitions ending ≤2023 (707 rows), test on
   2024→2025 transitions only (362 rows), with the 20–80 scaler fit on ≤2023 season-rows only
   and applied to 2024/2025 rows.

### What changed / key finding
**The player-grouped R² does *not* collapse relative to the pooled estimate — if anything it is
marginally higher.** This is the opposite of what the phase was scoped to find, and we report it
plainly rather than searching for a way to manufacture a drop. A plausible explanation: with only
5–10 low-dimensional real-valued features per transition and ~570 players averaging ~2.5
transitions each, there is little room for a linear or shallow-RF model to "memorize" a specific
player's idiosyncratic year-over-year trajectory — the original pooled 5-fold estimate in the
paper was already a reasonably fair estimate of out-of-player generalization for this particular
model class and feature set. The refit-scaling variant is numerically **identical** to the
player-grouped/no-refit variant to at least 6 decimal places, exactly as the affine-invariance
argument predicts — confirming the min–max pooling flagged in Phase 0 has no measurable effect on
this R².

The **forward-time holdout** is the one place a real (if modest) gap appears: the random forest's
R² for ΔwRC+ drops from 0.679 (grouped CV) to 0.636 (forward holdout on 2024→2025), and similarly
0.677→0.644 for ΔWAR. The linear regression is comparably stable on wRC+/WAR (0.708→0.702,
0.703→0.692) and, interestingly, *improves* on ΔOPS (0.708→0.728) — plausibly sampling variance
from evaluating on a single 362-row test slice rather than a genuine second effect; this single
split should not be over-interpreted, but the RF's modest degradation is consistent with the
model fitting some era-specific structure (e.g., league-wide offensive-environment shifts) that a
grouped-but-still-2021–2025-pooled CV cannot detect.

### Numbers
| Target | Pooled KFold (orig.) LR / RF | Player-grouped LR / RF | + refit scaling LR / RF | Forward holdout (≤23→24-25) LR / RF |
|---|---|---|---|---|
| ΔwRC+ | 0.707 / 0.677 | 0.708 / 0.679 | 0.708 / 0.679 | 0.702 / 0.636 |
| ΔWAR  | 0.704 / 0.673 | 0.703 / 0.677 | 0.703 / 0.677 | 0.692 / 0.644 |
| ΔOPS  | 0.707 / 0.678 | 0.708 / 0.678 | 0.708 / 0.678 | 0.728 / 0.691 |

(Pooled-KFold numbers recomputed here on the audited dataset match the paper's reported 0.71 /
0.70 / 0.71 (LR) and ~0.68 (RF) to two decimal places, confirming the Muncy relabeling doesn't
perturb the headline numbers.)

### Plain-language summary (for paper, Sec. 4.3 and Limitations)
We re-ran the ΔTool → Δvalue regression under a player-grouped 5-fold cross-validation
(`GroupKFold` keyed on player identity, so no player's transitions appear in both the training
and test fold) as a check on whether the paper's original pooled cross-validated $R^2=0.71$
partly reflects the model having implicitly "seen" a given player's trajectory during training.
It does not: player-grouped $R^2$ is statistically indistinguishable from the pooled estimate
(0.708 vs. 0.707 for ΔwRC+, 0.703 vs. 0.704 for ΔWAR, 0.708 vs. 0.707 for ΔOPS; linear
regression), with the ten-feature random forest showing the same pattern (0.68 vs. 0.68 across
targets). We additionally refit the 20–80 scouting-scale scaling within each training fold and
found this makes no measurable difference, consistent with the fact that both linear regression
and random-forest $R^2$ are invariant to the kind of per-feature affine rescaling the min–max
step performs. A secondary forward-time check — training on all transitions ending in 2023 or
earlier and testing only on 2024→2025 transitions — shows the random forest's $R^2$ degrade
modestly (0.68→0.64 for ΔwRC+ and ΔWAR) while linear regression remains stable, suggesting some
of the random forest's fit reflects era-specific offensive-environment structure rather than a
purely player-level pattern. We conclude the original $R^2=0.71$ figure is not an artifact of
player-level leakage, but flag the forward-time gap as a more honest estimate of how the model
would perform when forecasting a genuinely future season.

---

## 2026-08-08 — Phase 2: Confidence Intervals for Table 1 Correlations

**Script:** `scripts/revision/phase2_confidence_intervals.py`
**Outputs:** `outputs/revision/phase2_confidence_intervals/table1_with_ci.csv`,
`outputs/revision/phase2_confidence_intervals/encoding_pairwise_diffs.csv`,
`outputs/revision/phase2_confidence_intervals/bootstrap_distributions.npz`

### What was run
Player-level cluster bootstrap (2,000 resamples, seed 42) on the audited
`data/v3_results_clean.csv`: each iteration resamples the 827 unique players with replacement,
takes every season-row belonging to each resampled player (preserving the up-to-5-row block per
player rather than resampling individual rows), and recomputes the pooled Pearson $r$ between
pentagon area and WAR/wRC+/OPS for all three encodings from that same resampled dataset. Using
one shared resample per iteration (rather than resampling independently per encoding) also
supports a **paired** bootstrap on the difference between encodings (e.g. ZScore $r$ − PCA $r$,
computed from the same resampled players each iteration), which is the statistically correct way
to test whether one encoding's correlation is really higher than another's — simply checking
whether two independently-computed CIs overlap is a conservative/misleading heuristic here,
because the three encodings are computed on the same players and are themselves highly
correlated with each other.

### What changed / key finding
Point estimates match the paper's Table 1 to two decimal places (ZScore 0.81/0.65/0.63, PCA
0.79/0.63/0.61, JointVAE 0.74/0.51/0.49 vs. WAR/wRC+/OPS), confirming the audited dataset
reproduces the original numbers. The bootstrap adds two things the paper currently lacks:

1. **Uncertainty on each point estimate.** 95% CIs are roughly ±0.03–0.04 for WAR and
   ±0.04 for wRC+/OPS — not negligible, and worth stating whenever these r-values are quoted.
2. **A statistically rigorous test of the paper's "Z-score wins, PCA nearly indistinguishable,
   JointVAE clearly behind" claim (Sec. 4, current text).** The *marginal* CIs for Z-score and
   PCA overlap substantially (e.g. WAR: [0.780, 0.830] vs. [0.766, 0.819]), which on its own
   would suggest "not distinguishable." But the **paired** bootstrap — which accounts for the
   fact that Z-score and PCA are computed on the same players and are highly correlated with
   each other — shows Z-score's edge over PCA is small (+0.01 to +0.02 across all three metrics)
   but **statistically significant at 95%** for all three metrics (none of the three paired-diff
   CIs include 0). Z-score's edge over JointVAE is both larger (+0.05 to +0.16) and significant
   for all three metrics. So the paper's qualitative ordering (ZScore > PCA > JointVAE) is
   correct and holds up under a proper uncertainty analysis, even though a naive glance at
   overlapping marginal CIs would have wrongly suggested Z-score and PCA are indistinguishable.
   This is a case where the paper's existing hedge ("PCA nearly indistinguishable") is the right
   call qualitatively (the gap is small) but can now be stated with a precise, defensible
   magnitude and significance rather than an impression.

### Numbers
| Encoding | WAR r [95% CI] | wRC+ r [95% CI] | OPS r [95% CI] |
|---|---|---|---|
| Z-score | 0.806 [0.780, 0.830] | 0.649 [0.613, 0.683] | 0.626 [0.588, 0.663] |
| PCA | 0.794 [0.766, 0.819] | 0.634 [0.598, 0.669] | 0.610 [0.571, 0.648] |
| JointVAE | 0.739 [0.706, 0.768] | 0.506 [0.466, 0.545] | 0.486 [0.442, 0.528] |

Paired differences (ZScore − PCA), 95% CI on the difference, all significant:
WAR +0.012 [+0.011, +0.014]; wRC+ +0.014 [+0.013, +0.016]; OPS +0.016 [+0.014, +0.018].

### Ready-to-paste replacement for Table 1 and its caption
Replace the table body with:

| Encoding | WAR | wRC+ | OPS |
|---|---|---|---|
| Z-score averaging | **0.81** [0.78, 0.83] | **0.65** [0.61, 0.68] | **0.63** [0.59, 0.66] |
| PCA | 0.79 [0.77, 0.82] | 0.63 [0.60, 0.67] | 0.61 [0.57, 0.65] |
| JointVAE | 0.74 [0.71, 0.77] | 0.51 [0.47, 0.55] | 0.49 [0.44, 0.53] |

Suggested caption/note addition: *"95% confidence intervals from a player-level cluster
bootstrap (2,000 resamples; players, not season-rows, are the resampling unit, since a player's
multiple seasons are not independent draws). A paired bootstrap on the same resamples confirms
Z-score's advantage over PCA, though only 0.01–0.02 in magnitude, is statistically significant
(95% CI on the difference excludes 0) for all three metrics; the larger gap to JointVAE
(0.05–0.16) is significant a fortiori."*

### Plain-language summary (for paper)
We attached 95% confidence intervals to Table 1's correlations using a player-level cluster
bootstrap, which correctly treats a player's several seasons as one resampling unit rather than
independent observations. The intervals are moderate in width (roughly ±0.03 to ±0.04) and,
more importantly, a paired bootstrap analysis (comparing encodings using the same resampled
players rather than independently-computed intervals) confirms that the paper's ordering of the
three encodings is statistically real: Z-score's small edge over PCA (0.01–0.02 in $r$) and its
larger edge over JointVAE (0.05–0.16) are both significant at the 95% level for all three value
metrics. This upgrades "Z-score wins, PCA nearly indistinguishable" from a qualitative
impression to a quantitatively supported claim with a defensible magnitude.

---

## 2026-08-08 — Phase 3: Ablations

**Script:** `scripts/revision/phase3_ablations.py`
**Outputs:** `outputs/revision/phase3_ablations/shrinkage_sensitivity.csv`,
`outputs/revision/phase3_ablations/permutation_sensitivity_all_encodings.csv`

### Data constraint on 3a (documented, not silently worked around)
Before running 3a we re-confirmed the raw-data constraint from Phase 0 by actually attempting a
live fetch in this session: `pybaseball.batting_stats(2024, qual=100)` and a direct `requests`
call to `fangraphs.com/leaders-legacy.aspx` both return **HTTP 403** from this environment's IP,
matching the README's documented anti-scraping warning. Since the Bayesian stabilization step
(`src/baseball2vec/data.py`) that the shrinkage-strength hyperparameter controls operates on raw
per-stat columns (K%, BB%, AVG, ...) that are never persisted past the intermediate pipeline
stage, a literal re-run of the pipeline under alternative PA-equivalent thresholds is not
possible here. 3a instead runs a clearly-labeled **proxy** ablation on the same underlying
question, described below, rather than fabricating raw-stat values.

### 3a. Shrinkage prior strength (proxy ablation)
**What was run:** each tool's already-computed composite `ZScore_{Tool}` score is blended toward
its own population mean by an extra fraction λ ∈ {0 (baseline), 0.10, 0.25, 0.50}, and pentagon
area / Table-1 correlations are recomputed directly on the blended values (deliberately *without*
re-applying the 20–80 min–max rescaling — an earlier version of this script did re-rescale after
blending and found every λ gave results identical to baseline to floating-point precision, because
independently re-normalizing each shrunk column back to a fixed [20, 80] range exactly cancels any
prior affine shrinkage; this null result is itself informative and is why the script fixes the
rescaling to a single shared reference frame instead).

This proxy can only add shrinkage (λ > 0), not remove it: the paper's already-applied
stabilization discarded information (raw values were shrunk toward the league mean before being
saved), so there is no way to recover a "less-shrunk" version of the composite score from the
processed dataset alone. Read the result as "how sensitive is Table 1's r to *additional*
shrinkage beyond what's already applied," not literally "±50% of the current PA-equivalent
thresholds" as the phase brief originally specified.

**Finding:** correlations are essentially insensitive to a wide range of additional shrinkage.
Even λ=0.50 (blending each tool halfway to its population mean) moves $r$ by at most 0.003 in
absolute terms, in both directions (WAR moves slightly down, wRC+/OPS move slightly up) — far
smaller than the CI widths from Phase 2 (±0.03–0.04). This is a negative result reported plainly:
the paper's shrinkage thresholds, while acknowledged as heuristic rather than estimated, do not
appear to be a load-bearing assumption for the headline correlations, at least not in the
direction (more shrinkage) this proxy can test.

| λ (extra shrinkage) | r(WAR) | r(wRC+) | r(OPS) |
|---|---|---|---|
| 0.00 (baseline) | 0.8061 | 0.6488 | 0.6261 |
| 0.10 | 0.8058 (−0.0003) | 0.6495 (+0.0007) | 0.6266 (+0.0005) |
| 0.25 | 0.8052 (−0.0009) | 0.6504 (+0.0016) | 0.6273 (+0.0013) |
| 0.50 | 0.8040 (−0.0021) | 0.6518 (+0.0030) | 0.6283 (+0.0023) |

### 3b. Axis-permutation sensitivity, all three encodings
**What was run:** pentagon area recomputed under all 5! = 120 orderings of the five tool axes,
for each of the three encodings, reporting the range (min–max) of the resulting Pearson $r$
against WAR/wRC+/OPS. This needed no raw data — only the already-computed, per-encoding
20–80-scaled Tool columns in `data/v3_results_clean.csv`.

**Finding:** the Z-score band reproduces the paper's existing claim almost exactly (paper states
$r$ vs. WAR in [0.79, 0.82] for Z-score; recomputed here as [0.791, 0.820]). Extending the same
check to PCA and JointVAE (not previously reported) shows a more nuanced picture than "Z-score is
most robust": **PCA has the widest permutation band of the three encodings in every metric**, both
in absolute terms (e.g. OPS: PCA 0.050 vs. Z-score 0.043 vs. JointVAE 0.036) and as a fraction of
its own point estimate (PCA ≈8% of $r$ vs. Z-score ≈6–7% vs. JointVAE ≈6–8%, metric-dependent).
JointVAE is comparable to or slightly more stable than Z-score in relative terms for wRC+/OPS,
despite its much lower absolute correlation. This complicates a clean "Z-score wins on robustness
too" narrative: Z-score's advantage over the alternatives is its higher absolute $r$ (Phase 2
confirmed this is significant), not that it is uniquely insensitive to axis ordering — PCA is
consistently the most order-sensitive of the three, and JointVAE's low order-sensitivity is
somewhat beside the point given how much lower its correlations already are.

| Encoding | WAR band [min, max] | wRC+ band [min, max] | OPS band [min, max] |
|---|---|---|---|
| Z-score | 0.028 [0.791, 0.820] | 0.044 [0.612, 0.656] | 0.043 [0.592, 0.635] |
| PCA | 0.033 [0.778, 0.811] | 0.051 [0.594, 0.645] | 0.050 [0.572, 0.623] |
| JointVAE | 0.021 [0.738, 0.760] | 0.031 [0.500, 0.531] | 0.036 [0.479, 0.516] |

### Plain-language summary (for paper, Sec. 4.4 and Limitations)
Two robustness checks were run against the paper's two heuristic-hyperparameter admissions. First,
the Bayesian shrinkage-prior strength: because raw FanGraphs stats are unavailable in this
environment (confirmed by a live 403 from FanGraphs during this session, consistent with the
README's documented restriction), we could not literally re-run the pipeline at alternative
PA-equivalent thresholds; instead we tested the same question via a proxy that adds extra
shrinkage to the already-computed tool scores. Even substantially more aggressive shrinkage
(halfway to the population mean) moves Table 1's correlations by at most 0.003 — the current
shrinkage thresholds, though heuristic, are not a load-bearing methodological choice. Second, we
extended the paper's existing axis-permutation check (currently reported only for Z-score, Sec.
4.4) to PCA and JointVAE. The Z-score band matches the paper's stated [0.79, 0.82] almost exactly.
PCA turns out to be the most permutation-sensitive of the three encodings, both in absolute and
relative terms, while JointVAE — despite its much lower correlation with value metrics — is
comparably or more stable to axis reordering than Z-score. This means Z-score's advantage over the
alternatives should be attributed to its higher, statistically significant correlation with value
metrics (Phase 2), not to any special robustness to the pentagon's axis-ordering choice.

---

## 2026-08-08 — Phase 4: External Validation Pilot (Qualitative)

**Script:** `scripts/revision/phase4_external_pilot.py`
**Outputs:** `outputs/revision/phase4_external_pilot/pilot_table.csv`,
`outputs/revision/phase4_external_pilot/notes.md`

**Labeled explicitly as a pilot, not a validation result — no correlation or aggregate statistic
is computed across the 6 players; six points cannot support one.**

### What was run
Identified 6 players with (a) a specific, numerically graded, publicly cited 20-80 pre-MLB
scouting report (all from Baseball America player pages or stories, cited individually — no grade
estimated or guessed; missing digits left blank rather than filled in) and (b) at least one
qualifying (≥100 PA) MLB season in the audited dataset: Bobby Witt Jr., Julio Rodríguez, Adley
Rutschman, Gunnar Henderson, Corbin Carroll, Elly De La Cruz. For each, placed their first
full-PA-season Baseball2Vector (Z-score) vector next to the public grade, comparing on the three
axes that map reasonably cleanly (Contact↔Hit, Power↔Power, Speed↔Run) and separately discussing
Defense (public sources split Field/Arm; Baseball2Vector doesn't) and Discipline (no public
20-80 analogue exists in any source found — noted, not forced).

### What was found (qualitative, per-player detail in `notes.md`)
- **Speed/Run showed the most consistent agreement**: Witt (Speed his standout rookie tool),
  Carroll (Speed 78.8, closely tracking a scouted 80-grade run tool — the cleanest single match in
  the panel), and De La Cruz (Speed his highest tool, consistent with a scouted 70) all lined up.
- **Power was the most consistent disagreement**, generally reading lower in Baseball2Vector than
  the scouted raw-power grade (Rodríguez: Power 58.6 vs. an 80-grade raw-power scouting report;
  De La Cruz: Power 51.7 vs. a scouted 70). A plausible, non-dismissive explanation is offered:
  scouted power grades often describe raw batting-practice ceiling, while Baseball2Vector's Power
  tool is built entirely from in-game production (ISO, SLG, Barrel%, HardHit%, HR/FB) — a
  real difference in what's being measured, not necessarily an error.
- **Defense disagreements recurred** (Witt's rookie year, Henderson, Carroll all read lower on
  Baseball2Vector's Defense tool than their scouted Field grade) and plausibly trace to the
  taxonomy mismatch noted above — Baseball2Vector's single Defense column can't separately
  register an elite or weak Arm grade the way the public Field/Arm split can (e.g. Carroll's
  scouted 45-grade Arm alongside a 60 Field very plausibly pulls his combined Defense score down).
- Two structural caveats apply throughout and limit how far any of this can be pushed: (1) the two
  20-80 scales are calibrated against different reference populations (all minor-league prospects
  vs. only players who already reached MLB regular status), so literal number-matching isn't
  statistically justified — only relative profile shape is compared; (2) small-PA debut seasons
  (excluded here in favor of each player's first full-PA season where a debut was <200 PA) are
  noisy, and even full rookie seasons carry far more year-to-year variance than the paper's Table 1
  population-level correlations.

### Plain-language summary (for paper, "Future directions" or a new short subsection — not framed
as a validated result)
As a small-scale, qualitative check on Baseball2Vector's tool profiles against independent public
scouting information, we compared six recently-graduated top prospects' rookie-season
Baseball2Vector vectors against their pre-MLB Baseball America scouting grades on the three axes
that map reasonably well across taxonomies (Contact↔Hit, Power↔Power, Speed↔Run); Defense and
Plate Discipline were discussed separately since they don't map cleanly onto the public grading
taxonomy. Speed showed the most consistent agreement across players; Power showed the most
consistent disagreement, plausibly reflecting a genuine difference between scouted raw power
potential and Baseball2Vector's in-game-production-based Power tool rather than a flaw in the
method. This is offered as a pilot-scale, hypothesis-generating check only — with six players and
two non-commensurable 20-80 scales, it does not and cannot establish external validity, but the
patterns (particularly the systematic raw-vs-game power gap) suggest concrete directions for a
larger-sample follow-up.

---

## 2026-08-08 — Phase 5: Reproducibility Packaging

**Script:** `scripts/revision/phase5_repro_package.py`
**Outputs:** `outputs/revision/phase5_reproducibility/environment_used.txt`,
`outputs/revision/phase5_reproducibility/phase1_fold_assignments.csv`

### What was run
Recorded the exact package versions installed in this session, the exact Phase 1 GroupKFold
fold assignments (which player's transitions landed in which of the 5 test folds, seed 42,
verified no player spans two folds), and added a "Reproducing the Revision Pipeline" section to
the top-level `README.md` documenting the one-command-per-phase reproduction path for Phases 0–5.

### Findings

**Package version drift.** `requirements.txt` pins `pandas==3.0.2`, `numpy==2.4.4` under Python
3.12. This session's environment had `pandas==3.0.3`, `numpy==2.5.1` pre-installed under Python
3.14.6 — a real, if likely minor, drift from what generated the paper's original numbers.
`matplotlib` (3.10.9) and `pybaseball` (2.2.7) were installed fresh this session and match
`requirements.txt`'s pins exactly. `torch` and `seaborn` were not installed at all — not needed,
since Phases 0–4 never call `joint_vae.py` or the seaborn-based parts of `viz.py`. This drift did
not block anything (all Phase 1–4 recomputations of pooled/baseline numbers matched the paper's
published values to 2 decimal places — see Phase 0/1/2 entries above), but exact bit-for-bit
reproduction of the original v3 pipeline (JointVAE training, which the repo's own README already
flags as hardware-sensitive) should not be assumed from this session's environment without
re-pinning to the exact versions.

**A real repo-hygiene bug found and fixed:** `.gitignore` had a blanket `outputs/**` rule with
exceptions carved out only for `outputs/figures/.gitkeep` and `outputs/tables/.gitkeep` — meaning
every artifact generated by Phases 0–4 of this revision pass (`outputs/revision/**`) was silently
untracked and would **not** have been committed even after `git add`. Added
`!outputs/revision/` and `!outputs/revision/**` exceptions; verified with `git add -n
outputs/revision/` that all 12 generated files are now stageable. This was caught only because we
checked `git status` before assuming the outputs were safely trackable — worth being aware of if
new `outputs/` subdirectories are added in the future, since the same blanket rule will silently
swallow them too.

**Data pipeline provenance (from raw FanGraphs pull to `data/v3_results_clean.csv`), for the
record:** `scripts/01_build_dataset.py` (`pybaseball.batting_stats()`, qual≥100 PA, 2021–2025) →
`preprocess()` (type coercion + within-season Bayesian stabilization, `src/baseball2vec/data.py`)
→ `scripts/03_joint_vae.py` (within-season Z-scoring, direction alignment, Z-score/PCA/JointVAE
tool scores, pooled 20–80 scaling, pentagon area) → `data/processed/v3_results.csv` → **this
session's** `scripts/revision/phase0_data_audit.py` (Max Muncy 2025 disambiguation) →
`data/v3_results_clean.csv`, which all of Phases 1–4 read.

**Random seeds used throughout:** 42, consistently — `GroupKFold`/`KFold` shuffling (Phase 1),
`RandomForestRegressor` (Phases 1, all instances), the cluster-bootstrap RNG (Phase 2,
`np.random.default_rng(42)`). Phase 0, 3, and 4 are fully deterministic (no sampling involved;
Phase 3b enumerates all 120 permutations rather than sampling). Phase 1's exact fold assignments
are saved to `outputs/revision/phase5_reproducibility/phase1_fold_assignments.csv` for audit —
anyone can verify a specific player landed in a specific fold without re-running anything.

### Is the repo ready to be made public?

**Mostly, with two things to resolve first:**

1. **FanGraphs data redistribution.** `data/processed/v3_results.csv` (and now
   `data/v3_results_clean.csv`) are *derived* statistics (Z-scores, PCA components, pentagon
   areas) computed from FanGraphs data, not a raw republication of FanGraphs' own tables — but
   FanGraphs' Terms of Service (linked in `README.md`) should be checked explicitly for whether
   a derived, non-commercial research dataset at this level of transformation is permitted to be
   committed to a public GitHub repo, versus only usable locally / distributed on request. This
   wasn't re-verified in this session (out of scope for the technical revision work) and is
   flagged here as an open item for Choi to confirm before a public push, not resolved.
2. **`data/raw/` is empty and FanGraphs blocks this environment's IP** (confirmed via a live
   fetch attempt in Phase 3, HTTP 403) — this is already disclosed in the existing README's "Data
   availability" section and needs no further action, just noted as consistent.

**No secrets or API keys were found** in any file touched by this revision session (`git grep`
for common secret/key/token patterns across the files this session created or modified turned up
nothing). Two files elsewhere in the repo (`LM_experiments/src/lm_experiments/lm_arms/anonymize.py`,
`LM_experiments/config.yaml`) matched a generic keyword search but were **not inspected**, since
`LM_experiments/` (the RQ2 LM-agent benchmark work) is explicitly out of scope for this session —
flagged here only so it isn't forgotten, not assessed.

### Plain-language summary (for paper / repo release notes)
The revision pipeline (Phases 0–4) is fully reproducible from the committed, already-processed
dataset with one command per phase, documented in the top-level README. We recorded the exact
package versions used to generate these outputs, noting a minor drift from `requirements.txt`'s
pins that didn't affect any recomputed number to the precision reported, and the exact
player-to-fold assignments behind Phase 1's grouped cross-validation for full auditability. We
also caught and fixed a `.gitignore` bug that would have silently prevented every revision-phase
output from ever being committed. Before making the repository public, FanGraphs' terms of
service should be explicitly checked for the processed/derived dataset (not yet done in this
session); no other blockers (secrets, keys, or raw-data licensing issues within scope) were found.
