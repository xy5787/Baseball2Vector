# Agent Pilot V2/V2.1 Report

Run: `AGENT_PILOT_NONPUBLICATION_V2`

This is a six-record diagnostic pilot, not a publication-grade experiment.
The records comprise two elite, two strength-up, and two decliner cases from
the 2024→2025 holdout. One agent instance handled all six records within each
arm, so observations within an arm are not independent agent runs.

## Main result

| Model | WAR MAE | wRC+ MAE | OPS MAE | WAR sign | wRC+ sign | OPS sign |
|---|---:|---:|---:|---:|---:|---:|
| Agent B (blind) | **0.783** | 16.833 | 0.0640 | 0.50 | 1.00 | 1.00 |
| Agent C (named) | 0.883 | 15.833 | 0.0593 | 0.50 | 1.00 | 1.00 |
| Agent D v2.1 (tools) | 0.914 | 9.774 | **0.0281** | 0.50 | 0.83 | 0.83 |
| LR basic | 0.904 | 10.286 | 0.0316 | 0.50 | 1.00 | 1.00 |
| LR extended | 0.912 | **9.760** | **0.0281** | 0.50 | 0.83 | 1.00 |
| KNN k=10 | 1.333 | 10.000 | 0.0333 | 0.50 | 0.67 | 0.83 |

Agent D v2.1 is numerically almost identical to the extended regression
anchor. Small differences come from rounded prompt inputs. It therefore does
not demonstrate incremental LM value over the regression tool.

Agent B has the lowest WAR MAE, while Agent C's access to player identity
slightly improves wRC+/OPS calibration but worsens WAR. With only six selected
cases, neither contrast is stable evidence of a general effect.

The two decliner cases are the clearest failure mode. B, C, and D all miss
both WAR directions. D also misses the wRC+/OPS direction for Masataka
Yoshida. Tool-vector changes alone do not capture all realized year-to-year
value changes.

## Leakage and tool-protocol audit

- The subject-excluded comparable pool removes every historical row belonging
  to the target player before nearest-neighbor fitting.
- Direct subject leakage in returned D comparables: 0/6 in v2 and 0/6 in v2.1.
- Comparable ID format violations: 0.
- The first v2 D agent mistakenly queried `find_comparables` with delta
  vectors rather than current 20–80 vectors. Its mean Jaccard@10 against the
  correct reference was 0.0.
- The gateway and schema now reject comparable query values outside [20, 80],
  and the v2 prompt explicitly requires the season-t base vector.
- A fresh v2.1 D agent used the exact base vectors. Mean and minimum
  Jaccard@10 were both 1.0 (10/10 overlap for every record).
- Correcting comparables did not materially change prediction errors. This is
  evidence that this D run relied on the regression anchor, not comparable
  reasoning.

## Interpretation

The benchmark machinery is promising: arm boundaries, subject exclusion,
overlapping-stratum scoring, and semantic tool-input validation exposed
failure modes that a single aggregate score would hide.

The current pilot is not evidence that LM reasoning beats classical models.
The strongest defensible conclusion is narrower: a tool-using agent can
reliably carry a fitted regression estimate through the prediction contract,
but no incremental value from identity knowledge or comparables has yet been
shown.

V1 and V2 use different tiny selected samples, so changes in MAE across runs
must not be read as improvement or regression. A useful next experiment is a
larger preregistered holdout with fresh agent instances per record/arm,
stateful tool authorization, query logging, repeated seeds, and paired
bootstrap intervals for B−C, B−D, and D−LR differences.

## Reproduction

Run:

```bash
python LM_experiments/scripts/07_score_agent_pilot_v2.py
python -m unittest discover -s LM_experiments/tests -v
```

No external model API was used. The predictions were produced by isolated
sub-agents in the current Codex session. This execution path is suitable for a
small diagnostic pilot only.
