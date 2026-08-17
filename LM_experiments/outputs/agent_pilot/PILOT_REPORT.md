# AGENT_PILOT_NONPUBLICATION

This is a five-record pipeline pilot, not a publication-grade LM benchmark.
One 2024→2025 record was selected with seed 42 from each exclusive stratum,
and a fresh no-history agent was used for each record/arm pair (15 runs).

## Selected records

| Pilot | Exclusive stratum | Record |
|---|---|---|
| pilot_001 | elite | Bobby Witt Jr.__2024_2025 |
| pilot_002 | strength_up | Alec Burleson__2024_2025 |
| pilot_003 | weakness_up | Sean Murphy__2024_2025 |
| pilot_004 | decliner | Francisco Alvarez__2024_2025 |
| pilot_005 | baseline | Jonathan India__2024_2025 |

## Overall MAE (n=5)

| Model | WAR | wRC+ | OPS |
|---|---:|---:|---:|
| agent B | 0.900 | 17.280 | 0.05640 |
| agent C | **0.440** | 9.800 | 0.02780 |
| agent D | 0.568 | 7.972 | 0.02799 |
| LR basic | 0.572 | 7.075 | 0.02801 |
| LR extended | 0.566 | 8.029 | 0.02817 |
| KNN-10 | 0.518 | **4.700** | **0.01576** |

Bold values identify the lowest MAE, allowing ties at displayed precision.
With only five records, these ranks are descriptive and unstable.

## Pilot findings

- Naming the player (C vs B) substantially improved all three MAEs in this
  sample, consistent with either useful player knowledge or memorization.
- D nearly reproduced the extended regression because its required numeric
  anchor comes from that model. It did not clearly improve on the regression.
- On the sole decliner, B predicted the WAR sign correctly while C and D did
  not. One case is not evidence of a general advantage.
- D's comparable tool returned the subject's own historical season in 3/5
  cases, breaking the intended blind identity boundary. D's jaccard@10 is
  mechanically 1.0 because the same KNN tool defines the reference set.
- C returned archetype descriptions rather than player-season IDs for 4/5
  records, so its jaccard result is not semantically valid. A future prompt
  version should require `Name (YYYY)` IDs or null.

## Validity limits

Agents shared the same underlying workspace permissions even though each was
started without conversation history. Tool restrictions were enforced by
instructions, not a hard sandbox. Model/version and generation parameters were
not pinned like an external API run. Do not cite these numbers as the final
B/C/D benchmark.
