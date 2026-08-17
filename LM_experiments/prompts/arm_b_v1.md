# Arm B — Blind vector reasoning (v1)

You are predicting an MLB hitter's observed change in value from period t to
t+1. The subject's identity, team, and calendar seasons are intentionally
hidden. Do not infer or invent them.

Each tool is on the 20–80 scouting scale (50 is league average; 10 points is
one standard deviation). The delta is an observed vector change, not a
hypothetical intervention.

Anonymous subject: `$anon_id`

| Input | Power | Contact | Discipline | Defense | Speed |
|---|---:|---:|---:|---:|---:|
| Vector at t | $vec_t_power | $vec_t_contact | $vec_t_discipline | $vec_t_defense | $vec_t_speed |
| Observed delta | $delta_vec_power | $delta_vec_contact | $delta_vec_discipline | $delta_vec_defense | $delta_vec_speed |

Predict the accompanying `delta_war`, `delta_wrc`, and `delta_ops` using only
these numbers. Do not perform hidden lookups or identify the subject. Submit
exactly one `submit_prediction` tool call. Set `comparables` to null because
this arm has no comparable-search tool. Keep `rationale` brief.
