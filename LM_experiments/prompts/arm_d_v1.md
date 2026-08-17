# Arm D — Blind vector reasoning with tools (v1)

You are predicting an MLB hitter's observed change in value from period t to
t+1. The subject's identity, team, and calendar seasons are intentionally
hidden. Do not use tool results to reverse-identify the subject.

Each tool is on the 20–80 scouting scale (50 is league average; 10 points is
one standard deviation). The delta is an observed vector change, not a
hypothetical intervention.

Anonymous subject: `$anon_id`

| Input | Power | Contact | Discipline | Defense | Speed |
|---|---:|---:|---:|---:|---:|
| Vector at t | $vec_t_power | $vec_t_contact | $vec_t_discipline | $vec_t_defense | $vec_t_speed |
| Observed delta | $delta_vec_power | $delta_vec_contact | $delta_vec_discipline | $delta_vec_defense | $delta_vec_speed |

Available tools:

- `predict_value_delta(delta_vec, base_vec)`: code-computed regression anchor.
- `check_delta_feasibility(tool, current_value, magnitude)`: empirical delta percentile.
- `find_comparables(vec5, k)`: nearest historical player-season IDs.
- `get_player_vector(player_id, season)`: vector for an available historical comparable.
- `apply_tool_delta(vec5, tool, magnitude)`: code-computed one-tool shift.

Use tools for numeric calculations; do not mentally recreate their arithmetic.
Predict `delta_war`, `delta_wrc`, and `delta_ops`, then make exactly one final
`submit_prediction` tool call. If comparables influenced the answer, return at
most 10 IDs from `find_comparables`. Briefly identify the tool outputs used in
`rationale`.
