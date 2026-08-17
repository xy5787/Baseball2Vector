# Arm C — Named player reasoning (v1)

You are predicting an MLB hitter's observed change in value from one named
season to the next. This arm intentionally exposes identity and seasons so its
performance can measure the contribution of text knowledge or memorization.

Each tool is on the 20–80 scouting scale (50 is league average; 10 points is
one standard deviation). The delta is an observed vector change, not a
hypothetical intervention.

Player: **$player_name**  
Transition: **$season_t → $season_t1**

| Input | Power | Contact | Discipline | Defense | Speed |
|---|---:|---:|---:|---:|---:|
| Vector in $season_t | $vec_t_power | $vec_t_contact | $vec_t_discipline | $vec_t_defense | $vec_t_speed |
| Observed delta | $delta_vec_power | $delta_vec_contact | $delta_vec_discipline | $delta_vec_defense | $delta_vec_speed |

Predict the accompanying `delta_war`, `delta_wrc`, and `delta_ops`. You may use
knowledge associated with the named player and seasons, but do not fabricate
facts when uncertain. Submit exactly one `submit_prediction` tool call. Keep
`rationale` brief and state whether player-specific knowledge affected it.
