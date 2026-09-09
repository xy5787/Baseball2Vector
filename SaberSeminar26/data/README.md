# Data

`processed/` contains the smallest redistributable inputs needed to validate the paper:

- `v3_results.csv`: published 2021–2025 representation output (SHA-256 `70bea280...0873`).
- `player_id_audit.csv` and `chadwick_registry_subset.csv`: stable-ID provenance.
- `player_seasons.csv`: deterministic merge used by the final analyses.
- `transitions.csv`: adjacent-season stable-ID table (SHA-256 `62f69a78...900`).

Place any FanGraphs source export in `data/raw/`. That directory is intentionally ignored by Git.
Run `python scripts/build_representation.py` to verify the committed merge, or add `--write` to rebuild it.
