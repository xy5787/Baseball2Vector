# Baseball2Vector academic validation experiments

This directory is an isolated Stage A workspace. It reads the authoritative inputs from the
repository but never writes to `paper/`, `data/`, `outputs/`, or the original source tree.

The committed processed input does not contain the raw constituent statistics or stable player
IDs. The pipeline therefore retrieves the Chadwick Register only for identity and birth-year
metadata, maps rows conservatively, and excludes unresolved identities. Analyses requiring the
missing FanGraphs raw export are emitted with an explicit `not_run_missing_raw` status rather
than approximated.

Run from the repository root with the existing Python 3.12 environment:

```bash
/home/i2slab4/miniconda3/envs/Nymeria/bin/python academic_validation_experiments/scripts/00_audit_and_ids.py
/home/i2slab4/miniconda3/envs/Nymeria/bin/python academic_validation_experiments/scripts/01_prospective_prediction.py
/home/i2slab4/miniconda3/envs/Nymeria/bin/python academic_validation_experiments/scripts/02_stability_ablation.py
/home/i2slab4/miniconda3/envs/Nymeria/bin/python academic_validation_experiments/scripts/03_profile_diversity.py
/home/i2slab4/miniconda3/envs/Nymeria/bin/python academic_validation_experiments/scripts/04_generate_report.py
/home/i2slab4/miniconda3/envs/Nymeria/bin/python -m pytest academic_validation_experiments/tests -q
```

`run_stage_a.sh` records the same commands and redirects each command's output to `logs/`.

The manuscript hash is checked before and after the run. Stage A does not edit the authoritative
manuscript. Proposed integration language is stored only in `manuscript_preview/`.

