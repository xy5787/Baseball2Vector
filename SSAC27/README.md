# Baseball2Vector — SSAC 2027

This is a self-contained experiment package. It never imports from `../SaberSeminar26`.

## Run and validate

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_all.py              # all tasks (Task 6 needs local raw exports)
python scripts/run_all.py --task 1     # one task
python -m pytest -q tests
```

Tasks 1, 2, 4, and 5 use only `data/processed/transitions.csv`. Tasks 3 and 6 also need the local raw exports whose placement is documented in `data/README.md`. Results are grouped under `results/task1/` through `results/task6/`; paper-level values are collected in `results/headline_results.csv`, and provenance is under `results/provenance/`.
