# Baseball2Vector — SaberSeminar 2026

The canonical manuscript is `paper/baseball2vector_en.tex`. Historical previews and dated run directories are retained in Git history, not in the public tree.

## Reproduce and validate

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_pipeline.py
```

The default command verifies the committed representation, headline results, manuscript references, temporal split, and artifact hashes. Use `--regenerate` to rerun the bootstrap analyses and rebuild paper figures/tables. Raw inputs, when needed for the original encoder training, belong in `data/raw/`.

## Layout

- `src/baseball2vec/`: representation and final validation implementation
- `scripts/`: project-local entrypoints
- `data/processed/`: redistributable analysis inputs
- `results/`: canonical and supporting final results
- `paper/`: the one canonical TeX manuscript and only its referenced artifacts
- `tests/`: reproducibility, consistency, and leakage checks

The paper's direct numeric claims are indexed in `results/headline_results.csv` and checked automatically against the manuscript and source result files.
