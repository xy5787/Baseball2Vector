# Baseball2Vector

Reproducibility repository for two independently runnable conference projects:

- [`SaberSeminar26/`](SaberSeminar26/) — canonical paper, representation code, final academic validation, and paper artifacts.
- [`SSAC27/`](SSAC27/) — rolling-origin prediction, constituent controls, Marcel comparison, comparable-player retrieval, encoder cross-checks, and the extended window.

Each project owns its `src/`, `scripts/`, `data/`, `results/`, dependency files, and tests. SSAC does not import from SaberSeminar. Raw data belongs only in each project's ignored `data/raw/` directory.

## Validation

```bash
cd SaberSeminar26 && python scripts/run_pipeline.py
cd ../SSAC27 && python -m pytest -q tests
```

The canonical Saber manuscript is `SaberSeminar26/paper/baseball2vector_en.tex`; historical previews and timestamped runs remain available through Git history. Headline claims are machine-readable in each project's `results/headline_results.csv`.

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). No redistribution license has been selected yet; `LICENSE` intentionally reserves rights until the copyright holder chooses one.
