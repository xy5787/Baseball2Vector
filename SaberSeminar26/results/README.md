# SaberSeminar 2026 results

- `headline_results.csv`: claims directly checked against the manuscript.
- `prediction_results.csv`: final forecast-safe model metrics (byte-identical copy of the validated `calibrated_prediction.csv`).
- `robustness_results.csv`: PA-stratified stability results (byte-identical copy of `stability_by_pa.csv`).
- `provenance.json`: canonical manuscript/data hashes and reproduced headline values.

The other small files are the supporting final-run outputs used to regenerate figures, tables, or paired comparisons. Large row-level bootstrap/matching files are regenerated and remain uncommitted.
