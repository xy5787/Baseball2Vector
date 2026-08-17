# Baseball2Vector final academic revision

This timestamped directory is an isolated, reproducible revision run. Nothing
in `paper/` was edited. The authoritative original SHA-256 remained
`a873b4beb3b9acbcdd44c5f13d859b8c2a326241234ef49794fe20ff65e9f00d`.

## Scientific result

On the fixed 2024→2025 holdout (358 stable-ID players), the forecast-safe B2V
Ridge had MAE 18.027 versus 19.220 for calibrated current-wRC+ Ridge. The paired
MAE improvement was 1.193 (95% player-bootstrap CI 0.403–2.012). The combined
scalar+B2V+age+PA model improved on scalar+age+PA by 1.125 (0.304–1.929), but
did not improve on B2V+age+PA. Age and PA had no clear incremental MAE gain.

## Inputs

- `data_intermediate/player_seasons_with_ids.csv`: copied stable-ID artifact,
  SHA-256 `0c810b9ea6aab754cd389ae7c5ab401a139bb251c6490a84b6e4b71d6cd07963`.
- Repository source: `data/processed/v3_results.csv`, SHA-256
  `70bea280ec725ac78376e94edf41fae6fd036fe8b7a0b71270869eb1cfdf0873`.
- Original manuscript: `paper/baseball2vector_en.tex`, SHA-256 above.
- Constituent raw statistics and pre-min-max dimension scores were unavailable.

Exact paths, modification times, versions, and hashes are in
`results/input_provenance.json`; fixed settings are in
`config/experiment_config.json`.

## Reproduction

Run from this directory:

```bash
bash experiments/scripts/run_all.sh
```

Equivalent individual commands:

```bash
python experiments/scripts/run_calibrated_baselines.py
python experiments/scripts/audit_temporal_scaling.py
python experiments/scripts/run_matching_analysis.py
python experiments/scripts/run_stability_analysis.py
python experiments/scripts/generate_revised_figures.py
python experiments/scripts/generate_tables_and_audit.py
uv run --isolated --python 3.12 --with pytest --with pandas --with numpy --with scipy --with scikit-learn python -m pytest -q experiments/tests
(cd manuscript_preview && ../tools/tectonic baseball2vector_en_revision.tex --keep-logs --keep-intermediates)
python experiments/scripts/render_pdf_pages.py
```

Seed 42 is used for random pairs and bootstrap resampling; each reported
interval uses 2,000 resamples. The Ridge grid is
`[0.01, 0.1, 1, 10, 100, 1000]`, selected by five-fold stable-player-ID
`GroupKFold` using training-period MAE.

## Principal artifacts

- Revised source/PDF/diff: `manuscript_preview/`
- Machine-readable estimates: `results/`
- Generated LaTeX tables: `tables/`
- Corrected figures in PDF and PNG: `figures/`
- Tests: `experiments/tests/`
- Complete Korean report: `report_ko.md`
- Commands and runtime output: `logs/`

The host lacked `latexmk`, `pdflatex`, and the repository's referenced
`cvpr.sty`. The preview therefore uses standard article packages and the
official Tectonic 0.16.9 binary stored under `tools/`. Six PDF pages were
rendered to `manuscript_preview/rendered_pages/` and visually inspected.
