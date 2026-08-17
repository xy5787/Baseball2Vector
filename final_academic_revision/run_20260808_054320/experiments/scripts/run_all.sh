#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$RUN_ROOT"

python experiments/scripts/run_calibrated_baselines.py 2>&1 | tee logs/run_calibrated_baselines.log
python experiments/scripts/audit_temporal_scaling.py 2>&1 | tee logs/audit_temporal_scaling.log
python experiments/scripts/run_matching_analysis.py 2>&1 | tee logs/run_matching_analysis.log
python experiments/scripts/run_stability_analysis.py 2>&1 | tee logs/run_stability_analysis.log
python experiments/scripts/generate_revised_figures.py 2>&1 | tee logs/generate_revised_figures.log
python experiments/scripts/generate_tables_and_audit.py 2>&1 | tee logs/generate_tables_and_audit.log
uv run --isolated --python 3.12 --with pytest --with pandas --with numpy --with scipy --with scikit-learn python -m pytest -q experiments/tests 2>&1 | tee logs/tests.log
(cd manuscript_preview && ../tools/tectonic baseball2vector_en_revision.tex --keep-logs --keep-intermediates) 2>&1 | tee logs/latex_compile.log
python experiments/scripts/render_pdf_pages.py 2>&1 | tee logs/render_pdf_pages.log
