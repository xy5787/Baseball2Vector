#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/i2slab4/miniconda3/envs/Nymeria/bin/python"
EXP_DIR="academic_validation_experiments"
mkdir -p "${EXP_DIR}/logs"

"${PYTHON_BIN}" "${EXP_DIR}/scripts/00_audit_and_ids.py" 2>&1 | tee "${EXP_DIR}/logs/00_audit_and_ids.log"
"${PYTHON_BIN}" "${EXP_DIR}/scripts/01_prospective_prediction.py" 2>&1 | tee "${EXP_DIR}/logs/01_prospective_prediction.log"
"${PYTHON_BIN}" "${EXP_DIR}/scripts/02_stability_ablation.py" 2>&1 | tee "${EXP_DIR}/logs/02_stability_ablation.log"
"${PYTHON_BIN}" "${EXP_DIR}/scripts/03_profile_diversity.py" 2>&1 | tee "${EXP_DIR}/logs/03_profile_diversity.log"
"${PYTHON_BIN}" "${EXP_DIR}/scripts/04_generate_report.py" 2>&1 | tee "${EXP_DIR}/logs/04_generate_report.log"
"${PYTHON_BIN}" -m pytest "${EXP_DIR}/tests" -q 2>&1 | tee "${EXP_DIR}/logs/tests.log"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="${EXP_DIR}/manuscript_preview" "${EXP_DIR}/manuscript_preview/validation_preview.tex" 2>&1 | tee "${EXP_DIR}/logs/latex_preview.log"
else
  echo "SKIPPED: latexmk is not installed in this environment." | tee "${EXP_DIR}/logs/latex_preview.log"
fi
