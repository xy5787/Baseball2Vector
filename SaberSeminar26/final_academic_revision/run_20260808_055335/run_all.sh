#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$RUN_ROOT/../.." && pwd)"
PYTHONHASHSEED=0
export PYTHONHASHSEED

cd "$REPO_ROOT"

uv run python "$RUN_ROOT/experiments/scripts/run_calibrated_baselines.py" \
  2>&1 | tee "$RUN_ROOT/logs/run_calibrated_baselines.log"
uv run python "$RUN_ROOT/experiments/scripts/audit_temporal_scaling.py" \
  2>&1 | tee "$RUN_ROOT/logs/audit_temporal_scaling.log"
uv run python "$RUN_ROOT/experiments/scripts/run_matching_analysis.py" \
  2>&1 | tee "$RUN_ROOT/logs/run_matching_analysis.log"
uv run python "$RUN_ROOT/experiments/scripts/run_stability_analysis.py" \
  2>&1 | tee "$RUN_ROOT/logs/run_stability_analysis.log"
uv run python "$RUN_ROOT/experiments/scripts/generate_artifacts.py" \
  2>&1 | tee "$RUN_ROOT/logs/generate_artifacts.log"

cp "$RUN_ROOT/figures/figure1_alignment_revised.pdf" "$RUN_ROOT/manuscript_preview/figures/"
cp "$RUN_ROOT/figures/figure2_profile_diversity_revised.pdf" "$RUN_ROOT/manuscript_preview/figures/"
cp "$RUN_ROOT/figures/figure3_delta_association_revised.pdf" "$RUN_ROOT/manuscript_preview/figures/"

uv run --with pytest pytest -q "$RUN_ROOT/experiments/tests" \
  2>&1 | tee "$RUN_ROOT/logs/pytest.log"
