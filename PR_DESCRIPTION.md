# PR: Baseball2Vec — Clean Repo for Saberseminar 2026

**Branch:** `cleanup/saberseminar-prep`  
**Reviewer note:** Numbers in the abstract were locked before this PR was opened. All changes are structural; no model logic was modified.

---

## What changed

### Repository structure
The original `Fivetools_VAE/` monolith was split into a proper Python package and thin scripts:

```
src/baseball2vec/
    data.py        — load_raw(), preprocess(), Bayesian stabilization
    tools.py       — feature groups, direction flip, Z-Score, 20-80 scaling, pentagon area
    baselines.py   — Z-Score avg and PCA baselines
    joint_vae.py   — JointVAE model, KL/alignment/disentanglement losses, train()
    roi.py         — build_delta_dataset(), run_regression(), compute_individual_roi()
    viz.py         — all figure functions (outputs/figures/)

scripts/
    01_build_dataset.py  — fetch/cache raw FanGraphs data
    02_baseline_comparison.py
    03_joint_vae.py      — train VAE → v3_results.csv + all v3 figures
    04_roi_analysis.py   — regression + counterfactual ROI figures and tables

run_all.py   — orchestrator (--from STEP, --refetch flags)
```

### Verified unchanged (bit-identical)

| Abstract claim | Expected | Verified |
|---|---|---|
| Pentagon-WAR Pearson r | 0.81 (full: 0.8060958222864365) | ✓ |
| ΔwRC+ LR R² (5-fold CV) | 0.7068 | ✓ |
| Byron Buxton Contact ROI | +14.02 wRC+ | ✓ |
| Dataset rows | 2,309 player-seasons | ✓ |

Two consecutive runs of `python run_all.py --from 04` produced SHA256-identical `individual_roi.csv` and `lr_results.csv`.

### Reproducibility fixes applied
- `torch.backends.cudnn.deterministic = True` — was missing from scripts
- RF `n_jobs=1` (was `-1`) — parallel float summation caused sub-ULP drift between runs
- Fixed sklearn 1.8 API: `pd_res["grid_values"]` replaces `pd_res["values"]` in `plot_partial_dependence`

### Added
- `requirements.txt` — pinned to the conda env's exact versions
- `tests/test_reproducibility.py` — 4 assertions covering all abstract numbers (run with `pytest`)
- `data/processed/v3_results.csv` — committed so step 04 runs without FanGraphs access
- `outputs/.gitkeep`, `outputs/figures/.gitkeep`, `outputs/tables/.gitkeep`

---

## What was NOT changed

- No model architecture changes (JointVAE, Cholesky covariance, loss weights)
- No data changes (same 2,309 player-seasons, same stabilization thresholds)
- `Fivetools_VAE/` left completely untouched

---

## Items requiring researcher decision before merging

1. **LICENSE** — Repo has no license file. MIT recommended for a public Saberseminar submission; confirm before opening to the public.

2. **Old scripts in `Fivetools_VAE/`** — 10+ superseded scripts (`Multiyear_5tools.py`, `Multiyear_5tools_VAE_v2.py`, `AgingCurve_Analysis.py`, etc.) remain untouched. Options:
   - Move to `archive/` in this repo
   - Leave as-is in the original directory (separate from this repo)
   - Delete if no longer needed

3. **Raw data cache** — `data/raw/batting_stats_2021_2025.csv` does not exist in this repo. It must be generated on a machine with FanGraphs access:
   ```bash
   conda activate baseballEmbedding
   python scripts/01_build_dataset.py --refetch
   ```
   Once generated, `python run_all.py` (full pipeline) will work end-to-end.

4. **Step 03 re-training** — `03_joint_vae.py` has not been end-to-end verified in the clean repo (requires raw data). The abstract numbers come from `v3_results.csv` which was carried over verified from the original pipeline. If retraining from scratch, confirm numbers match before Saberseminar submission.

---

## How to run

```bash
# Step 04 only (uses committed v3_results.csv — no FanGraphs needed)
python run_all.py --from 04

# Full pipeline (requires data/raw/batting_stats_2021_2025.csv)
python run_all.py

# Reproducibility tests
pytest tests/test_reproducibility.py -v
```

Expected test output:
```
tests/test_reproducibility.py::test_row_count PASSED
tests/test_reproducibility.py::test_pentagon_war_correlation PASSED
tests/test_reproducibility.py::test_lr_r2 PASSED
tests/test_reproducibility.py::test_buxton_roi PASSED
4 passed in ~151s
```
