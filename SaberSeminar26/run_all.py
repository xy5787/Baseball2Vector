"""Baseball2Vec — single-command pipeline orchestrator.

Usage
-----
    python run_all.py              # full pipeline (01 → 04)
    python run_all.py --from 04   # start from a specific step
    python run_all.py --refetch   # force re-fetch raw data from pybaseball

Steps
-----
    01  Build dataset (fetch / load cache → preprocess)
    02  Baseline comparison (Z-Score / PCA → heatmap)
    03  Joint Encoder VAE (train → v3_results.csv + figures)
    04  ROI analysis (ΔTool regression → figures + tables)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = {
    "01": Path("scripts/01_build_dataset.py"),
    "02": Path("scripts/02_baseline_comparison.py"),
    "03": Path("scripts/03_joint_vae.py"),
    "04": Path("scripts/04_roi_analysis.py"),
}

PYTHON = sys.executable

parser = argparse.ArgumentParser(description="Run the Baseball2Vec pipeline end-to-end")
parser.add_argument(
    "--from",
    dest="start_from",
    default="01",
    choices=SCRIPTS.keys(),
    metavar="STEP",
    help="Start from this step (default: 01)",
)
parser.add_argument(
    "--refetch",
    action="store_true",
    help="Pass --refetch to step 01 to force re-fetch from pybaseball",
)
args = parser.parse_args()

steps = [k for k in sorted(SCRIPTS) if k >= args.start_from]
print(f"Running steps: {', '.join(steps)}\n")

for step in steps:
    script = SCRIPTS[step]
    cmd = [PYTHON, str(script)]
    if step == "01" and args.refetch:
        cmd.append("--refetch")

    print("=" * 60)
    print(f"  Step {step}: {script.name}")
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\nStep {step} failed (exit code {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s\n")

print("=" * 60)
print("All steps completed successfully.")
print("Figures -> outputs/figures/")
print("Tables  -> outputs/tables/")
print("=" * 60)
