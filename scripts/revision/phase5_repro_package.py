"""Phase 5: Reproducibility packaging.

Records the exact package versions and random-fold assignments used to produce the Phase 1-3
revision outputs in this session, so they can be audited or exactly reproduced later.

Outputs:
  outputs/revision/phase5_reproducibility/environment_used.txt
  outputs/revision/phase5_reproducibility/phase1_fold_assignments.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from baseball2vec.roi import build_delta_dataset  # noqa: E402

DATA = ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase5_reproducibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
PACKAGES = ["pandas", "numpy", "scikit-learn", "matplotlib", "scipy", "pybaseball", "torch", "seaborn"]


def write_environment_file() -> None:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    installed = {
        line.split("==")[0].lower(): line.strip()
        for line in freeze.splitlines()
        if "==" in line
    }

    lines = [
        "# Environment actually used to produce outputs/revision/phase{0,1,2,3,4}/*",
        f"# Python: {sys.version.split()[0]}",
        "",
        "## Packages required by requirements.txt, versions actually installed this session:",
    ]
    for pkg in PACKAGES:
        key = pkg.lower()
        status = installed.get(key, "NOT INSTALLED in this session")
        lines.append(f"{pkg}: {status}")

    lines += [
        "",
        "## Deviation from requirements.txt",
        "requirements.txt pins pandas==3.0.2, numpy==2.4.4 (Python 3.12). This session's sandbox",
        f"had pandas=={installed.get('pandas', '?')}, numpy=={installed.get('numpy', '?')} pre-installed",
        f"under Python {sys.version.split()[0]}, and torch/seaborn were not installed at all (not needed",
        "for Phases 0-4, which never call joint_vae.py or the seaborn-based parts of viz.py).",
        "matplotlib and pybaseball were installed fresh this session, pinned to the versions",
        "requirements.txt already specified (3.10.9 / 2.2.7 respectively) -- those two match exactly.",
        "Recommendation: re-pin requirements.txt to the versions below before treating Phases 0-4",
        "as bit-exact reproducible, or note the drift explicitly if reproducing on Python 3.12.",
    ]

    (OUT_DIR / "environment_used.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved: {OUT_DIR / 'environment_used.txt'}")


def write_fold_assignments() -> None:
    df = pd.read_csv(DATA)
    delta_clean = build_delta_dataset(df)
    groups = delta_clean["Name"].values

    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_col = pd.Series(-1, index=delta_clean.index, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(gkf.split(delta_clean, groups=groups)):
        fold_col.iloc[test_idx] = fold_idx

    out = delta_clean[["Name", "Season_From", "Season_To"]].copy()
    out["group_kfold_test_fold"] = fold_col
    out.to_csv(OUT_DIR / "phase1_fold_assignments.csv", index=False)
    print(f"Saved: {OUT_DIR / 'phase1_fold_assignments.csv'} ({len(out)} rows, seed={SEED})")

    # Sanity: no player split across folds.
    per_player_folds = out.groupby("Name")["group_kfold_test_fold"].nunique()
    assert (per_player_folds == 1).all(), "A player's transitions span multiple folds!"
    print("Verified: every player's transitions fall in exactly one GroupKFold test fold.")


def main() -> None:
    write_environment_file()
    write_fold_assignments()


if __name__ == "__main__":
    main()
