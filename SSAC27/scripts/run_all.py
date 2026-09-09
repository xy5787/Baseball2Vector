"""Regenerate every number in SSAC27/results/ in one command.

    python SSAC27/scripts/run_all.py            # all tasks
    python SSAC27/scripts/run_all.py --task 1   # one task

Tasks are independent and can be run in any order or on their own.

Tasks 1-5 are anchored to the archived 2021-2025 tool scores, so they reproduce
the published Finding 2 numbers exactly. Task 3 additionally reads local raw exports for Marcel history. Task 6 rebuilds the cohort from the raw
2019-2025 exports to widen the study to five rolling origins; it is an
extension, not a replacement.

Every task runs three targets: next-season wRC+ (the published one), next-season
WAR, and next-season WAR per 600 PA. wRC+ is batting-only and prices neither
Defense nor Speed -- two of the five tools -- so WAR is the target on which the
tool structure has something to gain that wRC+ cannot show. Counting WAR is also
a playing-time forecast, which is why the per-600 rate is reported beside it.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

TASKS: dict[str, str] = {
    "1": "task1_constituent_control",
    "2": "task2_rolling_origin",
    "3": "task3_marcel_incremental",
    "4": "task4_comparable_players",
    "5": "task5_encoder_crosscheck",
    "6": "task6_extended_window",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        action="append",
        choices=sorted(TASKS),
        help="run only these tasks (repeatable); default is all",
    )
    args = parser.parse_args()
    selected = args.task or sorted(TASKS)

    for task in selected:
        module_name = TASKS[task]
        print("=" * 72)
        print(f"Task {task}: {module_name}")
        print("=" * 72)
        started = time.time()
        importlib.import_module(module_name).main()
        print(f"\n  Task {task} finished in {time.time() - started:.1f}s\n")


if __name__ == "__main__":
    main()
