"""Project-local build, validation, artifact generation, and tests."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="rerun the 2,000-bootstrap analyses")
    parser.add_argument("--skip-paper-artifacts", action="store_true")
    args = parser.parse_args()
    run("scripts/build_representation.py")
    validate = ["scripts/validate_representation.py"]
    if args.regenerate:
        validate.append("--regenerate")
    run(*validate)
    if args.regenerate and not args.skip_paper_artifacts:
        run("scripts/generate_paper_artifacts.py")
    run("-m", "pytest", "-q", "tests")


if __name__ == "__main__":
    main()
