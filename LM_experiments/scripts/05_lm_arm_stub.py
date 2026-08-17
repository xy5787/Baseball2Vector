"""Phase 5: build one LM-arm request locally; never call an external API.

Examples:
    python scripts/05_lm_arm_stub.py --arm b \
      --record-id "Juan Soto__2021_2022"

    python scripts/05_lm_arm_stub.py --arm d \
      --record-id "Juan Soto__2021_2022" \
      --output outputs/requests/soto_arm_d_v1.json

The JSON envelope contains local record_id metadata for later scoring. A future
provider adapter must send only its prompt and tools fields, never the envelope
record_id, for blind Arms B and D. See API_INTEGRATION.md.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.anonymize import anon_id  # noqa: E402
from lm_experiments.lm_arms.client import build_request  # noqa: E402
from lm_experiments.lm_arms.prompts import render  # noqa: E402


def rounded_template_vars(row: pd.Series) -> dict:
    """Return prompt inputs with deterministic display rounding."""
    values = row.to_dict()
    for key, value in values.items():
        if not isinstance(value, float):
            continue
        values[key] = round(value, 3 if "ops" in key else 1)
    return values


def audit_masking(arm: str, prompt: str, row: pd.Series) -> None:
    """Fail closed if a blind prompt exposes identity or calendar seasons."""
    unresolved = re.findall(r"\$[A-Za-z_][A-Za-z0-9_]*", prompt)
    if unresolved:
        raise ValueError(f"unresolved prompt placeholders: {sorted(set(unresolved))}")
    if arm in {"b", "d"}:
        if str(row["player_name"]) in prompt:
            raise ValueError(f"Arm {arm.upper()} prompt leaked player_name")
        years = {str(int(row["season_t"])), str(int(row["season_t1"]))}
        if any(re.search(rf"\b{year}\b", prompt) for year in years):
            raise ValueError(f"Arm {arm.upper()} prompt leaked calendar season")
    else:
        if str(row["player_name"]) not in prompt:
            raise ValueError("Arm C prompt must expose player_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=["b", "c", "d"], required=True)
    parser.add_argument(
        "--record-id", required=True, help="record_id from transitions_stratified.csv"
    )
    parser.add_argument(
        "--output", type=Path, help="optional local JSON request-envelope path"
    )
    parser.add_argument(
        "--show-ground-truth",
        action="store_true",
        help="print labels after the request for researcher inspection only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    path = cfg["output"]["tables_dir"] / "transitions_stratified.csv"
    transitions = pd.read_csv(path)
    rows = transitions[transitions["record_id"] == args.record_id]
    if rows.empty:
        raise SystemExit(f"record_id {args.record_id!r} not found in {path}")
    if len(rows) != 1:
        raise SystemExit(f"record_id {args.record_id!r} is not unique")
    row = rows.iloc[0]

    values = rounded_template_vars(row)
    values["anon_id"] = anon_id(
        str(row["player_id"]), cfg["lm_arms"]["anon_salt"]
    )
    version = cfg["lm_arms"]["prompt_version"]
    prompt = render(args.arm, version=version, **values)
    audit_masking(args.arm, prompt, row)
    request = build_request(args.record_id, args.arm, prompt)

    print("=" * 60)
    print(f"Arm {args.arm.upper()} request stub — prompt {version}")
    print("=" * 60)
    print(prompt)
    print("\n--- tools offered ---")
    for tool in request["tools"]:
        print(f"  {tool['name']}")
    print("\n[stub only — no network or API call was made]")

    if args.output:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved local request envelope -> {output_path}")

    if args.show_ground_truth:
        print(
            "\nResearcher-only ground truth: "
            f"delta_war={row['delta_war']:.3f}, "
            f"delta_wrc={row['delta_wrc']:.3f}, "
            f"delta_ops={row['delta_ops']:.3f}"
        )


if __name__ == "__main__":
    main()
