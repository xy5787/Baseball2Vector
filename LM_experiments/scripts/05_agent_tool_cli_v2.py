"""Subject-excluding local tool gateway for AGENT_PILOT_NONPUBLICATION_V2."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.tools import ToolBox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument(
        "--name",
        required=True,
        choices=[
            "apply_tool_delta",
            "predict_value_delta",
            "find_comparables",
            "check_delta_feasibility",
        ],
    )
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (root / "outputs" / "agent_pilot_v2" / "manifest.json").read_text()
    )
    matches = [row for row in manifest["records"] if row["pilot_id"] == args.pilot_id]
    if len(matches) != 1:
        raise SystemExit("unknown or ambiguous pilot-id")
    subject = matches[0]["record_id"].split("__", 1)[0]
    tool_input = json.loads(args.input_json)
    result = ToolBox(load_config(), excluded_player_id=subject).dispatch(
        args.name, tool_input
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
