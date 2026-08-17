"""Restricted local tool gateway for the non-publication Arm-D agent pilot."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.tools import ToolBox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        required=True,
        choices=[
            "get_player_vector",
            "apply_tool_delta",
            "predict_value_delta",
            "find_comparables",
            "check_delta_feasibility",
        ],
    )
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    tool_input = json.loads(args.input_json)
    if not isinstance(tool_input, dict):
        raise SystemExit("--input-json must decode to an object")
    result = ToolBox(load_config()).dispatch(args.name, tool_input)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
