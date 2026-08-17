"""Prepare a deterministic five-record, three-arm agent pilot manifest."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.anonymize import anon_id  # noqa: E402
from lm_experiments.lm_arms.client import build_request  # noqa: E402
from lm_experiments.lm_arms.prompts import render  # noqa: E402


def rounded_values(row: pd.Series) -> dict:
    values = row.to_dict()
    for key, value in values.items():
        if isinstance(value, float):
            values[key] = round(value, 3 if "ops" in key else 1)
    return values


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    tables = cfg["output"]["tables_dir"]
    transitions = pd.read_csv(tables / "transitions_stratified.csv")
    final_window = tuple(cfg["transitions"]["windows"][-1])
    test = transitions[
        (transitions["season_t"] == final_window[0])
        & (transitions["season_t1"] == final_window[1])
    ]
    strata = ["elite", "strength_up", "weakness_up", "decliner", "baseline"]
    output_dir = Path(__file__).parents[1] / "outputs" / "agent_pilot"
    requests_dir = output_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for number, stratum in enumerate(strata, start=1):
        candidates = test[test["stratum"] == stratum]
        if candidates.empty:
            raise RuntimeError(f"no exclusive test candidate for {stratum}")
        row = candidates.iloc[int(rng.integers(0, len(candidates)))]
        pilot_id = f"pilot_{number:03d}"
        values = rounded_values(row)
        values["anon_id"] = anon_id(
            str(row["player_id"]), cfg["lm_arms"]["anon_salt"]
        )
        request_paths = {}
        for arm in ("b", "c", "d"):
            prompt = render(
                arm, version=cfg["lm_arms"]["prompt_version"], **values
            )
            request = build_request(str(row["record_id"]), arm, prompt)
            request_path = requests_dir / f"{pilot_id}_arm_{arm}.json"
            request_path.write_text(
                json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            request_paths[arm] = str(request_path.relative_to(output_dir.parent.parent))
        records.append(
            {
                "pilot_id": pilot_id,
                "record_id": row["record_id"],
                "exclusive_stratum": stratum,
                "season_t": int(row["season_t"]),
                "season_t1": int(row["season_t1"]),
                "requests": request_paths,
            }
        )

    manifest = {
        "run_id": "AGENT_PILOT_NONPUBLICATION",
        "seed": cfg["seed"],
        "selection": "one seeded record per exclusive test stratum",
        "records": records,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)
    for record in records:
        print(record["pilot_id"], record["exclusive_stratum"], record["record_id"])


if __name__ == "__main__":
    main()
