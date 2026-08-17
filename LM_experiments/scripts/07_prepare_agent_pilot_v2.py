"""Prepare six subject-excluded v2 agent-pilot records."""

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


def main() -> None:
    cfg = load_config()
    rng = np.random.default_rng(cfg["seed"] + 1)
    root = Path(__file__).parents[1]
    out = root / "outputs" / "agent_pilot_v2"
    req = out / "requests"
    req.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(cfg["output"]["tables_dir"] / "transitions_stratified.csv")
    window = cfg["transitions"]["windows"][-1]
    gt = gt[(gt.season_t == window[0]) & (gt.season_t1 == window[1])]
    v1 = json.loads((root / "outputs" / "agent_pilot" / "manifest.json").read_text())
    excluded = {row["record_id"] for row in v1["records"]}
    records = []
    number = 1
    for stratum in ("elite", "strength_up", "decliner"):
        candidates = gt[(gt.stratum == stratum) & ~gt.record_id.isin(excluded)]
        chosen = rng.choice(candidates.index.to_numpy(), size=2, replace=False)
        for index in chosen:
            row = gt.loc[index]
            pilot_id = f"pilot_v2_{number:03d}"
            values = row.to_dict()
            for key, value in values.items():
                if isinstance(value, float):
                    values[key] = round(value, 3 if "ops" in key else 1)
            values["anon_id"] = anon_id(str(row.player_id), cfg["lm_arms"]["anon_salt"])
            for arm in "bcd":
                prompt = render(arm, version="v2", **values)
                envelope = build_request(str(row.record_id), arm, prompt)
                (req / f"{pilot_id}_arm_{arm}.json").write_text(
                    json.dumps(envelope, ensure_ascii=False, indent=2)
                )
            records.append(
                {"pilot_id": pilot_id, "record_id": row.record_id, "stratum": stratum}
            )
            number += 1
    manifest = {
        "run_id": "AGENT_PILOT_NONPUBLICATION_V2",
        "seed": cfg["seed"] + 1,
        "prompt_version": "v2",
        "subject_exclusion": True,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for row in records:
        print(row["pilot_id"], row["stratum"], row["record_id"])


if __name__ == "__main__":
    main()
