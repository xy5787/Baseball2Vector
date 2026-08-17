"""Normalize, audit, and score the six-record subject-excluded v2 pilot."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.client import validate_prediction  # noqa: E402
from lm_experiments.lm_arms.tools import ToolBox  # noqa: E402
from lm_experiments.scoring import score  # noqa: E402


COMPARABLE_RE = re.compile(r"^.+ \(\d{4}\)$")


def main() -> None:
    cfg = load_config()
    root = Path(__file__).parents[1]
    pilot_dir = root / "outputs" / "agent_pilot_v2"
    manifest = json.loads((pilot_dir / "manifest.json").read_text())
    manifest_rows = pd.DataFrame(manifest["records"])
    record_map = dict(zip(manifest_rows["pilot_id"], manifest_rows["record_id"]))
    stratum_map = dict(zip(manifest_rows["pilot_id"], manifest_rows["stratum"]))

    raw = [
        json.loads(line)
        for line in (pilot_dir / "normalized_agent_predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    normalized = []
    for item in raw:
        prediction = validate_prediction(item, record_map[item["pilot_id"]])
        prediction.update(
            {
                "pilot_id": item["pilot_id"],
                "arm": f"agent_{item['arm']}",
                "selection_stratum": stratum_map[item["pilot_id"]],
                "run_id": manifest["run_id"],
            }
        )
        normalized.append(prediction)
    agent_predictions = pd.DataFrame(normalized)
    agent_arms = ["agent_b", "agent_c", "agent_d"]
    if "d_v21" in {item["arm"] for item in raw}:
        agent_arms.append("agent_d_v21")
    expected = {(p, arm) for p in record_map for arm in agent_arms}
    observed = set(zip(agent_predictions["pilot_id"], agent_predictions["arm"]))
    if observed != expected:
        raise ValueError(f"Pilot/arm coverage mismatch: missing={expected-observed}, extra={observed-expected}")

    agent_predictions.to_json(
        pilot_dir / "agent_predictions.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    csv_predictions = agent_predictions.copy()
    csv_predictions["comparables"] = csv_predictions["comparables"].map(
        lambda value: json.dumps(value, ensure_ascii=False) if value is not None else None
    )
    csv_predictions.to_csv(pilot_dir / "agent_predictions.csv", index=False)

    ground_truth_all = pd.read_csv(cfg["output"]["tables_dir"] / "transitions_stratified.csv")
    selected_ids = set(manifest_rows["record_id"])
    ground_truth = ground_truth_all[ground_truth_all["record_id"].isin(selected_ids)].copy()
    if len(ground_truth) != 6:
        raise ValueError(f"Expected 6 ground-truth rows, got {len(ground_truth)}")

    reference = {}
    for record_id in selected_ids:
        subject = record_id.split("__", 1)[0]
        row = ground_truth[ground_truth["record_id"] == record_id].iloc[0]
        base = {
            tool: float(row[f"vec_t_{tool.lower()}"])
            for tool in cfg["transitions"]["tool_order"]
        }
        reference[record_id] = ToolBox(
            cfg, excluded_player_id=subject
        ).find_comparables(base, k=cfg["baselines"]["knn_k"])

    overall_parts = []
    stratum_parts = []
    jaccard_parts = []
    for arm in agent_arms:
        predictions = agent_predictions[agent_predictions["arm"] == arm]
        report = score(predictions, ground_truth, reference_comparables=reference)
        for key, sink in (("overall", overall_parts), ("by_stratum", stratum_parts)):
            table = report[key].copy()
            table.insert(0, "model", arm)
            sink.append(table)
        if "jaccard" in report and not report["jaccard"].empty:
            table = report["jaccard"].copy()
            table.insert(0, "model", arm)
            jaccard_parts.append(table)

    arm_a_long = pd.read_csv(cfg["output"]["tables_dir"] / "arm_a_predictions.csv")
    rename = {"WAR": "delta_war", "wRC+": "delta_wrc", "OPS": "delta_ops"}
    for model in arm_a_long["model"].unique():
        subset = arm_a_long[
            (arm_a_long["model"] == model) & arm_a_long["record_id"].isin(selected_ids)
        ]
        wide = (
            subset.pivot(index="record_id", columns="target", values="y_pred")
            .reset_index()
            .rename(columns=rename)
        )
        report = score(wide, ground_truth)
        for key, sink in (("overall", overall_parts), ("by_stratum", stratum_parts)):
            table = report[key].copy()
            table.insert(0, "model", model)
            sink.append(table)

    overall = pd.concat(overall_parts, ignore_index=True)
    by_stratum = pd.concat(stratum_parts, ignore_index=True)
    jaccard = pd.concat(jaccard_parts, ignore_index=True) if jaccard_parts else pd.DataFrame()
    overall.to_csv(pilot_dir / "scores_overall.csv", index=False)
    by_stratum.to_csv(pilot_dir / "scores_by_stratum.csv", index=False)
    jaccard.to_csv(pilot_dir / "jaccard_at_10.csv", index=False)

    truths = ground_truth.set_index("record_id")
    error_rows = []
    for row in agent_predictions.itertuples():
        truth = truths.loc[row.record_id]
        for target in ("war", "wrc", "ops"):
            predicted = float(getattr(row, f"delta_{target}"))
            actual = float(truth[f"delta_{target}"])
            error_rows.append(
                {
                    "pilot_id": row.pilot_id,
                    "record_id": row.record_id,
                    "selection_stratum": row.selection_stratum,
                    "arm": row.arm,
                    "target": target,
                    "actual": actual,
                    "predicted": predicted,
                    "absolute_error": abs(actual - predicted),
                    "sign_correct": bool(np.sign(actual) == np.sign(predicted)),
                }
            )
    pd.DataFrame(error_rows).to_csv(pilot_dir / "record_errors.csv", index=False)

    audit_rows = []
    for row in agent_predictions.itertuples():
        subject = row.record_id.split("__", 1)[0]
        items = row.comparables or []
        audit_rows.append(
            {
                "pilot_id": row.pilot_id,
                "arm": row.arm,
                "subject": subject,
                "identity_leak": any(item.startswith(f"{subject} (") for item in items),
                "invalid_comparable_format_count": sum(
                    COMPARABLE_RE.fullmatch(item) is None for item in items
                ),
                "comparable_count": len(items),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(pilot_dir / "comparable_audit.csv", index=False)
    d_audit = audit[audit["arm"].isin(["agent_d", "agent_d_v21"])].copy()
    d_audit.to_csv(pilot_dir / "arm_d_identity_leakage.csv", index=False)

    log = {
        "run_id": manifest["run_id"],
        "n_records": 6,
        "n_agent_predictions": len(agent_predictions),
        "agent_instance_per_arm": True,
        "fresh_agent_per_record_arm": False,
        "subject_exclusion": True,
        "publication_grade": False,
        "arm_d_audit_by_variant": {
            arm: {
                "identity_leak_count": int(part["identity_leak"].sum()),
                "denominator": len(part),
                "invalid_comparable_format_count": int(
                    part["invalid_comparable_format_count"].sum()
                ),
            }
            for arm, part in d_audit.groupby("arm")
        },
    }
    (pilot_dir / "summary.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\nComparable audit")
    print(d_audit.to_string(index=False))


if __name__ == "__main__":
    main()
