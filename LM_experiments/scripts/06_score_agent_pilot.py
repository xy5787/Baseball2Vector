"""Normalize and score the five-record non-publication agent pilot."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from lm_experiments.baseline_models import (  # noqa: E402
    build_knn_pool,
    knn_comparables,
    time_split,
)
from lm_experiments.config import load_config  # noqa: E402
from lm_experiments.lm_arms.client import validate_prediction  # noqa: E402
from lm_experiments.scoring import score  # noqa: E402
from lm_experiments.transitions import drop_ambiguous_keys, load_v3_results  # noqa: E402


def main() -> None:
    cfg = load_config()
    root = Path(__file__).parents[1]
    pilot_dir = root / "outputs" / "agent_pilot"
    manifest = json.loads((pilot_dir / "manifest.json").read_text())
    manifest_rows = pd.DataFrame(manifest["records"])
    record_map = dict(zip(manifest_rows["pilot_id"], manifest_rows["record_id"]))
    stratum_map = dict(
        zip(manifest_rows["pilot_id"], manifest_rows["exclusive_stratum"])
    )

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

    ground_truth_all = pd.read_csv(
        cfg["output"]["tables_dir"] / "transitions_stratified.csv"
    )
    selected_ids = set(manifest_rows["record_id"])
    ground_truth = ground_truth_all[
        ground_truth_all["record_id"].isin(selected_ids)
    ].copy()
    assert len(ground_truth) == 5

    windows = [tuple(window) for window in cfg["transitions"]["windows"]]
    source = load_v3_results(cfg["data"]["v3_results"])
    clean, _ = drop_ambiguous_keys(source)
    train_seasons = sorted({season for window in windows[:-1] for season in window})
    pool = build_knn_pool(
        clean,
        train_seasons,
        cfg["transitions"]["tool_order"],
        cfg["transitions"]["min_pa"],
    )
    transitions = pd.read_csv(cfg["output"]["tables_dir"] / "transitions.csv")
    _, test = time_split(transitions, windows[-1])
    reference = knn_comparables(
        test,
        pool,
        cfg["transitions"]["tool_order"],
        k=cfg["baselines"]["knn_k"],
    )

    overall_parts = []
    stratum_parts = []
    jaccard_parts = []
    for arm in ("agent_b", "agent_c", "agent_d"):
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
            (arm_a_long["model"] == model)
            & (arm_a_long["record_id"].isin(selected_ids))
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
    jaccard = (
        pd.concat(jaccard_parts, ignore_index=True)
        if jaccard_parts
        else pd.DataFrame()
    )
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

    d_rows = agent_predictions[agent_predictions["arm"] == "agent_d"]
    leaks = []
    for row in d_rows.itertuples():
        subject = row.record_id.split("__", 1)[0]
        leaked = any(item.startswith(f"{subject} (") for item in (row.comparables or []))
        leaks.append({"pilot_id": row.pilot_id, "subject": subject, "identity_leak": leaked})
    leak_df = pd.DataFrame(leaks)
    leak_df.to_csv(pilot_dir / "arm_d_identity_leakage.csv", index=False)

    log = {
        "run_id": manifest["run_id"],
        "n_records": 5,
        "n_agent_predictions": len(agent_predictions),
        "fresh_agent_per_record_arm": True,
        "publication_grade": False,
        "arm_d_identity_leak_count": int(leak_df["identity_leak"].sum()),
        "arm_d_identity_leak_denominator": len(leak_df),
    }
    (pilot_dir / "summary.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print("\nD identity leakage")
    print(leak_df.to_string(index=False))


if __name__ == "__main__":
    main()
