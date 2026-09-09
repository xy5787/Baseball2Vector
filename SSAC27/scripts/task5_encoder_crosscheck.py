"""Task 5 -- does the result come from the tool structure or from one encoder?

The five tool scores can be built three ways, all of them already archived in
`v3_results.csv`:

    ZScore    the group mean of the within-season z-scored constituents
    PCA       the first principal component of each group
    JointVAE  the latent means of the Cholesky full-covariance joint VAE

All three share the same *grouping* -- the same 22 statistics assigned to the
same 5 tools -- and differ only in how each group is collapsed to one number.
So if all three beat the scalar baseline, the result belongs to the five-tool
structure. If only one does, the result belongs to that encoder.

Every encoding is run against the same scalar baseline, on the same
rolling-origin folds, with alpha re-selected per fold per encoding, for all
three targets.

Run:  python SSAC27/scripts/task5_encoder_crosscheck.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as sc

GRID = sc.ALPHA_GRID_WIDE
ENCODING_LABELS = {
    "zscore": "ZScore",
    "pca": "PCA",
    "vae": "JointVAE",
}


def model_name(encoding: str, suffix: str = "") -> str:
    return f"M1 {ENCODING_LABELS[encoding]} 5 tools{suffix}"


def build_specs(target: sc.Target) -> dict[str, list[str]]:
    specs = {"M0 scalar baseline": [target.baseline]}
    for encoding in ENCODING_LABELS:
        specs[model_name(encoding)] = sc.tool_features(encoding)
        specs[model_name(encoding, " + age + PA")] = [
            *sc.tool_features(encoding),
            *sc.CONTEXT_FEATURES,
        ]
    return specs


def pairs_for() -> list[tuple[str, str]]:
    pairs = [(model_name(e), "M0 scalar baseline") for e in ENCODING_LABELS]
    pairs += [
        (model_name(e, " + age + PA"), "M0 scalar baseline") for e in ENCODING_LABELS
    ]
    # Head-to-head between encodings, ZScore as the published reference.
    pairs += [(model_name("zscore"), model_name(e)) for e in ("pca", "vae")]
    return pairs


def run_target(
    transitions: pd.DataFrame, target: sc.Target
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = sc.prepare_for_target(transitions, target)
    specs = build_specs(target)
    pairs = pairs_for()

    fold_frames, paired_frames, tuning_frames = [], [], []
    per_fold: list[pd.DataFrame] = []
    for fold in sc.ROLLING_FOLDS:
        train, test = sc.split_fold(frame, fold)
        results, tuning = sc.run_models(
            train, test, specs, GRID, fold_label=fold, target=target.column
        )
        tuning.insert(0, "target", target.key)
        tuning_frames.append(tuning)

        predictions = sc.attach_fit_metadata(
            pd.concat([r.predictions for r in results], ignore_index=True), results
        )
        predictions["target"] = target.key
        per_fold.append(predictions)

        summary, paired = sc.evaluate_predictions(
            predictions, fold, pairs, extra={"target": target.key}
        )
        fold_frames.append(summary)
        paired_frames.append(paired)

    pooled = pd.concat(per_fold, ignore_index=True)
    pooled_summary, pooled_paired = sc.evaluate_predictions(
        pooled, "pooled", pairs, pooled=True, extra={"target": target.key}
    )
    fold_frames.append(pooled_summary)
    paired_frames.append(pooled_paired)

    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(paired_frames, ignore_index=True),
        pooled,
        pd.concat(tuning_frames, ignore_index=True),
    )


def verdict_table(paired: pd.DataFrame) -> pd.DataFrame:
    """Per target and encoding: how many folds beat the scalar, and by how much."""
    rows: list[dict] = []
    folds = list(sc.ROLLING_FOLDS)
    for target in sc.TARGETS.values():
        for encoding, label in ENCODING_LABELS.items():
            for suffix in ("", " + age + PA"):
                block = paired[
                    paired["target"].eq(target.key)
                    & paired["candidate"].eq(model_name(encoding, suffix))
                    & paired["baseline"].eq("M0 scalar baseline")
                ].set_index("fold")
                deltas = [float(block.loc[f, "improvement_mae"]) for f in folds]
                excludes = [bool(block.loc[f, "excludes_zero"]) for f in folds]
                rows.append(
                    {
                        "target": target.key,
                        "encoding": label,
                        "features": f"5 tools{suffix}",
                        "folds_beating_scalar": sum(d > 0 for d in deltas),
                        "all_three_same_sign": len({np.sign(d) for d in deltas}) == 1,
                        "folds_ci_excludes_zero": sum(excludes),
                        "pooled_improvement": float(
                            block.loc["pooled", "improvement_mae"]
                        ),
                        "pooled_ci_low": float(block.loc["pooled", "ci_low"]),
                        "pooled_ci_high": float(block.loc["pooled", "ci_high"]),
                        "pooled_excludes_zero": bool(
                            block.loc["pooled", "excludes_zero"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    (sc.RESULTS_DIR / "task5").mkdir(parents=True, exist_ok=True)
    players = sc.load_player_seasons()
    transitions = sc.build_transitions(players)

    blocks = [run_target(transitions, target) for target in sc.TARGETS.values()]
    performance = pd.concat([b[0] for b in blocks], ignore_index=True)
    paired = pd.concat([b[1] for b in blocks], ignore_index=True)
    verdicts = verdict_table(paired)

    performance.to_csv(sc.RESULTS_DIR / "task5" / "fold_performance.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task5" / "paired_differences.csv", index=False)
    verdicts.to_csv(sc.RESULTS_DIR / "task5" / "encoder_verdicts.csv", index=False)
    pd.concat([b[2] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task5" / "predictions.csv", index=False
    )
    pd.concat([b[3] for b in blocks], ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task5" / "alpha_tuning.csv", index=False
    )
    (sc.RESULTS_DIR / "task5" / "environment.json").write_text(
        json.dumps(
            {**sc.environment_stamp(), "encodings": list(ENCODING_LABELS.values())},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    for target in sc.TARGETS.values():
        block = performance[performance["target"].eq(target.key)]
        print(f"\n=== {target.label} -- MAE by encoding x fold ===")
        print(block.pivot(index="model", columns="fold", values="mae").round(4).to_string())

    print("\n=== Does every encoding beat the scalar baseline? ===")
    print(verdicts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
