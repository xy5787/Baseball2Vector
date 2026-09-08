"""Task 4 -- comparable-player retrieval: profile distance vs scalar distance.

Question: is a comparable found in the five-dimensional tool space more useful
than one found by matching WAR and wRC+?

For each focal player-season X that has a next season, retrieve k comparables,
average *their* next-season outcome, and use that as the prediction for X. There
is no model and nothing is fitted, so the two methods differ only in the
distance they search under.

    Method A (scalar)  WAR and wRC+, standardized within the season, 2-D Euclidean
    Method B (B2V)     the five tool scores, 5-D Euclidean

Two candidate pools are reported, and the distinction matters:

* ``same_season`` -- the pool is the focal player's own season, as specified in
  the brief. **This is a retrieval evaluation, not a forecast.** It reads the
  comparables' next-season outcomes, which at real forecast time have not
  happened yet. It is a fair A-vs-B comparison because both methods get exactly
  the same information, but no number from it may be quoted as forecasting
  accuracy.
* ``past_seasons`` -- the pool is restricted to player-seasons strictly earlier
  than the focal season, whose following season has therefore already been
  observed. This *is* deployable, and it is the variant to quote if the paper
  makes any claim about using comparables in practice.

Run:  python SSAC27/scripts/task4_comp_search.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ssac_common as sc

K_VALUES = (3, 5, 10)
SCALAR_FEATURES = ["WAR_t", "wRC+_t"]
POOLS = ("same_season", "past_seasons")

METHODS = {
    "A scalar (WAR, wRC+)": SCALAR_FEATURES,
    "B B2V (5 tools)": [f"safe_ZScore_{tool}_t" for tool in sc.TOOL_NAMES],
}


def standardize_within_season(
    frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    """Put each feature on a common within-season scale before measuring distance.

    The tool features are already within-season standardized; WAR and wRC+ are
    not, and a raw Euclidean distance over them would be dominated by wRC+'s
    much larger numeric range. Standardizing uses only the focal season's own
    rows, so it reads nothing from the future.
    """
    out = np.empty((len(frame), len(features)), dtype=float)
    for index, feature in enumerate(features):
        values = frame.groupby("Season_t")[feature].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
        )
        out[:, index] = values.to_numpy(float)
    return out


def retrieve(
    focal: pd.DataFrame,
    candidates: pd.DataFrame,
    coordinates_focal: np.ndarray,
    coordinates_candidates: np.ndarray,
    target_column: str,
    k: int,
) -> np.ndarray:
    """Mean next-season outcome of the k nearest candidates, focal row excluded."""
    outcomes = candidates[target_column].to_numpy(float)
    candidate_keys = list(
        zip(candidates["player_id"], candidates["Season_t"].astype(int))
    )
    key_index = {key: position for position, key in enumerate(candidate_keys)}

    predictions = np.empty(len(focal), dtype=float)
    focal_keys = list(zip(focal["player_id"], focal["Season_t"].astype(int)))
    for row, key in enumerate(focal_keys):
        distances = np.linalg.norm(
            coordinates_candidates - coordinates_focal[row], axis=1
        )
        self_position = key_index.get(key)
        if self_position is not None:
            distances[self_position] = np.inf  # never a comparable to itself
        nearest = np.argpartition(distances, k)[:k]
        predictions[row] = float(outcomes[nearest].mean())
    return predictions


def build_predictions(
    transitions: pd.DataFrame, target: sc.Target, pool: str
) -> pd.DataFrame:
    """One prediction row per (focal, method, k) over every rolling-origin fold."""
    frame = sc.prepare_for_target(transitions, target)
    rows: list[pd.DataFrame] = []

    for fold in sc.ROLLING_FOLDS:
        train_inputs, test_input = sc.ROLLING_FOLDS[fold]
        test = frame[frame["Season_t"].eq(test_input)]
        if pool == "same_season":
            candidates = test
        else:
            candidates = frame[frame["Season_t"].isin(train_inputs)]
        if candidates.empty:
            continue

        for method, features in METHODS.items():
            # Standardize focal and candidates together only where they share a
            # season; the helper standardizes inside each Season_t independently.
            combined = pd.concat([test, candidates], ignore_index=True)
            coordinates = standardize_within_season(combined, features)
            coordinates_focal = coordinates[: len(test)]
            coordinates_candidates = coordinates[len(test) :]

            for k in K_VALUES:
                if len(candidates) <= k:
                    continue
                predicted = retrieve(
                    test, candidates, coordinates_focal, coordinates_candidates,
                    target.column, k,
                )
                block = test[["player_id", "Name", "Season_t", "Season_t1"]].copy()
                block["y_true"] = test[target.column].to_numpy(float)
                block["prediction"] = predicted
                block["fold"] = fold
                block["model"] = f"{method} k={k}"
                block["method"] = method
                block["k"] = k
                block["pool"] = pool
                block["target"] = target.key
                block["n_candidates"] = len(candidates)
                rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    players = sc.load_player_seasons()
    transitions = sc.build_transitions(players)

    summaries, paired_frames, prediction_frames = [], [], []
    for target in sc.TARGETS.values():
        for pool in POOLS:
            predictions = build_predictions(transitions, target, pool)
            prediction_frames.append(predictions)
            pairs = [
                (f"B B2V (5 tools) k={k}", f"A scalar (WAR, wRC+) k={k}")
                for k in K_VALUES
            ]
            extra = {"target": target.key, "pool": pool}
            for fold in sorted(predictions["fold"].unique()):
                block = predictions[predictions["fold"].eq(fold)]
                summary, paired = sc.evaluate_predictions(block, fold, pairs, extra=extra)
                summaries.append(summary)
                paired_frames.append(paired)
            pooled_summary, pooled_paired = sc.evaluate_predictions(
                predictions, "pooled", pairs, pooled=True, extra=extra
            )
            summaries.append(pooled_summary)
            paired_frames.append(pooled_paired)

    summary = pd.concat(summaries, ignore_index=True)
    paired = pd.concat(paired_frames, ignore_index=True)
    summary.to_csv(sc.RESULTS_DIR / "task4_retrieval_performance.csv", index=False)
    paired.to_csv(sc.RESULTS_DIR / "task4_paired_differences.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        sc.RESULTS_DIR / "task4_predictions.csv", index=False
    )
    (sc.RESULTS_DIR / "task4_environment.json").write_text(
        json.dumps(
            {**sc.environment_stamp(), "k_values": list(K_VALUES),
             "scalar_features": SCALAR_FEATURES, "pools": list(POOLS)},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    for target in sc.TARGETS.values():
        for pool in POOLS:
            block = summary[summary["target"].eq(target.key) & summary["pool"].eq(pool)]
            if block.empty:
                continue
            print(f"\n=== {target.label} | candidate pool: {pool} -- MAE ===")
            print(block.pivot(index="model", columns="fold", values="mae").round(4).to_string())
            print(f"\n  B minus A, paired (pool={pool}):")
            print(
                paired[paired["target"].eq(target.key) & paired["pool"].eq(pool)][
                    ["fold", "candidate", "improvement_mae", "ci_low", "ci_high",
                     "excludes_zero"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
