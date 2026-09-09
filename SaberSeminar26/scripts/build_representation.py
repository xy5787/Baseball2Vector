"""Build or verify the public stable-ID player-season analysis table."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def build_player_seasons() -> pd.DataFrame:
    scores = pd.read_csv(PROCESSED / "v3_results.csv")
    scores.insert(0, "source_row", range(len(scores)))
    audit = pd.read_csv(PROCESSED / "player_id_audit.csv")
    registry = pd.read_csv(PROCESSED / "chadwick_registry_subset.csv")
    out = scores.merge(
        audit[["source_row", "id_status", "id_candidate_count", "id_candidates"]],
        on="source_row", validate="one_to_one",
    )
    out["key_fangraphs"] = pd.to_numeric(
        out["id_candidates"].str.extract(r"fg:(\d+)")[0], errors="coerce"
    ).astype("Int64")
    out["key_mlbam"] = pd.to_numeric(
        out["id_candidates"].str.extract(r"mlbam:(\d+)")[0], errors="coerce"
    ).astype("Int64")
    out["player_id"] = out["key_fangraphs"].map(
        lambda value: f"fg:{value}" if pd.notna(value) else pd.NA
    )
    excluded = ~out["id_status"].eq("matched")
    out.loc[excluded, ["player_id", "key_fangraphs", "key_mlbam"]] = pd.NA
    birth_year = registry.set_index("key_fangraphs")["birth_year"]
    out["birth_year"] = out["key_fangraphs"].map(birth_year)
    out["age_t"] = out["Season"] - out["birth_year"]
    original = [c for c in scores if c != "source_row"]
    order = ["source_row", *original, "player_id", "key_mlbam", "key_fangraphs",
             "birth_year", "age_t", "id_status", "id_candidate_count", "id_candidates"]
    return out[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    built = build_player_seasons()
    target = PROCESSED / "player_seasons.csv"
    assert len(built) == 2309
    assert built["id_status"].value_counts().to_dict() == {
        "matched": 2286, "ambiguous": 15, "unmatched": 8
    }
    matched = built[built["id_status"].eq("matched")]
    assert matched["player_id"].nunique() == 818
    if args.write:
        built.to_csv(target, index=False)
    else:
        committed = pd.read_csv(target)
        pd.testing.assert_frame_equal(built, committed, check_dtype=False)
    print(f"verified {len(built)} player-seasons ({len(matched)} stable-ID matched)")


if __name__ == "__main__":
    main()
