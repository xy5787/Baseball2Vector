"""Audit authoritative inputs, reproduce headline values, and attach stable IDs."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation_utils import (  # noqa: E402
    EXP_ROOT,
    REPO_ROOT,
    TOOL_COLS,
    ensure_dirs,
    load_config,
    normalize_name,
    sha256_file,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from baseball2vec.roi import build_delta_dataset, run_regression  # noqa: E402


def load_chadwick(url: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.content
    archive_hash = __import__("hashlib").sha256(payload).hexdigest()
    archive = zipfile.ZipFile(io.BytesIO(payload))
    people_files = [
        name
        for name in archive.namelist()
        if "/data/people-" in name and name.endswith(".csv")
    ]
    people = pd.concat(
        [pd.read_csv(archive.open(name), low_memory=False) for name in people_files],
        ignore_index=True,
    )
    names_path = next(name for name in archive.namelist() if name.endswith("/data/names.csv"))
    aliases = pd.read_csv(archive.open(names_path), low_memory=False)
    return people, aliases, archive_hash


def build_alias_index(people: pd.DataFrame, aliases: pd.DataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}

    def add(key_person: object, first: object, last: object) -> None:
        if pd.isna(first) or pd.isna(last):
            return
        key = normalize_name(f"{first} {last}")
        if key:
            index.setdefault(key, set()).add(str(key_person))

    for row in people.itertuples(index=False):
        add(row.key_person, row.name_first, row.name_last)
        if pd.notna(row.name_given):
            given_parts = str(row.name_given).split()
            if given_parts:
                add(row.key_person, given_parts[0], row.name_last)
    for row in aliases.itertuples(index=False):
        add(row.key_person, row.altname_first, row.altname_last)
        if pd.notna(row.altname_given):
            parts = str(row.altname_given).split()
            if parts:
                add(row.key_person, parts[0], row.altname_last)
    return index


def attach_ids(df: pd.DataFrame, people: pd.DataFrame, aliases: pd.DataFrame) -> pd.DataFrame:
    people = people.copy()
    people = people[people["mlb_played_first"].notna()].copy()
    people["key_person"] = people["key_person"].astype(str)
    by_key = people.set_index("key_person", drop=False)
    alias_index = build_alias_index(people, aliases)
    output = df.copy()
    output.insert(0, "source_row", np.arange(len(output), dtype=int))
    player_ids: list[object] = []
    mlbam_ids: list[object] = []
    fg_ids: list[object] = []
    birth_years: list[object] = []
    statuses: list[str] = []
    counts: list[int] = []
    candidate_lists: list[str] = []
    for row in output.itertuples(index=False):
        keys = alias_index.get(normalize_name(row.Name), set())
        candidates = by_key.loc[list(keys)] if keys else people.iloc[0:0]
        if isinstance(candidates, pd.Series):
            candidates = candidates.to_frame().T
        candidates = candidates[
            (candidates["mlb_played_first"] <= row.Season)
            & (candidates["mlb_played_last"] >= row.Season)
        ]
        candidates = candidates.drop_duplicates("key_person")
        counts.append(len(candidates))
        candidate_lists.append(
            ";".join(
                f"fg:{int(r.key_fangraphs) if pd.notna(r.key_fangraphs) else 'NA'}"
                f"/mlbam:{int(r.key_mlbam) if pd.notna(r.key_mlbam) else 'NA'}"
                for r in candidates.itertuples(index=False)
            )
        )
        if len(candidates) != 1:
            player_ids.append(pd.NA)
            mlbam_ids.append(pd.NA)
            fg_ids.append(pd.NA)
            birth_years.append(pd.NA)
            statuses.append("unmatched" if len(candidates) == 0 else "ambiguous")
            continue
        candidate = candidates.iloc[0]
        fg = candidate.get("key_fangraphs")
        mlbam = candidate.get("key_mlbam")
        if pd.notna(fg):
            player_id = f"fg:{int(fg)}"
        elif pd.notna(mlbam):
            player_id = f"mlbam:{int(mlbam)}"
        else:
            player_id = pd.NA
        player_ids.append(player_id)
        mlbam_ids.append(int(mlbam) if pd.notna(mlbam) else pd.NA)
        fg_ids.append(int(fg) if pd.notna(fg) else pd.NA)
        birth_years.append(candidate.get("birth_year", pd.NA))
        statuses.append("matched" if pd.notna(player_id) else "unmatched_no_stable_id")
    output["player_id"] = player_ids
    output["key_mlbam"] = pd.array(mlbam_ids, dtype="Int64")
    output["key_fangraphs"] = pd.array(fg_ids, dtype="Int64")
    output["birth_year"] = pd.to_numeric(birth_years, errors="coerce")
    output["age_t"] = output["Season"] - output["birth_year"]
    output["id_status"] = statuses
    output["id_candidate_count"] = counts
    output["id_candidates"] = candidate_lists
    matched = output[output["id_status"].eq("matched")]
    duplicate_ids = matched.duplicated(["player_id", "Season"], keep=False)
    if duplicate_ids.any():
        bad_ids = set(matched.loc[duplicate_ids, "player_id"])
        mask = output["player_id"].isin(bad_ids)
        output.loc[mask, "id_status"] = "ambiguous_duplicate_id_season"
        output.loc[mask, "player_id"] = pd.NA
    return output


def main() -> None:
    ensure_dirs()
    cfg = load_config()
    source = REPO_ROOT / cfg["source_data"]
    manuscript = REPO_ROOT / cfg["authoritative_manuscript"]
    df = pd.read_csv(source)
    assert len(df) == 2309
    assert sorted(df["Season"].unique().tolist()) == cfg["years"]
    assert (df["PA"] >= cfg["minimum_pa"]).all()
    assert set(TOOL_COLS).issubset(df.columns)

    correlations = []
    for encoding in ["ZScore", "PCA", "JointVAE"]:
        for target in ["WAR", "wRC+", "OPS"]:
            correlations.append(
                {
                    "encoding": encoding,
                    "target": target,
                    "pearson_r": df[f"{encoding}_Area"].corr(df[target]),
                }
            )
    pd.DataFrame(correlations).to_csv(EXP_ROOT / "results" / "headline_correlations.csv", index=False)

    original_delta = build_delta_dataset(df)
    original_reg = run_regression(original_delta, random_state=cfg["seed"])
    reproduction = {
        "rows": len(df),
        "unique_display_names": int(df["Name"].nunique()),
        "unique_name_season_keys": int(df["UniqueName"].nunique()),
        "seasons": sorted(int(v) for v in df["Season"].unique()),
        "season_counts": {str(int(k)): int(v) for k, v in df["Season"].value_counts().sort_index().items()},
        "name_season_collisions": int(df.duplicated(["Name", "Season"]).sum()),
        "zscore_area_war_r": float(df["ZScore_Area"].corr(df["WAR"])),
        "name_keyed_transition_rows": int(len(original_delta)),
        "name_keyed_delta_wrc_lr_cv_r2": float(next(r for r in original_reg if r.target == "ΔwRC+").lr_r2_cv),
    }

    people, aliases, registry_hash = load_chadwick(cfg["chadwick_register_url"])
    with_ids = attach_ids(df, people, aliases)
    with_ids.to_csv(EXP_ROOT / "data_intermediate" / "player_seasons_with_ids.csv", index=False)
    registry_cols = [
        "key_person", "key_mlbam", "key_fangraphs", "key_bbref", "key_retro",
        "name_first", "name_last", "birth_year", "birth_month", "birth_day",
        "mlb_played_first", "mlb_played_last",
    ]
    matched_people = people[people["key_fangraphs"].isin(with_ids["key_fangraphs"].dropna())]
    matched_people[registry_cols].to_csv(
        EXP_ROOT / "data_intermediate" / "chadwick_registry_subset.csv", index=False
    )
    id_audit = with_ids[
        ["source_row", "Name", "Season", "PA", "id_status", "id_candidate_count", "id_candidates"]
    ]
    id_audit.to_csv(EXP_ROOT / "results" / "player_id_audit.csv", index=False)

    matched = with_ids[with_ids["id_status"].eq("matched")]
    sample_flow = pd.DataFrame(
        [
            {"stage": "source_player_seasons", "n_rows": len(df), "n_players": df["Name"].nunique()},
            {"stage": "stable_id_matched_player_seasons", "n_rows": len(matched), "n_players": matched["player_id"].nunique()},
            {"stage": "excluded_ambiguous_or_unmatched", "n_rows": len(df) - len(matched), "n_players": with_ids.loc[~with_ids["id_status"].eq("matched"), "Name"].nunique()},
        ]
    )
    sample_flow.to_csv(EXP_ROOT / "results" / "sample_flow.csv", index=False)

    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_mtime_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
        "manuscript_path": str(manuscript),
        "manuscript_sha256_stage_a_start": sha256_file(manuscript),
        "chadwick_url": cfg["chadwick_register_url"],
        "chadwick_archive_sha256": registry_hash,
        "raw_fangraphs_cache_expected": str(REPO_ROOT / "data/raw/batting_stats_2021_2025.csv"),
        "raw_fangraphs_cache_present": (REPO_ROOT / "data/raw/batting_stats_2021_2025.csv").exists(),
    }
    (EXP_ROOT / "results" / "input_provenance.json").write_text(json.dumps(provenance, indent=2))
    (EXP_ROOT / "results" / "headline_reproduction.json").write_text(json.dumps(reproduction, indent=2))
    print(json.dumps({"reproduction": reproduction, "id_status": with_ids["id_status"].value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    main()
