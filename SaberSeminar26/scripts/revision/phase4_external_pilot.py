"""Phase 4: Qualitative external validation pilot vs. public scouting-style grades.

This is explicitly a small (n=6), qualitative, discussion-only pilot -- NOT a validation claim.
No correlation or aggregate statistic is computed; see outputs/revision/phase4_external_pilot/notes.md
for the per-player discussion this script's table is meant to support.

Public tool grades below were located via web search in this session and are cited to the
specific source page per player (see SOURCE_URL / SOURCE_NOTE). Where a source did not report a
specific 20-80 number for a tool, that cell is left blank (pd.NA) rather than estimated -- see
notes.md for which tools are missing per player and why.

Grades are on the 20-80 scale but are NOT numerically interchangeable with Baseball2Vector's
ZScore_{Tool} columns: public scouting grades are calibrated against the full population of
minor-league prospects (most of whom never reach the majors), while Baseball2Vector's 20-80 scale
is calibrated only against qualified (>=100 PA) MLB regulars, 2021-2025 -- a much narrower, more
elite reference population. A "60" on each scale does not mean the same percentile. The
comparison below should be read as a check on each player's *relative profile shape* (which tool
is their carrying tool vs. their weakest), not on literal number matching.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "v3_results_clean.csv"
OUT_DIR = ROOT / "outputs" / "revision" / "phase4_external_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Public scouting grades (20-80 scale), hand-researched and cited per player. Missing tools are
# left as None (not estimated). "season_used" = each player's first >=100 PA MLB season in
# data/v3_results_clean.csv, i.e. the closest available Baseball2Vector reading to a pre-debut
# prospect grade.
PUBLIC_GRADES = [
    {
        "player": "Bobby Witt Jr.",
        "season_used": 2022,
        "hit": 60, "power": 60, "run": 60, "arm": 60, "field": 60, "overall_fv": 65,
        "source_url": "https://www.baseballamerica.com/players/8310-bobby-witt/",
        "source_note": "Baseball America player-page grade block (Hit/Power/Run/Arm/Field/Overall).",
    },
    {
        "player": "Julio Rodriguez",
        "season_used": 2022,
        "hit": 70, "power": 80, "run": None, "arm": 70, "field": None, "overall_fv": None,
        "source_url": "https://www.baseballamerica.com/players/7235-julio-rodriguez/",
        "source_note": (
            "BA: Hit 70, 80-grade raw power, Arm 70; Run described only as \"average\" and Field "
            "only as \"should be at least an average defender\" -- no specific digit reported in "
            "the excerpt found, so left blank rather than guessed."
        ),
    },
    {
        "player": "Adley Rutschman",
        "season_used": 2022,
        "hit": 70, "power": None, "run": None, "arm": None, "field": None, "overall_fv": None,
        "source_url": "https://www.baseballamerica.com/players/4052-adley-rutschman/",
        "source_note": (
            "BA: Hit 70. MLB Pipeline described him qualitatively as having \"plus grades in each "
            "of the four tools that matter for catchers\" (no specific digits found for "
            "Power/Run/Arm/Field in available source excerpts) -- left blank rather than guessed."
        ),
    },
    {
        "player": "Gunnar Henderson",
        "season_used": 2023,  # 2022 was only 132 PA; 2023 is his first full qualifying season
        "hit": 60, "power": 70, "run": 60, "arm": 70, "field": 60, "overall_fv": None,
        "source_url": "https://www.baseballamerica.com/players/8124-gunnar-henderson/",
        "source_note": "Baseball America player-page grade block.",
    },
    {
        "player": "Corbin Carroll",
        "season_used": 2023,  # 2022 was only 115 PA; 2023 is his first full qualifying season
        "hit": 60, "power": 60, "run": 80, "arm": 45, "field": 60, "overall_fv": 65,
        "source_url": "https://www.baseballamerica.com/players/8111-corbin-carroll/",
        "source_note": "Baseball America player-page grade block; BA Grade/Risk 65/Medium.",
    },
    {
        "player": "Elly De La Cruz",
        "season_used": 2023,
        "hit": 42, "power": 70, "run": 70, "arm": 70, "field": None, "overall_fv": None,
        "source_url": "https://www.baseballamerica.com/stories/elly-de-la-cruz-pegs-the-top-of-the-scouting-scale/",
        "source_note": (
            "BA: Power/Run/Arm all 70; Hit reported as a 40-45 range (midpoint 42 used here, "
            "noted as a range in notes.md, not a precise digit). No specific Field grade found in "
            "available source excerpts -- left blank rather than guessed."
        ),
    },
]


def main() -> None:
    b2v = pd.read_csv(DATA)

    rows = []
    for entry in PUBLIC_GRADES:
        match = b2v[(b2v["Name"] == entry["player"]) & (b2v["Season"] == entry["season_used"])]
        assert len(match) == 1, f"Expected 1 row for {entry['player']} {entry['season_used']}, got {len(match)}"
        r = match.iloc[0]
        rows.append(
            {
                "player": entry["player"],
                "season_used": entry["season_used"],
                "pa": int(r["PA"]),
                "public_hit": entry["hit"],
                "public_power": entry["power"],
                "public_run": entry["run"],
                "public_arm": entry["arm"],
                "public_field": entry["field"],
                "public_overall_fv": entry["overall_fv"],
                "b2v_contact": round(r["ZScore_Contact"], 1),
                "b2v_power": round(r["ZScore_Power"], 1),
                "b2v_speed": round(r["ZScore_Speed"], 1),
                "b2v_defense": round(r["ZScore_Defense"], 1),
                "b2v_discipline_no_public_analogue": round(r["ZScore_Discipline"], 1),
                "wRC+": r["wRC+"],
                "WAR": r["WAR"],
                "source_url": entry["source_url"],
                "source_note": entry["source_note"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "pilot_table.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'pilot_table.csv'}")


if __name__ == "__main__":
    main()
