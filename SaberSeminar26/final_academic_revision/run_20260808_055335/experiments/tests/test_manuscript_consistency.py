"""Static consistency checks for the revised manuscript and artifacts."""

from pathlib import Path
import hashlib
import re

import pandas as pd


RUN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RUN_ROOT.parents[1]
TEX = RUN_ROOT / "manuscript_preview" / "baseball2vector_en_revision.tex"
ORIGINAL = REPO_ROOT / "paper" / "baseball2vector_en.tex"
ORIGINAL_SHA256 = "a873b4beb3b9acbcdd44c5f13d859b8c2a326241234ef49794fe20ff65e9f00d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_is_unchanged_and_revision_is_distinct() -> None:
    assert sha256(ORIGINAL) == ORIGINAL_SHA256
    assert TEX.exists()
    assert TEX.read_bytes() != ORIGINAL.read_bytes()


def test_references_labels_and_figure_files_resolve() -> None:
    text = TEX.read_text(encoding="utf-8")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    cites = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", text)
        for key in group.split(",")
    }
    assert refs <= labels
    assert cites <= bibitems
    assert len(labels) == len(re.findall(r"\\label\{([^}]+)\}", text))
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert (TEX.parent / relative).exists(), relative


def test_removed_unverifiable_and_roi_material_stays_removed() -> None:
    text = TEX.read_text(encoding="utf-8").lower()
    forbidden = [
        "baseball america", "tab:pilot", "external-pilot",
        "return on investment", "roi_coefficients", "synthetic $+5$",
    ]
    assert not any(term in text for term in forbidden)


def test_manuscript_numbers_match_result_files() -> None:
    text = TEX.read_text(encoding="utf-8")
    prediction = pd.read_csv(RUN_ROOT / "results" / "calibrated_prediction.csv")
    b2v = prediction[
        prediction["model"].eq("B2V Ridge") & prediction["metric"].eq("mae")
    ].iloc[0]
    assert f"{b2v.estimate:.3f}" in text
    paired = pd.read_csv(RUN_ROOT / "results" / "paired_model_differences.csv")
    scalar = paired[
        paired["candidate"].eq("B2V Ridge")
        & paired["baseline"].eq("Calibrated wRC+ Ridge")
        & paired["metric"].eq("mae")
    ].iloc[0]
    assert f"{scalar.improvement:.3f}" in text
    matching = pd.read_csv(RUN_ROOT / "results" / "matched_profile_distance_summary.csv")
    for pair_type in [
        "same_player_adjacent", "scalar_matched", "random_same_season_pa_stratum"
    ]:
        row = matching[matching["pair_type"].eq(pair_type)].iloc[0]
        assert f"{row['median']:.3f}" in text
    stability = pd.read_csv(RUN_ROOT / "results" / "stability_by_pa.csv")
    assert set(stability["n_pairs"]) == {340, 507, 567, 1414}


def test_tex_structure_is_balanced_and_methods_are_not_empty() -> None:
    text = TEX.read_text(encoding="utf-8")
    assert text.count("{") == text.count("}")
    begins = re.findall(r"\\begin\{([^}]+)\}", text)
    ends = re.findall(r"\\end\{([^}]+)\}", text)
    assert sorted(begins) == sorted(ends)
    assert not re.search(
        r"\\paragraph\{[^}]+\}\s*\n\s*\\(?:section|subsection|paragraph)", text
    )
