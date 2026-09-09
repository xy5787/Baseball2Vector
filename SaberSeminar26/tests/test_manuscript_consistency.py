from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "baseball2vector_en.tex"

def test_only_one_canonical_manuscript():
    assert [p.name for p in (ROOT / "paper").glob("baseball2vector_en*.tex")] == ["baseball2vector_en.tex"]

def test_references_figures_and_tables_resolve():
    text = TEX.read_text()
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    assert refs <= labels
    for relative in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
        assert (TEX.parent / relative).is_file(), relative
    for relative in re.findall(r"\\input\{([^}]+)\}", text):
        assert (TEX.parent / relative).is_file(), relative
    assert (TEX.parent / "references.bib").is_file()

def test_headline_registry_matches_manuscript():
    text = TEX.read_text()
    rows = pd.read_csv(ROOT / "results" / "headline_results.csv", dtype={"manuscript_value": str})
    for value in rows.manuscript_value:
        assert value in text
