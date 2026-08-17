from pathlib import Path

import numpy as np
import pandas as pd

EXP_ROOT = Path(__file__).resolve().parents[1]


def test_generated_tables_agree_with_machine_results() -> None:
    results = pd.read_csv(EXP_ROOT / "results" / "prospective_prediction.csv")
    table = pd.read_csv(EXP_ROOT / "tables" / "prospective_prediction_table.csv")
    source = results[
        results["target"].eq("wRC+_t1")
        & results["model"].eq("b2v_age_pa_ridge")
        & results["metric"].eq("r2")
    ].iloc[0]
    displayed = table[table["model"].eq("B2V + age + PA Ridge")].iloc[0]
    assert np.isclose(float(str(displayed["r2"]).split()[0]), round(source["estimate"], 3))

    machine_match = pd.read_csv(
        EXP_ROOT / "results" / "matched_profile_distance_summary.csv"
    ).sort_values("pair_type")
    table_match = pd.read_csv(
        EXP_ROOT / "tables" / "matched_profile_distance_table.csv"
    ).sort_values("pair_type")
    assert machine_match["pair_type"].tolist() == table_match["pair_type"].tolist()
    assert np.allclose(machine_match["median"], table_match["median"])

    stability = pd.read_csv(EXP_ROOT / "results" / "representation_stability.csv")
    stability_table = pd.read_csv(EXP_ROOT / "tables" / "stability_table.csv")
    source_contact = stability[
        stability["subset"].eq("overall") & stability["metric"].eq("spearman")
        & stability["dimension"].eq("Contact")
    ].iloc[0]
    displayed_contact = stability_table[stability_table["dimension"].eq("Contact")].iloc[0]
