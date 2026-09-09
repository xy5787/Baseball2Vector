from pathlib import Path
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def test_published_inputs_have_expected_hashes():
    data = ROOT / "data" / "processed"
    assert sha(data / "v3_results.csv") == "70bea280ec725ac78376e94edf41fae6fd036fe8b7a0b71270869eb1cfdf0873"
    assert sha(data / "transitions.csv") == "62f69a7840caa7cf362550b4bb91d2914c3487607cc8aa2f90cdbb8d82d3c900"

def test_canonical_result_copies_are_byte_identical():
    results = ROOT / "results"
    assert (results / "prediction_results.csv").read_bytes() == (results / "calibrated_prediction.csv").read_bytes()
    assert (results / "robustness_results.csv").read_bytes() == (results / "stability_by_pa.csv").read_bytes()

def test_stable_id_counts():
    frame = pd.read_csv(ROOT / "data" / "processed" / "player_seasons.csv")
    assert frame.id_status.value_counts().to_dict() == {"matched": 2286, "ambiguous": 15, "unmatched": 8}
    assert frame.loc[frame.id_status.eq("matched"), "player_id"].nunique() == 818
