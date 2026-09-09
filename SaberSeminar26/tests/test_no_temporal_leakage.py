from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_temporal_split_and_predictors():
    frame = pd.read_csv(ROOT / "data" / "processed" / "transitions.csv")
    train = frame[frame.Season_t1.isin([2022, 2023, 2024])]
    test = frame[(frame.Season_t == 2024) & (frame.Season_t1 == 2025)]
    assert len(train) == 1056 and len(test) == 358
    assert train.Season_t1.max() < test.Season_t1.min()
    predictors = [c for c in frame if c.endswith("_t")] + ["age_t_t"]
    assert not any(c.endswith("_t1") for c in predictors)
