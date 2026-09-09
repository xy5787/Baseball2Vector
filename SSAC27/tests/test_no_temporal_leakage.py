from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import common

def test_rolling_folds_are_ordered():
 d=common.load_transitions()
 for fold in common.ROLLING_FOLDS:
  train,test=common.split_fold(d,fold)
  assert train.Season_t1.max() <= test.Season_t.min()
  assert (test.Season_t1==test.Season_t+1).all()

def test_ssac_has_no_saber_path_dependency():
 for path in (ROOT/"scripts").glob("*.py"):
  assert "../SaberSeminar26" not in path.read_text()
