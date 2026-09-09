from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_headline_values_match_task_sources():
 h=pd.read_csv(ROOT/"results/headline_results.csv").set_index("claim_id")
 p=pd.read_csv(ROOT/"results/task6/paired_differences.csv")
 q=p[(p.target=="wrc_plus")&(p.window=="2019-2025 (all)")&(p.fold=="pooled")
     &(p.candidate=="M1 B2V 5 tools")&(p.baseline=="M0 scalar baseline")].iloc[0]
 assert abs(h.loc["b2v_vs_scalar","estimate"]-q.improvement_mae)<1e-12
 assert int(q.n_test)==1706

def test_each_task_has_results():
 for task in range(1,7):
  files=list((ROOT/f"results/task{task}").glob("*.csv"))
  assert files, task
  assert (ROOT/f"results/task{task}/environment.json").is_file()
