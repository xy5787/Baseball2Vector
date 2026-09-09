from pathlib import Path
import hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_transition_hash_and_keys():
 p=ROOT/"data/processed/transitions.csv"
 assert hashlib.sha256(p.read_bytes()).hexdigest()=="7fc2f490b1a354fc78bd51fb3f92645e7bfda042d6a07ca82c25284bcffc1aa4"
 d=pd.read_csv(p)
 assert len(d)==1414
 assert not d.duplicated(["player_id","Season_t","Season_t1"]).any()
