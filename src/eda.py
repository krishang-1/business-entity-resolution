import sys, os, time, collections
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from io_utils import read_tsv

DATA = sys.argv[1] if len(sys.argv) > 1 else "dataset"
WORK = sys.argv[2] if len(sys.argv) > 2 else "work"
os.makedirs(WORK, exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)

files = {f"{sp}_s{i}": f"{DATA}/{sp}/{sp}_source{i}.tsv" for sp in ("train", "test") for i in (1, 2, 3)}
counts = {}
for k, p in files.items():
    t = time.time()
    df = read_tsv(p)
    if k == "train_s1":
        print(df.shape, df.columns.tolist(), df.dtypes.tolist()); print(df.sample(5, random_state=0))
    counts[k] = (len(df), df["country"].value_counts().to_dict(),
                 (df["business_name"] == "").mean().round(4), (df["business_address"] == "").mean().round(4),
                 df["business_name"].str.len().median(), df["business_address"].str.len().median(),
                 df["entity_id"].is_unique)
    df.to_parquet(f"{WORK}/raw_{k}.parquet", index=False)
    print(k, counts[k], f"{time.time()-t:.0f}s", flush=True)
    del df

gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
print(gt.shape, gt.columns.tolist()); print(gt.head())
m = gt["matched_entity_ids"].str.split(",")
n = m.map(lambda x: 0 if x == [""] else len(x))
print("singleton frac", (n == 0).mean(), "\nmatches/S1 dist\n", n.clip(upper=8).value_counts().sort_index())
ex = gt.assign(mid=m).explode("mid"); ex = ex[ex["mid"] != ""]
print("pairs", len(ex), "S2 share", ex["mid"].str.startswith("S2").mean())
vc = ex["mid"].value_counts()
print("S2/S3 ids matched to >1 S1:", (vc > 1).sum(), "max", vc.max())
ex["src"] = ex["mid"].str[:2]
per = ex.groupby(["source1_entity_id", "src"]).size()
print("S1 with >1 from same source frac (of matched S1):", (per.groupby(level=0).max() > 1).mean())
print("dist of per-source counts\n", per.clip(upper=6).value_counts().sort_index())
ex[["source1_entity_id", "mid"]].to_parquet(f"{WORK}/gt_pairs.parquet", index=False)
matched_ids = set(ex["mid"])
for k in ("train_s2", "train_s3"):
    ids = pd.read_parquet(f"{WORK}/raw_{k}.parquet", columns=["entity_id"])["entity_id"]
    print(k, "frac matching nothing:", 1 - ids.isin(matched_ids).mean())
s1 = pd.read_parquet(f"{WORK}/raw_train_s1.parquet")
print("GT S1 == train S1 set:", set(gt["source1_entity_id"]) == set(s1["entity_id"]))
