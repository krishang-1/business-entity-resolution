import os
"""Phase 2b (full pool) + 2c: cluster precision/recall on the full train pool, and blocking gain from cluster expansion
on the 88,272 val S1 (same split). Expansion: cluster-mates of each S1's top-N candidates by stage-1 p1, cap +EXTRA."""
import sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as C
from run_pipeline import Source, make_split, load_gt
from evaluate import f05

tag = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EXTRA = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cl = pd.read_parquet(f"work/pool_cl_train_{tag}.parquet")["cl"].values
gi = pd.read_parquet("work/gt_int.parquet")
S1 = Source("work", "train", "s1")
ids = S1.col("entity_id").to_numpy(zero_copy_only=False); ctry = S1.col("country").to_numpy(zero_copy_only=False)
gt = load_gt("dataset", "work")
tr, va = make_split(ids, ctry, gt)

# 2b on the full pool: pairwise precision/recall of pool clusters vs owners
owner = np.full(len(cl), -1, np.int64); owner[gi["p"].values] = gi["q"].values
m = cl >= 0
d = pd.DataFrame({"cl": cl[m], "o": owner[m]})
pairs_tot = (d.groupby("cl").size() * (d.groupby("cl").size() - 1)).sum()
do = d[d.o >= 0]
same = (do.groupby(["cl", "o"]).size() * (do.groupby(["cl", "o"]).size() - 1)).sum()
owned_pairs = (do.groupby("cl").size() * (do.groupby("cl").size() - 1)).sum()
og = pd.Series(owner[owner >= 0]).value_counts()
print(f"[{tag}] full pool: clustered {m.mean():.3f} | pairwise precision (both owned) {same / max(owned_pairs, 1):.4f} "
      f"| recall {same / max((og * (og - 1)).sum(), 1):.4f} | pairs with unowned {1 - owned_pairs / max(pairs_tot, 1):.4f}")

# 2c: expansion on val S1
cands = pd.concat([pd.read_parquet(f) for f in glob.glob(f"work/cands_train_*_{C.BLOCK_VERSION}_*.parquet")], ignore_index=True)
cv = cands[np.isin(cands["q"].values, va)][["q", "p", "p1"]].copy()
cv["cl"] = cl[cv["p"].values]
top = cv.sort_values(["q", "p1"], ascending=[True, False])
top = top[top.groupby("q").cumcount() < TOPN]
top = top[top["cl"] >= 0][["q", "cl"]].drop_duplicates()
members = pd.DataFrame({"p": np.flatnonzero(cl >= 0)}); members["cl"] = cl[members["p"].values]
add = top.merge(members, on="cl")[["q", "p"]]
key = lambda df: df["q"].values.astype(np.int64) << 32 | df["p"].values.astype(np.int64)
add = add[~np.isin(key(add), key(cv))].drop_duplicates()
add = add[add.groupby("q").cumcount() < EXTRA]
exp = pd.concat([cv[["q", "p"]], add], ignore_index=True)
gv = gi[np.isin(gi["q"].values, va)]
gk = key(gv)
tc = gv.groupby("q").size().reindex(va).fillna(0).values
def stats(df, name):
    hit = np.isin(key(df), gk)
    tp = pd.Series(hit.astype(int)).groupby(df["q"].values).sum().reindex(va).fillna(0).values
    # ceiling: perfect matcher on candidates -> P=1, R=tp/tc
    f = np.where(tc == 0, 1.0, np.where(tp > 0, 1.25 * (tp / np.maximum(tc, 1)) / (0.25 + tp / np.maximum(tc, 1)), 0.0))
    n = df.groupby("q").size().reindex(va).fillna(0)
    print(f"  {name}: pair_recall={hit.sum() / len(gv):.4f} ceiling_F05={f.mean():.4f} mean_cands={n.mean():.2f} p95={n.quantile(.95):.0f}"
          f" | added pairs precision={hit[len(cv):].mean() if len(df) > len(cv) else float('nan'):.4f}")
stats(cv, "stage-1 pruned (current)")
stats(exp, f"+ cluster expansion top{TOPN} cap+{EXTRA}")
