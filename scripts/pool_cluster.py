import os
"""Phase 2a/2b: unsupervised pool-side clustering (S2 ∪ S3, per country) + quality check against train GT.
Links (no labels used):
  L1 exact name_core AND same addr_norm
  L2 exact name_core AND shared house number
  L3 exact name_core AND one address empty
  L4 same non-empty addr_norm AND skeleton-name token_set >= 80
  L5 same skeleton name AND shared house number
Keys with pool frequency > MAX_KEY are skipped. Union-find with a component size cap."""
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rapidfuzz import fuzz
from normalize import skeleton

MAX_KEY = 50
CAP = 12


class UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64); self.s = np.ones(n, np.int32)

    def find(self, x):
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb or self.s[ra] + self.s[rb] > CAP:
            return False
        if self.s[ra] < self.s[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra; self.s[ra] += self.s[rb]
        return True


def key_groups(keys, valid):
    """Return list of index arrays for key groups of size 2..MAX_KEY."""
    pos = np.flatnonzero(valid)
    k = np.asarray(keys)[valid]
    g = pd.Series(k).groupby(k).indices
    return [pos[v] for v in g.values() if 2 <= len(v) <= MAX_KEY]


def cluster(df):
    n = len(df)
    uf = UF(n)
    name = df["name_core"].values; addr = df["addr_norm"].values
    nums = df["addr_nums"].values
    skel = np.array([" ".join(skeleton(t) for t in s.split()) for s in name], dtype=object)
    idx = np.arange(n)
    stats = {}
    # explode house numbers (first 2 numbers per record)
    num_rows = [(i, x) for i, s in enumerate(nums) for x in s.split()[:2]]
    nr = pd.DataFrame(num_rows, columns=["i", "num"])
    def link_groups(groups, tag, check=None):
        c = 0
        for g in groups:
            base = g[0]
            for j in g[1:]:
                if check is None or check(base, j):
                    c += uf.union(base, j)
        stats[tag] = c
    has_name = name != ""
    if "L1" in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["L1", "L2", "L4"]):
        link_groups(key_groups(name + "|" + addr, has_name & (addr != "")), "L1")
    kk = pd.Series(name[nr.i.values] + "|" + nr.num.values)
    rules = sys.argv[3].split(",") if len(sys.argv) > 3 else ["L1", "L2", "L4"]
    for tag, keyser in (("L2", kk), ("L5", pd.Series(skel[nr.i.values] + "|" + nr.num.values))):
        if tag not in rules:
            continue
        g = keyser.groupby(keyser.values).indices
        link_groups([nr.i.values[v] for v in g.values() if 2 <= len(v) <= MAX_KEY], tag)
    # L3: empty-address record joins a same-name record with an address (only if the name is not common)
    if "L3" not in (sys.argv[3].split(",") if len(sys.argv) > 3 else []):
        empt = np.zeros(n, bool)
    else:
        empt = (addr == "") & has_name
    ng = pd.Series(name[has_name]).groupby(name[has_name]).indices
    hn = idx[has_name]
    c = 0
    for v in ng.values():
        if 2 <= len(v) <= MAX_KEY:
            v = hn[v]
            e = v[empt[v]]; ne = v[~empt[v]]
            if len(e) and len(ne):
                for x in e:
                    c += uf.union(ne[0], x)
    stats["L3"] = c
    # L4: same address, similar skeleton name
    if "L4" in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["L1", "L2", "L4"]):
        link_groups(key_groups(addr, addr != ""), "L4", check=lambda a, b: fuzz.token_set_ratio(skel[a], skel[b]) >= 80)
    roots = np.fromiter((uf.find(i) for i in range(n)), np.int64, n)
    return roots, stats


if __name__ == "__main__":
    country = sys.argv[1]
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    t = time.time()
    cols = ["entity_id", "country", "name_core", "addr_norm", "addr_nums"]
    pool = pd.concat([pd.read_parquet(f"work/norm_train_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    pool = pool[pool.country == country].reset_index(drop=True)
    gt = pd.read_parquet("work/gt_pairs.parquet")
    owner = pd.Series(gt["source1_entity_id"].values, index=gt["mid"].values)
    pool["owner"] = owner.reindex(pool["entity_id"].values).fillna("").values
    if frac < 1:  # sample by owner so that clusters stay intact
        own = pool["owner"].unique()
        keep = set(np.random.RandomState(0).choice(own, int(len(own) * frac), replace=False))
        pool = pool[pool["owner"].isin(keep) | ((pool["owner"] == "") & (np.random.RandomState(1).rand(len(pool)) < frac))].reset_index(drop=True)
    roots, stats = cluster(pool)
    pool["cl"] = roots
    sz = pool.groupby("cl")["cl"].transform("size")
    print(f"{country} n={len(pool)} links={stats} clusters>1: {(sz > 1).mean():.3f} of records; size dist:",
          sz.clip(upper=13).value_counts().sort_index().to_dict(), f"{time.time()-t:.0f}s", flush=True)
    # pairwise quality within clusters
    m = pool[sz > 1]
    pr = m[["cl", "owner"]].merge(m[["cl", "owner"]], on="cl")
    pr = pr[pr.index % 1 == 0]
    tot = len(pr) - len(m)  # exclude self-pairs (approx: identical rows)
    both = (pr.owner_x != "") & (pr.owner_y != "")
    same = both & (pr.owner_x == pr.owner_y)
    n_both = both.sum() - (m.owner != "").sum(); n_same = same.sum() - (m.owner != "").sum()
    print(f"pairwise precision (both owned): {n_same / max(n_both, 1):.4f}  | pairs with an unowned member: "
          f"{((pr.owner_x == '') ^ (pr.owner_y == '')).mean():.4f}")
    own = pool[pool.owner != ""]
    ow = own.groupby("owner").size()
    tot_same = int((ow * (ow - 1)).sum())
    print(f"pairwise recall (same-owner pool pairs linked): {n_same / max(tot_same, 1):.4f}")
    pool[["entity_id", "cl"]].to_parquet(f"work/pool_clusters_train_{country}.parquet", index=False)
