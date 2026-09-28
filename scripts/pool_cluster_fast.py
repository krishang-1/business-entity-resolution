import os
"""Vectorized pool clustering over the full train/test pool (global pool index = S2 rows then S3 rows).
Rules: L1 exact name_core+addr_norm; L2 exact name_core + shared house number (first 2 numbers); L4 same non-empty
addr_norm and skeleton-name token_set >= 80. Keys with frequency > MAX_KEY skipped; components > CAP dropped.
Usage: pool_cluster_fast.py <split> <rules per country e.g. US=L1,L2,L4;India=L1,L4;*=L1,L4>"""
import sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from rapidfuzz import fuzz
from normalize import skeleton

MAX_KEY, CAP = 50, 12


def chain_edges(keys, rows):
    """Edges between consecutive members of each key group of size 2..MAX_KEY. rows: global row ids aligned to keys."""
    s = pd.DataFrame({"k": keys, "r": rows})
    s["n"] = s.groupby("k")["r"].transform("size")
    s = s[(s["n"] >= 2) & (s["n"] <= MAX_KEY)].sort_values(["k", "r"])
    same = s["k"].values[1:] == s["k"].values[:-1]
    return np.stack([s["r"].values[:-1][same], s["r"].values[1:][same]], 1)


def cluster_country(df, rules):
    n = len(df)
    name, addr = df["name_core"].values, df["addr_norm"].values
    loc = np.arange(n)
    E = []
    if "L1" in rules:
        m = (name != "") & (addr != "")
        E.append(chain_edges((name + "|" + addr)[m], loc[m]))
    if "L2" in rules:
        nr = df["addr_nums"].str.split().str[:2].explode().dropna()
        nr = nr[nr != ""]
        ii = nr.index.values
        m = name[ii] != ""
        E.append(chain_edges(name[ii][m] + "|" + nr.values[m], ii[m]))
    if "L4" in rules:
        m = addr != ""
        e = chain_edges(addr[m], loc[m])
        if len(e):
            sk = {}
            def skl(i):
                if i not in sk:
                    sk[i] = " ".join(skeleton(t) for t in name[i].split())
                return sk[i]
            ok = np.fromiter((fuzz.token_set_ratio(skl(a), skl(b)) >= 80 for a, b in e), bool, len(e))
            E.append(e[ok])
    E = np.concatenate([x for x in E if len(x)]) if any(len(x) for x in E) else np.zeros((0, 2), int)
    g = sp.coo_matrix((np.ones(len(E), np.int8), (E[:, 0], E[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab)[lab]
    lab = np.where((size >= 2) & (size <= CAP), lab, -1)
    return lab, len(E)


if __name__ == "__main__":
    split = sys.argv[1]
    spec = dict(x.split("=") for x in sys.argv[2].split(";"))
    cols = ["entity_id", "country", "name_core", "addr_norm", "addr_nums"]
    pool = pd.concat([pd.read_parquet(f"work/norm_{split}_s{i}.parquet", columns=cols) for i in (2, 3)], ignore_index=True)
    cl = np.full(len(pool), -1, np.int64)
    off = 0
    for c in pool["country"].unique():
        t = time.time()
        idx = np.flatnonzero(pool["country"].values == c)
        rules = spec.get(c, spec.get("*", "L1,L4")).split(",")
        lab, ne = cluster_country(pool.iloc[idx].reset_index(drop=True), rules)
        m = lab >= 0
        cl[idx[m]] = lab[m] + off
        off += lab.max() + 1 if m.any() else 0
        print(f"{split} {c}: n={len(idx)} rules={rules} edges={ne} clustered={m.mean():.3f} clusters={len(np.unique(lab[m]))} {time.time()-t:.0f}s", flush=True)
    tag = sys.argv[3] if len(sys.argv) > 3 else "v1"
    pd.DataFrame({"cl": cl}).to_parquet(f"work/pool_cl_{split}_{tag}.parquet", index=False)
