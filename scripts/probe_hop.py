import os
"""Second-hop retrieval probe: queries = each sampled val S1's top-N candidates (by stage-1 p1); retrieve top-k pool
neighbours of each query (name / addr / combined spaces, pool-only IDF); new candidates = union minus current set.
Reports pair recall / ceiling / cands before vs after, with a per-S1 cap on new candidates (ranked by hop score)."""
import sys, glob, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import config as C
from blocking import hash_texts, Space, _retrieval, topk_search
from run_pipeline import Source, make_split, load_gt

if __name__ == "__main__":
    country, n_s1, TOPN, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    S1, P = Source("work", "train", "s1"), Source("work", "train", "pool")
    ids = S1.col("entity_id").to_numpy(zero_copy_only=False); ctry = S1.col("country").to_numpy(zero_copy_only=False)
    tr, va = make_split(ids, ctry, load_gt("dataset", "work"))
    va_c = va[ctry[va] == country]
    samp = np.sort(np.random.RandomState(0).choice(va_c, min(n_s1, len(va_c)), replace=False))
    cands = pd.concat([pd.read_parquet(f, columns=["q", "p", "p1"]) for f in glob.glob(f"work/cands_train_{country}_{C.BLOCK_VERSION}_*.parquet")])
    cv = cands[np.isin(cands.q.values, samp)]
    pc = P.col("country").to_numpy(zero_copy_only=False)
    pidx = np.flatnonzero(pc == country)
    loc = pd.Series(np.arange(len(pidx)), index=pidx)
    pf = P.frame(["name_core", "addr_norm"], pidx)
    t = time.time()
    pn, pa = hash_texts("name", pf.name_core.tolist()), hash_texts("addr", pf.addr_norm.tolist())
    sn, sa = Space(C.NAME_MAX_DF).fit([pn]), Space(C.ADDR_MAX_DF).fit([pa])
    Pn, Pa = sn.weight(pn), sa.weight(pa)
    del pn, pa
    keep = np.concatenate([sn.keep, sa.keep])
    Pc = sp.hstack([Pn, Pa], format="csr")
    print(f"hash+weight {time.time()-t:.0f}s", flush=True)
    top = cv.sort_values(["q", "p1"], ascending=[True, False])
    top = top[top.groupby("q").cumcount() < TOPN]
    qrows = loc[top.p.values].values
    res = []
    for nm, Q, PP, kp in (("name", Pn, Pn, sn.keep), ("addr", Pa, Pa, sa.keep), ("comb", Pc, Pc, keep)):
        t = time.time()
        bi, bs = topk_search(_retrieval(Q[qrows], kp), _retrieval(PP, kp), K + 1)
        r = pd.DataFrame({"q": np.repeat(top.q.values, K + 1), "pl": bi.ravel(), "s": bs.ravel(), "src": np.repeat(qrows, K + 1)})
        r = r[(r.pl >= 0) & (r.pl != r.src)]
        res.append(r[["q", "pl", "s"]])
        print(f"topk {nm} {time.time()-t:.0f}s", flush=True)
    hop = pd.concat(res).groupby(["q", "pl"], as_index=False)["s"].max()
    hop["p"] = pidx[hop.pl.values]
    key = lambda d: d.q.values.astype(np.int64) << 32 | d.p.values.astype(np.int64)
    hop = hop[~np.isin(key(hop), key(cv))].sort_values(["q", "s"], ascending=[True, False])
    gi = pd.read_parquet("work/gt_int.parquet"); gv = gi[np.isin(gi.q.values, samp)]
    gk = key(gv); tc = gv.groupby("q").size().reindex(samp).fillna(0).values
    def rep(df, name):
        hit = np.isin(key(df), gk)
        tp = pd.Series(hit.astype(int)).groupby(df.q.values).sum().reindex(samp).fillna(0).values
        R = tp / np.maximum(tc, 1)
        f = np.where(tc == 0, 1.0, np.where(tp > 0, 1.25 * R / (0.25 + R), 0.0))
        n = df.groupby("q").size().reindex(samp).fillna(0)
        print(f"  {name}: pair_recall={hit.sum()/len(gv):.4f} ceiling={f.mean():.4f} mean_cands={n.mean():.2f} p95={n.quantile(.95):.0f}", flush=True)
    print(f"{country} n_s1={len(samp)} TOPN={TOPN} K={K}")
    rep(cv[["q", "p"]], "current")
    for cap in (3, 5, 8, 12, 20):
        h = hop[hop.groupby("q").cumcount() < cap]
        rep(pd.concat([cv[["q", "p"]], h[["q", "p"]]]), f"+hop cap {cap} (new-pair precision {np.isin(key(h), gk).mean():.3f})")
