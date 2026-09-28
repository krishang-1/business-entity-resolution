"""Candidate generation, per country partition.
Hashed TF-IDF spaces: name (words, 4-char prefixes/suffixes, consonant skeletons, concatenation, bigrams) and
address (word uni+bigrams). Three top-k lists per S1: name, address, combined (name+address). Top-k is computed as
S1-chunk x pool-chunk sparse products merged with argpartition; features with doc freq > MAX_DF are excluded from the
retrieval product (keeps it sparse => sub-quadratic). Final candidates = top-a name ∪ top-b addr ∪ top-c comb,
capped at MAX_CAND per S1 by best list rank."""
import gc
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

import config as C
from evaluate import Stage
from normalize import skeleton

NF = 2 ** 22


def name_tokens(s):
    toks = s.split()
    out = list(toks) + ["~" + skeleton(t) for t in toks if len(t) >= 3]
    for t in toks:
        if len(t) >= 6:
            out.append("<" + t[:4]); out.append(">" + t[-4:])
    if toks:
        out.append("#" + "".join(toks))
    for a, b in zip(toks, toks[1:]):
        out.append(a + "_" + b)
    return out


def addr_tokens(s):
    toks = s.split()
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


def _hv(kind):
    return HashingVectorizer(analyzer=name_tokens if kind == "name" else addr_tokens, n_features=NF,
                             alternate_sign=False, norm=None, dtype=np.float32)


def _hash_job(args):
    kind, texts = args
    m = _hv(kind).transform(texts)
    m.data = np.log1p(m.data, dtype=np.float32)  # sublinear tf
    return m


def hash_texts(kind, texts, chunk=200_000, n_jobs=6):
    jobs = [(kind, texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
    if not jobs:
        return sp.csr_matrix((0, NF), dtype=np.float32)
    if len(jobs) == 1:
        return _hash_job(jobs[0])
    with Pool(n_jobs) as p:
        mats = p.map(_hash_job, jobs)
    return sp.vstack(mats, format="csr")


class Space:
    """Hashed TF-IDF space fitted on a partition (S1 + pool texts)."""

    def __init__(self, max_df):
        self.max_df = max_df

    def fit(self, mats):
        n = sum(m.shape[0] for m in mats)
        df = np.zeros(NF, np.int64)
        for m in mats:
            df += np.bincount(m.indices, minlength=NF)
        self.idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
        self.keep = (df <= self.max_df).astype(np.float32)
        return self

    def weight(self, m):
        m = m.copy()
        m.data *= self.idf[m.indices]
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel()).astype(np.float32)
        norms[norms == 0] = 1
        return sp.diags(1 / norms).dot(m).tocsr().astype(np.float32)


def _retrieval(m, keep):
    r = m.copy()
    r.data *= keep[r.indices]
    r.eliminate_zeros()
    return r


_PT = None


def _init_worker(PT):
    global _PT
    _PT = PT


def _chunk_topk(args):
    """Top-k of one S1 chunk against the worker's pool chunk. Returns local (idx, score) arrays."""
    Qc, k, p0 = args
    R = (Qc @ _PT).tocsr()
    n = R.shape[0]
    bi = np.full((n, k), -1, np.int32)
    bs = np.full((n, k), -1, np.float32)
    ip, ix, dt = R.indptr, R.indices, R.data
    for r in range(n):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d, c = dt[a:b], ix[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k)[:k]
            d, c = d[sel], c[sel]
        bs[r, :len(d)] = d
        bi[r, :len(d)] = c + p0
    return bi, bs


def _merge(best_i, best_s, rows, ci, cs, k):
    s_all = np.concatenate([best_s[rows], cs], axis=1)
    i_all = np.concatenate([best_i[rows], ci], axis=1)
    top = np.argpartition(-s_all, k - 1, axis=1)[:, :k]
    best_s[rows] = np.take_along_axis(s_all, top, 1)
    best_i[rows] = np.take_along_axis(i_all, top, 1)


def topk_search(Q, P, k, s1_chunk=C.S1_CHUNK, pool_chunk=C.POOL_CHUNK, n_jobs=C.TOPK_JOBS):
    """Q: (nq,F) csr, P: (np,F) csr. Returns idx (nq,k) int32 (-1 = none) and score (nq,k) float32.
    Loops over pool chunks; within a pool chunk, S1 chunks are scored in parallel worker processes."""
    nq = Q.shape[0]
    best_i = np.full((nq, k), -1, np.int32)
    best_s = np.full((nq, k), -1, np.float32)
    starts = list(range(0, nq, s1_chunk))
    for p0 in range(0, P.shape[0], pool_chunk):
        PT = P[p0:p0 + pool_chunk].T.tocsr()
        jobs = [(Q[q0:q0 + s1_chunk], k, p0) for q0 in starts]
        if n_jobs > 1 and len(jobs) > 1:
            with Pool(n_jobs, initializer=_init_worker, initargs=(PT,)) as pl:
                for q0, (ci, cs) in zip(starts, pl.imap(_chunk_topk, jobs)):
                    _merge(best_i, best_s, slice(q0, q0 + len(ci)), ci, cs, k)
        else:
            _init_worker(PT)
            for q0, job in zip(starts, jobs):
                ci, cs = _chunk_topk(job)
                _merge(best_i, best_s, slice(q0, q0 + len(ci)), ci, cs, k)
            _init_worker(None)
        del PT, jobs
        gc.collect()
    return best_i, best_s


def _ranked(idx, score, q0=0):
    """Flatten a top-k result to (q, p, rank) with rank 1 = best; drops empty slots. q offset by q0."""
    order = np.argsort(-score, axis=1, kind="stable")
    idx = np.take_along_axis(idx, order, 1)
    nq, k = idx.shape
    q = np.repeat(np.arange(q0, q0 + nq, dtype=np.int32), k)
    rank = np.tile(np.arange(1, k + 1, dtype=np.int16), nq)
    p = idx.ravel()
    m = p >= 0
    return q[m], p[m], rank[m]


def rowwise_cos(A, B):
    return np.asarray(A.multiply(B).sum(1)).ravel().astype(np.float32)


STAGE1_FEATS = ["r_name", "r_addr", "r_comb", "name_cos", "addr_cos", "comb", "s1_rank_comb", "s1_rank_name",
                "s1_gap_comb", "s1_gap_name", "s1_gap_addr", "s1_n_union", "n_lists"]


def stage1_features(df):
    """Cheap per-pair blocking features; all per-S1 aggregates, so chunking by S1 is exact."""
    g = df.groupby("q")
    df["comb"] = df["name_cos"] + df["addr_cos"]
    df["s1_rank_comb"] = g["comb"].rank(ascending=False, method="first").astype(np.float32)
    df["s1_rank_name"] = g["name_cos"].rank(ascending=False, method="first").astype(np.float32)
    df["s1_gap_comb"] = (g["comb"].transform("max") - df["comb"]).astype(np.float32)
    df["s1_gap_name"] = (g["name_cos"].transform("max") - df["name_cos"]).astype(np.float32)
    df["s1_gap_addr"] = (g["addr_cos"].transform("max") - df["addr_cos"]).astype(np.float32)
    df["s1_n_union"] = g["p"].transform("size").astype(np.float32)
    df["n_lists"] = ((df["r_name"] < 99).astype(np.int8) + (df["r_addr"] < 99) + (df["r_comb"] < 99)).astype(np.float32)
    return df


def prune(df, pruner):
    """Apply the learned stage-1 blocker: keep p1 >= thr, always keep each S1's top-1 by comb and by name_cos,
    cap at MAX_CAND by p1. Output = final candidate set scored by the matcher."""
    bst, thr, max_cand = pruner
    df["p1"] = bst.predict(df[STAGE1_FEATS].values.astype(np.float32), num_threads=C.TOPK_JOBS)
    keep = (df["p1"].values >= thr) | (df["s1_rank_comb"].values == 1) | (df["s1_rank_name"].values == 1)
    d = df[keep].sort_values(["q", "p1"], ascending=[True, False])
    d = d[d.groupby("q").cumcount() < max_cand]
    return d.sort_values(["q", "p"]).reset_index(drop=True)


def block_partition(s1, pool, k_name=C.K_NAME, k_addr=C.K_ADDR, k_comb=C.K_COMB, pruner=None, q_chunk=100_000):
    """s1/pool: normalized DataFrames (positional). Returns the union of the three top-k lists with ranks (99 = not
    in that list), cosines and stage-1 features; pruned by `pruner` when given. Union/pruning run in S1 chunks."""
    with Stage(f"hash n_s1={len(s1)} n_pool={len(pool)}"):
        qn, pn = hash_texts("name", s1["name_core"].tolist()), hash_texts("name", pool["name_core"].tolist())
        qa, pa = hash_texts("addr", s1["addr_norm"].tolist()), hash_texts("addr", pool["addr_norm"].tolist())
        sn, sa = Space(C.NAME_MAX_DF).fit([qn, pn]), Space(C.ADDR_MAX_DF).fit([qa, pa])
        Qn, Pn, Qa, Pa = sn.weight(qn), sn.weight(pn), sa.weight(qa), sa.weight(pa)
        del qn, pn, qa, pa
        gc.collect()
    with Stage("topk name"):
        ln = topk_search(_retrieval(Qn, sn.keep), _retrieval(Pn, sn.keep), k_name)
    with Stage("topk addr"):
        la = topk_search(_retrieval(Qa, sa.keep), _retrieval(Pa, sa.keep), k_addr)
    with Stage("topk combined"):
        keep = np.concatenate([sn.keep, sa.keep])
        Qc = _retrieval(sp.hstack([Qn, Qa], format="csr"), keep)
        Pc = _retrieval(sp.hstack([Pn, Pa], format="csr"), keep)
        lc = topk_search(Qc, Pc, k_comb)
        del Qc, Pc
        gc.collect()
    out = []
    with Stage("union + cosines + stage1" + (" + prune" if pruner else "")):
        for c0 in range(0, len(s1), q_chunk):
            c1 = c0 + q_chunk
            parts = []
            for col, (ii, ss) in (("r_name", ln), ("r_addr", la), ("r_comb", lc)):
                q, p, r = _ranked(ii[c0:c1], ss[c0:c1], c0)
                d = pd.DataFrame({"q": q, "p": p, "r_name": np.int16(99), "r_addr": np.int16(99), "r_comb": np.int16(99)})
                d[col] = r
                parts.append(d)
            df = pd.concat(parts, ignore_index=True).groupby(["q", "p"], sort=False).min().reset_index()
            del parts
            name_cos = np.empty(len(df), np.float32); addr_cos = np.empty(len(df), np.float32)
            step = 1_000_000
            for i in range(0, len(df), step):
                qq, pp = df["q"].values[i:i + step], df["p"].values[i:i + step]
                name_cos[i:i + step] = rowwise_cos(Qn[qq], Pn[pp])
                addr_cos[i:i + step] = rowwise_cos(Qa[qq], Pa[pp])
            df["name_cos"], df["addr_cos"] = name_cos, addr_cos
            df = stage1_features(df)
            if pruner:
                df = prune(df, pruner)
            out.append(df)
            gc.collect()
    return pd.concat(out, ignore_index=True)


def select(df, a=C.A_NAME, b=C.A_ADDR, c=C.A_COMB, max_cand=C.MAX_CAND):
    """Per-list rank cap then per-S1 cap by best rank (ties broken by name_cos+addr_cos)."""
    d = df[(df["r_name"] <= a) | (df["r_addr"] <= b) | (df["r_comb"] <= c)]
    best = np.minimum(np.minimum(d["r_name"].values, d["r_addr"].values), d["r_comb"].values)
    d = d.assign(_b=best, _s=-(d["name_cos"].values + d["addr_cos"].values)).sort_values(["q", "_b", "_s"])
    d = d[d.groupby("q").cumcount() < max_cand]
    return d.drop(columns=["_b", "_s"]).sort_values(["q", "p"]).reset_index(drop=True)


def partitions(s1_country, pool_country):
    """Yield (label, s1_mask, pool_mask); labels with too few records fall back to the whole pool."""
    s1_country = pd.Series(s1_country); pool_country = pd.Series(pool_country)
    pc = pool_country.value_counts()
    for lab, n in s1_country.value_counts().items():
        s1m = (s1_country == lab).values
        if n >= C.MIN_PARTITION and pc.get(lab, 0) >= C.MIN_PARTITION:
            yield lab, s1m, (pool_country == lab).values
        else:
            yield lab, s1m, np.ones(len(pool_country), bool)
