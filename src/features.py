"""Pair features. String similarities via rapidfuzz in a multiprocessing pool over aligned pair chunks."""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from normalize import skeleton

FUZZ_COLS = ["n_ratio", "n_partial", "n_tsort", "n_tset", "n_jw", "n_nospace", "nn_tset", "n_first_eq", "n_acr",
             "n_tok_jacc", "a_tset", "a_partial", "a_tsort", "am_tset", "a_tok_jacc", "num_jacc", "num_first_eq",
             "num_conflict", "num_both", "pc_eq", "pc_conflict", "legal_eq", "legal_conflict", "legal_one",
             "n_len_a", "n_len_b", "n_ntok_b", "a_len_b", "num_near", "n_skel", "n_contain"]


def _jacc(a, b):
    if not a and not b:
        return -1.0
    return len(a & b) / len(a | b)


def _num_near(u, v):
    for x in u:
        for y in v:
            if x == y or (len(x) >= 2 and len(y) >= 2 and (x.startswith(y) or y.startswith(x) or x.endswith(y)
                                                          or y.endswith(x))) or Levenshtein.distance(x, y) <= 1:
                return 1.0
    return 0.0


def _skel(s):
    return " ".join(skeleton(t) for t in s.split())


def _pair_chunk(args):
    na, nb, nna, nnb, aa, ab, ama, amb, numa, numb, pca, pcb, la, lb = args
    out = np.zeros((len(na), len(FUZZ_COLS)), np.float32)
    for i in range(len(na)):
        a, b = na[i], nb[i]
        ta, tb = a.split(), b.split()
        sa, sb = set(ta), set(tb)
        acr_a = "".join(t[0] for t in ta) if len(ta) > 1 else ""
        acr_b = "".join(t[0] for t in tb) if len(tb) > 1 else ""
        xa, xb = aa[i], ab[i]
        stxa, stxb = set(xa.split()), set(xb.split())
        nua, nub = numa[i].split(), numb[i].split()
        su, sv = set(nua), set(nub)
        both = bool(su) and bool(sv)
        pa, pb = pca[i], pcb[i]
        ga, gb = set(la[i].split()), set(lb[i].split())
        out[i] = (
            fuzz.ratio(a, b), fuzz.partial_ratio(a, b), fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b),
            JaroWinkler.normalized_similarity(a, b), fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")),
            fuzz.token_set_ratio(nna[i], nnb[i]),
            float(bool(ta) and bool(tb) and ta[0] == tb[0]),
            float(bool(acr_a and acr_a == b.replace(" ", "")) or bool(acr_b and acr_b == a.replace(" ", ""))),
            _jacc(sa, sb),
            fuzz.token_set_ratio(xa, xb) if xa and xb else -1, fuzz.partial_ratio(xa, xb) if xa and xb else -1,
            fuzz.token_sort_ratio(xa, xb) if xa and xb else -1,
            fuzz.token_set_ratio(ama[i], amb[i]) if ama[i] and amb[i] else -1,
            _jacc(stxa, stxb),
            _jacc(su, sv) if both else -1, float(both and nua[0] == nub[0]), float(both and not (su & sv)), float(both),
            float(bool(pa) and pa == pb), float(bool(pa) and bool(pb) and pa != pb),
            float(bool(ga) and ga == gb), float(bool(ga) and bool(gb) and not (ga & gb)), float(bool(ga) != bool(gb)),
            len(a), len(b), len(tb), len(xb),
            _num_near(su, sv) if both else -1, fuzz.token_set_ratio(_skel(a), _skel(b)),
            float(bool(sa) and bool(sb) and (sa <= sb or sb <= sa)),
        )
    return out


def string_features(s1, pool, q, p, n_jobs=6, chunk=100_000):
    """q, p: global positional index arrays into s1/pool normalized frames."""
    cols = {}
    for c in ["name_core", "name_norm", "addr_norm", "addr_main", "addr_nums", "postcode", "name_legal"]:
        cols[c] = (s1[c].values, pool[c].values)
    jobs = []
    for i in range(0, len(q), chunk):
        qq, pp = q[i:i + chunk], p[i:i + chunk]
        args = []
        for c in ["name_core", "name_norm", "addr_norm", "addr_main", "addr_nums", "postcode", "name_legal"]:
            A, B = cols[c]
            args += [A[qq].tolist(), B[pp].tolist()]
        # reorder to match _pair_chunk signature
        na, nb, nna, nnb, aa, ab, ama, amb, numa, numb, pca, pcb, la, lb = args
        jobs.append((na, nb, nna, nnb, aa, ab, ama, amb, numa, numb, pca, pcb, la, lb))
    res = []
    with Pool(n_jobs) as pl:
        for r in pl.imap(_pair_chunk, jobs):
            res.append(r)
    del jobs
    arr = np.vstack(res) if res else np.zeros((0, len(FUZZ_COLS)), np.float32)
    return pd.DataFrame(arr, columns=FUZZ_COLS)


def context_features(df):
    """df has q, p, name_cos, addr_cos, comb (all S1 candidates of the processed set)."""
    out = pd.DataFrame(index=df.index)
    g = df.groupby("q")
    out["rank_comb"] = g["comb"].rank(ascending=False, method="min").astype(np.float32)
    out["rank_name"] = g["name_cos"].rank(ascending=False, method="min").astype(np.float32)
    out["rank_addr"] = g["addr_cos"].rank(ascending=False, method="min").astype(np.float32)
    out["gap_comb"] = (g["comb"].transform("max") - df["comb"]).astype(np.float32)
    out["gap_name"] = (g["name_cos"].transform("max") - df["name_cos"]).astype(np.float32)
    out["gap_addr"] = (g["addr_cos"].transform("max") - df["addr_cos"]).astype(np.float32)
    out["n_cands"] = g["p"].transform("size").astype(np.float32)
    return out
