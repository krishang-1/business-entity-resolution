import os
import sys
import time
import numpy as np
import pandas as pd


def rss_mb():
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return r, r
    except Exception:
        return float("nan"), float("nan")


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        cur, peak = rss_mb()
        print(f"[stage] {self.name}: {time.time()-self.t:.1f}s rss={cur:.0f}MB peak={peak:.0f}MB", flush=True)


def f05(pred, true):
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def score(pred_map, true_map, s1_ids, country=None, verbose=True):
    """pred_map/true_map: dict s1 -> set. Returns dict of metrics."""
    fs, tp, npred, ntrue = [], 0, 0, 0
    sing_ok, sing_n = 0, 0
    for s in s1_ids:
        p, t = pred_map.get(s, set()), true_map.get(s, set())
        fs.append(f05(p, t))
        tp += len(p & t); npred += len(p); ntrue += len(t)
        if not t:
            sing_n += 1; sing_ok += (not p)
    fs = np.array(fs)
    out = dict(F05=fs.mean(), P=tp / max(npred, 1), R=tp / max(ntrue, 1), singleton_acc=sing_ok / max(sing_n, 1), n=len(fs))
    if country is not None:
        c = pd.Series(country, index=range(len(fs)))
        for k, idx in c.groupby(c).groups.items():
            out[f"F05_{k}"] = fs[list(idx)].mean()
        nt = np.array([min(len(true_map.get(s, ())), 3) for s in s1_ids])
        for b in range(4):
            if (nt == b).any():
                out[f"F05_m{b}{'+' if b == 3 else ''}"] = fs[nt == b].mean()
    if verbose:
        print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()), flush=True)
    return out


def sanity():
    assert f05({"a", "b"}, {"a", "b"}) == 1.0
    assert abs(f05({"a", "b", "c"}, {"a", "c"}) - 0.7142857) < 1e-6
    assert f05(set(), set()) == 1.0 and f05({"a"}, set()) == 0.0
    tm = {"x": set(), "y": {"a"}, "z": set(), "w": {"b"}}
    assert score({}, tm, list(tm), verbose=False)["F05"] == 0.5
    assert score(tm, tm, list(tm), verbose=False)["F05"] == 1.0
    print("scorer sanity OK (perfect=1.0, empty=singleton frac, example=0.714)")


def blocking_report(cand, gt_pairs, s1_ids, true_map):
    """cand: DataFrame(s1, cid). gt_pairs: DataFrame(source1_entity_id, mid)."""
    s1set = set(s1_ids)
    g = gt_pairs[gt_pairs["source1_entity_id"].isin(s1set)]
    key_c = set(zip(cand["s1"], cand["cid"]))
    hit = np.fromiter(((a, b) in key_c for a, b in zip(g["source1_entity_id"], g["mid"])), bool, len(g))
    per = cand.groupby("s1").size().reindex(s1_ids, fill_value=0)
    cmap = cand.groupby("s1")["cid"].agg(set).to_dict()
    ceil = np.mean([f05(true_map.get(s, set()) & cmap.get(s, set()), true_map.get(s, set())) for s in s1_ids])
    out = dict(pair_recall=hit.mean(), ceiling_F05=ceil, mean_cands=per.mean(), median_cands=per.median(),
               p95_cands=per.quantile(0.95), max_cands=per.max(), total_pairs=len(cand))
    print("blocking: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()), flush=True)
    return out


if __name__ == "__main__":
    sanity()
