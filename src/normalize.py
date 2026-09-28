"""Country-agnostic text normalization. All dictionaries are hand-written constants."""
import re
import os
import time
from multiprocessing import Pool

import pandas as pd
from unidecode import unidecode

LEGAL = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt",
    "plc": "plc", "pc": "pc", "pllc": "pllc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "tbk": "tbk", "lllp": "llp", "ag": "ag",
    "bv": "bv", "nv": "nv", "pty": "pty", "opc": "opc",
}
NAME_STOP = {"the", "and", "of", "et", "de", "du", "des", "la", "le", "les", "l", "d"}
WEB_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|fr|biz|info|us|co)$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
NONALNUM = re.compile(r"[^a-z0-9 ]+")
MIXED = re.compile(r"(?=.*[a-z])(?=.*\d)")
SPACES = re.compile(r"\s+")

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bld": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway", "pkwy": "parkway",
    "cv": "cove", "ct": "court", "cir": "circle", "pl": "place", "sq": "square", "ter": "terrace",
    "trl": "trail", "tr": "trail", "pk": "park", "mt": "mount", "ft": "fort", "twp": "township",
    "ste": "suite", "fl": "floor", "flr": "floor", "apt": "apartment", "bldg": "building", "nr": "near",
    "opp": "opposite", "r": "rue", "rte": "route", "ch": "chemin", "imp": "impasse", "all": "allee",
    "fbg": "faubourg", "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast",
    "nw": "northwest", "se": "southeast", "sw": "southwest", "mg": "marg", "extn": "extension",
    "ext": "extension", "sec": "sector", "ph": "phase", "vill": "village", "po": "post", "dist": "district",
    "cplx": "complex", "hsg": "housing", "soc": "society", "stn": "station", "jn": "junction",
}
ADDR_DROP = {"null", "none", "na", "no", "number", "cdp", "pmb", "unit", "suite", "apartment", "floor",
             "de", "du", "des", "la", "le", "les", "l", "d", "the", "of", "and", "cedex"}
STATES = {  # code -> name, applied to address tokens (US + India codes, hand-written)
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "newhampshire", "nj": "newjersey", "nm": "newmexico", "ny": "newyork", "nc": "northcarolina",
    "nd": "northdakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhodeisland", "sc": "southcarolina", "sd": "southdakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "westvirginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "districtofcolumbia",
}
STATE_NAMES = {  # multiword state names -> single token
    "new hampshire": "newhampshire", "new jersey": "newjersey", "new mexico": "newmexico", "new york": "newyork",
    "north carolina": "northcarolina", "north dakota": "northdakota", "rhode island": "rhodeisland",
    "south carolina": "southcarolina", "south dakota": "southdakota", "west virginia": "westvirginia",
    "district of columbia": "districtofcolumbia",
    "uttar pradesh": "uttarpradesh", "madhya pradesh": "madhyapradesh", "andhra pradesh": "andhrapradesh",
    "himachal pradesh": "himachalpradesh", "arunachal pradesh": "arunachalpradesh", "tamil nadu": "tamilnadu",
    "west bengal": "westbengal", "jammu and kashmir": "jammukashmir", "jammu kashmir": "jammukashmir",
}
IN_STATES = {"up": "uttarpradesh", "mp": "madhyapradesh", "ap": "andhrapradesh", "hp": "himachalpradesh",
             "tn": "tamilnadu", "wb": "westbengal", "mh": "maharashtra", "rj": "rajasthan", "gj": "gujarat",
             "ka": "karnataka", "kl": "kerala", "dl": "delhi", "hr": "haryana", "pb": "punjab", "br": "bihar",
             "tg": "telangana", "ts": "telangana", "or": "odisha", "od": "odisha", "jh": "jharkhand",
             "uk": "uttarakhand", "cg": "chhattisgarh", "jk": "jammukashmir"}
# Devanagari state names transliterated by unidecode (hand-written aliases)
TRANSLIT = {"uttr prdesh": "uttarpradesh", "raajsthaan": "rajasthan", "mhaaraassttr": "maharashtra",
            "dillii": "delhi", "hriyaannaa": "haryana", "gujraat": "gujarat", "krnaattk": "karnataka",
            "bihaar": "bihar", "mdhy prdesh": "madhyapradesh", "tmilnaaddu": "tamilnadu",
            "pshcim bngaal": "westbengal", "telngaanaa": "telangana", "pnjaab": "punjab", "kerl": "kerala"}
STATE_CODE = {v: k for k, v in STATES.items()}
STATE_CODE.update({v: k for k, v in IN_STATES.items() if k not in ("ts", "od", "or")})
LANDMARK = {"near", "opposite", "behind", "next", "beside", "adjacent", "above", "below"}
NUM_RE = re.compile(r"\d+")


SKEL_SUBS = [("ph", "f"), ("sh", "s"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
             ("ch", "c"), ("ck", "k"), ("q", "k"), ("w", "v"), ("z", "j"), ("x", "ks"), ("c", "k")]
VOWELS = re.compile(r"(?<!^)[aeiouyh]")
REPEAT = re.compile(r"(.)\1+")
TRANSLIT_LEGAL = {"prvt", "prbt", "prvr", "prvtt", "lmtd", "lmtt", "lmrd", "lmtdd", "lmd", "prpt", "prpr"}
# Several Indian scripts (e.g. Tamil, Malayalam) do not distinguish voiced/unvoiced consonants, so their romanizations
# swap b/p, d/t, g/k, j/c; merging them makes the skeleton transliteration-invariant.
SKEL_VOICE = os.environ.get("ER_SKEL_VOICE", "0") == "1"
VOICE_MAP = str.maketrans({"b": "p", "d": "t", "g": "k", "j": "c", "v": "p", "z": "s"})


def skeleton(t):
    for a, b in SKEL_SUBS:
        t = t.replace(a, b)
    s = VOWELS.sub("", t)
    if SKEL_VOICE:
        s = s.translate(VOICE_MAP)
    return REPEAT.sub(r"\1", s)


def _ascii(s):
    return s if s.isascii() else unidecode(s)


def norm_name(raw):
    s = _ascii(raw).lower().replace("�", "").replace("&", " and ")
    toks = []
    for t in s.split():
        m = WEB_RE.match(t.strip(".,;:()[]"))
        toks.append(m.group(1).replace("-", "") if m else t)
    s = NONALNUM.sub(" ", " ".join(toks))
    out = []
    for t in s.split():
        if MIXED.match(t) and sum(c.isalpha() for c in t) >= 2:
            t = t.translate(LEET)
        out.append(t)
    norm = " ".join(out)
    legal = sorted({LEGAL[t] for t in out if t in LEGAL})
    core = [t for t in out if t not in LEGAL and t not in NAME_STOP and skeleton(t) not in TRANSLIT_LEGAL]
    if not core:
        core = out
    return norm, " ".join(core), " ".join(legal)


def norm_addr(raw):
    s = _ascii(raw).lower().replace("�", "")
    s = NONALNUM.sub(" ", s.replace("#", " "))
    s = SPACES.sub(" ", s).strip()
    for k, v in STATE_NAMES.items():
        if k in s:
            s = s.replace(k, v)
    for k, v in TRANSLIT.items():
        if k in s:
            s = re.sub(r"\b" + k + r"\b", v, s)
    toks, nums, main = [], [], []
    skip = 0
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
            nums.append(t)
        else:
            if t in ADDR_DROP:
                continue
            t = ADDR_ABBR.get(t, t)
            t = STATE_CODE.get(t, t)
        toks.append(t)
        if t in LANDMARK:
            skip = 3
        elif skip:
            skip -= 1
        else:
            main.append(t)
    postcode = ""
    for n in nums:
        if len(n) in (5, 6):
            postcode = n
            break
    return " ".join(toks), " ".join(dict.fromkeys(nums)), postcode, " ".join(main)


def _norm_chunk(args):
    names, addrs = args
    n = [norm_name(x) for x in names]
    a = [norm_addr(x) for x in addrs]
    return n, a


def normalize_df(df, n_jobs=6, chunk=100_000):
    """Return DataFrame with normalized fields for a raw source DataFrame."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(names), chunk)]
    del names, addrs
    res_n, res_a = [], []
    with Pool(n_jobs) as p:
        for n, a in p.imap(_norm_chunk, jobs):
            res_n.extend(n)
            res_a.extend(a)
    del jobs
    out = pd.DataFrame({"entity_id": df["entity_id"].values, "country": df["country"].values})
    out["name_norm"], out["name_core"], out["name_legal"] = zip(*res_n) if res_n else ([], [], [])
    del res_n
    out["addr_norm"], out["addr_nums"], out["postcode"], out["addr_main"] = zip(*res_a) if res_a else ([], [], [], [])
    return out


def normalize_split(work, split, force=False):
    """Normalize raw parquet for split in {'train','test'} -> work/norm_<split>_s{1,2,3}.parquet"""
    for i in (1, 2, 3):
        dst = f"{work}/norm_{split}_s{i}.parquet"
        if os.path.exists(dst) and not force:
            continue
        t = time.time()
        df = pd.read_parquet(f"{work}/raw_{split}_s{i}.parquet")
        out = normalize_df(df)
        out.to_parquet(dst, index=False)
        print(f"normalized {split}_s{i} rows={len(out)} {time.time()-t:.0f}s", flush=True)
        del df, out


if __name__ == "__main__":
    import sys
    ex = ["Meek Pr�ncipa1 Co", "johnsviking.com", "THE INTERSTATE BIOPHARMA INC.", "Family Charities-II Corporation",
          "Quiroz [Yhn]", "राम मार्केटिंग प्राइवेट लिमिटेड", "Supérieure Sante EURL", "Maa Security Private Ltd",
          "LLC Global Compeelfmte Vendome", "Tayl0r Battery Holdings"]
    for e in ex:
        print(e, "->", norm_name(e))
    ad = ["##2620 SAWMILL RD, FORT RECOVERY, OH", "02620 Sawmill Rd, Recovery Twp, Ohio",
          "KH NO-436/437, KHUSHAL PARK MAIN ROAD, GHAZIABAD, उत्तर प्रदेश", "राजस्थान, P NO 4, CAFCUTTA FARM HOUSE, JANPATH, JAIPUR, NULL",
          "3100-3104 SEMINOLE LOOP, CROSSVILLE, TN", "12 RUE de lOrne, 44800, Saint-Herblain, Pays de la Loire",
          "51 R DE LA PLANCHE AU GUE, NANTES, Pays de la Loire", "Shop 4, Near SBI ATM, MG Rd, Pune 411001",
          "56 BELKNAP ST, NULL, WESTOROUGH CDP, MA", "Pessac, Gironde, 60 AVENUE DU PARC D'ESPAGNE"]
    for e in ad:
        print(e, "->", norm_addr(e))
