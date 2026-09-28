# Methodology — Business Entity Resolution (Amazon ML Challenge 2026)

Author: Krishang Raju · 2026-09-28

---

## 1. Executive Summary
A four-stage, fully offline pipeline: (1) a country-partitioned, sub-quadratic wide net (hashed TF-IDF top-k
retrieval over three lists: name, address, combined name+address), (2) a learned first-stage blocker (small LightGBM on
cheap blocking features) that prunes the wide net to the final candidate set, (3) a LightGBM (MIT) pair classifier on
55 string/number/context features, and (4) a stacked pass-2 LightGBM that adds per-S1 probability context and sibling
support, followed by a decision rule (probability threshold + one-owner assignment + per-S1 relative rule) tuned directly
for macro F0.5. On a held-out validation set of 88,272 train S1 entities it reaches
**macro F0.5 = 0.9589 (precision 0.9877, recall 0.9175)**, with blocking pair recall 0.954 at 21.4 candidates per S1.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured EDA facts (train unless noted):
- Train: 2,206,821 S1 (US 1,323,633 / India 883,188), 5,034,616 S2, 5,285,603 S3. Test: 1,732,544 S1
  (India 809,986 / US 663,106 / **France 259,452**), 4,887,273 S2, 5,082,316 S3. France never appears in train.
- Singleton S1 (no match): 5.58%. Matches per S1: 0:123k, 1:119k, 2:375k, 3:531k, 4:484k, 5:322k, 6:165k, 7:64k, 8+:23k
  (7,638,365 true pairs; 48.4% from S2).
- **No S2/S3 record matches more than one S1** → a one-owner constraint is valid.
- 81% of matched S1 have >1 match in the *same* source → no "top-1 per source" rule.
- 25–27% of S2/S3 records match nothing (distractors). Empty names 0%; empty addresses ~3% of S2/S3.
- Noise patterns seen: typos and leet substitutions (`Principa1`, `Tayl0r`), U+FFFD replacement characters, website
  names (`johnsviking.com`), junk names with the correct address (`Zephyumacalo`), legal suffixes added/dropped/moved,
  `Services`/`Center` suffixes, brackets; names in native Indian scripts (Devanagari, Tamil, Telugu, Bengali,
  Gujarati, Malayalam); addresses with reordered components, `ST/RD/DR` abbreviations, state codes vs full names vs
  Devanagari state names, `NULL`, `##`, leading zeros (`02620`), number ranges (`3100-3104`), dropped digits,
  truncated addresses. France: SARL/SAS/EURL/SCI/SA, `R`/`RUE`, `BD`, `AV`, `BIS`, occasional 5-digit postcodes.

### 2.2 Solution Strategy
**Approach Type:** Wide-net blocking + learned pruning + Classifier (LightGBM) + constrained decision rule
**Core Innovation:** Three complementary retrieval lists (name / address / combined) in hashed TF-IDF spaces with a
phonetic consonant-skeleton token (robust to transliteration), unioned into a wide net and pruned by a learned
first-stage blocker whose threshold is set to keep 99.5% of the wide net's true pairs; country used only as a blocking
partition key, so the method transfers to unseen countries (France) without any country-specific code.

---

## 3. Candidate Generation (Blocking)

- **Normalization (country-agnostic, hand-written rules in `normalize.py`):** unidecode, lowercase, `&`→`and`,
  website → name, leet fixes inside alphanumeric tokens, legal-form extraction/removal (US/IN/FR forms incl.
  transliterated `praaivett`/`limittedd` via skeleton), address abbreviation canonicalization (st/rd/ave/av/bd/r/…),
  state full name → code (applied per country, so US/IN collisions are harmless), house numbers without leading zeros,
  5/6-digit postcode extraction, landmark-phrase stripping (`near …`, `opposite …`).
- **Blocking keys / representation:** per country partition (fallback to the whole pool if a label has < 50 records).
  Name space: words, 4-char prefixes/suffixes, consonant skeletons (`tirupati`/`tirupti` → `trpt`), whole-name
  concatenation, word bigrams. Address space: word unigrams + bigrams. Both hashed (2^22), sublinear TF, IDF fitted
  on the partition's S1+pool (unsupervised), L2 normalized.
- **Retrieval (wide net):** cosine top-20 by name, top-30 by address, top-40 by combined name+address, computed as S1-chunk
  (2,000) × pool-chunk (1M) sparse products in 4 worker processes, merged with `argpartition`. Features with
  document frequency > 2,000 are excluded from the retrieval product, which keeps the products sparse
  (sub-quadratic in practice); they still count in the full cosines used as model features.
- **Learned first-stage blocker (final candidate set = `candidate_pairs.tsv`, exactly what the matcher scores):** the
  union of the three lists (58.8 candidates/S1 on validation, pair recall 0.9571) is scored by a 100-tree LightGBM on 13
  cheap features (the three list ranks, name/address/combined cosines, rank and gap to the S1's best by combined and
  name cosine, gaps by address, union size, number of lists that retrieved the pair), trained on train-split S1 only.
  Kept: p ≥ 0.001 (the largest threshold keeping ≥ 99.5% of the wide net's validation true pairs), plus each S1's
  top-1 by combined and by name cosine; capped at 40 per S1.
- **Candidate pairs generated (test):** 1,732,544 S1; mean 23.38, median 22, p95 40, max 40 per S1
  (US 19.61, India 22.39, France 36.08); ≈40.5M pairs out of ~1.7M × 10M possible. France keeps more candidates:
  the stage-1 blocker is less confident on the unseen country, so more pairs survive (often up to the cap).
- **Validation blocking quality (88,272 val S1 vs the full per-country train pool):** pair recall **0.9543**,
  entity-level ceiling macro F0.5 **0.9836**, mean 21.37 / median 20 / p95 40 / max 40 candidates per S1
  (previous rank-cap blocking: 0.9354 / 0.9769 / mean 18.45).

Stage-1 threshold sweep on validation (kept = share of the wide net's true pairs retained):

| stage-1 threshold | kept | pair recall | mean cands/S1 | p95 |
|---|---|---|---|---|
| 0.1 | 0.961 | 0.920 | 5.08 | 8 |
| 0.03 | 0.977 | 0.935 | 6.00 | 10 |
| 0.01 | 0.986 | 0.944 | 7.65 | 13 |
| 0.005 | 0.990 | 0.947 | 9.41 | 17 |
| 0.003 | 0.993 | 0.950 | 11.79 | 22 |
| 0.002 | 0.994 | 0.952 | 14.58 | 28 |
| **0.001 (chosen)** | **0.997** | **0.954** | **21.37** | **40** |

At threshold 0.03 the learned blocker matches the old rank-cap recall (0.935) with 6.0 instead of 18.45 candidates/S1.

Earlier rank-cap blocking (version 2, per-list caps; kept for reference):

| Config (name/addr/comb caps, MAX_CAND) | mean cands/S1 | pair recall | val F0.5 |
|---|---|---|---|
| 5 / 10 / 15, 20 (v2 submission) | 18.45 | 0.9354 | 0.9512 |
| 5 / 10 / 10, 15 | 14.29 | 0.9259 | 0.9478 |
| 5 / 5 / 10, 12 | 11.44 | 0.9151 | 0.9451 |
| 3 / 5 / 10, 12 | 11.16 | 0.9172 | 0.9458 |
| 5 / 5 / 5, 10 | 8.84 | 0.8850 | 0.9396 |
| 3 / 5 / 5, 8 | 7.32 | 0.8675 | 0.9363 |

Probe recall per list at depth 20 (1% India / 0.7% US probes): name 0.53 / 0.65, address 0.83 / 0.84,
combined 0.87 / 0.95, union 0.93 / 0.97. Remaining misses (India probe, bucketed): 16% empty pool address,
10% identical but very common name (median 64 same-name records) with a truncated address, the rest native-script
transliterations and heavily truncated addresses.

- **How we ensured true matches were not lost:** three complementary deep lists (a junk-name record is still found by
  address; an empty-address record by name), phonetic skeletons for transliterations, and a learned pruner whose
  threshold is set on validation to keep 99.5% of the wide net's true pairs, plus unconditional top-1 per S1.

---

## 4. Matching Model

**Features used (per candidate pair):**
- Name: rapidfuzz ratio, partial_ratio, token_sort, token_set, Jaro-Winkler on the legal-form-stripped name; ratio
  without spaces; token_set on the full normalized name; token Jaccard; first-token equality; acronym match;
  containment; consonant-skeleton token_set; name lengths/token counts; legal form equal / conflicting / one missing;
  hashed TF-IDF name cosine.
- Address: token_set, partial, token_sort; token_set without landmark phrases; token Jaccard; house-number Jaccard,
  first-number equality, conflict, fuzzy number agreement (prefix/suffix/1-edit); postcode equal / conflict;
  address length; hashed TF-IDF address cosine.
- Other/context: combined cosine; rank in the name/address/combined retrieval lists; the 13 stage-1 blocking
  features (union-level rank/gap/size, number of lists); rank and gap to the best
  candidate of the same S1 (combined/name/address); number of candidates of the S1; candidate source (S2/S3);
  frequency of the candidate's name and of the S1's name in the country pool (common vs rare name);
  frequency of the candidate's exact address. **No country feature, no id-derived feature.**

**Model type:** LightGBM binary classifier (MIT), lr 0.08, 63 leaves, min_child_samples 50, feature/bagging fraction
0.8, early stopping on 10% held-out S1 groups (best iteration 443), seed 42. Trained on candidates of 220,683 train
S1 (10% of train S1 per country × has-match stratum), blocked against the **full** per-country train pool.
Top features by gain: combined cosine, combined-list rank, number conflict, number Jaccard, address token_set,
name partial ratio, S1-level combined rank, name ratio, landmark-free address token_set.

**Pass-2 stacking (`stack.py`):** pass-1 probabilities for training rows are produced out-of-fold (5 folds grouped by
S1, same hyperparameters, 443 rounds); validation/test rows use the pass-1 model trained on the train split. Pass-2 adds
per-S1 context of the pass-1 probabilities (best, second best, gap to best, rank, counts above 0.5/0.9, sum, candidate
count) and sibling support (token_set name/address similarity of the candidate to the S1's top-3 other candidates with
p1 ≥ 0.3, the best of these, the pass-1 probability of the most similar sibling, and the number of siblings), on top of
all 55 pass-1 features. Pass-2 early-stopped at 96 rounds; its top features are p1, the gap to the S1's best,
the S1's best probability, and sibling-address support. Pass-2 with only the stacking features reached 0.9572 (+0.0014,
not shipped); with all features 0.9589 (+0.0031, shipped).

**Threshold selection method:** grid search on validation for macro F0.5 over probability threshold and a per-S1
relative rule (keep p ≥ REL·max p of that S1), with one-owner assignment (each S2/S3 record is kept only for the S1
with the highest probability, justified by EDA). Pass-1 alone: threshold 0.68, REL 0.0. Final (pass-2): threshold 0.44, REL 0.7.

---

## 5. Results & Error Analysis

Validation = 88,272 train S1 (4% per country × has-match, disjoint from training S1), scored with the exact macro F0.5.

- **F_0.5 Score (macro):** **0.9589** (precision 0.9877, recall 0.9175, singleton accuracy 0.9353)
  - by country: US 0.9699, India 0.9424
  - by true match count: 0 → 0.935, 1 → 0.886, 2 → 0.948, 3+ → 0.969
  - progression: v1 rank-cap blocking + 28 fuzz features 0.9491 → v2 + frequency / fuzzy-number / skeleton
    features 0.9512 → v3 wide net + learned stage-1 blocker 0.9558 → **v5 pass-2 stacking 0.9589**
  - split thresholds for an S1's first vs additional matches were tried (0.9513 vs 0.9512) and not adopted.
- **Loss decomposition (v5, 0.9589):** missed by blocking 0.0164 (India 0.0104), matcher false negatives 0.0136,
  matcher false positives 0.0075, singleton false positives 0.0036.
- **Loss decomposition (v2, 0.9512):** missed by blocking 0.0231, matcher false negatives 0.0148, matcher false
  positives on matched entities 0.0073, false positives on singletons 0.0036 (points of macro F0.5). Blocking was the
  largest bucket, which motivated v3: its oracle ceiling rose from 0.9769 to 0.9836.
- **Leave-one-country-out (proxy for unseen France; measured on the v2 pipeline):** train US → eval India 0.8072
  (in-domain 0.9322); train India → eval US 0.9413 (in-domain 0.9638). India's native-script names are the hard direction; France is
  Latin-script like US, and the final model is trained on both countries.
- **Test sanity (France):** predicted matches per S1 France 3.20 (empty 5.9%) vs US 3.41 (5.7%) and India 3.26 (6.2%).
  France receives more candidates (36.1/S1 vs 19.6 US, 22.4 India) because the stage-1 blocker is less confident on it.
- **Common false positives (wrong merges):** different businesses with similar names at the same or a neighbouring
  address (e.g. `shah ladder` vs `shah legal` in the same building; `ocean city village` vs `ocean city`), and exact
  name matches where the candidate has an empty address. Because validation uses a subsample of S1, the true owner of
  such a record is often absent from validation, so one-owner assignment cannot remove it there; on test, all S1 compete.
- **Common false negatives (missed matches):** candidate address empty and name shared by many businesses; house
  number with a dropped digit (`341` vs `41`); native-script transliterations (`blue marketing` vs
  `pulluu maarkketttting`); blocking misses (4.6% of true pairs never become candidates, down from 6.5% in v2).

---

## 6. Conclusion
Country-partitioned, multi-list sparse retrieval followed by a learned first-stage blocker gives 0.954 pair recall at
21 candidates per S1 on 10M-record pools within 8 GB RAM, and a LightGBM matcher with a one-owner, F0.5-tuned decision
rule with pass-2 sibling-support stacking reaches 0.959 macro F0.5 on validation with 0.988 precision. A learned blocker beats hand-set rank caps clearly:
at equal recall it needs 6 instead of 18 candidates per S1. The main lessons: address carries most of the signal (junk and transliterated
names are common), precision-first decisions are cheap (a small recall loss), and the transliterated native-script
names are the main source of both blocking and matching errors.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run_pipeline.py` — entry point: `python src/run_pipeline.py --data-dir <DATA_DIR> --out-dir ./output --work-dir ./work --mode full`
  (writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`; `--mode val` prints validation metrics only).
- `src/config.py` (all parameters, seed 42), `src/io_utils.py`, `src/normalize.py`, `src/blocking.py`,
  `src/features.py`, `src/model.py`, `src/decide.py`, `src/evaluate.py`.
- `README.md` (setup, run, runtime), `requirements.txt` (pinned: lightgbm 4.7.0, numpy 2.2.6, pandas 2.3.3,
  pyarrow 25.0.1, RapidFuzz 3.14.5, scikit-learn 1.7.2, scipy 1.15.3, Unidecode 1.4.0; Python 3.10.5).

### B. Additional Results
- Runtime (Windows 11 laptop, 8 cores, 7.8 GB RAM, CPU only): normalization ~15 min for 22M records. v3: train wide-net
  blocking US 330 s / India 242 s, stage-1 training + threshold 219 s; test blocking with inline pruning India 628 s
  (809,986 S1 vs 4,717,565 pool), US 438 s, France 175 s; test features ~135–225 s per 150k S1; peak process RSS 3.85 GB.
- Reproducibility: the v2 pipeline was re-run end to end from an empty work directory with the README command
  (6,793 s = 113 min, exit 0) and its outputs were byte-identical to the v2 submission. The v3 run that produced the
  submitted files reused only the cached normalized parquet files (deterministic, seed 42) and recomputed blocking,
  stage-1, features, model and decision from scratch; a second clean end-to-end v3 run was not performed.
- **Compliance:** no external data, APIs, geocoding or downloaded lists; all dictionaries hand-written; no pretrained
  models; LightGBM (MIT). TF-IDF document frequencies are fitted unsupervised on the records being blocked (per country,
  S1 + pool of the same split). No test labels exist or were used; thresholds were tuned on validation only.
- **Subsampling disclosure:** because of the 7.8 GB RAM limit, training and validation use a stratified subsample of
  train S1 (220,683 training + 88,272 validation of 2,206,821), always blocked against the **full** per-country train
  S2+S3 pool so the distractor density is realistic. Test is processed in full (all 1,732,544 S1).
