# Review part 1: AMAZON data pipeline and graph construction

**Scope**
- `Preprocessor.cs` and `AmazonGraphFilter.cs`, now in `hgb/extraction_cs/`
- `hgb/dat_generator.py` and `hgb/data_loader.py`
- the AMAZON branch of `hgb/utils.py::load_dataset`, plus feature propagation

**Bottom line:** the data behind every AMAZON result so far was contaminated, mostly by the raw SNAP
data itself. 73% of val/test edges had a copy of the same interaction in train, and one fake "user"
had reviewed 61% of all products. Both are fixed here. The Fleet v2 searches have to be re-run on
the rebuilt data.

## How it was checked

1. **Code review** of the full chain, from `amazon-meta.txt` to the DGL graph.
2. **Full-data audit.** `hgb/audit_amazon_data.py` was run on the `Filtered_Output` behind the Fleet v2
   runs (90,659 products / 39,754 categories / 171,285 users / 2,612,766 user-product edges).
   - Summary: [`evidence/audit_before_fix.json`](evidence/audit_before_fix.json).
   - The 17,726 duplicate clusters it found: [`evidence/dup_clusters_before_fix.csv`](evidence/dup_clusters_before_fix.csv).
3. **Raw-data check on the first 15,001 products of `amazon-meta.txt`.**
   - The original C# was compiled and run unchanged (.NET 8, CsvHelper 33.1.0).
   - Its output was compared with an independent parser and audited.
   - This separates what comes from the dataset from what comes from the code.

## Verdicts

| # | Issue | Origin | Evidence | Verdict |
|---|---|---|---|---|
| 1 | Edition duplicates | **dataset** (filter had no cross-ASIN step) | See below | **Must fix: fixed** |
| 2 | Placeholder customers | **dataset** | See below | **Must fix: fixed** |
| 3 | Products linked to every ancestor category | code (`dat_generator`) | Median 13 categories per product vs 4 leaves; one category holds 56.5% of products; P-C-P walks 1.18·P² vs 0.035·P² for leaves only (33×) | Fixed: `--category-links leaf` is the default |
| 4 | Weak title features | code (500-token bag of words) | 27.2% of products had no title token in the vocabulary | Fixed: `--title-features lsa` (TF-IDF + SVD, 256 dims) is the default. On the 14,857 snippet products it gives a title signal to 96.4% vs 76.9% |
| 5 | Untraceable node names | code | Random user names (~400 combinations, regenerated every run); category names collide ("General") | Fixed: customer ID for users, "Name [ID]" for categories |
| 6 | Locale-dependent `decimal.TryParse` | code (C#) | On a vi-VN machine "4.5" parses as 45. Only AvgRating is affected, and nothing uses it | Fixed: invariant culture |
| 7 | Title parsing keeps only the last fragment when a title contains "title:" | code (C#) | 0 cases in the snippet | Fixed |
| 8 | Self-loops on PP and CC | design (inherited from ACM) | 88.2% of products have at least one also-bought neighbour | Kept; document |
| 9 | Helpfulness filter; ratings ≥ 4 only | design | Removes 1.35% of positive reviews; keeps 78.6% of reviews | Kept; document |
| 10 | Random edge split ignores review dates; also-bought is a 2006 snapshot | design | n/a | Limitation; document |
| 11 | Truncated review lists (`downloaded < total`) | dataset | 148 of 14,857 snippet products; 2,529 reviews | Ignore |
| 12 | Both C# constructors call `Directory.Delete(outDir)` | code (C#) | Pointing `outDir` at the input would delete it | Unchanged; operational caution |

### Issue 1: edition duplicates

Amazon shows one review on every edition or format of a work (hardcover/paperback, DVD/VHS, volumes
of a set). SNAP crawled every edition page, so the same review line appears verbatim under several
ASINs. `AmazonGraphFilter` deduplicated (customer, ASIN) pairs only, so the copies survived.

**Full data, before the fix**
- 17,726 clusters cover 44,591 products (49%).
- 55% of all edges are copies of an interaction already in the graph.
- **72.9% of val edges and 73.0% of test edges have a copy in train.**
- 68.7% of the raw review rows are shared with another product.
- The clusters are genuine:
  - 99.1% have the same review count on every member.
  - 85% share the first three title words.
  - The largest cluster has 24 members.

**Raw snippet:** "Final Fantasy – The Spirits Within" on DVD and VHS share 659 of 660 review lines.

**The fix:** products sharing ≥ 3 identical review lines (customer, date, rating, votes, helpful)
that cover ≥ 50% of the smaller product are merged into the member with the most review lines.
This happens **before** the k-core.

### Issue 2: placeholder customers

Two customer IDs hold the anonymous reviews:
- `ATVPDKIKX0DER`, Amazon's own seller ID: 1995–2004, up to **741 reviews on one product**.
- `A3UN6WX5RRO2AG`: 2000–2005, up to 579 reviews on one product.

Real heavy reviewers never exceed 3 reviews on one product (spammers reach 34). On the full data
the first ID alone had 55,572 edges, covering 61% of all products.

**The fix:** drop both IDs, plus any customer with ≥ 50 reviews on a single product.

Verified correct and unchanged:
- relation directions and the positional unpacking in `load_dataset`
- UP **and** PU both replaced by the train-only adjacency
- the PP/CC merge, which prevents the propagated-key collision
- explicit node counts
- filter order and k-core convergence
- users emitted from surviving reviews
- the `dat_generator` format contract

## Verification of the fix (15,001-product snippet)

**Equivalence checks**

| Comparison | Result |
|---|---|
| `amazon_extract.py --legacy` vs the original C# | all 7 filtered files byte-identical (user names aside), same 9 k-core iterations |
| fixed C# vs fixed `amazon_extract.py` | all 8 files byte-identical, edition map included |
| fixed C# `--legacy` vs the original C# | byte-identical (user names now IDs) |
| `dat_generator.py --title-features bow --category-links all` vs the old `dat_generator.py` | `link.dat`, label files and every node id, type and feature identical; only the name column differs |

**Before vs after on the snippet**

| | Original extraction | Fixed extraction |
|---|---|---|
| Products / users / edges after the 5-core | 828 / 1,169 / 10,822 | 569 / 879 / 7,121 |
| Largest user degree | 605 (`ATVPDKIKX0DER`, 73% of products) | 61 |
| Duplicate clusters (audit) | 41 | **0** |
| Val / test edges with a train copy | 15.1% / 16.0% | **0% / 0%** |

The snippet is a prefix, so most editions of a work fall outside it. That is why its leak (15%) is
lower than on the full data (73%).

**Regression test:** `python hgb/tests/test_amazon_extract.py`.

## Consequences

- **Do not report or reuse any Fleet v2 output.** This covers:
  - `hgb/logs/Fleet v2`
  - the 1-SE shortlists (`arch.py` keys `amazon_lp_f4_*` and `amazon_lp_f6_*`)
  - the README result tables
  - any retraining or HAN numbers built on them
- **The headline finding has to be re-established.** That finding is "PPUPUP, PUPPUP and PUPUP are
  the core relations". Co-review paths are exactly the best copy detectors: two editions have nearly
  the same reviewers.
- **Order of work:** finish the code reviews of parts 2 (split/leakage) and 3 (search fidelity)
  before re-running the fleets, so that every fix goes into one rerun.

## Rebuilding the data (from `hgb/`)

```
python amazon_extract.py --input <amazon-meta.txt> --output ..\data\AMAZON_filtered
python dat_generator.py --input ..\data\AMAZON_filtered --output ..\data\AMAZON --verify
python audit_amazon_data.py --filtered ..\data\AMAZON_filtered --out ..\data\AMAZON\audit.json
```

**Acceptance:** audit section 2 shows no hub user, section 3 finds 0 duplicate clusters, and
section 4 shows about 0% leakage.

**C# route:**
```
dotnet run -c Release --project extraction_cs -- <amazon-meta.txt> <raw_dir> <filtered_dir>
```
It writes identical files.

**Reproducing the old data:** `amazon_extract.py --legacy` plus
`dat_generator.py --title-features bow --category-links all`.

**Expected size, first order:**
- Merging editions leaves about 63.8k products and 1.18M unique user-product edges, before
  dropping the placeholders and re-running the 5-core (2.61M edges before the fix).
- Measured on the snippet, extraction takes about 2 minutes and 2–4 GB of RAM for the full 1 GB
  file.
