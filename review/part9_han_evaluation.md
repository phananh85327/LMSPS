# Review part 9: what the search hands over to HAN

**Scope**
- What LMSPS hands to the HAN repo (`phananh85327/HAN-improve`, or any meta-path model):
  - the path strings and what they mean
  - the split and evaluation negatives
  - the data
- How the HAN repo reads them, at `f69b997`.

**Not in scope yet:** the HAN repo itself. Its code is not yet reviewed or tested by its author,
so it is parked. Nothing there is changed, and what I noticed is listed at the end for later.

**Bottom line:** one must-fix on the LMSPS side.

**The search scored paths with self-loops, which no meta-path model uses.**
- `load_dataset` added self-loops to P–P and C–C (copied from upstream's ACM branch), so a P–P
  or C–C step could stay put. The paper's Eq. 1 has no self-loops. HAN's meta-path neighbours
  and metapath2vec's walks follow the plain relations.
- With the loops, a product with no also-bought partner (11.8% of products on the old data) got
  its own `PUP` feature as `PPUP`.
- On a small graph, `PPUPUP` and `PUPPUP` came out identical to `PUPUP` (median cosine 1.000).
  Fleet v2's "core trio" `PPUPUP`, `PUPPUP`, `PUPUP` was therefore partly one signal counted
  three times.
- Its paths with an inner P–P or C–C step are not the graphs a meta-path model builds from the
  same strings.

**Fixed:** plain relations by default; `--same-type-loops` reproduces Fleet v2.

The rest of the hand-over is sound:
- The HAN repo builds its split with the same `build_link_split` code (identical, diffed), so
  the train/val/test edges and the 99 negatives per edge match the search's for the same split
  seed.
- Its graphs, walks and union graph use training edges only.

Two conditions on the HAN side for later: it must use split seed 1 (its scripts default to 0),
and its graphs must be rebuilt from the new data (it finds graph files by name only).

A first check on the old data says the search's ranking of path sets is a weak predictor of
HAN's. The supernet's averaged set scores (part 5) and HAN's validation loss on the Fleet v2
shortlist sets agree with Spearman +0.54 (4-path sets, n = 6) and +0.75 (6-path sets, n = 7).
That is one HAN seed per set, from a repo not yet reviewed: a hint for part 4, not a result.

## How it was checked

1. **Code:**
   - `load_dataset`'s AMAZON branch, against upstream's ACM branch and the paper's Eq. 1.
   - The HAN repo's `link_data.py` and `data_loader.py`, diffed against LMSPS. Identical split
     and negatives; HAN only adds an edition-twin report, which draws no random numbers.
   - Its graph builder, walks and union graph, read for which edges they use.
2. **LMSPS's own feature propagation** on the k3 graph (1,730 products, split 1), with and
   without self-loops (`scratchpad/part9/loops_features.py`, which calls `utils.load_dataset`
   and `hg_propagate_feat_dgl`).
3. **The search end to end** on k3 with the new default: a 20-epoch smoke run. Its JSON records
   `same_type_loops: false`.
4. **Tests:** `test_amazon_same_type_loops` in `hgb/tests/test_link_split.py` (10 tests, all
   pass). Two deliberate breaks are both caught: loops always on, loops never on.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| P1 | The search's paths carried self-loops on P–P and C–C | See "Self-loops" below | **Must fix: fixed** (`--same-type-loops` for reproduction; README deviation 9) |
| P2 | Same split and negatives on both sides | `build_link_split` identical; HAN's graphs and walks use training edges only | OK |
| P3 | The downstream split seed | The search picks its set with split 1's validation edges; on another split some of those are test edges. LMSPS records split 1 in `arch.py` and the search JSON; the HAN scripts default to 0, though `run_all.bat` passes 1 | Hand-over rule: split 1. HAN-side fix parked |
| P4 | Old HAN graph files reused on the new data | HAN finds graphs by file name (`<path>_top20_split1.npz`), which doesn't change with the data | Parked (HAN side); rebuild the graph folder after the data rebuild |
| P5 | Does the search predict HAN? | Spearman +0.54 / +0.75 on the old data, one seed per set | Part 4 on the rerun |
| P6 | Two-user-hop features are nearly the same vector even without loops | k3: `PPUPUP` vs `PUPUP` median cosine 0.979, `PUPPUP` vs `PUPUP` 0.942. Mean aggregation over two user hops averages over most of the graph | Document: one reason so many sets tie (part 5). Not a bug |
| P7 | `main_path.py` reads the same `load_dataset` | It follows the new default and has no flag | Ignore (out of the flow) |

## Self-loops (P1)

With a self-loop, mean aggregation over P–P gives the product itself a weight of 1/(d+1), where d
is its number of also-bought partners. For d = 0 the "also-bought" path is just the shorter path.
On the old data that was 11.8% of products, and every P–P step of every path did it again. With
leaf-only categories, C–C works the same way: `PCCP` became `PCP` plus a little, while without
the loop it is nearly empty.

LMSPS's own features on k3, with and without the loops:

| Key | Products with no instance of the path (plain) | Median cosine, loops vs plain (where both exist) |
|---|---|---|
| `PPUP` | 87% | 0.988 |
| `PPUPUP` | 87% | 0.996 |
| `PUPPUP` | 12% | 0.946 |
| `PCCP` | 85% | 0.830 |
| `PCCPUP` | 85% | 0.910 |
| `PCCPP` | 91% | 0.658 |
| `PCCCP` | 0% | 0.995 |

The first column is the products whose feature the loops filled in with a shorter path's feature.
k3 keeps few also-bought pairs (13% of products have one, against 88% on the full data), so the
`PP`-rows are exaggerated here; the `CC`-rows are not, since the rebuilt data links products to
leaf categories only. Under the loops, `PPUPUP` and `PUPPUP` were the same vector as `PUPUP`
(median cosine 1.000).

**Why change the search rather than the models:**
- A meta-path is a composite relation: HAN's meta-path neighbours, metapath2vec's walks and the
  HAN repo's builder all use the plain relations.
- The paper's Eq. 1 has no self-loops.
- The fleets are being rerun anyway.

The cost: a product with no also-bought partner now has an all-zero `PP…` feature. That is
correct, since the path carries no information for it, and the search already handles all-zero
user-path features for products without training users.

## Parked: the HAN repo, for when you review it

Measured with that repo's code, so read these as leads rather than results. My changes are saved
as `scratchpad/deliver/han_improve_part9_parked.patch` (16 files, not applied or committed
anywhere).

1. **Split-seed default 0** in every script; split 1 needed.
2. **Graph files and result files are found by name only.** After the rebuild, old graphs would be
   reused, and `run_all.bat` would skip runs whose result file already exists.
3. **Self-inclusion in HAN training.** A training edge sits inside its product's neighbourhood:
   - k3: 13% of a training positive's `PUP` neighbours are the user's own items, against 3% for a
     test edge.
   - Graphs that leave out each edge's fold raised single-path test Hit@10 by 0.03–0.04 (3 seeds).
   - On a 4-path set they made HAN's path attention collapse onto one path in 2 of 3 seeds
     (Hit@10 0.27–0.28 against 0.42).
   - A control with the same fold-by-fold sampling but full graphs did not collapse, so the
     collapse comes from the fold graphs.
4. **Comparison design.**
   - The manual set has 5 paths, matching neither 4 nor 6. In the Fleet v2 HAN runs, 6-path sets
     beat 4-path sets by 0.012 Hit@10.
   - Its `PCCP` is nearly empty with leaf categories.
   - One random draw per size; no popularity row (popularity alone gives 0.41 Hit@10 on k3).
5. **Graph construction.** Pruning to 20 neighbours after every hop turns long paths into beams;
   symmetrisation makes popular products hubs.
6. **Docs:** 504-d features, split 0, and reference numbers from the duplicate-inflated data.

## What changed

Not committed (pending your approval):
- `hgb/utils.py`: P–P / C–C self-loops only with `--same-type-loops`.
- `hgb/train_search.py`: the flag.
- `hgb/tests/test_link_split.py`: `test_amazon_same_type_loops`; `_load_amazon` passes extra
  flags.
- `hgb/README.md`: the change note and deviation 9.
- `hgb/run_fleet_v2.bat`: reproducing Fleet v2 also needs `--same-type-loops`.
- This write-up.

## How to verify

```
python hgb/tests/test_link_split.py
```
