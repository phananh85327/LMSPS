# Review part 10: synthesis of both repos

**Scope:** every review note in this repo (`review/part1-9`, `part4b`, `recheck_full_repo.md`;
part 6 was dropped when `main_path.py` left the flow) and in the HAN repo (`review/h1-h8`,
`wrap_up_before_runs.md`, the paper check in H3), with the final results: the Fleet v3 search and
the HAN repo's rerun on the rebuilt data (2026-10-27 to 2026-10-30).

**Bottom line**
- **Both repos are sound for the thesis's experiment.** Every must-fix found is fixed and
  committed. The last passes (this repo's full re-check, H8) found no bug.
- **Every reported number comes from the rebuilt data.** The data behind all earlier results was
  contaminated (duplicate editions, a placeholder customer), which inflated them.
- **The answer to the thesis question:** on Amazon user-to-product link prediction, the meta-path
  sets LMSPS finds do not improve HAN or metapath2vec over expert-chosen sets. On the metric fixed
  in advance they are clearly worse; on full ranking HAN ties. They also do no better than random
  draws from the search's own candidates. A type-blind GAT beats both meta-path models.
- **All review parts are done.** What is left is optional: two cheap checks of the explanation
  (H8) and further work.

## 1. What each part found and fixed

| Part | Main finding | What was fixed |
|---|---|---|
| **LMSPS 1** data | 73% of validation and test edges had a copy of the same review in training (duplicate editions); one placeholder "customer" had reviewed 61% of products; products were linked to every ancestor category; weak title features | **Must fix:** the data rebuilt: editions merged, placeholders removed, leaf-only category links, LSA title features |
| **LMSPS 2** split, leakage | The split is clean. Training edges sit inside their own user-path features, which biased alpha against user paths | **Must fix:** alpha learns on validation edges (`--alpha-edges val`) |
| **LMSPS 3** search vs paper | A faithful port of the paper and the authors' code. The same self-inclusion biased the weight step; the supernet overfits | Fold-by-fold weight features (`--wstep-kfold 4`); `--dropout 0.5` |
| **LMSPS 4, 4b** Fleet v3 | 60 searches, deterministic, agreeing beyond chance. Their sets: `PUPUP`, `PUPPUP`, one more user path, and one or three category paths that reach nothing for 76-81% of products. 4b: the search does not predict HAN | **Must fix (docs):** hand over two one-SE sets and the consensus per size; HAN's validation, not its test, picks a set |
| **LMSPS 5** selection | The old "1-SE rule" was a vote that creates agreement by construction | **Must fix (reporting):** averaged scores with a real one-SE rule, the vote's null printed |
| **LMSPS 7** reproducibility | A search record did not say which data it ran on | Records carry the data build, node counts and a training-edge fingerprint |
| **LMSPS 8** docs, rerun | Rerunning the Fleet v2 script would have overwritten its records; re-selection printed only the vote's sets | `run_fleet_v3.bat`; `arch.py` entries for the averaged sets; five README claims corrected |
| **LMSPS 9** hand-over | The search's features had self-loops on P-P and C-C, which no meta-path model uses | **Must fix:** plain relations |
| **LMSPS re-check** | Every file in the flow read again against the Fleet v3 logs; all 62 records consistent | Compare searched sets with others on test numbers only |
| **HAN H1** data, split | The split matches the search's. Nothing knew which data a graph or a result came from; this repo could silently run on the old data | **Must fix:** fingerprints in every graph and result, a `meta.json` guard, `check_handover.py`, the table keeps data versions apart |
| **HAN H2** graphs | No must-fix. For long paths the 20 kept neighbours are a sample of an almost full neighbourhood; ties barely matter | Documented |
| **HAN H3** model | Correct (gradcheck, equals DGL's GAT, the paper's eq. 7-9), but **the order of the paths decided which path HAN used** | **Must fix:** every path's GAT starts from the same weights. Warm-up 15 kept: without it attention collapses onto one path. The paper check found no bug |
| **HAN H4** baselines | GAT was scored through the very edge it predicted; no popularity floor | **Must fix:** GAT removes each step's target edges; popularity row added |
| **HAN H5** reporting | Runs with different settings were averaged as seeds; sampled metrics flatter popularity | **Must fix:** `variant` column; full-ranking columns |
| **HAN H6** protocol | The old `run_all.bat` could not run the rerun | **Must fix:** rewritten (random draws, seeds 2-3, popularity, checks); the epoch cap no longer splits the table (P11) |
| **HAN wrap-up** | `run_all.bat` missed the two `_c` sets; the run logs were git-ignored | **Must fix:** both |
| **HAN H8** results | The runs are valid; the results below | The table fix (P11) |

## 2. The data: why the old numbers were too high

- **Before the rebuild:** 90,659 products, 171,285 users, 2.6M user-product edges. 73% of the
  test edges had a copy of the same review in training. For 50% the copy came from an exact
  duplicate edition (an identical reviewer set), and HAN scored 0.93 Hit@10 on those against 0.78
  on the rest.
- **After:** 48,883 products, 37,411 categories, 53,569 users, 819,847 edges; 40 of 81,984 test
  edges keep a twin. Split seed 1, training-edge fingerprint `c5f7bccbf61c1b0f` in every search
  record, graph and result.
- **What that did to the numbers:** HAN 0.85-0.87 to 0.62-0.71 Hit@10, metapath2vec 0.78-0.82 to
  0.55-0.64, while GAT stayed at about 0.79 because its target-edge fix lifted it. metapath2vec's
  code did not change, so its drop is the data. Compare numbers only within one data version.

## 3. The results

**Main comparison** (test, 3 seeds, mean +/- std; HAN H8):

| | Hit@10 (99 sampled negatives; fixed in advance) | Full-ranking Hit@10 |
|---|---|---|
| HAN: LMSPS `v3_f4_a1` vs M5 | 0.635 vs 0.664 (p 0.0003) | 0.0207 vs 0.0211 (tie) |
| HAN: LMSPS `v3_f6_a1` vs M6 | 0.666 vs 0.687 (p 0.008) | 0.0245 vs 0.0246 (tie) |
| metapath2vec: `v3_f4_a1` vs M5 | 0.600 vs 0.640 (p 0.001) | 0.0279 vs 0.0272 (tie) |
| metapath2vec: `v3_f6_a1` vs M6 | 0.589 vs 0.638 (p < 0.0001) | 0.0263 vs 0.0314 (manual wins) |
| GAT | 0.787 +/- 0.004 | 0.048 +/- 0.005 |
| Popularity | 0.423 | 0.009 |

- **Against chance (part 4b):** among the 8 sets per size drawn from the search's own 36
  candidates, HAN ranks the three LMSPS sets last, on test and on validation. The expert sets do
  about as well as an average random draw.
- **Path level (part 4b):** the search and HAN agree that the paths through users matter. They
  disagree on category paths: HAN drops most of the near-empty ones the search prefers, and uses
  the dense ones the search ranks near the bottom (`PCP`, `PCCCP`, `PCPCP`).
- **The best models:** GAT on every metric. The best HAN set by validation is a random draw
  (`rand6_d2`, 0.706). HAN beats metapath2vec on the sampled metric in all 24 sets, metapath2vec
  wins on full ranking in 14.

## 4. What the thesis can and cannot claim

**Can claim**
- On this task and data, the searched meta-paths do not carry over to HAN or metapath2vec. That
  goes against the paper's Table 4 (HAN retrained with searched paths gains on all four of its
  node-classification datasets), with the conditions below.
- The search's choice does no better than random draws from its own candidates for HAN.
- A type-blind GAT beats both meta-path models, most clearly for users with few purchases.
- The earlier numbers were inflated by duplicate editions; the rebuilt data removes that.

**Cannot claim**
- That LMSPS fails in general: one dataset, one task, one split. The paper's results are node
  classification.
- Why exactly the searched sets lose. H8 and part 4b give a likely reason (the LMSPS sets have no
  category path that reaches books and music, 75% of the test edges) and two checks for it.
- That HAN is a weak model in general: here its users are free vectors (27.4M of its 28M
  parameters), and the paper's tasks give every node a graph neighbourhood.
- Fine rankings among sets run at one seed: differences under about 0.01 Hit@10 are seed noise.

## 5. Threats to validity

| Threat | How it was handled |
|---|---|
| One random split (seed 1), not by date | Shared by every model; stated as a limitation (part 1 #10, part 2 S8) |
| Seeds vary only training | 3 seeds for the main comparison; the spread (0.002-0.005 Hit@10) is far below the gaps |
| Sampled metric flatters popularity | Popularity floor reported; full-ranking columns for every run |
| Same hyperparameters for every set | Fixed in advance, sized for a 12 GB card; the paper's own settings collapse HAN's attention (H3) |
| Early stopping per improvement of 0.0005 | Every run stopped early; the runs with most left are non-LMSPS sets and GAT, so more training could only widen the gaps (H8 R7) |
| The search chose its sets on the validation edges HAN also uses | LMSPS compared with other sets on test numbers only |
| Category paths empty for 81% of products | A consequence of leaf-only category links, chosen because all-ancestor links put 56.5% of products in one category (part 1). The category-path findings are tied to that choice |
| Self-inclusion in HAN's user-path graphs | Measured (H3); the fold fix did not help HAN, so it is documented, not changed. The search corrects for it, HAN does not |
| The search is smaller than the paper's (150 epochs, 4-6 sampled paths, pool 10) | Sized for the hardware; its runs agree beyond chance (part 4) |

## 6. Questions the council is likely to ask

1. **Why are your numbers lower than before?** The old test set was half copies of training data
   (section 2).
2. **Why does GAT beat HAN?** GAT builds each user from the user's own items; HAN and metapath2vec
   learn a free vector per user. GAT's lead is largest for users with 1-5 purchases (0.756 against
   0.639 for the best HAN set). Lv et al. (KDD 2021) found well-tuned plain GNNs match or beat most
   heterogeneous GNNs on the HGB benchmark this data format comes from.
3. **Why do the LMSPS sets lose?** They are built for the search's own model, where a near-empty
   category path still carries a signal (a zero feature marks books and music, and less popular
   products). In HAN that channel adds little, and the sets lack the dense category paths HAN uses.
   A likely reason, tested by the H8 checks.
4. **Is the comparison fair?** Same data, split, negatives, evaluation code and settings for every
   set; selection by validation; LMSPS judged on test numbers; 3 seeds for the main comparison.
5. **Did the models train long enough?** Every run stopped early, far below its cap; one GAT run
   that reached its cap was rerun (H6 P11, H8 R7).
6. **Is the sampled metric misleading?** Partly: the full-ranking check changes LMSPS against the
   manual sets from a loss to a tie in HAN, and swaps HAN and metapath2vec. The main conclusion
   holds under both metrics: the searched sets are never significantly better.
7. **Can it be reproduced?** Deterministic search (pilot = repeat), fingerprints in every record,
   graph and result, every command in `run_fleet_v3.bat` and `run_all.bat`, logs committed.

## 7. Roadmap

**Optional, before writing up** (HAN H8, about 25 GPU minutes):
1. `review/evidence/h8_by_group.py` on the saved runs: does LMSPS lose mostly on books and music?
   Is HAN's top 10 more popular than metapath2vec's?
2. Two HAN runs with `PCP` in place of `PPCCPP` in `v3_f4_a1` and `v3_f6_a1`: does a dense
   category path close the gap?

**Optional, to strengthen the thesis:**
- HAN on one path at a time for the 14 paths of the `a1` and `_c` sets, M5 and M6 (2-3 GPU hours): the paper's
  single-path baseline, its Fig. 5 claim, and a path-level comparison with the search.
- GAT with `--keep-target-edges` (about 20 minutes): the before number of the H4 fix on the full
  data.

**Further work, beyond this thesis:**
- A user side for HAN built from interactions (user meta-paths), to test whether the gap to GAT is
  the free user vectors.
- A search whose proxy model aggregates like HAN (attention over a few neighbours, self-loops).
- Several splits, or a split by date; full-ranking selection; other datasets; the all-ancestor
  category variant.

## 8. Status

| Part | Status |
|---|---|
| LMSPS 1-5, 7-9, 4b, full re-check | Done |
| LMSPS 6 | Dropped (`main_path.py` is out of the flow) |
| HAN H1-H8, wrap-up, paper check | Done |
| LMSPS 10 (this note) | Done |

## What changed

Not committed yet:
- `review/part10_synthesis.md`: this note.
- `review/part4b_search_vs_han.md`: cites the paper's Table 4 (from part 3).
- HAN repo: `review/wrap_up_before_runs.md` (status) and the README pointer.
