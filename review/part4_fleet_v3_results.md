# Review part 4, first half: the Fleet v3 results

**Scope**
- the 60 Fleet v3 searches (`hgb/logs/Fleet v3/final_ns4|final_ns6`, seeds 1-30), the two pilot
  runs, and the re-selection (`reselect_d010` / `_d005`, `reselect run.txt`)
- judged against the rules fixed before the rerun (parts 5, 8, 9) and against Fleet v2
- **Not yet:** whether the search's ranking of path sets predicts HAN's. That needs the HAN
  runs and is the second half of this part.

**Bottom line:** the fleet is valid and can go to HAN. Nothing in the LMSPS code needs fixing.
Two things in the hand-over do:
1. **Don't give HAN every one-SE set.** The ns 6 window holds 32 sets, and all 32 are
   `PUPUP` + `PUPPUP` + one more user path + three category paths. Training all of them would
   train nearly the same model 32 times, for roughly 20 extra GPU hours with metapath2vec. Hand
   over the first two per ns (simplest first, as before), plus the vote's consensus.
2. **Part 5 said "HAN's test results pick the final set".** It is HAN's validation results; the
   test set only reports the choice. Corrected.

**What the fleet found**
- Almost every run's best set has `PUPUP` or `PUPPUP` (products two user hops away; 28 and 30 of
  30 runs), two or three paths through users in all, and one (ns 4) or three (ns 6) of the
  category paths below. After re-selection, `PUPUP` and `PUPPUP` are in 28 and 26 of the 30 picks
  (ns 4) and 29 and 29 (ns 6).
- Those category paths (`PCCP`, `PCCPP`, `PCCPPP`, `PCCPCP`, `PCCPUP`, `PPCCPP`) reach nothing
  for 76-81% of products: almost all books and music. For DVDs and videos (90% of them) they
  reach the products filed one category level up or down.
- The supernet wants this mix. In ns 6, the best set without any category path scores 0.047
  worse than the run's best; three category paths is the best. A fourth user path makes the
  validation loss worse.
- The paths of the expert sets (`PP`, `PCP`, `PPP`, `PCCCP`) never reach a final pool. That is a
  clear prediction for HAN, which the second half of this part tests.

**Better than Fleet v2?**
- **More trustworthy.** It runs on the rebuilt data, without the three search biases fixed in
  parts 2, 3 and 9. Every record carries its data fingerprint, and a run repeats exactly.
- **At least as stable.** Score noise is about the same. The vote's agreement beats chance in
  both fleets; in Fleet v2 only for ns 4.
- **Three times faster** per epoch.
- **Its scores can't be compared with Fleet v2's:** the data differ. Whether its sets are better
  for HAN, only HAN can tell.

## How it was checked

1. **Data and settings:** all 62 records (60 searches, 2 pilots) carry the training-edge
   fingerprint `c5f7bccbf61c1b0f` and identical arguments: split 1, 5 hops, dropout 0.5, alpha on
   validation edges, fold-by-fold weight steps, plain relations, the same 36 candidates. The
   candidates are also Fleet v2's, so the frozen random sets in the HAN repo still apply.
2. **Determinism:** `compare_search_records.py` prints IDENTICAL for `pilot_a` against `pilot_b`
   and for `pilot_a` against fleet seed 1.
3. **Run health:** every log has 150 epochs and no error or NaN. Best validation BPR 0.211-0.226
   (ns 4) and 0.203-0.220 (ns 6).
4. **Path coverage** on the rebuilt data, with the search's own split (the script checks the
   fingerprint): `evidence/part4_category_paths.py`.
5. **Path effects and what each run's best set contains:** `evidence/part4_fleet_paths.py`.
6. **Part 5's split-half test of the selection rules**, rerun on Fleet v3:
   `evidence/part4_split_half.py`.
7. The LMSPS tests still pass (`test_amazon_extract`, `test_link_split`, `test_reselect_fleet`).

## Verdicts

| # | Item | Evidence | Verdict |
|---|---|---|---|
| V1 | The runs used the data HAN will use, with one configuration | 62/62 records: same fingerprint, same arguments | OK |
| V2 | Same seed, same run | Pilot a = pilot b = fleet seed 1 | OK |
| V3 | Runs finished cleanly | 150 epochs each, no errors | OK |
| V4 | The runs agree beyond chance | Vote against its shuffled-score null: ns 4 at 3% (delta 0.010) and 0% (0.005); ns 6 at 11% and 1% | OK. Fleet v2's ns 6 never did (52% / 61%) |
| V5 | What they agree on | `PUPUP` and `PUPPUP` in 26-29 of 30 re-selected picks; one of `PPUPP` / `PPPUPP` / `PPUP`; one or three category paths | Finding (below) |
| V6 | The category paths reach nothing for 76-81% of products | Books 2%, music 5%, DVDs and videos 90% have an endpoint | Finding, not a bug. Report it. Run the sets unchanged |
| V7 | The averaged ranking can't separate its top sets | One-SE window: 2 sets (ns 4), 32 (ns 6); 333 / 169 sets compared at SE 0.0025-0.0030 | Report. `a1` stands for a plateau, not for one best set |
| V8 | Does the `a1` rule still pick well? | Split halves: within 0.003 of every other rule, below the score noise (0.0055-0.0068) | Keep `a1` as the main LMSPS set (fixed in the HAN protocol before any result) |
| V9 | ns 4's `a1`/`a2` have two category paths and no `PUPPUP` | 24 of 30 runs' best sets have one category path; `PUPPUP` is in 26 of 30 vote picks | Add the vote's consensus `_c` as a second LMSPS row, chosen now, before any HAN result |
| V10 | Hand-over: "paste the `_a<i>` entries", and `run_all.bat` trains every one | ns 6 has 32 | **Must fix (docs): fixed.** The first two `_a<i>` per ns plus `_c` |
| V11 | Part 5: "HAN's test results pick the final set" | Choosing on test inflates the number reported | **Must fix (docs): fixed.** Validation |
| V12 | The manual-sets note says to drop paths that leave more than half the products without neighbours | That rule was for designing the expert sets | Clarified: never applied to LMSPS sets |
| V13 | README run times | "~1 h per run" was Fleet v2's | Fixed: ~20 min |

## Fleet v3 against Fleet v2

| | v2 ns 4 | **v3 ns 4** | v2 ns 6 | **v3 ns 6** |
|---|---|---|---|---|
| Data | old | rebuilt | old | rebuilt |
| Seconds per epoch (median) | 19.7 | **6.9** | 22.9 | **8.1** |
| Score noise of one set | 0.0063 | 0.0055 | 0.0050 | 0.0068 |
| Rank agreement of two runs (Spearman) | +0.72 | +0.68 | +0.67 | +0.59 |
| Vote, delta 0.010: distinct picks / null median (null fleets as low) | 15 / 19 (4%) | 16 / 21 (**3%**) | 19 / 19 (52%) | 19 / 22 (**11%**) |
| Vote, delta 0.005 | 19 / 25 (0%) | 20 / 26 (**0%**) | 25 / 25 (61%) | 20 / 25 (**1%**) |
| Averaged one-SE window | 7 sets | 2 | 4 | 32 |
| Out of sample: gap of the averaged pick to the other half's best | 0.0107 | 0.0057 | 0.0065 | 0.0060 |

Fleet v2's numbers are from part 5. The last row is on different data, so it shows only that
Fleet v3's picks hold up in new runs at least as well as Fleet v2's did.

The pools changed in a way that fits the part 9 fix. In Fleet v2, `PPUPUP`, `PUPPUP` and
`PUPUP` were in (nearly) every pool, and `PPUPUP` had one of the two strongest effects; with the
old self-loops the three were close to one signal counted three times (part 9). In Fleet v3,
`PUPUP` and `PUPPUP` stay, while `PPUPUP` drops to 21 pools (ns 4) and 17 (ns 6) with an effect
near zero.

## What each run's best set contains

The best set among those with k category paths, as a gap to the run's best (median over runs):

| k category paths | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| ns 4 | 0.0077 | **0** | 0.0041 | 0.0485 | 0.394 | | |
| ns 6 | 0.0474 | 0.0219 | 0.0049 | **0** | 0.0082 | 0.0555 | 0.401 |

Runs' own best sets: ns 4, 24 with one category path and 6 with two. ns 6: 5 with two, 21 with
three, 4 with four. A set of category paths alone fails (0.39-0.40 worse).

Pick rates (re-selected sets, delta 0.010, among the runs whose pool held the path):
- ns 4: `PUPUP` 100%, `PUPPUP` 90%, `PPUPP` 100%, `PCCPP` 93%, `PPPUPP` 52%, `PCCPUP` 57%
- ns 6: `PUPUP`, `PUPPUP`, `PPCCPP`, `PCCPUP`, `PCCP` and `PPUPP` all 100%

Over all scored sets, the paths through users have the largest effects (`PUPUP` -0.010,
`PUPPUP` -0.009 in ns 4). A category path on average makes a set worse (+0.001 to +0.006),
because in most sets it takes the place of a needed user path. Next to two or three user paths
it is the best choice.

## The category paths

The rebuilt data, split 1, plain relations, training edges only:

| Path | No endpoint | Median endpoints when non-empty |
|---|---|---|
| `PCCP` | 81.1% | 128 |
| `PCCPP` | 81.2% | 470 |
| `PCCPPP` | 81.2% | 1,828 |
| `PCCPCP` | 81.1% | 12,324 |
| `PCCPUP` | 81.1% | 26,690 |
| `PPCCPP` | 76.5% | 1,346 |
| `PCCCCP` | 1.2% | 1,787 |
| `PP`, `PPP`, ... and every `PP`-start path | 11.3% | |
| every other path | 0% | |

- **Why:** the one C-C step of `PCCP` moves from a category the product is filed under to that
  category's parent or child. With leaf-only links (part 1), that neighbour holds products only
  where the catalogue files products one level up or down as well: DVD and video categories do,
  book and music categories almost never. Two C-C steps (`PCCCP`, sibling categories) reach
  almost every product.
- **Who has an endpoint:** 18.9% of products. By group: books 2.2%, music 5.3%, DVDs 90.1%,
  videos 90.2%.
- **They are more popular:** mean training degree 19.6 against 12.0. They receive 27.7% of the
  training positives and 27.6% of the validation positives, but only 18.9% of the sampled
  negatives. So the paths carry real information for DVDs and videos, plus a popularity hint.
- **In HAN:** these channels see only themselves (self-loops) for most books and music. Run the
  sets as selected. Removing their category paths would no longer test LMSPS. The rule in the
  manual-sets note ("drop a path that leaves more than half the products without neighbours")
  was for designing the expert sets, which use `PCCCP` (0% empty) instead.

## Selection: what `a1` is

- **The averaged ranking has a plateau.** ns 4 compares 333 sets seen in at least 5 runs, ns 6
  169, each with SE 0.0025-0.0030. The best of that many noisy averages is partly luck, so the
  one-SE window holds 2 sets for ns 4 and 32 for ns 6.
- **The 32 ns 6 sets differ only in the last slots:** all contain `PUPPUP`, `PUPUP` and three
  category paths; the sixth path is `PPUPP` (14 sets), `PPPUPP` (11), `PPUP` (5) or `PPUPUP` (2).
- **Out of sample** (200 random half splits, each rule picks from 15 runs, judged on the other
  15):

| Rule | ns 4: gap to the other half's best | ns 6 |
|---|---|---|
| `a1` (averaged, one-SE, simplest) | 0.0061 | 0.0047 |
| averaged, lowest | 0.0057 | 0.0060 |
| vote | 0.0061 | 0.0025 |
| one run's own winner | 0.0062 | 0.0077 |

Head to head, on the runs that can score both picks: `a1` beats the vote by 0.0009 in ns 4 and
loses by 0.0007 in ns 6; it beats one run's winner by 0.0006 and 0.0029. All differences are
below the score noise of one set. No rule is clearly better, so `a1` stays the main LMSPS set.

- **Why `_c` as well:** ns 4's `a1` and `a2` each have two category paths and lack `PUPPUP`,
  while most runs' best sets have one category path and `PUPPUP`. The vote's consensus
  (`PCCPP PPUPP PUPPUP PUPUP`) is that typical set; for ns 6 the consensus is itself in the
  window (`a9`). It is the same at both deltas. Adding it now, before any HAN result, keeps the
  choice of LMSPS rows independent of HAN's numbers.

## For HAN

Paste into the HAN repo's `arch.py`, with the `# ----` provenance lines from
`reselect_d010.txt`:

| Key | Paths |
|---|---|
| `amazon_lp_v3_f4_a1` (main comparison against M5) | `PCCPCP PPCCPP PPUPP PUPUP` |
| `amazon_lp_v3_f4_a2` | `PCCPPP PPCCPP PPPUPP PUPUP` |
| `amazon_lp_v3_f4_c` | `PCCPP PPUPP PUPPUP PUPUP` |
| `amazon_lp_v3_f6_a1` (main comparison against M6) | `PCCP PCCPP PPCCPP PPUP PUPPUP PUPUP` |
| `amazon_lp_v3_f6_a2` | `PCCP PCCPCP PPCCPP PPUP PUPPUP PUPUP` |
| `amazon_lp_v3_f6_c` | `PCCP PCCPUP PPCCPP PPUPP PUPPUP PUPUP` |

- The main comparison does not change: `a1` against M5 and M6, seeds 1-3.
- `run_all.bat` in the HAN repo trains the `_a<i>` keys only. The two `_c` keys need a one-line
  addition there, or two runs by hand.
- **What to watch in the second half:** the search ranks the expert sets' `PP`, `PCP`, `PPP`
  and `PCCCP` near the bottom of its 36 candidates (mean alpha rank 22-30; never in a final pool)
  and `PUP`, `PPUP` in the middle. If the search is a good proxy for HAN, the LMSPS sets should
  beat M5 and M6.

## What changed

Not committed yet:
- `review/part4_fleet_v3_results.md`: this note.
- `review/evidence/part4_category_paths.py`, `part4_fleet_paths.py`, `part4_split_half.py`: new.
- `review/part5_fleet_selection.md`: "test results" corrected to validation; the cap noted.
- `hgb/README.md`: run flow steps 1-2 (time per run) and 4-5 (which sets to hand over, unchanged).
- `hgb/run_fleet_v3.bat`: the same in the header comment (no command changed).

## How to verify

From the repo root, in `lmsps_env`:
```
python review/evidence/part4_category_paths.py data "hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"
python review/evidence/part4_fleet_paths.py "hgb/logs/Fleet v3/final_ns4/search_*.json"
python review/evidence/part4_fleet_paths.py "hgb/logs/Fleet v3/final_ns6/search_*.json"
python review/evidence/part4_split_half.py "hgb/logs/Fleet v3/final_ns6/search_*.json"
python hgb/compare_search_records.py "hgb/logs/Fleet v3/pilot/pilot_a.json" "hgb/logs/Fleet v3/pilot/pilot_b.json"
```
The split-half test takes several minutes per fleet.
