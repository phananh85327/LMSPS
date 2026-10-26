# Review part 5: fleet statistics and the 1-SE rule

**Scope**
- `hgb/reselect_fleet.py`: the fleet-level re-selection that turns 30 searches into the HAN
  shortlist
- the README's fleet results and the claims built on them
- the Fleet v2 logs (`logs/Fleet v2/final_ns4|final_ns6`, 30 runs each)

**Bottom line:** the script does what the README describes and reproduces the committed results
exactly. Three of the claims built on it do not hold:
1. **The "1-SE rule" is a vote that creates agreement by construction.** Replayed on a fleet whose
   scores are shuffled (no information in them at all), it already produces 37% duplicate picks.
   Fleet v2's 50% for ns 4 is only partly real; its 37% for ns 6 is entirely the rule.
2. **The noise estimate is inflated, and the "no information" premise is wrong.** "Cross-seed
   noise" (0.0094) re-scores a run's *winner*, so it includes the winner's curse; one set's score
   noise is 0.0063. Two runs rank their shared sets alike (Spearman about 0.7), so scores can be
   averaged across runs instead of voted on.
3. **Path counts mostly reflect pool availability.** The "core" trio was in all 30 pools, and in
   Fleet v2 alpha decided the pools with the bias found in part 2. The README's "core paths in
   every shortlisted set" is false for ns 4.

Out of sample, no rule beats one run's own winner by much. Averaging scores is slightly better
than the vote; tie-group membership is worst. `reselect_fleet.py` now also prints averaged scores
with a real one-SE rule, the duplicate rate's null, the true score noise and pick rates when
available. With the part 2/3 fixes, the runs on the small graph agree for real.

## How it was checked

1. **Reproduction:** all four committed `reselect_d0xx.json` files (ns 4 and 6, delta 0.010 and
   0.005) come out identical from the logs.
2. **Null test:** 300 fleets per setting in which every run keeps its own final pool and
   tie-group size, but its 210 scores are shuffled across its sets. The exact same rule is
   replayed.
3. **Noise and agreement:** from the sets two runs both scored (434 run pairs for ns 4, 118 for
   ns 6).
4. **Out-of-sample test:** 200 random splits of the 30 runs into halves. Each rule picks one set
   from half A; the pick is judged by its gap to the best in half B's runs that can score it.
5. **Averaged scores:** a two-way model, loss = run effect + set effect, fitted by alternating
   means. It matches exact least squares to 3.5e-12 over 2,037 sets.
6. **Tests:** `hgb/tests/test_reselect_fleet.py`, 4 tests on synthetic fleets with known answers
   plus a regression check on the committed files. Each was checked with a deliberate break.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| R1 | The "1-SE rule" is a leave-one-out vote, not Breiman's rule, and makes runs agree by construction | Shuffled-score null: 37% duplicates for both ns. Observed: 50% (ns 4, at the null's 5% tail) and 37% (ns 6, = the null) | **Must fix the reporting: fixed.** The duplicate rate is printed next to its null; README corrected |
| R2 | "Cross-seed noise" includes the winner's curse | ns 4: 0.0094 vs a set's own noise 0.0063. ns 6: 0.0059 from 7 run pairs vs 0.0050 from 118 | **Fixed:** the script prints the per-set noise. Keep delta 0.010 for the vote (pre-registered); the averaged ranking does not use it |
| R3 | README premise: a run has no seed-independent information inside its tie group | Two runs rank their shared sets with Spearman +0.72 (ns 4) / +0.67 (ns 6) | **Fixed:** averaged scores added |
| R4 | Path counts confounded by pool availability | PCCPUP picked in 9 of the 9 runs whose pool held it; PPPUPP in 13 of 22. The core trio was in 30/30 pools | **Fixed:** pick rates when available are printed |
| R5 | README: "the three core paths are in every shortlisted set under both deltas" | False for ns 4: 2 of 6 sets lack PUPUP at delta 0.010; 4 of 6 lack PUPUP or PUPPUP at 0.005. True for ns 6 | **Fixed** in the README |
| R6 | The vote's headline ns 4 set (`PPPUPP PPUPUP PUPPUP PUPUP`, picked 5×) | Averaged over 22 runs it scores 0.010 worse than the best sets (SE 0.0013), which swap PUPUP for a category path | Information: the Fleet v2 finding was pre-fix anyway; re-check on the rerun |
| R7 | Which rule picks best? | Out of sample: see the table below | Rank by averaged scores; HAN decides among its one-SE sets |
| R8 | The "consensus" is the most frequent paths assembled, not necessarily any run's pick | ns 6: picked by only 2 runs (the README notes it) | Ignore; superseded by the averaged one-SE sets |
| R9 | README: "a vote by fleet-mean marginal effect … is worse" | If judged by the duplicate rate, the comparison is circular: a vote maximises duplicates | Document; compare rules out of sample instead |
| R10 | Key naming in `arch.py` / README (D1) | — | Fixed in part 8 (aliases; `_a<i>` entries for the averaged sets) |
| R11 | The `*.json` glob also matches `reselect_d0xx.json` (D6) | Skipped with a warning | Fixed in part 8 (`run_fleet_v3.bat` reads `search_*.json`) |

### R1: what the duplicate rate measures

| | ns 4, δ 0.010 | ns 4, δ 0.005 | ns 6, δ 0.010 | ns 6, δ 0.005 |
|---|---|---|---|---|
| observed distinct picks (of 30) | 15 | 19 | 19 | 25 |
| shuffled-score null, median (90% range) | 19 (16–22) | 25 (22–28) | 19 (16–23) | 25 (22–28) |
| null fleets with as few distinct picks | 5% | 0% | 54% | 61% |

Each run picks, among its near-ties, the set made of the paths other runs picked most. When pools
overlap (7.5 of 10 paths shared on average in ns 4), that alone steers runs to the same sets. Only
duplicates *beyond* the null show that the runs' evaluations agree.

### R7: which rule picks sets that hold up elsewhere

Gap to the best set, in the half of the fleet the rule did not see (ns 4; ns 6 in brackets):

| Pick | Gap |
|---|---|
| the other half's own best (a floor, not achievable) | 0.0037 (0.0035) |
| **averaged scores** | **0.0107 (0.0065)** |
| one random run's own winner | 0.0101 (0.0073) |
| the vote's most frequent pick | 0.0115 (0.0084) |
| a random set from a random tie group | 0.0129 (0.0117) |
| tie-group membership | 0.0135 (0.0105) |

A pick is judged only on the other-half runs whose pools contain it, so the numbers above carry a
selection effect. A random run's winner, for example, could be scored in only 112 of 200 splits
for ns 4 and 20 for ns 6. Head to head on the runs that can score both picks:
- averaging beats the vote by 0.0011 (ns 4) and 0.0016 (ns 6);
- the vote loses to one random run's winner by 0.0028 (ns 4).

The fleet step's real value is averaging, which the vote throws away.

### With the part 2/3 fixes

The six combined-fix runs on the small graph (part 3) give 2 distinct vote picks, where the
shuffled-score null gives 5–6. So the runs genuinely agree. The averaged one-SE sets all pair
PCCPUP with three of PPPUPP, PPUPUP, PUPPUP, PUPUP. Six runs on 1,730 products is a hint; the
full rerun decides.

## For the rerun

1. Run `reselect_fleet.py` on each fleet as before, and report:
   - the averaged scores and their one-SE sets (simplest first);
   - the score noise and ranking agreement;
   - the duplicate rate *with* its null;
   - path pick rates when available.
2. Give HAN the one-SE sets, plus the manual sets (M1–M8). **HAN's test results pick the final
   set** (part 9). Supernet score differences of 0.001–0.003 are too small to decide on their own.
3. "Simplest first" counts total hops: shorter paths also make sparser HAN graphs (part 9).

## What changed

Committed in `a04ab60`:
- `hgb/reselect_fleet.py`:
  - new: `score_noise`, `averaged_scores`, `null_distinct`, `spearman`
  - new outputs: averaged scores with a one-SE rule, the per-set noise and agreement, the
    duplicate rate's null, pick rates when available, and the matching JSON fields
  - new flags `--min-runs`, `--null-draws`
  - the vote itself is unchanged: the committed Fleet v2 files are reproduced
- `hgb/tests/test_reselect_fleet.py`: new, 4 tests.
- `hgb/README.md`: the noise sentence, the "Breiman" label, the D2 claim, and the reporting rule
  for the rerun.

## How to verify

```
python hgb/tests/test_reselect_fleet.py
python hgb/reselect_fleet.py "hgb/logs/Fleet v2/final_ns4/search_*.json" --delta 0.010
```
