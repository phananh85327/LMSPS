# Review part 4, second half: does the search rank path sets as HAN does?

**Scope:** the 60 Fleet v3 records against the HAN repo's runs on the rebuilt data
(`output/AMAZON/v3`, HAN review H8): 24 sets at seed 1 for HAN and metapath2vec, and the main
comparison at seeds 1-3.

**The question, from the first half:** if the search is a good proxy for HAN, the LMSPS sets
should beat M5 and M6.

**Answer: no.**
- **Against the expert sets:** the LMSPS sets lose on the protocol's metric (Hit@10 against 99
  sampled negatives): HAN -0.029 (4 paths) and -0.021 (6 paths), metapath2vec -0.040 and -0.049,
  all beyond seed noise. On full ranking HAN ties.
- **Against chance:** of the 8 sets per size drawn from the search's own 36 candidates (the 3
  LMSPS sets and 5 random draws), HAN puts the LMSPS sets last, on test Hit@10 and on validation
  BPR, at both sizes. On full ranking, and in metapath2vec, they land between 2nd and 7th.
- **The search and HAN agree on the paths through users and disagree on the category paths.** HAN
  drops most of the near-empty category paths the search prefers, and uses the dense category paths
  the search ranks near the bottom.
- **No bug explains it:** the search reproduces exactly, its runs agree beyond chance (part 4),
  and the HAN side passed H1-H8. It is a finding about the search as a proxy on this data.

## Verdicts

| # | Item | Evidence | Verdict |
|---|---|---|---|
| B1 | LMSPS against the expert sets | 3 seeds: HAN Hit@10 0.635 vs M5 0.664 (p 0.0003), 0.666 vs M6 0.687 (p 0.008); metapath2vec 0.600 vs 0.640, 0.589 vs 0.638. Full ranking, HAN: 0.0207 vs 0.0211, 0.0245 vs 0.0246 (p 0.32, 0.87) | Finding: the searched sets do not improve on the expert sets |
| B2 | LMSPS against random draws from its own candidates | HAN Hit@10 and validation BPR: LMSPS 6th, 7th and 8th of 8 at both sizes. Full ranking: 3rd, 4th, 7th (4 paths), 5th-7th (6 paths). metapath2vec: 2nd, 3rd, 5th and 5th-7th | Finding: the search does no better than chance for HAN, and worst on the metric fixed in advance |
| B3 | Does it depend on which LMSPS set? | `a1`, `a2` and `_c` all land at the bottom for HAN | No |
| B4 | Set level: the search's preference against HAN's result | Mean alpha rank of a set's paths against HAN Hit@10, the 16 candidate sets: Spearman +0.54 (p 0.035; validation BPR -0.61, p 0.017): the more the search likes a set's paths, the worse HAN does. Among the 10 random draws alone +0.28 (p 0.43) | The inverse relation rests on the LMSPS sets themselves; beyond them the search's preference says little about HAN |
| B5 | Path level | Table below | Agree on users, disagree on categories |
| B6 | A bug on either side? | Search: deterministic, no defect in parts 1-3, 9 and the full re-check. HAN: H1-H8; its validation agrees with its test (Spearman -0.94 over 24 sets); the main comparison holds in metapath2vec too | No |

## Path level

HAN's path weights (beta, seed 1) as a multiple of an equal share, against the search's mean
alpha rank (1 = most preferred of 36, mean of the 4- and 6-path fleets):

| Kind of path | Paths | Search rank | Uses in HAN's 24 sets | Dropped by HAN (beta < 0.01) | Median HAN weight |
|---|---|---|---|---|---|
| Through users | 12 | 12.7 | 46 | 0 | 1.34 |
| Category paths that reach nothing for 76-81% of products | 6 | 11.8 | 25 | 16 | 0.00 |
| Other category paths | 10 | 23.8 | 22 | 3 | 1.12 |
| Product-product only | 4 | 28.8 | 16 | 2 | 0.85 |

- **Agree:** `PUPPUP` and `PUPUP`, the search's two favourites, and `PUP` (its fourth) are never
  dropped and get 1.3-1.8 times an equal share.
- **Disagree:** the near-empty category paths (`PCCP`, `PCCPP`, `PPCCPP`, ...) rank 10-14 in the
  search; in every LMSPS set HAN keeps one of them and drops the rest. `PPCCPP` is dropped in all 6
  sets that have it. The expert sets' `PCP`, `PCCCP` and `PPP`, ranked 22-29 and never in a final
  pool, get about an equal share; `PCPCP` (rank 25-28) nearly twice that.
- **Only the LMSPS sets** (among the 4- and 6-path sets) **have no category path that reaches
  (almost) every product,** and at both sizes every set with one beats every set without.

## Why the search may disagree (hypotheses)

- **A zero feature is a signal in the search, not in HAN.** In the search, a product with no
  `P-C-C-P` endpoint gets a zero propagated feature. Products with an endpoint are mostly DVDs and
  videos, and more popular (27.7% of training positives, 18.9% of sampled negatives: part 4), so
  zero against non-zero tells the supernet the group and part of the popularity. In HAN every
  channel keeps the product's own features (self-loops), so that cue is mostly gone and the
  extra near-empty channels are redundant.
- **Mean aggregation against attention over 20 neighbours.** The search averages a path's features
  over all its endpoints; a `PCP` average spans a whole leaf category. HAN's graphs keep 20
  neighbours per product and attend over them, which may make the dense category paths useful
  where an average is not.

The HAN repo's optional checks (H8) test the HAN side of this: where the LMSPS sets lose (Hit@10
by product group), and whether a dense category path closes the gap (a run with `PCP` in place of
`PPCCPP`).

## For the thesis

- On this task and data the searched meta-paths do not carry over to HAN or metapath2vec: the
  searched sets lose to the expert sets on the metric fixed in advance and do no better than
  random draws from the search's own candidates. If the thesis sets this against the LMSPS paper's
  own claims about using its meta-paths in other models, quote the paper's wording.
- Report the comparison with the random draws as their spread, not one draw, on test numbers.
- Present the reasons above as hypotheses unless the optional checks confirm them.

## What changed

Not committed yet:
- `review/part4b_search_vs_han.md`: this note.
- `review/evidence/part4b_search_vs_han.py`: every number above.
- `review/part4_fleet_v3_results.md`: the scope line points here.

## How to verify

From the repo root, with the HAN repo next to it:

    python review/evidence/part4b_search_vs_han.py "hgb/logs/Fleet v3" ../HAN-improve

Standard library only; it reads the HAN repo's `arch.py`, `output/AMAZON/v3/results.csv` and the
seed-1 HAN logs.
