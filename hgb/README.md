# LMSPS on HGB datasets



## Search stage

```bash
python train_search.py  --dataset DBLP --num-hops 6 --residual --amp  --all_path 

python train_search.py  --dataset IMDB --num-hops 5 --amp   --all_path 

python train_search.py  --dataset ACM --num-hops 5 --amp   --all_path
```



## Training stage

After search stage, you can add you search result in `arch.py`, and change the `args.arch`.

You can also **directly run the following command** to get the results in our paper with our searched meta-paths.

```bash
seeds='1 2 3 4 5 6 7 8 9 10'

python main_path.py  --dataset DBLP  --num-hops 6  --residual --amp --seeds $seeds --arch dblp 

python main_path.py  --dataset IMDB  --num-hops 5  --n-layers-2 4 --amp --seeds $seeds  --arch imdb   

python main_path.py  --dataset ACM  --num-hops 5   --amp --seeds $seeds --arch acm  --dropout 0.55

```

## AMAZON link prediction (thesis pipeline)

This fork adds a link-prediction task (`--task link_prediction`) for user-product
recommendation on the Amazon-Meta dataset: BPR loss, sampled-negative ranking
evaluation (99 negatives per test edge), single-tower design (meta-path search on
the product side, `--user-tower embedding` on the user side).

> **Data rebuilt 2026-10-26: the Fleet v2 results below are from the pre-fix data.**
> The original extraction kept two problems of the raw SNAP data:
> - **Edition duplicates.** One review appears on every edition of a work, so 73% of val/test
>   edges had a copy in train.
> - **Placeholder customers.** `ATVPDKIKX0DER` alone reviewed 61% of all products.
>
> Every number in "Fleet v2 final results", the `amazon_lp_f4_*` / `amazon_lp_f6_*` shortlists
> and anything trained on them must be redone on the rebuilt data.
> Details and evidence: `../review/part1_data_pipeline.md`.
>
> **Search changed 2026-10-26:** alpha now trains on the validation edges (`--alpha-edges val`,
> paper Eq. 5). Fleet v2 trained it on training edges whose features contain the edge itself
> (deviation 4 below; `../review/part2_split_negatives_leakage.md`). The weight step now reads
> each training edge's features from a graph without it (`--wstep-kfold 4`, deviation 8;
> `../review/part3_search_fidelity.md`). P-P and C-C no longer get self-loops, so a path
> means what a meta-path model will build from it (`--same-type-loops` restores them,
> deviation 9; `../review/part9_han_evaluation.md`). Reproducing Fleet v2 needs the old
> data, `--alpha-edges train-half`, `--wstep-kfold 0` and `--same-type-loops`.
>
> **The rerun is `run_fleet_v3.bat`** (Fleet v3, `logs/Fleet v3`): the Fleet v2 protocol on the
> rebuilt data with the current defaults plus `--dropout 0.5` (review part 3). See "Run flow".

Data: `amazon-meta.txt` -> `amazon_extract.py` -> `dat_generator.py` -> HGB-format `data/AMAZON`.
Node types 0=P(roduct) 1=C(ategory) 2=U(ser).

```bash
python amazon_extract.py --input <amazon-meta.txt> --output ../data/AMAZON_filtered
python dat_generator.py --input ../data/AMAZON_filtered --output ../data/AMAZON --verify
python audit_amazon_data.py --filtered ../data/AMAZON_filtered --out ../data/AMAZON/audit.json
python tests/test_amazon_extract.py
```

What each step does:
- **`amazon_extract.py`** ports the C# extraction (`extraction_cs/`, Preprocessor -> AmazonGraphFilter).
  Both write identical files. Products are merged per work (editions sharing >= 3 identical review
  lines covering >= 50% of the smaller one) and placeholder customers are dropped, both before the
  5-core. The edition map goes to `filtered_edition_map.csv`; per-phase counts go to
  `extraction_report.json`.
- **`dat_generator.py`** now defaults to:
  - leaf-only product->category links (`--category-links leaf`)
  - TF-IDF + SVD title features (`--title-features lsa --title-dim 256`)
  - IDs as node names

  It records its settings in `data/AMAZON/meta.json`.
- **The audit** must show no hub user (section 2), 0 duplicate clusters (section 3) and about 0%
  leakage (section 4).
- **Reproducing the old data:** `amazon_extract.py --legacy` plus
  `dat_generator.py --title-features bow --category-links all`.

### Search stage (meta-path discovery)

Final protocol (2026-10-18, plus the 2026-10-26 fixes): split-seed 1, stage 150, 36
reverse-deduplicated candidates, 15 warm-up epochs, 4 paths per step, exhaustive noise-free
selection, deterministic kernels, dropout 0.5. Thirty seeds per `--ns`, then the fleet-level
re-selection (`run_fleet_v3.bat` runs all of it).

```bash
python train_search.py --dataset AMAZON --task link_prediction --target-relation "U,P" \
    --num-hops 5 --ns 4 --num_final 10 --split-seed 1 --tgt-ended-paths --dedupe-reversed \
    --warmup-epochs 15 --dropout 0.5 --half-feats --stage 150 --deterministic --eval-no-gumbel \
    --eval-exhaustive-cap 1000 --seeds 1
python reselect_fleet.py "logs/Fleet v3/final_ns4/search_*.json" --delta 0.010 --prefix amazon_lp_v3_f4
```

`run_fleet_v3.bat final4 | final6 | reselect` runs the whole thing (30 seeds each,
~1 h per run; logs and JSON records in `logs/Fleet v3/final_ns4|final_ns6`).
`run_fleet_v2.bat` keeps the Fleet v2 commands. Both scripts activate `lmsps_env`
themselves; `execute.bat` is only a wrapper for `run_fleet_v2.bat`.

**Run flow for the rerun (Fleet v3, 2026-10-26)**

| step | command (from `hgb/`) | time | produces |
|---|---|---|---|
| 0 | move the old `../data/AMAZON` aside, rebuild it (see "Data" above) and check `audit.json` | minutes | the data every later step uses (`dat_generator.py` overwrites its output folder) |
| 0b | `run_fleet_v3.bat pilot` (optional) | ~2 h | `logs/Fleet v3/pilot`: same seed twice, must print `IDENTICAL` |
| 1 | `run_fleet_v3.bat final4` | 30 x ~1 h | `logs/Fleet v3/final_ns4/search_ns4_seed<s>.txt` + `.json` (a run whose `.json` exists is skipped) |
| 2 | `run_fleet_v3.bat final6` | 30 x ~1 h | same for ns 6 |
| 3 | `run_fleet_v3.bat reselect` | seconds | per ns, saved as `reselect_d010.txt/.json` (and `_d005`): averaged scores and their one-SE sets, the vote with its shuffled-score null, path pick rates, and `arch.py` entries |
| 4 | paste into `arch.py` | - | the `amazon_lp_v3_f<ns>_a<i>` entries (averaged one-SE sets, simplest first) with their provenance line; the vote's `_c` / `_s<i>` entries only for comparison |
| 5 | downstream, all with `--split-seed 1` | per arch | HAN repo: the `_a<i>` sets plus the manual sets (`../review/manual_metapath_sets.md`), on the same rebuilt data with graphs rebuilt from it (review part 9). `main_path.py` (the LMSPS model's own numbers) is optional |
| 6 | report | - | the averaged ranking and its one-SE sets, score noise and rank agreement, the duplicate rate next to its null, pick rates (review part 5) |

`run_fleet_v3.bat all` runs steps 1-3 and commits and pushes `logs/Fleet v3` after each fleet
and after the re-selection (`run_fleet_v3.bat push` does it by hand). Fleet v2's own flow is in
`run_fleet_v2.bat` and its header.

Nothing from steps 1-2 is wasted if the rule changes: step 3 only reads the saved
tables and can be re-run with any `--delta`.

**Why the per-run argmin cannot be the unit of comparison.** Per run, the 210
exhaustively scored 4-sets contain 12-78 sets within 0.010 val BPR of the best
(4-33 within 0.005), and the seed alone moves a set's score by ~0.006 (ns 4; 0.005
for ns 6; review part 5, from the sets two runs share). Re-scoring one run's winner
in another run's table costs ~0.009, more, because a winner also carries the
winner's curse. Inside that tie
group a run has no seed-independent information: every self-contained rule we
replayed on the 20 saved tables (argmin; plateau centre at delta 0.005-0.020;
parsimony; loss + lambda * hops) leaves 7-10 distinct sets out of 10. Averaging
over seeds is the only thing that works, and doing it inside a run (K
sub-trainings per run) was simulated at <= 39 % agreement for K = 5, i.e. 5 h per
run. So each run now prints and records its own **tie group** (`--report-delta`,
default 0.010: every set within delta of its best) before any fleet step, and the
fleet result is a count over those declared ties:

- `reselect_fleet.py` first prints the **membership table**: in how many of the N
  runs a given set belongs to the tie group. On the first 10 old-schedule runs,
  `PPUPUP PUPCP PUPUP PUPUPP` and `PPUPUP PUPCCP PUPUP PUPUPP` are tied-best in
  7/10 runs (each is also the raw argmin of one run), while the other eight raw
  argmin sets are tied-best in at most 4/10. That is a count, not a re-pick.
- It then prints the **1-SE table**: per run, among the sets within delta of its
  best, the one whose paths were selected most often by the *other* runs
  (leave-one-out, so no run votes for itself), ties by loss. Old-schedule runs:
  10 distinct -> 3 (x7). This is a leave-one-out vote, not Breiman's rule (which
  picks the simplest model within one SE of the best), and it pulls the runs
  together by construction: a null fleet with every run's scores shuffled already
  reaches 37% duplicates for both ns, against the observed 50% (ns 4) and 37%
  (ns 6). So the duplicate rate is not evidence of stability (review part 5).
- delta is fixed in advance to the measured cross-seed noise (0.010) and the
  0.005 line is reported as sensitivity. Choosing delta after seeing which value
  maximises duplicates would be the thing to avoid.
- Report both tables; HAN trains the consensus plus every 1-SE set seen >= 2 times.
- **Since review part 5:** two runs rank their shared sets alike (Spearman ~0.7), so
  scores can be averaged. `reselect_fleet.py` also prints **averaged scores** (all
  runs pooled, run effect + set effect) with a real one-SE rule: every set within
  one SE of the best, fewest total hops first. Out of sample (split halves of fleet
  v2) it picked sets at least as good as the vote. For the rerun, report that
  ranking, give HAN its one-SE sets, read the duplicate rate only against its
  printed null, and read path counts as pick rates when available.

**Fleet v2 final results (2026-10-22; `logs/Fleet v2/final_ns4|final_ns6`, 30 seeds each,
identical config, no warnings). PRE-FIX DATA: superseded, see the 2026-10-26 note at the top
of this section.**

| | ns 4 | ns 6 |
|---|---|---|
| raw argmin: distinct sets | 29 / 30 | 30 / 30 |
| 1-SE delta 0.010: distinct (duplicate rate) | 15 (50 %) | 19 (37 %) |
| 1-SE delta 0.005 (sensitivity) | 19 (37 %) | 25 (17 %) |
| most frequent 1-SE set | PPPUPP PPUPUP PUPPUP PUPUP x5 | PCPPUP PPPPUP PPPUPP PPUPUP PUPPUP PUPUP x5 |
| core paths in the 1-SE sets | PPUPUP 30, PUPPUP 27, PUPUP 23 | PUPPUP 30, PUPUP 30, PPUPUP 29 |
| tie-group size, median of 210 | 23.5 | 46 |
| cross-seed noise (median) | 0.0094 | 0.0059 |

Reading: the three "co-reviewed" relations {PPUPUP, PUPPUP, PUPUP} are the finding;
every further slot is a filler whose fleet-mean marginal effect is ~0 or positive
(PPPUPP +0.003, PPPPUP +0.005, PUP +0.013 = hurts). The duplicate rate is lower
than the 10-run old-schedule replay suggested because tie groups are larger with 36
candidates and the filler slot is split by pool availability (PCCPUP, the best
filler by marginal -0.007, is in only 9/30 pools). A vote by fleet-mean marginal
effect instead of argmin frequency was replayed and is worse (rewards rarely
available C-paths), so the pre-registered rule stands. For ns 6 the assembled
path-frequency consensus was itself picked by only 2 runs; the most frequent 1-SE
set (x5) is the better single representative. HAN shortlist = `arch.py` keys
`amazon_lp_f4_c, _s2.._s6` and `amazon_lp_f6_s1, _s2, _s3, _c, _s5.._s7` (the full
delta-0.010 shortlists, 13 sets; run in count order, the first three per ns cover
the consensus and the strong fillers), all with `--split-seed 1`.
`reselect_fleet.py` prints this shortlist directly (`--min-count`, default 2);
`--prefix` is only the key-name stem.

**Why delta 0.010 (not 0.005) is the final rule.** It was fixed on 2026-10-18,
before the final fleets ran, from the measured cross-seed noise of the old group
(0.008-0.009); the final ns-4 fleet measured 0.0094, confirming it. The ns-6 fleet
measured 0.0059, but from only 7 evaluable pairs (the 6-sets rarely share a pool),
so that estimate is weak, and switching delta per ns after seeing the results would
be exactly the post-hoc tuning the rule is meant to exclude. Substantively the
choice changes fillers: the three core paths are in every ns-6 shortlisted set,
but not for ns 4, where two of the six delta-0.010 sets lack PUPUP and four of the
six delta-0.005 sets lack PUPUP or PUPPUP (review part 5). The delta-0.010 shortlists contain 3 of the
6 (ns 4) and 3 of the 4 (ns 6) delta-0.005 sets, including the top ones. The
delta-0.005 files (`reselect_d005.json`) are the sensitivity line in the report.

**Why `--ns-search` is not in the final protocol.** Sampling 8 paths per step while
selecting 4 trains the shared weights on 8-path mixtures and evaluates 4-subsets
off-distribution (softmax weights 1/8 -> 1/4, dot-product scale doubles): the
cross-seed noise of a set's score rose from 0.008 to 0.018 and the 1-SE rule stopped
working (9 distinct / 10). With warm-up (and 36 instead of 55 candidates), a path eliminated
in Fleet v2 had been sampled in 12 epochs on average for ns 4 and 18 for ns 6 (medians 9.5 and
14), against about 8 before.

Notes:
- `--stage` IS the search budget -- there is deliberately NO early stopping here.
  The epoch index drives two annealing schedules: progressive shrinking
  (`eps = 1 - epoch/(stage-1)`, the candidate pool only reaches `--num_final` at
  the last epoch) and the Gumbel temperature (`tau_max -> tau_min`). Stopping
  early leaves the search unfinished; the discovered arch comes from the final
  alphas + `--repeat` re-evaluations after the last epoch, not from a best
  checkpoint. Pick `--stage` as a budget: in the first fleet the validation BPR was
  still falling at epoch 149 (means 0.130 / 0.125 for ns 4 / 6), so 150 is a floor;
  the paper uses 200 (convergence between 100 and 180 in its Table 11).
- `--num_final` must be smaller than the candidate count (55 target-ended paths
  at 5 hops, 36 with `--dedupe-reversed`) or shrinking is inert.
- **Reproducible search (recommended, all opt-in).** The 60 same-seed runs in
  `logs/Top K 4|6` were bit-identical for 33-60 epochs, then drifted by 1e-4 in a
  loss (GPU float nondeterminism) and ended in 16/30 resp. 26/30 different path
  sets -- although their final top-10 pools agreed on >= 8 of 10 paths. The
  spread came from the LAST step: every set was scored by ONE Gumbel draw
  (same-set re-scores varied by up to 0.02, set-to-set gaps were 0.001-0.005),
  and 200 uniform draws over 210 subsets re-scored duplicates.
  `--deterministic` = cuBLAS workspace + `torch.use_deterministic_algorithms`
  (same seed -> same run; vary `--seeds` for the stability test, same-seed runs
  are copies); `--eval-no-gumbel` scores each set with the noise-free path
  strength softmax(alpha/tau) (one weighting for all node batches);
  `--eval-exhaustive-cap` scores every subset of the final pool once when
  C(num_final, ns) is small (C(10,4) = C(10,6) = 210, C(10,5) = 252 ~ the cost of
  `--repeat 200`). `--num_final 10` is the paper's V = 2M only for M = 5 (2.5M for
  M = 4, 1.7M for M = 6). With
  `--num_final 20 --repeat 60`, 60 draws cover 0.4 % of the 15,504 possible
  5-sets -- the selection is then essentially random.
  Every search writes a JSON record via `--search-json` (default
  `output/AMAZON/search_<time>_seed<s>.json`: args, candidates, alpha ranking,
  final pool, every scored set, chosen paths); `analyze_search_fleet.py` accepts
  these JSON files as well as console logs.
- FREEZE THE ARCH: paste the printed path list into `arch.py` once and never
  casually re-run the search -- GPU nondeterminism means even the same seed can
  discover a different set, which invalidates every downstream result built on
  the previous arch. Search-seed robustness is a separate (optional) analysis:
  report path overlap across search runs, not performance tables per search seed.

### Retraining stage (the LMSPS table row)

Optional: the HAN flow does not use it. It trains the LMSPS model on a found set, for the
LMSPS model's own numbers and `lmsps_emb.pt`.

```bash
python main_path.py --dataset AMAZON --task link_prediction --target-relation "U,P" \
    --num-hops 5 --arch amazon_lp_han --split-seed 1 --half-feats --patience 20 --seeds 1 2 3
```

- `--split-seed` fixes the edge split + eval negatives (same exam for every
  model) and MUST equal the split seed of the search that produced the arch
  (the arch was chosen by validation BPR on that split; current protocol: 1);
  `--seeds` are RUN seeds -- the script loops internally and prints one
  metrics dict per seed; report mean +/- std.
- Early stopping: best-val checkpoint always kept; only improvements
  > `--min-delta` (default 5e-4) reset patience.
- Test output reports per-edge metrics, per-user pooled metrics, and a
  breakdown by user training degree.
- Embeddings persist to `output/AMAZON/lmsps_emb.pt` (last seed) and are served
  or re-evaluated by the standalone HAN repo's `recommend.py` / `eval_emb.py`.

Downstream consumers of the discovered paths (HAN, metapath2vec, GAT baselines,
serving, offline re-evaluation) live in the standalone HAN repo
(`HAN-improve` / `han-recsys`) -- see its README and METAPATH_CONTRACT.md.

### Reproducibility and tests

- **What a search record holds.** Besides the flags, the alpha ranking and every scored
  set, each `--search-json` record stores the data it ran on (`data`: `meta.json` from
  `dat_generator.py`, node counts, and a fingerprint of the training edges, which changes
  with the data and with the split) and the environment (`env`: Python, torch, dgl, numpy,
  device). `reselect_fleet.py` prints the fingerprint in its provenance line and warns when
  one fleet mixes data versions, e.g. seeds run before and after a data rebuild.
- **Same seed, same run.** With `--deterministic`, two runs of one seed must be identical.
  `run_fleet_v3.bat pilot` checks it on your machine (two full runs of seed 1, compared by
  `compare_search_records.py`); on the CPU it holds for the current code (review part 7).
- **Tests** need the training environment (`requirements.txt`) and run as scripts or with
  pytest (`python -m pytest hgb/tests`):
  - `tests/test_amazon_extract.py`: the extraction and `dat_generator.py`
  - `tests/test_link_split.py`: split, negatives, leakage guards, fold features, plain
    relations, data provenance
  - `tests/test_reselect_fleet.py`: fleet statistics, `arch.py` entries, the pilot check
    (standard library only)
- `sparse_tools` is a git submodule (`Yangxc13/sparse_tools`, the commit the authors' README
  clones): `git submodule update --init sparse_tools`, then build it as `requirements.txt` says.
  Only node classification and `ogbn/` need it; the link-prediction flow stubs it out.
- References to `chat/` notes and "chat log" numbers point to the author's local notes; that
  folder is git-ignored.

### Deviations from the paper (Li et al., NeurIPS 2024)

The search (Eq. 1-5), sampling evaluation (Eq. 6) and target-net (Eq. 7) are the
authors' code; BPR over dot products replaces the cross-entropy head because the
paper covers node classification only. What differs, and must be stated when
the results are reported as "LMSPS":

1. **Search space.** `--tgt-ended-paths` keeps only P...P walks of length >= 2
   (55 candidates at 5 hops) so every discovered path is a HAN-usable adjacency.
   The paper searches ALL target-related meta-paths, including the 0-hop raw
   feature and paths ending at other types (125 keys here). Category and user
   raw features therefore never enter the model.
2. **M and V.** The paper uses M = 30 selected paths and V = 2M = 60 for every
   dataset, 200 search epochs. Here M = `--ns` (4-6), V = `--num_final` (10-20),
   `--stage` 30-150. The paper's own rule (App. C.1) would skip progressive
   sampling at K = 55 <= 2M = 60; the small M is what makes shrinking matter.
3. **Single tower.** Only products are meta-path encoded; users are a free
   embedding table. Call it "LMSPS product encoder + MF-style user vectors".
4. **Alpha update data (`--alpha-edges`, changed 2026-10-26).** The default `val`
   follows paper Eq. 5: weights on all train edges, alpha on the validation
   edges, which also pick the final set in `project_op_lp` (upstream's
   `project_op` likewise scores the half that trained alpha). Fleet v2 trained
   alpha on an internal half of the TRAIN edges (`train-half`, kept for
   reproduction only). Those edges are also message-passing edges, so each one's
   user-path features contain the edge itself, and alpha learned to reward paths
   for recognising edges already in the graph (`../review/part2_split_negatives_leakage.md`).
   `heldout-half` keeps the internal split but removes the alpha half from the graph.
5. **Inherited from the authors' code** (not this fork): no 20-epoch lambda
   warm-up (now available as `--warmup-epochs`, off by default), candidate pool
   ranked by raw alpha rather than Gumbel strength, sampling evaluation draws
   uniformly from the top-V instead of strength-weighted, single linear
   projection per path instead of a 2-layer MLP, initialised uniform(-0.5, 0.5)
   instead of Xavier (about 6x wider for these matrix sizes; review part 3).
6. **`--ns-search` (opt-in).** The paper samples the same M paths per step that
   it selects at the end. With M = 4 of K = 55 every eliminated path got only
   ~8 looks (2-4 for early cuts) in fleet v2, so `--ns-search S` samples S paths
   per step while still selecting `--ns`. Using S != ns is a departure; say so.
7. **`--dedupe-reversed` (recommended for AMAZON).** Reverse(X) has the transpose
   of X's walk-count matrix, i.e. the same relation, and HAN symmetrises anyway.
   Without the flag, 5 of the first 10 fleet-v2 runs spent 2 of their 4 slots on
   PPUPUP + PUPUPP. The flag keeps one member per reverse pair (55 -> 36
   candidates: 17 palindromes + 19 representatives). The paper has no such step
   because its target-net concatenates all M paths and never reuses them as
   graphs; state it as a search-space choice.
8. **`--wstep-kfold K` (link prediction only, default 4).** A training edge is also
   a message-passing edge, so its product's user-path features contain the edge
   itself. With alpha on the validation edges (deviation 4), the weight step still
   learned to rely on that self-signal, and user paths then looked worse on
   validation than they are. The training edges are split into K folds, features
   are propagated once per fold without it, and the weight step reads each edge's
   own fold; alpha and the final scoring keep the full training graph. Node
   classification has no counterpart: labels are never part of the graph. Costs K
   extra propagations and K copies of the user-path candidates in CPU RAM.
9. **Plain same-type relations (changed 2026-10-26).** Upstream adds self-loops to
   ACM's P-P adjacency, and Fleet v1/v2 did the same for AMAZON's P-P and C-C. A
   self-loop lets a step stay put, so the search's `PPUP` also contained `PUP`: a
   product without an also-bought partner (11.8% of them on the old data) got its
   own `PUP` feature as `PPUP`, and on a small graph `PPUPUP` and `PUPPUP` came out
   identical to `PUPUP`. Meta-path models build a path from the plain relations (HAN's
   meta-path neighbours, metapath2vec's walks), where `PPUP` is only "reviewed together
   with an also-bought product"; with leaf-only categories `PCCP` is then nearly empty.
   The default now matches them, and the paper's Eq. 1 has no self-loops either;
   `--same-type-loops` is for reproducing Fleet v2.

**Search stability (fleet v2, seeds 21-30, ns 4, 2026-10-18; old schedule: 55 candidates
without `--dedupe-reversed`, no warm-up, so it includes PUPUPP and PUPCCP).** Same seed ->
identical run. Different seeds -> different 4-sets, but PPUPUP in 9/10,
PUPUPP and PUPCCP 6/10, PUPUP 5/10 (consensus PPUPUP PUPUPP PUPCCP PUPUP),
pairwise overlap 1.8/4. Causes: redundant candidates (2-13 sets tie within
0.002 val BPR per run), few looks per path before elimination, one-shot weight
sharing. Report path frequencies + consensus + downstream equivalence, not the
exact set. Details: chat log #18, Addendum 4.

**Split-seed rule:** the search and every model trained on its paths (HAN, metapath2vec,
the retrain) MUST use the same `--split-seed`. The final path set is chosen by validation BPR
on the search split; training on a different split can put those validation edges into the
test set. `reselect_fleet.py` prints a provenance line (split seed, stage, ns, num_final and
the search flags, read from the runs) above its `arch.py` entries: paste it with them.

Full review with file/line references: `chat/2026-10-16_18_lmsps-paper-fidelity-review.md`.