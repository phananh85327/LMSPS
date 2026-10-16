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

Data: C# preprocessing (Preprocessor -> AmazonGraphFilter) -> `dat_generator.py
--input <Filtered_Output> --output <data/AMAZON> --verify` -> HGB-format
`data/AMAZON`. Node types 0=P(roduct) 1=C(ategory) 2=U(ser).

### Search stage (meta-path discovery)

```bash
python train_search.py --dataset AMAZON --task link_prediction --target-relation "U,P" \
    --num-hops 5 --ns 5 --num_final 20 --repeat 60 --split-seed 0 --tgt-ended-paths --half-feats --stage 30
```

Notes:
- `--stage` IS the search budget -- there is deliberately NO early stopping here.
  The epoch index drives two annealing schedules: progressive shrinking
  (`eps = 1 - epoch/(stage-1)`, the candidate pool only reaches `--num_final` at
  the last epoch) and the Gumbel temperature (`tau_max -> tau_min`). Stopping
  early leaves the search unfinished; the discovered arch comes from the final
  alphas + `--repeat` re-evaluations after the last epoch, not from a best
  checkpoint.
- `--num_final` must be smaller than the candidate count (55 target-ended paths
  at 5 hops) or shrinking is inert.
- **Reproducible search (recommended, all opt-in).** The 60 same-seed runs in
  `logs/Top K 4|6` were bit-identical for 33-60 epochs, then drifted by 1e-4 in a
  loss (GPU float nondeterminism) and ended in 16/30 resp. 26/30 different path
  sets -- although their final top-10 pools agreed on >= 8 of 10 paths. The
  spread came from the LAST step: every set was scored by ONE Gumbel draw
  (same-set re-scores varied by up to 0.02, set-to-set gaps were 0.001-0.005),
  and 200 uniform draws over 210 subsets re-scored duplicates. Use
  ```bash
  python train_search.py --dataset AMAZON --task link_prediction --target-relation "U,P" \
      --num-hops 5 --ns 5 --num_final 10 --tgt-ended-paths --half-feats --stage 150 \
      --split-seed 0 --seeds 1 --deterministic --eval-no-gumbel --eval-exhaustive-cap 1000
  ```
  `--deterministic` = cuBLAS workspace + `torch.use_deterministic_algorithms`;
  `--eval-no-gumbel` scores each set with the noise-free path strength
  softmax(alpha/tau) (one weighting for all node batches); `--eval-exhaustive-cap`
  scores every subset of the final pool once when C(num_final, ns) is small
  (C(10,5) = 252 ~ the cost of `--repeat 200`). `--num_final 10` is the paper's
  V = 2M for M = 5. With `--num_final 20 --repeat 60`, 60 draws cover 0.4 % of
  the 15,504 possible 5-sets -- the selection is then essentially random.
  Every search now writes `output/AMAZON/search_<time>_seed<s>.json` (args,
  candidates, alpha ranking, final pool, every scored set, chosen paths);
  `analyze_search_fleet.py` accepts these JSON files as well as console logs.
- FREEZE THE ARCH: paste the printed path list into `arch.py` once and never
  casually re-run the search -- GPU nondeterminism means even the same seed can
  discover a different set, which invalidates every downstream result built on
  the previous arch. Search-seed robustness is a separate (optional) analysis:
  report path overlap across search runs, not performance tables per search seed.

### Retraining stage (the LMSPS table row)

```bash
python main_path.py --dataset AMAZON --task link_prediction --target-relation "U,P" \
    --num-hops 5 --arch amazon_lp_han --split-seed 0 --half-feats --patience 20 --seeds 1 2 3
```

- `--split-seed` fixes the edge split + eval negatives (same exam for every
  model); `--seeds` are RUN seeds -- the script loops internally and prints one
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
4. **Alpha update data.** Alpha is trained on an internal half of the TRAIN
   edges, not on the real validation set (paper Eq. 5). The real validation
   edges are used only to pick the final set in `project_op_lp`.
5. **Inherited from the authors' code** (not this fork): no 20-epoch lambda
   warm-up, candidate pool ranked by raw alpha rather than Gumbel strength,
   sampling evaluation draws uniformly from the top-V instead of
   strength-weighted, single linear projection per path instead of a 2-layer MLP.

**Split-seed rule:** search and retrain MUST use the same `--split-seed`. The
final path set is chosen by validation BPR on the search split; retraining on a
different split can put those validation edges into the test set. Record the
split seed, stage, ns and num_final next to every `arch.py` entry.

Full review with file/line references: `chat/2026-10-16_18_lmsps-paper-fidelity-review.md`.