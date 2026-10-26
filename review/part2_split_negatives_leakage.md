# Review part 2: link split, negative sampling and leakage

**Scope**
- `hgb/link_data.py`: the edge split, evaluation negatives and the training sampler
- the link-prediction branches of `utils.load_dataset` and `train_search.py`
- what the search sees of the val/test edges

**Flow note (2026-10-26):** the discovered paths are only used to train HAN in the other repo, so
`main_path.py` (the LMSPS retrain) is not part of the flow and not reviewed. Questions about
training on the found paths move to part 9 (HAN).

**Bottom line:** the split is clean. No val or test edge reaches training, the propagated
features or the search, and new tests guard that. The real problem is subtler: every **training**
edge is also a message-passing edge, so its product's user paths (PUP, PPUP, PUPUP, ...) contain
the edge itself, and held-out edges never get that. The search trained alpha on such in-graph
edges (README deviation 4). On a small graph built from your data, that made alpha rank PUP 4th
of 36; taking alpha's edges out of the graph moved PUP to 20th and made alpha agree with
validation (Spearman +0.78 instead of +0.20). Fixed: alpha now learns on the validation edges
(`--alpha-edges val`, paper Eq. 5). The same effect on the search's shared weights goes to
part 3, and on HAN training to part 9.

## How it was checked

1. **Code trace** of the split, the samplers, the graph swap in `load_dataset` and the search's
   bi-level step.
2. **Tests:** `hgb/tests/test_link_split.py`, 7 tests. Each was checked by breaking the code on
   purpose; every break is caught by exactly the test written for it:

   | Deliberate break | Caught by |
   |---|---|
   | `load_dataset` swaps U-P but not P-U (val/test edges stay in the reverse direction) | `test_amazon_graph_uses_train_edges_only` |
   | evaluation negatives avoid only train positives | `test_eval_negatives_never_hit_a_positive` |
   | the training sampler also avoids val/test positives (tells the trainer what is held out) | `test_train_negatives_only_avoid_train_positives` |
   | `remove_target_edges` removes only one direction | `test_remove_target_edges` |
   | `remove_target_edges` passes int32 ids (a bug in the first draft of this fix) | `test_remove_target_edges` |

3. **Measurement on real data.** Mini graphs from the 15,001-product snippet, built with the fixed
   extraction and `dat_generator.py`, with the k-core lowered so they keep enough edges:
   - k5: 569 products / 879 users / 7,121 edges
   - k3: 1,730 / 3,542 / 19,002
   - k2: 3,635 / 10,395 / 35,314

   A scipy replica of the mean aggregation reproduces the real `load_dataset` +
   `hg_propagate_feat_dgl` features to 3.4e-7.
4. **Experiment:** the search with alpha trained three ways, 6 seeds each, on k3 (details below).

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| S1 | Is the split strict? | Disjoint, complete, deterministic 80/10/10 split; evaluation negatives avoid every positive; training negatives avoid only train positives; U-P **and** P-U swapped to train edges; test edges never touch the search; one `--split-seed`, which HAN must reuse (part 9) | **OK.** Tests added |
| S2a | Self-inclusion biases **alpha** | See below | **Must fix: fixed** (`--alpha-edges val`) |
| S2b | Self-inclusion in the **weights** (search, and HAN downstream) | Same mechanism: the model trains on features that contain each target edge and is tested on features that don't | Search: **fixed in part 3** (`--wstep-kfold 4`). HAN: parked with the HAN repo review (part 9) |
| S3 | Users the weight step never trains | The user tower is only in `optimizer_w`, so a user whose train edges all landed in the alpha half keeps its random init. Full pre-fix data: 3.1% of users. k3: 13% of users, 14–15% of alpha-step edges | **Fixed by S2a.** Left: cold-start users with no train edge (0.9% of k3 val edges); HAN's evaluation should report them separately (parked with the HAN repo review) |
| S4 | Uniform evaluation negatives are easy | Training popularity alone, no learning: test HR@10 0.316 (k5), 0.415 (k3), 0.498 (k2) vs 0.10 for random. Val positives have 1.8–6.1× the training degree of their negatives | HAN's results need a popularity baseline next to every table (parked with the HAN repo review, part 9) |
| S5 | `TrainEdgeSampler` draws with replacement | About 63% of edges per epoch, all edges over the run | Ignore |
| S6 | `sample_negs` loops forever if a user has fewer than `k` non-positive items | Needs `k > num_dst - degree`: impossible with 99 negatives and tens of thousands of products | Ignore |
| S7 | `--cpu` crashes | Unguarded `torch.cuda.synchronize()`, 4 calls in `train_search.py` | **Fixed.** (`main_path.py` has the same 4 calls; left as is, it is out of the flow) |
| S8 | Random split ignores review dates | Test edges can predate training edges; also-bought is a 2006 snapshot (part 1 #10) | Limitation; document |
| S9 | The 5-core runs before the split | A user's val/test edges help it pass the filter, so some users have < 5 train edges | Ignore; standard practice |
| S10 | Validation does several jobs after the fix | Alpha and the final selection both use val (and HAN's early stopping, if it uses val) | Accept: upstream does the same. Report test numbers only |

### S2: a training edge is inside its own features

`load_dataset` builds the message-passing graph from the training edges, which is correct for
val/test. But the same training edges are the BPR positives. For a training pair (u, p):
- `PUP(p)` = mean over p's users of (mean over each user's products of the title features).
  u is one of p's users, so p's feature contains u's own history.
- A val/test pair never gets this: the edge is not in the graph.

Measured by removing a sampled training edge and recomputing the features (300 edges per graph):

| Path | Share of the feature that is the edge itself (k5 / k3) | Pull toward the user's taste (k5 / k3) | Edges pulled toward the user |
|---|---|---|---|
| PUP | 5.0% / 4.2% | +0.039 / +0.050 | 100% |
| PPUP | 4.6% / 3.7% | +0.037 / +0.046 | 99% |
| PUPUP | 3.0% / 3.3% | +0.025 / +0.040 | 100% |
| PPUPUP | 2.9% / 2.8% | +0.024 / +0.037 | 99% |
| PCPUP | 2.3% / 1.2% | +0.021 / +0.018 | 99% |
| PPP, PCP | 0 | 0 | none |

"Pull" is cos(feature, user's other products) with the edge minus without it.

Two consequences:
- **Alpha learned on the wrong targets.** It was trained on in-graph train edges, so it rewarded
  paths for recognising an edge that is already in the graph. PUP-type paths do that best. The
  final set is then picked on val edges, which carry no such signal. This matches Fleet v2
  (ns 4, 30 runs, pre-fix data): PUP was alpha's #3–#8 path in every run, one-user-hop paths
  took 6.2 of the 10 pool slots, and 77% of them then made validation worse. 90% of the pooled
  two-user-hop paths made it better. The measured self-inclusion follows the same split: 3.6–5.0%
  of the feature for one-user-hop paths through products (PUP, PPUP, PPUPP), 2.8–3.3% for the
  two-user-hop paths.
- **Upstream guards against exactly this for labels.** SeHGNN/LMSPS build label features as
  `remove_diag(A) @ labels`, so a node's own label never flows back into its own features
  (`train_search.py:594-606`). In link prediction the labels are the U-P
  edges, they are in the graph, and nothing removes them. SEAL and NBFNet remove the target link
  during training for the same reason, and PyG's `RandomLinkSplit(disjoint_train_ratio=...)`
  exists for it.

### The experiment: alpha trained three ways

**Setup**
- Graph: mini k3.
- Flags: the Fleet v2 `final4` flags (`run_fleet_v2.bat` COMMON + FINAL: ns 4, 36 candidates,
  stage 150, warm-up 15, exhaustive evaluation of the 210 subsets), CPU, seeds 1–6 per variant.
- One change: `--batch-size 256` instead of 10,000.
  - With 10,000 the mini graph gets one step per epoch, 150 in total. Training BPR was still
    0.67 at epoch 40, so that run was stopped.
  - Full AMAZON gets about 100 steps per epoch. 256 gives about 30 on the mini graph, and the
    runs then overfit the way the full-data runs did.

**Variants**
- `train-half` (Fleet v2): weights on half of the train edges, alpha on the other half. Both
  halves are in the graph.
- `heldout-half`: the same halves, but the alpha half is removed from the graph. Same data and the
  same number of steps as `train-half`. The only difference is whether alpha's edges are in the
  graph, so this is the clean test of the mechanism.
- `val` (the fix): weights on all train edges, alpha on the val edges.

**Results** (one-user-hop = PUP family, two-user-hop = PUPUP family; "hurt" / "help" = the path's
marginal effect on the validation BPR of the scored sets)

| | Fleet v2 ns4, full pre-fix data (30 runs) | k3 `train-half` (6 runs) | k3 `heldout-half` (6 runs) | k3 `val` (6 runs) |
|---|---|---|---|---|
| alpha's top 4: one-user-hop paths (11 of 36 candidates) | 1.2 of 4 | 2.2 of 4 | 1.0 of 4 | 1.3 of 4 |
| PUP alpha rank (of 36) | 4.4 (3-8) | 4.2 (2-11) | 20.0 (8-29) | 19.2 (9-33) |
| final pool of 10: one-user-hop paths (11 candidates) | 6.2 | 6.5 | 2.0 | 4.2 |
| final pool of 10: two-user-hop paths (3 candidates) | 3.0 | 3.0 | 1.7 | 2.7 |
| pooled one-user-hop paths that hurt val | 142/185 (77%) | 27/39 (69%) | 6/12 (50%) | 18/25 (72%) |
| pooled two-user-hop paths that help val | 81/90 (90%) | 15/18 (83%) | 2/10 (20%) | 12/16 (75%) |
| selected 4: one- / two-user-hop paths | 1.6 / 1.9 | 1.2 / 2.3 | 0.7 / 0.8 | 1.0 / 1.8 |
| alpha's top 4 vs selected set (overlap) | 2.1 of 4 | 2.2 of 4 | 3.0 of 4 | 3.0 of 4 |
| Spearman(alpha, val marginal) over the pool | +0.51 | +0.20 | +0.78 | +0.82 |
| distinct selected sets | 29 of 30 | 6 of 6 | 5 of 6 | 6 of 6 |

**Reading**
- **The small graph reproduces Fleet v2.** Under `train-half`, PUP is alpha's #4 path, one-user-hop
  paths fill about 6 of the 10 pool slots, and most of them then hurt validation. The small graph
  was built with the fixed extraction, so this pattern does not come from the duplicates or the
  placeholder customers.
- **Self-inclusion is the cause.** `heldout-half` changes only whether alpha's edges are in the
  graph. PUP drops from #4 to #20, one-user-hop paths in the pool from 6.5 to 2.0, and alpha now
  agrees with validation (+0.78 vs +0.20). Validation never trained alpha in this variant, so the
  agreement is not circular.
- **`val` (the fix) behaves the same way** (PUP #19, agreement +0.82; that number is partly
  circular because val also trained alpha). It is the default because it searches on the full
  training graph, the same graph HAN will use. On `heldout-half`'s graph, which keeps only 40% of
  the edges, two-user-hop paths helped validation in 20% of cases instead of 75–90%: what the
  search concludes about a path depends on the graph it is computed on.
- **Scale caveat.** The small graph has 1,730 products. What carries over is the direction of the
  effect; the rerun on the rebuilt data will show the real pattern.

### S2b: options for the weights (parts 3 and 9)

The search's shared weights also train on features that contain each target edge, and so does
HAN if its meta-path graphs are built from the same training edges it learns from. That is not
test leakage, and the test numbers are not inflated. But the model learns to trust user paths more
than they deserve on held-out edges, which can penalise exactly the paths this project is trying
to rank. Options, cheapest first:
1. **Disjoint supervision:** hold out part of the train edges from the graph and use them only as
   BPR positives (PyG `disjoint_train_ratio`). The cost is a sparser graph during training.
2. **K-fold message passing:** build the user-path features or graphs K times, each time without
   one fold of the train edges, and train each edge on its own fold's version. Test uses the full
   train graph.
3. **Leave as is**, and say so when reporting.

Part 3 fixed this for the search; for HAN it is parked with the HAN repo review (part 9).

## What changed

Committed in `c49b776`:
- `train_search.py`: `--alpha-edges {val,heldout-half,train-half}`, default `val`.
  - `val`: weights on all train edges, alpha on the val edges. Alpha negatives avoid train and val
    positives. Each epoch has about twice as many steps as before (all train edges instead of
    half), so the epoch loop takes about twice as long.
  - `heldout-half`: the old internal split, with the alpha half removed from the graph
    (`link_data.remove_target_edges`).
  - `train-half`: the Fleet v2 behaviour, for reproduction only. Same losses as the old code,
    epoch for epoch.
- `train_search.py`: `torch.cuda.synchronize()` only without `--cpu`.
- `hgb/tests/test_link_split.py`: new.
- `hgb/README.md`: deviation 4 and the rebuild note.
- `hgb/run_fleet_v2.bat`: a header note that the default changed (`--alpha-edges train-half`
  reproduces `logs\Fleet v2`).

## How to verify

```
python hgb/tests/test_link_split.py
python hgb/tests/test_amazon_extract.py
```

Both run on CPU in under a minute.
