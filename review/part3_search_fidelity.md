# Review part 3: the search vs the paper and the upstream code

**Scope**
- `hgb/model_search.py` (the supernet), the link-prediction search in `hgb/train_search.py`, and
  its helpers in `hgb/utils.py` (training step, set scoring, feature propagation)
- compared with:
  - **the paper** (Li et al., NeurIPS 2024): Eq. 1–7, Algorithm 1, App. A.3, B.4, C.1, E.3, E.4,
    Table 4 and Table 11
  - **the authors' code**, JHL-HUST/LMSPS. Its latest commit is `1a1e2e8` (2024-12-04), the same
    commit as in this repo's history, so nothing upstream is newer than what the fork started
    from.

**Bottom line:** the search is a faithful port of the paper and the authors' code: same
supernet, schedule, bi-level step, selection and hyperparameters, and no implementation bug was
found. Two problems come from link prediction, which the paper (node classification) never had
to face:
- **The weights still trained on edges inside their own features.** Part 2 moved alpha off those
  edges; the weight step stayed on them. That biased the search *against* user paths. Fixed with
  fold-by-fold features (`--wstep-kfold 4`, now the default): PUP climbs back from alpha rank 19
  to 9, and seeds agree more.
- **The supernet overfits**, because link prediction adds a free vector per user. `--dropout 0.5`
  cuts the late rise of the validation loss by more than half. Recommended for the rerun as a
  flag; no code change. With both fixes the search overfits least and its picks agree most
  across seeds.

Two README numbers are wrong (part 8). The paper's Table 4 is exactly your flow (HAN retrained
with searched paths), which part 9 should reproduce.

## How it was checked

1. **Line-by-line comparison with upstream** (`git diff 1a1e2e8`):
   - `model_search.py`
   - the link-prediction branch of `train_search.py` against upstream's node-classification
     search, plus upstream's OGBN-MAG search (`ogbn/`), the large-scale case closest to AMAZON
   - `utils.train_search_lp` against `train_search_new`, and `project_op_lp` against `project_op`
2. **Every paper reference in the README and the code**, checked against the paper.
3. **The Fleet v2 logs** (30 runs each for ns 4 and ns 6): alpha spread, looks per path, pool
   contents.
4. **Experiments** on the k3 mini graph from part 2, same flags as part 2, 6 seeds each:
   - fold-by-fold weight-step features (F8)
   - dropout 0.5 (F9)

## Verdicts

| # | Item | Evidence | Verdict |
|---|---|---|---|
| F1 | Supernet: a projection per path, Gumbel-softmax over the sampled paths (Eq. 2, 4), PReLU | Identical to upstream. Link prediction only drops the classifier head, so the product embedding is `dropout(PReLU(Σ w·XW))`. Two inherited differences from the paper (App. B.4): one linear map per path instead of a two-layer MLP, and `uniform(-0.5, 0.5)` initialisation instead of Xavier (about 6× wider for these matrix sizes) | OK. The README lists the first; part 8 adds the second |
| F2 | Schedule | Pool shrinks linearly to V (Eq. 3), temperature 8 → 4, warm-up of 20 / 200 epochs available as `--warmup-epochs` (App. B.4). One path sample per **epoch**, as in App. E.4 and upstream's OGBN-MAG code. The paper ranks the pool by the noisy strength q_k (Eq. 3); the code by alpha, without noise | OK. Differences already in the README (deviation 5) |
| F3 | Bi-level step | Algorithm 1 lines 13–14, Eq. 5 and App. A.3: weights on the training loss, alpha on the validation loss, first-order. Same alternation as upstream's `train_search_new`; alpha is not in the weight optimizer; gradients are zeroed correctly | OK |
| F4 | Final selection | Paper (Eq. 6): 200 subsets drawn in proportion to strength. Upstream: 200 uniform draws, with Gumbel noise. Fork: every one of the 210 subsets scored once, without noise | OK; documented |
| F5 | Hyperparameters | Paper (App. B.4) and upstream agree: learning rates 1e-3, weight decay 0, hidden 512, α initialised to 1, temperature 8 → 4, M = 30, V = 60, 200 epochs. The fleets use M = 4–6, V = 10, 150 epochs | OK; README deviation 2 |
| F6 | Loss and scoring | BPR averaged over every positive–negative pair; dot products scored in chunks | OK |
| F7 | `--all_path` in upstream's README commands | Defined but never read, in upstream as well | Ignore |
| F8 | The search's weights still train on edges inside their own features (part 2, S2b) | Weights trained with each edge inside its own features learn a self-signal that validation edges lack, so user paths look worse than they are. Fold-by-fold features on k3: PUP alpha rank 19.2 → 9.3; seeds' selected sets share 2.5 of 4 paths instead of 1.9, their pools 6.9 of 10 instead of 5.4 | **Should fix: fixed** (`--wstep-kfold 4`, default; `0` = old behaviour) |
| F9 | The supernet overfits | Training BPR falls to 0.03 while validation BPR is best at epoch 54 (0.49) and ends at 0.66, the weights the final sets are scored with. Link prediction adds a free 512-dim vector per user; the paper uses weight decay 0 and upstream dropout 0. `--dropout 0.5` on k3: rise +0.17 → +0.07, final 0.66 → 0.58, pools 6.6 of 10 shared instead of 5.4 | **Recommended:** add `--dropout 0.5` to the rerun's flags; check the validation curve on the full data |
| F10 | `--alpha-edges val` doubles the weight steps per epoch | k3: alpha spread 2.4 → 4.2. On the rebuilt full data (about 1 M edges) that is still about 80 steps per epoch, fewer than Fleet v2's 105 | No action |
| F11 | Adam keeps moving an unsampled path's alpha | At most about 0.01 (10 × the learning rate), against a 0.10 gap at the pool boundary in Fleet v2 | Ignore |
| F12 | Products' own features are not a candidate (target-ended paths only) | HAN always has them, so the search could overvalue self-heavy paths. It doesn't: PP, PPP, PPPP never reached a final pool in Fleet v2 or on k3 | Ignore; document |
| F13 | The reverse-pair representative is the alphabetically smaller key (L10) | HAN's graphs are identical for both; only the search's feature differs | Ignore; document |
| F14 | Set scores weight paths by softmax(alpha / 4) | In Fleet v2 the selected paths got 0.12–0.40 of the weight, so a set's score is not an equal-weight score. Same as the paper (Eq. 4 weights inside Eq. 6's evaluation) | Part 4 |
| F15 | README: "warm-up gives eliminated paths ~24 looks" | Measured in the final fleets: 12 on average for ns 4, 18 for ns 6 | Part 8: correct the number |
| F16 | README: "`--num_final 10` is the paper's V = 2M for M = 4–6" | V = 2M (App. C.1) gives 8, 10 and 12 for M = 4, 5, 6 | Part 8: reword |

**Paper claims in the README and code, now all checked:** Eq. 1–7, Algorithm 1, the 20-epoch
warm-up (App. B.4), "skip progressive sampling when K ≤ 2M" (App. C.1), convergence within
100–180 epochs (Table 11 / App. E.3), M = 30 and V = 60 (App. B.4). All are accurate, including
README deviation 5's "pool ranked by raw alpha rather than Gumbel strength": Eq. 3 does rank by
the noisy strength.

**Your flow is the paper's Table 4.** The paper retrains HAN with LMSPS's searched meta-paths
("HAN*") and compares it with HAN on its original meta-paths: HAN* wins on all four datasets
(e.g. ACM 90.79 → 92.28). That is the validation part 9 should reproduce for AMAZON: HAN with the
searched paths against HAN with hand-picked and random path sets, on the same split.

## F8: the weights still learned from edges inside their own features

**Mechanism.** Part 2 stopped alpha from learning on in-graph training edges. The weight step
still learns from them: for a training pair (u, p), p's user-path features contain u's own
history. The weights learn to use that self-signal. Validation edges never carry it, so when alpha
and the final scoring evaluate user paths on validation edges, those paths underperform and look
worse than they are.

**Upstream has no such problem.** In node classification the labels are never part of the graph,
and label features drop each node's own label (`remove_diag`). So upstream's weight step never
sees its own target in its features either. Fold-by-fold features are the link-prediction
counterpart.

**The fix (`--wstep-kfold K`).**
- The weight step's training edges are split into K folds.
- For each fold, the features are propagated on a graph **without that fold**.
- Every training edge reads its own fold's features, for its positive and its negative product.
- The alpha step and the final scoring keep the full training graph, the same graph HAN will use.

**Evidence** (k3, 6 seeds each, same flags):

| | `val` (part 2 fix) | `val` + fold-by-fold weight step |
|---|---|---|
| PUP alpha rank (of 36) | 19.2 (9–33) | 9.3 (5–18) |
| one-user-hop paths in the final pool of 10 | 4.2 | 5.5 |
| two-user-hop paths in the selected 4 | 1.8 | 2.2 |
| seed agreement: shared paths between two selected sets | 1.9 of 4 | 2.5 of 4 |
| seed agreement: shared paths between two final pools | 5.4 of 10 | 6.9 of 10 |

For reference, the old biased setup (`train-half`) put PUP at rank 4.2. Part 2's fix removed that
bias but over-corrected; this fix settles PUP between the two.

**Cost.**
- K extra feature propagations: about 33 s each on the old full data, so about 2 minutes on a
  50-minute search at K = 4.
- K copies of the 14 user-path candidates in CPU RAM: about 2 GB on the rebuilt data at K = 4
  (estimated). Each fold keeps only the final candidates in the final dtype.

**Checks.**
- The repo code reproduces the experiment run epoch for epoch, and `--wstep-kfold 0` reproduces
  the part 2 runs.
- Two new tests (fold features, sampler fold IDs); five deliberate breaks, all caught.

## F9: the supernet overfits

**Evidence** (k3, `val`, 6 seeds): training BPR falls to 0.03; validation BPR is best at epoch 54
(0.49) and then climbs to 0.66 by the last epoch. The final sets are scored with those
end-of-search weights. Fleet v2 couldn't show this: its in-graph alpha edges looked easy.

**Cause.** Link prediction adds a free 512-dimensional vector per user, which can memorise each
user's few training edges. The paper uses weight decay 0 (App. B.4) and does not specify dropout;
upstream's default dropout is 0 (its code comment says the original was 0.5). Node classification
has no per-node free parameters, so it never needed more.

**Dropout 0.5** (upstream's original value), k3, 6 seeds:

| | `val` | `val` + dropout 0.5 |
|---|---|---|
| training BPR at the end | 0.029 | 0.113 |
| validation BPR: best → end | 0.489 → 0.664 (+0.174) | 0.507 → 0.578 (+0.071) |
| seed agreement: final pools | 5.4 of 10 | 6.6 of 10 |
| seed agreement: selected sets | 1.9 of 4 | 1.9 of 4 |

It halves the overfitting and steadies the pools. It does not fix F8's bias (PUP stays at rank
17), so the two fixes are complementary.

**Both together** (`--wstep-kfold 4 --dropout 0.5`, the repo code, k3, 6 seeds):

| | `val` | + dropout | + folds | + folds + dropout |
|---|---|---|---|---|
| validation BPR rise after its best | +0.174 | +0.071 | +0.145 | **+0.045** |
| validation BPR at the end | 0.664 | 0.578 | 0.626 | **0.548** |
| PUP alpha rank | 19.2 | 17.0 | 9.3 | 10.2 |
| seed agreement: selected sets | 1.9 of 4 | 1.9 of 4 | 2.5 of 4 | **2.7 of 4** |
| seed agreement: final pools | 5.4 of 10 | 6.6 of 10 | 6.9 of 10 | 5.8 of 10 |
| distinct selected sets | 6 of 6 | 6 of 6 | 6 of 6 | **5 of 6** |

No bad interaction: the combination overfits least and picks the most consistent sets. It
converges on a clear core: **PUPUP and PPUPUP in all 6 runs, PUPPUP in 5**, plus one category path
in the fourth slot. That is the trio Fleet v2 called its core, now found on clean data with both
leaks fixed (part 1 had warned it might be an artifact of edition duplicates). On a 1,730-product
graph this is a hint, not proof; the full-data rerun decides. These three paths are exactly the
ones that turn into near-complete graphs when HAN keeps only reachability, so how the HAN repo
builds its graphs (part 9) decides whether HAN can use them.

**On the full data,** "val bpr" in the search log is now the alpha step's loss on the validation
edges, so the curve is visible every epoch. If it still climbs after its minimum with dropout 0.5,
regularise the user vectors further (weight decay on the user tower, or smaller user vectors).

## What changed

Not committed yet (pending your approval):
- `hgb/utils.py`: `fold_user_path_feats()`; `train_search_lp(..., fold_stack=)`.
- `hgb/link_data.py`: `TrainEdgeSampler(..., fold_of=)` also returns each sampled edge's fold,
  with the same random stream.
- `hgb/train_search.py`: `--wstep-kfold` (default 4).
- `hgb/tests/test_link_split.py`: `test_sampler_fold_ids`, `test_fold_user_path_feats` (9 tests).
- `hgb/README.md` (deviation 8, the rebuild note) and `hgb/run_fleet_v2.bat` (header note).

Recommended for the rerun's flags, no code change: `--dropout 0.5`.
