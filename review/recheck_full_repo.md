# Full re-check of the repo (2026-10-29, after Fleet v3)

**Scope:** every file in the link-prediction flow, read again line by line now that the Fleet v3
logs exist to test it against:
- the data path: `amazon_extract.py`, `dat_generator.py`, `data_loader.py`
- the split, features and search: `link_data.py`, `utils.py` (`load_dataset`, feature
  propagation, fold features, the training step, set scoring), `train_search.py`,
  `model_search.py`
- the selection, scripts and docs: `reselect_fleet.py`, `analyze_search_fleet.py`,
  `compare_search_records.py`, `run_fleet_v3.bat`, `execute.bat`, `arch.py`, the READMEs

`main_path.py` is out of the flow (part 2) and was not re-read.

**Bottom line:** no bug and no must-fix in the code; the Fleet v3 results stand. One reporting
rule was missing and is added (README). One run time in the README was stale.

## What was checked

| Area | Checked | Result |
|---|---|---|
| Training step | Alpha's gradient from the weight step is zeroed before the alpha step; the weights' and the user tower's gradients from the alpha step are zeroed before the next weight step; alpha is not in the weights' optimizer | OK |
| Path weights | The channels of the stacked features and the weights `ws` follow the same index order in training (`sorted` samples) and in the exhaustive scoring (sorted pool, `itertools.combinations`) | OK |
| Features | Message direction of every relation (row = head, column = tail in `data_loader`, `row = dst` in `load_dataset`), user-product edges from the training split only, relations in `link.dat` in the order `load_dataset` unpacks them (checked on the rebuilt file) | OK |
| Fold features | Each fold's graph drops that fold's training edges in both directions; only keys through users differ by fold; validation edges are in no graph | OK (also `test_fold_user_path_feats`) |
| Leakage | No search code reads `test_edges` or `test_negs`; the full adjacency is used only to keep evaluation negatives off positives | OK |
| Seeding | Python, numpy and torch seeded; split, samplers and folds on their own generators; the pilot reproduces exactly | OK |
| Data path | Placeholder removal, edition union-find, the rating and helpfulness filters, one interaction per user and work, the 5-core, leaf category links, LSA title features | OK |
| Records | All 62 Fleet v3 records: the pool is the top 10 by alpha, all 210 subsets scored once, the reported set is the argmin, the tie group matches, no NaN (`evidence/recheck_records.py`) | OK |
| Tests | `test_amazon_extract`, `test_link_split`, `test_reselect_fleet` | Pass |

## Changed

Not committed yet:
- `hgb/README.md`:
  - the split-seed rule now also says to compare the searched sets with manual and random sets
    on test numbers only, because the search chose its sets on the same validation edges
    (their validation numbers lean in their favour)
  - "~1 h per run" corrected to ~20 min on the rebuilt data
  - a pointer to the Fleet v3 review
- `review/evidence/recheck_records.py`: the record check above.
- `review/recheck_full_repo.md`: this note.
