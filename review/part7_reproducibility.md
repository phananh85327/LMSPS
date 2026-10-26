# Review part 7: reproducibility, scripts and repo hygiene

**Scope:** whether the rerun's results can be traced and reproduced. That covers what a search
record holds, same-seed determinism of the changed code, the run scripts, the environment, the
tests, and loose ends in the repo. The rerun script itself was part 8.

**Bottom line:** one real gap.

**Gap: a search record didn't say which data it ran on.**
- The records stored every flag, but nothing about the data. Fleet v2's records can't be told
  apart from runs on the rebuilt data, except by the date.
- A fleet folder that mixed runs from before and after the rebuild would have looked normal.

**Fixed:**
- Each record now stores:
  - the data build settings (`meta.json`) and node counts
  - a fingerprint of the training edges, which changes with the data and with the split
  - library versions and the device
- `reselect_fleet.py` prints the fingerprint and warns when a fleet mixes versions.

**Determinism:** two same-seed runs of the current code on the CPU are identical: alpha, final
pool and all 210 scored sets (k3, seed 7, 12 epochs; no warning about a non-deterministic op). The GPU check is the new `run_fleet_v3.bat pilot`
(two full runs of seed 1, compared automatically). Fleet v2 ran the same pilot on the old code.

Everything else is either fine or a documentation matter.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| R1 | Search records don't say which data / split they ran on | Records held args only; the data was rebuilt in part 1 | **Fixed:** `data` (meta.json, node counts, training-edge fingerprint) and `env` (Python, torch, dgl, numpy, device) in every record; `reselect_fleet.py` shows the fingerprint and warns on a mix |
| R2 | Same-seed determinism after the part 2/3/9 changes | CPU, k3, seed 7, 12 epochs: see above | **Check added:** `run_fleet_v3.bat pilot` + `compare_search_records.py` (exit 0 = identical) |
| R3 | `E:\miniconda3` hard-coded | Every `.bat` | `run_fleet_v3.bat` takes `CONDA_ROOT`; the others left as they are |
| R4 | How to run the tests | Not documented | README section "Reproducibility and tests"; all 21 tests pass, also under pytest |
| R5 | Version pins | `requirements.txt` pins and explains every package (torch 2.1.2+cu121, torch_sparse 0.6.18, dgl 2.2.1, …) | OK; records now store the versions actually used |
| D3 | `execute.bat` with no argument ran 60 searches | — | Fixed in part 8 |
| D6 | The reselect glob matched its own summaries | — | Fixed in part 8 (`search_*.json`) |
| D7 | Docs cite material that isn't in the repo | Only `logs/Fleet v2/final_ns4|final_ns6` are committed. `pilot`, `ablation_old55`, `ablation_ns8`, `Fleet v1`, `Top K 4|6` and `chat/` are not | README note. **If the thesis cites the schedule ablation or the determinism pilot, commit those JSON records from your PC** |
| D11 | `sparse_tools` is a gitlink with no URL | A fresh clone gets an empty folder; upstream has no `sparse_tools` at all | Documented. Not removed: only node classification needs it, and removing the gitlink could disturb your local compiled copy |
| D8 | Windows-only scripts | — | Ignore: your machine runs Windows; the README's Python commands work anywhere |
| D10 | `Fleet v2` vs "V2" in your notes | Windows paths are case-insensitive | Ignore |
| D12 | `main_path.py` has no `--deterministic` | Out of the flow | Ignore |

## What changed

Not committed (pending your approval):
- `hgb/link_data.py`: `edges_fingerprint`, `data_provenance`.
- `hgb/train_search.py`: `data` and `env` in the search record.
- `hgb/reselect_fleet.py`: the data fingerprint in the provenance line; a warning when a fleet
  mixes data versions.
- `hgb/compare_search_records.py`: new; the pilot's check.
- `hgb/run_fleet_v3.bat`: `pilot` stage, `CONDA_ROOT`.
- `hgb/tests/test_link_split.py`: `test_edges_fingerprint_and_data_provenance`.
- `hgb/tests/test_reselect_fleet.py`: `test_mixed_data_versions_are_flagged`,
  `test_compare_search_records`.
- Deliberate breaks of the new code, all caught:
  - an order-dependent fingerprint
  - a fingerprint of the validation edges
  - no mixed-data warning
- `hgb/README.md`: "Reproducibility and tests".

## How to verify

```
python -m pytest hgb/tests          # or each file as a script
run_fleet_v3.bat pilot              # on the GPU machine, once before the fleets
```
