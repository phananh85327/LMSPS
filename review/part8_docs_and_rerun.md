# Review part 8: README, `arch.py` and the rerun scripts (targeted)

**Scope:** only what would break or mislead the rerun, or end up in the thesis as a wrong
number. Cosmetic issues and references to files that live outside the repo are left alone.

**Bottom line:** two things would have hurt the rerun directly.
1. **Running the Fleet v2 script again would have overwritten the Fleet v2 records.**
   `run_fleet_v2.bat final4/final6` writes to the same `logs/Fleet v2/final_ns*` files. It also
   lacks the `--dropout 0.5` recommended in part 3. `execute.bat` with no argument ran all 60
   searches, although its header says it only re-selects. **Fixed:**
   - a new `run_fleet_v3.bat` with its own folder and resume
   - optional commit-and-push of the logs
   - `execute.bat` now does what its header says
2. **The re-selection printed `arch.py` entries only for the vote.** Part 5 recommends the
   averaged one-SE sets for HAN, and the entries carried a hard-coded "fleet v2, split-seed 1"
   label. The documented consensus key `amazon_lp_f4_c` didn't exist in `arch.py` (KeyError).
   **Fixed:**
   - entries for the averaged one-SE sets (`<prefix>_a<i>`)
   - a provenance line read from the runs' own settings
   - aliases for the Fleet v2 consensus keys

Five README claims were wrong or out of date; all are corrected.

## Verdicts

| # | Issue | Evidence | Verdict |
|---|---|---|---|
| N1 | Rerunning `run_fleet_v2.bat` overwrites the Fleet v2 logs; no `--dropout 0.5` | Same output paths, no skip logic | **Fixed:** `run_fleet_v3.bat` |
| D3 | `execute.bat` with no argument runs `run_fleet_v2.bat all` (60 searches, ~58 h, over the committed logs) | Header says "No argument = reselect (seconds, harmless)" | **Fixed:** no argument = reselect |
| N2 | `reselect_fleet.py` gives `arch.py` entries only for the vote's sets, labelled "fleet v2 final schedule, split-seed 1" whatever the runs were | Part 5: the averaged one-SE sets are the ranking to hand to HAN | **Fixed:** `<prefix>_a<i>` entries first; provenance from the runs' args |
| D1 | Consensus key: README `amazon_lp_f4_consensus` / `_r1` and `amazon_lp_f4_c`; `arch.py` `_s1` (ns 4) and `_s4` (ns 6) | `--arch amazon_lp_f4_c` raised KeyError | **Fixed:** aliases in `arch.py`; run-flow table uses the script's names |
| D6 | The `*.json` glob also matches `reselect_d0xx.json` | Skipped with a warning | Fixed in v3 (`search_*.json`); harmless in v2 |
| C1 | "Warm-up alone gives eliminated paths ~24 looks" | Fleet v2 logs: 12 epochs on average for ns 4, 18 for ns 6 (medians 9.5 / 14) | **Fixed** |
| C2 | "`--num_final 10` is the paper's V = 2M for M = 4-6" | 10 = 2M only for M = 5 | **Fixed** (README and `run_fleet_v2.bat`) |
| C3 | "55 target-ended paths at 5 hops" as the candidate count | 36 with `--dedupe-reversed` | **Fixed** |
| C4 | Deviation 5 omits the initialisation | uniform(-0.5, 0.5) instead of Xavier, about 6x wider (part 3) | **Fixed** |
| C5 | Split-seed rule names only the retrain | HAN and metapath2vec must match too (part 9) | **Fixed** |
| D4 | "Search stability" section lists PUPUPP and PUPCCP, which `--dedupe-reversed` removes | From the old schedule (55 candidates, no warm-up) | Labelled as such |
| D5 | Run flow: `main_path.py` as a downstream step; the vote's shortlist for HAN | Out of the flow (part 2); part 5 replaced the vote | **Fixed:** rerun flow; `main_path.py` optional |
| D9 | Purpose of `amazon_lp_unique_strings` undocumented | Union of the 13 shortlist paths | Documented |
| — | Cited files not in the repo (chat logs, Fleet v1, `Top K`), folder-name casing, platform lock-in, the orphan `sparse_tools` gitlink | — | Not changed (part 7 if at all) |

## `run_fleet_v3.bat`

Same protocol as Fleet v2's `final4` / `final6`, with:
- the current defaults: alpha on validation edges, fold-by-fold weight steps, plain relations
- `--dropout 0.5`
- `logs/Fleet v3` as its folder

A run whose JSON record exists is skipped, so the script can be relaunched after an interruption.
`reselect` saves its output next to the runs (`reselect_d010.txt/.json`, `_d005`). `push` commits
`logs/Fleet v3` and pushes the current branch. `all` runs final4, push, final6, push, reselect,
push. It can't be run here (Windows batch); it follows `run_fleet_v2.bat` line for line except
for those changes.

## What changed

Not committed (pending your approval):
- `hgb/run_fleet_v3.bat`: new.
- `hgb/execute.bat`: no argument = reselect.
- `hgb/reselect_fleet.py`:
  - `provenance()`
  - `arch.py` entries for the averaged one-SE sets
  - `averaged_one_se_keys` and `provenance` in the summary JSON
- `hgb/tests/test_reselect_fleet.py`: `test_arch_entries_and_provenance`. Three deliberate breaks
  are all caught: keys numbered from 0, no old-behaviour defaults, entries out of order.
- `hgb/arch.py`: `amazon_lp_f4_c` / `amazon_lp_f6_c` aliases; note on
  `amazon_lp_unique_strings`.
- `hgb/README.md`: rerun note and run flow, search command (`--dropout 0.5`, v3 paths), C1–C5,
  D4, retraining marked optional.
- `hgb/run_fleet_v2.bat`: the V = 2M comment.
- `hgb/train_search.py`: `--split-seed` help.

## How to verify

```
python hgb/tests/test_reselect_fleet.py
python hgb/reselect_fleet.py "hgb/logs/Fleet v2/final_ns4/search_*.json" --prefix amazon_lp_f4
```
