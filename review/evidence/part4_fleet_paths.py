"""Review part 4: what the search's own validation loss says about single paths, per fleet.

1. Path effects: val_bpr(run, set) = run effect + sum of path effects over the set, least
   squares over every scored set of every run (each run scores all subsets of its 10-path
   pool). Centred: only differences between paths are identified. Negative = the path lowers
   validation loss. With how often the path reached a final pool and its mean alpha rank.
2. Near-empty category paths (no endpoint for 76-81% of products on the rebuilt data,
   part4_category_paths.py): how many each run's best set holds, and how far the best set with
   k of them is from the run's best.

    python review/evidence/part4_fleet_paths.py "hgb/logs/Fleet v3/final_ns4/search_*.json"
"""
import glob
import json
import sys

import numpy as np

NEAR_EMPTY = {'PCCP', 'PCCPP', 'PCCPPP', 'PCCPCP', 'PCCPUP', 'PPCCPP'}

recs = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1]))]
recs = [d for d in recs if 'eval_table' in d]
cands = recs[0]['candidates']
idx = {p: i for i, p in enumerate(cands)}
R, P = len(recs), len(cands)
X, y = [], []
for r, d in enumerate(recs):
    for e in d['eval_table']:
        row = np.zeros(R + P)
        row[r] = 1
        row[R + np.array([idx[p] for p in e['paths']])] = 1
        X.append(row)
        y.append(e['val_bpr'])
X, y = np.array(X), np.array(y)
seen = X[:, R:].sum(0) > 0
cols = np.r_[np.arange(R), R + np.flatnonzero(seen)]
coef = np.linalg.lstsq(X[:, cols], y, rcond=None)[0]
eff = np.full(P, np.nan)
eff[seen] = coef[R:]
eff -= np.nanmean(eff)
pool = [sum(p in d['final_pool'] for d in recs) for p in cands]
rank = [np.mean([sorted(d['alpha'], key=lambda q: -d['alpha'][q]).index(p) + 1 for d in recs]) for p in cands]
print(f'{R} runs, {len(y)} scored sets')
print(f'{"path":<8} {"effect":>8} {"in pool":>8} {"mean alpha rank (of 36)":>24}')
for i in np.argsort(np.where(seen, eff, 9)):
    print(f'{cands[i]:<8} {(f"{eff[i]:+.4f}" if seen[i] else "-"):>8} {pool[i]:>5}/{R} {rank[i]:>18.1f}')

ns = len(recs[0]['eval_table'][0]['paths'])
gaps = {k: [] for k in range(ns + 1)}
best_k = []
for d in recs:
    t = d['eval_table']
    scores = [e['val_bpr'] for e in t]
    ks = [sum(p in NEAR_EMPTY for p in e['paths']) for e in t]
    best_k.append(ks[int(np.argmin(scores))])
    for k in gaps:
        s = [v for v, kk in zip(scores, ks) if kk == k]
        if s:
            gaps[k].append(min(s) - min(scores))
print(f'\nnear-empty category paths in each run\'s best set: '
      f'{dict(sorted((k, best_k.count(k)) for k in set(best_k)))}')
print(f'{"k near-empty":>13} {"runs":>6} {"best such set vs the run best: median":>40} {"mean":>8}')
for k, g in gaps.items():
    if g:
        print(f'{k:>13} {len(g):>6} {np.median(g):>40.4f} {np.mean(g):>8.4f}')
