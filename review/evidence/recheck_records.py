"""Full re-check (2026-10-29): is every search record internally consistent?
For each record: the final pool is the top 10 paths by alpha, every subset of the pool of
size ns was scored exactly once, the reported set is the table's argmin, the tie group
holds exactly the sets within tie_delta, and no score is NaN.

    python review/evidence/recheck_records.py "hgb/logs/Fleet v3/final_ns*/search_*.json" "hgb/logs/Fleet v3/pilot/*.json"
"""
import glob
import json
import math
import sys

files = sorted(f for pattern in sys.argv[1:] for f in glob.glob(pattern))
failures = []
for f in files:
    d = json.load(open(f))
    table, pool, ns = d['eval_table'], set(d['final_pool']), d['ns_select']
    best = min(table, key=lambda e: e['val_bpr'])
    top = sorted(d['alpha'], key=lambda k: -d['alpha'][k])[:len(pool)]
    checks = {
        'pool is the top paths by alpha': set(top) == pool,
        'every subset scored once': len(table) == math.comb(len(pool), ns) == len({tuple(sorted(e['paths'])) for e in table}),
        'subsets inside the pool': all(set(e['paths']) <= pool for e in table),
        'reported set is the argmin': sorted(best['paths']) == sorted(d['paths']),
        'tie group matches tie_delta': len(d['tie_sets']) == sum(e['val_bpr'] - best['val_bpr'] <= d['tie_delta'] for e in table),
        'no NaN score': all(e['val_bpr'] == e['val_bpr'] for e in table),
    }
    failures += [(f, name) for name, ok in checks.items() if not ok]
print(f'{len(files)} records checked; failures: {failures or "none"}')
sys.exit(1 if failures or not files else 0)
