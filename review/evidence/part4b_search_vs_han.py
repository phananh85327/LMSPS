"""Review part 4, second half: does the search rank paths and path sets as HAN does?

Reads the Fleet v3 records (alpha per candidate path) and, from the HAN repo, arch.py, the
results table and the seed-1 HAN logs (the path weights beta).

    python review/evidence/part4b_search_vs_han.py "hgb/logs/Fleet v3" ../HAN-improve

1. per candidate path: the search's mean alpha rank (1 = most preferred of 36) against HAN's
   weight for it (beta x number of paths: 1 = an equal share) and how often HAN drops it
2. the same by kind of path
3. the 16 sets drawn from the search's own 36 candidates (6 LMSPS, 10 random), per size: HAN's
   order, and where the LMSPS sets land
4. Spearman between the search's preference for a set's paths and HAN's result, with
   permutation p-values
5. sets with and without a category path that reaches (almost) every product
"""
import csv
import glob
import importlib.util
import json
import os
import random
import re
import statistics as st
import sys

fleet, han_repo = sys.argv[1], sys.argv[2]
EMPTY = {'PCCP', 'PCCPP', 'PCCPPP', 'PCCPCP', 'PCCPUP', 'PPCCPP'}   # no endpoint for 76-81% of products (part 4)

recs = {ns: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(fleet, f'final_ns{ns}', 'search_*.json')))]
        for ns in (4, 6)}
cands = recs[4][0]['candidates']
rank = {}
for ns, rs in recs.items():
    for d in rs:
        order = sorted(d['alpha'], key=lambda k: -d['alpha'][k])
        for p in cands:
            rank.setdefault((p, ns), []).append(order.index(p) + 1)
rank = {k: st.mean(v) for k, v in rank.items()}

spec = importlib.util.spec_from_file_location('han_arch', os.path.join(han_repo, 'arch.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sets = {k: v[0] for k, v in mod.archs.items() if re.match(r'amazon_lp_(manual_\d|rand\d_d\d|v3_f\d_)', k)}
rows = list(csv.DictReader(open(os.path.join(han_repo, 'output', 'AMAZON', 'v3', 'results.csv'), encoding='utf-8')))
res = {(r['model'], r['arch']): r for r in rows if r['seed'] == '1'}
beta = {}
for k in sets:
    txt = open(os.path.join(han_repo, 'output', 'AMAZON', 'v3', 'logs', f'han_{k}_s1.txt'), encoding='utf-8',
               errors='replace').read()
    m = re.search(r'Semantic attention over meta-paths:\n((?:\s+\S+: [\d.]+\n)+)', txt)
    beta[k] = {a: float(b) for a, b in re.findall(r'(\S+): ([\d.]+)', m.group(1))}
kind = lambda p: ('near-empty category' if p in EMPTY else 'through users' if 'U' in p
                  else 'other category' if 'C' in p else 'P-P only')
setkind = lambda k: 'LMSPS' if '_v3_' in k else 'random' if 'rand' in k else 'manual'
print(f'{len(cands)} candidates, {len(recs[4])} + {len(recs[6])} records; {len(sets)} HAN sets')

print('\n1. per path: search rank (ns 4 / ns 6) | HAN: sets, weight relative to an equal share, dropped (beta < 0.01)')
use = {}
for k, b in beta.items():
    for p, v in b.items():
        use.setdefault(p, []).append(v * len(b))
for p in sorted(use, key=lambda p: rank[(p, 4)] + rank[(p, 6)]):
    print(f'   {p:8s} {rank[(p, 4)]:5.1f} / {rank[(p, 6)]:5.1f} | {len(use[p]):2d} sets, {st.mean(use[p]):4.2f}, '
          f'dropped {sum(beta[k][p] < 0.01 for k in beta if p in beta[k])}  ({kind(p)})')

print('\n2. by kind of path (paths used by at least one HAN set)')
for kd in ('through users', 'near-empty category', 'other category', 'P-P only'):
    ps = [p for p in use if kind(p) == kd]
    n_use = sum(len(use[p]) for p in ps)
    dropped = sum(1 for k in beta for p, v in beta[k].items() if kind(p) == kd and v < 0.01)
    print(f'   {kd:20s} {len(ps):2d} paths, mean search rank {st.mean((rank[(p, 4)] + rank[(p, 6)]) / 2 for p in ps):4.1f} | '
          f'HAN: {n_use} uses, dropped {dropped}, median relative weight {st.median(x for p in ps for x in use[p]):.2f}')

table = []
for k, ps in sets.items():
    n = len(ps)
    ns = 4 if n <= 4 else 6 if n == 6 else None
    score = st.mean(rank[(p, ns)] if ns else (rank[(p, 4)] + rank[(p, 6)]) / 2 for p in ps)
    h, m = res[('HAN', k)], res[('metapath2vec', k)]
    table.append(dict(set=k, n=n, kind=setkind(k), search=score, han=float(h['hit10']), val=float(h['val_bpr']),
                      full=float(h['hit10_full']), m2v=float(m['hit10']),
                      dense=any('C' in p and p not in EMPTY for p in ps)))

print('\n3. the sets drawn from the 36 candidates, seed 1, best first')
for n in (4, 6):
    sub = [t for t in table if t['n'] == n and t['kind'] != 'manual']
    for col, rev in (('han', True), ('val', False), ('full', True), ('m2v', True)):
        order = sorted(sub, key=lambda t: t[col], reverse=rev)
        pos = [i + 1 for i, t in enumerate(order) if t['kind'] == 'LMSPS']
        print(f'   {n} paths, {col:4s}: LMSPS at {pos} of {len(order)}: '
              + ' > '.join(t['set'][10:] for t in order))


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for t in range(i, j + 1):
            r[o[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    return (sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
            / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5)


def corr(label, sub, col):
    a, b = [t['search'] for t in sub], [t[col] for t in sub]
    r0, rng, bb, hit = spearman(a, b), random.Random(0), list(b), 0
    for _ in range(20000):
        rng.shuffle(bb)
        hit += abs(spearman(a, bb)) >= abs(r0) - 1e-12
    print(f'   {label:40s} n={len(sub):2d}  rho {r0:+.2f}  p {(hit + 1) / 20001:.3f}')


print('\n4. search preference (mean alpha rank of the set\'s paths; LOW = preferred) against HAN')
print('   a positive rho on Hit@10 means: the more the search likes a set\'s paths, the worse HAN does')
cand = [t for t in table if t['kind'] != 'manual']
corr('16 candidate sets, HAN Hit@10', cand, 'han')
corr('16 candidate sets, HAN validation BPR', cand, 'val')
corr('16 candidate sets, HAN full-ranking Hit@10', cand, 'full')
corr('10 random draws only, HAN Hit@10', [t for t in table if t['kind'] == 'random'], 'han')
corr('all 24 sets, HAN Hit@10', table, 'han')
corr('all 24 sets but M4, metapath2vec Hit@10', [t for t in table if t['set'] != 'amazon_lp_manual_4'], 'm2v')

print('\n5. with and without a category path that reaches (almost) every product (PCP, PCCCP, PCPCP, ...)')
for n in (4, 6):
    sub = [t for t in table if t['n'] == n]
    yes, no = [t for t in sub if t['dense']], [t for t in sub if not t['dense']]
    print(f'   {n} paths: with ({len(yes)} sets) HAN Hit@10 {min(t["han"] for t in yes):.4f}-{max(t["han"] for t in yes):.4f}; '
          f'without ({len(no)}: {", ".join(t["set"][10:] for t in no)}) {min(t["han"] for t in no):.4f}-{max(t["han"] for t in no):.4f}')
