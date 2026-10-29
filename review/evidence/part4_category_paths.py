"""Review part 4: how empty each candidate path is on the rebuilt AMAZON data, as the Fleet v3
search sees it (plain P-P / C-C relations, U-P = the split's TRAINING edges, as in
load_dataset), and who the products with a P-C-C-P endpoint are.

For each path: the share of products with no endpoint at all (their mean-aggregated feature is
zero), the share whose only endpoint is themselves, and the median number of endpoints (from a
random sample of 400 products). Then, for P-C-C-P: product groups, training degree, and the
share of training and validation positives and of the sampled validation negatives.

    python review/evidence/part4_category_paths.py data "hgb/logs/Fleet v3/final_ns4/search_ns4_seed1.json"

Needs the LMSPS environment (hgb/link_data.py imports torch_sparse). Reads node.dat only for the
node types, so it runs on data/AMAZON as rebuilt.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hgb'))
from link_data import build_link_split, edges_fingerprint  # noqa: E402

root, record = sys.argv[1], sys.argv[2]
rec = json.load(open(record))
types = np.array([int(l.rstrip('\n').split('\t')[2]) for l in open(f'{root}/AMAZON/node.dat', encoding='utf-8')])
shift = {t: int(np.flatnonzero(types == t)[0]) for t in (0, 1, 2)}
n = {t: int((types == t).sum()) for t in (0, 1, 2)}
rel = {r: ([], []) for r in range(8)}
for line in open(f'{root}/AMAZON/link.dat'):
    h, t, r, _ = line.split('\t')
    rel[int(r)][0].append(int(h))
    rel[int(r)][1].append(int(t))


def adj(r, rt, ct):
    h, t = np.array(rel[r][0]) - shift[rt], np.array(rel[r][1]) - shift[ct]
    return sp.csr_matrix((np.ones(len(h)), (h, t)), shape=(n[rt], n[ct]))


# the search's own split, through a minimal stand-in for data_loader
total = len(types)
meta = {0: (0, 0), 1: (0, 0), 2: (1, 1), 3: (1, 1), 4: (0, 1), 5: (1, 0), 6: (2, 0), 7: (0, 2)}


class DL:
    nodes = {'count': n, 'shift': shift}
    links = {'meta': meta, 'data': {r: sp.csr_matrix((np.ones(len(rel[r][0])), (rel[r][0], rel[r][1])),
                                                     shape=(total, total)) for r in range(8)}}


split = build_link_split(DL, 2, 0, seed=rec['split_seed'])
fp = edges_fingerprint(split['train_edges'])
print(f'training-edge fingerprint {fp} (search record: {rec["data"]["train_edges_hash"]})')
assert fp == rec['data']['train_edges_hash'], 'not the data or split the search ran on'


def plain(m):
    m = (m + m.T).sign().tocsr()
    m.setdiag(0)
    m.eliminate_zeros()
    return m


pc = adj(4, 0, 1).sign().tocsr()
up = split['train_adj_src2dst'].sign().tocsr()
A = {'PP': plain(adj(0, 0, 0) + adj(1, 0, 0)), 'CC': plain(adj(2, 1, 1) + adj(3, 1, 1)),
     'PC': pc, 'CP': pc.T.tocsr(), 'UP': up, 'PU': up.T.tocsr()}
size = {'P': n[0], 'C': n[1], 'U': n[2]}


def nonempty(path):
    """Products with >= 1 endpoint: an indicator propagated backwards from the last type."""
    v = np.ones(size[path[-1]])
    for i in range(len(path) - 2, -1, -1):
        v = (A[path[i] + path[i + 1]] @ v > 0).astype(float)
    return v > 0


sample = np.sort(np.random.default_rng(0).choice(n[0], 400, replace=False))
rows = []
for p in rec['candidates']:
    m = A[p[0] + p[1]][sample]
    for i in range(1, len(p) - 1):
        m = (m @ A[p[i] + p[i + 1]]).sign().tocsr()
    d = np.diff(m.indptr)
    self_only = np.mean([d[k] == 1 and m.indices[m.indptr[k]] == sample[k] for k in range(len(sample))])
    rows.append((p, 1 - nonempty(p).mean(), self_only, np.median(d[d > 0]) if (d > 0).any() else 0))
print(f'\n{"path":<8} {"no endpoint":>12} {"only itself":>12} {"median endpoints (non-empty)":>30}')
for p, e, s, md in sorted(rows, key=lambda r: -r[1]):
    print(f'{p:<8} {e:>12.1%} {s:>12.1%} {md:>30.0f}')

grp = nonempty('PCCP')
deg = np.asarray(split['train_adj_src2dst'].sum(0)).ravel()
print(f'\nproducts with a PCCP endpoint: {grp.mean():.1%}')
print(f'  training degree: median {np.median(deg[grp]):.0f} vs {np.median(deg[~grp]):.0f} without, '
      f'mean {deg[grp].mean():.1f} vs {deg[~grp].mean():.1f}')
print(f'  share of training positives {deg[grp].sum() / deg.sum():.1%}, of validation positives '
      f'{grp[split["val_edges"][:, 1]].mean():.1%}, of the sampled validation negatives {grp[split["val_negs"].ravel()].mean():.1%}')
label = {}
for f in ('label.dat', 'label.dat.test'):
    for line in open(f'{root}/AMAZON/{f}', encoding='utf-8'):
        x = line.rstrip('\n').split('\t')
        label[int(x[0])] = int(x[3])
g = np.array([label.get(i + shift[0], -1) for i in range(n[0])])
names = {0: 'Book', 1: 'DVD', 2: 'Music', 3: 'Video'}   # dat_generator: sorted group names
for v in sorted(set(g)):
    print(f'  {names.get(v, v)}: {np.mean(g == v):.1%} of products, {grp[g == v].mean():.1%} of them with a PCCP endpoint')
