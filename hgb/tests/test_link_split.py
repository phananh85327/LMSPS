"""Leakage guards of the link-prediction split (review part 2).

    python hgb/tests/test_link_split.py      (or: python -m pytest hgb/tests)

Needs the training environment (torch, torch_sparse, dgl, scikit-learn): it imports
link_data.py and utils.py the way train_search.py does.
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
import scipy.sparse as sp

HGB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HGB)
from link_data import TrainEdgeSampler, build_link_split, get_relation_edges  # noqa: E402

N_P, N_C, N_U = 300, 5, 200


class FakeLoader:
    """The parts of data_loader that link_data reads: node counts/shifts and one
    U->P relation stored as a (total x total) matrix with global ids."""

    def __init__(self, reverse=False, seed=0):
        rng = np.random.default_rng(seed)
        self.nodes = {'count': {0: N_P, 1: N_C, 2: N_U}, 'shift': {0: 0, 1: N_P, 2: N_P + N_C}}
        total = N_P + N_C + N_U
        u = np.repeat(np.arange(N_U), 12)
        p = rng.integers(0, N_P, size=u.size)
        h, t = (p, u + N_P + N_C) if reverse else (u + N_P + N_C, p)
        m = sp.csr_matrix((np.ones(u.size), (h, t)), shape=(total, total))
        self.links = {'meta': {6: (0, 2) if reverse else (2, 0)}, 'data': {6: m}}
        self.edges = np.unique(np.stack([u, p], axis=1), axis=0)


def test_partition_and_determinism():
    dl = FakeLoader()
    s = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    parts = [s['train_edges'], s['val_edges'], s['test_edges']]
    n = len(dl.edges)
    assert [len(x) for x in parts] == [n - 2 * int(0.1 * n), int(0.1 * n), int(0.1 * n)]
    allp = np.concatenate(parts)
    assert len(np.unique(allp, axis=0)) == n                       # disjoint ...
    assert np.array_equal(np.unique(allp, axis=0), dl.edges)       # ... and complete
    again = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    other = build_link_split(dl, 2, 0, seed=2, num_neg_eval=20)
    for k in ('train_edges', 'val_edges', 'test_edges', 'val_negs', 'test_negs'):
        assert np.array_equal(s[k], again[k]), k
    assert not np.array_equal(s['val_edges'], other['val_edges'])


def test_adjacencies():
    dl = FakeLoader()
    s = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    tr = s['train_adj_src2dst'].tocoo()
    assert np.array_equal(np.unique(np.stack([tr.row, tr.col], 1), axis=0), np.unique(s['train_edges'], axis=0))
    assert s['full_adj_src2dst'].nnz == len(dl.edges)


def test_eval_negatives_never_hit_a_positive():
    dl = FakeLoader()
    s = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    full = s['full_adj_src2dst']
    for e, negs in ((s['val_edges'], s['val_negs']), (s['test_edges'], s['test_negs'])):
        assert negs.shape == (len(e), 20)
        assert np.asarray(full[np.repeat(e[:, 0], 20), negs.ravel()]).ravel().max() == 0   # no train/val/test positive
        assert all(len(set(row)) == 20 for row in negs.tolist())                       # no duplicates per edge


def test_reverse_stored_relation():
    a, b = FakeLoader(reverse=False), FakeLoader(reverse=True)
    for dl in (a, b):
        src, dst = get_relation_edges(dl, 2, 0)
        assert np.array_equal(np.unique(np.stack([src, dst], 1), axis=0), a.edges)


def test_train_negatives_only_avoid_train_positives():
    dl = FakeLoader()
    s = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    sampler = TrainEdgeSampler(s['train_edges'], s['train_adj_src2dst'], seed=0)
    hit_heldout = 0
    heldout = set(map(tuple, np.concatenate([s['val_edges'], s['test_edges']]).tolist()))
    for _ in range(200):
        src, pos, neg = sampler.sample_batch(256)
        src, pos, neg = (np.array(x.numpy()) for x in (src, pos, neg))   # scipy indexing needs writeable arrays
        assert np.asarray(s['train_adj_src2dst'][src, pos]).min() == 1    # positives are train edges
        assert np.asarray(s['train_adj_src2dst'][src, neg]).max() == 0    # negatives avoid train positives
        hit_heldout += sum((a, b) in heldout for a, b in zip(src.tolist(), neg.tolist()))
    # held-out positives MUST stay drawable as training negatives: excluding them would tell
    # the trainer which items are in val/test. ~2.4 held-out items per user out of 300
    # products -> ~0.8% of 51,200 draws, so a sampler that excluded them would show 0 here.
    assert hit_heldout > 0, 'training negatives never hit a held-out positive: is val/test being excluded?'


def _write_filtered(out, seed=0):
    """A small random filtered_*.csv set (AmazonGraphFilter format)."""
    rnd = random.Random(seed)
    os.makedirs(out, exist_ok=True)

    def w(name, header, rows):
        with open(os.path.join(out, name), 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(header)
            wr.writerows(rows)

    words = 'love war peace night day river stone fire king queen city song road home heart'.split()
    asins = ['B%09d' % i for i in range(120)]
    w('filtered_products.csv', ['ID', 'ASIN', 'Discontinued', 'Title', 'Group', 'SalesRank', 'AvgRating'],
      [(str(i), a, 'False', ' '.join(rnd.sample(words, 3)), rnd.choice(['Book', 'Music']), '1', '4')
       for i, a in enumerate(asins)])
    w('filtered_categories.csv', ['ID', 'Name'], [('1', 'Root'), ('2', 'Left'), ('3', 'Right')])
    w('filtered_category_edges.csv', ['ParentID', 'ChildID'], [('1', '2'), ('1', '3')])
    w('filtered_category_paths.csv', ['ASIN', 'No', 'Seq', 'CategoryID'],
      [r for a in asins for r in ((a, '0', '0', '1'), (a, '0', '1', rnd.choice('23')))])
    w('filtered_also_bought.csv', ['ASIN', 'AlsoASIN'], [(a, rnd.choice(asins)) for a in asins[:60]])
    users = ['A%05d' % u for u in range(80)]
    reviews = {(u, a) for u in users for a in rnd.sample(asins, 8)}
    w('filtered_reviews.csv', ['ASIN', 'Date', 'Customer', 'Rating', 'Votes', 'Helpful'],
      [(a, '2003-1-1', u, '5', '0', '0') for u, a in sorted(reviews)])
    w('filtered_users.csv', ['ID', 'Name'], [(u, u) for u in users])


def _load_amazon(tmp, generate=True, **flags):
    """utils.load_dataset on a small generated AMAZON graph (generate=False reloads it)."""
    import utils   # torch, torch_sparse, dgl
    if generate:
        _write_filtered(os.path.join(tmp, 'filtered'))
        subprocess.run([sys.executable, os.path.join(HGB, 'dat_generator.py'), '--input', os.path.join(tmp, 'filtered'),
                        '--output', os.path.join(tmp, 'AMAZON'), '--title-features', 'bow'],
                       check=True, stdout=subprocess.DEVNULL)
    args = argparse.Namespace(root=tmp, dataset='AMAZON', task='link_prediction', target_relation='U,P',
                              split_seed=1, num_neg_eval=20, edge_mask_ratio=0, **flags)
    g, adjs, *_, link_split = utils.load_dataset(args)
    return g, link_split


def _message_edges(g):
    """(user, product) pairs carried by U-P and by P-U."""
    u2p = set(zip(*[x.tolist() for x in g.edges(etype='U-P')]))
    p2u = set((u, p) for p, u in zip(*[x.tolist() for x in g.edges(etype='P-U')]))
    return u2p, p2u


def test_amazon_graph_uses_train_edges_only():
    """load_dataset must replace BOTH directions of U-P with the training edges, or
    val/test edges leak into the propagated features through the other direction."""
    with tempfile.TemporaryDirectory() as tmp:
        g, link_split = _load_amazon(tmp)
        train = {tuple(e) for e in link_split['train_edges'].tolist()}
        heldout = {tuple(e) for e in np.concatenate([link_split['val_edges'], link_split['test_edges']]).tolist()}
        assert heldout, 'fixture too small: no held-out edges'
        u2p, p2u = _message_edges(g)
        assert u2p == train and p2u == train, 'U-P message edges must be exactly the training edges'
        assert not (u2p & heldout) and not (p2u & heldout)



def test_amazon_same_type_loops():
    """P-P and C-C carry no self-loops unless --same-type-loops: meta-path models build paths
    from the plain relations, so the search's 'PPUP' must not contain 'PUP'."""
    with tempfile.TemporaryDirectory() as tmp:
        g, _ = _load_amazon(tmp)
        looped, _ = _load_amazon(tmp, generate=False, same_type_loops=True)
        for et, nt in (('P-P', 'P'), ('C-C', 'C')):
            plain = set(zip(*[x.tolist() for x in g.edges(etype=et)]))
            assert plain and all(a != b for a, b in plain), f'{et}: self-loop without --same-type-loops'
            loops = {(i, i) for i in range(g.num_nodes(nt))}
            assert set(zip(*[x.tolist() for x in looped.edges(etype=et)])) == plain | loops, et

def test_remove_target_edges():
    """--alpha-edges heldout-half: the alpha half must leave both directions of the graph
    (the split's int32 edge arrays included)."""
    from link_data import remove_target_edges
    with tempfile.TemporaryDirectory() as tmp:
        g, link_split = _load_amazon(tmp)
        tr = link_split['train_edges']
        held = tr[np.random.default_rng(0).permutation(len(tr))[len(tr) // 2:]]
        g = remove_target_edges(g, held.astype(np.int32), 'U,P')     # int32, as scipy gives on real data
        u2p, p2u = _message_edges(g)
        kept = {tuple(e) for e in tr.tolist()} - {tuple(e) for e in held.tolist()}
        assert u2p == kept and p2u == kept, 'both directions must lose exactly the removed edges'


def test_sampler_fold_ids():
    """--wstep-kfold: the sampler returns each sampled edge's fold and draws exactly the same edges
    and negatives as without fold ids, so --wstep-kfold 0 reproduces earlier runs."""
    import torch
    dl = FakeLoader()
    s = build_link_split(dl, 2, 0, seed=1, num_neg_eval=20)
    tr = s['train_edges']
    fold_of = np.random.default_rng(0).integers(0, 4, size=len(tr))
    plain = TrainEdgeSampler(tr, s['train_adj_src2dst'], seed=3)
    folded = TrainEdgeSampler(tr, s['train_adj_src2dst'], seed=3, fold_of=fold_of)
    index = {tuple(e): i for i, e in enumerate(tr.tolist())}       # train edges are unique
    for _ in range(20):
        a, b = plain.sample_batch(128), folded.sample_batch(128)
        assert len(a) == 3 and len(b) == 4
        assert all(torch.equal(x, y) for x, y in zip(a, b[:3])), 'fold ids must not change the random stream'
        assert b[3].tolist() == [int(fold_of[index[(u, p)]]) for u, p in zip(b[0].tolist(), b[1].tolist())]


def test_fold_user_path_feats():
    """--wstep-kfold: fold f's user-path features equal a propagation on the graph without fold f
    (both directions), only kept user-path keys come back in the requested dtype, and g is untouched."""
    import torch
    import utils
    from link_data import remove_target_edges
    with tempfile.TemporaryDirectory() as tmp:
        g, link_split = _load_amazon(tmp)
        tr = link_split['train_edges']
        fold_of = np.random.default_rng(0).integers(0, 3, size=len(tr))
        keep = lambda k: len(k) >= 2 and k[-1] == 'P'
        before = {nt: {k: v.clone() for k, v in g.nodes[nt].data.items()} for nt in g.ntypes}
        folds = utils.fold_user_path_feats(g, 'P', 3, 4, tr, fold_of, 'U,P', keep=keep, dtype=torch.float16)
        after = {nt: dict(g.nodes[nt].data.items()) for nt in g.ntypes}
        assert {nt: set(d) for nt, d in after.items()} == {nt: set(d) for nt, d in before.items()}, 'g gained/lost features'
        assert all(torch.equal(after[nt][k], v) for nt, d in before.items() for k, v in d.items()), 'g features changed'
        assert len(folds) == 3
        for f, ff in enumerate(folds):
            assert ff and all('U' in k and keep(k) and v.dtype == torch.float16 for k, v in ff.items())
            ref, _ = _load_amazon(tmp, generate=False)
            ref = utils.hg_propagate_feat_dgl(remove_target_edges(ref, tr[fold_of == f], 'U,P'), 'P', 3, 4)
            expected = {k for k in ref.nodes['P'].data.keys() if 'U' in k and keep(k)}
            assert set(ff) == expected, f
            for k, v in ff.items():
                assert torch.equal(v, ref.nodes['P'].data[k].half()), (f, k)
        assert _message_edges(g) == ({tuple(e) for e in tr.tolist()},) * 2, 'g lost edges'


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
