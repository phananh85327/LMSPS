"""
Link-prediction data preparation for LMSPS.

Builds a train/val/test split of a single target relation (e.g. Author-Paper for
ACM, eventually User-Product for AMAZON) for BPR-style link prediction, plus fixed
negative candidates for val/test ranking evaluation. Kept dataset-agnostic: callers
pass the two node-type ids to relate; the dataset-specific wiring (which adjacency
dict keys to patch before meta-path propagation) lives in train_search.py/main_path.py.
"""

import hashlib
import json
import os

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor


def get_relation_edges(dl, src_type, dst_type):
    """
    Return (src_local_id, dst_local_id) arrays for the relation between src_type and
    dst_type, regardless of which direction HGB's link.dat happened to store it in.
    IDs are local to each type (0-indexed within that type's node range).
    """
    src_shift = dl.nodes['shift'][src_type]
    dst_shift = dl.nodes['shift'][dst_type]
    for r, (h_type, t_type) in dl.links['meta'].items():
        if h_type == src_type and t_type == dst_type:
            coo = dl.links['data'][r].tocoo()
            return coo.row - src_shift, coo.col - dst_shift
        if h_type == dst_type and t_type == src_type:
            coo = dl.links['data'][r].tocoo()
            return coo.col - src_shift, coo.row - dst_shift
    raise ValueError(f'No relation found between node types {src_type} and {dst_type}')


def build_link_split(dl, src_type, dst_type, seed=0, val_ratio=0.1, test_ratio=0.1, num_neg_eval=99):
    """
    Splits the (src_type -> dst_type) relation into train/val/test edges for link
    prediction and prepares fixed negative candidates for val/test evaluation.

    Returns a dict:
      num_src, num_dst        : node counts for each type
      train_edges/val_edges/test_edges : [N, 2] local (src, dst) id arrays
      val_negs/test_negs       : [N, num_neg_eval] local dst ids, disjoint from ANY
                                  observed edge of that src (train+val+test), fixed by seed
      full_adj_src2dst         : scipy CSR (num_src, num_dst) of ALL edges. Used ONLY to
                                  build the fixed val/test eval negatives above (a "negative"
                                  that is secretly a positive would corrupt the ranking).
                                  Never feed it to the model and never use it to filter
                                  TRAINING negatives: rejecting a user's held-out test item
                                  as a training negative tells the trainer which items are
                                  in the test set (Rendle-style leak).
      train_adj_src2dst        : scipy CSR (num_src, num_dst) of TRAIN edges only --
                                  substitute this in place of the full adjacency before
                                  meta-path feature propagation so val/test edges can't
                                  leak into the features the model is scored on, and pass
                                  it (NOT full_adj) to TrainEdgeSampler
    """
    rng = np.random.default_rng(seed)

    num_src = dl.nodes['count'][src_type]
    num_dst = dl.nodes['count'][dst_type]

    src_ids, dst_ids = get_relation_edges(dl, src_type, dst_type)
    edges = np.unique(np.stack([src_ids, dst_ids], axis=1), axis=0)

    perm = rng.permutation(edges.shape[0])
    edges = edges[perm]

    n_total = edges.shape[0]
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train + n_val]
    test_edges = edges[n_train + n_val:]

    full_adj_src2dst = sp.csr_matrix(
        (np.ones(n_total, dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_src, num_dst))
    train_adj_src2dst = sp.csr_matrix(
        (np.ones(n_train, dtype=np.float32), (train_edges[:, 0], train_edges[:, 1])),
        shape=(num_src, num_dst))

    def sample_negs(pos_edges, k):
        # Vectorized rejection sampling: draw everything at once, then redraw only
        # entries that hit an observed edge of their src (checked in one scipy CSR
        # fancy-index call) or duplicate another negative in the same row. With
        # collision probability ~(degree + k) / num_dst this converges in 2-3
        # rounds; the per-edge Python loop it replaces took minutes at ~10^5 edges.
        n = pos_edges.shape[0]
        if n == 0:
            return np.empty((0, k), dtype=np.int64)
        src = pos_edges[:, 0]
        negs = rng.integers(0, num_dst, size=(n, k))
        row_idx = np.repeat(np.arange(n), k)
        while True:
            bad = (np.asarray(full_adj_src2dst[src[row_idx], negs.ravel()]).ravel() > 0).reshape(n, k)
            order = np.argsort(negs, axis=1)
            sorted_negs = np.take_along_axis(negs, order, axis=1)
            dup_sorted = np.zeros((n, k), dtype=bool)
            dup_sorted[:, 1:] = sorted_negs[:, 1:] == sorted_negs[:, :-1]
            dup = np.zeros((n, k), dtype=bool)
            np.put_along_axis(dup, order, dup_sorted, axis=1)
            bad |= dup
            n_bad = int(bad.sum())
            if n_bad == 0:
                return negs.astype(np.int64, copy=False)
            negs[bad] = rng.integers(0, num_dst, size=n_bad)

    val_negs = sample_negs(val_edges, num_neg_eval)
    test_negs = sample_negs(test_edges, num_neg_eval)

    return {
        'num_src': num_src,
        'num_dst': num_dst,
        'train_edges': train_edges,
        'val_edges': val_edges,
        'test_edges': test_edges,
        'val_negs': val_negs,
        'test_negs': test_negs,
        'full_adj_src2dst': full_adj_src2dst,
        'train_adj_src2dst': train_adj_src2dst,
    }


def edges_fingerprint(edges):
    """Order-free short hash of an edge set [N, 2]. Over the training edges it changes
    with the data and with the split, so search records can be told apart by it."""
    e = np.unique(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=0)
    return hashlib.blake2b(e.tobytes(), digest_size=8).hexdigest()


def data_provenance(data_dir, dl, link_split):
    """What a search ran on, for its JSON record: dat_generator.py's meta.json (how the
    data was built; None for data built before it existed), node counts, the number of
    target edges and a fingerprint of the training edges."""
    meta_file = os.path.join(data_dir, 'meta.json')
    meta = None
    if os.path.isfile(meta_file):
        with open(meta_file, encoding='utf-8') as f:
            meta = json.load(f)
    return {'dir': data_dir, 'meta': meta,
            'node_counts': {int(t): int(c) for t, c in dl.nodes['count'].items()},
            'num_target_edges': int(link_split['full_adj_src2dst'].nnz),
            'train_edges_hash': edges_fingerprint(link_split['train_edges'])}


def remove_target_edges(g, edges, target_relation):
    """Remove (src, dst) target edges from BOTH directions of the message-passing
    graph (etypes 'S-D' and 'D-S' for --target-relation "S,D"), so features
    propagated afterwards cannot see them."""
    s, d = target_relation.split(',')
    src, dst = torch.LongTensor(edges[:, 0]), torch.LongTensor(edges[:, 1])  # edge_ids needs int64
    g.remove_edges(g.edge_ids(src, dst, etype=f'{s}-{d}'), etype=f'{s}-{d}')
    g.remove_edges(g.edge_ids(dst, src, etype=f'{d}-{s}'), etype=f'{d}-{s}')
    return g


def csr_to_sparse_tensor(csr, transpose=False):
    """Convert a scipy CSR matrix to a torch_sparse.SparseTensor, matching the
    row=dst/col=src convention used by load_dataset's adjs dict."""
    coo = csr.tocoo()
    if transpose:
        row, col, sizes = coo.col, coo.row, (csr.shape[1], csr.shape[0])
    else:
        row, col, sizes = coo.row, coo.col, csr.shape
    return SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sizes)


class TrainEdgeSampler:
    """Yields (src, pos_dst, neg_dst) batches for BPR training. Negatives are
    resampled fresh every call (not cached) so the model sees varied negatives.

    exclude_adj_src2dst must be the TRAIN-ONLY adjacency: training negatives may
    only be rejected for colliding with TRAIN positives. Excluding val/test
    positives here would leak the held-out edges into training (the model would
    never be pushed to down-score them -- a signal unavailable in deployment)."""

    def __init__(self, train_edges, exclude_adj_src2dst, seed=0, fold_of=None):
        self.train_edges = train_edges
        self.exclude_adj_src2dst = exclude_adj_src2dst
        self.num_dst = exclude_adj_src2dst.shape[1]
        self.rng = np.random.default_rng(seed)
        # optional fold id per train edge (--wstep-kfold): sample_batch then also
        # returns each sampled edge's fold, drawing the same random numbers
        self.fold_of = fold_of

    def num_batches(self, batch_size):
        return (self.train_edges.shape[0] - 1) // batch_size + 1

    def sample_batch(self, batch_size):
        idx = self.rng.integers(0, self.train_edges.shape[0], size=batch_size)
        batch = self.train_edges[idx]
        src = batch[:, 0]
        pos_dst = batch[:, 1]
        # vectorized rejection sampling: one scipy CSR fancy-index call checks the
        # whole batch against the exclusion adjacency, then only offenders are
        # redrawn and rechecked (collision probability ~degree/num_dst per draw)
        neg_dst = self.rng.integers(0, self.num_dst, size=batch_size)
        pending = np.arange(batch_size)
        while pending.size:
            hits = np.asarray(self.exclude_adj_src2dst[src[pending], neg_dst[pending]]).ravel() > 0
            pending = pending[hits]
            if pending.size:
                neg_dst[pending] = self.rng.integers(0, self.num_dst, size=pending.size)
        if self.fold_of is not None:
            return (torch.LongTensor(src), torch.LongTensor(pos_dst), torch.LongTensor(neg_dst),
                    torch.LongTensor(self.fold_of[idx]))
        return torch.LongTensor(src), torch.LongTensor(pos_dst), torch.LongTensor(neg_dst)
