import os
import sys
import gc
import math
import random
import itertools

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.append('../data')
from data_loader import data_loader
from link_data import build_link_split, csr_to_sparse_tensor, TrainEdgeSampler
# import src.utils_lib

import warnings
warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

NODE_TYPE_IDS = {'ACM': {'P': 0, 'A': 1, 'C': 2, 'K': 3},
                 'AMAZON': {'P': 0, 'C': 1, 'U': 2}}


def project_op(keys, model, criterion, eval_loader, device, trainval_point, valtest_point, labels, repeat):  #
    ''' operation '''
    # num_ops = self.num_sampled
    # candidate_flags = model.candidate_flags[cell_type]
    # proj_crit = args.proj_crit[cell_type]
    compare = lambda x, y: x < y   ######################attention########################
    crit_extrema = None
    best_index = 0
    for opid in range(repeat):
        index_sampled = model.epoch_sample(0)
        crit = infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, labels)  #weights=weights
        print("loss {}\n".format(crit))
        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_index = index_sampled
    path = []
    label_path = []
    for i, index in enumerate(best_index):
        if index < len(keys):
            path.append(keys[index])
        # else:
        #     label_path.append((label_keys[index - len(keys)], i))
    # import code
    # code.interact(local=locals())
    return [path, label_path]


def infer_eval(model, criterion, eval_loader, device, index_sampled, trainval_point, valtest_point, labels):
    # objs = src.utils_lib.utils.AvgrageMeter()
    # top1 = src.utils_lib.utils.AvgrageMeter()
    model.eval()
    raw_preds = []
    meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in index_sampled]
    label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in index_sampled]
    with torch.no_grad():

        for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
            # # batch = batch.to(device)
            batch_feats = {k: x.to(device) for k, x in batch_feats.items() if
                           k in meta_path_sampled or (model.residual and k == model.tgt_key)}
            batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items() if k in label_meta_path_sampled}

            raw_preds.append(model(index_sampled, batch_feats, batch_labels_feats, meta_path_sampled,
                                   label_meta_path_sampled).cpu())  # id_paths
        # logits = model(data, weights_dict=weights_dict)
    raw_preds = torch.cat(raw_preds, dim=0)
    loss_train = criterion(raw_preds[:trainval_point], labels[:trainval_point]).item()
    loss_val = criterion(raw_preds[trainval_point:valtest_point], labels[trainval_point:valtest_point]).item()


    return loss_train + loss_val



def bpr_loss(pos_scores, neg_scores):
    """BPR pairwise ranking loss. pos_scores: [N]; neg_scores: [N] or [N, K]
    (broadcast against pos_scores when K sampled negatives per positive)."""
    if neg_scores.dim() > pos_scores.dim():
        pos_scores = pos_scores.unsqueeze(-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def score_edges(src_emb, dst_emb, src_idx, dst_idx, chunk_size=1024):
    """Dot-product score for (src_idx[i], dst_idx[i]) pairs. dst_idx may be
    [N] (one dst per src) or [N, K] (K candidates per src, e.g. sampled negatives).
    The [N, K] case is scored in row chunks: gathering dst_emb[dst_idx] in one
    shot materializes an [N, K, hidden] tensor -- ~53 GB for AMAZON's ~260k val
    edges x 99 negatives x 512 dims -- while chunks stay around 200 MB."""
    if dst_idx.dim() == 1:
        return (src_emb[src_idx] * dst_emb[dst_idx]).sum(-1)
    out = torch.empty(dst_idx.shape, dtype=src_emb.dtype, device=src_emb.device)
    for start in range(0, dst_idx.size(0), chunk_size):
        end = min(start + chunk_size, dst_idx.size(0))
        s = src_emb[src_idx[start:end]].unsqueeze(1)
        out[start:end] = (s * dst_emb[dst_idx[start:end]]).sum(-1)
    return out


def evaluate_ranking(pos_scores, neg_scores, ks=(5, 10, 20)):
    """pos_scores: [N], neg_scores: [N, K] sampled-negative scores per positive
    edge. Ranks the single positive among its K negatives and computes
    Precision@k/Recall@k/NDCG@k/Hit@k, averaged over all N edges. With exactly
    one relevant item per edge, Hit@k is the binary top-k indicator and
    Recall@k coincides with it; Precision@k = Hit@k / k."""
    num_candidates = neg_scores.size(1) + 1
    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    ranks = (all_scores >= all_scores[:, :1]).sum(dim=1).float()

    results = {}
    for k in ks:
        k = min(k, num_candidates)
        hit = (ranks <= k).float()
        ndcg = hit / torch.log2(ranks + 1)
        results[f'Precision@{k}'] = (hit / k).mean().item()
        results[f'Recall@{k}'] = hit.mean().item()
        results[f'NDCG@{k}'] = ndcg.mean().item()
        results[f'Hit@{k}'] = hit.mean().item()
    return results


def evaluate_ranking_by_user(pos_scores, neg_scores, test_edges, test_negs,
                             ks=(5, 10, 20)):
    """Per-USER ranking metrics: all test edges of a user are ranked together
    against the union of that user's sampled negatives, so the four metrics
    genuinely differ (unlike the per-edge protocol, where a single positive
    forces Recall@k == Hit@k and Precision@k == Hit@k / k).

    For user u with P_u test positives and candidate set C_u = positives +
    deduplicated negatives (negatives are sampled per edge, so the same item can
    appear for several of u's edges -- scored identically, kept once):
      Precision@k = |top-k ∩ P_u| / k
      Recall@k    = |top-k ∩ P_u| / |P_u|
      NDCG@k      = binary-gain DCG over top-k / IDCG(min(|P_u|, k))
      Hit@k       = 1 if any positive in top-k
    averaged over users (macro), so heavy users get ONE vote here versus one
    vote per edge in evaluate_ranking. Ties rank pessimistically (same rule as
    evaluate_ranking). test_edges: [N, 2] (src, dst); test_negs: [N, K] dst ids.
    """
    pos = pos_scores.detach().cpu().numpy().astype(np.float64)
    neg = neg_scores.detach().cpu().numpy().astype(np.float64)
    users = np.asarray(test_edges)[:, 0]
    pos_items = np.asarray(test_edges)[:, 1]
    neg_items = np.asarray(test_negs)

    order = np.argsort(users, kind='stable')
    users_s, pos_s, pitems_s = users[order], pos[order], pos_items[order]
    neg_s, nitems_s = neg[order], neg_items[order]
    starts = np.flatnonzero(np.r_[True, users_s[1:] != users_s[:-1]])
    bounds = np.r_[starts, users_s.size]

    idcg_table = np.cumsum(1.0 / np.log2(np.arange(2, max(ks) + 2)))
    sums = {k: np.zeros(4) for k in ks}  # precision, recall, ndcg, hit
    n_users = starts.size
    for i in range(n_users):
        lo, hi = bounds[i], bounds[i + 1]
        p_sc, p_it = pos_s[lo:hi], pitems_s[lo:hi]
        n_it = nitems_s[lo:hi].ravel()
        n_sc = neg_s[lo:hi].ravel()
        uniq, first = np.unique(n_it, return_index=True)
        keep = ~np.isin(uniq, p_it)
        cand = np.concatenate([p_sc, n_sc[first[keep]]])
        ranks = (cand[None, :] >= p_sc[:, None]).sum(1)  # >= counts self
        npos = hi - lo
        for k in ks:
            in_top = ranks <= k
            h = int(in_top.sum())
            dcg = (1.0 / np.log2(ranks[in_top] + 1)).sum()
            idcg = idcg_table[min(npos, k) - 1]
            sums[k] += (h / k, h / npos, dcg / idcg, 1.0 if h else 0.0)

    results = {'n_users': n_users}
    for k in ks:
        prec, rec, ndcg, hit = sums[k] / n_users
        results[f'Precision@{k}'] = prec
        results[f'Recall@{k}'] = rec
        results[f'NDCG@{k}'] = ndcg
        results[f'Hit@{k}'] = hit
    return results


def print_by_user(res):
    res = dict(res)
    n = res.pop('n_users')
    print(f'Per-user ranking metrics ({n} users, positives pooled per user):')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')


def evaluate_ranking_stratified(pos_scores, neg_scores, train_degrees,
                                buckets=((0, 0), (1, 5), (6, 20), (21, None)),
                                ks=(10,)):
    """Ranking metrics per USER-training-degree bucket (how many train
    interactions the user behind each test edge has). Shows WHERE a model wins:
    free-embedding / graph-aggregation user sides excel on well-observed users
    but have no signal for near-cold ones, while feature towers degrade more
    gracefully. train_degrees: [N] array aligned with pos_scores rows."""
    out = {}
    deg = np.asarray(train_degrees)
    for lo, hi in buckets:
        mask = (deg >= lo) if hi is None else ((deg >= lo) & (deg <= hi))
        if hi is None:
            name = f'deg>={lo}'
        elif lo == hi:
            name = f'deg={lo}'
        else:
            name = f'deg {lo}-{hi}'
        n = int(mask.sum())
        if n == 0:
            continue
        m = torch.from_numpy(mask).to(pos_scores.device)
        out[name] = {'n_edges': n,
                     **evaluate_ranking(pos_scores[m], neg_scores[m], ks=ks)}
    return out


def print_stratified(strat):
    """Compact per-bucket report (NDCG/Hit only -- Precision/Recall duplicate
    Hit in the 1-positive protocol)."""
    for name, res in strat.items():
        metrics = '  '.join(f'{k} {v:.4f}' for k, v in res.items()
                            if k.startswith(('NDCG', 'Hit')))
        print(f'  {name:<9} ({res["n_edges"]:>7} edges)  {metrics}')


def infer_full_embeddings_search(model, eval_loader, device, index_sampled, meta_path_sampled, label_meta_path_sampled,
                                  keep_on_device=False):
    """Compute full target-type-node embeddings with the LMSPS_Se (search-stage)
    model, reusing infer_eval's eval_loader node-batching pattern."""
    model.eval()
    embs = []
    with torch.no_grad():
        for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
            batch_feats = {k: x.to(device).float() for k, x in batch_feats.items() if
                           k in meta_path_sampled or (model.residual and k == model.tgt_key)}
            batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items() if k in label_meta_path_sampled}
            out = model(index_sampled, batch_feats, batch_labels_feats, meta_path_sampled,
                        label_meta_path_sampled)
            embs.append(out if keep_on_device else out.cpu())
    return torch.cat(embs, dim=0)


def infer_eval_lp(model, eval_loader, device, index_sampled, meta_path_sampled, label_meta_path_sampled,
                   tower_src_emb, val_edges, val_negs):
    """Search-stage link-prediction validation BPR loss -- used as project_op_lp's
    selection criterion, mirroring infer_eval's role for classification."""
    dst_emb = infer_full_embeddings_search(model, eval_loader, device, index_sampled,
                                            meta_path_sampled, label_meta_path_sampled,
                                            keep_on_device=True)
    src_idx = val_edges[:, 0]
    pos_dst_idx = val_edges[:, 1]
    pos_scores = score_edges(tower_src_emb, dst_emb, src_idx, pos_dst_idx)
    neg_scores = score_edges(tower_src_emb, dst_emb, src_idx, val_negs)
    return bpr_loss(pos_scores, neg_scores).item()


def infer_full_embeddings_final(model, eval_loader, device, keep_on_device=False):
    """Compute full target-type-node embeddings with the LMSPS (final-retrain)
    model, reusing the same node-batching pattern."""
    model.eval()
    embs = []
    with torch.no_grad():
        for batch, batch_feats, batch_labels_feats, batch_mask in eval_loader:
            batch_feats = {k: x.to(device).float() for k, x in batch_feats.items()}
            batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
            out = model(batch, batch_feats, batch_labels_feats, batch_mask)
            embs.append(out if keep_on_device else out.cpu())
    return torch.cat(embs, dim=0)


def project_op_lp(keys, model, eval_loader, device, tower_src_emb, val_edges, val_negs, repeat,
                  exhaustive_cap=0):
    """Link-prediction counterpart of project_op: score candidate path sets by
    validation BPR loss and keep the argmin (paper Eq. 6 / Algorithm 1 l.17-21).

    Default: `repeat` uniform draws of ns paths from the final top-num_final pool
    (upstream behaviour; with C(10,4)=210 subsets, 200 draws hit ~125 distinct
    sets and re-score the rest). If exhaustive_cap > 0 and C(pool, ns) <= cap,
    every subset is scored exactly once instead. Combine with
    model.eval_gumbel=False so a set's score does not depend on the Gumbel draw.

    Returns ([path, label_path], table) where table is a list of
    (sorted path-index list, val bpr loss) for every evaluation performed."""
    # one-time device residency: every repeat scores on the GPU, so keep the
    # user embeddings and eval edges there instead of re-transferring per repeat
    tower_src_emb = tower_src_emb.to(device)
    val_edges = val_edges.to(device)
    val_negs = val_negs.to(device)

    pool = sorted(int(i) for i in model.epoch_pool(0).cpu().numpy())
    n_sets = math.comb(len(pool), model.num_sampled)
    if exhaustive_cap and n_sets <= exhaustive_cap:
        candidates = [list(c) for c in itertools.combinations(pool, model.num_sampled)]
        print(f'sampling evaluation: exhaustive over all {n_sets} subsets of the final pool {pool}')
    else:
        candidates = None
        print(f'sampling evaluation: {repeat} uniform draws from the final pool {pool} '
              f'({n_sets} possible subsets)')

    compare = lambda x, y: x < y
    crit_extrema = None
    best_index = 0
    table = []
    n_evals = len(candidates) if candidates is not None else repeat
    for opid in range(n_evals):
        if candidates is not None:
            index_sampled = candidates[opid]
            print(f"sampled: {index_sampled}")
        else:
            index_sampled = model.epoch_sample(0)
        meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in index_sampled]
        label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats, model.num_paths) if i in index_sampled]
        crit = infer_eval_lp(model, eval_loader, device, index_sampled, meta_path_sampled, label_meta_path_sampled,
                              tower_src_emb, val_edges, val_negs)
        print("val bpr loss {}\n".format(crit))
        table.append(([int(i) for i in index_sampled], crit))
        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_index = index_sampled
    path = []
    label_path = []
    for i, index in enumerate(best_index):
        if index < len(keys):
            path.append(keys[index])
    return [path, label_path], table


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluator(gt, pred):
    gt = gt.cpu().squeeze()
    pred = pred.cpu().squeeze()
    return f1_score(gt, pred, average='micro'), f1_score(gt, pred, average='macro')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def hg_propagate_feat_dgl(g, tgt_type, num_hops, max_length, echo=False):
    for hop in range(1, max_length):
        #reserve_heads = [ele[:hop] for ele in extra_metapath if len(ele) > hop]

        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)
            # if hop == args.num_hops and dtype != tgt_type: continue
            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if (hop == num_hops and dtype != tgt_type ) \
                      or (hop > num_hops):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo:
            print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())
            print(f'------\n')

    return g


def hg_propagate_feat_dgl_path(g, tgt_type, num_hops, max_length, meta_path, echo=False):
    # A propagated key is built source-outward: pushing key k along stype->dtype
    # yields f'{dtype}{k}', so producing path X requires exactly X's suffixes as
    # intermediates ('PUPP' needs 'P' -> 'PP' -> 'UPP' -> 'PUPP'). Restricting to
    # those suffixes skips every other candidate path -- at AMAZON hops-5 scale
    # that is ~20 keys instead of 125+, cutting propagation time and peak RAM.
    # (Length-1 raw-feature keys are inputs, never produced, so never skipped.)
    wanted = set()
    for ele in meta_path:
        for i in range(len(ele) - 1):
            wanted.add(ele[i:])
    for hop in range(1, max_length):
        for etype in g.etypes:
            stype, _, dtype = g.to_canonical_etype(etype)

            for k in list(g.nodes[stype].data.keys()):
                if len(k) == hop:
                    current_dst_name = f'{dtype}{k}'
                    if current_dst_name not in wanted:
                        continue
                    if (hop == num_hops and dtype != tgt_type ) \
                      or (hop > num_hops):
                        continue
                    if echo: print(k, etype, current_dst_name)
                    g[etype].update_all(
                        fn.copy_u(k, 'm'),
                        fn.mean('m', current_dst_name), etype=etype)

        # remove no-use items
        for ntype in g.ntypes:

            if ntype == tgt_type: continue
            removes = []
            for k in g.nodes[ntype].data.keys():

                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo:
            print(f'-- hop={hop} ---')
            for ntype in g.ntypes:
                for k, v in g.nodes[ntype].data.items():
                    print(f'{ntype} {k} {v.shape}', v[:,-1].max(), v[:,-1].mean())

        if echo: print(f'------\n')

    return g


def hg_propagate_sparse_pyg(adjs, tgt_types, num_hops, max_length, extra_metapath, prop_feats=False, echo=False, prop_device='cpu'):
    store_device = 'cpu'
    if type(tgt_types) is not list:
        tgt_types = [tgt_types]

    label_feats = {k: v.clone() for k, v in adjs.items() if prop_feats or k[-1] in tgt_types} # metapath should start with target type in label propagation
    adjs_g = {k: v.to(prop_device) for k, v in adjs.items()}

    for hop in range(2, max_length):
        reserve_heads = [ele[-(hop+1):] for ele in extra_metapath if len(ele) > hop]
        new_adjs = {}
        for rtype_r, adj_r in label_feats.items():
            metapath_types = list(rtype_r)
            if len(metapath_types) == hop:
                dtype_r, stype_r = metapath_types[0], metapath_types[-1]
                for rtype_l, adj_l in adjs_g.items():
                    dtype_l, stype_l = rtype_l
                    if stype_l == dtype_r:
                        name = f'{dtype_l}{rtype_r}'
                        if (hop == num_hops and dtype_l not in tgt_types and name not in reserve_heads) \
                          or (hop > num_hops and name not in reserve_heads):
                            continue
                        if name not in new_adjs:
                            if echo: print('Generating ...', name)
                            if prop_device == 'cpu':
                                new_adjs[name] = adj_l.matmul(adj_r)
                            else:
                                with torch.no_grad():
                                    new_adjs[name] = adj_l.matmul(adj_r.to(prop_device)).to(store_device)
                        else:
                            if echo: print(f'Warning: {name} already exists')
        label_feats.update(new_adjs)

        removes = []
        for k in label_feats.keys():
            metapath_types = list(k)
            if metapath_types[0] in tgt_types: continue  # metapath should end with target type in label propagation
            if len(metapath_types) <= hop:
                removes.append(k)
        for k in removes:
            label_feats.pop(k)
        if echo and len(removes): print('remove', removes)
        del new_adjs
        gc.collect()

    if prop_device != 'cpu':
        del adjs_g
        torch.cuda.empty_cache()

    return label_feats


def check_acc(preds_dict, condition, init_labels, train_nid, val_nid, test_nid, show_test=True, loss_type='ce'):
    mask_train, mask_val, mask_test = [], [], []
    remove_label_keys = []
    k = list(preds_dict.keys())[0]
    v = preds_dict[k]
    if loss_type == 'ce':
        na, nb, nc = len(train_nid), len(val_nid), len(test_nid)
    elif loss_type == 'bce':
        na, nb, nc = len(train_nid) * v.size(1), len(val_nid) * v.size(1), len(test_nid) * v.size(1)

    for k, v in preds_dict.items():
        if loss_type == 'ce':
            pred = v.argmax(1)
        elif loss_type == 'bce':
            pred = (v > 0).int()

        a, b, c = pred[train_nid] == init_labels[train_nid], \
                  pred[val_nid] == init_labels[val_nid], \
                  pred[test_nid] == init_labels[test_nid]
        ra, rb, rc = a.sum() / na, b.sum() / nb, c.sum() / nc

        if loss_type == 'ce':
            vv = torch.log(v / (v.sum(1, keepdim=True) + 1e-6) + 1e-6)
            la, lb, lc = F.nll_loss(vv[train_nid], init_labels[train_nid]), \
                         F.nll_loss(vv[val_nid], init_labels[val_nid]), \
                         F.nll_loss(vv[test_nid], init_labels[test_nid])
        else:
            vv = (v / 2. + 0.5).clamp(1e-6, 1-1e-6)
            la, lb, lc = F.binary_cross_entropy(vv[train_nid], init_labels[train_nid].float()), \
                         F.binary_cross_entropy(vv[val_nid], init_labels[val_nid].float()), \
                         F.binary_cross_entropy(vv[test_nid], init_labels[test_nid].float())
        if condition(ra, rb, rc, k):
            mask_train.append(a)
            mask_val.append(b)
            mask_test.append(c)
        else:
            remove_label_keys.append(k)
        # if show_test:
        #     print(k, ra, rb, rc, la, lb, lc, (ra/rb-1)*100, (ra/rc-1)*100, (1-la/lb)*100, (1-la/lc)*100)
        # else:
        #     print(k, ra, rb, la, lb, (ra/rb-1)*100, (1-la/lb)*100)
    print(set(list(preds_dict.keys())) - set(remove_label_keys))

    print((torch.stack(mask_train, dim=0).sum(0) > 0).sum() / na)
    print((torch.stack(mask_val, dim=0).sum(0) > 0).sum() / nb)
    if show_test:
        print((torch.stack(mask_test, dim=0).sum(0) > 0).sum() / nc)


def train_multi_stage(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, enhance_loader, evaluator, predict_prob, gama, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    loss_l1, loss_l2 = 0., 0.
    iter_num = 0
    y_true, y_pred = [], []

    for idx_1, idx_2 in zip(train_loader, enhance_loader):
        idx = torch.cat((idx_1, idx_2), dim=0)
        L1_ratio = len(idx_1) * 1.0 / (len(idx_1) + len(idx_2))
        L2_ratio = len(idx_2) * 1.0 / (len(idx_1) + len(idx_2))

        if isinstance(feats, list):
            batch_feats = [x[idx].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[idx].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[idx].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[idx].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[idx_1]
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            extra_weight = 2 * torch.abs(predict_prob[idx_2] - 0.5)
            extra_y = (predict_prob[idx_2] > 0.5).float()
        else:
            extra_weight, extra_y = predict_prob[idx_2].max(dim=1)
        extra_weight = extra_weight.to(device)
        extra_y = extra_y.to(device)

        # teacher_soft = predict_prob[idx_2].to(device)
        # teacher_conf = torch.max(teacher_soft, dim=1, keepdim=True)[0]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(None, batch_feats, batch_labels_feats, batch_mask)
                L1 = loss_fcn(output_att[:len(idx_1)], batch_y)
                if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
                    L2 = F.binary_cross_entropy_with_logits(output_att[len(idx_1):], extra_y, reduction='none')
                else:
                    L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
                L2 = (L2 * extra_weight).sum() / len(idx_2)
                loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(None, batch_feats, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att[:len(idx_1)], batch_y)
            if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
                L2 = F.binary_cross_entropy_with_logits(output_att[len(idx_1):], extra_y, reduction='none')
            else:
                L2 = F.cross_entropy(output_att[len(idx_1):], extra_y, reduction='none')
            L2 = (L2 * extra_weight).sum() / len(idx_2)
            loss_train = L1_ratio * L1 + gama * L2_ratio * L2
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att[:len(idx_1)].data.cpu() > 0.).int())
        else:
            y_pred.append(output_att[:len(idx_1)].argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        loss_l1 += L1.item()
        loss_l2 += L2.item()
        iter_num += 1

    print(loss_l1 / iter_num, loss_l2 / iter_num)
    loss = total_loss / iter_num
    approx_acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, approx_acc


def train(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        ## batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc



def train_search(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, epoch_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader)).to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, epoch_sampled, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, epoch_sampled, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(val_batch, val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_mask)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward()
            scalar.step(optimizer_a)
            scalar.update()
        else:
            val_output_att = model(val_batch, val_batch_feats, epoch_sampled, val_batch_labels_feats, val_batch_mask)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val


def train_search_new(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, epoch_sampled, meta_path_sampled, label_meta_path_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader))
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            # import code
            # code.interact(local=locals())
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(epoch_sampled, batch_feats, batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(epoch_sampled, batch_feats, batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
                val_loss_train = loss_fcn(val_output_att, val_batch_y)
            scalar.scale(val_loss_train).backward()
            scalar.step(optimizer_a)
            scalar.update()
        else:
            val_output_att = model(epoch_sampled, val_batch_feats, val_batch_labels_feats, meta_path_sampled, label_meta_path_sampled)
            L1 = loss_fcn(val_output_att, val_batch_y)
            val_loss_train = L1
            val_loss_train.backward()
            optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val





def train_search_lp(model, tower, feats, label_feats, src_attr, epoch_sampled, meta_path_sampled,
                     label_meta_path_sampled, optimizer_w, optimizer_a, train_sampler, val_sampler,
                     batch_size, device, scalar=None):
    """Link-prediction counterpart of train_search_new: same bi-level (DARTS)
    optimizer_w/optimizer_a structure, but batches are edge triplets
    (src, pos_dst, neg_dst) instead of node indices, and the loss is BPR over
    dot-product scores instead of a classification criterion."""
    model.train()
    tower.train()
    total_loss = 0.
    val_total_loss = 0.
    iter_num = train_sampler.num_batches(batch_size)
    src_attr_dev = src_attr.to(device)

    def run(dst_ids):
        batch_feats = {k: x[dst_ids].to(device).float() for k, x in feats.items()
                       if k in meta_path_sampled or (model.residual and k == model.tgt_key)}
        batch_label_feats = {k: x[dst_ids].to(device) for k, x in label_feats.items()
                              if k in label_meta_path_sampled}
        return model(epoch_sampled, batch_feats, batch_label_feats, meta_path_sampled, label_meta_path_sampled)

    for _ in range(iter_num):
        src, pos_dst, neg_dst = train_sampler.sample_batch(batch_size)
        val_src, val_pos_dst, val_neg_dst = val_sampler.sample_batch(batch_size)

        ########################################train
        optimizer_w.zero_grad()
        dst_emb = run(torch.cat([pos_dst, neg_dst]))
        pos_emb, neg_emb = dst_emb[:len(pos_dst)], dst_emb[len(pos_dst):]
        src_emb = tower(src_attr_dev[src.to(device)])
        loss_train = bpr_loss((src_emb * pos_emb).sum(-1), (src_emb * neg_emb).sum(-1))
        loss_train.backward()
        optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        val_dst_emb = run(torch.cat([val_pos_dst, val_neg_dst]))
        val_pos_emb, val_neg_emb = val_dst_emb[:len(val_pos_dst)], val_dst_emb[len(val_pos_dst):]
        val_src_emb = tower(src_attr_dev[val_src.to(device)])
        val_loss_train = bpr_loss((val_src_emb * val_pos_emb).sum(-1), (val_src_emb * val_neg_emb).sum(-1))
        val_loss_train.backward()
        optimizer_a.step()

        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()

    return total_loss / iter_num, val_total_loss / iter_num


def train_lp(model, tower, feats, label_feats, src_attr, optimizer, train_sampler, batch_size, device, scalar=None):
    """Link-prediction counterpart of train (final-retrain stage): edge-triplet
    batches and BPR loss instead of node-index batches and a classification
    criterion."""
    model.train()
    tower.train()
    total_loss = 0.
    iter_num = train_sampler.num_batches(batch_size)
    src_attr_dev = src_attr.to(device)

    for _ in range(iter_num):
        src, pos_dst, neg_dst = train_sampler.sample_batch(batch_size)
        dst_ids = torch.cat([pos_dst, neg_dst])
        batch_feats = {k: x[dst_ids].to(device).float() for k, x in feats.items()}
        batch_label_feats = {k: x[dst_ids].to(device) for k, x in label_feats.items()}

        optimizer.zero_grad()
        dst_emb = model(dst_ids, batch_feats, batch_label_feats)
        pos_emb, neg_emb = dst_emb[:len(pos_dst)], dst_emb[len(pos_dst):]
        src_emb = tower(src_attr_dev[src.to(device)])
        loss_train = bpr_loss((src_emb * pos_emb).sum(-1), (src_emb * neg_emb).sum(-1))
        loss_train.backward()
        optimizer.step()

        total_loss += loss_train.item()

    return total_loss / iter_num


def train_search_two(model, feats, label_feats, labels_cuda, loss_fcn, optimizer_w, optimizer_a, train_loader, val_loader, meta_sampled, label_sampled, evaluator, mask=None, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []
    val_total_loss = 0
    val_y_true, val_y_pred = [], []
    ###################  optimize w  ##################
    for batch in train_loader:
        # batch = batch.to(device)
        val_batch = next(iter(val_loader))
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
            val_batch_feats = [x[val_batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
            val_batch_feats = {k: x[val_batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        val_batch_labels_feats = {k: x[val_batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
            val_batch_mask = {k: x[val_batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
            val_batch_mask = None
        batch_y = labels_cuda[batch]
        val_batch_y = labels_cuda[val_batch]

        ########################################train
        optimizer_w.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch, batch_feats, meta_sampled, label_sampled, batch_labels_feats, batch_mask)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward(retain_graph=True)
            scalar.step(optimizer_w)
            scalar.update()
        else:
            output_att = model(batch, batch_feats, meta_sampled, label_sampled, batch_labels_feats, batch_mask)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward(retain_graph=True)
            optimizer_w.step()

        ########################################val  update a
        optimizer_a.zero_grad()
        # if scalar is not None:
        #     with torch.cuda.amp.autocast():
        #         val_output_att = model(val_batch, val_batch_feats, val_batch_labels_feats, val_batch_mask)
        #         loss_train = loss_fcn(val_output_att, val_batch_y)
        #     scalar.scale(loss_train).backward()
        #     scalar.step(optimizer_a)
        #     scalar.update()
        # else:
        val_output_att = model(val_batch, val_batch_feats, meta_sampled, label_sampled, val_batch_labels_feats, val_batch_mask)
        L1 = loss_fcn(val_output_att, val_batch_y)
        val_loss_train = L1
        val_loss_train.backward()
        optimizer_a.step()

        ########################################
        y_true.append(batch_y.cpu().to(torch.long))
        val_y_true.append(val_batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
            val_y_pred.append((val_output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            val_y_pred.append(val_output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        val_total_loss += val_loss_train.item()
        iter_num += 1


    loss_train = total_loss / iter_num
    acc_train = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    loss_val = val_total_loss / iter_num
    acc_val = evaluator(torch.cat(val_y_true, dim=0), torch.cat(val_y_pred, dim=0))

    return loss_train, loss_val, acc_train, acc_val



def train_flag(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, step_size, m, mask=None, scalar=None):

    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    # step_size = 1e-2
    # m = 3
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)
        if isinstance(feats, list):
            batch_feats = [x[batch].to(device) for x in feats]
        elif isinstance(feats, dict):
            batch_feats = {k: x[batch].to(device) for k, x in feats.items()}
        else:
            assert 0
        batch_labels_feats = {k: x[batch].to(device) for k, x in label_feats.items()}
        if mask is not None:
            batch_mask = {k: x[batch].to(device) for k, x in mask.items()}
        else:
            batch_mask = None
        batch_y = labels_cuda[batch]


        def forward(k, perturb):

            #perturb_feats = {k: x+perturb[k] for k, x in batch_feats.items()}
            batch_feats[k] = batch_feats[k] + perturb

            #print (batch_feats[k][0][0:10])
            return model(batch, batch_feats, batch_labels_feats, batch_mask)
        model_forward = (model,forward)
        feats_shape = {k: x.shape for k, x in batch_feats.items()}
        if scalar is not None:
            with torch.cuda.amp.autocast():
                loss_train, output_att = flag(model_forward, feats_shape, batch_y, step_size, m, optimizer, device, F.nll_loss, scalar)
        else:
            loss, output_att = flag(model_forward, batch_feats.shape, batch_y, step_size, m, optimizer, device, F.nll_loss)


        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc






def train_2l(model, feats, label_feats, labels_cuda, loss_fcn, optimizer, train_loader, evaluator, tgt_type, scalar=None):
    model.train()
    device = labels_cuda.device
    total_loss = 0
    iter_num = 0
    y_true, y_pred = [], []

    for batch in train_loader:
        # batch = batch.to(device)

        layer2_feats = {k: x[batch] for k, x in feats.items() if k[0] == tgt_type}
        batch_labels_feats = {k: x[batch] for k, x in label_feats.items()}

        involved_keys = {}
        for k, v in layer2_feats.items():
            src = k[-1]
            if src not in involved_keys:
                involved_keys[src] = []
            involved_keys[src].append(torch.unique(v.storage.col()))
        involved_keys = {k: torch.unique(torch.cat(v)) for k, v in involved_keys.items()}

        for k, v in layer2_feats.items():
            src = k[-1]
            old_nnz = v.nnz()
            layer2_feats[k] = v[:, involved_keys[src]]
            assert layer2_feats[k].nnz() == old_nnz

        layer1_feats = {k: v[involved_keys[k[0]]] for k, v in feats.items() if k[0] in involved_keys}

        batch1 = {k: v.to(device) for k,v in involved_keys.items()}
        layer1_feats = {k: v.to(device) for k,v in layer1_feats.items()}
        batch2 = batch.to(device)
        layer2_feats = {k: v.to(device) for k,v in layer2_feats.items()}
        batch_labels_feats = {k: x.to(device) for k, x in batch_labels_feats.items()}
        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(layer1_feats, batch1, layer2_feats, batch2, batch_labels_feats)
            L1 = loss_fcn(output_att, batch_y)
            loss_train = L1
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        if isinstance(loss_fcn, nn.BCEWithLogitsLoss):
            y_pred.append((output_att.data.cpu() > 0.).int())
        else:
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    return loss, acc

def edge_mask(etypes,adjs,edge_mask_ratio):
    from collections import Counter
    etypes_ditc ={}

    generator = torch.Generator().manual_seed(1)

    for etype, adj in zip(etypes, adjs): 
        etypes_ditc[etype] = [adj,False]

    for key in etypes: 
        if not etypes_ditc[key][1]:
            etypes_ditc[key][1] = True
            row = etypes_ditc[key][0].storage._row
            col = etypes_ditc[key][0].storage._col

            row_list,col_list = row.cpu().tolist(),col.cpu().tolist()
            row_Counter = Counter(row_list)
            col_Counter = Counter(col_list)
            row_Counter = set([key for key in row_Counter.keys() if row_Counter[key] == 1])
            col_Counter = set([key for key in col_Counter.keys() if col_Counter[key] == 1])

            single_edge = torch.zeros(row.shape) == 0.0
            for index in range(len(row_list)):
                if row_list[index] in row_Counter or col_list[index] in col_Counter:
                    single_edge[index] = False

            edge_mask = torch.zeros(row.shape) != 0.0
            mid_mask = torch.rand((single_edge.sum().item()), generator=generator) < edge_mask_ratio
            edge_mask[single_edge] = mid_mask

            edge_mask = ~edge_mask

            row = row[edge_mask]
            col = col[edge_mask]

            etypes_ditc[key][0].storage._row,etypes_ditc[key][0].storage._col = row,col
            new_key = (key[2],key[1][::-1],key[0])
            if new_key in etypes_ditc and not etypes_ditc[new_key][1]:
                etypes_ditc[new_key][1] = True
                etypes_ditc[new_key][0].storage._row,etypes_ditc[new_key][0].storage._col = col,row


def load_dataset(args):
    dl = data_loader(f'{args.root}/{args.dataset}')

    link_split = None
    if getattr(args, 'task', 'classification') == 'link_prediction':
        src_name, dst_name = args.target_relation.split(',')
        type_ids = NODE_TYPE_IDS[args.dataset]
        assert src_name in type_ids and dst_name in type_ids, (
            f'--target-relation "{args.target_relation}" names unknown node types for '
            f'{args.dataset} (valid: {sorted(type_ids)}; e.g. ACM "A,P", AMAZON "U,P")')
        # split seed is fixed independently of --seeds so the search stage and the
        # final retrain (and every repeat seed) share the same train/val/test edges
        link_split = build_link_split(dl, type_ids[src_name], type_ids[dst_name],
                                       seed=args.split_seed, num_neg_eval=args.num_neg_eval)

    edge_mask_ratio = args.edge_mask_ratio

    # use one-hot index vectors for nods with no attributes
    # === feats ===
    features_list = []
    for i in range(len(dl.nodes['count'])):
        th = dl.nodes['attr'][i]
        if th is None:
            features_list.append(torch.eye(dl.nodes['count'][i]))
        else:
            features_list.append(torch.FloatTensor(th))

    idx_shift = np.zeros(len(dl.nodes['count'])+1, dtype=np.int32)
    for i in range(len(dl.nodes['count'])):
        idx_shift[i+1] = idx_shift[i] + dl.nodes['count'][i]

    # === labels ===
    num_classes = dl.labels_train['num_classes']
    init_labels = np.zeros((dl.nodes['count'][0], num_classes), dtype=int)

    val_ratio = 0.2
    train_nid = np.nonzero(dl.labels_train['mask'])[0]
    np.random.shuffle(train_nid)
    split = int(train_nid.shape[0]*val_ratio)
    val_nid = train_nid[:split]
    train_nid = train_nid[split:]
    train_nid = np.sort(train_nid)
    val_nid = np.sort(val_nid)
    test_nid = np.nonzero(dl.labels_test['mask'])[0]
    test_nid_full = np.nonzero(dl.labels_test_full['mask'])[0]

    init_labels[train_nid] = dl.labels_train['data'][train_nid]
    init_labels[val_nid] = dl.labels_train['data'][val_nid]
    init_labels[test_nid] = dl.labels_test['data'][test_nid]
    if args.dataset != 'IMDB':
        init_labels = init_labels.argmax(axis=1)

    print(len(train_nid), len(val_nid), len(test_nid), len(test_nid_full))
    init_labels = torch.LongTensor(init_labels)

    # === adjs ===
    # print(dl.nodes['attr'])
    # for k, v in dl.nodes['attr'].items():
    #     if v is None: print('none')
    #     else: print(v.shape)
    adjs = [] if args.dataset != 'Freebase' else {}
    for i, (k, v) in enumerate(dl.links['data'].items()):
        v = v.tocoo()
        src_type_idx = np.where(idx_shift > v.col[0])[0][0] - 1
        dst_type_idx = np.where(idx_shift > v.row[0])[0][0] - 1
        row = v.row - idx_shift[dst_type_idx]
        col = v.col - idx_shift[src_type_idx]
        sparse_sizes = (dl.nodes['count'][dst_type_idx], dl.nodes['count'][src_type_idx])
        adj = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), sparse_sizes=sparse_sizes)
        if args.dataset == 'Freebase':
            name = f'{dst_type_idx}{src_type_idx}'
            assert name not in adjs
            adjs[name] = adj
        else:
            adjs.append(adj)
            #print(adj)


    if args.dataset == 'DBLP':
        # A* --- P --- T
        #        |
        #        V
        # author: [4057, 334]
        # paper : [14328, 4231]
        # term  : [7723, 50]
        # venue(conference) : None
        A, P, T, V = features_list
        AP, PA, PT, PV, TP, VP = adjs

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-A', 'A'),
            ('A', 'A-P', 'P'),
            ('T', 'T-P', 'P'),
            ('V', 'V-P', 'P'),
            ('P', 'P-T', 'T'),
            ('P', 'P-V', 'V'),
        ]
        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)


                    
        
        
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        # for i, etype in enumerate(g.etypes):
        #     src, dst, eid = g._graph.edges(i)
        #     adj = SparseTensor(row=dst.long(), col=src.long())
        #     print(etype, adj)

        # g.ndata['feat']['A'] = A # not work
        g.nodes['A'].data['A'] = A
        g.nodes['P'].data['P'] = P
        g.nodes['T'].data['T'] = T
        g.nodes['V'].data['V'] = V
    elif args.dataset == 'IMDB':
        # A --- M* --- D
        #       |
        #       K
        # movie    : [4932, 3489]
        # director : [2393, 3341]
        # actor    : [6124, 3341]
        # keywords : None
        M, D, A, K = features_list
        MD, DM, MA, AM, MK, KM = adjs
        assert torch.all(DM.storage.col() == MD.t().storage.col())
        assert torch.all(AM.storage.col() == MA.t().storage.col())
        assert torch.all(KM.storage.col() == MK.t().storage.col())

        assert torch.all(MD.storage.rowcount() == 1) # each movie has single director

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('D', 'D-M', 'M'),
            ('M', 'M-D', 'D'),
            ('A', 'A-M', 'M'),
            ('M', 'M-A', 'A'),
            ('K', 'K-M', 'M'),
            ('M', 'M-K', 'K'),
        ]
        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)
        g = dgl.heterograph(new_edges)

        g.nodes['M'].data['M'] = M
        g.nodes['D'].data['D'] = D
        g.nodes['A'].data['A'] = A
        if args.num_hops > 2 :#or args.two_layer:
            g.nodes['K'].data['K'] = K
    elif args.dataset == 'ACM':
        # A --- P* --- C
        #       |
        #       K
        # paper     : [3025, 1902]
        # author    : [5959, 1902]
        # conference: [56, 1902]
        # field     : None
        P, A, C, K = features_list
        PP, PP_r, PA, AP, PC, CP, PK, KP = adjs
        row, col = torch.where(P)
        assert torch.all(row == PK.storage.row()) and torch.all(col == PK.storage.col())
        assert torch.all(AP.matmul(PK).to_dense() == A)
        assert torch.all(CP.matmul(PK).to_dense() == C)

        assert torch.all(PA.storage.col() == AP.t().storage.col())
        assert torch.all(PC.storage.col() == CP.t().storage.col())
        assert torch.all(PK.storage.col() == KP.t().storage.col())

        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()
        adjs = [PP] + adjs[2:]

        if link_split is not None:
            PA = csr_to_sparse_tensor(link_split['train_adj_src2dst'], transpose=True)
            AP = csr_to_sparse_tensor(link_split['train_adj_src2dst'], transpose=False)
            adjs[1] = PA
            adjs[2] = AP

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('A', 'A-P', 'P'),
            ('P', 'P-A', 'A'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
        ]

        if edge_mask_ratio != 0:
            edge_mask(etypes,adjs,edge_mask_ratio)
        if args.ACM_keep_F:
            etypes += [
                ('K', 'K-P', 'P'),
                ('P', 'P-K', 'K'),
            ]
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        # dgl.heterograph infers node counts from the max id in the edge lists;
        # with the train-only link-prediction split some nodes have no edge left,
        # so pass explicit counts to keep all nodes (and their features) present.
        num_nodes_dict = {'P': dl.nodes['count'][0], 'A': dl.nodes['count'][1], 'C': dl.nodes['count'][2]}
        if args.ACM_keep_F:
            num_nodes_dict['K'] = dl.nodes['count'][3]
        g = dgl.heterograph(new_edges, num_nodes_dict=num_nodes_dict)

        g.nodes['P'].data['P'] = P # [3025, 1902]
        g.nodes['A'].data['A'] = A # [5959, 1902]
        g.nodes['C'].data['C'] = C # [56, 1902]
        if args.ACM_keep_F:
            g.nodes['K'].data['K'] = K # [1902, 1902]
    elif args.dataset == 'AMAZON':
        # U --- P* --- C     (P-P also-bought, C-C category hierarchy)
        # product : title-BoW + group one-hot
        # category: name-BoW
        # user    : hashing-trick sparse binary code
        P, C, U = features_list
        PP, PP_r, CC, CC_r, PC, CP, UP, PU = adjs

        # merge both directions of each same-type relation into ONE symmetric
        # adjacency (as ACM does for PP): hg_propagate_feat_dgl names propagated
        # tensors f'{dtype}{src_key}', so two parallel same-type etypes would
        # write the same key and the second update_all would silently win.
        row0, col0, _ = PP.coo()
        row1, col1, _ = PP_r.coo()
        PP = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=PP.sparse_sizes())
        PP = PP.coalesce()
        PP = PP.set_diag()

        row0, col0, _ = CC.coo()
        row1, col1, _ = CC_r.coo()
        CC = SparseTensor(row=torch.cat((row0, row1)), col=torch.cat((col0, col1)), sparse_sizes=CC.sparse_sizes())
        CC = CC.coalesce()
        CC = CC.set_diag()

        if link_split is not None:
            # swap BOTH directions of the target relation for the train-only
            # adjacency, or val/test edges leak back in via the reverse relation
            UP = csr_to_sparse_tensor(link_split['train_adj_src2dst'], transpose=False)
            PU = csr_to_sparse_tensor(link_split['train_adj_src2dst'], transpose=True)

        adjs = [PP, CC, PC, CP, UP, PU]

        new_edges = {}
        ntypes = set()
        etypes = [ # src->tgt
            ('P', 'P-P', 'P'),
            ('C', 'C-C', 'C'),
            ('C', 'C-P', 'P'),
            ('P', 'P-C', 'C'),
            ('P', 'P-U', 'U'),
            ('U', 'U-P', 'P'),
        ]

        if edge_mask_ratio != 0:
            edge_mask(etypes, adjs, edge_mask_ratio)
        for etype, adj in zip(etypes, adjs):
            stype, rtype, dtype = etype
            dst, src, _ = adj.coo()
            src = src.numpy()
            dst = dst.numpy()
            new_edges[(stype, rtype, dtype)] = (src, dst)
            ntypes.add(stype)
            ntypes.add(dtype)

        # explicit counts: the train-only link-prediction split can leave nodes
        # edgeless, which would shrink dgl's inferred per-type node counts
        num_nodes_dict = {'P': dl.nodes['count'][0], 'C': dl.nodes['count'][1], 'U': dl.nodes['count'][2]}
        g = dgl.heterograph(new_edges, num_nodes_dict=num_nodes_dict)

        g.nodes['P'].data['P'] = P
        g.nodes['C'].data['C'] = C
        g.nodes['U'].data['U'] = U
    elif args.dataset == 'Freebase':
        # 0*: 40402  2/4/7 <-- 0 <-- 0/1/3/5/6
        #  1: 19427  all <-- 1
        #  2: 82351  4/6/7 <-- 2 <-- 0/1/2/3/5
        #  3: 1025   0/2/4/6/7 <-- 3 <-- 1/3/5
        #  4: 17641  4 <-- all
        #  5: 9368   0/2/3/4/6/7 <-- 5 <-- 1/5
        #  6: 2731   0/4 <-- 6 <-- 1/2/3/5/6/7
        #  7: 7153   4/6 <-- 7 <-- 0/1/2/3/5/7
        for i in range(8):
            kk = str(i)
            print(f'==={kk}===')
            for k, v in adjs.items():
                t, s = k
                assert s == t or f'{s}{t}' not in adjs
                if s == kk or t == kk:
                    if s == t:
                        print(k, v.sizes(), v.nnz(),
                              f'symmetric {v.is_symmetric()}; selfloop-ratio: {v.get_diag().sum()}/{v.size(0)}')
                    else:
                        print(k, v.sizes(), v.nnz())

        adjs['00'] = adjs['00'].to_symmetric()
        g = None
    else:
        assert 0

    if args.dataset == 'DBLP':
        adjs = {'AP': AP, 'PA': PA, 'PT': PT, 'PV': PV, 'TP': TP, 'VP': VP}
    elif args.dataset == 'ACM':
        adjs = {'PP': PP, 'PA': PA, 'AP': AP, 'PC': PC, 'CP': CP}
    elif args.dataset == 'AMAZON':
        adjs = {'PP': PP, 'CC': CC, 'PC': PC, 'CP': CP, 'UP': UP, 'PU': PU}
    elif args.dataset == 'IMDB':
        adjs = {'MD': MD, 'DM': DM, 'MA': MA, 'AM': AM, 'MK': MK, 'KM': KM}
    elif args.dataset == 'Freebase':
        new_adjs = {}
        for rtype, adj in adjs.items():
            dtype, stype = rtype
            if dtype != stype:
                new_name = f'{stype}{dtype}'
                assert new_name not in adjs
                new_adjs[new_name] = adj.t()
        adjs.update(new_adjs)
    else:
        assert 0

    return g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full, link_split


class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
