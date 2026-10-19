import os
import json
import math
import time
import uuid
import argparse
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import dgl
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

from model_search import *
from utils import *
try:
    from sparse_tools import SparseAdjList
except ImportError as _e:  # compiled extension (python39 build in ../sparse_tools); unused by link prediction
    _sparse_tools_error = _e

    def SparseAdjList(*args, **kwargs):
        raise ImportError('sparse_tools is required for the node-classification / Freebase paths: '
                          f'build it with `cd ../sparse_tools && python setup.py develop` ({_sparse_tools_error})')



def main_link_prediction(args, g, dl, link_split):
    if args.dataset == 'ACM':
        tgt_type = 'P'
    elif args.dataset == 'AMAZON':
        tgt_type = 'P'
    else:
        assert 0, 'link prediction is currently only wired up for ACM and AMAZON'

    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'

    # =======
    # neighbor aggregation (item/tgt side only -- searched via meta-paths)
    # =======
    max_length = args.num_hops + 1
    prop_tic = datetime.datetime.now()
    g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, echo=True)

    feats = {}
    keys = list(g.nodes[tgt_type].data.keys())
    print(f'For tgt {tgt_type}, feature keys {keys}')
    for k in keys:
        feats[k] = g.nodes[tgt_type].data.pop(k)
    if args.tgt_ended_paths:
        # keep only tgt...tgt walks (len >= 2): every candidate is then a real
        # target-type neighbor graph, directly usable as a HAN meta-path adjacency
        assert not args.residual, '--tgt-ended-paths drops the raw tgt key that --residual needs'
        feats = {k: v for k, v in feats.items() if len(k) >= 2 and k[-1] == tgt_type}
        print(f'--tgt-ended-paths: {len(feats)} candidates {list(feats.keys())}')
    if args.dedupe_reversed:
        assert args.tgt_ended_paths, '--dedupe-reversed needs --tgt-ended-paths (reverse of a tgt-ended path is tgt-ended)'
        # reverse(X)'s walk-count matrix is the transpose of X's, and
        # build_metapath_graphs symmetrizes, so X and reverse(X) yield
        # near-identical HAN graphs. Result6/8: the search selected such a pair
        # (PCPPUP + PUPPCP) and HAN's semantic attention put ALL its weight on
        # it -- one relation occupying two of the five slots. Keeping one
        # canonical member per reverse-pair makes every discovered set
        # redundancy-free by construction (palindromes are their own reverse).
        before = len(feats)
        feats = {k: v for k, v in feats.items() if k <= k[::-1]}
        print(f'--dedupe-reversed: {before} -> {len(feats)} candidates {list(feats.keys())}')
    if args.half_feats:
        # fp16 storage halves CPU RAM for the big feats dict; the gather points in
        # train_search_lp / infer_full_embeddings_* upcast to fp32 on the GPU
        feats = {k: v.half() for k, v in feats.items()}
    data_size = {k: v.size(-1) for k, v in feats.items()}
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # no label propagation for link prediction -- see plan's out-of-scope note
    label_feats = {}

    # =======
    # counterpart side (author/user) -- raw attributes only, no meta-path search
    # =======
    src_name, dst_name = args.target_relation.split(',')
    src_type_id = NODE_TYPE_IDS[args.dataset][src_name]
    if args.user_tower == 'embedding':
        # tower consumes indices: arange makes every src_attr[...] gather a passthrough
        src_attr = torch.arange(link_split['num_src'])
    else:
        src_raw_attr = dl.nodes['attr'][src_type_id]
        if src_raw_attr is None:
            # identity fallback is a dense [num_src, num_src]: fine for a tiny
            # type, tens-of-GB OOM at AMAZON user scale -- refuse loudly instead
            assert link_split['num_src'] <= 20000, (
                f'node type {src_name} has no attributes and is too large for a '
                f'torch.eye fallback -- use --user-tower embedding')
            src_attr = torch.eye(link_split['num_src'])
        else:
            src_attr = torch.FloatTensor(src_raw_attr)

    # =======
    # search-internal train/val split of the train_edges for the bi-level (DARTS)
    # update: weights step on one half, alpha step on the other. NOTE: the paper
    # (Eq. 5 / App. A.3) updates alpha on the real validation set; we keep the
    # real val_edges out of the alpha updates and use them ONLY in project_op_lp
    # below, which scores the `--repeat` sampled path sets by validation BPR to
    # pick the final arch (paper Eq. 6). test_edges never touch the search.
    # Consequence: the retrain MUST use the same --split-seed as the search, or
    # the val edges that chose the arch may be test edges of the retrain split.
    # =======
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(link_split['train_edges'].shape[0])
    half = len(perm) // 2
    lp_train_edges = link_split['train_edges'][perm[:half]]
    lp_val_edges = link_split['train_edges'][perm[half:]]

    train_sampler = TrainEdgeSampler(lp_train_edges, link_split['train_adj_src2dst'], seed=args.seed)
    val_sampler = TrainEdgeSampler(lp_val_edges, link_split['train_adj_src2dst'], seed=args.seed + 1)

    # =======
    # eval loader over ALL tgt-type (item) nodes, for full embedding inference
    # =======
    num_dst = link_split['num_dst']
    batchsize = 2 * args.batch_size
    eval_loader = []
    for batch_idx in range((num_dst - 1) // batchsize + 1):
        batch_start = batch_idx * batchsize
        batch_end = min(num_dst, (batch_idx + 1) * batchsize)
        batch = torch.arange(batch_start, batch_end)
        batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
        eval_loader.append((batch, batch_feats, {}, None))

    if args.ns_linear:
        ns_ratio = 2 * args.num_hops / (math.e ** (0.6 * args.num_hops))
        if ns_ratio > 0.5:
            ns_ratio = 0.5
        args.ns = math.floor(len(feats) * ns_ratio)
    if args.ns > len(feats):
        args.ns = len(feats)
    print(f"num sampled :{args.ns}")
    # per-step sampling width during the search (paper: == ns). --ns-search widens
    # it so every path gets more looks per epoch and alpha's softmax compares each
    # path against more companions; the FINAL set still has --ns paths.
    ns_search = args.ns_search if args.ns_search > 0 else args.ns
    assert ns_search <= args.num_final, '--ns-search must not exceed --num_final (the final pool must hold a full sample)'
    assert ns_search <= len(feats), '--ns-search exceeds the candidate count'
    assert 0 <= args.warmup_epochs < args.stage, '--warmup-epochs must be in [0, stage)'
    print(f'search schedule: {args.stage} epochs, warm-up {args.warmup_epochs} (pool fixed at {len(feats)}), '
          f'then linear shrink to {args.num_final}; {ns_search} paths sampled per step, {args.ns} selected at the end')

    # =======
    # Construct network
    # =======
    torch.cuda.empty_cache()
    gc.collect()

    model = LMSPS_Se(args.hidden, 1, feats.keys(), label_feats.keys(), tgt_type,
        args.dropout, args.input_drop, device, args.num_final, args.residual,
        bns=args.bns, data_size=data_size, num_sampled=ns_search, task='link_prediction',
        num_select=args.ns)
    model = model.to(device)
    if args.user_tower == 'embedding':
        tower = UserEmbeddingTower(link_split['num_src'], args.hidden).to(device)
    else:
        tower = ShallowNodeTower(src_attr.shape[1], args.hidden).to(device)

    print(model)
    print(tower)
    print("# Params:", get_n_params(model) + get_n_params(tower))

    optimizer_w = torch.optim.Adam(list(model.parameters()) + list(tower.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.alphas(), lr=args.alr)

    train_times = []
    # disable=None: no progress bar when stdout is redirected to a log file, so
    # the per-epoch "sampled: [...]" print lands on its own line (with the bar,
    # it is glued to tqdm's carriage-return line and log parsers miss it)
    for epoch in tqdm(range(args.stage), disable=None):
        gc.collect()
        torch.cuda.synchronize()

        # paper App. B.4: lambda = 1 during the warm-up epochs (no shrinking), then
        # linear to 0; tau anneals over the full stage regardless
        if epoch < args.warmup_epochs:
            eps = 1.0
        else:
            eps = 1 - (epoch - args.warmup_epochs) / max(1, args.stage - 1 - args.warmup_epochs)
        epoch_sampled = model.epoch_sample(eps)
        meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in epoch_sampled]
        label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats, model.num_paths) if i in epoch_sampled]

        start = time.time()
        model.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.stage - 1))

        loss_w, loss_a = train_search_lp(model, tower, feats, label_feats, src_attr,
                                          epoch_sampled, meta_path_sampled, label_meta_path_sampled,
                                          optimizer_w, optimizer_a, train_sampler, val_sampler,
                                          args.batch_size, device)
        torch.cuda.synchronize()
        end = time.time()
        train_times.append(end - start)
        print(f'Epoch {epoch}, train bpr {loss_w:.4f}, val bpr {loss_a:.4f}')

    ##################direct evaluation##############
    tower.eval()
    with torch.no_grad():
        tower_src_emb = tower(src_attr.to(device))  # stays on GPU; project_op_lp scores there

    # final architecture parameters: the independent ranking the pool is built from
    keys = list(feats.keys())
    with torch.no_grad():
        alpha = model.alpha.detach().float().cpu()
        strength = torch.softmax(alpha / model.tau, dim=-1)
        final_pool = [int(i) for i in model.epoch_pool(0).cpu().numpy()]
    order = torch.argsort(alpha, descending=True).tolist()
    print(f'final alpha ranking (tau={model.tau:.3f}; * = in final pool of {len(final_pool)}):')
    for rank, i in enumerate(order):
        print(f'  {rank + 1:>3} {"*" if i in final_pool else " "} {keys[i]:<10} alpha {alpha[i].item():+.6f}  strength {strength[i].item():.5f}')

    model.eval_gumbel = not args.eval_no_gumbel
    geno_out, eval_table = project_op_lp(keys, model, eval_loader, device,
                                         tower_src_emb, torch.LongTensor(link_split['val_edges']),
                                         torch.LongTensor(link_split['val_negs']), args.repeat,
                                         exhaustive_cap=args.eval_exhaustive_cap)
    print('average train times', sum(train_times) / len(train_times))
    print("paths {}\n".format(geno_out))

    # The run's own statement of its tie group: every scored set whose val BPR is
    # within --report-delta of the best. Within one run these sets are not
    # distinguishable (the seed alone moves a set's score by ~0.008 on AMAZON), so
    # the argmin above is one arbitrary member of this group. Printed here, before
    # any fleet-level step, so a later selection among tied sets is a count over
    # declared ties and not a post-hoc re-pick.
    tie_sets = []
    if args.report_delta > 0 and eval_table:
        best_loss = min(loss for _, loss in eval_table)
        tie_sets = sorted(((idx, loss - best_loss) for idx, loss in eval_table if loss - best_loss <= args.report_delta),
                          key=lambda t: t[1])
        print(f'tie group (within {args.report_delta:.4f} of the best val bpr {best_loss:.4f}): {len(tie_sets)} of {len(eval_table)} sets')
        for idx, gap in tie_sets:
            print(f'  +{gap:.4f}  {[keys[i] for i in idx]}')
        print()

    if args.search_json:
        # provenance record: everything needed to reproduce / audit this search
        # result (split seed, flags, candidates, alpha, pool, every scored set)
        save_path = args.search_json
        if save_path == 'auto':
            os.makedirs(f'./output/{args.dataset}', exist_ok=True)
            save_path = f'./output/{args.dataset}/search_{datetime.datetime.now():%Y%m%d_%H%M%S}_seed{args.seed}.json'
        record = {
            'dataset': args.dataset, 'task': args.task, 'target_relation': args.target_relation,
            'split_seed': args.split_seed, 'seed': args.seed, 'args': vars(args),
            'candidates': keys,
            'alpha': {keys[i]: alpha[i].item() for i in range(len(keys))},
            'final_tau': float(model.tau),
            'final_pool': [keys[i] for i in final_pool],
            'ns_search': ns_search, 'ns_select': args.ns, 'warmup_epochs': args.warmup_epochs,
            'eval_mode': ('exhaustive' if args.eval_exhaustive_cap and
                          math.comb(len(final_pool), args.ns) <= args.eval_exhaustive_cap else 'sampled'),
            'eval_gumbel': model.eval_gumbel,
            'eval_table': [{'paths': [keys[i] for i in idx], 'val_bpr': loss} for idx, loss in eval_table],
            'paths': geno_out[0], 'label_paths': geno_out[1],
            'tie_delta': args.report_delta,
            'tie_sets': [{'paths': [keys[i] for i in idx], 'gap': gap} for idx, gap in tie_sets],
            'avg_epoch_time_s': sum(train_times) / len(train_times),
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=1)
        print(f'search record saved to {save_path}')


def main(args):
    if args.deterministic:
        # Same-seed searches in hgb/logs were bit-identical for 33-60 epochs and
        # then drifted by 1e-4 in a loss value (cuBLAS / reduction-order
        # nondeterminism), after which argsort(alpha) + random.sample amplify the
        # drift into different path sets. These settings remove the drift at some
        # speed cost. The cuBLAS workspace variable must be set before the first
        # CUDA call. warn_only=True: an op without a deterministic implementation
        # is reported in the log (UserWarning) instead of aborting the run.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # torch < 1.11 has no warn_only; then an offending op raises instead
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        dgl.seed(args.seed)
        print(f'deterministic mode: CUBLAS_WORKSPACE_CONFIG={os.environ["CUBLAS_WORKSPACE_CONFIG"]}, '
              f'torch.use_deterministic_algorithms(True, warn_only=True)')
    if args.seed > 0:
        set_random_seed(args.seed)

    g, adjs, init_labels, num_classes, dl, train_nid, val_nid, test_nid, test_nid_full, link_split \
        = load_dataset(args)

    if args.task == 'link_prediction':
        main_link_prediction(args, g, dl, link_split)
        return

    for k in adjs.keys():
        adjs[k].storage._value = None
        adjs[k].storage._value = torch.ones(adjs[k].nnz()) / adjs[k].sum(dim=-1)[adjs[k].storage.row()]

    # =======
    # rearange node idx (for feats & labels)
    # =======
    search_node_nums= (len(train_nid)+len(val_nid))
    train_node_nums = search_node_nums // 2
    valid_node_nums = search_node_nums // 2
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = train_node_nums + valid_node_nums + test_node_nums


    num_nodes = dl.nodes['count'][0]

    if total_num_nodes < num_nodes:
        flag = np.ones(num_nodes, dtype=bool)
        flag[train_nid] = 0
        flag[val_nid] = 0
        flag[test_nid] = 0
        extra_nid = np.where(flag)[0]
        print(f'Find {len(extra_nid)} extra nid for dataset {args.dataset}')
    else:
        extra_nid = np.array([])

    init2sort = torch.LongTensor(np.concatenate([train_nid, val_nid, test_nid, extra_nid]))
    sort2init = torch.argsort(init2sort)
    assert torch.all(init_labels[init2sort][sort2init] == init_labels)
    labels = init_labels[init2sort]

    # =======
    # neighbor aggregation
    # =======
    if args.dataset == 'DBLP':
        tgt_type = 'A'
        node_types = ['A', 'P', 'T', 'V']
        extra_metapath = []
    elif args.dataset == 'ACM':
        tgt_type = 'P'
        node_types = ['P', 'A', 'C']
        extra_metapath = []
    elif args.dataset == 'IMDB':
        tgt_type = 'M'
        node_types = ['M', 'A', 'D', 'K']
        extra_metapath = []
    elif args.dataset == 'Freebase':
        tgt_type = '0'
        node_types = [str(i) for i in range(8)]
        extra_metapath = []
    else:
        assert 0
    extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_hops + 1]

    print(f'Current num hops = {args.num_hops}')

    if args.dataset == 'Freebase':
        prop_device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    else:
        prop_device = 'cpu'
    store_device = 'cpu'

    if args.dataset == 'Freebase':
        if not os.path.exists('./Freebase_adjs'):
            os.makedirs('./Freebase_adjs')
        num_tgt_nodes = dl.nodes['count'][0]

    # compute k-hop feature
    prop_tic = datetime.datetime.now()
    if args.dataset != 'Freebase':
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1


        g = hg_propagate_feat_dgl(g, tgt_type, args.num_hops, max_length, echo=True)


        feats = {}
        keys = list(g.nodes[tgt_type].data.keys())
        print(f'For tgt {tgt_type}, feature keys {keys}')
        for k in keys:
            feats[k] = g.nodes[tgt_type].data.pop(k)

    else:
        if len(extra_metapath):
            max_length = max(args.num_hops + 1, max([len(ele) for ele in extra_metapath]))
        else:
            max_length = args.num_hops + 1

        save_name = f'./Freebase_adjs/feat_seed{args.seed}_hop{args.num_hops}'
        if args.seed > 0 and os.path.exists(f'{save_name}_00_int64.npy'):
            # meta_adjs = torch.load(save_name)
            meta_adjs = {}
            for srcname in tqdm(dl.nodes['count'].keys()):
                tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                for k in tmp.keys:
                    assert k not in meta_adjs
                meta_adjs.update(tmp.load_adjs(expand=True))
                del tmp
        else:
            meta_adjs = hg_propagate_sparse_pyg(adjs, tgt_type, args.num_hops, max_length, extra_metapath, prop_feats=True, echo=True, prop_device=prop_device)

            meta_adj_list = []
            for srcname in dl.nodes['count'].keys():
                keys = [k for k in meta_adjs.keys() if k[-1] == str(srcname)]
                tmp = SparseAdjList(f'{save_name}_0{srcname}', keys, meta_adjs, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                meta_adj_list.append(tmp)

            for srcname in dl.nodes['count'].keys():
                tmp = SparseAdjList(f'{save_name}_0{srcname}', None, None, num_tgt_nodes, dl.nodes['count'][srcname], with_values=True)
                tmp_adjs = tmp.load_adjs(expand=True)
                print(srcname, tmp.keys)
                for k in tmp.keys:
                    assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                    assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                    assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                del tmp_adjs, tmp
                gc.collect()

        feats = {k: v.clone() for k, v in meta_adjs.items() if len(k) <= args.num_hops + 1 or k in extra_metapath}
        assert '0' not in feats
        feats['0'] = SparseTensor.eye(dl.nodes['count'][0])
        print(f'For tgt {tgt_type}, Involved keys {feats.keys()}')

    if args.dataset in ['DBLP', 'ACM', 'IMDB']:
        data_size = {k: v.size(-1) for k, v in feats.items()}
        feats = {k: v[init2sort] for k, v in feats.items()}
    elif args.dataset == 'Freebase':
        data_size = dict(dl.nodes['count'])

        for k, v in tqdm(feats.items()):
            if len(k) == 1:
                continue

            if k[0] == '0' and k[-1] == '0':

                feats[k], _ = v.sample_adj(init2sort, -1, False) # faster, 50% time acceleration
            elif k[0] == '0':

                feats[k] = v[init2sort]
            else:
                assert args.two_layer, k
                if k[-1] == tgt_type:
                    feats[k] = v[:, init2sort]
                else:
                    feats[k] = v
    else:
        assert 0
    prop_toc = datetime.datetime.now()
    print(f'Time used for feat prop {prop_toc - prop_tic}')
    gc.collect()

    # =======
    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)
    checkpt_file = checkpt_folder + uuid.uuid4().hex
    #print('checkpt_file', checkpt_file)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = 'cuda:{}'.format(args.gpu) if not args.cpu else 'cpu'
    if args.dataset != 'IMDB':
        labels_cuda = labels.long().to(device)
    else:
        labels = labels.float()
        labels_cuda = labels.to(device)

    for stage in [0]:
        epochs = args.stage

        # =======
        # labels propagate alongside the metapath
        # =======
        label_feats = {}
        if args.label_feats:
            if args.dataset != 'IMDB':
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = F.one_hot(init_labels[train_nid], num_classes).float()
            else:
                label_onehot = torch.zeros((num_nodes, num_classes))
                label_onehot[train_nid] = init_labels[train_nid].float()

            if args.dataset == 'DBLP':
                extra_metapath = []
            elif args.dataset == 'IMDB':
                extra_metapath = []
            elif args.dataset == 'ACM':
                extra_metapath = []
            elif args.dataset == 'Freebase':
                extra_metapath = []
            else:
                assert 0

            extra_metapath = [ele for ele in extra_metapath if len(ele) > args.num_label_hops + 1]
            if len(extra_metapath):
                max_length = max(args.num_label_hops + 1, max([len(ele) for ele in extra_metapath]))
            else:
                max_length = args.num_label_hops + 1

            print(f'Current label-prop num hops = {args.num_label_hops}')
            # compute k-hop feature
            prop_tic = datetime.datetime.now()
            if args.dataset == 'Freebase' and args.num_label_hops <= args.num_hops and len(extra_metapath) == 0:
                meta_adjs = {k: v for k, v in meta_adjs.items() if k[-1] == '0' and len(k) < max_length}
            else:
                if args.dataset == 'Freebase':
                    save_name = f'./Freebase_adjs/label_seed{args.seed}_hop{args.num_label_hops}'
                    if args.seed > 0 and os.path.exists(f'{save_name}_int64.npy'):
                        meta_adj_list = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        meta_adjs = meta_adj_list.load_adjs(expand=True)
                    else:
                        meta_adjs = hg_propagate_sparse_pyg(
                            adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)
                        meta_adj_list = SparseAdjList(save_name, meta_adjs.keys(), meta_adjs, num_tgt_nodes, num_tgt_nodes, with_values=True)

                        tmp = SparseAdjList(save_name, None, None, num_tgt_nodes, num_tgt_nodes, with_values=True)
                        tmp_adjs = tmp.load_adjs(expand=True)
                        for k in tmp.keys:
                            assert torch.all(meta_adjs[k].storage.rowptr() == tmp_adjs[k].storage.rowptr())
                            assert torch.all(meta_adjs[k].storage.col() == tmp_adjs[k].storage.col())
                            assert torch.all(meta_adjs[k].storage.value() == tmp_adjs[k].storage.value())
                        del tmp_adjs, tmp
                        gc.collect()
                else:
                    # try:
                    #     meta_adjs = torch.load(f'./cache/{args.dataset}_label_prop_hop{args.num_label_hops}.pt')
                    # except:
                    meta_adjs = hg_propagate_sparse_pyg(
                        adjs, tgt_type, args.num_label_hops, max_length, extra_metapath, prop_feats=False, echo=True, prop_device=prop_device)
                    # torch.save(meta_adjs, f'./cache/{args.dataset}_label_prop_hop{args.num_label_hops}.pt')

            if args.dataset == 'Freebase':
                if 0:
                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()

                    condition = lambda ra,rb,rc,k: rb > 0.2
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False)

                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(label_feats.keys())) - set(left_keys))
                    for k in remove_keys:
                        label_feats.pop(k)
                else:
                    left_keys = ['00', '000', '0000', '0010', '0030', '0040', '0050', '0060', '0070']
                    remove_keys = list(set(list(meta_adjs.keys())) - set(left_keys))
                    for k in remove_keys:
                        meta_adjs.pop(k)

                    label_onehot_g = label_onehot.to(prop_device)
                    for k, v in tqdm(meta_adjs.items()):
                        if args.dataset != 'Freebase':
                            label_feats[k] = remove_diag(v) @ label_onehot
                        else:
                            label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)

                    del label_onehot_g
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                for k, v in tqdm(meta_adjs.items()):
                    if args.dataset != 'Freebase':
                        label_feats[k] = remove_diag(v) @ label_onehot
                    else:
                        label_feats[k] = (remove_diag(v).to(prop_device) @ label_onehot_g).to(store_device)
                gc.collect()

                if args.dataset == 'IMDB':
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=False, loss_type='bce')
                else:
                    condition = lambda ra,rb,rc,k: True
                    check_acc(label_feats, condition, init_labels, train_nid, val_nid, test_nid, show_test=True)
            print('Involved label keys', label_feats.keys())

            label_feats = {k: v[init2sort] for k,v in label_feats.items()}
            prop_toc = datetime.datetime.now()
            print(f'Time used for label prop {prop_toc - prop_tic}')

        # =======
        # Train & eval loaders
        # =======
        train_loader = torch.utils.data.DataLoader(
            torch.arange(valtest_point//2), batch_size=args.batch_size, shuffle=True, drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            torch.arange(valtest_point//2, valtest_point), batch_size=args.batch_size, shuffle=True, drop_last=False)



        # =======
        # Mask & Smooth
        # =======
        with_mask = False

        eval_loader, full_loader = [], []
        batchsize = 2 * args.batch_size


        for batch_idx in range((total_num_nodes-1) // batchsize + 1):
            batch_start = batch_idx * batchsize
            batch_end = min(total_num_nodes, (batch_idx+1) * batchsize)
            batch = torch.arange(batch_start, batch_end)

            batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
            if with_mask:
                # batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                ...
            else:
                batch_mask = None
            eval_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))

        for batch_idx in range((num_nodes-total_num_nodes-1) // batchsize + 1):
            batch_start = batch_idx * batchsize + total_num_nodes
            batch_end = min(num_nodes, (batch_idx+1) * batchsize + total_num_nodes)
            batch = torch.arange(batch_start, batch_end)

            batch_feats = {k: x[batch_start:batch_end] for k, x in feats.items()}
            batch_labels_feats = {k: x[batch_start:batch_end] for k, x in label_feats.items()}
            if with_mask:
                # batch_mask = {k: x[batch_start:batch_end] for k, x in full_mask.items()}
                ...
            else:
                batch_mask = None
            full_loader.append((batch, batch_feats, batch_labels_feats, batch_mask))


        if args.ns_linear:
            ns_ratio = 2 * args.num_hops / (math.e ** (0.6 * args.num_hops))
            if ns_ratio > 0.5:
                ns_ratio = 0.5
            args.ns = math.floor((len(feats) + len(label_feats)) * ns_ratio)

        if args.ns > (len(feats) + len(label_feats)):
            args.ns = (len(feats) + len(label_feats))

        print(f"num sampled :{args.ns}")

        # =======
        # Construct network
        # =======
        torch.cuda.empty_cache()
        gc.collect()

        print(data_size.keys(), feats.keys(), label_feats.keys())

        model = LMSPS_Se(args.hidden, num_classes, feats.keys(), label_feats.keys(), tgt_type,
            args.dropout, args.input_drop, device, args.num_final, args.residual, bns=args.bns, data_size=data_size, num_sampled=args.ns)

        print(model)

        model = model.to(device)



        if args.seed == args.seeds[0]:
            #print(model)
            print("# Params:", get_n_params(model))

        if args.dataset == 'IMDB':
            loss_fcn = nn.BCEWithLogitsLoss()
        else:
            loss_fcn = nn.CrossEntropyLoss()
        optimizer_w = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        optimizer_a = torch.optim.Adam(model.alphas(), lr=args.alr)  #,weight_decay=args.weight_decay

        best_epoch = -1
        best_val_loss = 1000000
        best_test_loss = 0
        best_val = (0,0)
        best_test = (0,0)
        val_loss_list, test_loss_list = [], []
        val_acc_list, test_acc_list = [], []
        actual_loss_list, actual_acc_list = [], []
        store_list = []
        best_pred = None
        count = 0

        train_times = []



        for epoch in tqdm(range(args.stage)):
            gc.collect()
            torch.cuda.synchronize()


            # determain eps for operation selection
            eps = 1 - epoch/(args.stage - 1)

            epoch_sampled = model.epoch_sample(eps)
            meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats) if i in epoch_sampled]
            label_meta_path_sampled = [model.all_meta_path[i] for i in range(model.num_feats,model.num_paths) if i in epoch_sampled]

            epoch_feats = {k:v for k,v in feats.items() if k in meta_path_sampled or (model.residual and k==model.tgt_key)}  #
            epoch_label_feats = {k:v for k,v in label_feats.items() if k in label_meta_path_sampled}


            start = time.time()

            # determain tau
            model.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.stage - 1))



            loss_w, loss_a, acc_w, acc_a = train_search_new(model, epoch_feats, epoch_label_feats, labels_cuda, loss_fcn, 
                                                            optimizer_w, optimizer_a, train_loader, val_loader,
                                                              epoch_sampled, meta_path_sampled, label_meta_path_sampled, 
                                                              evaluator, scalar=scalar)
            torch.cuda.synchronize()
            end = time.time()





            log = "" #"Epoch {}, training Time(s): {:.4f}, estimated train loss {:.4f}, acc {:.4f}, {:.4f}\n".format(epoch, end - start,loss, acc[0]*100, acc[1]*100)
            # print ("paths {}, label_path {}\n".format(path, label_path))

            torch.cuda.empty_cache()
            train_times.append(end-start)

        ##################direct evaluation##############
        # logging.info('Using valid loss crit for arch selection...')
        # model_new._initialize_flags()
        geno_out_vloss = project_op(list(feats.keys()), model, loss_fcn, eval_loader, device, trainval_point, valtest_point, labels, args.repeat)
        # return geno_out_vloss
        print('average train times', sum(train_times) / len(train_times))
        print("paths {}\n".format(geno_out_vloss))


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='LMSPS')
    ## For environment costruction
    parser.add_argument("--seeds", nargs='+', type=int, default=[0],
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action='store_true', default=False)
    parser.add_argument("--root", type=str, default="../data/")
    parser.add_argument("--stage", type=int, default=200, help="The epoch setting for each stage.")  # default 200
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    parser.add_argument("--label-feats", action='store_true', default=False,
                        help="whether to use the label propagated features")
    parser.add_argument("--num-label-hops", type=int, default=2,
                        help="number of hops for propagation of raw features")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0,  # original 0.5
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0,  # original 0.1
                        help="input dropout of input features")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to add residual branch the raw input features")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--alr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    parser.add_argument("--drop-metapath", type=float, default=0,
                        help="whether to process the input features")
    ## for ablation
    parser.add_argument("--nh", type=int, default=1)
    parser.add_argument("--ns", type=int, default=30)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--dy", action='store_true', default=False)
    parser.add_argument("--no_path", nargs='+', type=str, default=[])
    parser.add_argument("--no_label", nargs='+', type=str, default=[])
    parser.add_argument("--identity", action='store_true', default=False)
    parser.add_argument("--topn", type=int, default=0)
    parser.add_argument("--all_path", action='store_true', default=False)
    parser.add_argument("--ns_linear", action='store_true', default=False)
    parser.add_argument('--tau_max', type=float, default=8, help='for gumbel softmax search gradient max value')
    parser.add_argument('--tau_min', type=float, default=4, help='for gumbel softmax search gradient min value')
    parser.add_argument("--edge_mask_ratio", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--num_final", type=int, default=60)
    parser.add_argument("--ACM_keep_F", action='store_true', default=False,
                        help="whether to use Field type")
    ## for link prediction
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "link_prediction"])
    parser.add_argument("--target-relation", type=str, default="A,P",
                        help="src_type,dst_type node-type letters for the link-prediction relation")
    parser.add_argument("--num-neg-eval", type=int, default=99,
                        help="number of sampled negatives per positive edge for val/test ranking")
    parser.add_argument("--split-seed", type=int, default=0,
                        help="fixed seed for the link-prediction edge split; independent of --seeds so search and final retrain share the same split")
    parser.add_argument("--tgt-ended-paths", action="store_true",
                        help="restrict search candidates to paths that start AND end at the target type, so every discovered path is usable as a HAN meta-path adjacency")
    parser.add_argument("--dedupe-reversed", action="store_true",
                        help="with --tgt-ended-paths: keep one canonical candidate per reverse-pair "
                             "(X vs reversed X). Their symmetrized HAN graphs are near-identical, so "
                             "a set containing both wastes a path slot (seen in Result5/6/8)")
    parser.add_argument("--half-feats", action="store_true",
                        help="store propagated features as fp16 on CPU (halves feats RAM; upcast to fp32 on GPU at gather time)")
    parser.add_argument("--user-tower", type=str, default="embedding", choices=["embedding", "feats"],
                        help="link-prediction user side: 'embedding' = free per-user trained vectors (same per-user capacity as metapath2vec/GAT -- use for comparisons); 'feats' = linear projection of raw user attributes (inductive; original behavior)")
    ## search reproducibility (all opt-in; defaults reproduce the upstream behaviour)
    parser.add_argument("--eval-no-gumbel", action="store_true",
                        help="sampling evaluation scores each path set with the noise-free path strength "
                             "softmax(alpha/tau) instead of a fresh Gumbel draw per node batch, so a set's "
                             "val loss no longer depends on the draw (same-set re-scores varied by up to "
                             "0.02 in hgb/logs while set-to-set gaps were 0.001-0.005)")
    parser.add_argument("--eval-exhaustive-cap", type=int, default=0,
                        help="if > 0 and C(num_final, ns) <= cap, score EVERY subset of the final pool once "
                             "instead of --repeat uniform draws (C(10,4)=C(10,6)=210; 200 draws reach ~125 "
                             "distinct sets). 0 = off")
    parser.add_argument("--deterministic", action="store_true",
                        help="CUBLAS_WORKSPACE_CONFIG=:4096:8 + torch.use_deterministic_algorithms(True, "
                             "warn_only=True) + cudnn.deterministic + dgl.seed, so same-seed searches match")
    parser.add_argument("--search-json", type=str, default="auto",
                        help="where to save the search provenance record (args, candidates, alpha, pool, "
                             "every scored set, chosen paths); 'auto' = output/<dataset>/search_<time>_seed<s>.json, "
                             "'' = do not save")
    ## search schedule (link prediction; opt-in, defaults = upstream schedule)
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="epochs with the full candidate pool before progressive shrinking starts "
                             "(paper App. B.4 uses 20 of 200). Without it, paths cut early in fleet v2 had "
                             "only 2-4 looks; eliminated paths averaged 8")
    parser.add_argument("--ns-search", type=int, default=0,
                        help="paths sampled per training step (0 = --ns, the paper's choice). A wider "
                             "sample gives every path more looks per epoch and compares it against more "
                             "companions in alpha's softmax; the final set still has --ns paths. "
                             "Must be <= --num_final. DEPARTURE from the paper when != --ns")
    parser.add_argument("--report-delta", type=float, default=0.010,
                        help="after the sampling evaluation, print and record every scored set whose val "
                             "BPR is within this tolerance of the best (the run's own tie group). Use the "
                             "measured cross-seed noise (~0.008-0.010 on AMAZON). 0 = off")
    return parser.parse_args(args)

if __name__ == '__main__':
    args = parse_args()

    args.seed = args.seeds[0]
    print(args)


    for seed in args.seeds:
        args.seed = seed
        print('Restart with seed =', seed)
        main(args)

