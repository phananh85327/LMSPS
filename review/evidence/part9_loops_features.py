"""Part 9 evidence (review/part9_han_evaluation.md): propagate the search's candidate
features on a mini AMAZON graph with LMSPS's own load_dataset / hg_propagate_feat_dgl, with
and without P-P / C-C self-loops, and compare them per product.

    python review/evidence/part9_loops_features.py hgb <data root holding AMAZON/>
"""
import argparse
import sys
import numpy as np
import torch

hgb, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, hgb)
import utils  # noqa: E402

def feats(loops):
    args = argparse.Namespace(root=root, dataset='AMAZON', task='link_prediction', target_relation='U,P',
                              split_seed=1, num_neg_eval=99, edge_mask_ratio=0, same_type_loops=loops)
    g, *_ = utils.load_dataset(args)
    g = utils.hg_propagate_feat_dgl(g, 'P', 5, 6)
    return {k: v.float() for k, v in g.nodes['P'].data.items() if len(k) >= 2 and k[-1] == 'P' and k <= k[::-1]}

plain, loops = feats(False), feats(True)

def cos(a, b):
    na, nb = a.norm(dim=1), b.norm(dim=1)
    ok = (na > 0) & (nb > 0)
    c = (a * b).sum(1)[ok] / (na[ok] * nb[ok])
    return c, ok

pp = plain['PP'].norm(dim=1) > 0
print(f'products: {len(pp)}; with an also-bought partner: {pp.float().mean():.1%}')
print(f'{"key":<8} {"zero without loops":>18} {"cos(loops, plain) median":>26} {"share cos<0.9":>14}')
for k in sorted(plain, key=lambda k: (len(k), k)):
    if not any(s in k[1:] for s in ('PP', 'CC')) and not k.startswith('PP') and 'CC' not in k:
        continue
    c, ok = cos(plain[k], loops[k])
    zero = (plain[k].norm(dim=1) == 0).float().mean()
    if c.numel():
        print(f'{k:<8} {zero:>18.1%} {c.median():>26.3f} {(c < 0.9).float().mean():>14.1%}')
print('similarity to PUPUP, per product (median cosine): with loops vs plain')
for k in ('PPUPUP', 'PUPPUP'):
    cl, _ = cos(loops[k], loops['PUPUP']); cp, _ = cos(plain[k], plain['PUPUP'])
    print(f'  {k}: {cl.median():.3f} (loops) vs {cp.median():.3f} (plain)')
