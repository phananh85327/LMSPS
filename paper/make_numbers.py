#!/usr/bin/env python3
r"""make_numbers.py -- every number in the paper "Is Meta-Path Search Worth Its Cost for HAN-Based
Product Recommendation? A Controlled Study of LMSPS on Amazon-Meta", from the run records of its two
repositories.

    python make_numbers.py --lmsps <LMSPS checkout> --han <HAN-improve checkout>
                           [--out <paper folder>] [--extraction-report <extraction_report.json>]
                           [--lang vi]

The script is kept in the LMSPS repository as paper/make_numbers.py. From the root of that
checkout, with the HAN repository cloned next to it:

    python paper/make_numbers.py --lmsps . --han ../han-improve --out <paper folder>

It only reads the two checkouts. It writes, into --out (by default the folder that holds it),
    <out>/numbers.tex        \setval{key}{value} for every number; the paper prints one with \val{key}
    <out>/figures/*.dat      the data behind figures F2-F4
and prints a short list of key values for a check by eye.

With --lang vi, numbers.tex is the one of the Vietnamese version (paper_v2_vn): a decimal comma and
a dot between thousands (0,666; 1.147.920), "thứ 6, 7 và 8" for a list of places and "triệu" for
million. Version numbers, commits, branches, the split fingerprint, the GPU's name, the path sets
and the names of sets stay as they are (vi_value). The figure data are the same in both languages,
and without --lang vi the output is what it was before the option existed.

Sources (all committed, except the optional extraction report):
  LMSPS  hgb/logs/Fleet v3/final_ns{4,6}/search_ns*_seed*.{json,txt}  the 60 search runs
         hgb/logs/Fleet v3/final_ns{4,6}/reselect_d0{10,05}.json     the fleet decisions
         hgb/reselect_fleet.py                                        averaged scores (imported)
         review/evidence/audit_before_fix.json, data/AMAZON/audit.json  the data before / after
         data/AMAZON/meta.json, link.dat, label.dat, label.dat.test     graph, split, product groups
  HAN    output/AMAZON/v3/results.csv, output/AMAZON/v3/logs/          the 68 runs (and the warm-up check)
         arch.py                                                        the 24 path sets
The raw-file counts come from the full run's extraction_report.json: the file given with
--extraction-report, else the first one found in the LMSPS checkout (review/evidence/, then
data/AMAZON_filtered/), else the copy of its fields kept in this script (REPORT_COPY, 4 November 2026).
numbers.tex says which one was used, and the commit and branch of each checkout.

The split is rebuilt as hgb/link_data.py builds it (unique edges, numpy default_rng(split seed)
permutation, 80/10/10) and checked against the training-edge fingerprint in every run record, so the
test-edge numbers are exact. Needs Python 3.8+ and numpy; nothing else.
"""
import argparse
import csv
import datetime
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import Counter

import numpy as np

FINGERPRINT = 'c5f7bccbf61c1b0f'
DROP = 0.01                                                         # a path weight below this counts as dropped
EMPTY = {'PCCP', 'PCCPP', 'PCCPPP', 'PCCPCP', 'PCCPUP', 'PPCCPP'}   # no endpoint for 76-81% of products
GROUPS = ['Book', 'DVD', 'Music', 'Video']                          # label order of dat_generator.py (sorted)
DEG = [('d0', 'deg=0'), ('d1', 'deg 1-5'), ('d2', 'deg 6-20'), ('d3', 'deg>=21')]
METRICS = [('hit', 'hit10'), ('ndcg', 'ndcg10'), ('fhit', 'hit10_full'), ('fndcg', 'ndcg10_full'),
           ('frec', 'recall20_full'), ('val', 'val_bpr')]
# The fields of the full run's extraction_report.json (hgb/amazon_extract.py) that the paper uses, as
# the file stood on 4 November 2026; used only when no report file is found.
REPORT_COPY = {
    'settings': {'pos_threshold': 4.0, 'helpful_min_votes': 5, 'helpful_min_ratio': 0.2, 'k_core': 5,
                 'edition_min_shared': 3, 'edition_min_overlap': 0.5, 'edition_max_run': 50,
                 'placeholder_min_repeats': 50, 'drop_customers': ['ATVPDKIKX0DER', 'A3UN6WX5RRO2AG']},
    'raw': {'blocks': 548552, 'usable_products': 542684, 'review_lines': 7593244, 'categories': 49732},
    'placeholders_dropped': {
        'ATVPDKIKX0DER': {'reason': 'known placeholder', 'review_lines': 945065, 'max_on_one_product': 1056},
        'A3UN6WX5RRO2AG': {'reason': 'known placeholder', 'review_lines': 201770, 'max_on_one_product': 1193},
        'A2HS1WB42BT31I': {'reason': '>= 50 reviews on one product', 'review_lines': 685, 'max_on_one_product': 149},
        'A3UBZ958HZJ2ES': {'reason': '>= 50 reviews on one product', 'review_lines': 91, 'max_on_one_product': 81},
        'A1FHNBAA0Y8NBA': {'reason': '>= 50 reviews on one product', 'review_lines': 227, 'max_on_one_product': 57},
        'A1ELIM4ZVL13TR': {'reason': '>= 50 reviews on one product', 'review_lines': 82, 'max_on_one_product': 82}},
    'editions': {'pairs': 64814, 'clusters': 30648, 'products_in_clusters': 72280,
                 'cluster_sizes': {'2': 24366, '3': 4036, '4': 1268, '5': 470, '6': 224, '7': 106, '8': 55,
                                   '9': 25, '10': 27, '11': 17, '12': 16, '13': 8, '14': 6, '15': 6, '16': 3,
                                   '17': 3, '18': 5, '20': 3, '22': 2, '23': 1, '24': 1}},
    'filtered': {'products': 48883, 'users': 53569, 'reviews': 819847, 'categories': 37411,
                 'also_bought': 121075, 'category_path_rows': 2288922, 'kcore_iterations': 13},
}

# ------------------------------------------------------------------------------------------------
pccp_iso_global = []   # products PCCP links to no other product (graph_numbers), for a cross-check
train_products_global = []   # the product of every training edge (split_numbers)

# the value store
# ------------------------------------------------------------------------------------------------
VALUES = []          # (key, value, note)
SEEN = set()


def put(key, value, note=''):
    if not re.fullmatch(r'[a-z0-9-]+', key):
        raise ValueError(f'bad key {key!r}: use lower-case letters, digits and hyphens')
    if key in SEEN:
        raise ValueError(f'duplicate key {key!r}')
    SEEN.add(key)
    VALUES.append((key, str(value), note))


def section(title):
    VALUES.append((None, title, ''))


def n(x):                      # 48,883; a median of 59.5 prints as 60 (half to even, as reselect_fleet.py)
    return f'{x:,.0f}'


def f(x, d):                   # fixed decimals
    return f'{x:.{d}f}'


def sg(x, d):                  # signed, safe in text and math
    s = f'{x:+.{d}f}'
    if float(s) == 0:
        s = f'{0:.{d}f}'
    return r'\ensuremath{' + s + '}'


def ms(x, d):                  # minus sign only (correlations, differences printed without a plus)
    return r'\ensuremath{' + f'{x:.{d}f}' + '}'


def pct(x, d=1):               # a fraction as a percentage, without the % sign
    return f'{100 * x:.{d}f}'


def pct_auto(x):               # small percentages keep two significant digits
    v = 100 * x
    return f'{v:.1f}' if v >= 1 else f'{v:.2f}' if v >= 0.1 else f'{v:.3f}'


def pval(p):
    if p < 0.001:
        return r'\ensuremath{<}0.001'
    return f'{p:.2g}' if p < 0.1 else f'{p:.2f}'


def ptext(v):                  # a formatted p as text: p<0.001 or p=0.0076
    lt = r'\ensuremath{<}'
    return r'\ensuremath{p<' + v[len(lt):] + '}' if v.startswith(lt) else r'\ensuremath{p=' + v + '}'


def ordinal(k):
    return {1: '1st', 2: '2nd', 3: '3rd'}.get(k, f'{k}th')


# values that --lang vi leaves as they are: versions, names and ids (and every repo-* key and path set)
VI_KEEP = {'env-python', 'env-torch', 'env-cuda', 'env-dgl', 'env-numpy', 'env-gpu', 'split-fingerprint',
           'han-valpick', 'mtv-valpick'}


def vi_value(key, value):
    """A value in the Vietnamese number format (--lang vi): 0.666 -> 0,666; 1,147,920 -> 1.147.920;
    6th, 7th and 8th -> thứ 6, 7 và 8; 142.8 million -> 142,8 triệu."""
    if key in VI_KEEP or key.startswith('repo-') or re.fullmatch(r'[PCU]+(?: [PCU]+)*', value):
        return value
    places = re.findall(r'(\d+)(?:st|nd|rd|th)\b', value)
    if places:
        return 'thứ ' + (', '.join(places[:-1]) + ' và ' + places[-1] if len(places) > 1 else places[0])
    value = value.replace(' million', ' triệu')
    value = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '\0', value)       # the thousands, kept apart for now
    value = re.sub(r'(?<=\d)\.(?=\d)', ',', value)                # the decimals
    value = value.replace('\0', '.')
    if re.search(r'[A-Za-z]', re.sub(r'\\(?:ensuremath|detokenize)|triệu|\bp\b', '', value)):
        raise ValueError(f'{key}: {value!r} holds a word that vi_value() has no Vietnamese for')
    return value


# ------------------------------------------------------------------------------------------------
# statistics without scipy (regularised incomplete beta by Lentz's continued fraction)
# ------------------------------------------------------------------------------------------------
def _betacf(a, b, x, itmax=500, eps=1e-15):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def welch(x, y):
    """Difference of means and Welch's two-sided p."""
    mx, my, vx, vy = st.mean(x), st.mean(y), st.variance(x), st.variance(y)
    se2 = vx / len(x) + vy / len(y)
    t = (mx - my) / math.sqrt(se2)
    df = se2 ** 2 / ((vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1))
    return mx - my, _betai(df / 2, 0.5, df / (df + t * t))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def perm_spearman(a, b, shuffles=20000, seed=0):
    """Spearman with a permutation p, exactly as LMSPS review/evidence/part4b_search_vs_han.py."""
    r0, rng, bb, hit = spearman(a, b), random.Random(seed), list(b), 0
    for _ in range(shuffles):
        rng.shuffle(bb)
        hit += abs(spearman(a, bb)) >= abs(r0) - 1e-12
    return r0, (hit + 1) / (shuffles + 1)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_json(*parts):
    with open(os.path.join(*parts), encoding='utf-8') as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------------
# the data
# ------------------------------------------------------------------------------------------------
def find_report(lmsps, explicit):
    """The extraction report to use, and a note saying where it came from."""
    paths = [explicit] if explicit else [os.path.join(lmsps, 'review', 'evidence', 'extraction_report.json'),
                                         os.path.join(lmsps, 'data', 'AMAZON_filtered', 'extraction_report.json')]
    for path in paths:
        if os.path.isfile(path):
            rep = read_json(path)
            for part in ('raw', 'editions', 'filtered'):
                for k, v in REPORT_COPY[part].items():
                    if rep[part].get(k) != v:
                        print(f'NOTE: {path} has {part}.{k} = {rep[part].get(k)}, the copy in this script {v}: '
                              'the data were rebuilt, so check every number')
            inside = os.path.abspath(path).startswith(os.path.abspath(lmsps) + os.sep)
            where = os.path.relpath(path, lmsps).replace(os.sep, '/') if inside else 'given with --extraction-report'
            return rep, f'extraction_report.json ({where})'
        if explicit:
            raise SystemExit(f'no such file: {path}')
    print('NOTE: no extraction_report.json found; the raw-file counts come from the copy in this script')
    return REPORT_COPY, 'extraction_report.json (copy in make_numbers.py)'


def data_numbers(lmsps, extraction_report):
    section('Data: the raw file, the fixes and the clean graph (LMSPS notes, Parts 2-4)')
    rep, src = find_report(lmsps, extraction_report)
    ph = rep['placeholders_dropped']
    found = [v for k, v in ph.items() if k not in rep['settings']['drop_customers']]
    sizes = {int(k): v for k, v in rep['editions']['cluster_sizes'].items()}
    raw = dict(blocks=rep['raw']['blocks'], usable=rep['raw']['usable_products'],
               review_lines=rep['raw']['review_lines'], categories=rep['raw']['categories'],
               placeholder_ids=len(ph), placeholder_lines=sum(v['review_lines'] for v in ph.values()),
               placeholder_main=ph['ATVPDKIKX0DER']['review_lines'],
               placeholder_second=ph['A3UN6WX5RRO2AG']['review_lines'],
               edition_pairs=rep['editions']['pairs'], edition_clusters=rep['editions']['clusters'],
               edition_products=rep['editions']['products_in_clusters'],
               kcore_iterations=rep['filtered']['kcore_iterations'],
               category_path_rows=rep['filtered']['category_path_rows'],
               also_bought_rows=rep['filtered']['also_bought'])
    before = read_json(lmsps, 'review', 'evidence', 'audit_before_fix.json')
    after = read_json(lmsps, 'data', 'AMAZON', 'audit.json')
    meta = read_json(lmsps, 'data', 'AMAZON', 'meta.json')
    assert raw['review_lines'] == before['raw']['rows'], 'raw review lines disagree with the audit'

    put('raw-records', n(raw['blocks']), src)
    put('raw-usable', n(raw['usable']), src)
    put('raw-reviews', n(raw['review_lines']), src)
    put('raw-categories', n(raw['categories']), src)
    put('raw-pos-pct', pct(before['raw']['pos_frac']), 'audit_before_fix.json: share of review lines rated 4 or 5')
    put('raw-unhelpful-pct', f(100 * before['raw']['unhelpful_frac'], 2),
        'positive reviews dropped as unhelpful (>= 5 votes, < 20% helpful)')
    put('placeholder-ids', n(raw['placeholder_ids']), src)
    put('placeholder-lines', n(raw['placeholder_lines']), src)
    put('placeholder-main-lines', n(raw['placeholder_main']), src + ': ATVPDKIKX0DER')
    put('placeholder-second-lines', n(raw['placeholder_second']), src + ': A3UN6WX5RRO2AG')
    put('edition-pairs', n(raw['edition_pairs']), src)
    put('edition-clusters', n(raw['edition_clusters']), src)
    put('edition-products', n(raw['edition_products']), src)
    put('raw-works', n(raw['usable'] - raw['edition_products'] + raw['edition_clusters']),
        'usable products after merging editions')
    put('kcore-iterations', n(raw['kcore_iterations']), src)
    put('catpath-rows', n(raw['category_path_rows']), src)
    put('placeholder-main-max', n(ph['ATVPDKIKX0DER']['max_on_one_product']), 'most reviews on one product')
    put('placeholder-second-max', n(ph['A3UN6WX5RRO2AG']['max_on_one_product']))
    put('placeholder-found', n(len(found)), 'IDs found by the rule, not named in advance')
    put('placeholder-found-max-lo', n(min(v['max_on_one_product'] for v in found)), 'their most reviews on one product')
    put('placeholder-found-max-hi', n(max(v['max_on_one_product'] for v in found)))
    put('placeholder-found-lines', n(sum(v['review_lines'] for v in found)))
    put('edition-size-min', n(min(sizes)), 'products in an edition cluster')
    put('edition-size-max', n(max(sizes)))
    put('edition-size-two-pct', pct(sizes.get(2, 0) / sum(sizes.values())), 'clusters of exactly two products')
    st_ = rep['settings']
    put('rule-pos-rating', n(st_['pos_threshold']), 'a positive review is rated at least this')
    put('rule-helpful-votes', n(st_['helpful_min_votes']), 'a review with at least this many votes ...')
    put('rule-helpful-pct', pct(st_['helpful_min_ratio'], 0), '... and fewer helpful ones than this is dropped')
    put('rule-kcore', n(st_['k_core']))
    put('rule-edition-shared', n(st_['edition_min_shared']), 'identical review lines two editions must share ...')
    put('rule-edition-overlap-pct', pct(st_['edition_min_overlap'], 0), '... covering this share of the smaller one')
    put('rule-edition-maxrun', n(st_['edition_max_run']), 'review lines on more products than this are ignored')
    put('rule-placeholder-repeats', n(st_['placeholder_min_repeats']), 'reviews on one product that mark a placeholder')

    for tag, a in (('before', before), ('after', after)):
        c = a['counts']
        put(f'{tag}-products', n(c['products']))
        put(f'{tag}-categories', n(c['categories']))
        put(f'{tag}-users', n(c['users']))
        put(f'{tag}-edges', n(c['edges']))
        put(f'{tag}-maxdeg', n(a['user_degree']['max']), 'largest user degree')
        put(f'{tag}-meddeg', n(a['user_degree']['median']), 'median user degree')
        put(f'{tag}-clusters', n(a['duplicates']['clusters']), 'duplicate-edition clusters')
        put(f'{tag}-clustered', n(a['duplicates']['clustered_products']), 'products in those clusters')
        put(f'{tag}-clustered-pct', pct_auto(a['duplicates']['clustered_products'] / c['products']))
        put(f'{tag}-copyedges-pct', pct_auto(a['duplicates']['repeat_user_cluster_edges_frac']),
            'edges that copy an interaction already in the graph')
        put(f'{tag}-valleak-pct', pct_auto(a['val_dup_leak_frac']), 'validation edges with a training copy')
        put(f'{tag}-testleak-pct', pct_auto(a['test_dup_leak_frac']), 'test edges with a training copy')
        put(f'{tag}-valtitle-pct', pct_auto(a['val_title_leak_frac']),
            'validation edges whose user has a training edge to a same-title product')
        put(f'{tag}-testtitle-pct', pct_auto(a['test_title_leak_frac']), 'the same for test edges')
        put(f'{tag}-alsobought', n(a['also_bought']['undirected']), 'distinct also-bought product pairs')
    put('after-zerocat-pct', pct(after['categories']['zero_feature_categories_frac']),
        'categories with an all-zero name vector')
    put('after-leafcats-med', n(after['categories']['per_product_leaf_median']), 'leaf categories per product')
    put('title-coverage-pct', pct(after['title_lsa_coverage']), 'products with a non-zero title vector')
    put('title-dim', n(meta['title_dim']))
    put('title-vocab', n(meta['lsa_vocab']))
    put('cat-vocab', n(meta['cat_vocab']))
    put('user-bits', n(meta['user_feat_bits']))
    put('user-dim', n(meta['user_feat_dim']))
    rel = meta['relations']
    put('rel-pp-rows', n(rel['P->P also-bought']), 'also-bought rows per direction')
    put('rel-cc', n(rel['C->C parent-child']))
    put('rel-pc', n(rel['P->C belongs-to']))
    put('rel-up', n(rel['U->P reviewed']))
    return before, after, meta


def split_numbers(lmsps, meta, split_seed=1):
    """Rebuild the edge split exactly as hgb/link_data.py does and check its fingerprint."""
    section('The split (rebuilt as hgb/link_data.py builds it; checked against the fingerprint)')
    counts = meta['counts']
    user_shift = counts['products'] + counts['categories']
    src, dst = [], []
    with open(os.path.join(lmsps, 'data', 'AMAZON', 'link.dat'), encoding='utf-8') as fh:
        for line in fh:
            a, b, r, _ = line.split('\t')
            if r == '6':                                   # U -> P reviewed
                src.append(int(a))
                dst.append(int(b))
    edges = np.unique(np.stack([np.array(src) - user_shift, np.array(dst)], axis=1), axis=0)
    edges = edges[np.random.default_rng(split_seed).permutation(edges.shape[0])]
    total = edges.shape[0]
    n_val = n_test = int(total * 0.1)
    n_train = total - n_val - n_test
    train, test = edges[:n_train], edges[n_train + n_val:]
    train_products_global[:] = train[:, 1].tolist()
    fp = hashlib.blake2b(np.unique(train.astype(np.int64), axis=0).tobytes(), digest_size=8).hexdigest()
    if fp != FINGERPRINT:
        raise SystemExit(f'rebuilt split has fingerprint {fp}, the runs used {FINGERPRINT}: numbers would be wrong')
    label = {}
    for name in ('label.dat', 'label.dat.test'):
        with open(os.path.join(lmsps, 'data', 'AMAZON', name), encoding='utf-8') as fh:
            for line in fh:
                p = line.rstrip('\n').split('\t')
                label[int(p[0])] = int(p[3])
    assert len(label) == counts['products']
    put('split-train', n(n_train))
    put('split-val', n(n_val))
    put('split-test', n(n_test))
    put('split-seed', n(split_seed))
    put('split-fingerprint', FINGERPRINT)
    put('split-negatives', '99')
    deg = np.bincount(train[:, 0], minlength=counts['users'])
    put('train-deg-med', n(np.median(deg)), 'training edges per user')
    put('train-deg-mean', f(deg.mean(), 1))
    put('train-deg-max', n(deg.max()))
    put('test-users', n(len(np.unique(test[:, 0]))), 'users with at least one test edge')
    g = Counter(label[int(p)] for p in range(counts['products']))
    gt = Counter(label[int(p)] for p in test[:, 1])
    for i, name in enumerate(GROUPS):
        put(f'group-{name.lower()}', n(g[i]))
        put(f'group-{name.lower()}-pct', pct(g[i] / counts['products']))
        put(f'test-{name.lower()}-pct', pct(gt[i] / n_test), 'share of test edges')
    put('group-bookmusic-pct', pct((g[0] + g[2]) / counts['products']))
    put('test-bookmusic-pct', pct((gt[0] + gt[2]) / n_test), 'books and music, share of test edges')
    return n_test, label


def graph_numbers(lmsps, han, label):
    """Category links, near-empty paths, edition twins: the facts the Data section states about the graph."""
    section('The graph: category links, near-empty paths and edition twins (audits, link.dat, HAN logs)')
    before = read_json(lmsps, 'review', 'evidence', 'audit_before_fix.json')
    after = read_json(lmsps, 'data', 'AMAZON', 'audit.json')
    cat = after['categories']
    put('before-topuser-pct', pct(before['top_user_degree'] / before['counts']['products'], 0),
        'share of products the largest user had reviewed before the fix')
    put('raw-shared-pct', pct(before['raw']['shared_signature_frac']),
        'raw review lines that also appear under another product')
    put('cat-all-median', n(cat['per_product_all_median']), 'categories per product if every ancestor were linked')
    put('cat-all-largest-pct', pct(cat['largest_category_frac']), 'products in one category, every ancestor linked')
    put('cat-pcp-ratio', f(cat['pcp_walks_over_P2_all'] / cat['pcp_walks_over_P2_leaf'], 1),
        'P-C-P walks with every ancestor linked, over leaves only')
    # PCCP and PCP on the full graph (link.dat): does the path reach any product but the start?
    products = after['counts']['products']
    pc, cp, cc = {}, {}, {}
    with open(os.path.join(lmsps, 'data', 'AMAZON', 'link.dat'), encoding='utf-8') as fh:
        for line in fh:
            a, b, r, _ = line.split('\t')
            if r == '4':                                    # P -> C, leaf category
                pc.setdefault(int(a), set()).add(int(b))
                cp.setdefault(int(b), set()).add(int(a))
            elif r in ('2', '3'):                           # C - C, both directions
                cc.setdefault(int(a), set()).add(int(b))
                cc.setdefault(int(b), set()).add(int(a))
    alone = lambda p, cats: all(cp.get(c, set()) <= {p} for c in cats)
    pcp_iso = [p for p in range(products) if alone(p, pc.get(p, ()))]
    pccp_iso = [p for p in range(products) if alone(p, {c2 for c in pc.get(p, ()) for c2 in cc.get(c, ())})]
    pccp_iso_global[:] = pccp_iso
    put('pcp-isolated', n(len(pcp_iso)), 'products whose leaf categories hold no other product')
    iso_set = set(pccp_iso)
    reach_train = sum(p not in iso_set for p in train_products_global) / len(train_products_global)
    put('pccp-reach-train-pct', pct(reach_train), 'training edges whose product PCCP links to another product')
    put('pccp-isolated-pct', pct(len(pccp_iso) / products), 'products PCCP links to no other product (full graph)')
    g = Counter(label[p] for p in range(products))
    gi = Counter(label[p] for p in pccp_iso)
    for i, name in enumerate(GROUPS):
        put(f'pccp-reach-{name.lower()}-pct', pct(1 - gi[i] / g[i]),
            f'share of {name} products that PCCP links to another product')
    # the same in HAN's graphs (top-20 per hop): the graph-bias audit of the HAN repository
    with open(os.path.join(han, 'output', 'AMAZON', 'v3', 'logs', 'audit_graph_bias.txt'), encoding='utf-8') as fh:
        iso = {m.group(1): int(m.group(2)) for m in
               re.finditer(r'^(P[PCU]*P)\s+[\d.]+\s+\d+\s+(\d+)\s', fh.read(), re.M)}
    assert len(iso) == 32, len(iso)
    shares = [iso[p] / products for p in EMPTY]
    put('empty-isolated-min-pct', pct(min(shares)), 'near-empty paths: products without a neighbour, HAN graphs')
    put('empty-isolated-max-pct', pct(max(shares)))
    put('pccp-graph-isolated-pct', pct(iso['PCCP'] / products), 'PCCP in the HAN graph (pruning adds a few)')
    put('pcccp-graph-isolated', n(iso['PCCCP']))
    put('pcp-graph-isolated', n(iso['PCP']))
    put('pp-isolated-pct', pct(iso['PP'] / products), 'products without an also-bought partner')
    # products with an identical reviewer set (HAN repository, check_edition_twins.py)
    with open(os.path.join(han, 'output', 'AMAZON', 'v3', 'logs', 'audit_edition_twins.txt'), encoding='utf-8') as fh:
        txt = fh.read()
    m = re.search(r'identical reviewer set elsewhere: (\d+) \([\d.]+%\) in (\d+) groups', txt)
    put('twin-products', n(int(m.group(1))), 'products with an identical reviewer set elsewhere')
    put('twin-groups', n(int(m.group(2))))
    put('twin-edges', n(int(re.search(r'U-P edges on such products: (\d+) of', txt).group(1))))
    put('twin-val', n(int(re.search(r'val edges whose user has a TRAIN edge .*?: (\d+) of', txt).group(1))))
    put('twin-test', n(int(re.search(r'test edges whose user has a TRAIN edge .*?: (\d+) of', txt).group(1))))
    # an example from the raw file (LMSPS review part 1)
    with open(os.path.join(lmsps, 'review', 'part1_data_pipeline.md'), encoding='utf-8') as fh:
        m = re.search(r'on DVD and VHS share (\d+) of (\d+) review lines', fh.read())
    put('example-shared', n(int(m.group(1))), 'review lines a DVD and a VHS of one film share (review part 1)')
    put('example-lines', n(int(m.group(2))))


# ------------------------------------------------------------------------------------------------
# the search
# ------------------------------------------------------------------------------------------------
def search_numbers(lmsps, out):
    section('The search: 30 runs per set size (LMSPS notes, Parts 1 and 5-9)')
    fleet = os.path.join(lmsps, 'hgb', 'logs', 'Fleet v3')
    rf = load_module('reselect_fleet', os.path.join(lmsps, 'hgb', 'reselect_fleet.py'))
    recs, rank, gaps, picks = {}, {}, {}, []
    total_s = 0.0
    for ns in (4, 6):
        files = sorted(glob.glob(os.path.join(fleet, f'final_ns{ns}', 'search_*.json')))
        recs[ns] = [read_json(x) for x in files]
        assert len(recs[ns]) == 30 and all(d['data']['train_edges_hash'] == FINGERPRINT for d in recs[ns])
        # every run has the same settings and environment; only the seed and the set size differ
        free = ('ns', 'seed', 'seeds', 'search_json')
        ref = {k: v for k, v in recs[4][0]['args'].items() if k not in free}
        for d in recs[ns]:
            assert {k: v for k, v in d['args'].items() if k not in free} == ref, d['seed']
            assert d['env'] == recs[4][0]['env'], d['seed']
        if ns == 4:
            a = recs[4][0]['args']
            put('search-cands', n(len(recs[4][0]['candidates'])))
            put('search-pool', n(a['num_final']))
            put('search-sets', n(len(recs[4][0]['eval_table'])), 'sets scored per run (10 choose 4 = 10 choose 6)')
            put('search-epochs', n(a['stage']))
            put('search-warmup', n(a['warmup_epochs']))
            put('search-hops', n(a['num_hops']))
            put('search-hidden', n(a['hidden']))
            put('search-batch', n(a['batch_size']))
            put('search-dropout', f(a['dropout'], 1))
            put('search-lr', f'{a["lr"]:g}')
            put('search-tau-max', n(a['tau_max']))
            put('search-tau-min', n(a['tau_min']))
            put('search-delta', f(a['report_delta'], 3))
            put('search-folds', n(a['wstep_kfold']), 'folds of training edges for the weight step')
            put('search-alr', f'{a["alr"]:g}', 'learning rate of the alpha step')
            put('search-weight-decay', f'{a["weight_decay"]:g}')
            put('search-input-drop', f'{a["input_drop"]:g}')
            # The settings the appendix states in words (Appendix B): the same in every record.
            assert a['half_feats'] and a['deterministic'] and a['eval_no_gumbel'], a
            assert a['alpha_edges'] == 'val' and not a['same_type_loops'] and a['user_tower'] == 'embedding', a
            assert a['eval_exhaustive_cap'] >= len(recs[4][0]['eval_table']), a
            env = recs[4][0]['env']
            put('env-gpu', env['device'].replace('NVIDIA GeForce ', ''), 'the run records')
            put('env-gpu-mem', '12', "GB, the card's specification")
            put('env-python', env['python'])
            put('env-torch', env['torch'].split('+')[0])
            put('env-cuda', env['torch'].split('+cu')[1][:-1] + '.' + env['torch'].split('+cu')[1][-1])
            put('env-dgl', env['dgl'])
            put('env-numpy', env['numpy'])
            put('search-runs', '30')
            put('search-runs-total', '60')
            txt = open(files[0].replace('.json', '.txt'), encoding='utf-8', errors='replace').read()
            put('search-cands-raw', re.search(r'--tgt-ended-paths: (\d+) candidates', txt).group(1),
                'P...P paths of at most 5 hops before merging reversed pairs')
        # alpha rank of every candidate (1 = most preferred), mean over the 30 runs
        for d in recs[ns]:
            order = sorted(d['alpha'], key=lambda k: -d['alpha'][k])
            for p in d['candidates']:
                rank.setdefault((p, ns), []).append(order.index(p) + 1)
        # run time: 150 epochs at the logged mean epoch time, plus the logged feature propagation
        ep = [d['avg_epoch_time_s'] for d in recs[ns]]
        prop = []
        for x in files:
            txt = open(x.replace('.json', '.txt'), encoding='utf-8', errors='replace').read()
            h, m, s = re.search(r'Time used for feat prop (\d+):(\d\d):([\d.]+)', txt).groups()
            prop.append(int(h) * 3600 + int(m) * 60 + float(s))
        run_s = [e * d['args']['stage'] + p for e, d, p in zip(ep, recs[ns], prop)]
        total_s += sum(run_s)
        put(f'search-epoch-s-{ns}', f(st.median(ep), 1), 'median seconds per training epoch')
        put(f'search-prop-s-{ns}', n(round(st.median(prop))), 'median seconds of feature propagation')
        put(f'search-run-min-{ns}', n(round(st.median(run_s) / 60)), 'median minutes per run (epochs + propagation)')
        put(f'search-fleet-h-{ns}', f(sum(run_s) / 3600, 1), 'GPU hours for the 30 runs')

        # one run cannot rank its top sets
        winners = Counter(tuple(sorted(min(d['eval_table'], key=lambda e: e['val_bpr'])['paths'])) for d in recs[ns])
        put(f'search-winners-{ns}', n(len(winners)), 'distinct winners among the 30 runs')
        sel = read_json(fleet, f'final_ns{ns}', 'reselect_d010.json')
        alt = read_json(fleet, f'final_ns{ns}', 'reselect_d005.json')
        if ns == 4:
            put('search-delta-alt', f(alt['delta'], 3), 'the second tolerance of the vote (sensitivity check)')
        ties = []
        for d in recs[ns]:
            best = min(e['val_bpr'] for e in d['eval_table'])
            ties.append(sum(e['val_bpr'] - best <= sel['delta'] for e in d['eval_table']))
        assert sorted(ties) == sorted(sel['tie_group_sizes'])
        put(f'search-tie-med-{ns}', n(st.median(ties)), 'median tie-group size (delta 0.010)')
        put(f'search-tie-min-{ns}', n(min(ties)))
        put(f'search-tie-max-{ns}', n(max(ties)))
        put(f'search-tie-seed1-{ns}', n(ties[[d['seed'] for d in recs[ns]].index(1)]), 'tie group of seed 1')
        put(f'search-noise-{ns}', f(sel['score_noise'], 4), 'score noise of one set')
        put(f'search-agree-{ns}', sg(sel['rank_agreement'], 2), 'Spearman of two runs on shared sets')
        put(f'search-xnoise-{ns}', f(sel['cross_seed_noise'], 4), 'gap of a winner re-scored in another run')
        for tag, s in (('', sel), ('-alt', alt)):
            null = sorted(s['null_distinct'])
            vote = len(s['reselected_unique'])
            put(f'search-vote{tag}-{ns}', n(vote), 'distinct vote picks')
            put(f'search-null-med{tag}-{ns}', n(st.median(null)))
            put(f'search-null-lo{tag}-{ns}', n(null[len(null) // 20]), '90% range, as reselect_fleet.py')
            put(f'search-null-hi{tag}-{ns}', n(null[-(len(null) // 20) - 1]))
            put(f'search-null-share{tag}-{ns}', pct(sum(x <= vote for x in null) / len(null)),
                'null fleets with as few distinct picks')
        put(f'search-cons-count-{ns}', n(sel['consensus_count']), 'runs whose vote pick is the consensus')
        if ns == 4:
            put('search-null-fleets', n(len(sel['null_distinct'])), 'shuffled-score fleets in the null')
        # averaged scores with the one-standard-error rule
        runs = rf.load_runs([os.path.join(fleet, f'final_ns{ns}', 'search_*.json')])
        seen = {s: v for s, v in rf.averaged_scores(runs).items() if v[1] >= 5}
        best = min(seen, key=lambda s: seen[s][0])
        se = sel['score_noise'] / seen[best][1] ** 0.5
        put(f'search-avg-sets-{ns}', n(len(seen)), 'sets scored in at least 5 runs')
        put(f'search-se-{ns}', f(se, 4), 'standard error of the best averaged set')
        window = [s for s in seen if seen[s][0] <= seen[best][0] + se]
        assert len(window) == len(sel['averaged_one_se'])
        put(f'search-window-{ns}', n(len(window)), 'sets within one SE of the best averaged score')

        # F2: seed 1's 210 scores, sorted, as gaps to its best
        d1 = recs[ns][[d['seed'] for d in recs[ns]].index(1)]
        v = sorted(e['val_bpr'] for e in d1['eval_table'])
        gaps[ns] = [x - v[0] for x in v]
        # F2: distinct picks
        null = sorted(sel['null_distinct'])
        picks.append((ns, len(winners), len(sel['reselected_unique']), st.median(null),
                      null[len(null) // 20], null[-(len(null) // 20) - 1]))
    # the pilot: two runs with the same seed must give the same record (Appendix D)
    pilot = [read_json(fleet, 'pilot', f'pilot_{x}.json') for x in 'ab']
    for k in ('alpha', 'final_pool', 'eval_table'):
        assert pilot[0][k] == pilot[1][k], f'the two same-seed pilot runs differ in {k}'
    put('pilot-sets', n(len(pilot[0]['eval_table'])), 'sets scored in each same-seed pilot run, all identical')
    put('search-total-h', f(total_s / 3600, 0), 'GPU hours for all 60 runs (epochs + propagation)')
    rank = {k: st.mean(v) for k, v in rank.items()}
    for p in ('PUPUP', 'PUPPUP', 'PUP', 'PCP', 'PCCCP', 'PCPCP', 'PPCCPP'):
        for ns in (4, 6):
            put(f'rank-{p.lower()}-{ns}', f(rank[(p, ns)], 1), f'mean alpha rank of {p} (1 = most preferred of 36)')

    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    with open(os.path.join(out, 'figures', 'f2_scores.dat'), 'w', encoding='utf-8') as fh:
        fh.write('% one run (seed 1) per set size: its 210 sets, best first; gap = validation BPR minus the best\n')
        fh.write('rank gapfour gapsix\n')
        for i, (a, b) in enumerate(zip(gaps[4], gaps[6])):
            fh.write(f'{i + 1} {a:.5f} {b:.5f}\n')
    with open(os.path.join(out, 'figures', 'f2_picks.dat'), 'w', encoding='utf-8') as fh:
        fh.write('% distinct picks among 30 runs; null = every run\'s scores shuffled (200 fleets), median and 90% range\n')
        fh.write('size own vote nullmed nulllo nullhi errlo errhi\n')
        for ns, own, vote, med, lo, hi in picks:
            fh.write(f'{ns} {own} {vote} {med:g} {lo} {hi} {med - lo:g} {hi - med:g}\n')
    return rank


# ------------------------------------------------------------------------------------------------
# HAN, metapath2vec, GAT and popularity
# ------------------------------------------------------------------------------------------------
def parse_log(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    lines = re.split(r'[\r\n]+', txt)
    r = {'file': os.path.basename(path)[:-4]}
    ts = [datetime.datetime.strptime(x, '%Y-%m-%d_%H:%M:%S')
          for x in re.findall(r'(?:Start|End) Time: (\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)', txt)]
    r['minutes'] = (ts[1] - ts[0]).total_seconds() / 60 if len(ts) >= 2 else None
    r['exit'] = int((re.search(r'exit code (\d+)', txt) or [None, -1])[1])
    r['fp'] = (re.search(r'training-edge fingerprint (\w+)', txt) or [None, None])[1]
    m = re.search(r'# Params: (\d+)', txt)
    r['params'] = int(m.group(1)) if m else None
    m = re.search(r'Semantic attention over meta-paths:\n((?:\s+\S+: [\d.]+\n)+)', txt)
    r['beta'] = {a: float(b) for a, b in re.findall(r'(\S+): ([\d.]+)', m.group(1))} if m else {}
    r['epochs'] = [float(v) for v in re.findall(r'^Epoch [\d.]+, \S+ \S+ [\d.]+, val bpr ([\d.]+)', txt, re.M)]
    ns = next((x for x in lines if x.startswith('Namespace(')), '')
    w = re.search(r'\bbeta_warmup=(\d+)', ns)
    r['warmup'] = int(w.group(1)) if w else 0
    m = re.search(r'Early stop at epoch ([\d.]+)', txt)
    r['stop'] = float(m.group(1)) if m else None
    m = re.search(r'--(?:stage|epochs) (\d+)', lines[0])
    r['cap'] = int(m.group(1)) if m else None
    blk = txt.split('By user training degree:')[1].split('By duplicate')[0] if 'By user training degree:' in txt else ''
    r['deg'] = {k.strip(): (int(e), float(h)) for k, e, h in
                re.findall(r'(deg[^(]*)\(\s*(\d+) edges\)\s+NDCG@10 [\d.]+\s+Hit@10 ([\d.]+)', blk)}
    m = re.search(r'Test ranking metrics:(.*?)Per-user', txt, re.S)
    r['hit10'] = float(re.search(r'Hit@10: ([\d.]+)', m.group(1)).group(1)) if m else None
    return r


def gain_last10(r):
    best, run = [], float('inf')
    for v in r['epochs'][r['warmup']:]:
        run = min(run, v)
        best.append(run)
    return best[-11] - best[-1] if len(best) > 10 else None


def short(arch):
    m = re.match(r'amazon_lp_(?:v3_f(\d)_(a\d|c)|manual_(\d)|rand(\d)_d(\d))$', arch)
    if not m:
        return None
    if m.group(1):
        return f'f{m.group(1)}{m.group(2)}'
    if m.group(3):
        return f'm{m.group(3)}'
    return f'r{m.group(4)}d{m.group(5)}'


def family(arch):
    return 'searched' if '_v3_' in arch else 'random' if 'rand' in arch else 'manual'


def model_numbers(han, rank, out, n_test):
    v3 = os.path.join(han, 'output', 'AMAZON', 'v3')
    logdir = os.path.join(v3, 'logs')
    with open(os.path.join(v3, 'results.csv'), encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    arch = load_module('han_arch', os.path.join(han, 'arch.py'))
    sets = {k: v[0] for k, v in arch.archs.items() if short(k)}
    assert len(sets) == 24
    tag = {'HAN': 'han', 'metapath2vec': 'mtv', 'GAT': 'gat', 'popularity': 'pop'}
    logs = {}
    for p in sorted(glob.glob(os.path.join(logdir, '*_s*.txt'))):
        r = parse_log(p)
        if r['exit'] != 0 and not r['file'].startswith(('smoke', 'check', 'dry')):
            raise SystemExit(f'{p}: exit code {r["exit"]}')
        logs[r['file']] = r
    logname = lambda m, a, s: {'HAN': f'han_{a}_s{s}', 'metapath2vec': f'm2v_{a}_s{s}',
                               'GAT': f'gat_union_s{s}', 'popularity': f'popularity_s{s}'}[m]
    by = {}
    for x in rows:
        assert x['data'] == FINGERPRINT
        by.setdefault((x['model'], x['arch']), []).append(x)
    vals = lambda m, a, c: [float(x[c]) for x in sorted(by[(m, a)], key=lambda x: int(x['seed']))]
    seed1 = lambda m, a: next(x for x in by[(m, a)] if x['seed'] == '1')

    section('Settings: the commands in the seed-1 logs of M5 (HAN, metapath2vec) and of GAT')
    def flags(name):
        with open(os.path.join(logdir, name), encoding='utf-8', errors='replace') as fh:
            cmd = fh.readline()
        return dict(re.findall(r'--([\w-]+) (\S+)', cmd))
    num = lambda v: n(float(v)) if float(v) >= 1000 else f'{float(v):g}'
    for key, name, keep in (
            ('han', 'han_amazon_lp_manual_5_s1.txt', ('topk', 'hidden', 'num-heads', 'feat-drop', 'attn-drop', 'lr',
                                                     'weight-decay', 'batch-size', 'stage', 'patience', 'min-delta',
                                                     'beta-warmup')),
            ('mtv', 'm2v_amazon_lp_manual_5_s1.txt', ('dim', 'walks-per-node', 'walk-length', 'window', 'negatives',
                                                     'lr', 'epochs', 'evals-per-epoch', 'patience', 'min-delta',
                                                     'batch-walks')),
            ('gat', 'gat_union_s1.txt', ('in-dim', 'hidden', 'num-heads', 'layers', 'feat-drop', 'attn-drop', 'lr',
                                         'weight-decay', 'batch-size', 'stage', 'patience', 'min-delta'))):
        fl = flags(name)
        for k in keep:
            put(f'{key}-set-{k}', num(fl[k]), f'--{k}')
    fl = flags('han_amazon_lp_manual_5_s1.txt')
    put('han-set-head-dim', n(int(fl['hidden']) // int(fl['num-heads'])), 'dimensions per attention head')
    txt = open(os.path.join(logdir, 'han_amazon_lp_manual_5_s1.txt'), encoding='utf-8', errors='replace').read()
    m = re.search(r'\(semantic\): SemanticAttention\(\s*\(project\): Sequential\(\s*\(0\): Linear\(in_features=(\d+), '
                  r'out_features=(\d+)', txt)
    assert int(m.group(1)) == int(fl['hidden']), m.groups()
    put('han-set-sem-dim', m.group(2), 'semantic attention vector (model print in the log)')
    put('han-set-in-dim', re.search(r'\(fc\): Linear\(in_features=(\d+),', txt).group(1),
        'product input features: LSA title vector and product group')
    put('han-set-semantic-tau', f"{float(re.search(r'semantic_tau=([\d.]+)', txt).group(1)):g}", 'softmax temperature')
    for x in glob.glob(os.path.join(logdir, 'han_*_s*.txt')):           # the same defaults in every HAN run
        t = open(x, encoding='utf-8', errors='replace').read()
        for flag in ('semantic_tau=1.0', 'independent_init=False', "user_tower='embedding'"):
            assert flag in t, (x, flag)

    section('Path sets (HAN repository, arch.py)')
    put('runs-total', n(len(rows)), 'result rows: every model, set and seed')
    cands = {p for (p, _) in rank}
    used = {p for ps in sets.values() for p in ps}
    assert used <= cands, used - cands
    put('sets-distinct-paths', n(len(used)), 'distinct paths over the 24 sets, all among the candidates')
    put('sets-size-min', n(min(len(ps) for ps in sets.values())), 'paths per set, over the 24 sets')
    put('sets-size-max', n(max(len(ps) for ps in sets.values())))
    put('sets-hops-max', n(max(len(p) - 1 for p in used)), 'longest path in the 24 sets, hops')
    put('perm-shuffles', n(20000), 'permutation tests of the Spearman correlations')
    for a, ps in sets.items():
        put(f'set-{short(a)}', ' '.join(ps), a)
        put(f'set-{short(a)}-n', n(len(ps)))
        put(f'set-{short(a)}-hops', n(sum(len(p) - 1 for p in ps)))

    section('Every set at seed 1: HAN and metapath2vec (results.csv)')
    digits = {'hit': 3, 'ndcg': 3, 'fhit': 4, 'fndcg': 4, 'frec': 4, 'val': 4}
    for m in ('HAN', 'metapath2vec'):
        for a in sets:
            x = seed1(m, a)
            for key, col in METRICS:
                put(f'{tag[m]}-{short(a)}-{key}-s1', f(float(x[col]), digits[key]))

    section('The main comparison: 3 seeds, Welch tests (HAN notes, Part 10)')
    main = {4: ('amazon_lp_v3_f4_a1', 'amazon_lp_manual_5'), 6: ('amazon_lp_v3_f6_a1', 'amazon_lp_manual_6')}
    for m in ('HAN', 'metapath2vec'):
        for ns, (a, b) in main.items():
            for key, col in METRICS[:5]:
                x, y = vals(m, a, col), vals(m, b, col)
                assert len(x) == len(y) == 3
                for k, z in ((short(a), x), (short(b), y)):
                    if f'{tag[m]}-{k}-{key}-mean' not in SEEN:
                        put(f'{tag[m]}-{k}-{key}-mean', f(st.mean(z), digits[key]))
                        put(f'{tag[m]}-{k}-{key}-sd', f(st.stdev(z), digits[key]))
                diff, p = welch(x, y)
                put(f'{tag[m]}-{ns}-{key}-diff', sg(diff, digits[key]), f'{short(a)} minus {short(b)}')
                put(f'{tag[m]}-{ns}-{key}-rel', sg(100 * diff / st.mean(y), 1), 'relative difference, %')
                put(f'{tag[m]}-{ns}-{key}-p', pval(p), 'Welch, two-sided')
                put(f'{tag[m]}-{ns}-{key}-pt', ptext(pval(p)))
                put(f'{tag[m]}-{ns}-{key}-diffabs', f(abs(diff), digits[key]))
                put(f'{tag[m]}-{ns}-{key}-relabs', f(abs(100 * diff / st.mean(y)), 1))
                put(f'{tag[m]}-{ns}-{key}-sdx', f(abs(diff) / max(st.stdev(x), st.stdev(y)), 0),
                    'difference in units of the larger seed SD')

    twins = {int(x['n_twin']) for x in rows if x['n_twin']}
    assert twins == {40}, twins
    put('twin-maxdiff', f(max(abs(float(x['hit10']) - float(x['hit10_notwin'])) for x in rows if x['hit10_notwin']), 4),
        'largest change of test Hit@10 when the twin edges are left out, any run')

    sds = [st.stdev(vals(m, a, 'hit10')) for m in ('HAN', 'metapath2vec') for pair in main.values() for a in pair]
    put('main-hit-sd-max', f(max(sds), 3), 'largest seed SD of test Hit@10 among the main runs')

    section('GAT and popularity')
    for key, col in METRICS[:5]:
        z = vals('GAT', '-', col)
        put(f'gat-{key}-mean', f(st.mean(z), digits[key]))
        put(f'gat-{key}-sd', f(st.stdev(z), digits[key]))
        put(f'pop-{key}', f(float(by[('popularity', '-')][0][col]), digits[key]))
    z = vals('GAT', '-', 'val_bpr')
    put('gat-val-mean', f(st.mean(z), digits['val']), 'validation BPR, mean of the 3 seeds')
    put('gat-val-sd', f(st.stdev(z), digits['val']))

    section('By the user\'s number of training edges: Hit@10, mean over the seeds run (HAN logs)')
    for g, label in DEG:
        put(f'testdeg-{g}', n(logs['popularity_s1']['deg'][label][0]), 'test edges in the group')
    assert sum(logs['popularity_s1']['deg'][lab][0] for _, lab in DEG) == n_test

    def deg_mean(m, a):
        rs = [logs[logname(m, a, x['seed'])] for x in by[(m, a)]]
        return {g: st.mean(r['deg'][lab][1] for r in rs) for g, lab in DEG}
    rows_deg = [('HAN', a) for a in sets] + [('metapath2vec', a) for a in sets] + [('GAT', '-'), ('popularity', '-')]
    degs = {}
    for m, a in rows_deg:
        degs[(m, a)] = deg_mean(m, a)
        k = short(a) if a != '-' else None
        for g, _ in DEG[1:]:
            put(f'{tag[m]}-{k}-hit-{g}' if k else f'{tag[m]}-hit-{g}', f(degs[(m, a)][g], 3))
    for ns, (a, b) in main.items():
        for g, _ in DEG[1:]:
            put(f'han-{ns}-hit-{g}-diff', sg(degs[('HAN', a)][g] - degs[('HAN', b)][g], 3))
    for g, _ in DEG[1:]:
        put(f'gat-lead-{g}', sg(degs[('GAT', '-')][g] - degs[('HAN', 'amazon_lp_rand6_d2')][g], 3),
            'GAT minus the HAN validation pick')

    section('All 24 sets at seed 1: families, order, agreement (HAN notes, Part 10)')
    han1 = {a: seed1('HAN', a) for a in sets}
    mtv1 = {a: seed1('metapath2vec', a) for a in sets}
    for m, res in (('han', han1), ('mtv', mtv1)):
        for ns in (4, 6):
            rnd = [a for a in sets if f'rand{ns}' in a]
            for key, col in (('hit', 'hit10'), ('fhit', 'hit10_full')):
                z = [float(res[a][col]) for a in rnd]
                put(f'{m}-r{ns}-{key}-mean', f(st.mean(z), digits[key]))
                put(f'{m}-r{ns}-{key}-min', f(min(z), digits[key]))
                put(f'{m}-r{ns}-{key}-max', f(max(z), digits[key]))
            cand = [a for a in sets if family(a) != 'manual' and (f'rand{ns}' in a or f'_f{ns}_' in a)]
            for key, col, rev in (('hit', 'hit10', True), ('val', 'val_bpr', False), ('fhit', 'hit10_full', True)):
                order = sorted(cand, key=lambda a: float(res[a][col]), reverse=rev)
                pos = [i + 1 for i, a in enumerate(order) if family(a) == 'searched']
                put(f'{m}-{ns}-pos-{key}', ', '.join(ordinal(k) for k in pos[:-1]) + ' and ' + ordinal(pos[-1]),
                    f'places of the searched sets among the {len(order)} candidate-drawn sets')
    for ns, b in ((4, 'amazon_lp_manual_5'), (6, 'amazon_lp_manual_6')):
        rnd = st.mean(float(han1[a]['hit10']) for a in sets if f'rand{ns}' in a)
        man = st.mean(vals('HAN', b, 'hit10'))
        put(f'han-r{ns}-vs-manual-diff', sg(rnd - man, 3), 'mean of the 5 draws (seed 1) minus the manual set (3 seeds)')
        put(f'han-r{ns}-vs-manual-rel', sg(100 * (rnd - man) / man, 1), '%')
    a24 = list(sets)
    put('han-valtest-rho', ms(spearman([float(han1[a]['val_bpr']) for a in a24], [float(han1[a]['hit10']) for a in a24]), 2))
    put('han-valfull-rho', ms(spearman([float(han1[a]['val_bpr']) for a in a24], [float(han1[a]['hit10_full']) for a in a24]), 2))
    a23 = [a for a in a24 if a != 'amazon_lp_manual_4']
    put('han-mtv-rho', sg(spearman([float(han1[a]['hit10']) for a in a23], [float(mtv1[a]['hit10']) for a in a23]), 2),
        'HAN against metapath2vec over the 23 sets (M4 left out)')
    put('han-wins-sampled', n(sum(float(han1[a]['hit10']) > float(mtv1[a]['hit10']) for a in a24)), 'of 24 sets')
    put('mtv-wins-full', n(sum(float(mtv1[a]['hit10_full']) > float(han1[a]['hit10_full']) for a in a23)), 'of 24 sets')
    pick = min(a24, key=lambda a: float(han1[a]['val_bpr']))
    put('han-valpick', short(pick), 'lowest validation BPR among the 24 HAN sets')
    mpick = min(a23, key=lambda a: float(mtv1[a]['val_bpr']))
    put('mtv-valpick', short(mpick), 'lowest validation BPR among the 23 metapath2vec sets')
    for ns in (4, 6):
        sub = [a for a in a24 if len(sets[a]) == ns]
        dense = lambda a: any('C' in p and p not in EMPTY for p in sets[a])
        yes = [float(han1[a]['hit10']) for a in sub if dense(a)]
        no = [float(han1[a]['hit10']) for a in sub if not dense(a)]
        put(f'han-{ns}-dense-min', f(min(yes), 3))
        put(f'han-{ns}-dense-max', f(max(yes), 3))
        put(f'han-{ns}-nodense-min', f(min(no), 3))
        put(f'han-{ns}-nodense-max', f(max(no), 3))

    for ns in (4, 6):
        sub = [a for a in a24 if len(sets[a]) == ns]
        z = [float(han1[a]['hit10']) for a in sub]
        put(f'han-{ns}-span', f(max(z) - min(z), 3), f'range of HAN test Hit@10 over all {ns}-path sets, seed 1')
        put(f'han-{ns}-nsets', n(len(sub)))
        srch = [float(han1[a]['hit10']) for a in sub if family(a) == 'searched']
        put(f'han-searched{ns}-mean-s1', f(st.mean(srch), 3), 'mean of the three searched sets, seed 1')
    srch_sets = [ps for a, ps in sets.items() if family(a) == 'searched']
    for path in ('PUPUP', 'PUPPUP'):
        put(f'searched-with-{path.lower()}', n(sum(path in ps for ps in srch_sets)), f'searched sets holding {path}')
    empt = [sum(p in EMPTY for p in ps) for a, ps in sets.items() if family(a) == 'searched']
    put('searched-empty-min', n(min(empt)), 'near-empty category paths per searched set')
    put('searched-empty-max', n(max(empt)))

    section('What HAN does with the paths: weights at seed 1 against the search (LMSPS part 4b)')
    beta = {a: logs[f'han_{a}_s1']['beta'] for a in sets}
    assert all(set(beta[a]) == set(ps) for a, ps in sets.items())
    kind = lambda p: ('empty' if p in EMPTY else 'users' if 'U' in p else 'cat' if 'C' in p else 'pp')
    use = {}
    for a, b in beta.items():
        for p, x in b.items():
            use.setdefault(p, []).append(x * len(b))
    for kd in ('users', 'empty', 'cat', 'pp'):
        ps = [p for p in use if kind(p) == kd]
        put(f'kind-{kd}-paths', n(len(ps)))
        put(f'kind-{kd}-rank', f(st.mean((rank[(p, 4)] + rank[(p, 6)]) / 2 for p in ps), 1))
        put(f'kind-{kd}-uses', n(sum(len(use[p]) for p in ps)))
        put(f'kind-{kd}-dropped', n(sum(1 for a in beta for p, x in beta[a].items() if kind(p) == kd and x < DROP)))
        put(f'kind-{kd}-weight', f(st.median(x for p in ps for x in use[p]), 2))
    put('han-dropped', n(sum(1 for a in beta for x in beta[a].values() if x < DROP)), 'weights below 0.01, 24 sets')
    put('weight-drop', f'{DROP:g}', 'a path weight below this counts as dropped')
    keep = {a: max(x for p, x in beta[a].items() if p in EMPTY) for a in sets if family(a) == 'searched'}
    rest = [x for a, x in keep.items() if a != 'amazon_lp_v3_f4_c']
    put('han-empty-kept-min', f(min(rest), 2), 'largest near-empty-path weight (beta) in a searched set, f4c aside')
    put('han-empty-kept-max', f(max(rest), 2))
    put('han-empty-kept-f4c', f(keep['amazon_lp_v3_f4_c'], 2))
    for p in ('PUPUP', 'PUPPUP', 'PUP', 'PCP', 'PPCCPP'):
        put(f'weight-{p.lower()}', f(st.mean(use[p]), 2), f'mean weight of {p} as a multiple of an equal share')
        put(f'uses-{p.lower()}', n(len(use[p])))
    table = []
    for a, ps in sets.items():
        ns = 4 if len(ps) <= 4 else 6 if len(ps) == 6 else None
        score = st.mean(rank[(p, ns)] if ns else (rank[(p, 4)] + rank[(p, 6)]) / 2 for p in ps)
        table.append(dict(a=a, family=family(a), search=score, han=float(han1[a]['hit10']),
                          val=float(han1[a]['val_bpr']), full=float(han1[a]['hit10_full']), m2v=float(mtv1[a]['hit10'])))
    cand = [t for t in table if t['family'] != 'manual']
    for key, sub, col in (('cand', cand, 'han'), ('cand-val', cand, 'val'), ('cand-full', cand, 'full'),
                          ('random', [t for t in table if t['family'] == 'random'], 'han'),
                          ('all', table, 'han'),
                          ('mtv', [t for t in table if t['a'] != 'amazon_lp_manual_4'], 'm2v')):
        r0, p = perm_spearman([t['search'] for t in sub], [t[col] for t in sub])
        put(f'pref-{key}-rho', sg(r0, 2), f'search preference against {col}, n = {len(sub)}')
        ps = f'{p:.3f}' if p < 0.1 else f(p, 2)
        put(f'pref-{key}-p', ps, 'permutation p, 20,000 shuffles, Random(0)')
        put(f'pref-{key}-pt', ptext(ps))
        put(f'pref-{key}-n', n(len(sub)))

    # F4: one point per path, by kind; selected paths labelled (anchor = side of the label)
    f4_labels = {'PUPUP': 'south', 'PUPPUP': 'north', 'PUP': 'east', 'PPCCPP': 'east', 'PCCPCP': 'east',
                 'PCCP': 'south', 'PCCCP': 'north', 'PCP': 'east', 'PCPCP': 'east', 'PP': 'east', 'PPP': 'west'}
    pos = {p: ((rank[(p, 4)] + rank[(p, 6)]) / 2, st.mean(use[p])) for p in use}
    for kd in ('users', 'empty', 'cat', 'pp'):
        with open(os.path.join(out, 'figures', f'f4_{kd}.dat'), 'w', encoding='utf-8') as fh:
            fh.write(f'% paths of kind "{kd}" HAN used at seed 1: the search\'s mean alpha rank over both sizes\n'
                     '% (1 = most preferred of 36) and HAN\'s mean weight as a multiple of an equal share\n')
            fh.write('rank weight uses path\n')
            for p in sorted((p for p in use if kind(p) == kd), key=lambda p: pos[p][0]):
                fh.write(f'{pos[p][0]:.2f} {pos[p][1]:.3f} {len(use[p])} {p}\n')
    with open(os.path.join(out, 'figures', 'f4_labels.dat'), 'w', encoding='utf-8') as fh:
        fh.write('x y label anchor\n')
        for p, anchor in f4_labels.items():
            fh.write(f'{pos[p][0]:.2f} {pos[p][1]:.3f} {p} {anchor}\n')

    # F3: HAN test Hit@10 of every set at seed 1; x = group (1: 2-3 paths, 2: 4, 3: 6) plus an
    # offset per family and an even spread within it, sets sorted by their result
    base, spread = {'searched': -0.25, 'manual': 0.0, 'random': 0.25}, {'searched': 0.05, 'manual': 0.07, 'random': 0.04}
    groups, xy = {}, {}
    for t in table:
        g = 1 if len(sets[t['a']]) <= 3 else 2 if len(sets[t['a']]) == 4 else 3
        groups.setdefault((g, t['family']), []).append(t)
    for (g, fam), ts in groups.items():
        ts.sort(key=lambda t: t['han'])
        for i, t in enumerate(ts):
            step = 0.15 if g == 1 else spread[fam]          # 2-3 paths: manual sets only, room to spread
            xy[t['a']] = (g + base[fam] + step * (i - (len(ts) - 1) / 2), t['han'], g)
    for fam in ('searched', 'manual', 'random'):
        with open(os.path.join(out, 'figures', f'f3_{fam}.dat'), 'w', encoding='utf-8') as fh:
            fh.write(f'% {fam} sets: HAN test Hit@10 (sampled, seed 1); x = group 1 (2-3 paths), 2 (4) or 3 (6)\n')
            fh.write('x hit set\n')
            for a in sorted((a for a in sets if family(a) == fam), key=lambda a: xy[a][0]):
                fh.write(f'{xy[a][0]:.3f} {xy[a][1]:.4f} {short(a)}\n')
    with open(os.path.join(out, 'figures', 'f3_labels.dat'), 'w', encoding='utf-8') as fh:
        fh.write('x y label anchor\n')
        for a in sets:
            x, y, g = xy[a]
            if family(a) == 'manual':
                k = short(a).upper()
                anchor = {'M5': 'north', 'M8': 'east'}.get(k, 'west')
                fh.write(f'{x:.3f} {y:.4f} {k} {anchor}\n')
            elif family(a) == 'searched' and g == 2:
                fh.write(f'{x:.3f} {y:.4f} {short(a)[2:]} east\n')
            elif family(a) == 'searched' and short(a) == 'f6a2':      # the middle one of three equal sets
                fh.write(f'{x:.3f} {y:.4f} a1,a2,c north\n')

    section('Where the runs differ: Hit@10 by product group, seed 1 (HAN review/evidence/by_group.txt)')
    with open(os.path.join(han, 'review', 'evidence', 'by_group.txt'), encoding='utf-8') as fh:
        txt = fh.read().replace('\r', '')
    head = re.search(r'P-C-C-P reaches ([\d.]+)% of products, ([\d.]+)% of test edges', txt)
    assert abs(float(head.group(1)) - (100 - 100 * len(pccp_iso_global) / 48883)) < 0.05
    put('bg-pccp-reach-products-pct', head.group(1), 'products a P-C-C-P walk reaches')
    put('bg-pccp-reach-test-pct', head.group(2), 'test edges whose product it reaches')
    bg = {}
    for b in re.findall(r'^(\S+)_s1\.pt: Hit@10 ([\d.]+)\n  by product group: Book ([\d.]+)  DVD ([\d.]+)  '
                        r'Music ([\d.]+)  Video ([\d.]+)\n  P-C-C-P reaches the product: ([\d.]+)   does not: '
                        r'([\d.]+)\n  full-ranking top 10 of (\d+) test users: mean training degree ([\d.]+) \(all '
                        r'products ([\d.]+)\), from the top 1% most-bought ([\d.]+)%', txt, re.M):
        bg[b[0]] = dict(hit=float(b[1]), book=float(b[2]), dvd=float(b[3]), music=float(b[4]), video=float(b[5]),
                        reach=float(b[6]), no=float(b[7]), users=int(b[8]), deg=float(b[9]), alldeg=float(b[10]),
                        top1=float(b[11]))
    for a in sets:                                   # the check recomputes each run's own Hit@10
        assert abs(bg['han_' + a]['hit'] - float(han1[a]['hit10'])) < 6e-5, a
    for ns in (4, 6):
        sub = [a for a in sets if len(sets[a]) == ns]
        srch = [a for a in sub if family(a) == 'searched']
        dense = [a for a in sub if any('C' in p and p not in EMPTY for p in sets[a])]
        for col in ('no', 'reach', 'book', 'music', 'dvd', 'video'):
            gap = st.mean(bg['han_' + a][col] for a in srch) - st.mean(bg['han_' + a][col] for a in dense)
            put(f'bg-{ns}-gap-{col}', sg(gap, 3), 'searched sets minus the sets with a dense category path')
            put(f'bg-{ns}-gapabs-{col}', f(abs(gap), 3))
        a1, man = main[ns]
        for col in ('no', 'reach'):
            put(f'bg-{ns}-a1gap-{col}', sg(bg['han_' + a1][col] - bg['han_' + man][col], 3), 'a1 minus the manual set')
    hk = [a for a in sets if a != 'amazon_lp_manual_4']
    put('bg-users', n(bg['popularity']['users']), 'test users whose full-ranking top 10 is examined')
    put('bg-all-deg', f(bg['popularity']['alldeg'], 1), 'mean number of training users of a product')
    put('bg-han-deg-med', f(st.median(bg['han_' + a]['deg'] for a in hk), 0), 'median over the sets, M4 aside')
    put('bg-han-deg-min', f(min(bg['han_' + a]['deg'] for a in hk), 0))
    put('bg-han-deg-max', f(max(bg['han_' + a]['deg'] for a in hk), 0))
    put('bg-mtv-deg-med', f(st.median(bg['m2v_' + a]['deg'] for a in hk), 0))
    put('bg-gat-deg', f(bg['gat_union']['deg'], 0))
    put('bg-pop-deg', f(bg['popularity']['deg'], 0))
    put('bg-han-top1-med', f(st.median(bg['han_' + a]['top1'] for a in hk), 0), '% of the top 10 from the 1% most-bought')
    put('bg-mtv-top1-max', f(max(bg['m2v_' + a]['top1'] for a in hk), 1))
    put('bg-gat-top1', f(bg['gat_union']['top1'], 0))
    for col in ('book', 'music', 'dvd', 'video'):
        put(f'bg-han-mtv-{col}', sg(st.median(bg['han_' + a][col] - bg['m2v_' + a][col] for a in hk), 3),
            'HAN minus metapath2vec, median over the 23 sets')

    section('Training and cost (HAN logs; HAN notes, Parts 9-10)')
    hanlogs = [r for k, r in logs.items() if k.startswith('han_')]
    gat = [r for k, r in logs.items() if k.startswith('gat_union')]
    m2v = [r for k, r in logs.items() if k.startswith('m2v_')]
    assert len(hanlogs) == 32 and len(m2v) == 32 and len(gat) == 3
    assert all(r['fp'] == FINGERPRINT for r in hanlogs + gat + m2v)
    for key, rs in (('han', hanlogs), ('mtv', m2v), ('gat', gat)):
        mins = [r['minutes'] for r in rs]
        put(f'{key}-runs', n(len(rs)))
        put(f'{key}-run-min', f(st.median(mins), 1), 'median minutes per run')
        put(f'{key}-hours', f(sum(mins) / 60, 1), 'GPU hours, all runs')
    put('models-hours', f(sum(r['minutes'] for r in hanlogs + m2v + gat) / 60, 0), 'HAN, metapath2vec and GAT')
    assert all(r['stop'] is not None for r in hanlogs + m2v + gat), 'a run ended without early stopping'
    for key, rs, d in (('han', hanlogs, 0), ('gat', gat, 0), ('mtv', m2v, 1)):
        put(f'{key}-stop-min', f(min(r['stop'] for r in rs), d), 'epoch (metapath2vec: pass) of the early stop')
        put(f'{key}-stop-max', f(max(r['stop'] for r in rs), d))
    caps = Counter(r['cap'] for r in gat)
    rerun = [r for r in gat if r['cap'] != caps.most_common(1)[0][0]]
    assert len(rerun) == 1
    put('gat-cap', n(caps.most_common(1)[0][0]), 'epoch cap of GAT')
    put('gat-rerun-seed', rerun[0]['file'].rsplit('_s', 1)[1], 'the GAT seed that hit the cap and was rerun')
    put('gat-rerun-cap', n(rerun[0]['cap']))
    put('gat-rerun-stop', f(rerun[0]['stop'], 0))
    for m, pre in (('han', 'han'), ('mtv', 'm2v')):
        six = [logs[f'{pre}_{a}_s1']['minutes'] for a in sets if '_f6_' in a]
        other = [logs[f'{pre}_{a}_s1']['minutes'] for a in sets if len(sets[a]) == 6 and '_f6_' not in a]
        put(f'{m}-searched6-min-lo', f(min(six), 1), 'minutes per run, the three searched 6-path sets, seed 1')
        put(f'{m}-searched6-min-hi', f(max(six), 1))
        put(f'{m}-other6-min-lo', f(min(other), 1), 'the other 6-path sets (M6, random), seed 1')
        put(f'{m}-other6-min-hi', f(max(other), 1))
    section('Repeatability: a same-seed repeat of one HAN run on the same GPU (logs/addition runs)')
    pair = [os.path.join(logdir, 'addition runs', 'han_amazon_lp_v3_f4_a1_s1 (val).txt'),
            os.path.join(logdir, 'han_amazon_lp_v3_f4_a1_s1.txt')]
    rep, orig = [parse_log(x) for x in pair]
    best = [re.search(r'Best epoch (\d+), val bpr ([\d.]+)', open(x, encoding='utf-8', errors='replace').read()).groups()
            for x in pair]
    assert best[0] == best[1] and rep['hit10'] == orig['hit10'] and rep['stop'] == orig['stop'], (best, rep['hit10'])
    assert rep['fp'] == FINGERPRINT
    put('repeat-han-best', best[0][0], 'best epoch, the same in both runs (HAN, a1 with 4 paths, seed 1)')
    put('repeat-han-val', f(float(best[0][1]), 4), 'validation BPR, the same in both runs')
    put('repeat-han-hit', f(rep['hit10'], 4), 'test Hit@10, the same in both runs')
    put('repeat-han-groupdiff', f(max(abs(rep['deg'][k][1] - orig['deg'][k][1]) for k in orig['deg']), 4),
        'largest Hit@10 difference in a user-degree group')
    params = [r['params'] for r in hanlogs]
    put('han-params-min', f(min(params) / 1e6, 1), 'millions')
    put('han-params-max', f(max(params) / 1e6, 1), 'millions')
    put('han-user-params', f(53569 * 512 / 1e6, 1), 'millions: one 512-d vector per user')
    gains = [gain_last10(r) for r in hanlogs]
    put('han-gain-med', f(st.median(gains), 4), 'validation BPR gained in the last 10 epochs')
    put('han-gain-max', f(max(gains), 4))
    g6 = [gain_last10(r) for r in gat]
    put('gat-gain-min', f(min(g6), 4))
    put('gat-gain-max', f(max(g6), 4))
    put('searched-gain-max', f(max(gain_last10(r) for r in hanlogs if '_v3_' in r['file']), 4))

    section('Side check: HAN with a 5-epoch warm-up, seed 1 (HAN notes, Part 10)')
    w5 = {}
    for a in sets:
        p = os.path.join(logdir, 'beta-warmup-5', f'han_{a}_s1.txt')
        w5[a] = parse_log(p)
        assert w5[a]['warmup'] == 5 and w5[a]['exit'] == 0
    put('wu5-warmup', '5', 'epochs of path-weight warm-up in the side check (asserted in every log)')
    d = [w5[a]['hit10'] - float(han1[a]['hit10']) for a in sets]
    put('wu5-diff-mean', sg(st.mean(d), 3))
    put('wu5-diff-min', sg(min(d), 3))
    put('wu5-diff-max', sg(max(d), 3))
    put('wu5-dropped', n(sum(1 for a in sets for x in w5[a]['beta'].values() if x < DROP)))
    put('wu5-rho', f(spearman([w5[a]['hit10'] for a in sets], [float(han1[a]['hit10']) for a in sets]), 2))
    for ns in (4, 6):
        cand_ = [a for a in sets if family(a) != 'manual' and (f'rand{ns}' in a or f'_f{ns}_' in a)]
        order = sorted(cand_, key=lambda a: -w5[a]['hit10'])
        pos = [i + 1 for i, a in enumerate(order) if family(a) == 'searched']
        put(f'wu5-{ns}-pos', ', '.join(ordinal(k) for k in pos[:-1]) + ' and ' + ordinal(pos[-1]))
    put('wu5-f6a1-hit', f(w5['amazon_lp_v3_f6_a1']['hit10'], 3))
    put('wu5-m6-hit', f(w5['amazon_lp_manual_6']['hit10'], 3))
    put('wu5-m6-kept', n(sum(x >= 0.01 for x in w5['amazon_lp_manual_6']['beta'].values())), 'of its 6 paths')


# ------------------------------------------------------------------------------------------------
def external_numbers():
    """Numbers from other sources, typed here with their source."""
    section('From other sources (typed in make_numbers.py, with the source)')
    page = "McAuley's Amazon product data page, 2014 version (cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)"
    put('mcauley-raw-reviews', '142.8 million', page + ', read 5 November 2026: raw reviews')
    put('mcauley-dedup-reviews', '83.68 million', 'the same page: reviews left after removing duplicate items')
    t4 = {'DBLP': (92.05, 93.54), 'IMDB': (64.63, 65.89), 'ACM': (90.79, 92.28), 'Freebase': (54.77, 57.13)}
    gains = [b - a for a, b in t4.values()]
    put('lmsps-t4-gain-min', f(min(gains), 1), 'LMSPS paper, Table 4: Micro-F1 of HAN* minus HAN, four datasets')
    put('lmsps-t4-gain-max', f(max(gains), 1))
    dacrema = 'Dacrema et al., RecSys 2019, abstract'
    put('dacrema-algos', '18', dacrema + ': neural recommenders analysed')
    put('dacrema-repro', '7', 'the same: reproduced with reasonable effort')
    put('dacrema-beaten', '6', 'the same: often outperformed by simple heuristics')
    gems = {'Amazon': 800, 'Yelp': 1500, 'Douban': 2000}             # GPU minutes per dataset
    diffmg = {'Amazon': 0.7, 'Yelp': 1.1, 'Douban': 1.6}             # total of three search runs
    put('diffmg-gems-min', n(min(gems.values())), 'DiffMG paper, Table 5: GEMS search cost, GPU minutes')
    put('diffmg-gems-max', n(max(gems.values())))
    put('diffmg-own-max', f(max(diffmg.values()), 1), 'the same table: DiffMG search cost (three runs)')


def git_head(repo):
    """Short commit and branch of a checkout, read from .git without running git."""
    try:
        gitdir = os.path.join(repo, '.git')
        if os.path.isfile(gitdir):                                   # a worktree: "gitdir: <path>"
            with open(gitdir, encoding='utf-8') as fh:
                gitdir = os.path.join(repo, fh.read().split(':', 1)[1].strip())
        with open(os.path.join(gitdir, 'HEAD'), encoding='utf-8') as fh:
            head = fh.read().strip()
        if not head.startswith('ref:'):
            return head[:7], 'detached'
        ref = head.split(':', 1)[1].strip()
        branch = ref[len('refs/heads/'):] if ref.startswith('refs/heads/') else ref
        loose = os.path.join(gitdir, *ref.split('/'))
        if os.path.isfile(loose):
            with open(loose, encoding='utf-8') as fh:
                return fh.read().strip()[:7], branch
        with open(os.path.join(gitdir, 'packed-refs'), encoding='utf-8') as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:7], branch
    except (OSError, IndexError):
        pass
    return 'unknown', 'unknown'


def provenance_numbers(lmsps, han):
    section('Provenance: the checkouts these numbers were computed from')
    for key, repo in (('lmsps', lmsps), ('han', han)):
        commit, branch = git_head(repo)
        put(f'repo-{key}-commit', r'\detokenize{' + commit + '}', 'read from .git/HEAD')
        put(f'repo-{key}-branch', r'\detokenize{' + branch + '}')


def write_tex(out, args):
    lines = [
        '% numbers.tex -- written by make_numbers.py; do not edit by hand, run the script again.',
        f'% {datetime.date.today().isoformat()}: python make_numbers.py'
        f' --lmsps {os.path.basename(os.path.abspath(args.lmsps))} --han {os.path.basename(os.path.abspath(args.han))}'
        + (' --extraction-report extraction_report.json' if args.extraction_report else '')
        + (' --lang vi' if args.lang == 'vi' else ''),
        '% The raw-file counts come from extraction_report.json; the block below names the copy used.',
    ] + ([
        '% Vietnamese number format (--lang vi): 0,666 and 1.147.920; versions, commits and path sets as in',
        '% English. The notes after the values stay in English.',
    ] if args.lang == 'vi' else []) + [
        '% \\val{key} prints a number; an unknown key prints ??key in bold and raises a LaTeX warning.',
        '\\makeatletter',
        '\\providecommand{\\setval}[2]{\\@namedef{val@#1}{#2}}',
        '\\DeclareRobustCommand{\\val}[1]{\\@ifundefined{val@#1}{\\textbf{??#1}%',
        '  \\PackageWarning{numbers}{No value `#1\' in numbers.tex}}{\\@nameuse{val@#1}}}',
        '\\makeatother',
    ]
    for key, value, note in VALUES:
        if key is None:
            lines += ['', f'% --- {value} ' + '-' * max(3, 92 - len(value))]
            continue
        lines.append(f'\\setval{{{key}}}{{{value}}}' + (f'  % {note}' if note else ''))
    with open(os.path.join(out, 'numbers.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lmsps', required=True, help='LMSPS checkout (branch claude/fix-amazon-extraction)')
    ap.add_argument('--han', required=True, help='HAN-improve checkout (branch claude/han-review)')
    ap.add_argument('--out', default=os.path.dirname(os.path.abspath(__file__)), help='paper folder')
    ap.add_argument('--extraction-report', help='extraction_report.json of the full run (default: look in the LMSPS '
                    'checkout, review/evidence/ then data/AMAZON_filtered/; else the copy in this script)')
    ap.add_argument('--lang', choices=('en', 'vi'), default='en',
                    help='number format of numbers.tex: en (0.666; 1,147,920) or vi for the Vietnamese version '
                    '(0,666; 1.147.920); the figure data are the same')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'figures'), exist_ok=True)
    _, _, meta = data_numbers(args.lmsps, args.extraction_report)
    n_test, label = split_numbers(args.lmsps, meta)
    graph_numbers(args.lmsps, args.han, label)
    rank = search_numbers(args.lmsps, args.out)
    model_numbers(args.han, rank, args.out, n_test)
    external_numbers()
    provenance_numbers(args.lmsps, args.han)
    if args.lang == 'vi':
        VALUES[:] = [(k, vi_value(k, v) if k else v, note) for k, v, note in VALUES]
    write_tex(args.out, args)
    vals = {k: v for k, v, _ in VALUES if k}
    print(f'{len(vals)} values written to {os.path.join(args.out, "numbers.tex")}')
    for k in ('han-f4a1-hit-mean', 'han-f4a1-hit-sd', 'han-m5-hit-mean', 'han-4-hit-p', 'han-f6a1-hit-mean',
              'han-m6-hit-mean', 'han-6-hit-p', 'mtv-4-hit-p', 'mtv-6-hit-p', 'han-4-fhit-p', 'han-6-fhit-p',
              'han-r4-hit-mean', 'han-r6-hit-mean', 'gat-hit-mean', 'pop-hit', 'search-tie-med-4',
              'search-tie-med-6', 'search-winners-4', 'search-winners-6', 'search-vote-4', 'search-vote-6',
              'search-null-med-4', 'search-null-med-6', 'search-avg-sets-4', 'search-avg-sets-6',
              'search-window-4', 'search-window-6', 'pref-cand-rho', 'pref-cand-p', 'han-valtest-rho',
              'search-run-min-4', 'search-run-min-6', 'search-total-h', 'han-run-min', 'han-hours',
              'test-bookmusic-pct', 'split-train', 'wu5-diff-mean', 'wu5-rho', 'placeholder-found-max-lo',
              'placeholder-found-max-hi', 'edition-size-max', 'repo-lmsps-commit', 'repo-han-commit'):
        print(f'  {k:24s} ' + re.sub(r'\\(?:ensuremath|detokenize)\{(.*)\}', r'\1', vals[k]))


if __name__ == '__main__':
    main()
