#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read-only audit of the AMAZON link-prediction data (review/part1_data_pipeline.md).

Runs on a filtered_*.csv directory written by amazon_extract.py (or the C#
extraction in extraction_cs/) and answers the questions that need the real data:
  1. edition duplicates: products carrying the SAME review rows (Amazon shares one
     review across editions/formats), and how many val/test edges they leak
     under the exact link_data.py split
  2. hub users
  3. category structure: ancestors vs leaves, size of the root categories,
     how dense P-C-P walks are
  4. also-bought coverage and reciprocity
  5. title bag-of-words coverage (same tokenizer/vocab as dat_generator.py)
  6. review dates
  7. (for Part 2) users that the search's weight step never trains
  8. optional: what each AmazonGraphFilter.cs rule removed (needs the
     Preprocessor.cs output directory)

Run it from hgb/ (it reuses dat_generator.py's tokenizer and node order) inside
lmsps_env, so the numpy version matches the one that produced the split:

    python audit_amazon_data.py --filtered <filtered csv dir> [--raw <Preprocessor.cs output dir>] ^
        --out audit.json --dump-clusters dup_clusters.csv

On correctly extracted data, section 3 finds no duplicate clusters, section 4
no leakage and section 2 no hub user.

Nothing is written except the optional --out JSON and --dump-clusters CSV.
Needs numpy + stdlib only (Python >= 3.6).
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:  # use dat_generator's own helpers so the numbers match data/AMAZON exactly
    from dat_generator import build_vocab, read_csv, tokenize
    HELPERS = 'dat_generator.py'
except ImportError:  # identical fallback copies
    HELPERS = 'built-in copies'
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'by',
                  'at', 'from', 'vol', 'volume', 'edition', 'series', 'set'}
    TOKEN_RE = re.compile(r'[0-9a-z]+')

    def read_csv(path):
        with open(path, 'r', encoding='utf-8-sig', newline='') as fh:
            for row in csv.DictReader(fh):
                yield row

    def tokenize(text):
        if not text:
            return []
        return [t for t in TOKEN_RE.findall(text.lower())
                if len(t) >= 2 and t not in STOP_WORDS and not t.isdigit()]

    def build_vocab(doc_counts, size):
        if size <= 0 or not doc_counts:
            return {}
        ordered = sorted(doc_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]
        return {tok: i for i, (tok, _) in enumerate(ordered)}

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
REPORT = {}


def section(title):
    print('\n' + '=' * 78 + '\n' + title + '\n' + '=' * 78)


def pct(a, b):
    return f'{100.0 * a / b:.2f}%' if b else 'n/a'


def dist(name, arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f'  {name}: (empty)')
        return {}
    q = np.percentile(arr, [50, 90, 99])
    d = {'min': int(arr.min()), 'median': float(q[0]), 'mean': float(arr.mean()),
         'p90': float(q[1]), 'p99': float(q[2]), 'max': int(arr.max())}
    print(f'  {name}: min {d["min"]}  median {d["median"]:.0f}  mean {d["mean"]:.2f}  '
          f'p90 {d["p90"]:.0f}  p99 {d["p99"]:.0f}  max {d["max"]}')
    return d


NORM_DROP = re.compile(r'[\(\[][^\)\]]*[\)\]]')   # "(Hardcover)", "[VHS]", "(Widescreen Edition)"
NORM_KEEP = re.compile(r'[^0-9a-z]+')


def norm_title(t):
    t = NORM_DROP.sub(' ', (t or '').lower())
    return ' '.join(NORM_KEEP.sub(' ', t).split())


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- load
def load(fdir):
    f = lambda name: os.path.join(fdir, name)
    asin_to_pid, asins, titles, groups = {}, [], [], []
    for row in read_csv(f('filtered_products.csv')):          # dat_generator.load_products order
        a = (row.get('ASIN') or '').strip()
        if not a or a in asin_to_pid:
            continue
        asin_to_pid[a] = len(asins)
        asins.append(a)
        titles.append(row.get('Title') or '')
        groups.append((row.get('Group') or '').strip())
    uid_to_u, uids = {}, []
    for row in read_csv(f('filtered_users.csv')):             # dat_generator.load_users order
        u = (row.get('ID') or '').strip()
        if not u or u in uid_to_u:
            continue
        uid_to_u[u] = len(uids)
        uids.append(u)
    eu, ep, sig, votes, years = [], [], [], [], []
    dangling = 0
    for row in read_csv(f('filtered_reviews.csv')):
        c = (row.get('Customer') or '').strip()
        a = (row.get('ASIN') or '').strip()
        u, p = uid_to_u.get(c), asin_to_pid.get(a)
        if u is None or p is None:
            dangling += 1
            continue
        d = (row.get('Date') or '').strip()
        r, v, h = (row.get('Rating') or '').strip(), (row.get('Votes') or '').strip(), (row.get('Helpful') or '').strip()
        eu.append(u)
        ep.append(p)
        sig.append(hash(f'{c}\t{d}\t{r}\t{v}\t{h}'))  # one review object = same customer/date/rating/votes/helpful
        votes.append(int(v) if v.isdigit() else 0)
        years.append(int(d[:4]) if d[:4].isdigit() else 0)
    return {'asins': asins, 'asin_to_pid': asin_to_pid, 'titles': titles, 'groups': groups,
            'uids': uids, 'eu': np.array(eu, np.int64), 'ep': np.array(ep, np.int64),
            'sig': np.array(sig, np.int64), 'votes': np.array(votes, np.int64),
            'years': np.array(years, np.int64), 'dangling': dangling}


# --------------------------------------------------------------------------- split (link_data.py)
def link_split(eu, ep, seed, val_ratio, test_ratio):
    edges = np.unique(np.stack([eu, ep], axis=1), axis=0)     # build_link_split: unique + sorted
    rng = np.random.default_rng(seed)
    edges = edges[rng.permutation(edges.shape[0])]
    n = edges.shape[0]
    n_val, n_test = int(n * val_ratio), int(n * test_ratio)
    n_train = n - n_val - n_test
    return edges[:n_train], edges[n_train:n_train + n_val], edges[n_train + n_val:]


# --------------------------------------------------------------------------- main
def main(a):
    D = load(a.filtered)
    P, U = len(D['asins']), len(D['uids'])
    eu, ep = D['eu'], D['ep']
    E = len(eu)
    pdeg = np.bincount(ep, minlength=P)
    udeg = np.bincount(eu, minlength=U)

    section(f'1. Basic counts  (helpers: {HELPERS})')
    n_cat = sum(1 for _ in read_csv(os.path.join(a.filtered, 'filtered_categories.csv')))
    print(f'  products {P:,}   categories {n_cat:,}   users {U:,}   U-P edges {E:,}   dangling review rows {D["dangling"]:,}')
    print('  (for reference: the pre-fix data behind the Fleet v2 logs had products 90,659 / '
          'categories 39,754 / users 171,285)')
    REPORT['counts'] = {'products': P, 'categories': n_cat, 'users': U, 'edges': E}
    gc = Counter(D['groups'])
    print('  product groups:', ', '.join(f'{g or "<empty>"} {c:,} ({pct(c, P)})' for g, c in gc.most_common()))
    REPORT['user_degree'] = dist('user degree   ', udeg)
    REPORT['product_degree'] = dist('product degree', pdeg)
    print(f'  users below 5 edges: {int((udeg < 5).sum())}   products below 5 edges: {int((pdeg < 5).sum())}  (5-core check)')

    section('2. Hub users (top 10 by degree)')
    top = np.argsort(-udeg)[:10]
    for u in top:
        print(f'  {D["uids"][u]:<16} {udeg[u]:>7,} edges  ({pct(udeg[u], P)} of all products)')
    REPORT['top_user_degree'] = int(udeg[top[0]])

    # ---------------------------------------------------------------- duplicates
    section('3. Edition duplicates: identical review rows on different products')
    order = np.argsort(D['sig'], kind='stable')
    s_sorted, p_sorted, v_sorted = D['sig'][order], ep[order], D['votes'][order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    bounds = np.r_[starts, s_sorted.size]
    run_len = np.diff(bounds)
    multi = np.flatnonzero(run_len >= 2)
    shared_rows = int(run_len[multi].sum())
    extra_copies = int((run_len[multi] - 1).sum())
    strong = int(sum(1 for i in multi if v_sorted[bounds[i]] >= 3))
    print(f'  review signatures on >= 2 products: {len(multi):,}  ({shared_rows:,} rows = {pct(shared_rows, E)} of edges; '
          f'{extra_copies:,} of them are extra copies = {pct(extra_copies, E)})')
    print(f'  ... of which with >= 3 helpfulness votes (near-certain shared review objects): {strong:,}')
    pair_shared = Counter()
    for i in multi:
        prods = sorted(set(p_sorted[bounds[i]:bounds[i + 1]].tolist()))
        if 2 <= len(prods) <= a.max_run:
            for x in range(len(prods)):
                for y in range(x + 1, len(prods)):
                    pair_shared[prods[x] * P + prods[y]] += 1
    dsu = DSU(P)
    dup_pairs, same_title = 0, 0
    nt = [norm_title(t) for t in D['titles']]
    for key, sh in pair_shared.items():
        x, y = divmod(key, P)
        if sh >= a.min_shared and sh >= a.min_overlap * min(pdeg[x], pdeg[y]):
            dsu.union(x, y)
            dup_pairs += 1
            same_title += nt[x] == nt[y] and nt[x] != ''
    root = np.array([dsu.find(i) for i in range(P)])
    csize = np.bincount(root, minlength=P)
    in_cluster = csize[root] >= 2
    n_clusters = int((csize >= 2).sum())
    print(f'  duplicate product pairs (>= {a.min_shared} shared rows AND >= {a.min_overlap:.0%} of the smaller product): '
          f'{dup_pairs:,}; identical normalised title in {pct(same_title, dup_pairs)} of them')
    print(f'  -> {n_clusters:,} duplicate clusters covering {int(in_cluster.sum()):,} products ({pct(in_cluster.sum(), P)}); '
          f'cluster sizes {dict(sorted(Counter(csize[csize >= 2].tolist()).items()))}')
    e_in = in_cluster[ep]
    print(f'  edges on clustered products: {int(e_in.sum()):,} ({pct(e_in.sum(), E)})')
    # one user, several products of the same cluster
    uc = np.unique(np.stack([eu[e_in], root[ep[e_in]]], axis=1), axis=0)
    multi_u = int(e_in.sum() - uc.shape[0])
    print(f'  edges that repeat a user->cluster link already present (pure duplicates of an interaction): '
          f'{multi_u:,} ({pct(multi_u, E)})')
    ex = [r for r in np.flatnonzero(csize >= 2)[:5]]
    for r in ex:
        members = np.flatnonzero(root == r)[:4]
        print('   e.g. ' + ' | '.join(f'{D["asins"][m]} "{D["titles"][m][:40]}" ({pdeg[m]} rev)' for m in members))
    tgroups = defaultdict(list)
    for i, t in enumerate(nt):
        if t:
            tgroups[t].append(i)
    tg = {t: v for t, v in tgroups.items() if len(v) >= 2}
    print(f'  identical normalised titles: {len(tg):,} groups covering {sum(map(len, tg.values())):,} products '
          f'(includes legit distinct works with the same title)')
    if a.dump_clusters:  # merge map for a fix: one line per cluster, ASINs + titles
        with open(a.dump_clusters, 'w', encoding='utf-8', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['cluster', 'ASIN', 'Title', 'reviews'])
            for r in np.flatnonzero(csize >= 2):
                for m in np.flatnonzero(root == r):
                    wr.writerow([int(r), D['asins'][m], D['titles'][m], int(pdeg[m])])
        print(f'  cluster membership written to {a.dump_clusters}')
    REPORT['duplicates'] = {'shared_signature_rows_frac': shared_rows / max(E, 1), 'extra_copies_frac': extra_copies / max(E, 1),
                            'strong_shared_signatures': strong, 'dup_pairs': dup_pairs, 'clusters': n_clusters,
                            'clustered_products': int(in_cluster.sum()), 'clustered_edges_frac': float(e_in.mean()) if E else 0,
                            'repeat_user_cluster_edges_frac': multi_u / max(E, 1)}

    # ---------------------------------------------------------------- split + leakage
    section(f'4. Leakage through duplicates under the actual split (split-seed {a.split_seed}, '
            f'{1 - a.val_ratio - a.test_ratio:.0%}/{a.val_ratio:.0%}/{a.test_ratio:.0%})')
    train, val, test = link_split(eu, ep, a.split_seed, a.val_ratio, a.test_ratio)
    print(f'  train {len(train):,}  val {len(val):,}  test {len(test):,}   (numpy {np.__version__}; '
          f'exact match to the trained split only inside lmsps_env)')
    nc = P

    def leak(edges, labels, valid):
        tr_ok = valid[train[:, 1]]
        keys = np.unique(train[tr_ok, 0] * nc + labels[train[tr_ok, 1]])
        ok = valid[edges[:, 1]]
        k = edges[:, 0] * nc + labels[edges[:, 1]]
        return int((ok & np.isin(k, keys)).sum())

    tid = np.full(P, -1)
    for gi, (t, v) in enumerate(tg.items()):
        tid[v] = gi
    for name, part in (('val', val), ('test', test)):
        lk = leak(part, root, in_cluster)
        lt = leak(part, np.where(tid >= 0, tid, 0), tid >= 0)
        print(f'  {name}: {lk:,} edges ({pct(lk, len(part))}) whose user has a TRAIN edge to another product of the same '
              f'duplicate cluster;  {lt:,} ({pct(lt, len(part))}) to a product with the identical normalised title')
        REPORT[f'{name}_dup_leak_frac'] = lk / max(len(part), 1)
        REPORT[f'{name}_title_leak_frac'] = lt / max(len(part), 1)
    tr_udeg = np.bincount(train[:, 0], minlength=U)
    tr_pdeg = np.bincount(train[:, 1], minlength=P)
    for name, part in (('val', val), ('test', test)):
        print(f'  {name}: cold users (0 train edges) {int((tr_udeg[part[:, 0]] == 0).sum()):,}, '
              f'cold products {int((tr_pdeg[part[:, 1]] == 0).sum()):,}')
    REPORT['train_user_degree'] = dist('train degree per user', tr_udeg[tr_udeg > 0])

    # ---------------------------------------------------------------- categories
    section('5. Categories: ancestors vs leaves')
    cname = {}
    cpath = os.path.join(a.filtered, 'filtered_categories.csv')
    for row in read_csv(cpath):
        cname[(row.get('ID') or '').strip()] = (row.get('Name') or '').strip()
    allc = defaultdict(set)
    leafseq = {}
    for row in read_csv(os.path.join(a.filtered, 'filtered_category_paths.csv')):
        p = D['asin_to_pid'].get((row.get('ASIN') or '').strip())
        if p is None:
            continue
        c = (row.get('CategoryID') or '').strip()
        allc[p].add(c)
        no = (row.get('No') or '').strip()
        sq = int((row.get('Seq') or '0').strip() or 0)
        key = (p, no)
        if key not in leafseq or sq > leafseq[key][0]:
            leafseq[key] = (sq, c)
    leafc = defaultdict(set)
    for (p, _), (_, c) in leafseq.items():
        leafc[p].add(c)
    n_all = np.array([len(allc.get(p, ())) for p in range(P)])
    n_leaf = np.array([len(leafc.get(p, ())) for p in range(P)])
    print(f'  products with no category: {int((n_all == 0).sum()):,}')
    dist('categories per product (all, = what dat_generator links)', n_all)
    dist('categories per product (leaves only)                  ', n_leaf)
    csz_all = Counter(c for p in allc for c in allc[p])
    csz_leaf = Counter(c for p in leafc for c in leafc[p])
    print('  largest categories (products linked):')
    for c, k in csz_all.most_common(12):
        print(f'    [{c:>7}] {cname.get(c, "?")[:34]:<34} {k:>8,}  ({pct(k, P)} of products)'
              f'{"   <- leaf for some product" if c in csz_leaf else ""}')
    walks_all = sum(k * k for k in csz_all.values())
    walks_leaf = sum(k * k for k in csz_leaf.values())
    print(f'  P-C-P walks (sum of n_c^2): all categories {walks_all:,.0f} = {walks_all / max(P * P, 1):.3f} x P^2;  '
          f'leaves only {walks_leaf:,.0f} = {walks_leaf / max(P * P, 1):.4f} x P^2')
    empty = sum(1 for c in csz_all if not cname.get(c))
    cdoc = Counter()
    for c in csz_all:
        cdoc.update(set(tokenize(cname.get(c, ''))))
    cvocab = build_vocab(cdoc, a.cat_vocab)
    zero_c = sum(1 for c in csz_all if not any(t in cvocab for t in tokenize(cname.get(c, ''))))
    print(f'  categories: {len(csz_all):,}; empty names {empty:,}; all-zero name features with --cat-vocab {a.cat_vocab}: '
          f'{zero_c:,} ({pct(zero_c, len(csz_all))})')
    REPORT['categories'] = {'per_product_all_median': float(np.median(n_all)), 'per_product_leaf_median': float(np.median(n_leaf)),
                            'largest_category_frac': csz_all.most_common(1)[0][1] / P if csz_all else 0,
                            'pcp_walks_over_P2_all': walks_all / max(P * P, 1), 'pcp_walks_over_P2_leaf': walks_leaf / max(P * P, 1),
                            'zero_feature_categories_frac': zero_c / max(len(csz_all), 1)}

    # ---------------------------------------------------------------- also-bought
    section('6. Also-bought (P-P) after filtering')
    directed = set()
    for row in read_csv(os.path.join(a.filtered, 'filtered_also_bought.csv')):
        x = D['asin_to_pid'].get((row.get('ASIN') or '').strip())
        y = D['asin_to_pid'].get((row.get('AlsoASIN') or '').strip())
        if x is not None and y is not None and x != y:
            directed.add((x, y))
    und = {(min(x, y), max(x, y)) for x, y in directed}
    deg = np.zeros(P, np.int64)
    for x, y in und:
        deg[x] += 1
        deg[y] += 1
    recip = sum(1 for x, y in directed if (y, x) in directed)
    in_dup = sum(1 for x, y in und if in_cluster[x] and root[x] == root[y])
    print(f'  directed pairs {len(directed):,}; undirected {len(und):,}; reciprocated {pct(recip, len(directed))}')
    print(f'  products with >= 1 also-bought neighbour: {int((deg > 0).sum()):,} ({pct((deg > 0).sum(), P)}); '
          f'with none (PP = self-loop only, so every P-P hop is an identity for them): {int((deg == 0).sum()):,}')
    dist('also-bought degree', deg)
    print(f'  also-bought pairs inside one duplicate cluster (edition links): {in_dup:,} ({pct(in_dup, len(und))})')
    REPORT['also_bought'] = {'undirected': len(und), 'covered_products_frac': float((deg > 0).mean()),
                             'pairs_in_dup_cluster_frac': in_dup / max(len(und), 1)}

    # ---------------------------------------------------------------- content
    section('7. Product title features')
    toks = [tokenize(t) for t in D['titles']]
    doc = Counter()
    for t in toks:
        doc.update(set(t))
    vocab = build_vocab(doc, a.title_vocab)
    nv = np.array([len({w for w in t if w in vocab}) for t in toks])
    print(f'  --title-features bow (--title-vocab {a.title_vocab}): products with ZERO title tokens in the vocab '
          f'(only the group one-hot left): {int((nv == 0).sum()):,} ({pct((nv == 0).sum(), P)})')
    dist('active title tokens per product', nv)
    REPORT['title_zero_frac'] = float((nv == 0).mean())
    try:  # the dat_generator default; needs dat_generator.py next to this script, scipy and scikit-learn
        from dat_generator import LsaProductAttrs
        lsa = LsaProductAttrs(toks, D['groups'], {}, a.title_dim, a.lsa_vocab, a.lsa_min_df)
        print(f'  --title-features lsa ({lsa.title_dim} dims, vocab {lsa.vocab_size:,}): products with title signal '
              f'{100 * lsa.coverage:.2f}%')
        REPORT['title_lsa_coverage'] = lsa.coverage
    except (ImportError, SystemExit) as exc:
        print(f'  (lsa coverage skipped: {exc})')

    # ---------------------------------------------------------------- dates
    section('8. Review dates (the split ignores them)')
    yc = Counter(D['years'].tolist())
    print('  ' + '  '.join(f'{y}:{c:,}' for y, c in sorted(yc.items())))

    # ---------------------------------------------------------------- Part 2 bonus
    section('9. (Part 2) users the search weight step never trains')
    n_tr = len(train)
    users_tr = np.flatnonzero(tr_udeg > 0)
    fr_u, fr_a, fr_v = [], [], []
    for s in a.search_seeds:
        perm = np.random.default_rng(s).permutation(n_tr)          # train_search.py internal 50/50 split
        half = n_tr // 2
        w_users = np.zeros(U, bool)
        w_users[train[perm[:half], 0]] = True
        untrained = (~w_users) & (tr_udeg > 0)
        fr_u.append(untrained[users_tr].mean())
        fr_a.append(untrained[train[perm[half:], 0]].mean())
        fr_v.append(untrained[val[:, 0]].mean())
    print(f'  over search seeds {a.search_seeds[0]}..{a.search_seeds[-1]}: users with train edges but none in the '
          f'weight half: mean {100 * np.mean(fr_u):.2f}%')
    print(f'  alpha-step edges whose user embedding stays at its random init: mean {100 * np.mean(fr_a):.2f}%')
    print(f'  real val edges (used to pick the final set) with such a user: mean {100 * np.mean(fr_v):.2f}%')
    REPORT['untrained_users_frac'] = float(np.mean(fr_u))
    REPORT['alpha_edges_untrained_frac'] = float(np.mean(fr_a))
    REPORT['val_edges_untrained_frac'] = float(np.mean(fr_v))

    # ---------------------------------------------------------------- raw (optional)
    if a.raw:
        section('10. What the filter removed (raw Preprocessor output)')
        n = n_cust = n_valid = n_pos = n_unhelp = n_dupe = 0
        valid = set()
        disc = 0
        for row in read_csv(os.path.join(a.raw, 'products.csv')):
            if (row.get('Discontinued') or '') == 'False' and row.get('ASIN'):
                valid.add(row['ASIN'].strip())
            else:
                disc += 1
        seen = set()       # hashed (customer, asin): ~7.8M raw rows, keep memory flat
        sig_buf = []
        for row in read_csv(os.path.join(a.raw, 'reviews.csv')):
            n += 1
            c, asin = (row.get('Customer') or '').strip(), (row.get('ASIN') or '').strip()
            if not c:
                continue
            n_cust += 1
            sig_buf.append(hash(f'{c}\t{row.get("Date")}\t{row.get("Rating")}\t{row.get("Votes")}\t{row.get("Helpful")}'))
            if asin not in valid:
                continue
            n_valid += 1
            try:
                r = float(row.get('Rating') or 'x')
            except ValueError:
                continue
            if r < 4:
                continue
            n_pos += 1
            v, h = (row.get('Votes') or ''), (row.get('Helpful') or '')
            if v.isdigit() and h.isdigit() and int(v) >= 5 and int(h) / int(v) < 0.2:
                n_unhelp += 1
                continue
            key = hash(f'{c}\t{asin}')
            if key in seen:
                n_dupe += 1
            seen.add(key)
        _, cnt = np.unique(np.array(sig_buf, np.int64), return_counts=True)
        shared = int(cnt[cnt >= 2].sum())
        print(f'  products not usable (discontinued / no ASIN): {disc:,}')
        print(f'  raw review rows {n:,}; with customer {n_cust:,}; on usable products {n_valid:,}')
        print(f'  rating >= 4: {n_pos:,} ({pct(n_pos, n_valid)}); dropped as unhelpful: {n_unhelp:,} ({pct(n_unhelp, n_pos)}); '
              f'repeat (user, product): {n_dupe:,}')
        print(f'  RAW review rows whose (customer, date, rating, votes, helpful) also appears on another row: '
              f'{shared:,} ({pct(shared, n_cust)})')
        REPORT['raw'] = {'rows': n, 'pos_frac': n_pos / max(n_valid, 1), 'unhelpful_frac': n_unhelp / max(n_pos, 1),
                         'shared_signature_frac': shared / max(n_cust, 1)}

    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(REPORT, fh, indent=1)
        print(f'\nsummary written to {a.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Read-only Part-1 audit of the AMAZON LP data.')
    ap.add_argument('--filtered', required=True, help='AmazonGraphFilter output dir (filtered_*.csv)')
    ap.add_argument('--raw', default='', help='optional Preprocessor output dir (products.csv, reviews.csv)')
    ap.add_argument('--title-vocab', type=int, default=500, help='as dat_generator.py --title-vocab (bow)')
    ap.add_argument('--title-dim', type=int, default=256, help='as dat_generator.py --title-dim (lsa)')
    ap.add_argument('--lsa-vocab', type=int, default=50000, help='as dat_generator.py --lsa-vocab')
    ap.add_argument('--lsa-min-df', type=int, default=2, help='as dat_generator.py --lsa-min-df')
    ap.add_argument('--cat-vocab', type=int, default=128)
    ap.add_argument('--split-seed', type=int, default=1)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--search-seeds', type=int, nargs='+', default=list(range(1, 31)))
    ap.add_argument('--min-shared', type=int, default=3, help='duplicate pair: at least this many identical review rows')
    ap.add_argument('--min-overlap', type=float, default=0.5, help='... and at least this share of the smaller product')
    ap.add_argument('--max-run', type=int, default=50, help='skip signatures shared by more products than this')
    ap.add_argument('--out', default='', help='optional JSON summary path')
    ap.add_argument('--dump-clusters', default='', help='optional CSV of the detected duplicate clusters (ASIN, title)')
    main(ap.parse_args())
