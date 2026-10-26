#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Extract the AMAZON link-prediction graph from SNAP's amazon-meta.txt.

Python port of the C# extraction (Preprocessor.cs -> AmazonGraphFilter.cs, see
extraction_cs/). It writes the filtered_*.csv files dat_generator.py reads, and
by default fixes two problems that are in the raw SNAP data itself:

1. Edition duplicates. Amazon shows one review on every edition/format of a work
   (hardcover/paperback, DVD/VHS, volumes of a set) and SNAP crawled every
   edition page, so the same review line appears verbatim under several ASINs.
   On the full data 55% of the filtered U-P edges were such copies and 73% of
   val/test edges had a copy in train. Products sharing >= --edition-min-shared
   identical review lines that cover >= --edition-min-overlap of the smaller
   product are merged into one product (the member with the most reviews)
   BEFORE the k-core.
2. Placeholder customers. ATVPDKIKX0DER (Amazon's own seller ID) and
   A3UN6WX5RRO2AG hold the anonymous reviews: up to 741 resp. 579 reviews on a
   single product, and the first one sat on 61% of the filtered products. They
   are dropped, as is any customer with >= --placeholder-min-repeats reviews on
   one product.

Users are written with their customer ID as name (Preprocessor.cs drew random
names). --legacy switches both fixes off and reproduces AmazonGraphFilter.cs
row for row (user names aside).

    python amazon_extract.py --input amazon-meta.txt --output Filtered_Output
    python dat_generator.py --input Filtered_Output --output ../data/AMAZON --verify
"""

import argparse
import gzip
import json
import os
import re
from array import array
from collections import Counter, OrderedDict
from decimal import Decimal, InvalidOperation

import numpy as np

# output files (same names and columns as AmazonGraphFilter.cs)
FILTERED_PRODUCTS = 'filtered_products.csv'
FILTERED_CATEGORIES = 'filtered_categories.csv'
FILTERED_CATEGORY_EDGES = 'filtered_category_edges.csv'
FILTERED_CATEGORY_PATHS = 'filtered_category_paths.csv'
FILTERED_ALSO_BOUGHT = 'filtered_also_bought.csv'
FILTERED_REVIEWS = 'filtered_reviews.csv'
FILTERED_USERS = 'filtered_users.csv'
EDITION_MAP = 'filtered_edition_map.csv'
REPORT = 'extraction_report.json'

PLACEHOLDER_CUSTOMERS = ('ATVPDKIKX0DER', 'A3UN6WX5RRO2AG')

# the regexes of Preprocessor.cs
CATEGORY_RE = re.compile(r'^\s*(.+?)\s*\[(\d+)\]\s*$', re.IGNORECASE)
SUMMARY_RE = re.compile(r'total:\s*(\d+)\s+downloaded:\s*(\d+)\s+avg\s+rating:\s*([0-9.]+)', re.IGNORECASE)
REVIEW_RE = re.compile(r'(?P<date>\d{4}-\d{1,2}-\d{1,2})\s+(?:cutomer|customer):\s*(?P<cust>\S+)\s+'
                       r'rating:\s*(?P<rating>[0-9]+(?:\.[0-9]+)?)\s+votes:\s*(?P<votes>\d+)\s+'
                       r'helpful:\s*(?P<helpful>\d+)', re.IGNORECASE)


def decimal_or_none(text):
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def decimal_text(text):
    """decimal.TryParse + ToString(InvariantCulture): '4.50' -> '4.50', junk -> '0'."""
    d = decimal_or_none(text)
    return '0' if d is None else str(d)


def int_text(text):
    """int.TryParse + ToString: junk -> '0'."""
    try:
        return str(int(text.strip()))
    except ValueError:
        return '0'


def csv_field(v):
    """WriteCsvRow quoting: only fields with , " CR or LF are quoted."""
    if ',' in v or '"' in v or '\n' in v or '\r' in v:
        return '"' + v.replace('"', '""') + '"'
    return v


def write_csv(path, header, rows):
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(csv_field(h) for h in header) + '\n')
        for row in rows:
            fh.write(','.join(csv_field(x) for x in row) + '\n')


# --------------------------------------------------------------------------- parsing
class Raw:
    """Everything Preprocessor.cs extracts, kept in memory (block order)."""

    def __init__(self):
        self.products = []      # (id, asin, discontinued, title, group, salesrank, avg_rating)
        self.similar = []       # per block: [also-bought ASIN, ...]
        self.paths = []         # per block: [(cat_id, ...), ...]
        self.categories = OrderedDict()   # cat_id -> name of its first occurrence
        self.cat_edges = OrderedDict()    # (parent, child) -> None, first-seen order
        self.r_block = []       # review -> block index
        self.r_date, self.r_cust, self.r_rating, self.r_votes, self.r_helpful = [], [], [], [], []


def iter_blocks(path):
    opener = gzip.open if path.endswith('.gz') else open
    block = []
    with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if not line.strip():
                if block:
                    yield block
                    block = []
                continue
            block.append(line.strip())
    if block:
        yield block


def first(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def parse(path, legacy_title=False):
    raw = Raw()
    intern = {}
    it = lambda s: intern.setdefault(s, s)
    n_blocks = 0
    for lines in iter_blocks(path):
        n_blocks += 1
        line = first(lines, 'Id:')
        pid = line.split('Id:')[1].strip() if line is not None else ''
        line = first(lines, 'ASIN:')
        asin = line.split('ASIN:')[1].strip() if line is not None else ''
        if not pid or not asin:
            continue
        b = len(raw.products)
        if first(lines, 'discontinued product') is not None:
            raw.products.append((pid, asin, 'True', '', '', '', ''))
            raw.similar.append([])
            raw.paths.append([])
            continue

        title = ''
        line = first(lines, 'title:')
        if line is not None:
            if legacy_title:   # Preprocessor.cs: a title containing "title:" keeps only its last fragment
                parts = line.split('title:')
                tail = ''
                for i in range(2, len(parts)):
                    tail = 'title:' + parts[i]
                title = (parts[1] + tail).strip()
            else:
                title = line[len('title:'):].strip()
        line = first(lines, 'group:')
        group = line.split('group:')[1].strip() if line is not None else ''
        line = first(lines, 'salesrank:')
        salesrank = int_text(line.split('salesrank:')[1]) if line is not None else '0'

        line = first(lines, 'similar:')
        tokens = [t for t in line.split(' ') if t] if line is not None else []
        similar = tokens[2:]

        paths = []
        line = first(lines, 'categories:')
        if line is not None:
            for l2 in lines[lines.index(line) + 1:]:
                if not l2.startswith('|'):
                    break
                path = []
                for part in (p for p in l2.split('|') if p):
                    part = part.strip()
                    if part.startswith('['):
                        cid, name = part.replace('[', '').replace(']', ''), ''
                    else:
                        m = CATEGORY_RE.match(part)
                        if not m:
                            continue
                        name, cid = m.group(1), m.group(2)
                    cid = it(cid)
                    if cid not in raw.categories:
                        raw.categories[cid] = name
                    path.append(cid)
                if path:
                    paths.append(tuple(path))
                    for i in range(1, len(path)):
                        raw.cat_edges.setdefault((path[i - 1], path[i]), None)

        avg = '0'
        line = first(lines, 'reviews:')
        if line is not None:
            m = SUMMARY_RE.search(line)
            if m:
                avg = decimal_text(m.group(3))
            for l2 in lines[lines.index(line) + 1:]:
                m = REVIEW_RE.search(l2)
                if not m:
                    continue
                raw.r_block.append(b)
                raw.r_date.append(it(m.group('date')))
                raw.r_cust.append(it(m.group('cust')))
                raw.r_rating.append(it(m.group('rating')))
                raw.r_votes.append(it(m.group('votes')))
                raw.r_helpful.append(it(m.group('helpful')))

        raw.products.append((pid, asin, 'False', title, group, salesrank, avg))
        raw.similar.append(similar)
        raw.paths.append(paths)
    print(f'parsed {n_blocks:,} blocks: {len(raw.products):,} products, {len(raw.r_block):,} reviews, '
          f'{len(raw.categories):,} categories')
    return raw


# --------------------------------------------------------------------------- fixes
def find_placeholders(raw, valid_block_of_asin, explicit, min_repeats):
    """Customers that are not people: the explicit IDs, plus anyone with
    >= min_repeats review lines on a single product."""
    per = Counter()
    for b, c in zip(raw.r_block, raw.r_cust):
        if c and raw.products[b][1] in valid_block_of_asin:
            per[(c, raw.products[b][1])] += 1
    worst = {}
    total = Counter()
    for (c, _), k in per.items():
        total[c] += k
        if k > worst.get(c, 0):
            worst[c] = k
    found = {}
    for c in explicit:
        if c in total:
            found[c] = {'reason': 'known placeholder', 'review_lines': total[c], 'max_on_one_product': worst[c]}
    if min_repeats > 0:
        for c, k in worst.items():
            if k >= min_repeats and c not in found:
                found[c] = {'reason': f'>= {min_repeats} reviews on one product', 'review_lines': total[c],
                            'max_on_one_product': k}
    return found


def find_editions(raw, canon, n_prod, keep_row, min_shared, min_overlap, max_run):
    """Union-find over products that carry the same review lines.

    canon[row] = canonical product index of a review row (-1 = ignore).
    A review line is identified by (customer, date, rating, votes, helpful).
    Returns (rep, n_sig, n_pairs): rep[p] = representative product of p's cluster,
    n_sig[p] = distinct review lines of p, n_pairs = product pairs merged."""
    sig_id = {}
    rows_p, rows_s = array('q'), array('q')   # compact: the full data has ~7.6M review lines
    for i, p in enumerate(canon):
        if p < 0 or not keep_row[i]:
            continue
        key = (raw.r_cust[i], raw.r_date[i], raw.r_rating[i], raw.r_votes[i], raw.r_helpful[i])
        rows_s.append(sig_id.setdefault(key, len(sig_id)))
        rows_p.append(p)
    del sig_id
    if not rows_p:
        return np.arange(n_prod), np.zeros(n_prod, np.int64), 0
    keys = np.unique(np.frombuffer(rows_s, np.int64) * n_prod + np.frombuffer(rows_p, np.int64))  # distinct (line, product)
    s_of, p_of = np.divmod(keys, n_prod)
    n_sig = np.bincount(p_of, minlength=n_prod)
    starts = np.flatnonzero(np.r_[True, s_of[1:] != s_of[:-1]])
    bounds = np.r_[starts, s_of.size]
    run = np.diff(bounds)
    shared = Counter()
    for i in np.flatnonzero((run >= 2) & (run <= max_run)):
        prods = p_of[bounds[i]:bounds[i + 1]].tolist()       # sorted, distinct
        for x in range(len(prods)):
            for y in range(x + 1, len(prods)):
                shared[prods[x] * n_prod + prods[y]] += 1
    parent = list(range(n_prod))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_pairs = 0
    for key, k in shared.items():
        a, b = divmod(key, n_prod)
        if k >= min_shared and k >= min_overlap * min(n_sig[a], n_sig[b]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
            n_pairs += 1
    root = np.array([find(p) for p in range(n_prod)])
    # representative: most review lines, ties -> earliest product
    order = np.lexsort((np.arange(n_prod), -n_sig, root))     # by root, then -n_sig, then index
    rep = np.empty(n_prod, np.int64)
    first_of_root = {}
    for p in order.tolist():
        r = root[p]
        if r not in first_of_root:
            first_of_root[r] = p
        rep[p] = first_of_root[r]
    return rep, n_sig, n_pairs


# --------------------------------------------------------------------------- pipeline
def extract(args):
    raw = parse(args.input, legacy_title=args.legacy)
    products = raw.products

    # PHASE 1: usable products (not discontinued, first block of each ASIN)
    canon_of_asin = OrderedDict()                     # asin -> canonical product index (block)
    for b, p in enumerate(products):
        if p[1] and p[2] == 'False' and p[1] not in canon_of_asin:
            canon_of_asin[p[1]] = b
    n_prod = len(products)
    canon = [canon_of_asin.get(products[b][1], -1) for b in raw.r_block]   # reviews follow their ASIN
    report = OrderedDict(settings=vars(args).copy(), raw=OrderedDict(
        blocks=len(products), usable_products=len(canon_of_asin), review_lines=len(raw.r_block),
        categories=len(raw.categories)))
    print(f'phase 1: {len(canon_of_asin):,} usable products')

    keep_row = [bool(c) for c in raw.r_cust]
    placeholders = {}
    if not args.legacy and not args.keep_placeholders:
        placeholders = find_placeholders(raw, canon_of_asin, args.drop_customers, args.placeholder_min_repeats)
        for i, c in enumerate(raw.r_cust):
            if c in placeholders:
                keep_row[i] = False
        for c, info in placeholders.items():
            print(f'  dropped placeholder customer {c}: {info["review_lines"]:,} review lines, '
                  f'up to {info["max_on_one_product"]} on one product ({info["reason"]})')
    report['placeholders_dropped'] = placeholders

    rep = np.arange(n_prod)
    if not args.legacy and not args.keep_editions:
        rep, _, n_pairs = find_editions(raw, canon, n_prod, keep_row, args.edition_min_shared,
                                        args.edition_min_overlap, args.edition_max_run)
        sizes = Counter(Counter(rep[list(canon_of_asin.values())].tolist()).values())
        merged = sum(k * v for k, v in sizes.items() if k >= 2)
        clusters = sum(v for k, v in sizes.items() if k >= 2)
        print(f'  edition duplicates: {n_pairs:,} product pairs -> {clusters:,} clusters covering {merged:,} products '
              f'(merged into {clusters:,})')
        report['editions'] = OrderedDict(pairs=n_pairs, clusters=clusters, products_in_clusters=merged,
                                         cluster_sizes={str(k): v for k, v in sorted(sizes.items()) if k >= 2})

    # PHASE 2: positive, not unhelpful, one interaction per (customer, product)
    pos = decimal_or_none(str(args.pos_threshold))
    min_ratio = Decimal(str(args.helpful_min_ratio))
    seen = set()
    kept = []                                          # review row indices
    for i, p in enumerate(canon):
        if p < 0 or not keep_row[i]:
            continue
        rating = decimal_or_none(raw.r_rating[i])
        if rating is None or rating < pos:
            continue
        votes, helpful = decimal_or_none(raw.r_votes[i]), decimal_or_none(raw.r_helpful[i])
        if votes is not None and helpful is not None and votes >= args.helpful_min_votes \
                and helpful / votes < min_ratio:
            continue
        key = (raw.r_cust[i], int(rep[p]))
        if key not in seen:
            seen.add(key)
            kept.append(i)
    print(f'phase 2: {len(kept):,} positive interactions')

    # PHASE 3: k-core on the bipartite customer-product graph
    cust_ids = {}
    ku = np.array([cust_ids.setdefault(raw.r_cust[i], len(cust_ids)) for i in kept], np.int64)
    kp = np.array([rep[canon[i]] for i in kept], np.int64)
    alive = np.ones(len(kept), bool)
    it = 0
    while True:
        uc = np.bincount(ku[alive], minlength=len(cust_ids))
        pc = np.bincount(kp[alive], minlength=n_prod)
        nxt = alive & (uc[ku] >= args.k_core) & (pc[kp] >= args.k_core)
        if nxt.sum() == alive.sum():
            break
        alive = nxt
        it += 1
        print(f'   -> iteration {it}: {int(alive.sum()):,} edges')
    kept = [i for i, a in zip(kept, alive) if a]
    surv_prod = set(int(rep[canon[i]]) for i in kept)
    surv_asin = {products[b][1] for b in surv_prod}
    users = list(OrderedDict.fromkeys(raw.r_cust[i] for i in kept))
    print(f'phase 3: k-core {args.k_core} converged: {len(surv_prod):,} products, {len(kept):,} reviews, '
          f'{len(users):,} users')

    # PHASE 4: write the surviving graph
    os.makedirs(args.output, exist_ok=True)
    out = lambda name: os.path.join(args.output, name)
    rep_asin = {a: products[int(rep[b])][1] for a, b in canon_of_asin.items()}   # asin -> representative asin

    ab_rows, ab_seen = [], set()
    for b, sims in enumerate(raw.similar):
        a = products[b][1]
        for s in sims:
            if a == s:
                continue
            ra, rs = rep_asin.get(a, a), rep_asin.get(s, s)
            if ra == rs:                                  # a link between two editions of one work
                continue
            if ra in surv_asin and rs in surv_asin and (ra, rs) not in ab_seen:
                ab_seen.add((ra, rs))
                ab_rows.append((ra, rs))
    write_csv(out(FILTERED_ALSO_BOUGHT), ['ASIN', 'AlsoASIN'], ab_rows)

    # category paths: a product's own paths; a merged product gets the distinct paths of all its editions
    members = OrderedDict()
    for b, p in enumerate(products):
        if p[2] == 'False' and p[1] in canon_of_asin:
            members.setdefault(int(rep[canon_of_asin[p[1]]]), []).append(b)
    cp_rows, valid_cats = [], set()
    for r in sorted(surv_prod):
        asin = products[r][1]
        if args.legacy:                                   # every block of this ASIN, own numbering
            paths_of = [raw.paths[b] for b in members[r]]
        else:
            uniq = list(OrderedDict.fromkeys(pth for b in members[r] for pth in raw.paths[b]))
            paths_of = [uniq]
        for paths in paths_of:
            for no, pth in enumerate(paths):
                for seq, cid in enumerate(pth):
                    valid_cats.add(cid)
                    cp_rows.append((asin, str(no), str(seq), cid))
    write_csv(out(FILTERED_CATEGORY_PATHS), ['ASIN', 'No', 'Seq', 'CategoryID'], cp_rows)
    write_csv(out(FILTERED_CATEGORY_EDGES), ['ParentID', 'ChildID'],
              [e for e in raw.cat_edges if e[0] in valid_cats and e[1] in valid_cats])
    write_csv(out(FILTERED_CATEGORIES), ['ID', 'Name'],
              [(c, n) for c, n in raw.categories.items() if c in valid_cats])
    write_csv(out(FILTERED_PRODUCTS), ['ID', 'ASIN', 'Discontinued', 'Title', 'Group', 'SalesRank', 'AvgRating'],
              [products[b] for b in canon_of_asin.values() if b in surv_prod])
    write_csv(out(FILTERED_REVIEWS), ['ASIN', 'Date', 'Customer', 'Rating', 'Votes', 'Helpful'],
              [(products[int(rep[canon[i]])][1], raw.r_date[i], raw.r_cust[i], raw.r_rating[i], raw.r_votes[i],
                raw.r_helpful[i]) for i in kept])
    write_csv(out(FILTERED_USERS), ['ID', 'Name'], [(u, u) for u in users])
    if not args.legacy and not args.keep_editions:   # every member of every merged work, representative included
        size = Counter(int(rep[b]) for b in canon_of_asin.values())
        write_csv(out(EDITION_MAP), ['ASIN', 'RepresentativeASIN', 'Title'],
                  [(a, rep_asin[a], products[b][3]) for a, b in canon_of_asin.items() if size[int(rep[b])] >= 2])
    report['filtered'] = OrderedDict(products=len(surv_prod), users=len(users), reviews=len(kept),
                                     categories=len(valid_cats), also_bought=len(ab_rows),
                                     category_path_rows=len(cp_rows), kcore_iterations=it)
    with open(out(REPORT), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1)
    print(f'phase 4: wrote {args.output} ({REPORT} has the per-phase counts)')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='amazon-meta.txt -> filtered_*.csv (AmazonGraphFilter.cs format).',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', required=True, help='SNAP amazon-meta.txt (or .txt.gz)')
    ap.add_argument('--output', required=True, help='output directory for filtered_*.csv')
    ap.add_argument('--pos-threshold', type=float, default=4.0, help='keep reviews rated at least this')
    ap.add_argument('--helpful-min-votes', type=int, default=5)
    ap.add_argument('--helpful-min-ratio', type=float, default=0.2,
                    help='drop a review with >= --helpful-min-votes votes and a helpful share below this')
    ap.add_argument('--k-core', type=int, default=5)
    ap.add_argument('--legacy', action='store_true',
                    help='reproduce AmazonGraphFilter.cs exactly: no edition merge, placeholders kept, '
                         "Preprocessor.cs's title parsing")
    ap.add_argument('--keep-editions', action='store_true', help='do not merge edition duplicates')
    ap.add_argument('--edition-min-shared', type=int, default=3,
                    help='an edition pair shares at least this many identical review lines ...')
    ap.add_argument('--edition-min-overlap', type=float, default=0.5,
                    help='... covering at least this share of the smaller product')
    ap.add_argument('--edition-max-run', type=int, default=50,
                    help='ignore review lines shared by more products than this')
    ap.add_argument('--keep-placeholders', action='store_true', help='do not drop placeholder customers')
    ap.add_argument('--drop-customers', nargs='*', default=list(PLACEHOLDER_CUSTOMERS),
                    help='customer IDs that are known placeholders')
    ap.add_argument('--placeholder-min-repeats', type=int, default=50,
                    help='also drop any customer with at least this many reviews on one product (0 = off)')
    return ap.parse_args(argv)


if __name__ == '__main__':
    extract(parse_args())
