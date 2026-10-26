#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Build HGB-format ``.dat`` files for the AMAZON (Amazon-Meta / SNAP) dataset from the
filtered CSVs produced by ``amazon_extract.py`` (or its C# twin in extraction_cs/).

Standalone replacement for ``DatGenerator.cs``. Depends on numpy + the stdlib; the
default --title-features lsa also needs scipy and scikit-learn (both in requirements.txt).

    python amazon_extract.py --input amazon-meta.txt --output <filtered_csv_dir>
    python dat_generator.py --input <filtered_csv_dir> --output data/AMAZON --verify

--------------------------------------------------------------------------------
FORMAT CONTRACT (derived from hgb/data_loader.py, not assumed)
--------------------------------------------------------------------------------
node.dat        id \t name \t type [\t a1,a2,...]
                * ids are GLOBAL and must form one contiguous block per type, with
                  the blocks ordered by type id -- load_nodes computes 'shift' by
                  accumulating count[0], count[1], ... in that order.
                * per type, EITHER every node carries an attribute vector or none
                  does: load_nodes only picks attributes up when the type's first
                  (shift) id is present in the attr dict.
                * 'name' must not contain a tab or newline -- load_nodes splits on
                  '\t' and raises unless the line has exactly 3 or 4 fields.
link.dat        head \t tail \t rel_id \t weight     (global ids)
label.dat       id \t name \t type \t label          (train split)
label.dat.test  same                                 (test split)

Schema written here:
    node types      0 = Product (P)   1 = Category (C)   2 = User (U)
    relations       0  P -> P  also-bought        1  reverse
                    2  C -> C  parent -> child    3  reverse
                    4  P -> C  belongs-to         5  reverse
                    6  U -> P  reviewed           7  reverse

Products MUST be type 0: utils.py::load_dataset allocates the label matrix as
``np.zeros((dl.nodes['count'][0], num_classes))`` and then indexes it with GLOBAL
node ids, which only works when the labelled type is the first block.

--------------------------------------------------------------------------------
WHY RELATION ORDER MATTERS
--------------------------------------------------------------------------------
utils.py::load_dataset iterates ``dl.links['data'].items()`` and appends to ``adjs`` in that
order; each dataset branch then unpacks POSITIONALLY (ACM: ``PP, PP_r, PA, AP, ...
= adjs``). ``links['data']`` is keyed by relation id in FIRST-APPEARANCE order in
link.dat -- it is not sorted. So link.dat is written here strictly grouped by
ascending relation id, and every relation is asserted non-empty; an empty relation
would silently shift the whole positional unpack.

Each ``adjs[i]`` carries row = head-type-local, col = tail-type-local, and the DGL
edge direction is tail -> head. So the AMAZON branch of utils.py::load_dataset
should read:

    PP, PP_r, CC, CC_r, PC, CP, UP, PU = adjs
    # merge PP+PP_r and CC+CC_r into ONE symmetric adjacency each (as the ACM
    # branch does for its PP pair): hg_propagate_feat_dgl names propagated tensors
    # f'{dtype}{src_key}', so two parallel same-type etypes would collide on the
    # same key and silently overwrite each other. Then:
    adjs = [PP, CC, PC, CP, UP, PU]
    etypes = [('P','P-P','P'), ('C','C-C','C'), ('C','C-P','P'),
              ('P','P-C','C'), ('P','P-U','U'), ('U','U-P','P')]

    # link prediction: --target-relation "U,P" (relation 6). BOTH adjs[6] (UP) and
    # adjs[7] (PU) must be replaced by the train-only adjacency, exactly as the ACM
    # branch does for its PA/AP pair, or val/test edges leak through
    # the reverse relation.

--------------------------------------------------------------------------------
NODE ATTRIBUTES
--------------------------------------------------------------------------------
Every type gets attributes, deliberately. utils.py::load_dataset substitutes
``torch.eye(count)`` for any attribute-less type -- unconditionally, before any
dataset branch runs -- so leaving the ~10^5 users bare would try to allocate a
dense [n_users, n_users] float tensor (tens of GB) and crash. The three encodings:

  Product   title text plus a one-hot of the product group. Default
            (--title-features lsa): TF-IDF over the title tokens reduced to
            --title-dim dimensions with truncated SVD (latent semantic analysis),
            rows L2-normalised. The former default (--title-features bow: a
            multi-hot over the 500 most frequent tokens, as in HGB-ACM) left 27% of
            the AMAZON products with no title signal at all, and a larger multi-hot
            multiplies the memory of every propagated meta-path feature.
  Category  bag-of-words multi-hot over the category-name tokens. Leakage-free
            (names are metadata, not interactions) and far cheaper than an identity.
  User      sparse random binary code (hashing trick): ``bits`` positions out of
            ``dim``, chosen from a BLAKE2b digest of the user id. This is a
            rank-``dim`` stand-in for the one-hot identity -- Linear(eye) is just a
            lookup table, and Linear(hash_code) is the same lookup table at low
            rank. Deterministic across runs and platforms. Amazon-Meta gives users
            no attributes other than their reviews, and any review-derived feature
            would leak the very U-P edges the model must predict (the train/val/test
            edge split happens later, in link_data.py).

Node names (2nd column) are identifiers: the ASIN for products, the customer ID
for users (Preprocessor.cs used to draw random display names, so users could not
be traced back) and "Name [ID]" for categories (names alone collide: "General").

--------------------------------------------------------------------------------
CATEGORY LINKS
--------------------------------------------------------------------------------
Amazon lists each of a product's category paths root to leaf.
--category-links leaf (default) links the product to the LEAF of every path; the
hierarchy stays reachable through the C-C relation. --category-links all (the
former behaviour) also links every ancestor: on AMAZON the median product then
sits in 13 categories instead of 4, one root category holds 56% of all products
and P-C-P walks are 33x denser, which flattens every category meta-path.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------- #
# schema constants
# --------------------------------------------------------------------------- #
PRODUCT_TYPE = 0
CATEGORY_TYPE = 1
USER_TYPE = 2

REL_ALSO_BOUGHT = 0
REL_ALSO_BOUGHT_REV = 1
REL_CATEGORY_CHILD = 2
REL_CATEGORY_CHILD_REV = 3
REL_PRODUCT_CATEGORY = 4
REL_PRODUCT_CATEGORY_REV = 5
REL_USER_PRODUCT = 6
REL_USER_PRODUCT_REV = 7

REL_NAMES = {
    REL_ALSO_BOUGHT: 'P->P also-bought',
    REL_ALSO_BOUGHT_REV: 'P->P also-bought (rev)',
    REL_CATEGORY_CHILD: 'C->C parent-child',
    REL_CATEGORY_CHILD_REV: 'C->C parent-child (rev)',
    REL_PRODUCT_CATEGORY: 'P->C belongs-to',
    REL_PRODUCT_CATEGORY_REV: 'C->P belongs-to (rev)',
    REL_USER_PRODUCT: 'U->P reviewed',
    REL_USER_PRODUCT_REV: 'P->U reviewed (rev)',
}

# file names written by AmazonGraphFilter.cs
FILTERED_PRODUCTS = 'filtered_products.csv'
FILTERED_CATEGORIES = 'filtered_categories.csv'
FILTERED_CATEGORY_EDGES = 'filtered_category_edges.csv'
FILTERED_CATEGORY_PATHS = 'filtered_category_paths.csv'
FILTERED_ALSO_BOUGHT = 'filtered_also_bought.csv'
FILTERED_REVIEWS = 'filtered_reviews.csv'
FILTERED_USERS = 'filtered_users.csv'

NODE_FILE = 'node.dat'
LINK_FILE = 'link.dat'
LABEL_FILE = 'label.dat'
LABEL_TEST_FILE = 'label.dat.test'

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with', 'by',
    'at', 'from', 'vol', 'volume', 'edition', 'series', 'set',
}

TOKEN_RE = re.compile(r'[0-9a-z]+')
UNSAFE_NAME_RE = re.compile(r'[\t\r\n]+')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def read_csv(path):
    """Stream dict rows from a CSV written by the C# preprocessor (UTF-8, RFC4180
    quoting, CRLF line endings)."""
    if not os.path.isfile(path):
        raise SystemExit('missing input file: {}'.format(path))
    with open(path, 'r', encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            yield row


def tokenize(text):
    """Lowercase alphanumeric runs, dropping stop words, 1-char tokens and pure
    numbers. Mirrors DatGenerator.cs::TokenizeTitle."""
    if not text:
        return []
    out = []
    for tok in TOKEN_RE.findall(text.lower()):
        if len(tok) < 2 or tok in STOP_WORDS or tok.isdigit():
            continue
        out.append(tok)
    return out


def safe_name(name, fallback):
    """node.dat/label.dat are tab-delimited and data_loader raises on a field count
    other than 3/4, so a tab or newline inside a name would corrupt the file."""
    name = UNSAFE_NAME_RE.sub(' ', name or '').strip()
    return name if name else fallback


def build_vocab(doc_counts, size):
    """Top-`size` tokens by document frequency, ties broken lexicographically so the
    vocabulary -- and therefore every feature column -- is reproducible run to run."""
    if size <= 0 or not doc_counts:
        return {}
    ordered = sorted(doc_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:size]
    return {tok: i for i, (tok, _) in enumerate(ordered)}


def hash_bits(key, dim, bits):
    """Deterministic sparse binary code: `bits` distinct positions in [0, dim)
    derived from a BLAKE2b digest of `key`. Stable across runs and platforms,
    unlike Python's salted built-in hash()."""
    chosen = []
    seen = set()
    counter = 0
    while len(chosen) < bits:
        digest = hashlib.blake2b('{}#{}'.format(key, counter).encode('utf-8'),
                                 digest_size=32).digest()
        for off in range(0, 32, 4):
            pos = int.from_bytes(digest[off:off + 4], 'big') % dim
            if pos not in seen:
                seen.add(pos)
                chosen.append(pos)
                if len(chosen) == bits:
                    break
        counter += 1
    return chosen


class MultiHotRenderer:
    """Renders a dense comma-separated 0/1 vector by patching a reusable buffer --
    O(active bits) per row instead of O(dim), which matters when dim ~ 10^3 and
    there are ~10^5 rows."""

    def __init__(self, dim):
        self.dim = dim
        buf = bytearray(b'0,' * dim)
        self.buf = buf[:-1] if dim else buf   # strip the trailing comma

    def render(self, indices):
        buf = self.buf
        for i in indices:
            buf[2 * i] = 0x31  # '1'
        out = bytes(buf)
        for i in indices:
            buf[2 * i] = 0x30  # '0'
        return out


def dedupe_pairs(heads, tails):
    """Unique (head, tail) pairs, order-stable, via a single int64 pack."""
    if not heads:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    h = np.asarray(heads, dtype=np.int64)
    t = np.asarray(tails, dtype=np.int64)
    packed = h * np.int64(1 << 32) + t
    _, keep = np.unique(packed, return_index=True)
    keep.sort()
    return h[keep], t[keep]


def write_edges(fh, heads, tails, rel):
    chunk = 200000
    for start in range(0, len(heads), chunk):
        hs = heads[start:start + chunk].tolist()
        ts = tails[start:start + chunk].tolist()
        block = '\n'.join('{}\t{}\t{}\t1.0'.format(h, t, rel) for h, t in zip(hs, ts))
        fh.write(block.encode('ascii'))
        fh.write(b'\n')


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def load_products(input_dir, title_vocab_size):
    asin_to_id = {}
    asins = []
    titles = []   # token list per product, aligned with node id
    groups = []   # group string per product ('' when absent)
    doc_counts = Counter()

    for row in read_csv(os.path.join(input_dir, FILTERED_PRODUCTS)):
        asin = (row.get('ASIN') or '').strip()
        if not asin or asin in asin_to_id:
            continue
        asin_to_id[asin] = len(asins)
        toks = tokenize(row.get('Title'))
        doc_counts.update(set(toks))
        titles.append(toks)
        groups.append((row.get('Group') or '').strip())
        asins.append(asin)

    return asin_to_id, asins, titles, groups, build_vocab(doc_counts, title_vocab_size)


def load_categories(input_dir, offset, cat_vocab_size):
    cat_to_id = {}
    keys = []
    names = []
    doc_counts = Counter()

    for row in read_csv(os.path.join(input_dir, FILTERED_CATEGORIES)):
        cid = (row.get('ID') or '').strip()
        if not cid or cid in cat_to_id:
            continue
        cat_to_id[cid] = offset + len(keys)
        name = (row.get('Name') or '').strip()
        toks = tokenize(name)
        doc_counts.update(set(toks))
        names.append((name, toks))
        keys.append(cid)

    return cat_to_id, keys, names, build_vocab(doc_counts, cat_vocab_size)


def load_users(input_dir, offset):
    user_to_id = {}
    keys = []
    for row in read_csv(os.path.join(input_dir, FILTERED_USERS)):
        uid = (row.get('ID') or '').strip()
        if not uid or uid in user_to_id:
            continue
        user_to_id[uid] = offset + len(keys)
        keys.append(uid)
    return user_to_id, keys


class BowProductAttrs:
    """--title-features bow: multi-hot over the title vocabulary + group one-hot."""

    def __init__(self, titles, groups, title_vocab, group_index):
        self.titles, self.groups = titles, groups
        self.title_vocab, self.group_index = title_vocab, group_index
        self.dim = len(title_vocab) + len(group_index)
        self.title_dim = len(title_vocab)
        self.coverage = sum(1 for t in titles if any(w in title_vocab for w in t)) / max(len(titles), 1)
        self.renderer = MultiHotRenderer(self.dim)

    def render(self, i):
        active = set(self.title_vocab[t] for t in self.titles[i] if t in self.title_vocab)
        if self.groups[i] in self.group_index:
            active.add(len(self.title_vocab) + self.group_index[self.groups[i]])
        return self.renderer.render(active)


class LsaProductAttrs:
    """--title-features lsa: TF-IDF over the title tokens (those in >= min_df titles,
    at most vocab_size of them) reduced to `dim` dimensions with truncated SVD, rows
    L2-normalised, followed by the group one-hot. A product whose tokens are all
    unique to it gets a zero text vector."""

    def __init__(self, titles, groups, group_index, dim, vocab_size, min_df, seed=0):
        import scipy.sparse as sp
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfTransformer

        doc_counts = Counter()
        for toks in titles:
            doc_counts.update(set(toks))
        vocab = build_vocab({t: c for t, c in doc_counts.items() if c >= min_df}, vocab_size)
        rows, cols = [], []
        for i, toks in enumerate(titles):
            for j in sorted({vocab[t] for t in toks if t in vocab}):
                rows.append(i)
                cols.append(j)
        x = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(titles), len(vocab)))
        x = TfidfTransformer().fit_transform(x)  # binary tf x smoothed idf, rows L2-normalised
        k = min(dim, x.shape[1] - 1, x.shape[0] - 1)
        if k < 1:
            raise SystemExit('too few titles/tokens for --title-features lsa; use --title-features bow')
        z = TruncatedSVD(n_components=k, n_iter=7, random_state=seed).fit_transform(x)
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        self.z = np.divide(z, norm, out=np.zeros_like(z), where=norm > 0).astype(np.float32)
        self.groups, self.group_index = groups, group_index
        self.vocab_size, self.title_dim = len(vocab), k
        self.dim = k + len(group_index)
        self.coverage = float((norm[:, 0] > 0).mean())

    def render(self, i):
        cells = ['%.5g' % v for v in self.z[i].tolist()]
        onehot = ['0'] * len(self.group_index)
        if self.groups[i] in self.group_index:
            onehot[self.group_index[self.groups[i]]] = '1'
        return ','.join(cells + onehot).encode('ascii')


def write_node_dat(out_dir, products, categories, users):
    asins, prod_attrs = products
    cat_keys, cat_names, cat_vocab = categories
    user_keys, user_dim, user_bits = users

    if prod_attrs.dim == 0:
        raise SystemExit('product attribute dimension is 0 -- raise --title-vocab')

    cat_renderer = MultiHotRenderer(len(cat_vocab)) if cat_vocab else None
    user_renderer = MultiHotRenderer(user_dim) if user_dim > 0 else None

    with open(os.path.join(out_dir, NODE_FILE), 'wb') as fh:
        for nid, asin in enumerate(asins):
            fh.write('{}\t{}\t{}\t'.format(nid, safe_name(asin, str(nid)),
                                           PRODUCT_TYPE).encode('utf-8'))
            fh.write(prod_attrs.render(nid))
            fh.write(b'\n')

        base = len(asins)
        for offset, (key, (name, toks)) in enumerate(zip(cat_keys, cat_names)):
            label = '{} [{}]'.format(name, key) if name else key
            fh.write('{}\t{}\t{}'.format(base + offset, safe_name(label, key),
                                         CATEGORY_TYPE).encode('utf-8'))
            if cat_renderer is not None:
                fh.write(b'\t')
                fh.write(cat_renderer.render(set(cat_vocab[t] for t in toks if t in cat_vocab)))
            fh.write(b'\n')

        base = len(asins) + len(cat_keys)
        for offset, key in enumerate(user_keys):
            fh.write('{}\t{}\t{}'.format(base + offset, safe_name(key, key),
                                         USER_TYPE).encode('utf-8'))
            if user_renderer is not None:
                fh.write(b'\t')
                fh.write(user_renderer.render(hash_bits(key, user_dim, user_bits)))
            fh.write(b'\n')

    return prod_attrs.dim


def write_labels(out_dir, asins, groups, train_ratio, seed):
    """Product 'group' as the node-classification label.

    Stratified, not a plain shuffle. data_loader derives ``num_classes`` from each
    file independently (max label + 1), and utils.py::load_dataset assigns
    ``labels_test['data'][test_nid]`` into a matrix sized by
    ``labels_train['num_classes']`` -- so if the highest class id is missing from
    either split the shapes disagree and the assignment raises. Stratifying makes
    every class present in both. A class with a single product cannot satisfy that,
    so those are dropped and the remaining label ids re-indexed contiguously.
    """
    by_group = defaultdict(list)
    for nid, group in enumerate(groups):
        if group:
            by_group[group].append(nid)

    dropped = sorted(g for g, ids in by_group.items() if len(ids) < 2)
    kept = sorted(g for g, ids in by_group.items() if len(ids) >= 2)
    label_index = {g: i for i, g in enumerate(kept)}

    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for group in kept:
        ids = np.array(by_group[group], dtype=np.int64)
        rng.shuffle(ids)
        n_train = int(round(len(ids) * train_ratio))
        n_train = min(max(n_train, 1), len(ids) - 1)  # keep both splits non-empty
        label = label_index[group]
        for nid in ids[:n_train].tolist():
            train_rows.append((nid, label))
        for nid in ids[n_train:].tolist():
            test_rows.append((nid, label))

    for rows, name in ((train_rows, LABEL_FILE), (test_rows, LABEL_TEST_FILE)):
        rows.sort()
        with open(os.path.join(out_dir, name), 'wb') as fh:
            for nid, label in rows:
                fh.write('{}\t{}\t{}\t{}\n'.format(
                    nid, safe_name(asins[nid], str(nid)), PRODUCT_TYPE, label).encode('utf-8'))

    return label_index, dropped, len(train_rows), len(test_rows)


def write_link_dat(out_dir, input_dir, asin_to_id, cat_to_id, user_to_id, category_links='leaf'):
    """Collect every relation first, then write link.dat grouped by ascending
    relation id (see the module docstring on positional adjs unpacking).
    category_links: 'leaf' = P->C for the last category of each path only,
    'all' = for every category on every path (ancestors included)."""
    heads = defaultdict(list)
    tails = defaultdict(list)
    dangling = Counter()

    def add(rel, rev, h, t):
        heads[rel].append(h)
        tails[rel].append(t)
        heads[rev].append(t)
        tails[rev].append(h)

    for row in read_csv(os.path.join(input_dir, FILTERED_ALSO_BOUGHT)):
        h = asin_to_id.get((row.get('ASIN') or '').strip())
        t = asin_to_id.get((row.get('AlsoASIN') or '').strip())
        if h is None or t is None:
            dangling['also_bought'] += 1
        elif h != t:
            add(REL_ALSO_BOUGHT, REL_ALSO_BOUGHT_REV, h, t)

    for row in read_csv(os.path.join(input_dir, FILTERED_CATEGORY_EDGES)):
        h = cat_to_id.get((row.get('ParentID') or '').strip())
        t = cat_to_id.get((row.get('ChildID') or '').strip())
        if h is None or t is None:
            dangling['category_edges'] += 1
        elif h != t:
            add(REL_CATEGORY_CHILD, REL_CATEGORY_CHILD_REV, h, t)

    leaf = {}  # (ASIN, path No) -> (Seq, CategoryID) of the deepest category seen
    for row in read_csv(os.path.join(input_dir, FILTERED_CATEGORY_PATHS)):
        asin = (row.get('ASIN') or '').strip()
        cid = (row.get('CategoryID') or '').strip()
        if category_links == 'leaf':
            key = (asin, (row.get('No') or '').strip())
            seq = int((row.get('Seq') or '0').strip() or 0)
            if key not in leaf or seq >= leaf[key][0]:
                leaf[key] = (seq, cid)
            continue
        h, t = asin_to_id.get(asin), cat_to_id.get(cid)
        if h is None or t is None:
            dangling['category_paths'] += 1
        else:
            # a product's several paths repeat shared ancestors; dedupe handles it
            add(REL_PRODUCT_CATEGORY, REL_PRODUCT_CATEGORY_REV, h, t)
    for (asin, _), (_, cid) in leaf.items():
        h, t = asin_to_id.get(asin), cat_to_id.get(cid)
        if h is None or t is None:
            dangling['category_paths'] += 1
        else:
            add(REL_PRODUCT_CATEGORY, REL_PRODUCT_CATEGORY_REV, h, t)

    for row in read_csv(os.path.join(input_dir, FILTERED_REVIEWS)):
        h = user_to_id.get((row.get('Customer') or '').strip())
        t = asin_to_id.get((row.get('ASIN') or '').strip())
        if h is None or t is None:
            dangling['reviews'] += 1
        else:
            add(REL_USER_PRODUCT, REL_USER_PRODUCT_REV, h, t)

    counts = {}
    with open(os.path.join(out_dir, LINK_FILE), 'wb') as fh:
        for rel in range(8):
            h, t = dedupe_pairs(heads.pop(rel, []), tails.pop(rel, []))
            if len(h) == 0:
                raise SystemExit(
                    'relation {} ({}) is empty. utils.py unpacks adjs positionally, '
                    'so a missing relation silently shifts every later adjacency -- '
                    'fix the filter output instead of proceeding.'.format(rel, REL_NAMES[rel]))
            write_edges(fh, h, t, rel)
            counts[rel] = len(h)
    return counts, dangling


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(out_dir):
    """Re-parse the output with the repo own loader -- the only check that proves
    the files are actually consumable."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(base, 'hgb') for base in
                  (script_dir, os.path.dirname(script_dir), os.getcwd())]
    hgb_dir = next((c for c in candidates if os.path.isfile(os.path.join(c, 'data_loader.py'))), None)
    if hgb_dir is None:
        print('  [skip] hgb/data_loader.py not found near {} or {}'.format(script_dir, os.getcwd()))
        return
    sys.path.insert(0, hgb_dir)
    try:
        from data_loader import data_loader
    except ImportError as exc:
        print('  [skip] could not import hgb/data_loader.py ({})'.format(exc))
        return

    dl = data_loader(out_dir)
    print('  nodes total       : {}'.format(dl.nodes['total']))
    for t in sorted(dl.nodes['count']):
        attr = dl.nodes['attr'][t]
        shape = 'None (loader substitutes torch.eye -- BAD)' if attr is None else str(attr.shape)
        print('    type {}: count={:>8}  shift={:>9}  attr={}'.format(
            t, dl.nodes['count'][t], dl.nodes['shift'][t], shape))

    keys = list(dl.links['data'].keys())
    print('  relation order    : {}'.format(keys))
    assert keys == sorted(keys), 'link.dat introduces relation ids out of order'
    for r in keys:
        print('    rel {}: {:<24} meta={} count={}'.format(
            r, REL_NAMES[r], dl.links['meta'][r], dl.links['count'][r]))

    ntr, nte = dl.labels_train['num_classes'], dl.labels_test['num_classes']
    print('  num_classes       : train={} test={}'.format(ntr, nte))
    assert ntr == nte, 'label.dat and label.dat.test disagree on class count'
    assert dl.links['meta'][REL_USER_PRODUCT] == (USER_TYPE, PRODUCT_TYPE)
    print('  OK -- parses cleanly and satisfies the positional-adjs invariant.')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='Generate HGB .dat files for the AMAZON dataset from filtered CSVs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input', required=True, help='directory holding filtered_*.csv')
    ap.add_argument('--output', required=True, help='output directory, e.g. data/AMAZON')
    ap.add_argument('--title-features', choices=['lsa', 'bow'], default='lsa',
                    help="lsa = TF-IDF + truncated SVD of the titles; bow = multi-hot over the "
                         "--title-vocab most frequent tokens (the former default)")
    ap.add_argument('--title-dim', type=int, default=256, help='lsa: SVD dimensions')
    ap.add_argument('--lsa-vocab', type=int, default=50000, help='lsa: maximum TF-IDF vocabulary')
    ap.add_argument('--lsa-min-df', type=int, default=2, help='lsa: a token must occur in this many titles')
    ap.add_argument('--title-vocab', type=int, default=500,
                    help='bow: title bag-of-words size (check the size report before raising)')
    ap.add_argument('--category-links', choices=['leaf', 'all'], default='leaf',
                    help='leaf = link a product to the last category of each path; all = also to '
                         'every ancestor (the former behaviour)')
    ap.add_argument('--cat-vocab', type=int, default=128,
                    help='category-name bag-of-words size; 0 leaves categories bare')
    ap.add_argument('--user-feat-dim', type=int, default=64,
                    help='user hash-code width; 0 leaves users bare (torch.eye blowup)')
    ap.add_argument('--user-feat-bits', type=int, default=8, help='set bits per user code')
    ap.add_argument('--train-ratio', type=float, default=0.8, help='label.dat share')
    ap.add_argument('--seed', type=int, default=42, help='label split seed')
    ap.add_argument('--verify', action='store_true',
                    help='re-parse the output with hgb/data_loader.py')
    args = ap.parse_args()

    if args.user_feat_dim and args.user_feat_bits >= args.user_feat_dim:
        raise SystemExit('--user-feat-bits must be smaller than --user-feat-dim')
    os.makedirs(args.output, exist_ok=True)

    print('Reading products...')
    asin_to_id, asins, titles, groups, title_vocab = load_products(args.input, args.title_vocab)
    group_index = {g: i for i, g in enumerate(sorted(set(g for g in groups if g)))}

    print('Reading categories...')
    cat_to_id, cat_keys, cat_names, cat_vocab = load_categories(
        args.input, len(asins), args.cat_vocab)

    print('Reading users...')
    user_to_id, user_keys = load_users(args.input, len(asins) + len(cat_keys))

    if args.title_features == 'lsa':
        print('Title features: TF-IDF + truncated SVD...')
        prod_attrs = LsaProductAttrs(titles, groups, group_index, args.title_dim, args.lsa_vocab,
                                     args.lsa_min_df)
    else:
        prod_attrs = BowProductAttrs(titles, groups, title_vocab, group_index)

    total = len(asins) + len(cat_keys) + len(user_keys)
    if total == 0:
        raise SystemExit('no nodes found -- check --input')
    if total >= (1 << 31):
        # dedupe_pairs packs a pair as h * 2**32 + t, so h must stay below 2**31
        # for the product to remain representable in int64.
        raise SystemExit('node ids exceed the packing width used for edge dedupe')

    print('Writing node.dat...')
    prod_dim = write_node_dat(
        args.output,
        (asins, prod_attrs),
        (cat_keys, cat_names, cat_vocab),
        (user_keys, args.user_feat_dim, args.user_feat_bits))

    print('Writing label.dat / label.dat.test...')
    label_index, dropped_groups, n_train, n_test = write_labels(
        args.output, asins, groups, args.train_ratio, args.seed)

    print('Writing link.dat...')
    counts, dangling = write_link_dat(args.output, args.input, asin_to_id, cat_to_id, user_to_id,
                                      args.category_links)

    # provenance: how this data/AMAZON was built (read back by nothing; for the record)
    meta = {
        'input': os.path.abspath(args.input), 'title_features': args.title_features,
        'title_dim': prod_attrs.title_dim, 'title_coverage': round(prod_attrs.coverage, 6),
        'category_links': args.category_links, 'cat_vocab': len(cat_vocab),
        'user_feat_dim': args.user_feat_dim, 'user_feat_bits': args.user_feat_bits,
        'counts': {'products': len(asins), 'categories': len(cat_keys), 'users': len(user_keys)},
        'relations': {REL_NAMES[r]: counts[r] for r in range(8)},
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__},
    }
    if args.title_features == 'lsa':
        import scipy
        import sklearn
        meta['lsa_vocab'] = prod_attrs.vocab_size
        meta['versions'].update(scipy=scipy.__version__, sklearn=sklearn.__version__)
    with open(os.path.join(args.output, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)

    # ----------------------------------------------------------------- report
    print('')
    print('--- summary ---------------------------------------------------')
    print('products {}  categories {}  users {}  total {}'.format(
        len(asins), len(cat_keys), len(user_keys), total))
    print('attr dims: product {} ({} title {} + {} groups; {:.1%} of products have title signal), '
          'category {}, user {}'.format(
              prod_dim, prod_attrs.title_dim, 'SVD dims' if args.title_features == 'lsa' else 'tokens',
              len(group_index), prod_attrs.coverage, len(cat_vocab), args.user_feat_dim))
    print('category links: {}'.format(args.category_links))
    print('labels: {} classes, {} train / {} test'.format(len(label_index), n_train, n_test))
    if dropped_groups:
        print('  dropped single-product groups (cannot appear in both splits): {}'.format(
            dropped_groups))
    for rel in range(8):
        print('  rel {} {:<24} {:>10} edges'.format(rel, REL_NAMES[rel], counts[rel]))
    for src, n in sorted(dangling.items()):
        print('  WARNING: {} rows in {} referenced nodes absent from the filter output'.format(
            n, src))

    # data_loader.load_nodes keeps a Python list of floats per attributed node before
    # np.array() collapses it -- that intermediate is what actually decides whether
    # this dataset loads on a 32 GB machine.
    cells = (len(asins) * prod_dim + len(cat_keys) * len(cat_vocab)
             + len(user_keys) * args.user_feat_dim)
    print('')
    print('attribute cells: {:.1f}M  ->  ~{:.1f} GB peak in data_loader.load_nodes '
          '(Python floats) + {:.1f} GB for the np.array'.format(
              cells / 1e6, cells * 32 / 1e9, cells * 8 / 1e9))
    if cells * 32 / 1e9 > 8:
        print('  WARNING: that peak is large. Lower --title-vocab, or patch load_nodes to '
              'parse straight into a preallocated np.float32 array.')

    if args.verify:
        print('')
        print('--- verify (hgb/data_loader.py) -------------------------------')
        verify(args.output)


if __name__ == '__main__':
    main()
