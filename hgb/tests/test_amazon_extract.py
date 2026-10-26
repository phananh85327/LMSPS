"""Regression test for amazon_extract.py (+ dat_generator.py's category links).

    python hgb/tests/test_amazon_extract.py      (or: python -m pytest hgb/tests)

The fixture is a hand-written amazon-meta.txt: a discontinued product, two editions of
one work carrying the same review lines and an also-bought link to each other, a
placeholder customer, a nameless category and the "cutomer:" spacing quirk.
"""
import contextlib
import csv
import io
import os
import subprocess
import sys
import tempfile

HGB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HGB)
import amazon_extract  # noqa: E402

COPIED = """    2001-1-1  cutomer: U1  rating: 5  votes:   3  helpful:   2
    2001-1-2  cutomer: U2  rating: 4  votes:   1  helpful:   1
    2001-1-3  cutomer:  U3  rating: 5  votes:   0  helpful:   0
    2001-1-4  cutomer: ATVPDKIKX0DER  rating: 5  votes:   0  helpful:   0"""

META = f"""# Full information about Amazon Share the Love products
Total items: 6

Id:   0
ASIN: D000000000
  discontinued product

Id:   1
ASIN: B000000001
  title: Work A (Hardcover)
  group: Book
  salesrank: 10
  similar: 2  B000000002  B000000003
  categories: 1
   |Books[283155]|Subjects[1000]|Fiction[17]|Drama[2159]
  reviews: total: 4  downloaded: 4  avg rating: 4.5
{COPIED}

Id:   2
ASIN: B000000002
  title: Work A (Paperback)
  group: Book
  salesrank: 20
  similar: 1  B000000001
  categories: 1
   |Books[283155]|Subjects[1000]|Fiction[17]|Plays[2160]
  reviews: total: 4  downloaded: 4  avg rating: 4.5
{COPIED}

Id:   3
ASIN: B000000003
  title: Work B
  group: Video
  salesrank: 30
  similar: 1  B000000004
  categories: 1
   |[139452]|VHS[404272]|Comedy[404278]
  reviews: total: 3  downloaded: 3  avg rating: 4
    2002-5-5  cutomer: U1  rating: 4  votes:   0  helpful:   0
    2002-5-6  cutomer: U2  rating: 5  votes:   0  helpful:   0
    2002-5-7  cutomer: U3  rating: 4  votes:   0  helpful:   0

Id:   4
ASIN: B000000004
  title: Work C
  group: Book
  salesrank: 40
  similar: 0
  categories: 1
   |Books[283155]|Subjects[1000]|History[9]
  reviews: total: 3  downloaded: 3  avg rating: 5
    2003-1-1  cutomer: U1  rating: 5  votes:   0  helpful:   0
    2003-1-2  cutomer: U2  rating: 5  votes:   0  helpful:   0
    2003-1-3  cutomer: U3  rating: 5  votes:   0  helpful:   0

Id:   5
ASIN: B000000005
  title: Only Anonymous Reviews
  group: Book
  salesrank: 50
  similar: 0
  categories: 1
   |Books[283155]|Subjects[1000]|History[9]
  reviews: total: 2  downloaded: 2  avg rating: 5
    2003-2-1  cutomer: ATVPDKIKX0DER  rating: 5  votes:   0  helpful:   0
    2003-2-2  cutomer: ATVPDKIKX0DER  rating: 4  votes:   0  helpful:   0
"""


def rows(path):
    with open(path, encoding='utf-8', newline='') as fh:
        return list(csv.DictReader(fh))


def run(tmp, *extra):
    meta = os.path.join(tmp, 'amazon-meta.txt')
    with open(meta, 'w', encoding='utf-8') as fh:
        fh.write(META)
    out = os.path.join(tmp, 'out' + ''.join(extra).replace('-', '_'))
    with contextlib.redirect_stdout(io.StringIO()):
        amazon_extract.extract(amazon_extract.parse_args(['--input', meta, '--output', out, '--k-core', '1', *extra]))
    return out


def test_fixed():
    with tempfile.TemporaryDirectory() as tmp:
        out = run(tmp)
        products = [r['ASIN'] for r in rows(os.path.join(out, 'filtered_products.csv'))]
        assert products == ['B000000001', 'B000000003', 'B000000004'], products   # editions merged, anonymous-only product gone
        reviews = rows(os.path.join(out, 'filtered_reviews.csv'))
        assert {r['Customer'] for r in reviews} == {'U1', 'U2', 'U3'}           # placeholder dropped, U3 parsed
        assert sum(r['ASIN'] == 'B000000001' for r in reviews) == 3             # copies count once
        also = [(r['ASIN'], r['AlsoASIN']) for r in rows(os.path.join(out, 'filtered_also_bought.csv'))]
        assert also == [('B000000001', 'B000000003'), ('B000000003', 'B000000004')], also  # edition link dropped
        paths = rows(os.path.join(out, 'filtered_category_paths.csv'))
        leaves = {r['CategoryID'] for r in paths if r['ASIN'] == 'B000000001' and r['Seq'] == '3'}
        assert leaves == {'2159', '2160'}, leaves                                # both editions' paths kept
        users = rows(os.path.join(out, 'filtered_users.csv'))
        assert all(u['ID'] == u['Name'] for u in users)                          # IDs as names
        assert {r['ASIN']: r['RepresentativeASIN'] for r in rows(os.path.join(out, 'filtered_edition_map.csv'))} == \
            {'B000000001': 'B000000001', 'B000000002': 'B000000001'}
        names = {r['ID']: r['Name'] for r in rows(os.path.join(out, 'filtered_categories.csv'))}
        assert names['139452'] == '' and names['404272'] == 'VHS'                # nameless category kept


def test_legacy():
    with tempfile.TemporaryDirectory() as tmp:
        out = run(tmp, '--legacy')
        products = [r['ASIN'] for r in rows(os.path.join(out, 'filtered_products.csv'))]
        assert products == ['B000000001', 'B000000002', 'B000000003', 'B000000004', 'B000000005'], products
        assert 'ATVPDKIKX0DER' in {r['Customer'] for r in rows(os.path.join(out, 'filtered_reviews.csv'))}
        assert not os.path.exists(os.path.join(out, 'filtered_edition_map.csv'))


def test_leaf_category_links():
    with tempfile.TemporaryDirectory() as tmp:
        out = run(tmp)
        counts = {}
        for mode in ('leaf', 'all'):
            dat = os.path.join(tmp, 'dat_' + mode)
            subprocess.run([sys.executable, os.path.join(HGB, 'dat_generator.py'), '--input', out, '--output', dat,
                            '--title-features', 'bow', '--category-links', mode],
                           check=True, stdout=subprocess.DEVNULL)
            with open(os.path.join(dat, 'link.dat')) as fh:
                counts[mode] = sum(1 for line in fh if line.split('\t')[2] == '4')   # rel 4 = P->C
        # leaves: B1 {Drama, Plays}, B3 {Comedy}, B4 {History} = 4; all: B1 5 + B3 3 + B4 3 = 11
        assert counts == {'leaf': 4, 'all': 11}, counts


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
