#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aggregate a fleet of train_search.py runs (the stress test): extract each run's
discovered arch, deduplicate, and report stability statistics.

    python analyze_search_fleet.py "logs/Top K 4/run_*.txt" --prefix amazon_lp_s4
    python analyze_search_fleet.py "output/AMAZON/search_*_seed*.json" --prefix amazon_lp_s4

Inputs may be console logs (the LAST line of the form
`paths [['PCPPUP', 'PPPPUP', ...], []]` is used) or the provenance JSON files
train_search.py writes via --search-json (key "paths"). Analyze each --ns group
separately.

Prints:
  1. per-file discovered arch (and which files failed to parse),
  2. unique archs with occurrence counts (duplicates filtered, order-insensitive),
  3. per-path frequency across all runs (the stability distribution),
  4. the CONSENSUS arch (top-k most frequent paths, k = the arch size seen),
  5. ready-to-paste arch.py entries for every unique arch + the consensus.
"""

import argparse
import ast
import glob
import json
import re
from collections import Counter

ARCH_RE = re.compile(r"^paths (\[\[.*?\], \[.*?\]\])\s*$", re.M)


def parse_log(path):
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            record = json.load(f)
        paths = record.get('paths')
        return list(paths) if paths else None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    matches = ARCH_RE.findall(text)
    if not matches:
        return None
    paths, label_paths = ast.literal_eval(matches[-1])
    return list(paths)


def main(args):
    files = []
    for pat in args.logs:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f'WARNING: nothing matches {pat!r}')
        files.extend(hits)
    if not files:
        raise SystemExit('no log files found')

    archs, failed = {}, []
    for f in files:
        arch = parse_log(f)
        if arch is None:
            failed.append(f)
        else:
            archs[f] = arch

    print(f'== {len(archs)} parsed / {len(files)} files'
          + (f'  (FAILED: {failed})' if failed else ''))
    for f, a in archs.items():
        print(f'  {f}: {a}')
    if not archs:
        return

    # dedupe order-insensitively: the arch is a SET of paths
    unique = Counter(tuple(sorted(a)) for a in archs.values())
    print(f'\n== {len(unique)} unique archs out of {len(archs)} runs '
          f'(duplicate rate {1 - len(unique) / len(archs):.0%})')
    for arch, n in unique.most_common():
        print(f'  x{n}  {list(arch)}')

    freq = Counter(p for a in archs.values() for p in a)
    print(f'\n== path frequency across {len(archs)} runs '
          f'({len(freq)} distinct paths ever chosen)')
    for p, n in freq.most_common():
        print(f'  {p:<8} {n:>3}/{len(archs)}  ({n / len(archs):.0%})')

    sizes = {len(a) for a in archs.values()}
    k = sizes.pop() if len(sizes) == 1 else max(sizes)
    consensus = [p for p, _ in freq.most_common(k)]
    print(f'\n== consensus arch (top {k} most frequent paths): {consensus}')

    print('\n== arch.py entries (paste into the archs dict)')
    def entry(key, paths):
        plist = ', '.join(f"'{p}'" for p in paths)
        return f'    "{key}": [\n    [{plist}], []],'
    print(entry(f'{args.prefix}_consensus', consensus))
    for i, (arch, n) in enumerate(unique.most_common(), 1):
        print(entry(f'{args.prefix}_u{i}', list(arch)) + f'  # seen x{n}')


def parse_args():
    ap = argparse.ArgumentParser(
        description='Deduplicate + analyze a train_search.py stress-test fleet.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('logs', nargs='+',
                    help='search console logs (globs ok); analyze each --ns group separately')
    ap.add_argument('--prefix', type=str, default='amazon_lp_fleet',
                    help='key prefix for the generated arch.py entries')
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
