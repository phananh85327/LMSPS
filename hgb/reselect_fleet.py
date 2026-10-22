#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fleet-level re-selection of LMSPS search results (1-SE rule).

Each train_search.py run saves a JSON record (--search-json) with the exhaustive
evaluation table of its final pool: every ns-subset and its validation BPR.
Within one run, many subsets tie within the cross-seed noise of a set's score
(measured median ~0.0085 on AMAZON), so the per-run argmin is an arbitrary pick
inside a near-tie group and almost never repeats across seeds.

This script re-selects every run's set with a one-standard-error rule:

    among the subsets within --delta of the run's best score, take the one whose
    paths were selected most often by the OTHER runs (leave-one-out frequency);
    ties by validation loss.

delta should be the measured cross-seed perturbation (printed below as
"cross-seed noise"); the rule is applied post hoc from the saved tables, so it
can be re-run for any delta and no GPU time is needed.

    python reselect_fleet.py "logs/Fleet v2/final_ns4/*.json" --delta 0.010 --prefix amazon_lp_f4

Prints: per-run argmin -> re-selected set, distinct counts and duplicate rates
for both, path frequencies, consensus, cross-seed noise, arch.py entries.
"""

import argparse
import glob
import json
import statistics
from collections import Counter


def canon(path):
    r = path[::-1]
    return path if path <= r else r


def load_runs(patterns):
    files = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f'WARNING: nothing matches {pat!r}')
        files.extend(hits)
    runs = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            rec = json.load(fh)
        table = rec.get('eval_table') or []
        if not table:
            print(f'WARNING: no eval_table in {f}, skipped')
            continue
        rows = [(tuple(sorted(e['paths'])), float(e['val_bpr'])) for e in table]
        rows.sort(key=lambda r: r[1])
        best = rows[0][1]
        runs.append({
            'file': f, 'seed': rec.get('seed'), 'rows': rows, 'best': best,
            'argmin': rows[0][0], 'lookup': {s: l for s, l in rows},
            'top5': [s for s, _ in rows[:5]],
            'ns_search': rec.get('ns_search'), 'warmup': rec.get('warmup_epochs'),
            'dedupe': (rec.get('args') or {}).get('dedupe_reversed'),
            'final_pool': rec.get('final_pool'),
        })
    return runs


def cross_seed_noise(runs):
    """Median gap of run A's argmin when scored in run B's table (where evaluable)."""
    gaps = []
    for i, a in enumerate(runs):
        for j, b in enumerate(runs):
            if i == j:
                continue
            loss = b['lookup'].get(a['argmin'])
            if loss is not None:
                gaps.append(loss - b['best'])
    return (statistics.median(gaps), len(gaps)) if gaps else (None, 0)


def reselect(runs, delta, freq_source):
    picks = []
    for i, run in enumerate(runs):
        freq = Counter()
        for j, other in enumerate(runs):
            if j == i:
                continue
            paths = other['argmin'] if freq_source == 'argmin' else {p for s in other['top5'] for p in s}
            freq.update(set(paths))
        near = [(s, l) for s, l in run['rows'] if l - run['best'] <= delta]
        near.sort(key=lambda sl: (-sum(freq[p] for p in sl[0]), sl[1]))
        picks.append((near[0][0], near[0][1], len(near)))
    return picks


def membership(runs, delta):
    """For every set: in how many runs does it lie within delta of that run's best
    (i.e. belongs to that run's tie group). A pure count over declared ties."""
    count = Counter()
    sizes = []
    for run in runs:
        tie = [s for s, l in run['rows'] if l - run['best'] <= delta]
        sizes.append(len(tie))
        count.update(tie)
    return count, sizes


def report_sets(label, sets, canonical):
    key = (lambda s: tuple(sorted({canon(p) for p in s}))) if canonical else (lambda s: tuple(s))
    uniq = Counter(key(s) for s in sets)
    n = len(sets)
    print(f'\n== {label}: {len(uniq)} distinct / {n} runs (duplicate rate {1 - len(uniq) / n:.0%})')
    for s, c in uniq.most_common():
        print(f'  x{c:<3} {" ".join(s)}')
    return uniq


def main(args):
    runs = load_runs(args.records)
    if len(runs) < 2:
        raise SystemExit('need at least two JSON records with eval_table')
    cfg = Counter((r['dedupe'], r['warmup'], r['ns_search']) for r in runs)
    print(f'== {len(runs)} runs; (dedupe, warmup, ns_search) configs: {dict(cfg)}')
    if len(cfg) > 1:
        print('WARNING: mixed configurations -- frequencies and noise are only meaningful within one schedule')

    noise, n_pairs = cross_seed_noise(runs)
    if noise is not None:
        print(f'== cross-seed noise: median gap of a run\'s argmin scored in another run = {noise:.4f} '
              f'({n_pairs} evaluable pairs); --delta {args.delta:.3f} is '
              f'{"below" if args.delta < noise else "at or above"} it')

    member, sizes = membership(runs, args.delta)
    argmin_member = {r['argmin']: member[r['argmin']] for r in runs}
    print(f'\n== tie-group membership (delta {args.delta:.3f}): tie-group sizes per run {sizes}')
    print(f'   sets ranked by the number of runs whose tie group contains them '
          f'(raw argmin sets reach at most {max(argmin_member.values())}/{len(runs)}):')
    for s, c in member.most_common(args.top):
        tag = '  <- some run\'s argmin' if s in argmin_member else ''
        print(f'  {c:>3}/{len(runs)}  {" ".join(s)}{tag}')

    if args.mode == 'membership':
        return

    picks = reselect(runs, args.delta, args.freq_source)
    print(f'\n== per run: argmin -> 1-SE re-selection (delta {args.delta:.3f}, freq from {args.freq_source})')
    for run, (s, l, k) in zip(runs, picks):
        changed = '' if s == run['argmin'] else '  <- changed'
        print(f'  seed {run["seed"]!s:>3}: {" ".join(run["argmin"])}  ->  {" ".join(s)}  '
              f'(gap {l - run["best"]:.4f}, {k} sets within delta){changed}')

    report_sets('argmin (original)', [r['argmin'] for r in runs], args.canonical)
    uniq = report_sets(f'1-SE re-selected (delta {args.delta:.3f})', [s for s, _, _ in picks], args.canonical)

    freq = Counter(p for s, _, _ in picks for p in s)
    print(f'\n== path frequency in the re-selected sets ({len(runs)} runs)')
    for p, c in freq.most_common():
        print(f'  {p:<8} {c:>3}/{len(runs)}')
    k = len(picks[0][0])
    consensus = tuple(sorted(p for p, _ in freq.most_common(k)))
    c_count = uniq.get(consensus, 0)
    print(f'\n== path-frequency consensus (the {k} most frequent paths, assembled; NOT necessarily any run\'s pick): '
          f'{" ".join(consensus)}  -- itself a 1-SE pick in {c_count}/{len(runs)} runs')

    # HAN shortlist: the sets worth training downstream -- every 1-SE set picked by
    # >= --min-count runs, plus the consensus if it is not already among them.
    shortlist = [(s, c) for s, c in uniq.most_common() if c >= args.min_count]
    if consensus not in dict(shortlist):
        shortlist.append((consensus, c_count))
    print(f'\n== HAN shortlist ({len(shortlist)} sets: 1-SE picks seen >= {args.min_count}x, plus the consensus)')
    for s, c in shortlist:
        tag = '  (consensus)' if s == consensus else ''
        print(f'  x{c:<3} {" ".join(s)}{tag}')

    print(f'\n== arch.py entries (prefix "{args.prefix}" is only the key name stem)')
    def entry(key, paths, note):
        return f'    # {note}\n    "{key}": [\n        [{", ".join(repr(p) for p in paths)}], []],'
    prov = f'fleet v2 final schedule, split-seed 1, 1-SE delta {args.delta:.3f}'
    for i, (s, c) in enumerate(shortlist, 1):
        key = f'{args.prefix}_c' if s == consensus else f'{args.prefix}_s{i}'
        what = 'path-frequency consensus' if s == consensus else '1-SE pick'
        print(entry(key, list(s), f'{prov}; {what} in {c}/{len(runs)} runs'))

    if args.out:
        summary = {
            'delta': args.delta, 'freq_source': args.freq_source, 'cross_seed_noise': noise,
            'tie_group_sizes': sizes,
            'membership': [{'paths': list(s), 'runs': c} for s, c in member.most_common(args.top)],
            'runs': [{'file': r['file'], 'seed': r['seed'], 'argmin': list(r['argmin']),
                      'reselected': list(s), 'gap': l - r['best'], 'n_within_delta': k}
                     for r, (s, l, k) in zip(runs, picks)],
            'reselected_unique': [{'paths': list(s), 'count': c} for s, c in uniq.most_common()],
            'consensus': list(consensus), 'consensus_count': c_count,
            'shortlist': [{'paths': list(s), 'count': c, 'is_consensus': s == consensus} for s, c in shortlist],
        }
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1)
        print(f'\nsummary written to {args.out}')


def parse_args():
    ap = argparse.ArgumentParser(description='1-SE fleet re-selection over train_search.py JSON records.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('records', nargs='+', help='JSON records written by train_search.py --search-json (globs ok)')
    ap.add_argument('--delta', type=float, default=0.010,
                    help='near-tie tolerance in validation BPR; use the printed cross-seed noise')
    ap.add_argument('--freq-source', choices=['argmin', 'top5'], default='argmin',
                    help="which sets of the other runs vote: their argmin, or their 5 best")
    ap.add_argument('--mode', choices=['both', 'membership', '1se'], default='both',
                    help='membership = only the tie-group count table (no re-picking); 1se = also the per-run 1-SE picks')
    ap.add_argument('--top', type=int, default=8, help='rows to show in the membership table')
    ap.add_argument('--min-count', type=int, default=2,
                    help='HAN shortlist = 1-SE sets picked by at least this many runs (plus the consensus)')
    ap.add_argument('--canonical', action='store_true',
                    help='collapse reverse pairs (X / reversed X) when counting distinct sets')
    ap.add_argument('--prefix', type=str, default='amazon_lp_fleet',
                    help='name stem for the generated arch.py keys (<prefix>_c = consensus, <prefix>_s1.. = shortlist); '
                         'cosmetic only, no effect on the analysis')
    ap.add_argument('--out', type=str, default='', help='optional summary JSON path')
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
