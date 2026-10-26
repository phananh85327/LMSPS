#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fleet-level re-selection of LMSPS search results (1-SE rule).

Each train_search.py run saves a JSON record (--search-json) with the exhaustive
evaluation table of its final pool: every ns-subset and its validation BPR.
Within one run, many subsets tie within the cross-seed noise of a set's score
(measured median ~0.0085 on AMAZON), so the per-run argmin is an arbitrary pick
inside a near-tie group and almost never repeats across seeds.

This script re-selects every run's set with a leave-one-out vote (called "1-SE"
below for continuity; it is not Breiman's rule, which picks the SIMPLEST model
within one SE of the best):

    among the subsets within --delta of the run's best score, take the one whose
    paths were selected most often by the OTHER runs (leave-one-out frequency);
    ties by validation loss.

delta should be the measured cross-seed perturbation (printed below as
"cross-seed noise"); the rule is applied post hoc from the saved tables, so it
can be re-run for any delta and no GPU time is needed.

    python reselect_fleet.py "logs/Fleet v2/final_ns4/search_*.json" --delta 0.010 --prefix amazon_lp_f4

Prints: per-run argmin -> re-selected set, distinct counts and duplicate rates
for both, path frequencies, consensus, cross-seed noise, arch.py entries. Since
review part 5 (2026-10-26) also:
- the score noise of one set and the runs' ranking agreement, from shared sets
- averaged scores (all runs pooled) and a real one-SE rule on them
- the duplicate rate a null fleet with shuffled scores gets from the vote alone
- each path's pick rate among the runs whose final pool held it
"""

import argparse
import glob
import json
import random
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


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float('nan')


def score_noise(runs, min_common=10):
    """Noise of ONE set's score, and how well two runs agree, from the sets both runs'
    tables contain (both pools hold all the set's paths). sigma: SD of loss_A - loss_B
    over the shared sets (each run's offset cancels), / sqrt(2). rho: Spearman
    correlation of the shared scores. Unlike cross_seed_noise, no winner is involved,
    so no winner's curse. Returns (median sigma, median rho, run pairs used)."""
    sigmas, rhos = [], []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            common = sorted(set(a['lookup']) & set(b['lookup']))
            if len(common) < min_common:
                continue
            la = [a['lookup'][s] for s in common]
            lb = [b['lookup'][s] for s in common]
            sigmas.append(statistics.pstdev([x - y for x, y in zip(la, lb)]) / 2 ** 0.5)
            rhos.append(spearman(la, lb))
    if not sigmas:
        return None, None, 0
    return statistics.median(sigmas), statistics.median(rhos), len(sigmas)


def averaged_scores(runs, iters=50):
    """Fleet-level score of every set: loss = run effect + set effect, fitted by
    alternating means (= least squares; runs score different pools, so plain means
    would mix in each run's offset). Returns {set: (effect, number of runs)}."""
    run_eff = [statistics.mean(l for _, l in r['rows']) for r in runs]
    set_eff = {}
    for _ in range(iters):
        acc = {}
        for i, r in enumerate(runs):
            for s, l in r['rows']:
                acc.setdefault(s, []).append(l - run_eff[i])
        set_eff = {s: statistics.mean(v) for s, v in acc.items()}
        run_eff = [statistics.mean(l - set_eff[s] for s, l in r['rows']) for r in runs]
    count = Counter(s for r in runs for s, _ in r['rows'])
    return {s: (e, count[s]) for s, e in set_eff.items()}


def null_distinct(runs, delta, freq_source, n=200, seed=0, key=lambda s: s):
    """Distinct re-selected sets when every run's scores are shuffled across its own
    sets: pools and tie-group sizes stay, any information in the scores is gone. The
    vote rule alone produces this many duplicates; only agreement beyond it counts."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        fake = []
        for r in runs:
            losses = [l for _, l in r['rows']]
            rng.shuffle(losses)
            rows = sorted(zip((s for s, _ in r['rows']), losses), key=lambda x: x[1])
            fake.append(dict(r, rows=rows, best=rows[0][1], argmin=rows[0][0], lookup=dict(rows),
                             top5=[s for s, _ in rows[:5]]))
        out.append(len({key(s) for s, _, _ in reselect(fake, delta, freq_source)}))
    return sorted(out)


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
    sigma, rho, n_sig = score_noise(runs)
    if sigma is not None:
        print(f'== score noise of one set: {sigma:.4f} (sets two runs share, {n_sig} run pairs); two runs rank '
              f'their shared sets with Spearman {rho:+.2f}. The cross-seed noise above re-scores a run\'s '
              f'WINNER, so it also carries the winner\'s curse')

    member, sizes = membership(runs, args.delta)
    argmin_member = {r['argmin']: member[r['argmin']] for r in runs}
    print(f'\n== tie-group membership (delta {args.delta:.3f}): tie-group sizes per run {sizes}')
    print(f'   sets ranked by the number of runs whose tie group contains them '
          f'(raw argmin sets reach at most {max(argmin_member.values())}/{len(runs)}):')
    for s, c in member.most_common(args.top):
        tag = '  <- some run\'s argmin' if s in argmin_member else ''
        print(f'  {c:>3}/{len(runs)}  {" ".join(s)}{tag}')

    # averaged scores: every run's evaluation of every set it could score, pooled.
    # Out of sample (split halves of fleet v2) this picked sets at least as good as
    # the leave-one-out vote below; the vote's duplicates are mostly its own product.
    hops = lambda s: sum(len(p) - 1 for p in s)
    seen = {s: v for s, v in averaged_scores(runs).items() if v[1] >= args.min_runs}
    best_eff, avg_short = None, []
    if seen and sigma is not None:
        best_set = min(seen, key=lambda s: seen[s][0])
        best_eff, best_n = seen[best_set]
        se = sigma / best_n ** 0.5
        avg_short = sorted((s for s in seen if seen[s][0] <= best_eff + se), key=lambda s: (hops(s), seen[s][0]))
        print(f'\n== averaged scores (loss = run effect + set effect, all runs pooled; sets seen in >= '
              f'{args.min_runs} runs; SE = score noise / sqrt(runs)):')
        for s in sorted(seen, key=lambda s: seen[s][0])[:args.top]:
            e, n = seen[s]
            print(f'  +{e - best_eff:.4f} (SE {sigma / n ** 0.5:.4f}, {n:>2} runs)  {" ".join(s)}')
        print(f'   one-SE rule: {len(avg_short)} set(s) within {se:.4f} of the best, simplest (fewest total hops) first:')
        for s in avg_short:
            print(f'    {hops(s):>2} hops  +{seen[s][0] - best_eff:.4f}  {" ".join(s)}')

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
    null = []
    if args.null_draws:
        key = (lambda s: tuple(sorted({canon(p) for p in s}))) if args.canonical else (lambda s: s)
        null = null_distinct(runs, args.delta, args.freq_source, n=args.null_draws, key=key)
        lo, hi = null[len(null) // 20], null[-(len(null) // 20) - 1]
        print(f'   null (every run\'s scores shuffled, pools kept): median {statistics.median(null):.0f} distinct '
              f'(90% range {lo}-{hi}); {sum(x <= len(uniq) for x in null) / len(null):.0%} of null fleets have '
              f'<= {len(uniq)}. Only duplicates beyond the null show agreement between runs')

    freq = Counter(p for s, _, _ in picks for p in s)
    pools = Counter(p for r in runs for p in (r['final_pool'] or []))
    print(f'\n== path frequency in the re-selected sets ({len(runs)} runs; the rate counts only runs whose final '
          f'pool held the path)')
    for p, c in freq.most_common():
        rate = f'   {c / pools[p]:.0%} of the {pools[p]} runs where available' if pools[p] else ''
        print(f'  {p:<8} {c:>3}/{len(runs)}{rate}')
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
            'score_noise': sigma, 'rank_agreement': rho,
            'averaged': [{'paths': list(s), 'vs_best': seen[s][0] - best_eff, 'runs': seen[s][1]}
                         for s in sorted(seen, key=lambda s: seen[s][0])[:args.top]] if best_eff is not None else [],
            'averaged_one_se': [list(s) for s in avg_short],
            'null_distinct': null,
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
    ap.add_argument('--min-runs', type=int, default=5,
                    help='averaged scores: only sets that at least this many runs could score')
    ap.add_argument('--null-draws', type=int, default=200,
                    help='fleets with shuffled scores for the duplicate-rate null (0 = skip)')
    ap.add_argument('--canonical', action='store_true',
                    help='collapse reverse pairs (X / reversed X) when counting distinct sets')
    ap.add_argument('--prefix', type=str, default='amazon_lp_fleet',
                    help='name stem for the generated arch.py keys (<prefix>_c = consensus, <prefix>_s1.. = shortlist); '
                         'cosmetic only, no effect on the analysis')
    ap.add_argument('--out', type=str, default='', help='optional summary JSON path')
    return ap.parse_args()


if __name__ == '__main__':
    main(parse_args())
