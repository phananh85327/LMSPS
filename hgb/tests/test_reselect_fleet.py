"""Fleet re-selection statistics (review part 5).

    python hgb/tests/test_reselect_fleet.py      (or: python -m pytest hgb/tests)

Standard library only, like reselect_fleet.py. Synthetic fleets have a known answer:
loss = run offset + sum of path effects + noise, every run scoring all 4-subsets of
its own pool of 8 paths.
"""
import glob
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile

HGB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HGB)
from reselect_fleet import averaged_scores, load_runs, null_distinct, reselect, score_noise  # noqa: E402

PATHS = ['P%d' % i for i in range(12)]


def _fleet(tmp, n_runs=20, noise=0.003, signal=True, run_spread=0.05, seed=0, args=None):
    """Write n_runs search records; returns (runs, true path effects)."""
    rng = random.Random(seed)
    effect = {p: (-0.004 * (12 - i) / 12 if signal else 0.0) for i, p in enumerate(PATHS)}
    for r in range(n_runs):
        pool = sorted(PATHS[:3] + rng.sample(PATHS[3:], 5))        # 3 paths in every pool, as in the fleets
        offset = rng.uniform(-run_spread, run_spread)
        table = [{'paths': list(s), 'val_bpr': offset + sum(effect[p] for p in s) + rng.gauss(0, noise)}
                 for s in itertools.combinations(pool, 4)]
        with open(os.path.join(tmp, f'search_seed{r}.json'), 'w') as fh:
            json.dump({'seed': r, 'eval_table': table, 'final_pool': pool, 'args': args or {}}, fh)
    return load_runs([os.path.join(tmp, 'search_seed*.json')]), effect


def test_averaged_scores_recover_the_best_set():
    """Runs carry large offsets (+-0.05) and score different pools: plain means would rank
    sets by which runs happened to score them; the two-way fit must not."""
    with tempfile.TemporaryDirectory() as tmp:
        runs, effect = _fleet(tmp)
        avg = averaged_scores(runs)
        seen = {s: e for s, (e, n) in avg.items() if n >= 5}
        true = {s: sum(effect[p] for p in s) for s in seen}
        best_true = min(true.values())
        picked = min(seen, key=seen.get)
        assert true[picked] - best_true <= 0.003, (picked, true[picked] - best_true)

        def corr(xs, ys):
            mx, my = statistics.mean(xs), statistics.mean(ys)
            return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (
                sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        plain = {}
        for r in runs:
            for s, l in r['rows']:
                plain.setdefault(s, []).append(l)
        two_way = corr([seen[s] for s in seen], [true[s] for s in seen])
        means = corr([statistics.mean(plain[s]) for s in seen], [true[s] for s in seen])
        assert two_way > 0.7 and two_way - means > 0.3, (two_way, means)   # 0.86 vs -0.41 on this fleet


def test_score_noise_measures_the_noise_put_in():
    """sigma recovers the noise despite the run offsets; rho is ~0 without a shared signal
    and high when the noise is small next to the set differences."""
    with tempfile.TemporaryDirectory() as tmp:
        runs, _ = _fleet(tmp, noise=0.004)
        sigma, rho, pairs = score_noise(runs)
        assert pairs > 50
        assert 0.0032 < sigma < 0.0048, sigma          # within 20% of 0.004
    with tempfile.TemporaryDirectory() as tmp:
        sigma, rho, _ = score_noise(_fleet(tmp, noise=0.0005)[0])
        assert rho > 0.8, rho
    with tempfile.TemporaryDirectory() as tmp:
        sigma, rho, _ = score_noise(_fleet(tmp, signal=False)[0])
        assert abs(rho) < 0.1, rho


def test_null_separates_agreement_from_the_vote():
    """Pure noise: the vote's duplicates are inside the shuffled-score null. A shared
    signal: the fleet beats the null (fewer distinct picks than 95% of null fleets)."""
    with tempfile.TemporaryDirectory() as tmp:
        runs, _ = _fleet(tmp, signal=False, seed=1)
        obs = len({s for s, _, _ in reselect(runs, 0.006, 'argmin')})
        null = null_distinct(runs, 0.006, 'argmin', n=100)
        assert null[5] <= obs <= null[-6], (obs, null[5], null[-6])
    with tempfile.TemporaryDirectory() as tmp:
        runs, _ = _fleet(tmp, noise=0.001, seed=2)
        obs = len({s for s, _, _ in reselect(runs, 0.002, 'argmin')})
        null = null_distinct(runs, 0.002, 'argmin', n=100)
        assert obs < null[5], (obs, null[5])


def test_arch_entries_and_provenance():
    """The printed arch.py entries: <prefix>_a<i> = the averaged one-SE sets in order, and the
    provenance line comes from the runs' own args (records without a flag get its old
    behaviour, as Fleet v2's do)."""
    for args, expect in (({'split_seed': 1, 'stage': 150, 'ns': 4, 'alpha_edges': 'val', 'wstep_kfold': 4,
                           'dropout': 0.5, 'same_type_loops': False},
                          'split-seed 1, stage 150, ns 4, alpha-edges val, wstep-kfold 4, dropout 0.5, plain relations'),
                         ({}, 'alpha-edges train-half, wstep-kfold 0, P-P/C-C self-loops')):
        with tempfile.TemporaryDirectory() as tmp:
            _fleet(tmp, noise=0.006, args=args)          # noisy enough for several one-SE sets
            out = os.path.join(tmp, 'summary.json')
            text = subprocess.run([sys.executable, os.path.join(HGB, 'reselect_fleet.py'),
                                   os.path.join(tmp, 'search_seed*.json'), '--delta', '0.006', '--prefix', 'k',
                                   '--null-draws', '0', '--out', out],
                                  check=True, capture_output=True, text=True).stdout
            summary = json.load(open(out, encoding='utf-8'))
            assert expect in summary['provenance'], summary['provenance']
            assert len(summary['averaged_one_se']) >= 2, 'fixture: need several one-SE sets to check their order'
            assert summary['averaged_one_se_keys'] == [f'k_a{i}' for i in range(1, len(summary['averaged_one_se']) + 1)]
            for key, paths in zip(summary['averaged_one_se_keys'], summary['averaged_one_se']):
                assert f'"{key}": [\n        [{", ".join(repr(p) for p in paths)}], []],' in text, key


def test_vote_reproduces_the_committed_fleet_v2_files():
    """The additions must not change the vote: recompute fleet v2 ns 4 at delta 0.010."""
    folder = os.path.join(HGB, 'logs', 'Fleet v2', 'final_ns4')
    ref_file = os.path.join(folder, 'reselect_d010.json')
    if not os.path.exists(ref_file):
        print('  (skipped: logs/Fleet v2 not present)')
        return
    ref = json.load(open(ref_file, encoding='utf-8'))
    runs = load_runs(sorted(glob.glob(os.path.join(folder, 'search_ns4_seed*.json'))))
    picks = reselect(runs, 0.010, 'argmin')
    assert [list(s) for s, _, _ in picks] == [r['reselected'] for r in ref['runs']]


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
