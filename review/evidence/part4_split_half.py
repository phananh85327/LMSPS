"""Review part 4: part 5's out-of-sample test of the fleet selection rules, on any fleet.
The runs are split into halves; each rule picks ONE set from half A, judged in half B by its gap
to the best set of every B run whose pool contains it. 200 random splits.
  a1        the rule behind <prefix>_a1: averaged scores, one-SE window, simplest first
  avg best  the lowest averaged score
  vote      the most frequent leave-one-out 1-SE pick (delta 0.010)
  one run   one random A run's own winner
In a half, the averaged scores use sets seen in >= 3 runs (5 in the full fleet).

    python review/evidence/part4_split_half.py "hgb/logs/Fleet v3/final_ns4/search_*.json"
"""
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'hgb'))
from reselect_fleet import load_runs, reselect, averaged_scores, score_noise  # noqa: E402


def hops(s):
    return sum(len(p) - 1 for p in s)


def pick_a1(runs, min_runs=3):
    seen = {s: v for s, v in averaged_scores(runs).items() if v[1] >= min_runs}
    sigma = score_noise(runs)[0]
    best = min(seen, key=lambda s: seen[s][0])
    window = [s for s in seen if seen[s][0] <= seen[best][0] + sigma / seen[best][1] ** 0.5]
    return min(window, key=lambda s: (hops(s), seen[s][0])), best, len(window)


def pick_vote(runs, delta=0.010):
    c = Counter(s for s, _, _ in reselect(runs, delta, 'argmin'))
    top = max(c.values())
    return min((s for s in c if c[s] == top),
               key=lambda s: st.mean(r['lookup'][s] - r['best'] for r in runs if s in r['lookup']))


def judge(s, runs):
    g = [r['lookup'][s] - r['best'] for r in runs if s in r['lookup']]
    return (st.mean(g) if g else None), len(g)


runs = load_runs([sys.argv[1]])
rng = random.Random(0)
res, cover, window, head = defaultdict(list), defaultdict(list), [], defaultdict(list)
for _ in range(200):
    order = list(range(len(runs)))
    rng.shuffle(order)
    A, B = [runs[i] for i in order[:len(runs) // 2]], [runs[i] for i in order[len(runs) // 2:]]
    a1, best, w = pick_a1(A)
    window.append(w)
    picks = {'a1': a1, 'avg best': best, 'vote': pick_vote(A), 'one run': rng.choice(A)['argmin']}
    for name, s in picks.items():
        g, k = judge(s, B)
        cover[name].append(k / len(B))
        if g is not None:
            res[name].append(g)
    for other in ('vote', 'one run', 'avg best'):
        both = [r for r in B if picks['a1'] in r['lookup'] and picks[other] in r['lookup']]
        if both:
            head[other].append(st.mean(r['lookup'][picks['a1']] - r['lookup'][picks[other]] for r in both))
print(f'{len(runs)} runs; one-SE window in a half: median {st.median(window)} sets')
for name in ('a1', 'avg best', 'vote', 'one run'):
    print(f'  {name:<9} gap to the other half\'s best: mean {st.mean(res[name]):.4f} (median {st.median(res[name]):.4f}), '
          f'judged in {len(res[name])}/200 splits, scorable in {st.mean(cover[name]):.0%} of the other half\'s runs')
for other, v in head.items():
    print(f'  a1 minus {other:<8} on the runs that score both: {st.mean(v):+.4f} over {len(v)} splits (negative = a1 better)')
