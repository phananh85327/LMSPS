#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Are two search records the same run? The determinism pilot (run_fleet_v3.bat pilot)
runs one seed twice with --deterministic; the two records must agree exactly in the
alpha values, the final pool and every scored set.

    python compare_search_records.py pilot_a.json pilot_b.json

Exit code 0 = identical, 1 = different (the differing fields are printed).
"""
import json
import sys

FIELDS = ('candidates', 'alpha', 'final_pool', 'eval_table', 'paths')


def compare(a, b):
    """Names of the FIELDS on which two records differ."""
    return [k for k in FIELDS if a.get(k) != b.get(k)]


def main(path_a, path_b):
    with open(path_a, encoding='utf-8') as fa, open(path_b, encoding='utf-8') as fb:
        a, b = json.load(fa), json.load(fb)
    diffs = compare(a, b)
    if diffs:
        print(f'DIFFERENT: {", ".join(diffs)} -- the same seed did not reproduce')
        return 1
    print(f'IDENTICAL: alpha, final pool and all {len(a.get("eval_table", []))} scored sets match')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
