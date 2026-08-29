"""Cut a stratified subset out of the FPS-Bench-Stream annotation, for a sweep.

An arm over all 990 streams is 165 hours of footage, and a frame-rate sweep multiplies
that by the frame rate: at 32 fps one arm is 19 M frames through the vision tower. A
sweep therefore runs on a subset, and *which* subset matters -- FPS-Bench-Stream's two
axes are question type (9 of them, each a different temporal demand) and needle position
(early/middle/late, which is retrieval distance). Taking the first N records
(`convert_fpsbench_stream.py --limit`) fixes neither, so an accuracy difference between
two frame rates on such a subset is partly a difference in which questions they contain.

This samples `--per_cell` streams from each of the 27 (question type x position bin)
cells, seeded, so every arm of the sweep sees exactly the same questions and the subset
is balanced on both axes. The smallest cell holds 32 streams, so --per_cell up to 32 is
available; 5 gives 135 streams, ~14% of the benchmark.

Written next to the full file, under a name that says what it is: run_eval.py tags the
results directory with the annotation's basename, so a subset run never lands on top of
the full arm's results.csv.

Usage:
    python scripts/make_fpsbench_subset.py --per_cell 5
    python scripts/make_fpsbench_subset.py --per_cell 2 --out data/fpsbench_stream/smoke.json
"""

import os
import json
import random
import argparse
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/fpsbench_stream/test_mc.json')
    parser.add_argument('--out', type=str, default=None,
                        help='Default: data/fpsbench_stream/sub<N>.json, N = the subset '
                             'size, so the results directory names the subset it ran on.')
    parser.add_argument('--per_cell', type=int, default=5,
                        help='Streams per (question type x position bin) cell. 27 cells, '
                             'smallest 32 deep.')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Fixed so the subset is the same file every time it is '
                             'rebuilt -- two arms sampled with different seeds are not '
                             'comparable.')
    args = parser.parse_args()

    records = json.load(open(args.src))
    cells = defaultdict(list)
    for rec in records:
        conv = rec['conversations'][0]
        cells[(conv['question_type'], conv['position_bin'])].append(rec)

    rng = random.Random(args.seed)
    picked, short = [], []
    # Sorted, not dict order: the sample has to depend on the seed alone, not on the order
    # the source file happened to list its questions in.
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda r: r['video_id'])
        if len(pool) < args.per_cell:
            short.append((key, len(pool)))
        picked.extend(rng.sample(pool, min(args.per_cell, len(pool))))
    picked.sort(key=lambda r: r['video_id'])

    out = args.out or f'data/fpsbench_stream/sub{len(picked)}.json'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(picked, f, indent=1)

    types = Counter(r['conversations'][0]['question_type'] for r in picked)
    bins = Counter(r['conversations'][0]['position_bin'] for r in picked)
    print(f'wrote {len(picked)} of {len(records)} streams -> {out}')
    print(f'tasks: {dict(sorted(types.items()))}')
    print(f'needle position: {dict(sorted(bins.items()))}')
    if short:
        print(f'cells thinner than --per_cell, taken whole: {short}')


if __name__ == '__main__':
    main()
