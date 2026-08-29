"""Assemble a frame-rate sweep of FPS-Bench-Stream into one table.

scripts/eval_fpsbenchstream.slurm writes one results directory per frame rate
(results/<model>/fpsbench_stream/<retrieve_size>-<fps>[-<trigger>][-<subset>][<reduction>]),
each already scored by video_qa/eval/eval_fpsbench_stream.py into a results.json. This
reads those back and prints them as one row per arm, which is the form the sweep's question
is actually asked in: does accuracy rise with the sampling rate, and where does it stop.

Three columns carry that answer, and they have to be read together:

* `acc` is over all questions, so at 1-2 fps it is a mixture dominated by questions whose
  evidence was never resolvable at that rate;
* `acc_res` is over the questions whose own `min_fps` was met (FPSBench's per-question
  claim about the rate its evidence needs), with `n_res` saying how many that was. It is
  the like-for-like number, but its *population* grows with the frame rate -- at 1 fps it
  is a handful of easy questions, at 16 fps nearly all of them -- so a flat `acc_res` next
  to a rising `acc` is the expected shape, not a null result;
* `hit` is the share of layers whose top-k retrieval reached the needle. It says whether a
  change in accuracy came from seeing more frames or from finding them.

`top-k%` is what fraction of memory the top-k returns: at low frame rates a 600 s stream is
only a few hundred blocks, retrieval returns most of them, and a high `hit` there is
arithmetic rather than retrieval quality.

Usage:
    python scripts/collect_fpsbench_sweep.py --model llava_ov_0.5b
    python scripts/collect_fpsbench_sweep.py --model llava_ov_0.5b --filter query sub135
    python scripts/collect_fpsbench_sweep.py --model llava_ov_0.5b --csv sweep.csv
"""

import os
import re
import csv
import json
import argparse

# 64-8.0-query-sub135-rlt0.2cosine -> retrieve_size, fps, and whatever tags follow.
DIR_RE = re.compile(r'^(?P<retrieve>\d+)-(?P<fps>[\d.]+)(?P<tags>.*)$')


def rows(root, filters):
    """One record per scored arm under `root`, newest scoring first is not assumed."""
    out = []
    for name in sorted(os.listdir(root)):
        m = DIR_RE.match(name)
        if not m:
            continue
        tags = m.group('tags')
        if any(f not in tags for f in filters):
            continue
        path = os.path.join(root, name, 'results.json')
        if not os.path.exists(path):
            # An arm that ran but was not scored is worth naming: silently dropping it
            # makes a hole in the sweep look like an arm that was never launched.
            out.append({'dir': name, 'fps': float(m.group('fps')), 'tags': tags,
                        'unscored': True})
            continue
        summary = json.load(open(path))
        out.append({'dir': name, 'fps': float(m.group('fps')), 'tags': tags,
                    'unscored': False, **summary})
    return sorted(out, key=lambda r: (r['tags'], r['fps']))


def fmt(value, spec='{:.1f}', scale=1.0):
    return '-' if value is None else spec.format(value * scale)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llava_ov_0.5b')
    parser.add_argument('--results_root', type=str, default='results')
    parser.add_argument('--filter', type=str, nargs='*', default=[],
                        help='Substrings every arm\'s tag suffix must contain, e.g. '
                             '--filter query sub135 to read one sweep out of a results tree '
                             'holding several. Written without their leading dash so '
                             'argparse does not read them as flags.')
    parser.add_argument('--csv', type=str, default=None,
                        help='Also write the table here, for plotting.')
    args = parser.parse_args()

    root = os.path.join(args.results_root, args.model, 'fpsbench_stream')
    if not os.path.isdir(root):
        raise SystemExit(f'no results under {root}')
    table = rows(root, args.filter)
    if not table:
        raise SystemExit(f'no arms under {root} matching {args.filter}')

    header = (f'{"fps":>5}  {"n":>4}  {"acc":>6}  {"acc_res":>7}  {"n_res":>5}  '
              f'{"hit":>6}  {"top-k%":>6}  {"keep":>6}  arm')
    print(f'{args.model} / fpsbench_stream\n')
    print(header)
    print('-' * len(header))
    for r in table:
        if r['unscored']:
            print(f'{r["fps"]:>5g}  {"-":>4}  {"not scored -- run "}'
                  f'video_qa/eval/eval_fpsbench_stream.py --save_dir '
                  f'{os.path.join(root, r["dir"])}')
            continue
        print(f'{r["fps"]:>5g}  {r.get("n_questions", 0):>4}  '
              f'{fmt(r.get("qa_acc")):>6}  {fmt(r.get("qa_acc_resolvable")):>7}  '
              f'{r.get("n_resolvable", "-"):>5}  '
              f'{fmt(r.get("layers_hit_frac_mean"), scale=100):>6}  '
              f'{fmt(r.get("retrieved_frac_of_memory"), scale=100):>6}  '
              f'{fmt(r.get("token_keep_rate"), scale=100):>6}  '
              f'{r["tags"] or "baseline"}')
    print(f'\nchance is {table[0].get("chance", 20.0):.0f}%. acc_res is computed on a '
          f'different (growing) population at every frame rate -- see n_res.')

    if args.csv:
        fields = ['fps', 'tags', 'n_questions', 'qa_acc', 'qa_acc_resolvable',
                  'n_resolvable', 'layers_hit_frac_mean', 'retrieved_frac_of_memory',
                  'token_keep_rate', 'qa_acc_needle_found', 'qa_acc_needle_lost', 'dir']
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in table:
                if not r['unscored']:
                    w.writerow(r)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
