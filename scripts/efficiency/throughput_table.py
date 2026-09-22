"""Streaming throughput per backbone and arm, straight from the span CSVs.

Why this exists next to collect_cost_model.py: that script's KV-Cache and GFLOPs columns
are derived from config.json, and it only reads the HF-nested Qwen2 layout
(`text_config`/`vision_config`). LongVA and Flash-VStream ship flat LLaVA-style configs
and name their tower as an external HF repo, so it raises on both. Throughput needs none
of that -- it is a measured wall-clock number sitting in the span CSV -- so this reads it
directly and works for any backbone.

Throughput is quoted from the steady-state span only (`local_window_full`), matching
collect_cost_model.py: before the local window fills, the eviction/offload path has not
engaged and the run is measuring a different system. An arm that never reached steady
state is reported with a '*' and its pre-steady figure, which is optimistic -- it must not
be compared with a steady row.

Validated against results/cost_model_backbones/llava_ov_7b-fps1-table.csv, which
collect_cost_model.py produced: baseline 7.107 +/- 0.108, rlt@0.5 14.667 +/- 0.240.

Usage:
    python scripts/efficiency/throughput_table.py --dir results/cost_model_backbones
"""

import os
import re
import csv
import glob
import argparse
import statistics as st
from collections import defaultdict

PAT = re.compile(r'^(?P<model>.+?)-fps(?P<fps>[0-9.]+)-v(?P<v>\d+)-'
                 r'(?P<arm>baseline|rlt[0-9.]+)-(?P<timing>span|per_chunk)'
                 r'(?P<prep>-gpuprep)?\.csv$')


def read_span(path):
    """(fps, reached_steady, keep_rate) for one arm CSV."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    steady = [r for r in rows if str(r.get('local_window_full', '')).strip().lower() == 'true']
    reached = bool(steady)
    used = steady if reached else rows
    frames = sum(float(r['num_frames']) for r in used)
    secs = sum(float(r['seconds']) for r in used)
    if secs <= 0:
        return None
    keep = None
    for r in rows:
        if r.get('keep_rate'):
            keep = float(r['keep_rate'])
            break
    return frames / secs, reached, (keep if keep is not None else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/cost_model_backbones')
    ap.add_argument('--timing', default='span', choices=['span', 'per_chunk'])
    ap.add_argument('--gpu_preprocess', action='store_true',
                    help='Table the -gpuprep arms instead of the CPU-preprocess ones. '
                         'The two differ in throughput and nothing else, so they are '
                         'never pooled.')
    ap.add_argument('--out', default=None, help='Optional CSV to write.')
    args = ap.parse_args()

    want_prep = '-gpuprep' if args.gpu_preprocess else None
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.dir, '*.csv'))):
        m = PAT.match(os.path.basename(path))
        if not m or m.group('timing') != args.timing or m.group('prep') != want_prep:
            continue
        rec = read_span(path)
        if rec is None:
            continue
        arm = m.group('arm')
        arm = 'baseline' if arm == 'baseline' else f"rlt@{float(arm[3:]):g}"
        data[(m.group('model'), float(m.group('fps')), arm)].append(rec)

    if not data:
        raise SystemExit(f'no span CSVs matched in {args.dir}')

    def arm_key(a):
        return (0, 0.0) if a == 'baseline' else (1, float(a.split('@')[1]))

    rows = []
    models = sorted({k[0] for k in data})
    print(f"{'backbone':<18}{'fps_in':>7}  {'arm':<10}{'n':>3}{'fps':>9}{'sd':>8}"
          f"{'speedup':>9}{'keep':>8}  steady")
    for model in models:
        for fps_in in sorted({k[1] for k in data if k[0] == model}):
            base = data.get((model, fps_in, 'baseline'))
            base_fps = st.mean([r[0] for r in base]) if base else None
            for arm in sorted({k[2] for k in data if k[:2] == (model, fps_in)}, key=arm_key):
                rs = data[(model, fps_in, arm)]
                v = [r[0] for r in rs]
                mean = st.mean(v)
                sd = st.stdev(v) if len(v) > 1 else 0.0
                keep = st.mean([r[2] for r in rs])
                all_steady = all(r[1] for r in rs)
                spd = mean / base_fps if base_fps else float('nan')
                flag = '' if all_steady else '  * PRE-STEADY'
                print(f"{model:<18}{fps_in:>7.0f}  {arm:<10}{len(v):>3}{mean:>9.3f}{sd:>8.3f}"
                      f"{spd:>8.2f}x{keep:>8.3f}{flag}")
                rows.append(dict(model=model, sample_fps=fps_in, arm=arm, n_streams=len(v),
                                 throughput_fps=mean, throughput_fps_sd=sd,
                                 speedup_vs_baseline=spd, keep_rate=keep,
                                 reached_steady_state=all_steady))
    print("\n* pre-steady: the local window never filled, so eviction/offload never "
          "engaged.\n  Optimistic; do not compare with a steady row.")

    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
