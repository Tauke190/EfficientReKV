"""Accuracy averaged over benchmarks, beside token reduction, throughput and KV growth.

Reads each benchmark's headline accuracy from its score file and the per-benchmark cost
from results/cost_model/benchmark_cost.csv (scripts/efficiency/benchmark_cost.py), and
averages both over the benchmarks where *every* arm -- baseline included -- is on disk
and complete. A benchmark missing one arm is left out of the average for all arms, so
every row averages the same set and the deltas compare like with like.

Headline metric per benchmark (what each scorer calls its headline):
    streamingbench_real   strict_letter.overall         (micro, StreamingBench's own)
    ovobench_*            strict_letter.<split>_average (mean over tasks, OVO-Bench's own)
    ovbench, odvbench     macro_average                 (their scorers' recommendation)
    streambench           macro_average                 (Llama-3-8B judge)
    rvs_*                 % judged 'yes'                (GPT-3.5 judge)

Token reduction is the pruner's (1 - tokens_kept / tokens_seen), i.e. what the method
drops. KV growth and throughput come from the cost model at each benchmark's KV keep
rate, which includes block-alignment padding and so can sit above the pruner's rate.

Usage:
    python scripts/efficiency/benchmark_cost.py && python scripts/efficiency/summary_table.py
"""

import os
import json
import argparse

import numpy as np
import pandas as pd

from benchmark_cost import load_cost_table, benchmark_keep, cost_at


def rvs_accuracy(model_dir, bench, run, videos):
    """RVS judge accuracy restricted to `videos`, counted the way eval_open_ended.py does."""
    res = json.load(open(f'{model_dir}/{bench}/{run}/results_gpt-3_5-turbo-0125.json'))
    preds = [v[0].get('pred', v[0].get('prev', '')).lower() for k, v in res.items()
             if isinstance(v, list) and k.rsplit('_', 1)[0] in videos]
    yes, no = sum('yes' in p for p in preds), sum('yes' not in p and 'no' in p for p in preds)
    return 100.0 * yes / (yes + no)


def headline(model_dir, bench, run):
    d = f'{model_dir}/{bench}/{run}'
    try:
        if bench == 'streamingbench_real':
            return json.load(open(f'{d}/streamingbench_scores.json'))['strict_letter']['overall']
        if bench.startswith('ovobench_'):
            split = bench.split('_', 1)[1]
            return json.load(open(f'{d}/ovobench_scores.json'))['strict_letter'][f'{split}_average']
        if bench in ('ovbench', 'odvbench', 'streambench'):
            return json.load(open(f'{d}/{bench}_scores.json'))['macro_average']
        if bench.startswith('rvs_'):
            res = json.load(open(f'{d}/results_gpt-3_5-turbo-0125.json'))
            if 'accuracy' in res:
                return 100.0 * res['accuracy']
            # Older files carry only the verdicts; count them the way eval_open_ended.py does.
            preds = [v[0].get('pred', v[0].get('prev', '')).lower() for v in res.values() if isinstance(v, list)]
            yes, no = sum('yes' in p for p in preds), sum('yes' not in p and 'no' in p for p in preds)
            return 100.0 * yes / (yes + no)
    except (FileNotFoundError, KeyError):
        return None
    return None


def n_videos(model_dir, bench, run):
    try:
        return pd.read_csv(f'{model_dir}/{bench}/{run}/results.csv', usecols=['video_id']).video_id.nunique()
    except (FileNotFoundError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['llava_ov_7b', 'llava_ov_0.5b'])
    p.add_argument('--results_root', default='results')
    p.add_argument('--cost_csv', default='results/cost_model/benchmark_cost.csv')
    p.add_argument('--arms', nargs='+', default=['baseline', '0.25', '0.5', '0.6', '0.7', '0.8', '0.9'])
    p.add_argument('--cost_dir', default='results/cost_model')
    p.add_argument('--gpuprep', action='store_true',
                   help='Must match the table --cost_csv was built from; used when a benchmark is re-scored on a video subset.')
    p.add_argument('--rvs_common_videos', action='store_true',
                   help='Re-score RVS arms on the videos every arm finished, instead of '
                        'taking each arm as it is on disk.')
    p.add_argument('--out', default='results/cost_model/summary_table.csv')
    args = p.parse_args()

    cost = pd.read_csv(args.cost_csv)
    per_bench, summary = [], []
    for model in args.models:
        mdir = f'{args.results_root}/{model}'
        benches = sorted(b for b in os.listdir(mdir) if b != 'fpsbench_stream' and os.path.isdir(f'{mdir}/{b}'))
        arm_names = ['baseline' if a == 'baseline' else f'rlt_ref{float(a):g}' for a in args.arms]
        runs = {n: ('64-1.0' if n == 'baseline' else f'64-1.0-{n}cosine') for n in arm_names}

        used, dropped, notes = [], {}, {}
        for b in benches:
            rows = {}
            for n, run in runs.items():
                c = cost[(cost.model == model) & (cost.benchmark == b) & (cost.arm == n)]
                rows[n] = dict(acc=headline(mdir, b, run), nv=n_videos(mdir, b, run),
                               cost=c.iloc[0] if len(c) else None)
            missing = [n for n, r in rows.items() if r['acc'] is None or r['cost'] is None]
            nvs = {r['nv'] for r in rows.values() if r['nv'] is not None}
            if missing:
                dropped[b] = 'missing ' + ', '.join(missing)
                continue
            if len(nvs) > 1:
                if not b.startswith('rvs_'):
                    dropped[b] = f'arms cover different video counts {sorted(nvs)}'
                    continue
            if len(nvs) > 1 and not args.rvs_common_videos:
                # Default: take each arm's numbers as they are on disk, even though some
                # arms stopped part-way and cover fewer videos.
                notes[b] = f'as on disk; arms cover different video counts {sorted(nvs)}'
            elif len(nvs) > 1:
                # Some arms stopped part-way. RVS verdicts are keyed <video>_<q>, so every
                # arm can be re-scored on the videos all of them finished -- accuracy and
                # keep rate both -- rather than dropping the benchmark.
                common = set.intersection(*(set(pd.read_csv(f'{mdir}/{b}/{run}/results.csv',
                                                            usecols=['video_id']).video_id)
                                            for run in runs.values()))
                path, cbase, curve, steady = load_cost_table(args.cost_dir, model, 1.0, args.gpuprep)
                for n, run in runs.items():
                    rows[n]['acc'] = rvs_accuracy(mdir, b, run, common)
                    rows[n]['nv'] = len(common)
                    k = (dict(keep_rate=1.0, pruner_keep_rate=1.0) if n == 'baseline'
                         else benchmark_keep(f'{mdir}/{b}/{run}/results.csv', common))
                    rows[n]['cost'] = pd.Series({**k, **cost_at(k['keep_rate'], cbase, curve, steady)})
                notes[b] = f'scored on the {len(common)} videos every arm finished'
            used.append(b)
            for n, r in rows.items():
                c = r['cost']
                per_bench.append(dict(model=model, benchmark=b, arm=n, accuracy=r['acc'],
                                      n_videos=r['nv'],
                                      token_reduction=100 * (1 - c.pruner_keep_rate),
                                      throughput_fps=c.throughput_fps,
                                      kv_gb_per_hour=c.kv_gb_per_hour,
                                      throughput_extrapolated=bool(c.throughput_extrapolated)))

        df = pd.DataFrame([r for r in per_bench if r['model'] == model])
        avg = df.groupby('arm', sort=False).agg(
            accuracy=('accuracy', 'mean'), token_reduction=('token_reduction', 'mean'),
            throughput_fps=('throughput_fps', 'mean'), kv_gb_per_hour=('kv_gb_per_hour', 'mean'),
            n_extrapolated=('throughput_extrapolated', 'sum')).reindex(arm_names)
        base = avg.loc['baseline']
        avg['d_accuracy'] = avg.accuracy - base.accuracy
        avg['speedup'] = avg.throughput_fps / base.throughput_fps
        avg['d_kv_pct'] = 100 * (avg.kv_gb_per_hour / base.kv_gb_per_hour - 1)
        avg.insert(0, 'model', model)
        avg['benchmarks'] = ' '.join(used)
        summary.append(avg.reset_index())

        print(f'\n{model}: averaged over {len(used)} benchmarks: {", ".join(used)}')
        for b, why in notes.items():
            print(f'  note {b}: {why}')
        for b, why in dropped.items():
            print(f'  excluded {b}: {why}')
        with pd.option_context('display.width', 200):
            print(avg.drop(columns=['model', 'benchmarks']).round(2).to_string())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.concat(summary).round(3).to_csv(args.out, index=False)
    pd.DataFrame(per_bench).round(3).to_csv(args.out.replace('.csv', '_per_benchmark.csv'), index=False)
    print(f'\nwrote {args.out} and its _per_benchmark.csv')


if __name__ == '__main__':
    main()
