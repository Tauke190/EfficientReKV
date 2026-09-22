"""Throughput and memory per benchmark, read off the one-stream cost model.

scripts/efficiency/cost_model.sh profiles a single FPS-Bench-Stream stream, so its table
is exact for that stream's keep rate and nothing else. Under rlt_ref the cost of a video
is set by how many of its tokens survive, which varies a lot between videos and between
benchmarks. This script takes each benchmark's actual keep rate from its results.csv and
maps it through the cost-model curve:

* KV-Cache (MiB/frame, GB/hour of video) -- linear in keep rate, exactly: every kept token
  costs the same bytes. baseline x keep.
* GFLOPs/frame -- analytic: the vision tower is untouched, the LM scales with keep rate.
  vision + (LM attention + LM linear) x keep. Same formula collect_cost_model.py uses.
* Throughput (frames/s) -- measured, not linear (the vision tower and preprocessing set a
  ceiling), so it is interpolated in keep rate between the profiled arms. Only arms that
  reached steady state are trusted; a keep rate below the lowest of them is flagged
  `throughput_extrapolated` and interpolated against the non-steady arms.
* Peak GPU memory is not reported per benchmark: old blocks are offloaded to CPU, so it
  is ~flat across arms (weights + local window) and does not reflect the saving.

Keep rate is `n_tokens_fed / tokens_seen` at each video's last question -- what the KV
cache actually holds, block alignment included -- pooled over the benchmark's videos, so
it is the cost per hour of that benchmark's video. The pruner's own rate
(`tokens_kept / tokens_seen`) is reported beside it, as is the per-video spread.

Usage:
    python scripts/efficiency/benchmark_cost.py                      # llava_ov_7b and 0.5b
    python scripts/efficiency/benchmark_cost.py --models llava_ov_7b --gpuprep
"""

import os
import re
import glob
import argparse

import numpy as np
import pandas as pd

ARM_RE = re.compile(r'^64-(?P<fps>[\d.]+)(?P<query>-query)?(?:-rlt_ref(?P<thr>[\d.]+)cosine)?$')


def load_cost_table(cost_dir, model, fps, gpuprep):
    path = f"{cost_dir}/{model}-fps{fps:g}{'-gpuprep' if gpuprep else ''}-table.csv"
    df = pd.read_csv(path)
    base = df[df.arm == 'baseline'].iloc[0]
    curve = df.sort_values('keep_rate')
    steady = curve[curve.reached_steady_state.astype(bool)]
    return path, base, curve, steady


def benchmark_keep(results_csv, videos=None):
    df = pd.read_csv(results_csv)
    need = {'video_id', 'tokens_seen', 'tokens_kept', 'n_tokens_fed'}
    if not need <= set(df.columns):
        return None
    if videos is not None:
        df = df[df.video_id.isin(videos)]
    # Streaming runs log cumulative counts per question; the last one covers the whole
    # stream the model saw. Offline runs repeat the same numbers on every row.
    last = df.sort_values('tokens_seen').groupby('video_id').tail(1)
    last = last[last.tokens_seen > 0]
    per_video = last.n_tokens_fed / last.tokens_seen
    return {
        'n_videos': len(last),
        'keep_rate': last.n_tokens_fed.sum() / last.tokens_seen.sum(),
        'pruner_keep_rate': last.tokens_kept.sum() / last.tokens_seen.sum(),
        'keep_p10': per_video.quantile(0.10),
        'keep_median': per_video.median(),
        'keep_p90': per_video.quantile(0.90),
    }


def cost_at(keep, base, curve, steady):
    lm = base.lm_attn_gflops_per_frame_analytic + base.lm_linear_gflops_per_frame_analytic
    extrapolated = keep < steady.keep_rate.min()
    pts = curve if extrapolated else steady
    fps = float(np.interp(keep, pts.keep_rate, pts.throughput_fps))
    return {
        'throughput_fps': fps,
        'speedup': fps / base.throughput_fps,
        'throughput_extrapolated': bool(extrapolated),
        'kv_mib_per_frame': base.kv_mib_per_frame * keep,
        'kv_gb_per_hour': base.kv_gb_per_hour * keep,
        'kv_reduction': 1.0 / keep,
        'gflops_per_frame': base.vision_gflops_per_frame_analytic + lm * keep,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['llava_ov_7b', 'llava_ov_0.5b'])
    p.add_argument('--results_root', default='results')
    p.add_argument('--cost_dir', default='results/cost_model')
    p.add_argument('--sample_fps', type=float, default=1.0,
                   help='Only arms at this rate: the keep rate, and so the cost, depends on it.')
    p.add_argument('--gpuprep', action='store_true',
                   help='Use the GPU-preprocessing throughput table. Off by default: '
                        'cost_model.sh recommends CPU preprocessing for any throughput '
                        'quoted beside an accuracy number.')
    p.add_argument('--out', default='results/cost_model/benchmark_cost.csv')
    args = p.parse_args()

    rows = []
    for model in args.models:
        path, base, curve, steady = load_cost_table(args.cost_dir, model, args.sample_fps, args.gpuprep)
        print(f'{model}: cost model {path} (baseline {base.throughput_fps:.2f} f/s, '
              f'{base.kv_gb_per_hour:.2f} GB/h; steady-state down to keep '
              f'{steady.keep_rate.min():.3f})')
        for d in sorted(glob.glob(f'{args.results_root}/{model}/*/64-*')):
            m = ARM_RE.match(os.path.basename(d))
            if not m or float(m['fps']) != args.sample_fps or not os.path.isfile(f'{d}/results.csv'):
                continue
            bench = os.path.basename(os.path.dirname(d)) + ('_query' if m['query'] else '')
            if m['thr'] is None:
                rows.append({'model': model, 'benchmark': bench, 'arm': 'baseline', 'keep_rate': 1.0,
                             'pruner_keep_rate': 1.0, **cost_at(1.0, base, curve, steady)})
                continue
            k = benchmark_keep(f'{d}/results.csv')
            if k is None:
                print(f'  skip {d}: no token counts in results.csv')
                continue
            rows.append({'model': model, 'benchmark': bench, 'arm': f"rlt_ref{float(m['thr']):g}",
                         **k, **cost_at(k['keep_rate'], base, curve, steady)})

    out = pd.DataFrame(rows)
    out['_thr'] = out.arm.str.extract(r'([\d.]+)$').astype(float).fillna(0)
    out = out.sort_values(['model', 'benchmark', '_thr']).drop(columns='_thr')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.round(4).to_csv(args.out, index=False)

    show = out[['model', 'benchmark', 'arm', 'keep_rate', 'throughput_fps', 'speedup',
                'kv_gb_per_hour', 'kv_reduction', 'gflops_per_frame', 'throughput_extrapolated']]
    with pd.option_context('display.width', 200, 'display.max_rows', None):
        print(show.round(3).to_string(index=False))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
