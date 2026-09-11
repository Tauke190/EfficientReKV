"""Assemble the efficiency table from a scripts/efficiency/cost_model.sh sweep.

Three columns, three different kinds of number, and the table is only readable if you
know which is which:

* **KV-Cache growth** -- measured. `calc_memory_usage()` at the end of the run, over the
  hours of video that produced it. Quoted per hour of *video*, not of wall-clock, because
  that is the axis it actually scales on and it makes runs of different lengths
  comparable. The analytic prediction is printed beside it as a check, not as the answer.

* **Streaming throughput** -- measured, and it is why every arm is a real run. Stage 2
  drops tokens before the LM prefill, ~84% of `_encode_video_chunk`, so this cannot be
  derived from the keep rate: the vision tower and preprocessing are untouched and set a
  ceiling on the speed-up. Reported from the steady-state chunks (those after the local
  window filled), since the pre-steady ones run without the eviction/offload path.

* **GFLOPs/frame** -- analytic, from config.json, times the measured keep rate. Nothing
  here is profiled: a profiler measures a kernel schedule, and what belongs in a cost
  table is the arithmetic the method actually removes. Cross-checked against parameter
  counts (implied non-embedding params come out at 0.358 B / 6.53 B for the 0.5B / 7B,
  which is Qwen2 with its tied 151k-vocab embedding excluded).

Two subtleties the arithmetic has to get right:

* **GQA.** The cache holds `num_key_value_heads`, not `num_attention_heads` -- 4 vs 28 on
  the 7B, a 7x error if confused.
* **Attention scales linearly with keep rate, not quadratically.** `n_local` is a budget
  in *tokens*, so the local window still holds n_local tokens under pruning; they just
  span more wall-clock time. Keys attended per query are unchanged and only the query
  count per frame falls. Counting the context as shrinking too would double the claimed
  saving.

Usage:
    python scripts/efficiency/collect_cost_model.py --model llava_ov_7b --sample_fps 1
"""

import os
import re
import json
import glob
import argparse

import numpy as np
import pandas as pd

# Model paths inlined rather than imported from video_qa.base, which pulls in torch and
# every model backend. This script only ever reads CSVs and a config.json, and the machine
# you re-table results on is usually a login node where torch cannot even map its shared
# objects. Same reasoning as video_qa/reduction_args.py, which exists so the launcher can
# parse a flag without importing a backend. Mirrors MODELS in video_qa/base.py.
MODEL_PATHS = {
    'llava_ov_0.5b': 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf',
    'llava_ov_7b': 'model_zoo/llava-onevision-qwen2-7b-ov-hf',
    'llava_ov_72b': 'model_zoo/llava-onevision-qwen2-72b-ov-hf',
    'video_llava_7b': 'model_zoo/Video-LLaVA-7B-hf',
    'longva_7b': 'model_zoo/LongVA-7B',
}


# ---------------------------------------------------------------------------------
# Closed forms, read off config.json. Nothing measured.
# ---------------------------------------------------------------------------------

def text_dims(cfg):
    """(d, m, n_layers, h, g, dh) for the Qwen2 backbone.

    `head_dim` is null in both shipped configs, HF's way of saying hidden_size /
    num_attention_heads; computing it rather than trusting the field keeps this correct
    if a future config sets it explicitly.
    """
    tc = cfg['text_config']
    d, h = tc['hidden_size'], tc['num_attention_heads']
    return (d, tc['intermediate_size'], tc['num_hidden_layers'], h,
            tc['num_key_value_heads'], tc.get('head_dim') or d // h)


def kv_bytes_per_token(cfg, dtype_bytes=2):
    """Bytes one token occupies in the KV-Cache: K and V, per layer, at the KV heads."""
    _, _, n_layers, _, g, dh = text_dims(cfg)
    return 2 * n_layers * g * dh * dtype_bytes


def lm_linear_gflops_per_token(cfg):
    """LM weight matmuls per token: q/k/v/o projections + gated MLP, 2 FLOPs per MAC.

    Exactly proportional to tokens fed, which is the term stage 2 shrinks linearly.
    Attention-over-context is not here; it depends on how much context is attended.
    """
    d, m, n_layers, h, g, dh = text_dims(cfg)
    return n_layers * 2 * (d * h * dh + 2 * d * g * dh + h * dh * d + 3 * d * m) / 1e9


def lm_attn_gflops_per_token(cfg, ctx_tokens):
    """QK^T and A.V per query token, over `ctx_tokens` keys."""
    _, _, n_layers, h, _, dh = text_dims(cfg)
    return n_layers * 4 * ctx_tokens * h * dh / 1e9


def vision_gflops_per_frame(cfg):
    """SigLIP tower + projector, per frame. Constant; stage 2 cannot touch it.

    Stage 2 sits after the projector, so every frame pays this whatever the threshold. It
    is the floor under per-frame cost and the reason the GFLOPs curve flattens rather than
    going to zero. (Stage 1, model/vision_reduction.py, is what attacks this term.)

    SigLIP is MHA with a plain two-matmul MLP, not Qwen2's gated three. The projector runs
    on the full 729-token grid: `apply_pooling` to 196 happens after it.
    """
    vc = cfg['vision_config']
    d, m, n_layers = vc['hidden_size'], vc['intermediate_size'], vc['num_hidden_layers']
    h = vc['num_attention_heads']
    dh = d // h
    n = (vc['image_size'] // vc['patch_size']) ** 2                  # 729
    tower = n_layers * (2 * n * (4 * d * dh * h + 2 * d * m) + 4 * n * n * h * dh)
    d_text = cfg['text_config']['hidden_size']
    projector = 2 * n * (d * d_text + d_text * d_text)               # pre-pooling
    return (tower + projector) / 1e9


# ---------------------------------------------------------------------------------

def read_arm(path):
    """One arm's CSV -> the handful of numbers the table needs.

    Throughput comes from the steady-state chunks only -- `local_window_full`, i.e. after
    the cached video tokens passed n_local and the eviction/offload path went live. Before
    that the run is measuring a different system. If an arm never reached steady state the
    overall figure is used and the row is flagged, rather than silently mixing the two.
    """
    df = pd.read_csv(path)
    steady = df[df.local_window_full] if 'local_window_full' in df else df.iloc[0:0]
    reached = len(steady) > 0
    used = steady if reached else df
    frames = df.num_frames.sum()
    return {
        'keep_rate': float(df.keep_rate.iloc[0]) if df.keep_rate.notna().any() else 1.0,
        'fps': used.num_frames.sum() / used.seconds.sum(),
        'fps_overall': frames / df.seconds.sum(),
        'steady': reached,
        'n_steady': len(steady),
        'kv_bytes': float(df.kv_cache_bytes.iloc[0]),
        # Recomputed rather than read: runs written before the fix stored a video_hours
        # covering only the timed frames, which inflates every per-hour figure by the
        # share that went to the FLOP counter instead.
        'video_hours': (frames + int(_col(df, 'flops_frames') or 0))
                       / float(df.sample_fps.iloc[0]) / 3600.0,
        'gpu_peak_bytes': float(df.gpu_peak_bytes.iloc[0]),
        'gpu_weights_bytes': float(df.gpu_weights_bytes.iloc[0]),
        'n_frames': int(frames),
        'n_frame_tokens': int(df.n_frame_tokens.iloc[0]),
        'n_tokens_fed': _col(df, 'n_tokens_fed'),
        'flops_keep_rate': _col(df, 'flops_keep_rate'),
        'gflops_measured': _col(df, 'gflops_per_frame'),
        'gflops_matmul': _col(df, 'gflops_matmul_per_frame'),
        'gflops_attn': _col(df, 'gflops_attn_per_frame'),
        'flops_frames': int(_col(df, 'flops_frames') or 0),
        'flops_backend': (df.flops_attn_backend.iloc[0]
                          if 'flops_attn_backend' in df else None),
    }


def _mean_or_none(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _col(df, name):
    """A run-wide scalar column, or None when the run did not record it."""
    if name not in df or not df[name].notna().any():
        return None
    return float(df[name].iloc[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llava_ov_7b',
                        choices=sorted(MODEL_PATHS))
    parser.add_argument('--sample_fps', type=float, default=1.0)
    parser.add_argument('--n_local', type=int, default=15000,
                        help='Context length the attention term is quoted at. Must match '
                             'the runs, or the GFLOPs column describes a different system.')
    parser.add_argument('--dir', type=str, default='results/cost_model')
    parser.add_argument('--timing', type=str, default='span',
                        choices=['span', 'per_chunk'],
                        help="Which timing protocol's arms to table. The two are not "
                             "comparable -- per_chunk syncs around every chunk, which at "
                             "chunk size 1 serializes CPU preprocessing against GPU "
                             "compute and reads low -- so a directory holding both is "
                             "filtered rather than pooled.")
    parser.add_argument('--out', type=str, default=None,
                        help='Table CSV (default <dir>/<model>-fps<f>-table.csv).')
    args = parser.parse_args()

    cfg = json.load(open(os.path.join(MODEL_PATHS[args.model], 'config.json')))
    kv_tok = kv_bytes_per_token(cfg)
    lm_lin = lm_linear_gflops_per_token(cfg)
    lm_att = lm_attn_gflops_per_token(cfg, args.n_local)
    vis = vision_gflops_per_frame(cfg)

    pat = os.path.join(args.dir,
                       f'{args.model}-fps{args.sample_fps:g}-v*-*-{args.timing}.csv')
    arms = {}
    for path in sorted(glob.glob(pat)):
        if path.endswith('_qa.csv') or path.endswith('-table.csv'):
            continue
        m = re.search(r'-v(\d+)-(baseline|rlt([0-9.]+))-(?:span|per_chunk)\.csv$', path)
        if not m:
            continue
        arm = 'baseline' if m.group(2) == 'baseline' else float(m.group(3))
        arms.setdefault(arm, []).append(read_arm(path))
    if not arms:
        raise SystemExit(f'no arm CSVs matched {pat} -- run scripts/efficiency/cost_model.sh first')

    def order(k):
        return (0, 0.0) if k == 'baseline' else (1, k)

    rows = []
    for arm in sorted(arms, key=order):
        rs = arms[arm]
        keep = float(np.mean([r['keep_rate'] for r in rs]))
        tok = rs[0]['n_frame_tokens'] * keep
        kv_meas_gb_h = float(np.mean([r['kv_bytes'] / r['video_hours'] for r in rs])) / 1024 ** 3
        # Tokens fed x exact bytes/token: measured, and unlike the offloaded figure it is
        # correct for an arm whose cache never left the GPU.
        toks = _mean_or_none([r['n_tokens_fed'] for r in rs])
        kv_gb_h = (float(np.mean([r['n_tokens_fed'] / r['video_hours'] for r in rs]))
                   * kv_tok / 1024 ** 3) if toks else kv_meas_gb_h
        rows.append({
            'arm': 'baseline' if arm == 'baseline' else f'rlt@{arm:g}',
            'n_streams': len(rs),
            'keep_rate': keep,
            'keep_rate_sd': float(np.std([r['keep_rate'] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
            'tokens_per_frame': tok,
            # measured
            'kv_gb_per_hour': kv_gb_h,
            'kv_mib_per_frame': kv_gb_h * 1024 / (args.sample_fps * 3600),
            'kv_gb_per_hour_offloaded': kv_meas_gb_h,
            'throughput_fps': float(np.mean([r['fps'] for r in rs])),
            'throughput_fps_sd': float(np.std([r['fps'] for r in rs], ddof=1)) if len(rs) > 1 else 0.0,
            'gpu_peak_gb': float(np.mean([r['gpu_peak_bytes'] for r in rs])) / 1024 ** 3,
            'reached_steady_state': all(r['steady'] for r in rs),
            # measured -- present only if the runs used --flops_frames
            'gflops_per_frame': _mean_or_none([r['gflops_measured'] for r in rs]),
            'gflops_matmul_per_frame': _mean_or_none([r['gflops_matmul'] for r in rs]),
            'gflops_attn_per_frame': _mean_or_none([r['gflops_attn'] for r in rs]),
            'flops_frames': rs[0]['flops_frames'],
            'flops_keep_rate': _mean_or_none([r['flops_keep_rate'] for r in rs]),
            # analytic, kept beside the measured figures as a cross-check
            'kv_gb_per_hour_analytic': kv_tok * tok * args.sample_fps * 3600 / 1024 ** 3,
            'gflops_per_frame_analytic': vis + (lm_lin + lm_att) * tok,
            'vision_gflops_per_frame_analytic': vis,
            'lm_linear_gflops_per_frame_analytic': lm_lin * tok,
            'lm_attn_gflops_per_frame_analytic': lm_att * tok,
        })
    t = pd.DataFrame(rows)
    # The measured column is the headline when the runs recorded one; the analytic one
    # stands in otherwise, and the header says which is being shown either way.
    rs0_backend = next((r['flops_backend'] for a in arms for r in arms[a]
                        if r['flops_backend']), 'unknown')
    # Per arm, not all-or-nothing: an arm that never reached steady state measured no
    # FLOPs (there was no steady frame to measure), and falling the whole table back to
    # the config-derived number because of it would throw away every arm that did.
    t['gflops_is_measured'] = t.gflops_per_frame.notna()
    t['gflops_per_frame'] = t.gflops_per_frame.fillna(t.gflops_per_frame_analytic)
    measured = bool(t.gflops_is_measured.any())
    base = t.iloc[0]
    t['kv_vs_baseline'] = base.kv_gb_per_hour / t.kv_gb_per_hour
    t['speedup_vs_baseline'] = t.throughput_fps / base.throughput_fps
    t['gflops_vs_baseline'] = base.gflops_per_frame / t.gflops_per_frame

    out = args.out or os.path.join(args.dir, f'{args.model}-fps{args.sample_fps:g}-table.csv')
    t.to_csv(out, index=False)

    n_fr = arms[sorted(arms, key=order)[0]][0]['n_frames']
    print(f'\n{args.model} @ {args.sample_fps:g} fps -- {n_fr} frames/stream, '
          f'{t.n_streams.iloc[0]} stream(s), 1 frame/forward (streaming), '
          f'{args.timing} timing')
    n_meas = int(t.gflops_is_measured.sum())
    if measured:
        print(f'GFLOPs/frame: MEASURED for {n_meas}/{len(t)} arms over '
              f'{int(t[t.gflops_is_measured].flops_frames.max())} steady-state frames each')
        print(f'              -- aten ops via FlopCounterMode plus LM attention from the '
              f'shapes append()\n                 was called with ({rs0_backend} backend). '
              f"Rows marked ~ are config-derived\n                 instead: no steady "
              f'frame existed to measure.')
    else:
        print(f'GFLOPs/frame: analytic (no --flops_frames in these runs). '
              f'Vision floor {vis:.0f}, attention at ctx={args.n_local}.')
    print(f'config-derived: {rows[0]["tokens_per_frame"]:.0f} tokens/frame baseline, '
          f'{kv_tok} KV bytes/token\n')

    hdr = (f'{"arm":>12} {"keep":>7} {"KV GB/h":>9} {"vs base":>8} '
           f'{"enc f/s":>8} {"speedup":>8} {"GF/frame":>9} {"vs base":>8} {"GPU GB":>7}')
    print(hdr)
    print('-' * len(hdr))
    for _, r in t.iterrows():
        flag = ('' if r.reached_steady_state else '  *') + ('' if r.gflops_is_measured else ' ~')
        print(f'{r.arm:>12} {r.keep_rate * 100:6.1f}% {r.kv_gb_per_hour:9.2f} '
              f'{r.kv_vs_baseline:7.2f}x {r.throughput_fps:8.2f} '
              f'{r.speedup_vs_baseline:7.2f}x {r.gflops_per_frame:9.1f} '
              f'{r.gflops_vs_baseline:7.2f}x {r.gpu_peak_gb:7.2f}{flag}')

    if args.timing == 'span':
        print('\nthroughput is the steady-state span: frames encoded after the local '
              'window filled,\ndivided by the wall clock across them, with three CUDA '
              'syncs in the whole run.')
    else:
        print('\nthroughput sums per-chunk timings, one sync pair per chunk. At 1 '
              'frame/forward those\nbarriers serialize preprocessing against compute, '
              'so read this as a lower bound.')

    if not t.reached_steady_state.all():
        # The window fills after n_local / (n_frame_tokens * keep) frames, so the harder
        # an arm prunes the longer it takes to get there -- exactly the arms most likely
        # to be flagged. Saying how many frames each one needed turns the flag into an
        # instruction.
        P = rows[0]['tokens_per_frame'] / max(rows[0]['keep_rate'], 1e-9)
        print('\n  * never filled the local window, so throughput is the whole-run figure '
              'and is not\n    comparable to the steady-state rows. Frames needed at '
              f'n_local={args.n_local}:')
        for _, r in t[~t.reached_steady_state].iterrows():
            need = args.n_local / (P * max(r.keep_rate, 1e-9))
            print(f'      {r.arm:>10}: {need:6.0f} frames '
                  f'({need / args.sample_fps:5.0f} s at {args.sample_fps:g} fps) '
                  f'-- had {n_fr}')
        print('    Re-run those arms with a larger NUM_FRAMES (and FPS, since 600 s is '
              'all there is).')

    if measured:
        m = t[t.gflops_is_measured]
        gap = (m.gflops_per_frame / m.gflops_per_frame_analytic - 1)
        print(f'\nGFLOPs check: measured vs the config-derived prediction differs by '
              f'{gap.min() * 100:+.1f}% to {gap.max() * 100:+.1f}%.')
        drift = m[m.flops_keep_rate.notna()]
        if len(drift):
            d = (drift.flops_keep_rate / drift.keep_rate - 1)
            print(f'  the sampled frames\' own keep rate vs the run\'s: '
                  f'{d.min() * 100:+.1f}% to {d.max() * 100:+.1f}% '
                  f'(how far the GFLOPs sample sits from its row)')
        print(f'  attention share of measured total: '
              + ', '.join(f'{r.arm} {r.gflops_attn_per_frame / r.gflops_per_frame * 100:.0f}%'
                          for _, r in t[t.gflops_is_measured].iterrows()))

    err = abs(t.kv_gb_per_hour / t.kv_gb_per_hour_analytic - 1).max()
    print(f'\nKV GB/h is the tokens the LM was actually fed x {kv_tok} B/token '
          f'(exact for the config).\n  Against the keep-rate prediction it differs by at '
          f'most {err * 100:.1f}%. The offloaded-only figure\n  (calc_memory_usage) is in '
          f'the CSV as kv_gb_per_hour_offloaded, and reads 0 for any arm\n  whose cache '
          f'never left the GPU.')
    if t.n_streams.iloc[0] == 1:
        print('n=1: keep_rate and everything scaling with it are a point estimate '
              '(between-video CV 21-42%).\n     The config-derived columns are exact.')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
