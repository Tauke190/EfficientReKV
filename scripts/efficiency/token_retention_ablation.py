"""Token retention of the three ablation pruners (rlt_ref / rlt_prev / rlt_frame).

Companion to the accuracy ablation in scripts/eval/ablation.slurm. That job reports how
well each arm *answers* at each threshold; this reports how much it *keeps* -- the token
retention curve the accuracy numbers sit against. The two together are the ablation: an
arm is only interesting where it answers as well while keeping fewer tokens.

    rlt_ref    ours        keep token iff dist(feat[t,p], last KEPT feat[p]) > thr
    rlt_prev   reference   keep token iff dist(feat[t,p], feat[t-1,p])       > thr
    rlt_frame  granularity keep ALL of frame t iff >50% of its tokens pass the rlt_ref test

Why this needs no LLM, and no QA run. Stage-2 pruning (model/token_pruning.py) decides
keep/drop purely from the projected+pooled features of the frames it has already seen; the
LM never influences that decision. So the keep rate of a (method, threshold) on a clip is
fully determined by the vision tower + projector output. This script stops at exactly the
tensor `_ingest_video_features` hands the pruner, which is why a smoke test on one video
takes seconds and the full benchmark minutes, where the accuracy ablation is four GPUs for
the better part of a day. The keep rates here are the keep rates the eval records in
`token_keep_rate`, not an approximation: each clip is encoded once, then replayed through
the real pruner class in 64-frame chunks exactly as `Abstract_ReKV.encode_video` feeds it.
(Under the default cosine metric that also equals the per-frame streaming result -- see the
chunk-invariance note in model/token_pruning.py.)

At equal threshold the three keep different numbers of tokens, so read this as a curve, not
a single point: rlt_prev's frame-to-frame distances are smaller, so it keeps less at the
same threshold; rlt_frame keeps whole frames. The accuracy ablation is compared at matched
keep rate, and this is the table that lets you line a threshold up with its keep rate.

Outputs (--out_dir):
  retention.csv   one row per (video, method, threshold): tokens_total, tokens_kept,
                  keep_rate, n_frames.
  summary.csv     one row per (method, threshold): token-weighted keep rate over all
                  videos (a 3-frame clip and a 200-frame clip do not cost the same, so the
                  figure that matters is total tokens kept / total tokens seen, not the
                  mean of per-clip rates).
A pivot of summary.csv (rows=threshold, cols=method) is printed at the end.

Usage (needs a GPU):
  # smoke test on one video first
  python scripts/efficiency/token_retention_ablation.py --video /path/to/test.mp4
  # then the whole StreamBench benchmark
  python scripts/efficiency/token_retention_ablation.py --anno_path data/streambench/full_oe.json
  # a subsample, or a single method, or a finer grid
  python scripts/efficiency/token_retention_ablation.py --num_videos 32
  python scripts/efficiency/token_retention_ablation.py --methods rlt_ref --thresholds 0.5 0.7

Normally driven by scripts/efficiency/token_retention_ablation.sh.
"""

import argparse
import os
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging as hf_logging

import logzero
from logzero import logger

# Reuse the exact frame sampling and encode path the rlt_ref sweep already uses, so the
# features these pruners see are bit-identical to the ones the eval produces.
from video_qa.analyze_rlt_threshold import load_video, encode_features, sample_clips
from video_qa.base import MODELS
from model.token_pruning import build_pruner, PRUNERS


def sweep(feats, methods, thresholds, metric, encode_chunk_size=64):
    """Replay cached features through a fresh pruner per (method, threshold).

    Returns {(method, threshold): tokens_kept}. A pruner is stateful and its reference
    updates depend on the threshold, so every (method, threshold) needs its own replay;
    the replay is cheap, and the vision-tower pass that produced `feats` -- shared across
    all of them -- is not. Fed in `encode_chunk_size` chunks exactly as the offline eval
    feeds it (chunk-invariant under cosine).
    """
    out = {}
    for method in methods:
        for thr in thresholds:
            pruner = build_pruner(method, threshold=thr, metric=metric)
            kept = 0
            for start in range(0, feats.shape[0], encode_chunk_size):
                _, counts = pruner(feats[start:start + encode_chunk_size])
                kept += int(sum(counts))
            out[(method, thr)] = kept
    return out


def build_anno(args):
    """Clips to measure: a single --video (smoke test) or an annotation file."""
    if args.video:
        assert os.path.exists(args.video), f'no such video file: {args.video}'
        vid = os.path.splitext(os.path.basename(args.video))[0]
        return [{'video_id': vid, 'video_path': args.video,
                 'conversations': [{}]}]
    import json
    anno = json.load(open(args.anno_path))
    return sample_clips(anno, args.num_videos, args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--video', type=str, default=None,
                     help='Single video file: the smoke-test path. Measures this clip '
                          'alone and ignores --anno_path/--num_videos.')
    src.add_argument('--anno_path', type=str, default='data/streambench/full_oe.json',
                     help='Annotation to sweep (full_oe.json schema: video_id, video_path, '
                          'conversations).')
    parser.add_argument('--model', type=str, default='llava_ov_7b')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['rlt_ref', 'rlt_prev', 'rlt_frame'],
                        help='Pruners to compare; any key in model.token_pruning.PRUNERS.')
    # Same grid as the rlt_ref/rlt_prev/rlt_frame arms in scripts/eval/ablation.slurm.
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.25, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--sample_fps', type=float, default=1.0,
                        help='Frame rate to sample at; StreamBench is evaluated at 1 fps.')
    parser.add_argument('--prune_metric', type=str, default='cosine',
                        choices=['cosine', 'l2'])
    parser.add_argument('--num_videos', type=int, default=-1,
                        help='Stratified subsample (<=0 or >= dataset size uses all). '
                             'Ignored with --video.')
    parser.add_argument('--max_frames', type=int, default=0,
                        help='Cap frames per clip (0 = no cap). Keeps the smoke test fast '
                             'on a long video; leave 0 for real numbers.')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--out_dir', type=str,
                        default='results/analysis/token_retention_ablation')
    parser.add_argument('--encode_chunk_size', type=int, default=64)
    args = parser.parse_args()

    hf_logging.set_verbosity_error()
    logzero.loglevel(20)  # INFO: the pruner logs per-chunk keep rates at DEBUG
    warnings.filterwarnings('ignore')
    os.makedirs(args.out_dir, exist_ok=True)

    unknown = [m for m in args.methods if m not in PRUNERS]
    assert not unknown, f'unknown methods {unknown}; registered: {sorted(PRUNERS)}'

    clips = build_anno(args)
    logger.info(f'{len(clips)} clips x {len(args.methods)} methods x '
                f'{len(args.thresholds)} thresholds @ {args.sample_fps} fps')

    cfg = MODELS[args.model]
    logger.info(f"Loading {cfg['model_path']}")
    model, _ = cfg['load_func'](model_path=cfg['model_path'], n_local=15000,
                                topk=64, chunk_size=1)

    rows = []
    for item in tqdm(clips, desc='clips'):
        try:
            video, native_fps, eff_fps = load_video(item['video_path'], args.sample_fps)
            if args.max_frames > 0:
                video = video[:args.max_frames]
            feats = encode_features(model, video, args.encode_chunk_size)
        except Exception as e:                         # a clip decord cannot open
            logger.warning(f"skip {item['video_id']}: {e}")
            continue
        n_frames, P = feats.shape[0], feats.shape[1]
        total = n_frames * P
        for (method, thr), kept in sweep(
                feats, args.methods, args.thresholds,
                args.prune_metric, args.encode_chunk_size).items():
            rows.append(dict(video_id=item['video_id'], method=method, threshold=thr,
                             n_frames=n_frames, effective_fps=eff_fps,
                             tokens_total=total, tokens_kept=kept,
                             keep_rate=kept / total))
        del feats
        torch.cuda.empty_cache()

    if not rows:
        logger.error('no clips produced any rows -- every clip failed to load. '
                     'Check the video path / annotation above.')
        return
    df = pd.DataFrame(rows)
    df.to_csv(f'{args.out_dir}/retention.csv', index=False)

    # Token-weighted keep rate per (method, threshold): tokens kept / tokens seen over all
    # clips, not the mean of per-clip rates (see the module docstring).
    summary = (df.groupby(['method', 'threshold'])
                 .apply(lambda g: pd.Series(dict(
                     n_videos=g.video_id.nunique(),
                     tokens_total=int(g.tokens_total.sum()),
                     tokens_kept=int(g.tokens_kept.sum()),
                     keep_rate=g.tokens_kept.sum() / g.tokens_total.sum())),
                     include_groups=False)
                 .reset_index())
    summary.to_csv(f'{args.out_dir}/summary.csv', index=False)
    logger.info(f'wrote {len(df)} rows -> {args.out_dir}/retention.csv '
                f'and {args.out_dir}/summary.csv')

    pivot = (summary.assign(keep_pct=(100 * summary.keep_rate).round(1))
                    .pivot(index='threshold', columns='method', values='keep_pct'))
    print('\ntoken retention % (token-weighted), rows=threshold, cols=method\n')
    print(pivot.to_string())


if __name__ == '__main__':
    main()
