"""Sweep RLT's threshold x sampling rate and report how many visual tokens survive.

Why this can be measured without generating a single answer: stage-2 pruning
(model/token_pruning.py) decides keep/drop purely from the projected+pooled features of
the frames it has already seen. The LM never influences that decision. So the keep rate
of a `--prune_threshold` on a given clip is fully determined by the vision tower +
projector output, and a full QA run -- four GPUs, ~45 min per sampling rate on FPSBench --
tells you nothing about token counts that this script cannot tell you in minutes. Run the
QA only for the thresholds whose cost/retention tradeoff you actually want scored.

The sweep reuses the encoded features across thresholds: each clip is pushed through the
vision tower once per sampling rate, cached as (T, 196, D), and then replayed through a
fresh `StreamingTokenPruner` per threshold. The replay is the real pruner class, fed in
64-frame chunks exactly as `Abstract_ReKV.encode_video` feeds it, so the keep rates here
are the keep rates the eval will produce -- not an approximation of them. (Under the
default cosine metric that also equals what the streaming path would produce frame by
frame; see the chunk-invariance note in model/token_pruning.py.)

Sampling rate is the interesting second axis on FPSBench specifically. Redundancy is a
property of the *sampled* sequence, not of the video: doubling the rate halves the motion
between consecutive samples, so more tokens fall under any fixed threshold. A threshold
calibrated at 1 fps therefore prunes much more aggressively at 8 fps -- which is exactly
where FPSBench's questions need the frames, since each one carries a `min_fps` below which
its evidence is unresolvable. Both the per-clip and per-frame outputs carry `min_fps`, so
retention can be read against the rate the question actually demands.

Note the effective rate is not the requested one: `BaseVQA.load_video` takes an integer
stride, so requesting 8 fps from a 30 fps file delivers 30/3 = 10 fps. The CSVs record
`effective_fps` alongside `requested_fps`; group by the former when comparing clips.

Outputs (--out_dir, default results/analysis/rlt_fpsbench):
  clips.csv   one row per (clip, requested_fps, threshold): token counts, keep rate, and
              the distance percentiles that produced them -- read those to pick the next
              threshold to try rather than guessing.
  frames.csv  one row per (clip, requested_fps, threshold, frame): survivors out of 196.
              Frame 0 is always 196 (nothing to diff against); the decay after it is the
              per-frame view of the same sweep.

Usage:
  python video_qa/analyze_rlt_threshold.py --num_videos 96
  python video_qa/analyze_rlt_threshold.py --thresholds 0.05 0.1 0.2 --sample_fps_list 1 8
"""

import argparse
import json
import os
import random
import warnings
from collections import defaultdict

import pandas as pd
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import logging as hf_logging

import logzero
from logzero import logger

from model.token_pruning import StreamingTokenPruner
from video_qa.base import MODELS


def sample_clips(anno, num_videos, seed):
    """Stratified sample over question_type.

    FPSBench's nine task families do not move at the same speed -- `blink_and_miss` and
    `fine_grained_motion` are near-static between samples where `repetitive_motion` is
    not -- so an unstratified sample would report a keep rate that is mostly a statement
    about which families it happened to draw.
    """
    if num_videos <= 0 or num_videos >= len(anno):
        return list(anno)
    by_type = defaultdict(list)
    for item in anno:
        by_type[item['conversations'][0].get('question_type', 'unknown')].append(item)
    rng = random.Random(seed)
    for items in by_type.values():
        rng.shuffle(items)
    picked, types = [], sorted(by_type)
    idx = 0
    while len(picked) < num_videos:
        added = False
        for t in types:
            if idx < len(by_type[t]) and len(picked) < num_videos:
                picked.append(by_type[t][idx])
                added = True
        if not added:
            break
        idx += 1
    return picked


def load_video(video_path, sample_fps):
    """Frame sampling identical to `BaseVQA.load_video`, plus the rate it really used."""
    vr = VideoReader(video_path, ctx=cpu(0))
    native_fps = round(vr.get_avg_fps())
    stride = max(1, int(native_fps / sample_fps))
    frame_idx = list(range(0, len(vr), stride))
    video = vr.get_batch(frame_idx).asnumpy()
    return video, native_fps, native_fps / stride


def pad_index(n_native, native_fps, target_fps):
    """Index map replicating native frames onto a `target_fps` grid (zero-order hold).

    For a target rate above the file's own rate there are no frames to sample -- the only
    thing "32 fps" can mean for a 30 fps clip is that some frames are repeated. Each target
    timestamp takes the most recent native frame, which is what a capture card holding the
    last frame would deliver.

    What this measures, and what it does not: a repeated frame is bit-identical, so its
    features are identical and its distance to the carried reference is exactly 0. RLT
    therefore drops every duplicate at any threshold, and the surviving token count is
    unchanged from the native run -- which is the point, and is worth showing: ingestion
    cost tracks content, not nominal frame rate. It is NOT a temporal-resolution
    condition: no visual evidence is added, so a question needing genuine 32 fps evidence
    stays unanswerable, and the keep *rate* is deflated by construction because the
    denominator counts frames that never existed. Read tokens/second from padded rows,
    never keep %.
    """
    n_target = max(1, int(round(n_native / native_fps * target_fps)))
    return [min(n_native - 1, int(t * native_fps / target_fps)) for t in range(n_target)]


@torch.inference_mode()
def encode_features(model, video, encode_chunk_size=64):
    """(T, H, W, 3) uint8 -> (T, 196, D) projected+pooled tokens, on GPU.

    Same preprocessing and same 64-frame batching as `Abstract_ReKV.encode_video`, minus
    the LM: this stops at exactly the tensor `_ingest_video_features` hands the pruner.
    """
    feats = []
    for start in range(0, video.shape[0], encode_chunk_size):
        chunk = video[start:start + encode_chunk_size]
        pixel_values = model.processor.video_processor(
            chunk, return_tensors="pt").pixel_values_videos.to(model.device, model.dtype)
        out = model._get_video_features(pixel_values)          # (1, T*196, D)
        feats.append(out.view(chunk.shape[0], model.n_frame_tokens, -1))
    return torch.cat(feats, dim=0)


def sweep_thresholds(feats, thresholds, metric, refresh_every, encode_chunk_size=64):
    """Replay cached features through a fresh pruner per threshold.

    Returns {threshold: (per_frame_counts, distance_percentiles)}. The pruner is stateful
    and its reference updates depend on the threshold, so every threshold needs its own
    replay -- there is no single distance matrix all of them can be read off. The replay
    is cheap; the vision tower pass that produced `feats` is not, and that is shared.

    Counts always come from the real pruner. The distances are re-derived by `ref_probe`,
    which mirrors the pruner's reference update; under "cosine" that mirror is exact, so
    the percentiles describe precisely the decisions that produced the counts. Under "l2"
    the pruner divides by a running mean token norm the probe does not track, so the
    percentiles would be on a different scale -- they are reported as NaN rather than as a
    number that cannot be compared to the threshold.
    """
    record_dists = metric == 'cosine'
    nan5 = [float('nan')] * 5
    out = {}
    for thr in thresholds:
        pruner = StreamingTokenPruner(threshold=thr, metric=metric, refresh_every=refresh_every)
        counts, dists = [], []
        ref_probe, frame_idx = None, 0
        for start in range(0, feats.shape[0], encode_chunk_size):
            chunk = feats[start:start + encode_chunk_size]
            if record_dists:
                chunk_f32 = chunk.float()
                for t in range(chunk.shape[0]):
                    f = chunk_f32[t]
                    if ref_probe is None:
                        ref_probe = f.clone()
                    else:
                        d = 1.0 - torch.nn.functional.cosine_similarity(f, ref_probe, dim=-1)
                        dists.append(d)
                        m = d > thr
                        if refresh_every > 0 and frame_idx % refresh_every == 0:
                            m = torch.ones_like(m)
                        ref_probe[m] = f[m]
                    frame_idx += 1
            _, c = pruner(chunk)
            counts.extend(c)
        pct = (torch.quantile(torch.cat(dists),
                              torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=feats.device)).tolist()
               if dists else nan5)
        out[thr] = (counts, pct)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--anno_path', type=str, default='data/fpsbench/test_mc.json')
    parser.add_argument('--model', type=str, default='llava_ov_7b')
    parser.add_argument('--sample_fps_list', type=float, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--pad_to_fps', type=float, nargs='+', default=[],
                        help='Extra conditions above the files own frame rate, reached by '
                             'repeating native frames (see pad_index). Duplicates carry no '
                             'new evidence and RLT drops all of them, so read tokens/second '
                             'from these rows, not keep rate.')
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    parser.add_argument('--prune_metric', type=str, default='cosine', choices=['cosine', 'l2'])
    parser.add_argument('--prune_refresh_every', type=int, default=0)
    parser.add_argument('--num_videos', type=int, default=96,
                        help='Stratified over question_type; <=0 or >= dataset size uses all.')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--out_dir', type=str, default='results/analysis/rlt_fpsbench')
    parser.add_argument('--encode_chunk_size', type=int, default=64)
    args = parser.parse_args()

    hf_logging.set_verbosity_error()
    logzero.loglevel(20)  # INFO: the pruner logs per-chunk keep rates at DEBUG
    warnings.filterwarnings('ignore')
    os.makedirs(args.out_dir, exist_ok=True)

    anno = json.load(open(args.anno_path))
    clips = sample_clips(anno, args.num_videos, args.seed)
    conditions = ([(f, False) for f in args.sample_fps_list]
                  + [(f, True) for f in args.pad_to_fps])
    logger.info(f'{len(clips)} clips x {len(conditions)} conditions '
                f'({len(args.pad_to_fps)} padded) x {len(args.thresholds)} thresholds')

    cfg = MODELS[args.model]
    logger.info(f"Loading {cfg['model_path']}")
    model, _ = cfg['load_func'](model_path=cfg['model_path'], n_local=15000, topk=64, chunk_size=1)

    clip_rows, frame_rows = [], []
    for item in tqdm(clips, desc='clips'):
        conv = item['conversations'][0]
        meta = dict(video_id=item['video_id'],
                    question_type=conv.get('question_type'),
                    min_fps=conv.get('min_fps'),
                    duration_sec=conv.get('clip_duration_sec'))
        native_feats, native_fps_cached = None, None
        for req_fps, padded in conditions:
            try:
                if padded:
                    # Every native frame, then repeated onto the target grid. Encoded once
                    # per clip and shared across pad targets: a repeated frame's features
                    # are the original's, so gathering rows is exact, not an approximation.
                    if native_feats is None:
                        video, native_fps_cached, _ = load_video(item['video_path'], 1e9)
                        native_feats = encode_features(model, video, args.encode_chunk_size)
                    native_fps = native_fps_cached
                    idx = pad_index(native_feats.shape[0], native_fps, req_fps)
                    feats, eff_fps = native_feats[idx], float(req_fps)
                else:
                    video, native_fps, eff_fps = load_video(item['video_path'], req_fps)
                    feats = encode_features(model, video, args.encode_chunk_size)
            except Exception as e:                      # a clip decord cannot open
                logger.warning(f"skip {item['video_id']} @ {req_fps}: {e}")
                continue
            n_frames, P = feats.shape[0], feats.shape[1]
            for thr, (counts, pct) in sweep_thresholds(
                    feats, args.thresholds, args.prune_metric,
                    args.prune_refresh_every, args.encode_chunk_size).items():
                kept = int(sum(counts))
                clip_rows.append(dict(**meta, requested_fps=req_fps, padded=padded,
                                      native_fps=native_fps,
                                      effective_fps=eff_fps, n_frames=n_frames,
                                      threshold=thr, tokens_total=n_frames * P,
                                      tokens_kept=kept, keep_rate=kept / (n_frames * P),
                                      dist_p10=pct[0], dist_p25=pct[1], dist_p50=pct[2],
                                      dist_p75=pct[3], dist_p90=pct[4]))
                for i, c in enumerate(counts):
                    frame_rows.append(dict(video_id=item['video_id'],
                                           question_type=meta['question_type'],
                                           min_fps=meta['min_fps'], requested_fps=req_fps,
                                           padded=padded, effective_fps=eff_fps,
                                           threshold=thr, frame_idx=i,
                                           tokens_kept=c, tokens_total=P))
            del feats
        native_feats = None
        torch.cuda.empty_cache()

    clip_df = pd.DataFrame(clip_rows)
    frame_df = pd.DataFrame(frame_rows)
    clip_df.to_csv(f'{args.out_dir}/clips.csv', index=False)
    frame_df.to_csv(f'{args.out_dir}/frames.csv', index=False)
    logger.info(f'wrote {len(clip_df)} clip rows and {len(frame_df)} frame rows '
                f'-> {args.out_dir}')

    # Token-weighted, not the mean of per-clip rates: a 3-frame clip and a 200-frame clip
    # do not cost the same, and the number that matters is tokens fed to the LM.
    pivot = clip_df.groupby(['threshold', 'requested_fps']).apply(
        lambda g: 100.0 * g.tokens_kept.sum() / g.tokens_total.sum(),
        include_groups=False).unstack()
    print('\nkeep rate % (token-weighted), rows=threshold, cols=requested fps\n')
    print(pivot.round(1).to_string())


if __name__ == '__main__':
    main()
