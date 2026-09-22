"""Per-frame motion signals for a streaming benchmark, on the exact frames the eval sees.

For each video, on the same 1 fps stride grid FrameStream uses by default (source frames
0, s, 2s, ... with s = round(avg_fps) / sample_fps), this records:

  flow    -- mean Farneback optical-flow magnitude between slot k's source frame and the
             source frame right after it, in % of frame width per second. Instantaneous
             motion, independent of the pruner: pixels, not the backbone's features.
  moving  -- fraction of pixels whose flow exceeds 1 px/frame (at the analysis width).
             Separates a small moving object from whole-frame (camera) motion.
  diff1s  -- mean |luma difference| (0-255) between consecutive slots, i.e. across the
             1 s gap the model actually sees. Slot 0 is 0.
  hist1s  -- 1 - HSV-histogram correlation between consecutive slots. Spikes at shot cuts;
             kept for splitting "abrupt" from "gradual" later. Slot 0 is 0.

Bucketing is a separate step (it needs the distribution first), so this writes the raw
series only -- one JSON for the whole benchmark, not a file per video.

Usage:
    python scripts/dataset_prep/motion_flow.py \
        --anno data/StreamingBench/real.json \
        --out data/StreamingBench/motion/real_flow_frames.json --workers 16
"""

import os
import json
import argparse
from multiprocessing import Pool

import numpy as np
import cv2
import decord

WIDTH = 320  # analysis width; flow is reported relative to it, so the unit is resolution-free


def analyse(args):
    video_id, path, sample_fps = args
    cv2.setNumThreads(1)
    try:
        probe = decord.VideoReader(path, num_threads=1)
        n_src, src_fps = len(probe), round(probe.get_avg_fps())
        h0, w0 = probe[0].shape[:2]
        del probe
        height = max(2, int(round(WIDTH * h0 / w0 / 2)) * 2)
        vr = decord.VideoReader(path, width=WIDTH, height=height, num_threads=1)

        stride = max(1, int(src_fps / sample_fps))          # FrameStream's default grid
        slots = list(range(0, n_src, stride))
        pairs = [min(i + 1, n_src - 1) for i in slots]
        need = sorted(set(slots) | set(pairs))
        pos = {i: j for j, i in enumerate(need)}
        frames = vr.get_batch(need).asnumpy()                 # (N, H, W, 3) uint8, small
        gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

        flow, moving, diff1s, hist1s = [], [], [], []
        prev_gray = prev_hist = None
        for i, j in zip(slots, pairs):
            g0 = gray[pos[i]]
            if j != i:
                fl = cv2.calcOpticalFlowFarneback(gray[pos[i]], gray[pos[j]], None,
                                                  0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.linalg.norm(fl, axis=-1)
                flow.append(float(mag.mean() / WIDTH * src_fps * 100))
                moving.append(float((mag > 1.0).mean()))
            else:                                             # last source frame: no successor
                flow.append(flow[-1] if flow else 0.0)
                moving.append(moving[-1] if moving else 0.0)
            hsv = cv2.cvtColor(frames[pos[i]], cv2.COLOR_RGB2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_gray is None:
                diff1s.append(0.0)
                hist1s.append(0.0)
            else:
                diff1s.append(float(np.abs(g0.astype(np.int16) - prev_gray).mean()))
                hist1s.append(float(1 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)))
            prev_gray, prev_hist = g0.astype(np.int16), hist

        r = lambda xs, n=4: [round(x, n) for x in xs]
        return video_id, {'src_fps': src_fps, 'stride': stride, 'n_src': n_src,
                          'analysis_hw': [height, WIDTH], 'n_slots': len(slots),
                          'flow': r(flow, 3), 'moving': r(moving), 'diff1s': r(diff1s, 3),
                          'hist1s': r(hist1s)}, None
    except Exception as e:  # one bad file must not take the benchmark down
        return video_id, None, f'{type(e).__name__}: {e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anno', default='data/StreamingBench/real.json')
    p.add_argument('--out', default='data/StreamingBench/motion/real_flow_frames.json')
    p.add_argument('--sample_fps', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    p.add_argument('--limit', type=int, default=None, help='first N videos only (smoke test)')
    args = p.parse_args()

    anno = json.load(open(args.anno))[:args.limit]
    jobs = [(s['video_id'], s['video_path'], args.sample_fps) for s in anno]
    videos, errors = {}, {}
    with Pool(args.workers) as pool:
        for k, (vid, res, err) in enumerate(pool.imap_unordered(analyse, jobs), 1):
            if err:
                errors[vid] = err
                print(f'[{k}/{len(jobs)}] {vid}: FAILED {err}', flush=True)
            else:
                videos[vid] = res
                if k % 25 == 0 or k == len(jobs):
                    print(f'[{k}/{len(jobs)}] done', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = {'anno': args.anno, 'sample_fps': args.sample_fps, 'grid': 'stride (FrameStream default)',
            'analysis_width': WIDTH, 'flow_unit': '% of frame width per second (Farneback, '
            'slot frame -> next source frame)', 'n_videos': len(videos), 'errors': errors}
    json.dump({'meta': meta, 'videos': videos}, open(args.out, 'w'))
    print(f'wrote {args.out}: {len(videos)} videos, {len(errors)} failed')


if __name__ == '__main__':
    main()
