"""Decode every video an annotation file references and report the ones that fail.

Written after `Real-Time Visual Understanding/sample_332` -- corrupt in the upstream
StreamingBench release, ~59% of its access units undecodable -- killed one worker of every
sighted StreamingBench arm 81 videos in, silently halving seven runs before anyone noticed.
`BaseVQA.analyze` now skips such files instead of aborting, but knowing which files they
are *before* committing a GPU to a 5-hour sweep is worth the CPU hour this costs.

The check is deliberately the same decode the evaluation performs -- decord, at the run's
`sample_fps`, through `open_video_reader` -- rather than a container probe. `ffprobe` reads
sample_332's header quite happily; only decoding its frames reveals it is broken.

    python scripts/scan_corrupt_videos.py --anno data/StreamingBench/real.json
    python scripts/scan_corrupt_videos.py --anno data/OVBench/full_mc.json --workers 8

Exit status is 1 when any video fails, so this can gate a sweep in a batch script.
"""
import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_one(task):
    """Decode one video on the run's frame grid. Returns (video_id, path, error-or-None)."""
    video_id, path, sample_fps = task
    # Imported in the worker: decord holds per-process FFmpeg state, and importing it in
    # the parent before fork has been a source of hangs.
    from decord import VideoReader, cpu
    from video_qa.base import decord_num_threads

    if not os.path.exists(path):
        return video_id, path, 'missing file'
    try:
        if path.endswith('.npy'):
            import numpy as np
            arr = np.load(path, mmap_mode='r')
            _ = np.asarray(arr[:: max(1, len(arr) // 64)])
            return video_id, path, None

        vr = VideoReader(path, ctx=cpu(0), num_threads=decord_num_threads())
        n = len(vr)
        fps = vr.get_avg_fps()
        if not n or not fps or fps != fps:  # nan
            return video_id, path, f'unusable header (frames={n}, fps={fps})'
        stride = max(1, int(round(fps / sample_fps)))
        idx = list(range(0, n, stride))
        # Batched the way FrameStream reads, so a file that only fails under decord's
        # threaded batch path -- which is how sample_332 fails -- is caught here too.
        for i in range(0, len(idx), 32):
            vr.get_batch(idx[i:i + 32]).asnumpy()
        return video_id, path, None
    except Exception as e:
        return video_id, path, f'{type(e).__name__}: {str(e).splitlines()[0][:200]}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--anno', required=True, help='Annotation JSON, e.g. data/StreamingBench/real.json')
    p.add_argument('--sample_fps', type=float, default=1.0,
                   help='Frame grid to test on; match the sweep you are gating.')
    p.add_argument('--workers', type=int, default=min(16, mp.cpu_count()))
    p.add_argument('--out', default=None, help='Report JSON (default: alongside --anno)')
    args = p.parse_args()

    with open(args.anno) as f:
        anno = json.load(f)

    seen, tasks = set(), []
    for s in anno:
        path = s.get('video_path') or s.get('video')
        if path in seen:
            continue
        seen.add(path)
        tasks.append((s['video_id'], path, args.sample_fps))

    out = args.out or os.path.splitext(args.anno)[0] + '.videoscan.json'
    print(f'scanning {len(tasks)} videos from {args.anno} at {args.sample_fps} fps '
          f'on {args.workers} workers', flush=True)

    t0 = time.time()
    bad = []
    # maxtasksperchild=1: a worker that survives a damaged file can leave FFmpeg state
    # behind that makes the *next* file look broken. A fresh process per video costs
    # milliseconds against a decode measured in seconds.
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for i, (video_id, path, err) in enumerate(pool.imap_unordered(check_one, tasks), 1):
            if err:
                bad.append({'video_id': video_id, 'video_path': path, 'error': err})
                print(f'  BAD  {video_id}: {err}', flush=True)
            if i % 25 == 0 or i == len(tasks):
                print(f'  {i}/{len(tasks)}  ({len(bad)} bad, {time.time() - t0:.0f}s)', flush=True)

    with open(out, 'w') as f:
        json.dump({'anno': args.anno, 'sample_fps': args.sample_fps,
                   'n_videos': len(tasks), 'n_bad': len(bad), 'bad': bad}, f, indent=2)

    print(f'\n{len(tasks) - len(bad)}/{len(tasks)} videos decoded cleanly in '
          f'{time.time() - t0:.0f}s; report: {out}')
    if bad:
        print(f'\n{len(bad)} UNUSABLE VIDEO(S):')
        for b in bad:
            print(f'  {b["video_id"]}  --  {b["error"]}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
