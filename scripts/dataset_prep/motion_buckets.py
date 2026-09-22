"""Motion labels from the raw signals scripts/dataset_prep/motion_flow.py wrote.

`--level video` (default) labels each video once, over all of its 1 fps frames: one row per
video, which is what a "these videos are static / gradual / abrupt / highly dynamic" split
needs. `--level question` instead labels each question over the frames the model had seen
when it was asked ([0, n_frames_seen)), which is finer -- a 4-minute video is rarely one
motion regime -- and lines up with the cumulative keep rate in results.csv.

Two independent axes, both pixel-level, so neither can be circular with the pruner's own
feature distance:

    flow  -- mean optical-flow magnitude (% of frame width per second): continuous motion.
    cuts  -- share of 1 s steps whose HSV-histogram distance exceeds --cut_threshold:
             shot changes, which flow alone cannot see.

`motion_class` crosses them, which is what separates "gradual" from "abrupt": a static
talking head cut between three cameras has near-zero flow and a high cut rate, while a
continuous pan has the opposite.

Usage:
    python scripts/dataset_prep/motion_buckets.py            # writes the CSV + prints the table
    python scripts/dataset_prep/motion_buckets.py --flow_edges 2 8 20 --cut_rate 0.08
"""

import json
import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', default='data/StreamingBench/motion/real_flow_frames.json')
    p.add_argument('--results', default='results/llava_ov_7b/streamingbench_real/64-1.0-rlt_ref0.5cosine/results.csv',
                   help='pruned run: supplies n_frames_seen per question, and token_keep_rate for the summary')
    p.add_argument('--level', choices=['video', 'question'], default='video')
    p.add_argument('--scheme', choices=['combined', 'flow'], default='combined',
                   help="combined: rank flow and cut rate together into four equal groups "
                        "(static -> gradual -> dynamic -> highly dynamic), so both kinds of "
                        "change count. flow: fixed --flow_edges on flow alone, cuts kept as a "
                        "separate 'abrupt' flag.")
    p.add_argument('--cut_weight', type=float, default=0.5,
                   help='--scheme combined: weight of the cut-rate rank in the score '
                        '(0 = flow only, 1 = cuts only).')
    p.add_argument('--out', default=None,
                   help='default: data/StreamingBench/motion/real_motion_<level>s.csv')
    p.add_argument('--flow_edges', type=float, nargs=3, default=[2.0, 8.0, 20.0],
                   help='%% frame width per second: static | gradual | dynamic | highly dynamic')
    p.add_argument('--cut_threshold', type=float, default=0.5,
                   help='1 - HSV histogram correlation above this counts the 1 s step as a cut')
    p.add_argument('--cut_rate', type=float, default=0.08,
                   help='cuts per frame above this makes the question "abrupt"')
    args = p.parse_args()
    out = args.out or f'data/StreamingBench/motion/real_motion_{args.level}s.csv'

    frames = json.load(open(args.frames))
    videos = frames['videos']
    res = pd.read_csv(args.results)

    lo, mid, hi = args.flow_edges

    def label(v, n, ids):
        """One row from the first `n` slots of video `v`, whatever `ids` identifies it by."""
        flow = np.asarray(v['flow'][:n])
        cuts = float((np.asarray(v['hist1s'][:n]) > args.cut_threshold).mean())
        f = float(flow.mean())
        band = ('static' if f < lo else 'gradual' if f < mid else 'dynamic' if f < hi
                else 'highly_dynamic')
        return {**ids,
                'flow_mean': round(f, 3), 'flow_median': round(float(np.median(flow)), 3),
                'flow_p90': round(float(np.percentile(flow, 90)), 3),
                'moving_frac': round(float(np.asarray(v['moving'][:n]).mean()), 4),
                'cut_rate': round(cuts, 4), 'flow_band': band,
                'abrupt': cuts > args.cut_rate,
                'motion_class': f'{band}+abrupt' if cuts > args.cut_rate else band}

    rows = []
    if args.level == 'video':
        for video_id, v in videos.items():
            rows.append(label(v, len(v['flow']),
                              {'video_id': video_id, 'n_frames': len(v['flow'])}))
    else:
        for q in res.itertuples():
            v = videos.get(q.video_id)
            if v is None:                 # video failed to decode in motion_flow.py
                continue
            n = min(int(q.n_frames_seen), len(v['flow']))
            if n < 2:                     # a question asked in the first second has no motion yet
                continue
            rows.append(label(v, n, {'question_id': q.question_id, 'video_id': q.video_id,
                                     'n_frames_seen': n}))

    d = pd.DataFrame(rows).sort_values('flow_mean', ascending=False)

    CLASSES = ['static', 'gradual', 'dynamic', 'highly_dynamic']
    if args.scheme == 'combined':
        # Percentile rank of each signal, so the two are on one scale despite different
        # units (% width per second vs cuts per frame), then four equal groups of the
        # weighted sum. Both flow and cut rate rise from one group to the next.
        fr = d.flow_mean.rank(pct=True)
        cr = d.cut_rate.rank(pct=True, method='average')
        d['motion_score'] = ((1 - args.cut_weight) * fr + args.cut_weight * cr).round(4)
        d['motion_class'] = pd.qcut(d.motion_score.rank(method='first'), 4, labels=CLASSES)
    else:
        d['motion_class'] = np.where(d.abrupt, d.flow_band + '+abrupt', d.flow_band)
    header = (f'# per-{args.level} motion labels; flow in %% of frame width per second\n'
              f'# flow_edges={args.flow_edges} cut_threshold={args.cut_threshold} '
              f'cut_rate={args.cut_rate}\n'
              f'# frames={args.frames} results={args.results}\n')
    with open(out, 'w') as fh:
        fh.write(header)
        d.to_csv(fh, index=False)
    print(f'wrote {out}: {len(d)} rows, {d.video_id.nunique()} videos')

    if 'token_keep_rate' not in res.columns:
        return
    if args.level == 'video':
        # One keep rate per video: its last question, which covers the longest stream the
        # model held for it. Accuracy is that video's questions pooled.
        last = res.sort_values('tokens_seen').groupby('video_id').tail(1)
        per_video = last.set_index('video_id').tokens_kept / last.set_index('video_id').tokens_seen
        j = d.merge(per_video.rename('token_keep_rate'), on='video_id')
        j = j.merge(res.groupby('video_id').qa_acc.mean().rename('qa_acc'), on='video_id')
        weight = res.groupby('video_id').size().rename('n_questions')
        j = j.merge(weight, on='video_id')
    else:
        j = d.merge(res[['question_id', 'token_keep_rate', 'qa_acc']], on='question_id')
    j['flow_band'] = pd.Categorical(j.flow_band, CLASSES, ordered=True)
    groupings = ([['motion_class']] if args.scheme == 'combined'
                 else [['flow_band'], ['flow_band', 'abrupt']])
    for keys in groupings:
        t = j.groupby(keys, observed=True).agg(
            n=('flow_mean', 'size'), flow=('flow_mean', 'mean'), cut_rate=('cut_rate', 'mean'),
            keep=('token_keep_rate', 'mean'), acc=('qa_acc', 'mean'))
        t['token_reduction'] = 1 - t.keep
        print(t.round(3).to_string(), '\n')


if __name__ == '__main__':
    main()
