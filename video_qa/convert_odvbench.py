"""Convert ODV-Bench's flat annotation file into ReKV's streaming video-level schema.

ODV-Bench (MCG-NJU/ODV-Bench on the Hub, from the StreamForest repo) is an *online* driving
benchmark: 6348 questions over 1190 dashcam clips, each question carrying a timestamp that
the model's view is supposed to stop at. It ships one record per question; ReKV's streaming
solver wants the transpose -- one record per video, questions sorted by timestamp -- so the
video is ingested once, incrementally, and questioned in between. Here that is a 5.3x
saving on encode cost, and it is also the only shape that can express the time limit.

**`[start, end]` is a visible-frame budget, not a grounding window.** `start` is 0 in all
6348 records, so the pair is always the prefix [0, end] -- the same contract as OVO-Bench's
`realtime`. `end` becomes `end_time`, and `rekv_ovobench_vqa.py` shows the model
floor(end_time * sample_fps) + 1 frames and no more.

That distinction is the whole benchmark, not a detail. `end` sits at a median of ~0.45 of
video duration, and 3928 of the 6348 questions (62%) are explicitly about what has not
happened yet -- "What *will* the position box of the pedestrian ... be", "What *will be*
the subsequent motion state", "Will there be significant traffic risks ... in the future".
Feeding a whole clip to an offline solver answers those from the frames they are asking the
model to predict. It scores well and measures nothing.

Three further facts about the source data, all verified against `ODVbench.json`:

* **`answer` is the option text and always appears verbatim in `candidates`** (6348/6348),
  with no duplicate options in any question, so `gt_index` is an unambiguous
  `candidates.index(answer)`. Unlike OVO-Bench, whose `answer` is often a paraphrase (see
  convert_ovobench.py) -- but the field is written out regardless, because the solver is
  shared and reads `gt_index`.
* **Option counts are 2 or 4**, well inside BaseVQA's eight `choice_letters`.
* **The `video` field's leading component is the zip's name** ('TS_Retrieval/test_video/
  x.mp4' for TS_Retrieval.zip, whose root is 'test_video/'), so extracting each zip into a
  directory named after it under `--video_root` reproduces those paths exactly. That is
  what scripts/dataset_prep/setup_odvbench.py does.

The 12 `subtask` values become `question_type` (the results CSV's `task` column, which
video_qa/eval/eval_odvbench.py aggregates over). The coarse 3-way `task` becomes
`task_group`; it is a pure function of the collection -- TS_Retrieval is exactly
'Targeting single static objects', TOI_Recognition 'single dynamic objects', TR_Analysis
'multi-object interaction scenarios or events' -- but carrying it explicitly beats
re-deriving it from a path in the scorer.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_odvbench.py \
        --src data/odvbench/ODVbench.json \
        --video_root data/odvbench \
        --out data/odvbench/full_mc.json
"""

import os
import json
import argparse
from collections import Counter, OrderedDict


def video_id_of(rel_path):
    """Filesystem-safe, collision-free id from ODV-Bench's relative video path.

    Basenames are not unique across the collections -- TR_Analysis names its clips by
    number under per-class directories ('CAP-DATA/10/009806.mp4'), and the same number
    recurs under other classes. The id also names the video in results.csv and has to
    survive being a single path component, so flatten the whole relative path.
    """
    stem = os.path.splitext(rel_path)[0]
    return stem.replace('/', '_').replace('\\', '_')


def probe_duration(video_path):
    """Video length in seconds, or None if it cannot be read.

    Metadata only -- nothing in the eval path reads `duration`; the stream is capped by
    the last question's `end_time`, not by this. It is reported here because the ratio of
    end_time to duration is what tells you how much of each clip the time limit is
    actually withholding.
    """
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        return round(len(vr) / fps, 2) if fps else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/odvbench/ODVbench.json',
                        help='ODVbench.json as shipped on the Hub.')
    parser.add_argument('--video_root', type=str, default='data/odvbench',
                        help="Directory the three zips were extracted into; the `video` "
                             "field is resolved relative to it.")
    parser.add_argument('--out', type=str, default='data/odvbench/full_mc.json')
    parser.add_argument('--no_probe', action='store_true',
                        help='Skip the decord duration probe (fast, leaves duration null).')
    parser.add_argument('--skip_missing', action='store_true',
                        help='Drop videos absent from --video_root instead of failing. Use '
                             'to evaluate on one collection without extracting the others.')
    args = parser.parse_args()

    src = json.load(open(args.src))

    # Insertion-ordered so the output follows the source file's order: run_eval splits the
    # annotation into contiguous per-GPU chunks, and a stable order keeps a resumed or
    # re-sharded run comparable with the one before it.
    by_video = OrderedDict()
    for qid, item in enumerate(src):
        # The source has no question id; its position in the file is the only stable one,
        # and it has to be assigned before the per-video sort below reorders anything.
        by_video.setdefault(item['video'], []).append((qid, item))

    out, missing, unprobed = [], [], 0
    for i, (rel_path, items) in enumerate(by_video.items()):
        video_path = os.path.join(args.video_root, rel_path)
        if not os.path.exists(video_path):
            missing.append(rel_path)
            continue

        duration = None if args.no_probe else probe_duration(video_path)
        if duration is None and not args.no_probe:
            unprobed += 1

        conversations = []
        # Sorted by timestamp because the solver ingests forward-only and never rewinds:
        # an out-of-order question would be answered against frames past its own limit.
        # rekv_ovobench_vqa asserts this rather than trusting it; sorting here is what
        # makes the assert hold. Ties keep source order via the qid secondary key.
        for qid, item in sorted(items, key=lambda p: (p[1]['end'], p[0])):
            assert item['start'] == 0, \
                (f"{rel_path}: start={item['start']}, expected 0 -- [start, end] is "
                 f"assumed to be the visible prefix [0, end]. Re-check the schema.")
            assert item['answer'] in item['candidates'], \
                f"{rel_path}: answer not among candidates -- {item['answer']!r}"
            assert len(set(item['candidates'])) == len(item['candidates']), \
                f'{rel_path}: duplicate options make gt_index by text ambiguous'
            conversations.append({
                'question_id': qid,
                'question': item['question'],
                'choices': item['candidates'],
                'answer': item['answer'],
                'gt_index': item['candidates'].index(item['answer']),
                'question_type': item['subtask'],
                'task_group': item['task'],
                # The time limit. start_time is carried for symmetry with OVO-Bench's
                # schema; it is 0 for every ODV-Bench question.
                'start_time': item['start'],
                'end_time': item['end'],
            })

        out.append({
            'video_id': video_id_of(rel_path),
            'video_path': video_path,
            'source_video': rel_path,
            'duration': duration,
            'conversations': conversations,
        })
        if not args.no_probe and (i + 1) % 100 == 0:
            print(f'probed {i + 1}/{len(by_video)}', flush=True)

    if missing:
        head = '\n  '.join(missing[:10])
        msg = (f'{len(missing)} of {len(by_video)} videos not found under '
               f'{args.video_root}:\n  {head}')
        if not args.skip_missing:
            raise SystemExit(f'{msg}\n\nRun scripts/dataset_prep/setup_odvbench.py first, or pass '
                             f'--skip_missing to evaluate on what is present.')
        print(f'WARNING: {msg}')

    ids = Counter(v['video_id'] for v in out)
    dupes = [k for k, n in ids.items() if n > 1]
    assert not dupes, f'video_id collisions: {dupes[:5]}'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)

    n_q = sum(len(v['conversations']) for v in out)
    print(f'\nwrote {args.out}: {len(out)} videos, {n_q} questions '
          f'({n_q / max(len(out), 1):.1f} per video)')
    if unprobed:
        print(f'{unprobed} videos have duration=null (decord could not read them)')

    # How much of each clip the time limit withholds. If this were ~1.0 the streaming
    # constraint would be vacuous and the offline solver would be equivalent; it is not.
    ratios = sorted(c['end_time'] / v['duration'] for v in out if v['duration']
                    for c in v['conversations'])
    if ratios:
        print(f'end_time / duration: p10={ratios[len(ratios) // 10]:.2f} '
              f'p50={ratios[len(ratios) // 2]:.2f} '
              f'p90={ratios[int(0.9 * len(ratios))]:.2f}  '
              f'(1.0 would mean the whole clip is visible)')
    print('collections: ' + ', '.join(
        f'{k}={n}' for k, n in Counter(v['source_video'].split('/')[0]
                                       for v in out).most_common()))
    print('subtasks: ' + ', '.join(
        f'{k}={n}' for k, n in Counter(c['question_type'] for v in out
                                       for c in v['conversations']).most_common()))


if __name__ == '__main__':
    main()
