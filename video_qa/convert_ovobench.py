"""Convert OVO-Bench's flat annotation file into ReKV's video-level schema.

OVO-Bench ships one record per query (`data/ovo_bench_new.json`), each naming a video and
a `realtime` timestamp: the model may only see frames in [0, realtime]. ReKV's streaming
solver wants the transpose of that -- one record per *video*, carrying every question
asked against it, in timestamp order -- so a video is ingested once and questioned N
times instead of being re-encoded per query. That is not a micro-optimisation here: the
realtime split is 837 queries over 237 videos (193 of them carry more than one query, up
to 15), so the grouping removes ~3.5x of the encode cost.

Two facts about the source data that this script encodes, both verified against
`ovo_bench_new.json` (1468 backward+realtime records):

* **`gt` is authoritative, not `answer`.** In 420 records `options[gt] != answer`, almost
  always because `answer` is the original free-form reference and the option is a
  paraphrase of it ("...going to drill a hole into the wires in front of them" vs "...going
  to drill a hole"). Only 8 records have the `answer` string sitting at a different index
  than `gt`. OVO-Bench's own scorer uses `chr(65 + gt)` and ignores `answer`, so we key on
  `gt` too. This matters because ReKV's existing offline solver derives the correct letter
  with `choices.index(answer)` (video_qa/rekv_offline_vqa.py) -- that convention would
  mislabel ~29% of OVO-Bench. `gt_index` is written out so the solver never has to guess.
* **Option counts vary (2-5) and 3 records contain duplicate option strings**, which is a
  second reason index-by-text is unusable.

FAR tasks (REC/SSR/CRR) are deliberately not converted: they need their own prompt
templates and Y/N-or-count scoring rather than multiple choice. `--modes` rejects them
explicitly rather than emitting something that would score as garbage.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_ovobench.py \
        --src data/ovo_bench/ovo_bench_new.json \
        --video_root data/ovo_bench/src_videos \
        --modes realtime \
        --out data/ovo_bench/realtime.json
"""

import os
import json
import argparse
from collections import Counter, defaultdict

# Same partition as OVO-Bench's constant.py.
BACKWARD_TASKS = ['EPM', 'ASI', 'HLD']
REALTIME_TASKS = ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']
FORWARD_TASKS = ['REC', 'SSR', 'CRR']

MODE_TASKS = {
    'realtime': REALTIME_TASKS,
    'backward': BACKWARD_TASKS,
}


def video_id_of(rel_path):
    """Filesystem-safe id from OVO's relative video path.

    Paths are nested ('Ego4D/clips/xxx.mp4', 'YouTube_Games/PLJ3...&index=1.mp4') and the
    basename alone is not unique across the source collections. It also names the video in
    results.csv, so it has to survive being a single path component.
    """
    stem = os.path.splitext(rel_path)[0]
    return stem.replace('/', '__').replace('&', '_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/ovo_bench/ovo_bench_new.json')
    parser.add_argument('--video_root', type=str, default='data/ovo_bench/src_videos',
                        help='Directory holding the unpacked src_videos tree '
                             '(AutoEvalMetaData/, Ego4D/, OpenEQA/, ...).')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--modes', type=str, default='realtime',
                        help="Comma-separated: realtime, backward. 'forward' is not "
                             'supported -- FAR tasks are not multiple choice.')
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose video file is absent instead of dropping '
                             'them. For inspecting the conversion before the videos are '
                             'staged; an eval run over these would crash on load.')
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for mode in modes:
        if mode not in MODE_TASKS:
            raise SystemExit(
                f"unsupported mode {mode!r}. Choose from {sorted(MODE_TASKS)}. "
                f"FAR tasks ({', '.join(FORWARD_TASKS)}) are answered with Yes/No or a "
                f"count and scored per-probe, so they need their own solver, not this "
                f"multiple-choice conversion.")
    keep_tasks = {t for mode in modes for t in MODE_TASKS[mode]}

    records = json.load(open(args.src))
    selected = [r for r in records if r['task'] in keep_tasks]

    by_video = defaultdict(list)
    for r in selected:
        by_video[r['video']].append(r)

    anno, missing, n_queries = [], [], 0
    for rel_path, group in sorted(by_video.items()):
        video_path = os.path.join(args.video_root, rel_path)
        if not os.path.exists(video_path):
            missing.append(rel_path)
            if not args.allow_missing:
                continue

        # Ascending timestamp is a correctness requirement, not a nicety: the streaming
        # solver only ever moves its ingestion cursor forward, so a question placed out of
        # order would be answered against frames from beyond its own `realtime` -- a
        # future-information leak that inflates accuracy while looking perfectly normal.
        # `id` breaks ties so the conversion is deterministic.
        group = sorted(group, key=lambda r: (float(r['realtime']), r['id']))

        conversations = []
        for r in group:
            gt = int(r['gt'])
            options = list(r['options'])
            assert 0 <= gt < len(options), f"id {r['id']}: gt {gt} out of range"
            conversations.append({
                'question_id': r['id'],
                'question': r['question'],
                'choices': options,
                # Set from gt, not from r['answer'] -- see the module docstring. Keeping
                # them consistent means any code path that recovers the letter by text
                # lookup agrees with the one that uses gt_index.
                'answer': options[gt],
                'gt_index': gt,
                'reference_answer': r.get('answer'),
                'question_type': r['task'],
                'realtime': float(r['realtime']),
                # ReKV's streaming solver gates ingestion on end_time alone; start_time is
                # carried only to match the schema of the other datasets. Backward-tracing
                # questions use the same gate as realtime ones -- 'backward' describes what
                # the question asks about, not what the model is allowed to see.
                'start_time': 0,
                'end_time': float(r['realtime']),
            })
        n_queries += len(conversations)
        anno.append({
            'video_id': video_id_of(rel_path),
            'video_path': video_path,
            'source_video': rel_path,
            'conversations': conversations,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(anno, f, indent=1)

    print(f'modes: {modes} -> tasks {sorted(keep_tasks)}')
    print(f'{len(selected)} queries in source, {n_queries} written '
          f'over {len(anno)} videos -> {args.out}')
    print('per-task:', dict(sorted(Counter(
        c['question_type'] for v in anno for c in v['conversations']).items())))
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'DROPPED'
        print(f'{len(missing)} videos not found under {args.video_root} and {verb}; '
              f'e.g. {missing[:3]}')


if __name__ == '__main__':
    main()
