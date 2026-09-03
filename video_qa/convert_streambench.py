"""Convert StreamBench's annotation into ReKV's video-level schema.

StreamBench (`streaming_bench_v0.3.json`) is 307 entries, each naming a video and holding
a `breakpoint` list: questions with a `time` past which the model may not see. Unlike
OVBench and like OVO-Bench the answers are **free-form sentences**, not letters, so this
is scored by an LLM judge (video_qa/eval/eval_open_ended_local.py) rather than by string
match -- see video_qa/run_eval.py:eval_streambench.

Four things about the source this encodes, each checked against the file:

* **32 videos appear under two entries**, so 307 entries cover only 275 distinct videos.
  Each entry carries its own 6 questions against the same footage. They are merged into
  one record per video, because the streaming solver ingests a record once and questions
  it N times: leaving them split would encode those 32 videos twice for no benefit, and
  the second pass would restart the KV-Cache from an empty stream rather than continuing
  the first. Merging interleaves the two question sets by timestamp, which is what a
  single viewing of the video would actually produce.
* **`video_name` is missing from 107 of the 307 entries**, so the id is built from
  `class_1` + `video_path` instead. Those two are present on all 307 and are what locates
  the file.
* **One video's `breakpoint` list is not in timestamp order.** The sort below is a
  correctness requirement, not tidiness -- the solver's ingestion cursor only moves
  forward, so an out-of-order question is answered against frames from beyond its own
  `time`, which inflates accuracy while looking entirely normal.
* **`class` is the question axis worth reporting** -- CI, KG, LM, OS, SF, SM, near-evenly
  balanced at ~300 each. KG in particular is world knowledge with no visual content
  ("From which ingredient is sesame oil extracted?"), so it is answerable blind and will
  sit far above the others in any run. It is carried through to results.csv so
  video_qa/eval/eval_streambench.py can separate it rather than averaging it in.

`class_1` (Ego / WebVideo / Movie) is carried as `source` for the same reason: the three
differ in length and in what the questions can ask, so a pooled figure hides them.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_streambench.py \
        --src data/streambench/streaming_bench_v0.3.json \
        --video_root data/streambench \
        --out data/streambench/full_oe.json
"""

import os
import json
import argparse
from collections import Counter, defaultdict


def video_id_of(class_1, video_path):
    """Filesystem-safe id. Names the video in results.csv, so it must be one field.

    Built from class_1 + the file stem rather than `video_name`, which 107 entries lack.
    The prefix matters: the three collections are separate directories and nothing
    guarantees a stem is unique across them.
    """
    return f'{class_1}__{os.path.splitext(video_path)[0]}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='data/streambench/streaming_bench_v0.3.json')
    parser.add_argument('--video_root', type=str, default='data/streambench',
                        help='Directory holding the Ego/, WebVideo/ and Movie/ folders.')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose video file is absent instead of dropping '
                             'them. For inspecting the conversion before the videos are '
                             'staged; an eval run over these would crash on load.')
    args = parser.parse_args()

    records = json.load(open(args.src))

    # Group by the file, not by the entry -- see the docstring on the 32 duplicates.
    by_video = defaultdict(list)
    for entry in records:
        info = entry['info']
        by_video[(info['class_1'], info['video_path'])].append(entry)

    anno, missing = [], []
    n_questions = 0
    classes, sources, topics = Counter(), Counter(), Counter()
    n_merged = 0
    for (class_1, video_path), group in sorted(by_video.items()):
        rel = os.path.join(class_1, video_path)
        full = os.path.join(args.video_root, rel)
        if not os.path.exists(full):
            missing.append(rel)
            if not args.allow_missing:
                continue
        if len(group) > 1:
            n_merged += 1

        vid = video_id_of(class_1, video_path)
        conversations = []
        for e_i, entry in enumerate(group):
            topic = entry['info'].get('class_2')
            for b_i, b in enumerate(entry['breakpoint']):
                conversations.append({
                    # Entry index is part of the id so the two merged question sets stay
                    # distinguishable after the sort below.
                    'question_id': f'{vid}#e{e_i}q{b_i}',
                    'question': b['question'],
                    # Free-form reference. No `choices`/`gt_index` here: an LLM judge
                    # decides whether pred_answer means the same thing as this.
                    'answer': b['answer'],
                    'question_type': b['class'],
                    'source': class_1,
                    'topic': topic,
                    'start_time': 0,
                    'end_time': float(b['time']),
                })
                classes[b['class']] += 1
                topics[topic] += 1

        # Ascending timestamp is a correctness requirement -- the ingestion cursor never
        # rewinds, so a question out of order would see future frames. question_id breaks
        # ties so the conversion is deterministic.
        conversations.sort(key=lambda c: (c['end_time'], c['question_id']))

        n_questions += len(conversations)
        sources[class_1] += 1
        anno.append({
            'video_id': vid,
            'video_path': full,
            'source_video': rel,
            'source': class_1,
            'conversations': conversations,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(anno, f, indent=1)

    print(f'{len(records)} entries -> {len(anno)} videos '
          f'({n_merged} had two entries merged), {n_questions} questions -> {args.out}')
    print('\nper source:')
    for s, n in sorted(sources.items()):
        print(f'  {s:<12} {n:4d} videos')
    print('\nper question class:')
    for c, n in sorted(classes.items()):
        print(f'  {c:<12} {n:5d}')
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'DROPPED'
        print(f'\n{len(missing)} videos not found under {args.video_root} and {verb}; '
              f'e.g. {missing[:3]}')


if __name__ == '__main__':
    main()
