"""Convert FPS-Bench-Stream's release JSONL into ReKV's video-level annotation schema.

FPS-Bench-Stream is FPSBench's needle-in-a-haystack build: each of the 996 FPSBench
questions keeps its clip (the *needle*, median 9 s) but that clip is spliced into a
600 s MLVU video (the *haystack*), so the evidence is 1.5% of the stream and sits at a
known timestamp. See /home/av354855/FPSBenchStream/README.md; this script reads the
canonical `fpsbench_stream_v1.jsonl` and the assembled `videos/` directory it describes.

Why this conversion is not `video_qa/convert_fpsbench.py` with a different --src:

* **Three clocks.** The release carries the same event in YouTube seconds (`time.*`), MLVU
  seconds (`stream.haystack.window_start_sec`) and assembled-stream seconds
  (`stream.timeline.*`). Only the third means anything to a solver reading the file it was
  handed, so this script emits `stream.timeline` and drops the other two -- mixing them is
  the one mistake the README calls out by name.
* **There is an answer key.** Unlike `fpsbench_questiononly.csv`, every record here carries
  `question.answer`, so `qa_acc` in results.csv is a real number and the run ends at a
  scorer (video_qa/eval/eval_fpsbench_stream.py) rather than at a submission file.
* **The needle window is the retrieval target.** `needle_start_sec`/`needle_end_sec` are
  written through so the solver can turn them into frame indices and ask what ReKV actually
  retrieved when the question came (video_qa/rekv_fpsbench_stream_vqa.py). That is the
  measurement this dataset exists for and the short-clip arm cannot make: there, the
  evidence is the whole video.

Choice order is A..E as the release letters them, so the letter ReKV predicts is the
letter FPSBench assigned -- 'E' is always "None of the above". `answer` is written as the
option *text* because that is what the solvers index on (`choices.index(answer)`), and
`answer_letter` is carried alongside for readers.

Six of the 996 streams were never built (their needle clip was unavailable at download
time; `fpsbench_stream_v1_stats.json` lists the ids). They are dropped with a warning --
an eval run over them would crash in decord.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_fpsbench_stream.py \
        --src /home/av354855/FPSBenchStream/fpsbench_stream_v1.jsonl \
        --video_root /home/av354855/FPSBenchStream/videos \
        --out data/fpsbench_stream/test_mc.json
"""

import os
import json
import argparse
from collections import Counter

# The release letters its options; ANSWER_CHOICES in video_qa/fpsbench_prompt.py is the
# same list, but importing it here would pull the prompt module into a pure-JSON script.
ANSWER_LETTERS = ['A', 'B', 'C', 'D', 'E']


def ordered_choices(question):
    """Option texts in letter order, and the letter of the correct one.

    The release stores choices as a {letter: text} object, whose key order is whatever the
    JSON parser produced. Sorting by ANSWER_LETTERS rather than trusting that order is what
    keeps the emitted list positional -- the solvers letter options by list position, so a
    reordering here would silently relabel every answer.
    """
    choices = question['choices']
    letters = [l for l in ANSWER_LETTERS if l in choices]
    texts = [str(choices[l]) for l in letters]
    answer_letter = str(question['answer']).strip().upper()
    return letters, texts, answer_letter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='/home/av354855/FPSBenchStream/fpsbench_stream_v1.jsonl')
    parser.add_argument('--video_root', type=str,
                        default='/home/av354855/FPSBenchStream/videos')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--limit', type=int, default=0,
                        help='Keep only the first N records that survive every other '
                             'filter. For a smoke run: one arm over all 990 streams is '
                             '~165 hours of footage, so a shape check wants a subset.')
    parser.add_argument('--position_bin', type=str, default=None,
                        choices=['early', 'middle', 'late'],
                        help='Keep only needles in this third of the stream. The release '
                             'stratifies 333/331/332, so a single bin is a third of the '
                             'benchmark with retrieval distance held roughly fixed.')
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose assembled stream is absent instead of '
                             'dropping them. For inspecting the conversion before the '
                             'videos are staged; an eval run over these would crash on load.')
    args = parser.parse_args()

    records, missing, unbuilt = [], [], []
    tasks, bins = Counter(), Counter()
    with open(args.src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stream = rec['stream']
            timeline = stream['timeline']
            stream_id = stream['stream_id']

            # `built: false` is the release saying the needle never downloaded, which is a
            # different failure from "the file is not staged here" -- keep them apart in
            # the report so a staging mistake cannot hide behind the six known gaps.
            if stream.get('built') is False:
                unbuilt.append(rec['id'])
                if not args.allow_missing:
                    continue

            video_path = os.path.join(args.video_root, f'{stream_id}.mp4')
            if not os.path.exists(video_path):
                missing.append(stream_id)
                if not args.allow_missing:
                    continue

            if args.position_bin and stream['insertion']['position_bin'] != args.position_bin:
                continue

            letters, texts, answer_letter = ordered_choices(rec['question'])
            if answer_letter not in letters:
                raise ValueError(f'{rec["id"]}: answer {answer_letter!r} is not one of '
                                 f'the presented options {letters}')

            duration = float(stream['target_duration_sec'])
            tasks[rec['question']['type']] += 1
            bins[stream['insertion']['position_bin']] += 1
            records.append({
                'video_id': stream_id,
                'video_path': video_path,
                'conversations': [{
                    'question': str(rec['question']['text']),
                    'choices': texts,
                    # The option text, not the letter: the solvers recover the letter with
                    # choices.index(answer).
                    'answer': texts[letters.index(answer_letter)],
                    'answer_letter': answer_letter,
                    'question_type': str(rec['question']['type']),
                    # The FPSBench question id, so a row here joins to the same question in
                    # the short-clip arm (results/*/fpsbench_stream_small/).
                    'question_id': rec['id'],
                    # FPSBench's claim about the needle: below this rate the certificate
                    # window is not resolvable at all, whatever retrieval does.
                    'min_fps': float(rec['temporal_requirements']['min_fps']),
                    # --- assembled-stream coordinates; nothing else is in this file ----
                    'clip_duration_sec': duration,
                    'needle_start_sec': float(timeline['needle_start_sec']),
                    'needle_end_sec': float(timeline['needle_end_sec']),
                    'cert_start_sec': float(timeline.get('certificate_start_sec',
                                                         timeline['needle_start_sec'])),
                    'cert_end_sec': float(timeline.get('certificate_end_sec',
                                                       timeline['query_time_sec'])),
                    # When a real-time protocol would ask (= certificate end). The default
                    # protocol here asks at the end of the stream instead; `end_time` is
                    # what the solver's --trigger query arm fires on.
                    'start_time': float(timeline['needle_start_sec']),
                    'end_time': float(timeline['query_time_sec']),
                    'query_time_sec': float(timeline['query_time_sec']),
                    'position_bin': stream['insertion']['position_bin'],
                    'position_frac': float(stream['insertion'].get('position_frac', 0.0)),
                    # The canvas the stream was normalised to. Recorded because it is the
                    # ceiling on --sample_fps: above it the exact grid only repeats frames.
                    'stream_fps': float(stream['normalization']['fps']),
                    'haystack_file': stream['haystack']['file'],
                }],
            })
            if args.limit and len(records) >= args.limit:
                break

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(records, f, indent=1)

    print(f'wrote {len(records)} records -> {args.out}')
    print(f'tasks: {dict(sorted(tasks.items()))}')
    print(f'needle position: {dict(sorted(bins.items()))}')
    if records:
        gaps = [r['conversations'][0]['clip_duration_sec']
                - r['conversations'][0]['needle_end_sec'] for r in records]
        gaps.sort()
        mid = gaps[len(gaps) // 2]
        # The distance the question has to reach back over when it is asked at the end of
        # the stream. It is the independent variable of this benchmark: at 1 fps a median
        # gap of ~300 s is ~300 frames of haystack between the evidence and the question,
        # all of it past n_local and reachable only through retrieval.
        print(f'seconds between the needle ending and the end of the stream: '
              f'min {gaps[0]:.0f} / median {mid:.0f} / max {gaps[-1]:.0f}')
        fps = sorted({r['conversations'][0]['stream_fps'] for r in records})
        print(f'stream canvas fps present: {[int(x) for x in fps]} '
              f'-- --sample_fps above the lowest of these repeats frames on those streams')
    if unbuilt:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'dropped'
        print(f'{len(unbuilt)} streams the release never built, {verb}: {sorted(unbuilt)}')
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'dropped'
        print(f'{len(missing)} streams absent from {args.video_root}, {verb}: '
              f'{sorted(missing)[:10]}{" ..." if len(missing) > 10 else ""}')


if __name__ == '__main__':
    main()
