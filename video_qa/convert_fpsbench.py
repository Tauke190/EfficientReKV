"""Convert FPSBench's flat question CSV into ReKV's video-level annotation schema.

FPSBench ships one CSV row per question (`fpsbench_questiononly.csv`, 1000 rows) plus a
directory of pre-cut clips named `<id>_<start>-<end>.mp4`. ReKV's solvers want one record
per *video*, carrying every question asked against it -- see
`video_qa/convert_ovobench.py` for the same transpose against OVO-Bench. Here the mapping
is 1:1 (ids are unique and each clip carries exactly one question), so the grouping is
trivial; what this script actually does is resolve ids to clip paths and normalise the
five choice columns into the `choices` list the solvers read.

Two facts about the released files that this script encodes:

* **The answer key is optional, and which CSV you point at decides whether you have
  one.** `fpsbench_questiononly.csv` (the release) has no `answer` column, so every record
  is written with `"answer": null`; the solver substitutes `choices[0]`, and the
  `qa_acc`/`correct_choice` columns in results.csv are then meaningless -- such a run ends
  at `video_qa/eval/export_fpsbench.py` rather than at a scorer. The full annotation CSV
  (`FPSBench/annotations/fpsbench_v1.csv`) does have `answer`/`answer_text`, and pointing
  --src at it writes the real key through, which is what local scoring -- and MBA
  (`video_qa/eval/eval_fpsbench_mba.py`) -- needs.

  Keeping the key out of the default path is deliberate, not an oversight: the prompt
  builder never sees the annotation, so a keyed file cannot leak into what the model is
  shown, and `data/*` is gitignored so it cannot leak into the repo either.
* **4 of the 1000 ids have no clip** (fpsbench_000148, 000184, 000659, 000660 in the
  release checked here -- videos that were unavailable at download time). They are dropped
  with a warning rather than emitted, since an eval run over them would crash in decord.
  Pass --allow_missing to keep them for inspection.
* **The temporal certificate is stated in source-video seconds**, like the clip range, so
  it has to be shifted into clip coordinates before a solver can use it. That is what
  `relative_certificate` below does, and it is why this conversion is a prerequisite for
  the streaming run: `end_time` is the timestamp at which
  video_qa/rekv_fpsbench_stream_small_vqa.py fires each question. An annotation file produced
  before this field existed will not work.

Choice order is the CSV column order (choice_a..choice_e), so the letter ReKV predicts is
the letter of the corresponding column -- 'E' is always "None of the above".

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    # question-only, no local scoring
    python video_qa/convert_fpsbench.py \
        --src fpsbench_questiononly.csv \
        --video_root /home/av354855/.cache/fpsbench/clips/clip \
        --out data/fpsbench/test_mc.json

    # keyed, for local scoring and MBA
    python video_qa/convert_fpsbench.py \
        --src /home/av354855/FPSBench/annotations/fpsbench_v1.csv \
        --video_root /home/av354855/.cache/fpsbench/clips/clip \
        --out data/fpsbench/test_mc_keyed.json
"""

import os
import re
import json
import glob
import math
import argparse
from collections import Counter

CHOICE_COLUMNS = ['choice_a', 'choice_b', 'choice_c', 'choice_d', 'choice_e']
# Positional: choice_a is 'A'. Same convention as fpsbench_prompt.ANSWER_CHOICES, kept
# separate because this module must not import a solver-side dependency.
ANSWER_LETTERS = ['A', 'B', 'C', 'D', 'E']

# `<id>_<clip_start>-<clip_end>.mp4`. The trailing range repeats clip_start_sec /
# clip_end_sec from the CSV, so the id alone is enough to key on -- and has to be, since
# the range in the filename is not always formatted the way the CSV columns are.
CLIP_RE = re.compile(r'^(fpsbench_\d+)_')


def relative_certificate(row):
    """Clip-relative temporal certificate, clamped to the clip, plus a "was clamped" flag.

    FPSBench states both the clip range and the certificate in *source video* seconds
    (clip 100-106, certificate 100-104), while the released clip file starts at 0. So the
    certificate has to be shifted by clip_start_sec before it means anything to a solver
    -- forgetting that would fire a question at t=100 s in a 6-second file, i.e. always at
    the end, which is exactly the offline behaviour this is meant to replace.

    Five of the 1000 rows do not survive the shift as-is (checked against the release
    here): 000159, 000453, 000700 and 000793 overshoot the clip end by 1-2 s, and 000995
    states a certificate of 0:02-0:05 for a clip covering 0:20-0:34, which is not a
    coordinate offset but a wrong annotation. Both are clamped into [0, duration] and the
    row is flagged `cert_clamped` so the ~0.5% affected can be excluded from any analysis
    that reads the trigger time as meaningful. A row clamped to a zero-length window falls
    back to the whole clip, which is the offline condition -- the safe direction, since it
    gives the model more to see rather than less.
    """
    duration = float(row.clip_duration_sec)
    start = float(row.temporal_certificate_start_sec) - float(row.clip_start_sec)
    end = float(row.temporal_certificate_end_sec) - float(row.clip_start_sec)
    clamped = not (0.0 <= start <= end <= duration)
    start = min(max(start, 0.0), duration)
    end = min(max(end, start), duration)
    if end <= 0.0:
        start, end = 0.0, duration
    return start, end, clamped


def index_clips(video_root):
    """Map fpsbench id -> clip path, erroring on any id that appears twice."""
    index = {}
    for path in sorted(glob.glob(os.path.join(video_root, '*.mp4'))):
        match = CLIP_RE.match(os.path.basename(path))
        if match is None:
            continue
        vid = match.group(1)
        if vid in index:
            raise ValueError(f'two clips claim id {vid}: {index[vid]} and {path}')
        index[vid] = path
    return index


def _missing(value):
    """True for a CSV cell pandas turned into NaN or None (i.e. an absent key)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def resolve_answer(row, choices):
    """The correct choice *text*, or None when the source CSV carries no key.

    Resolved through the `answer` letter rather than by matching `answer_text` against the
    choice columns. pandas normalises a numeric choice depending on what else shares its
    column -- "3" can arrive as 3, 3.0 or '3' -- so a string match between two columns is
    not reliably an identity even when the annotation is perfectly consistent. The letter
    is an index and cannot drift that way.

    `answer_text` is still read when present and checked against the letter, because a
    disagreement means the source CSV is internally inconsistent, and quietly trusting
    either field would put a wrong key into every run built from it.
    """
    letter = getattr(row, 'answer', None)
    if _missing(letter):
        return None
    letter = str(letter).strip().upper()
    valid = ANSWER_LETTERS[:len(choices)]
    if letter not in valid:
        raise ValueError(f'{row.id}: answer {letter!r} is not one of {valid}')
    answer = choices[ANSWER_LETTERS.index(letter)]

    text = getattr(row, 'answer_text', None)
    if not _missing(text) and str(text).strip() != answer.strip():
        raise ValueError(
            f'{row.id}: answer letter {letter} points at {answer!r} but answer_text says '
            f'{str(text).strip()!r} -- the source CSV disagrees with itself')
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='fpsbench_questiononly.csv')
    parser.add_argument('--video_root', type=str,
                        default='/home/av354855/.cache/fpsbench/clips/clip')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose clip is absent instead of dropping them. '
                             'For inspecting the conversion before the clips are staged; '
                             'an eval run over these would crash on load.')
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.src)
    clips = index_clips(args.video_root)

    records, missing, clamped_ids = [], [], []
    tasks = Counter()
    n_keyed = 0
    for row in df.itertuples(index=False):
        video_path = clips.get(row.id)
        if video_path is None:
            missing.append(row.id)
            if not args.allow_missing:
                continue
            video_path = os.path.join(args.video_root, f'{row.id}.mp4')

        choices = [str(getattr(row, col)) for col in CHOICE_COLUMNS]
        answer = resolve_answer(row, choices)
        n_keyed += answer is not None
        tasks[row.task_category] += 1
        cert_start, cert_end, clamped = relative_certificate(row)
        if clamped:
            clamped_ids.append(row.id)
        records.append({
            'video_id': row.id,
            'video_path': video_path,
            'conversations': [{
                'question': str(row.question_text),
                'choices': choices,
                # None when --src is the question-only release; the correct choice text
                # when it is the full annotation CSV. See module docstring.
                'answer': answer,
                'question_type': str(row.task_category),
                # Carried through for post-hoc analysis: min_fps is the frame rate
                # FPSBench says a clip needs before its certificate window is even
                # resolvable, so it is the number a sampling-rate sweep is read against.
                'min_fps': float(row.min_fps),
                'clip_duration_sec': float(row.clip_duration_sec),
                # Clip-relative temporal certificate. `end_time` is the key the streaming
                # solvers read to decide when a question fires (see
                # video_qa/rekv_fpsbench_stream_small_vqa.py); the offline solver ignores it and
                # its results are unchanged by this field existing.
                'start_time': cert_start,
                'end_time': cert_end,
                'cert_clamped': clamped,
            }],
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(records, f, indent=1)

    print(f'wrote {len(records)} records -> {args.out}')
    # Said out loud because it is the single fact that decides whether a run built from
    # this file can be scored at all: with no key the solver falls back to choices[0] and
    # every correct_choice becomes 'A'.
    if n_keyed == 0:
        print('answer key: ABSENT -- qa_acc/correct_choice will be meaningless; export '
              'predictions with video_qa/eval/export_fpsbench.py and score them upstream')
    elif n_keyed == len(records):
        print(f'answer key: present on all {n_keyed} records -- local scoring and MBA '
              f'(video_qa/eval/eval_fpsbench_mba.py) are available')
    else:
        raise ValueError(f'answer key on only {n_keyed}/{len(records)} records -- a '
                         f'partially keyed file would score a silent subset')
    print(f'tasks: {dict(sorted(tasks.items()))}')
    early = sum(1 for r in records
                if r['conversations'][0]['end_time'] < r['conversations'][0]['clip_duration_sec'])
    print(f'{early}/{len(records)} certificates end before the clip does -- those are the '
          f'questions a streaming run answers early (video_qa/rekv_fpsbench_stream_small_vqa.py)')
    if clamped_ids:
        print(f'{len(clamped_ids)} certificates fell outside their clip and were clamped '
              f'(flagged cert_clamped): {sorted(clamped_ids)}')
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'dropped'
        print(f'{len(missing)} ids with no clip, {verb}: {sorted(missing)}')


if __name__ == '__main__':
    main()
