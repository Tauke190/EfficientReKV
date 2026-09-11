"""Convert OVBench's annotation into ReKV's video-level schema.

OVBench (VideoChat-Online, MCG-NJU, CVPR 2025) ships one record per *video* already --
`ovbench.json`, 1463 videos carrying 7090 questions -- so unlike convert_ovobench.py this
is not a transpose. What it does is normalise the fields the streaming solver needs and
resolve each `video_id` to a file, because four things about the source are not what
video_qa/rekv_ovobench_vqa.py expects.

* **`answer` is a letter, not an index or a string.** Every answer is one of A-D and every
  option is prefixed with its own letter ("A. stationary"), verified across all 7090
  questions: 0 letters out of range, 0 options missing the `"<L>. "` prefix. So
  `gt_index = ord(answer) - 65` is exact, and it is written out explicitly -- the solver
  keys on `gt_index` and never recovers the letter by text lookup.
* **The prefixes have to come off.** base.format_mcqa_prompt renders choices as
  "(A) <text>", so leaving them in shows the model "(A) A. stationary". Options are also
  not unique across a question in general, which is the other reason index-by-text is
  unusable here.
* **Timestamps are absolute in the full video's timeline, not relative to `clip`.** This
  is the one thing that would silently destroy the benchmark if read the other way. Each
  entry carries `clip: [start, end]`, and for the three container sources `start` is often
  far from zero (AVA_RAW reaches 1000 s). Checked both readings against all 5143 container
  questions: 5143/5143 fall inside `[clip[0], clip[1]]`, while a clip-relative reading puts
  0/1396 AVA_RAW questions in range. So `middle_frame_timestamp` goes straight into
  `end_time` with no offset. The frame-derived sources all have `clip[0] == 0`, where the
  two readings coincide.
* **There are no question ids.** Synthesised as `<video_id>#q<i>` from the position in the
  source list, so an id is stable under the re-sort below and traceable back to the file.

Streams start at t=0, not at `clip[0]`. That is the honest streaming reading -- a model
watching the video has seen everything before the question -- and it is what the no-leak
invariant `n_frames_seen <= floor(end_time * sample_fps) + 1` is defined against. It costs
1.26x more ingestion than starting at `clip[0]` (78.0 vs 61.7 hours at 1 fps), almost all
of it in AVA_RAW (31.3 vs 15.2 h), because the other sources begin at zero anyway. Reading
it the other way would need a start offset the solver does not have.

Videos are resolved as `<video_root>/<video_id>` and then `<video_root>/<video_id>.mp4`.
Both spellings are needed: the container sources name a real file with its extension
("COIN/.../xyz.mp4"), while the seven JPEG-derived sources name a directory that
scripts/dataset_prep/setup_ovbench.py transcoded to a single .mp4 of the same name. All 1463 resolve.

Note that the transcoded clips are frequently *longer* than their own `clip` window (281
of 326, median +3.0 s), which is upstream's framing inherited from the frame directories,
not an artefact of the transcode. It is harmless here because ingestion is gated on
`end_time` rather than on the file length, and every question's timestamp lands inside its
video -- checked: 0 of 7090 fall past the end.

Lives here rather than beside the data because `data/*` is gitignored.

Usage:
    python video_qa/convert_ovbench.py \
        --src data/OVBench/ovbench.json \
        --video_root data/OVBench/videos \
        --out data/OVBench/full_mc.json
"""

import os
import json
import argparse
from collections import Counter, defaultdict

# answer_type -> its sub_answer_types. Fixed by the benchmark, and asserted below rather
# than trusted: a sub-task appearing under a new parent means the annotation changed under
# this converter. video_qa/eval/eval_ovbench.py groups its report by the same map.
TASK_GROUPS = {
    'Temporal Perception': ['Action Sequence', 'Object Existence State', 'Step Localization'],
    'Spatio Perception': ['Action Location', 'Object Position'],
    'Spatio Temporal Perception': ['Action Trajectory', 'Object Trajectory'],
    'Past Memory': ['Action Retrieval', 'Procedure Recall', 'Trajectory Retrieval'],
    'Future Prediction': ['Action Anticipation', 'Goal/Step Prediction', 'Movement Prediction'],
    'Temporal Hallucination Verification': ['Action Persistence', 'Object Presence',
                                            'Step Verification'],
}


def video_id_of(rel_path):
    """Filesystem-safe id from OVBench's `<source>/<rest>` path.

    The id names the video in results.csv, so it has to survive being a single field: the
    separator goes, and so does the extension the container sources carry, which would
    otherwise make 'COIN/x.mp4' and a hypothetical 'COIN/x.mkv' collide differently than
    their files do.
    """
    return rel_path.replace('/', '__')


def resolve(video_root, rel_path):
    """The file for a `video_id`, or None. See the module docstring on the two spellings."""
    for candidate in (os.path.join(video_root, rel_path),
                      os.path.join(video_root, rel_path + '.mp4')):
        if os.path.exists(candidate):
            return candidate
    return None


def strip_letter(option, i):
    """'A. stationary' -> 'stationary'. Asserted, not attempted."""
    prefix = f'{chr(65 + i)}. '
    assert option.startswith(prefix), f'option {i} lacks the {prefix!r} prefix: {option!r}'
    return option[len(prefix):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data/OVBench/ovbench.json')
    parser.add_argument('--video_root', type=str, default='data/OVBench/videos',
                        help='Output of scripts/dataset_prep/setup_ovbench.py -- one video per clip.')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--allow_missing', action='store_true',
                        help='Emit records whose video file is absent instead of dropping '
                             'them. For inspecting the conversion before the videos are '
                             'staged; an eval run over these would crash on load.')
    args = parser.parse_args()

    records = json.load(open(args.src))
    sub_to_group = {s: g for g, subs in TASK_GROUPS.items() for s in subs}

    anno, missing, n_questions = [], [], 0
    counts, groups = Counter(), Counter()
    for entry in records:
        rel_path = entry['video_id']
        video_path = resolve(args.video_root, rel_path)
        if video_path is None:
            missing.append(rel_path)
            if not args.allow_missing:
                continue
            video_path = os.path.join(args.video_root, rel_path)

        conversations = []
        for i, q in enumerate(entry['questions']):
            options = [strip_letter(o, j) for j, o in enumerate(q['options'])]
            gt = ord(q['answer']) - 65
            assert 0 <= gt < len(options), (
                f"{rel_path} q{i}: answer {q['answer']!r} outside {len(options)} options")
            sub = q['sub_answer_type']
            assert sub_to_group.get(sub) == q['answer_type'], (
                f"{rel_path} q{i}: sub-task {sub!r} is not listed under {q['answer_type']!r} "
                f'in TASK_GROUPS -- the annotation taxonomy changed, update the map')
            conversations.append({
                'question_id': f'{video_id_of(rel_path)}#q{i}',
                'question': q['question'],
                'choices': options,
                'answer': options[gt],
                'gt_index': gt,
                # The fine-grained axis is what the solver records as `task`; the 6 coarse
                # types are recovered from TASK_GROUPS at scoring time, so both levels are
                # reportable from one column.
                'question_type': sub,
                'task_group': q['answer_type'],
                # Absolute in the full video, never offset by clip[0] -- see the docstring.
                'start_time': 0,
                'end_time': float(q['middle_frame_timestamp']),
            })
            counts[sub] += 1
            groups[q['answer_type']] += 1

        # Ascending timestamp is a correctness requirement, not a nicety: the streaming
        # solver only ever moves its ingestion cursor forward, so a question placed out of
        # order would be answered against frames from beyond its own end_time -- a
        # future-information leak that inflates accuracy while looking perfectly normal.
        # question_id breaks ties so the conversion is deterministic.
        conversations.sort(key=lambda c: (c['end_time'], c['question_id']))

        n_questions += len(conversations)
        anno.append({
            'video_id': video_id_of(rel_path),
            'video_path': video_path,
            'source_video': rel_path,
            # Carried for reference only. Nothing gates on it: the stream starts at 0 and
            # stops at the last end_time.
            'clip': entry.get('clip'),
            'source_fps': entry.get('fps'),
            'conversations': conversations,
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(anno, f, indent=1)

    print(f'{n_questions} questions over {len(anno)} videos -> {args.out}')
    print('\nper answer_type:')
    for g in sorted(groups):
        print(f'  {g:<38} {groups[g]:5d}')
    print('\nper sub_answer_type:')
    for s in sorted(counts):
        print(f'  {s:<38} {counts[s]:5d}')
    if missing:
        verb = 'kept (--allow_missing)' if args.allow_missing else 'DROPPED'
        print(f'\n{len(missing)} videos not found under {args.video_root} and {verb}; '
              f'e.g. {missing[:3]}')
        print('Run scripts/dataset_prep/setup_ovbench.py to unpack them.')


if __name__ == '__main__':
    main()
