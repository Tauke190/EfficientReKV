"""FPS-Bench-Stream: an FPSBench clip hidden in a 600 s haystack, questioned at the end.

This is the retrieval arm. `video_qa/rekv_fpsbench_stream_small_vqa.py` streams FPSBench's
own 2-25 s clips and asks each question the moment its evidence has gone past, which tests
whether the motion was resolved in time. Here the same clip is spliced into a 600 s MLVU
video and the question is asked at the end of the stream, which tests something the short
clips cannot: whether ReKV can still *find* the evidence after ~300 s of unrelated footage
has been ingested on top of it.

The construction is what makes that a clean measurement (see
/home/av354855/FPSBenchStream/README.md):

* the needle is a median 9.0 s, i.e. 1.5% of the stream, so nothing but retrieval can put
  it in front of the model at question time;
* the padding is topically unrelated, so the answer stays unambiguous -- "how many dribbles"
  cannot be answered off the haystack;
* needle position is stratified early/middle/late (329/330/331), so retrieval distance is a
  variable you can read the results against rather than a confound;
* every question has a key, unlike the question-only FPSBench release, so `qa_acc` is a real
  number and the run ends at a scorer.

Three things this solver does that the short-clip one does not:

* **The question fires at the end of the stream** (`--trigger end`, the default): the model
  has ingested all 600 s when it is asked. `--trigger query` fires at `query_time_sec`
  instead -- the release's real-time protocol, where the needle is the most recent thing in
  memory and retrieval distance is 0. That arm is the control: it says what the model scores
  when the same frames are in the cache but no retrieval is needed to reach them, so the gap
  between the two is the cost of having to retrieve.

* **It records what retrieval actually pulled in.** Accuracy alone cannot separate "the
  needle was never retrieved" from "it was retrieved and the model still got it wrong", and
  those want opposite fixes. `layers_hit_frac` is the share of the backbone's layers whose
  top-k blocks included one representing a needle frame; `blocks_hit_mean` is how many of
  the k slots per layer went to the needle. Both come from the block indices the model
  captures at question time (`Abstract_ReKV.retrieval_hit_stats`).

* **It reads the needle window in stream coordinates.** `video_qa/convert_fpsbench_stream.py`
  writes `needle_start_sec`/`needle_end_sec` on the assembled-stream clock; the release also
  carries the same event on the YouTube and MLVU clocks, and those are build-time provenance
  that means nothing to a solver reading the assembled file.

Everything else is inherited: FPSBench's own prompt, the exact-fps arrival grid, one frame
per forward pass, and the streaming audit columns.

**What RLT is expected to do here.** Stage-2 pruning (model/token_pruning.py) drops visual
tokens that repeat what memory already holds, so the haystack -- static, low-motion MLVU
footage -- compresses hard while the needle, which is fast motion by construction, does not.
That shrinks the pool retrieval searches without shrinking the needle in it, which is the
mechanism by which pruning could *raise* retrieval hit rate rather than only cheapening it.
It can also go the other way: a block that spans a long static stretch plus the needle is a
worse retrieval target than a block holding the needle alone. The columns above are what
tells the two apart, at the same measured keep rate.

**No frame cache.** Decoding is in-process with decord and `REKV_FRAME_CACHE` is ignored
(inherited). Pre-extracting one arm would be 990 streams x 600 frames = ~594k JPEGs.

Run through `python -m video_qa.run_eval --dataset fpsbench_stream`; see
scripts/eval_fpsbench_stream.sh.
"""

import math

from video_qa.base import work
from video_qa.rekv_fpsbench_stream_small_vqa import (
    ReKVFPSBenchStreamSmallVQA, add_timing_args)


def add_args(parser):
    """Flags for the long-stream harness, registered through `work(add_args=...)`.

    The prompt flags repeat the short-clip solver's rather than importing its `add_args`:
    that one also registers --full_clip, whose meaning here ("trigger at the end of the
    clip") is this benchmark's *default* and would be a second, contradictory spelling of
    --trigger.
    """
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Decode budget. FPSBench asks for a letter plus a brief "
                             "explanation, so the usual 16 truncates mid-sentence.")
    parser.add_argument("--no_none_of_above", action="store_true",
                        help="Drop the 'None of the above' option (E) from the presented "
                             "choices. Diagnostic: FPSBench presents it by default.")
    parser.add_argument("--shuffle_choices", action="store_true",
                        help="Shuffle option texts across letters, re-lettered from A "
                             "('None of the above' stays last). Diagnostic for position "
                             "bias; the presented order goes into the results CSV.")
    parser.add_argument("--choice_seed", type=int, default=2024,
                        help="--shuffle_choices only: RNG seed, per question.")
    parser.add_argument("--trigger", type=str, default='end', choices=['end', 'query'],
                        help="When each question fires. 'end' (default) is this "
                             "benchmark's protocol: the whole 600 s stream has been "
                             "ingested, so the needle is a median 294 s back and only "
                             "retrieval can reach it. 'query' fires at query_time_sec "
                             "(= the certificate end), where the needle is the most recent "
                             "content in the cache -- the control arm for how much of any "
                             "gap is retrieval rather than perception.")
    # Shared with the short-clip solver rather than repeated: unlike the prompt flags
    # above, these two mean exactly the same thing on both benchmarks.
    add_timing_args(parser)


class ReKVFPSBenchStreamVQA(ReKVFPSBenchStreamSmallVQA):
    def __init__(self, *pos, args=None, **kw):
        super().__init__(*pos, args=args, **kw)
        self.trigger_mode = args.trigger

    def trigger_time(self, sample, clip_duration=None):
        """When this question is asked, in stream seconds.

        Both branches read the assembled-stream clock written by
        video_qa/convert_fpsbench_stream.py. `clip_duration` is the stream's target
        duration (600 s); ingestion clamps it to the frames the file actually has, so a
        stream that came out a few hundredths short does not become a lookahead violation.
        """
        if self.trigger_mode == 'end':
            return float(sample.get('clip_duration_sec') or clip_duration)
        if 'query_time_sec' not in sample and 'end_time' not in sample:
            raise KeyError(
                "no 'query_time_sec' in the annotation -- rebuild it with "
                'video_qa/convert_fpsbench_stream.py, which writes the release timeline in '
                'assembled-stream coordinates. --trigger query has nothing to fire on '
                'without it.')
        return float(sample.get('query_time_sec', sample.get('end_time')))

    def needle_frames(self, sample, n_visible):
        """Arrival slots holding the needle, as (first, last) inclusive.

        Slot k arrives at t = k / sample_fps, so a slot is inside the needle when
        needle_start <= k/fps <= needle_end. Below ~1/needle_duration fps that window can
        contain no slot at all -- a 2 s needle sampled at 1 fps lands between two arrivals
        about half the time -- and the honest answer is then the nearest slot to the middle
        of the needle: it is the frame that carries whatever survived of the evidence, and
        calling that "no needle frames" would report a retrieval miss for a sampling gap.
        `n_needle_frames` in the row says which case a row is, so the two can be separated.
        """
        fps = self.sample_fps
        start = float(sample['needle_start_sec'])
        end = float(sample['needle_end_sec'])
        first = int(math.ceil(start * fps))
        last = int(math.floor(end * fps))
        if last < first:
            mid = int(round((start + end) / 2 * fps))
            first = last = mid
        first = max(0, min(first, n_visible - 1))
        last = max(first, min(last, n_visible - 1))
        return first, last

    def extra_row_fields(self, sample, n_visible):
        """Needle geometry and what retrieval did with it, for this question's row."""
        first, last = self.needle_frames(sample, n_visible)
        trigger = self.trigger_time(sample, sample.get('clip_duration_sec'))
        tokens_after = self.qa_model.tokens_after_frame(last)
        return {
            'question_id': sample.get('question_id'),
            'position_bin': sample.get('position_bin'),
            'position_frac': sample.get('position_frac'),
            'needle_start_sec': sample.get('needle_start_sec'),
            'needle_end_sec': sample.get('needle_end_sec'),
            'query_time_sec': sample.get('query_time_sec'),
            'stream_fps': sample.get('stream_fps'),
            'haystack_file': sample.get('haystack_file'),
            'needle_first_frame': first,
            'needle_last_frame': last,
            'n_needle_frames': last - first + 1,
            # How far back the answer is when the question comes. The independent variable
            # of this benchmark, in the two units that matter: seconds of stream, and
            # arrival slots (which is what memory is actually indexed by). Negative under
            # --trigger query, where the question fires at the certificate end and the
            # needle clip runs a few seconds past it -- the honest value, since those rows
            # were asked before their needle had finished arriving.
            'retrieval_distance_sec': round(trigger - float(sample['needle_end_sec']), 3),
            'retrieval_distance_frames': max(0, n_visible - 1 - last),
            # Whether the needle still sat in the local window when the question came, and
            # by how much. Rows where it did are answered by ordinary sliding-window
            # attention and say nothing about retrieval either way -- the `--trigger query`
            # control is entirely such rows, and so is the tail of the 'late' position bin
            # at low frame rates, where the shortest gap in the release (59 s) is under
            # n_local. Every retrieval number has to be read on the complement of this.
            'tokens_after_needle': tokens_after,
            'needle_in_local_window': bool(tokens_after < self.qa_model.n_local),
            # Empty unless retrieval ran for this question -- i.e. unless the stream
            # overflowed n_local -- so a row with no hit columns is one where the whole
            # stream sat in the local window and there was nothing to retrieve.
            **self.qa_model.retrieval_hit_stats(first, last),
        }


if __name__ == "__main__":
    work(ReKVFPSBenchStreamVQA, add_args=add_args)
