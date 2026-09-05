"""FPS-Bench-Stream: an FPSBench clip hidden in a 600 s haystack, questioned at the end.

Each of the 990 built streams is one FPSBench question whose clip (the *needle*, median
9 s) was spliced into a 600 s MLVU video (the *haystack*), so the evidence is ~1.5% of the
stream and sits at a known timestamp. The question is asked at the end of the stream, which
is what makes this a retrieval measurement: by then the needle is a median 294 s back, well
past `n_local`, and only ReKV's memory can reach it.

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

Three things worth knowing about how this solver runs:

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

**Frames arrive one at a time, on the exact --sample_fps grid.** Slot k *is* the frame at
t = k / sample_fps, taking the most recent source frame -- not an integer stride. Everything
in this file that converts between seconds and frame indices (`trigger_time` -> `n_visible`,
`needle_frames`) depends on that identity, and the stride path does not have it: at native
30 fps a stride run delivers 30 fps whether you asked for 16 or 64, so a needle window in
seconds would land on the wrong slots. `--exact_fps` is therefore not consulted here; the
grid is always exact.

**No frame cache.** Decoding is in-process with decord, during the eval, like every
other solver here. Pre-extracting one arm would be 990 streams x 600 frames = ~594k JPEGs.

**What RLT is expected to do here.** Stage-2 pruning (model/token_pruning.py) drops visual
tokens that repeat what memory already holds, so the haystack -- static, low-motion MLVU
footage -- compresses hard while the needle, which is fast motion by construction, does not.
That shrinks the pool retrieval searches without shrinking the needle in it, which is the
mechanism by which pruning could *raise* retrieval hit rate rather than only cheapening it.
It can also go the other way: a block that spans a long static stretch plus the needle is a
worse retrieval target than a block holding the needle alone. The columns above are what
tells the two apart, at the same measured keep rate.

Run through `python -m video_qa.run_eval --dataset fpsbench_stream`; the run ends at
video_qa/eval/eval_fpsbench_stream.py (accuracy, broken down by needle position and by
whether retrieval reached the needle) and video_qa/eval/check_fpsbench_stream.py (the
no-lookahead audit).
"""

import math
import time

import torch
from logzero import logger
from decord import VideoReader, cpu

from video_qa.base import BaseVQA, work, open_video_reader
from video_qa.fpsbench_prompt import build_prompt, parse_letter
from video_qa.rekv_stream_vqa import FrameStream, StridedStream


def add_timing_args(parser):
    """The two flags that change what a latency number means.

    Separate from the prompt flags because these do not change the benchmark, they change
    the measurement: everything else in the row is recorded unconditionally.
    """
    parser.add_argument("--force_answer_length", action="store_true",
                        help="Decode exactly --max_new_tokens tokens per question, so QA "
                             "latency is measured over a fixed decode length. Answer "
                             "length is a dependent variable of anything that perturbs "
                             "the KV-Cache, so a latency comparison without this partly "
                             "measures how much each arm chose to say. Changes the answers.")
    parser.add_argument("--retrieval_breakdown", action="store_true",
                        help="Split retrieval+prefill out of QA latency into "
                             "retrieval_seconds / generation_seconds. Costs a CUDA sync "
                             "per question, which inflates latency_seconds, so take the "
                             "headline latency from a run without it.")


def add_args(parser):
    """Flags for this solver, registered through `work(add_args=...)`."""
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
    parser.add_argument("--sample_fps_list", type=str, default=None,
                        help="Comma-separated frame rates to run in one pass over each "
                             "video, e.g. '1,2,4', overriding --sample_fps. The video is "
                             "decoded once at the highest rate and the lower rates are "
                             "strided views of that decode (StridedStream), which deliver "
                             "bit-identical frames to separate runs because the exact "
                             "grids nest. Every rate must divide the highest exactly. "
                             "Rows carry their own sample_fps, so one CSV holds all rates; "
                             "split it with scripts/split_fpsbench_by_fps.py to get the "
                             "per-rate layout the scorer expects. Worth it only when the "
                             "rates would each get the same worker count anyway: this mode "
                             "holds the highest rate's KV-Cache, so a memory-limited run "
                             "pays the top rate's worker count for every rate.")
    parser.add_argument("--decode_hold_gb", type=float, default=6.0,
                        help="--sample_fps_list only: most decoded pixels one video may "
                             "hold. Sharing a decode means holding it, at source "
                             "resolution, for the whole video -- and the release is not "
                             "one resolution: 470 of 990 streams are 720x540 or smaller "
                             "(2.8 GB at 4 fps) but the 1920x8xx ones are 4.6 MB a frame "
                             "(11 GB), and the largest is 35 GB. A single unbudgeted "
                             "video would take the worker down, so any video whose shared "
                             "decode would exceed this falls back to one windowed decode "
                             "per rate -- slower for that video, identical results.")
    add_timing_args(parser)


def now():
    """Wall clock, with the GPU caught up first.

    CUDA launches are asynchronous, so a `perf_counter()` around a model call without this
    measures how long it took to *queue* the work.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class ReKVFPSBenchStreamVQA(BaseVQA):
    def __init__(self, *pos, args=None, **kw):
        super().__init__(*pos, **kw)
        self.trigger_mode = args.trigger
        self.max_new_tokens = args.max_new_tokens
        self.no_none_of_above = args.no_none_of_above
        self.shuffle_choices = args.shuffle_choices
        self.choice_seed = args.choice_seed
        self.force_answer_length = args.force_answer_length
        self.retrieval_breakdown = args.retrieval_breakdown
        self.fps_list = None
        if getattr(args, 'sample_fps_list', None):
            self.fps_list = sorted(float(x) for x in args.sample_fps_list.split(','))
            # self.sample_fps still drives every seconds<->slot conversion in this file;
            # analyze_a_video rebinds it per rate. Start it at the first rate so anything
            # read before the loop (the save path, logging) sees a member of the list.
            self.sample_fps = self.fps_list[0]
        self.decode_hold_gb = getattr(args, 'decode_hold_gb', 6.0)
        self.n_shared, self.n_fallback = 0, 0

    # --- ingestion ------------------------------------------------------------------

    def open_stream(self, video_sample):
        # exact=True regardless of --exact_fps, for the reason in the module docstring:
        # the stride grid does not put slot k at t = k / sample_fps, and every conversion
        # in this file assumes it does.
        if self.fps_list:
            # Decoded once at the highest rate; every lower rate reads it through a
            # StridedStream. Eager (window=0) rather than windowed: each rate is its own
            # pass over the same slots, and a windowed base would re-decode every block
            # once per pass, which is exactly the cost this mode exists to remove.
            #
            # That means holding the decode at source resolution, and the release spans
            # 0.23 to 14.75 MB a frame -- so the hold is budgeted per video rather than
            # assumed affordable. Over budget, this video runs one windowed decode per
            # rate: slower for that video, bit-identical results either way.
            top = max(self.fps_list)
            gb = self._shared_decode_gb(video_sample['video_path'], top)
            if gb <= self.decode_hold_gb:
                self.n_shared += 1
                return FrameStream(video_sample['video_path'], top, exact=True, window=0)
            self.n_fallback += 1
            logger.debug(f"{video_sample['video_id']}: shared decode would hold "
                         f'{gb:.1f} GB > {self.decode_hold_gb} GB; one decode per rate')
            return None
        return FrameStream(video_sample['video_path'], self.sample_fps, exact=True,
                           window=self.decode_window)

    def _shared_decode_gb(self, video_path, top_fps):
        """Pixels a shared decode at `top_fps` would hold, in GB.

        Reads one frame for its shape rather than trusting the annotation: the release
        normalizes to several resolutions and a wrong guess here is an OOM, not a slow
        video. One frame off a container that is about to be decoded in full is noise.
        """
        vr = open_video_reader(video_path)
        h, w, c = vr.get_batch([0]).asnumpy().shape[-3:]
        n_slots = max(1, int(round(len(vr) / round(vr.get_avg_fps()) * top_fps)))
        return n_slots * h * w * c / 1e9

    def ingest(self, stream, start, end):
        """Encode slots [start, end) as they arrive, one forward pass each.

        Returns the index one past the last slot encoded, i.e. `end` clamped to the length
        of the stream -- a stream can come out a few hundredths short of its target.
        """
        n = 0
        for frame in stream.frames(start, end):
            self.qa_model.encode_frame(frame)
            n += 1
        return start + n

    def n_visible_frames(self, trigger_sec, n_total):
        """How many slots have arrived at or before `trigger_sec`.

        floor-then-+1 rather than round: rounding up admits a frame recorded *after* the
        trigger, which under `--trigger query` is exactly the lookahead this benchmark's
        control arm must not have. Clamped to >= 1 (a question at t=0 still needs a frame
        to look at) and to the frames the file actually holds.
        """
        return max(1, min(int(math.floor(trigger_sec * self.sample_fps)) + 1, n_total))

    # --- the question ---------------------------------------------------------------

    def presented_choices(self, sample):
        """FPSBench's prompt for this question, and the letters it presents.

        The annotation stores choices as a list in release-letter order (A..E, E always
        "None of the above"); `build_prompt` wants the {letter: text} mapping FPSBench's
        own harness uses, so the list is re-lettered by position here rather than anywhere
        the order could drift.
        """
        choices = list(sample['choices'])
        letters = ['A', 'B', 'C', 'D', 'E'][:len(choices)]
        example = {'question': {'text': sample['question'],
                                'choices': dict(zip(letters, choices))}}
        return build_prompt(
            example,
            include_none_of_above=not self.no_none_of_above,
            shuffle=self.shuffle_choices,
            seed=self.choice_seed,
        )

    def ask(self, input_text):
        """One question against the current cache, with its latency.

        The breakdown wraps `_retrieve_and_prefill` for the duration of this call instead
        of timing inside the model: retrieval and the prefill it feeds are one forward pass
        there, and the decode loop that follows is what the split is trying to separate
        them from. Restoring the bound method with `del` rather than reassignment keeps the
        instance clean, so a run that raises mid-question does not leave every later row
        timed through a stale closure.
        """
        model = self.qa_model
        kwargs = {'max_new_tokens': self.max_new_tokens}
        if self.force_answer_length:
            kwargs['min_new_tokens'] = self.max_new_tokens

        if not self.retrieval_breakdown:
            t0 = now()
            answer = model.question_answering(input_text, **kwargs)
            return answer, {'latency_seconds': round(now() - t0, 4)}

        spent = []
        original = model._retrieve_and_prefill

        def timed(*a, **kw):
            t = now()
            out = original(*a, **kw)
            spent.append(now() - t)
            return out

        model._retrieve_and_prefill = timed
        try:
            t0 = now()
            answer = model.question_answering(input_text, **kwargs)
            total = now() - t0
        finally:
            del model._retrieve_and_prefill
        retrieval = sum(spent)
        return answer, {
            'latency_seconds': round(total, 4),
            'retrieval_seconds': round(retrieval, 4),
            'generation_seconds': round(total - retrieval, 4),
        }

    def video_close_qa(self, sample):
        """Ask this question and score it against the key.

        The correct letter is recovered from the *presented* order, not from the
        annotation's `answer_letter`: under --shuffle_choices the texts have been
        re-lettered, and scoring against the release letter would mark a correct answer
        wrong. The lookup is by text, which is what the shuffle permutes.
        """
        prompt_text, presented, _ = self.presented_choices(sample)
        letters = [letter for letter, _ in presented]
        correct_choice = next(
            (letter for letter, text in presented if text == sample['answer']), '')
        if not correct_choice:
            # Only reachable when the key was dropped from the presented set, i.e.
            # --no_none_of_above on a question whose answer is "None of the above". Such a
            # row is unanswerable and scores 0; say so, because a silent floor of zero on
            # part of the benchmark reads as a model failure.
            logger.warning(f'{sample.get("question_id")}: the key '
                           f'{sample["answer"]!r} is not among the presented choices; '
                           f'this question cannot be answered correctly')

        input_text = {
            # Retrieval keys on the bare question, not on the prompt: the system preamble
            # and the option block are the same on every question, so including them would
            # push every question's query vector toward the same point.
            'question': sample['question'],
            'prompt': self.qa_model.get_prompt(prompt_text),
        }
        pred_answer, timings = self.ask(input_text)
        pred_choice = parse_letter(pred_answer, valid=letters)
        return {
            'pred_answer': pred_answer.replace('\n', ' ').strip(),
            # None, not '', so an unparsable response is NaN in the CSV and the scorer's
            # `pred_choice.isna()` count is the number of prompt failures.
            'pred_choice': pred_choice or None,
            'correct_choice': correct_choice,
            'presented_choices': [text for _, text in presented],
            'acc': float(bool(pred_choice) and pred_choice == correct_choice),
            'n_generated_tokens': getattr(self.qa_model, 'last_generated_tokens', None),
            **timings,
        }

    # --- where the needle is, and what retrieval did with it ------------------------

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

    # --- the run --------------------------------------------------------------------

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        """One video, at every requested rate, over a single decode.

        Rates run ascending so the run reaches its peak KV-Cache (the highest rate) last,
        with every cheaper rate's cache already released -- the peak is one rate's, not
        the sum. `self.sample_fps` is rebound per rate because every seconds-to-slot
        conversion in this file reads it; it is restored afterwards so a failure mid-video
        cannot leave the next video running at the wrong rate.
        """
        base = self.open_stream(video_sample)
        if not self.fps_list:
            self._run_one_rate(video_sample, base)
            return

        original = self.sample_fps
        try:
            for fps in self.fps_list:
                self.sample_fps = fps
                if base is None:
                    # Over the hold budget: this video pays its own windowed decode per
                    # rate, exactly as a single-rate run would.
                    view = FrameStream(video_sample['video_path'], fps, exact=True,
                                       window=self.decode_window)
                else:
                    view = base if fps == base.sample_fps else StridedStream(base, fps)
                self._run_one_rate(video_sample, view)
        finally:
            self.sample_fps = original

    @torch.inference_mode()
    def _run_one_rate(self, video_sample, stream):
        n_loaded = len(stream)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        n_encoded = 0
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            trigger = self.trigger_time(sample, sample.get('clip_duration_sec'))
            n_visible = self.n_visible_frames(trigger, n_loaded)
            if n_visible > n_encoded:
                # One frame read, one forward pass, in arrival order, and not one frame
                # past the trigger. The cursor never rewinds, so a second question on the
                # same stream would see exactly these frames plus the ones after them.
                n_encoded = self.ingest(stream, n_encoded, n_visible)

            qa_results = self.video_close_qa(sample)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'task': sample.get('question_type'),
                'question': sample['question'],
                'answer': sample['answer'],
                'answer_letter': sample.get('answer_letter'),
                'min_fps': sample.get('min_fps'),
                'sample_fps': self.sample_fps,
                'trigger': self.trigger_mode,
                'trigger_time_sec': round(trigger, 3),
                # The audit trail for the no-leak property, checked in bulk by
                # video_qa/eval/check_fpsbench_stream.py: every row must satisfy
                # n_frames_seen <= floor(trigger_time_sec * sample_fps) + 1.
                'n_frames_seen': n_visible,
                'n_frames_loaded': n_loaded,
                'truncated': bool(n_visible < math.floor(trigger * self.sample_fps) + 1),
                'qa_acc': qa_results['acc'] * 100,
                **{k: v for k, v in qa_results.items() if k != 'acc'},
                # Whether retrieval ran at all for this question. False means the whole
                # stream fit inside n_local, so the row measures the backbone over a long
                # context rather than ReKV's memory -- and every retrieval column is empty.
                'retrieval_fired': bool(self.qa_model.last_retrieved_blocks),
                **self.extra_row_fields(sample, n_visible),
                # Read after the question, so the keep rate is the one this answer was
                # produced under.
                **self.reduction_stats(),
            })

        self.qa_model.log_reduction_summary(n_encoded)


if __name__ == "__main__":
    work(ReKVFPSBenchStreamVQA, add_args=add_args)
