"""FPS-Bench-Stream, needle only: the FPSBench clip on its own, with the haystack cut away.

The control for video_qa/rekv_fpsbench_stream_vqa.py. Each of the 990 streams is read on
exactly the same frame grid (`FrameStream(..., exact=True)` at --sample_fps over the
assembled 600 s file), but only the slots inside the needle window are encoded:

    needle_start_sec <= k / sample_fps <= needle_end_sec

so the model sees the same pixels, at the same rate and canvas, as the needle arm of the
full stream -- and nothing else. The KV-Cache starts empty at the needle's first frame and
the question fires right after its last one. What this measures is the backbone's
perception of the evidence at that frame rate and prune rate, with no haystack to compete
with and (almost always) nothing to retrieve. The gap to the full-stream run at the same
fps/prune setting is what the 600 s of padding costs.

`--needle_stop` picks where the window ends:

* `needle` (default): the whole needle clip, through `needle_end_sec`;
* `query`: stop at `query_time_sec` (= certificate end), the last frame the full-stream
  `--trigger query` arm can see. Use it for a frame-for-frame comparison with that arm.

Frame indexing. Rows keep the stream arm's column names so the same scorer
(video_qa/eval/eval_fpsbench_stream.py) and audit (video_qa/eval/check_fpsbench_stream.py)
run unchanged, but with memory coordinates: frame 0 is the needle's first slot, so
`needle_first_frame` = 0, `needle_last_frame` = `n_frames_seen` - 1, and
`trigger_time_sec` is measured from the needle start. The stream-clock slots are kept in
`stream_first_slot` / `stream_last_slot`.

`needle_in_local_window` here means the *whole* needle fit inside n_local. It does not at
high rates on long needles (4 fps x 25 s x 196 tokens > 15000), and on those rows retrieval
fires over a memory that is entirely needle, so its hit columns are trivially 1 and say
nothing about retrieval quality.

Run through `python -m video_qa.run_eval_fpsbench_needle`.
"""

import math

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_fpsbench_stream_vqa import ReKVFPSBenchStreamVQA, add_args
from video_qa.rekv_stream_vqa import FrameStream


def add_needle_args(parser):
    add_args(parser)
    parser.add_argument("--needle_stop", type=str, default='needle',
                        choices=['needle', 'query'],
                        help="Where the ingested window ends. 'needle' (default) feeds the "
                             "whole needle clip, through needle_end_sec. 'query' stops at "
                             "query_time_sec (= certificate end), matching the last frame "
                             "the full-stream --trigger query arm sees.")


class ReKVFPSBenchNeedleVQA(ReKVFPSBenchStreamVQA):
    def __init__(self, *pos, args=None, **kw):
        super().__init__(*pos, args=args, **kw)
        if self.fps_list:
            raise ValueError('--sample_fps_list is not supported for the needle-only arm; '
                             'run one --sample_fps per process.')
        self.needle_stop = args.needle_stop
        # Rows are tagged with this in place of the stream arm's 'end'/'query'.
        self.trigger_mode = f'needle-{args.needle_stop}'

    def window_sec(self, sample):
        start = float(sample['needle_start_sec'])
        if self.needle_stop == 'query':
            end = float(sample.get('query_time_sec', sample['needle_end_sec']))
        else:
            end = float(sample['needle_end_sec'])
        return start, end

    def window_slots(self, sample, n_slots):
        """Stream slots [first, last] inside the needle window, on the exact grid.

        Same rule as `needle_frames` in the stream solver, so the two arms mark the same
        slots as needle: slot k is in when start <= k / fps <= end, and a window too short
        to contain any slot at this rate falls back to the slot nearest its middle.
        """
        fps = self.sample_fps
        start, end = self.window_sec(sample)
        first = int(math.ceil(start * fps))
        last = int(math.floor(end * fps))
        if last < first:
            first = last = int(round((start + end) / 2 * fps))
        first = max(0, min(first, n_slots - 1))
        last = max(first, min(last, n_slots - 1))
        return first, last

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # Windowed, so only the blocks holding the needle are ever decoded.
        stream = FrameStream(video_sample['video_path'], self.sample_fps, exact=True,
                             window=self.decode_window or 64)
        n_loaded = len(stream)

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            # One question per stream in this release; a fresh cache per question keeps
            # that true even if a later annotation carries more.
            self.qa_model.clear_cache()
            self.qa_model.encode_init_prompt()

            first, last = self.window_slots(sample, n_loaded)
            n_seen = self.ingest(stream, first, last + 1) - first
            start_sec, end_sec = self.window_sec(sample)
            trigger = end_sec - start_sec

            qa_results = self.video_close_qa(sample)
            total_tokens = self.qa_model.tokens_after_frame(-1)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'task': sample.get('question_type'),
                'question': sample['question'],
                'answer': sample['answer'],
                'answer_letter': sample.get('answer_letter'),
                'min_fps': sample.get('min_fps'),
                'sample_fps': self.sample_fps,
                'trigger': self.trigger_mode,
                # Needle clock: 0 is the needle's first slot. The lookahead audit reads
                # this against n_frames_seen, which is in the same coordinates.
                'trigger_time_sec': round(max(trigger, 0.0), 3),
                'n_frames_seen': n_seen,
                'n_frames_loaded': n_loaded,
                'truncated': False,
                'qa_acc': qa_results['acc'] * 100,
                **{k: v for k, v in qa_results.items() if k != 'acc'},
                'retrieval_fired': bool(self.qa_model.last_retrieved_blocks),
                'question_id': sample.get('question_id'),
                'position_bin': sample.get('position_bin'),
                'position_frac': sample.get('position_frac'),
                'needle_start_sec': sample.get('needle_start_sec'),
                'needle_end_sec': sample.get('needle_end_sec'),
                'query_time_sec': sample.get('query_time_sec'),
                'stream_fps': sample.get('stream_fps'),
                'haystack_file': sample.get('haystack_file'),
                'stream_first_slot': first,
                'stream_last_slot': last,
                'needle_first_frame': 0,
                'needle_last_frame': n_seen - 1,
                'n_needle_frames': n_seen,
                'retrieval_distance_sec': 0.0,
                'retrieval_distance_frames': 0,
                'tokens_after_needle': 0,
                'needle_tokens': total_tokens,
                'needle_in_local_window': bool(total_tokens <= self.qa_model.n_local),
                **self.qa_model.retrieval_hit_stats(0, n_seen - 1),
                **self.reduction_stats(),
            })
            self.qa_model.log_reduction_summary(n_seen)


if __name__ == "__main__":
    work(ReKVFPSBenchNeedleVQA, add_args=add_needle_args)
