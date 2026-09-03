"""StreamBench solver: free-form answers asked mid-stream, at each breakpoint's timestamp.

StreamBench (StreamChat, arXiv 2501.13468) is the one benchmark here that is *both*
streaming and open-ended. Its contract is OVO-Bench's -- a question at `time` t may only
be answered from frames in [0, t] -- but its references are sentences, so there is no
letter to match and scoring goes through an LLM judge
(video_qa/eval/eval_open_ended_local.py with `--judge_preset streambench`).

That combination is why this is a distinct file rather than a reuse. The generation half
already exists in ReKVStreamVQA, which rvs_ego/rvs_movie use: streaming ingest gated on
`end_time`, `video_open_qa` for free text. What it does not do is record anything a
StreamBench report needs. Three columns are added here:

* **`task`** -- the question class (OS, LM, SM, CI, KG, SF). Near-evenly balanced at ~300
  each and measuring very different things: KG is Knowledge-based QA, world knowledge with
  no visual content at all ("From which ingredient is sesame oil extracted?"), so it is
  answerable with the video switched off and will sit far above the rest. A pooled figure
  that averages it with Object Search is not reporting either one. video_qa/eval/
  eval_streambench.py splits on this column.
* **`source`** -- Ego / WebVideo / Movie, which differ in length and in what can be asked.
* **`realtime` / `n_frames_seen` / `sample_fps`** -- the audit trail for the no-leak
  property. Every row must satisfy n_frames_seen <= floor(realtime * sample_fps) + 1, and
  without these three columns that is unverifiable after the fact. ReKVStreamVQA does not
  write them because RVS-Ego is scored as one pooled number; here the time limit *is* the
  benchmark, so it has to be checkable.

The visible-frame count follows the parent's convention -- `int(end_time * sample_fps)`,
not OVO-Bench's floor-then-plus-one -- so a StreamBench arm stays comparable with the
rvs_* arms this solver's base class produces. It is the stricter of the two by one frame
and never leaks. StreamBench's earliest breakpoint is at t=3 s, so it cannot produce the
zero-frame case that convention would hit at t=0.
"""

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_stream_vqa import ReKVStreamVQA


class ReKVStreamBenchVQA(ReKVStreamVQA):
    """Inherits open_stream/ingest/video_open_qa; overrides only what is recorded."""

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # Cap the stream at the last breakpoint. StreamBench videos run well past their
        # final question (Movie clips especially), and the tail is never visible to
        # anyone, so decoding it is pure cost. `len(stream)` comes from the container
        # header -- no frame is decoded to answer it.
        last = max(float(c['end_time']) for c in video_sample['conversations'])
        needed = max(1, int(last * self.sample_fps))
        stream = self.open_stream(video_sample, num_frames=needed)
        n_loaded = len(stream)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        n_encoded = 0
        last_realtime = -1.0
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            realtime = float(sample['end_time'])

            # The converter sorts by timestamp; assert it rather than trusting it, because
            # the failure mode is silent -- the cursor below never rewinds, so an
            # out-of-order question would be answered against future frames and score
            # suspiciously well. One video in streaming_bench_v0.3.json ships unsorted.
            assert realtime >= last_realtime, (
                f"{video_sample['video_id']}: conversations must be sorted by timestamp, "
                f'got {realtime} after {last_realtime}')
            last_realtime = realtime

            n_visible = min(max(1, int(realtime * self.sample_fps)), n_loaded)
            if n_visible > n_encoded:
                # One frame read, one forward pass, in arrival order -- the frames between
                # the last question and this one, and not one frame further.
                n_encoded = self.ingest(stream, n_encoded, n_visible)

            # 256 new tokens: references run to 59 words, and the judge compares meaning,
            # so an answer cut mid-sentence is scored as a wrong one. Matches the budget
            # ReKVStreamVQA uses for the rvs_* open-ended runs.
            qa_results = self.video_open_qa(sample['question'], max_new_tokens=256)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question_id': sample['question_id'],
                'task': sample['question_type'],
                'source': sample.get('source'),
                'question': sample['question'],
                # Free-form reference. eval_open_ended_local.py reads exactly
                # question/answer/pred_answer, so these three names are load-bearing.
                'answer': sample['answer'],
                'pred_answer': qa_results['pred_answer'],
                'realtime': realtime,
                'n_frames_seen': n_visible,
                'n_frames_loaded': n_loaded,
                'sample_fps': self.sample_fps,
                # Marks the questions whose video ended early -- those saw fewer frames
                # than the benchmark intends, through no fault of the model.
                'truncated': bool(n_visible < int(realtime * self.sample_fps)),
                # Read per question, not once per video: ingestion is interleaved with the
                # questions, so the keep rate here is the one this answer was produced
                # under.
                **self.reduction_stats(),
            })

        self.qa_model.log_reduction_summary(n_encoded)


if __name__ == '__main__':
    work(ReKVStreamBenchVQA)
