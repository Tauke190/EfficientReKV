"""OVO-Bench solver: multiple choice asked mid-stream, at each query's own timestamp.

OVO-Bench's contract is that a query at `realtime` t may only be answered from frames in
[0, t]. ReKV satisfies this natively rather than by re-encoding a truncated clip per
query: `encode_frame` appends to a persistent KV-Cache, and `question_answering` wraps its
forward in set_retrieval()/reset_retrieval() (model/llava_onevision_rekv.py), so the
question and its generated answer never land in that cache. One ingestion pass therefore
serves every query on the video, and question k+1 sees exactly the frames question k saw
plus the ones in between -- no contamination from the intervening QA turns.

The alternative -- OVO-Bench's own pre-chunked clips, one truncated mp4 per query -- gives
the same visible window but pays the encode again for every query. On the realtime split
that is 837 encodes instead of 237.
"""

import math

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_stream_vqa import ReKVStreamVQA


def n_visible_frames(realtime, sample_fps, n_total):
    """How many sampled frames lie at or before `realtime` seconds.

    The frame streams take source frames 0, stride, 2*stride, ... with stride =
    fps/sample_fps, so sampled frame k sits at t = k / sample_fps and the visible set is
    k <= realtime * sample_fps, i.e. floor(realtime * sample_fps) + 1 frames.

    Deliberately floor-then-+1 rather than round(realtime * sample_fps): rounding up
    admits a frame recorded *after* the query timestamp, which is precisely the leak this
    benchmark exists to detect. The clamp to >= 1 covers the queries at t=0 (the realtime
    split has 5 below 4 seconds, one at exactly 0.0) -- with no frames at all the model
    would be answering blind, and encoding an empty tensor would fail inside the vision
    tower. Clamping to n_total covers videos shorter than their own query timestamp.
    """
    return max(1, min(int(math.floor(realtime * sample_fps)) + 1, n_total))


class ReKVOVOBenchVQA(ReKVStreamVQA):
    """Inherits open_stream/ingest from the streaming solver: frames arrive one at a time
    and are encoded one per forward pass, and REKV_FRAME_CACHE applies here too."""

    def video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        # 16 tokens: get_prompt(mc=True) primes the model with 'Best option: (', so a
        # well-behaved answer is a letter and a paren. The budget only exists to bound
        # models that ignore the priming.
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=16)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # Cap the stream at the last question's timestamp. OVO-Bench draws on full-length
        # source videos (Ego4D/video/, MovieNet, ...) but its queries stop at their own
        # timestamps, so the tail is never visible to anyone and there is no reason to
        # read it. When the video is *shorter* than this cap the stream is simply shorter,
        # which is what `truncated` below detects. `len(stream)` comes from the file
        # listing or the container header -- no frame is decoded to answer it.
        needed = n_visible_frames(
            max(float(c['end_time']) for c in video_sample['conversations']),
            self.sample_fps, n_total=1 << 30)
        stream = self.open_stream(video_sample, num_frames=needed)
        n_loaded = len(stream)

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        n_encoded = 0  # frames pushed into the KV-Cache so far
        last_realtime = -1.0
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            realtime = float(sample['end_time'])

            # The converter sorts by timestamp; assert it rather than trusting it, because
            # the failure mode of an unsorted file is silent -- the cursor below never
            # rewinds, so an out-of-order query would simply be answered against future
            # frames and score suspiciously well.
            assert realtime >= last_realtime, (
                f"{video_sample['video_id']}: conversations must be sorted by timestamp, "
                f'got {realtime} after {last_realtime}')
            last_realtime = realtime

            n_visible = n_visible_frames(realtime, self.sample_fps, n_loaded)
            if n_visible > n_encoded:
                # One frame read, one forward pass, in arrival order -- the frames between
                # the last query and this one, and not one frame further.
                n_encoded = self.ingest(stream, n_encoded, n_visible)

            choices = sample['choices']
            # gt_index, never choices.index(answer): OVO-Bench options are not unique and
            # its `answer` field is often a paraphrase of the option rather than a copy of
            # it (see video_qa/convert_ovobench.py).
            correct_choice = self.choice_letters[int(sample['gt_index'])]
            qa_results = self.video_close_qa(sample['question'], choices, correct_choice)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question_id': sample['question_id'],
                'task': sample['question_type'],
                'question': sample['question'],
                'choices': choices,
                'answer': sample['answer'],
                'correct_choice': correct_choice,
                'pred_answer': qa_results['pred_answer'],
                'pred_choice': qa_results['pred_choice'],
                'qa_acc': qa_results['acc'] * 100,
                'realtime': realtime,
                # The audit trail for the no-leak property: every row must satisfy
                # n_frames_seen <= floor(realtime * sample_fps) + 1. Checked in bulk by
                # video_qa/eval/eval_ovobench.py, which refuses to score a CSV that
                # violates it. `truncated` marks the queries whose video ended early --
                # those saw fewer frames than the benchmark intends, through no fault of
                # the model.
                'n_frames_seen': n_visible,
                # Frames decoded for this video, i.e. everything up to the last query's
                # timestamp -- not the video's full length, which is never loaded.
                'n_frames_loaded': n_loaded,
                'sample_fps': self.sample_fps,
                'truncated': bool(n_visible < math.floor(realtime * self.sample_fps) + 1),
                # Read per question, not once per video: ingestion is interleaved with the
                # questions, so the keep rate at this point is the one this answer was
                # produced under.
                **self.reduction_stats(),
            })

        self.qa_model.log_reduction_summary(n_encoded)


if __name__ == '__main__':
    work(ReKVOVOBenchVQA)
