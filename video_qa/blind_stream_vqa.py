"""Blind control for the free-form streaming solvers -- StreamBench in particular.

Same idea as video_qa/blind_vqa.py, different answer type. That solver is multiple-choice:
it formats options, extracts a letter and scores the row itself. StreamBench's annotation
carries no `choices` at all (data/streambench/full_oe.json is question/answer sentences),
so the MCQA solver cannot run against it -- it raises KeyError('choices') on the first
sample. The generation half has to be the open-ended one, and the grading has to stay with
the LLM judge, exactly as in the sighted run.

**What "blind" means here** is what it means in blind_vqa: ReKV never passes a video at
question time -- frames are prefilled into the KV-Cache beforehand -- so this does not
suppress an input, it skips ingestion. `clear_cache()` and `encode_init_prompt()` run as
in a sighted run, leaving the system-prompt tokens in the cache, and no frame is decoded
or handed to the vision tower. `retrieved_indices=[[]]` routes through the *external*
retrieval branch so the backend skips the block search instead of performing an empty one
against an empty cache (which trips an assert in model/attention/kv_cache_manager.py).

**Why this number matters on StreamBench.** KG (Knowledge-based QA) is 298 of the 1838
questions and is pure world knowledge -- "From which ingredient is sesame oil extracted?"
-- answerable with the video off, and it is 16% of the pooled figure. The memory and
search classes (LM/SM/OS) are the opposite: they should collapse without frames. So the
blind row is not one floor but six, and a class that does not collapse is a class the
video was not contributing to. video_qa/eval/eval_streambench.py prints the per-class
split and already recognises the `blind` column this writes -- it skips the no-leak audit
(0 frames cannot leak) and stamps the output as a control rather than a score.

Rows carry the same columns as ReKVStreamBenchVQA, keyed by the same
(video_id, question_id), so the blind CSV joins directly to its sighted twin and goes
through the same judge. `max_new_tokens=256` is the parent's budget, kept identical: a
judge comparing meaning scores a truncated answer as a wrong one, and a control that was
cut off at a different length than the run it is the control for measures the truncation.

Usage (normally via `run_eval.py --dataset streambench --blind`):
    python video_qa/blind_stream_vqa.py --model llava_ov_7b \
        --anno_path data/streambench/full_oe.json \
        --save_dir results/llava_ov_7b/streambench/64-1.0-blind --sample_fps 1
"""

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_streambench_vqa import ReKVStreamBenchVQA

# Tells the backend to retrieve nothing rather than to search for the best blocks. One
# inner list per batch unit; ReKV runs batch_size 1 throughout.
NO_BLOCKS = [[]]


class BlindStreamVQA(ReKVStreamBenchVQA):
    """Inherits the open-ended prompt and the record layout; overrides only the two places
    that would otherwise touch the video."""

    def video_open_qa(self, question, max_new_tokens=256):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question),
        }
        pred_answer = self.qa_model.question_answering(
            input_text, max_new_tokens=max_new_tokens, retrieved_indices=NO_BLOCKS)
        return {'pred_answer': pred_answer.replace('\n', '')}

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # No stream is opened and no frame is decoded -- not even the container header,
        # which the sighted solver reads to cap the stream at the last breakpoint. The
        # cache is reset per video anyway, so a blind row cannot inherit state from the
        # video before it: the run is structurally identical to the sighted one, minus
        # ingestion.
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        last_realtime = -1.0
        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            realtime = float(sample['end_time'])
            # Asserted for the same reason as in the sighted solver: the two CSVs are
            # compared row by row, so they must walk the conversation in the same order.
            # One video in streaming_bench_v0.3.json ships unsorted.
            assert realtime >= last_realtime, (
                f"{video_sample['video_id']}: conversations must be sorted by timestamp, "
                f'got {realtime} after {last_realtime}')
            last_realtime = realtime

            qa_results = self.video_open_qa(sample['question'], max_new_tokens=256)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question_id': sample['question_id'],
                'task': sample['question_type'],
                'source': sample.get('source'),
                'question': sample['question'],
                # eval_open_ended_local.py reads exactly question/answer/pred_answer, so
                # these three names are load-bearing.
                'answer': sample['answer'],
                'pred_answer': qa_results['pred_answer'],
                'realtime': realtime,
                # Carried so the blind CSV joins to its sighted twin and so the leak check
                # has the columns it requires. Zero frames trivially satisfies the
                # invariant -- the check passes rather than being skipped.
                'n_frames_seen': 0,
                'n_frames_loaded': 0,
                'sample_fps': self.sample_fps,
                'truncated': False,
                # Makes the CSV self-identifying: a blind run left in a mislabelled
                # directory can still never be read as a real score.
                'blind': True,
            })


if __name__ == '__main__':
    work(BlindStreamVQA)
