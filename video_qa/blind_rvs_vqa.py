"""Blind control for RVS-Ego / RVS-Movie: answer every question with no video at all.

Same idea as video_qa/blind_stream_vqa.py, which is the StreamBench one, and it cannot be
reused here: it records `question_id` and `question_type`, and the RVS annotations
(data/rvs/{ego,movie}/*_oe.json) carry neither -- each conversation is only
question/answer/start_time/end_time. So this subclasses the rvs_* solver itself,
ReKVStreamVQA, and writes that solver's columns.

**What "blind" means here** is what it means in blind_vqa: ReKV never passes a video at
question time -- frames are prefilled into the KV-Cache beforehand -- so this does not
suppress an input, it skips ingestion. `clear_cache()` and `encode_init_prompt()` run as
in a sighted run, leaving the system-prompt tokens in the cache, and no frame is decoded
or handed to the vision tower. `retrieved_indices=[[]]` routes through the *external*
retrieval branch so the backend skips the block search instead of performing an empty one
against an empty cache (which trips an assert in model/attention/kv_cache_manager.py).

Rows keep the sighted solver's columns and order -- video_id/question/answer/pred_answer,
walked in annotation order -- so the blind CSV lines up with its sighted twin row by row
and goes through the same judge. `max_new_tokens=256` is the sighted budget, kept
identical: a judge comparing meaning scores a truncated answer as a wrong one. The extra
`blind` column makes the CSV self-identifying if it is ever moved out of its '-blind'
directory; the judges read only question/answer/pred_answer and ignore it.

Usage (normally via `run_eval.py --dataset rvs_movie --blind`):
    python video_qa/blind_rvs_vqa.py --model llava_ov_7b \
        --anno_path data/rvs/movie/movienet_oe.json \
        --save_dir results/llava_ov_7b/rvs_movie/64-1.0-blind --sample_fps 1
"""

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_stream_vqa import ReKVStreamVQA

# Tells the backend to retrieve nothing rather than to search for the best blocks. One
# inner list per batch unit; ReKV runs batch_size 1 throughout.
NO_BLOCKS = [[]]


class BlindRVSVQA(ReKVStreamVQA):
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
        # No stream is opened and no frame is decoded. The cache is still reset per video,
        # so the run is structurally identical to the sighted one, minus ingestion.
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            qa_results = self.video_open_qa(sample['question'], max_new_tokens=256)
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'pred_answer': qa_results['pred_answer'],
                'blind': True,
            })


if __name__ == '__main__':
    work(BlindRVSVQA)
