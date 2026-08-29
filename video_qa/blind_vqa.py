"""Blind control: answer every question with no video at all.

The number this produces is the floor that any real score has to be read against -- how
much of an accuracy is the language prior and the benchmark's own answer distribution,
before the model has seen a single frame. On ODV-Bench that floor is not a formality:
2472 of 6348 questions are binary, all 123 Hallucination-detection answers are the same
string ("Unable to say."), 73.4% of Risk Prediction is "No" and 67.9% of Action Prediction
is "Stopped". Four subtasks totalling 3234 questions have a majority answer well above
their own chance line, and those are exactly the strings a language model guesses unaided.

**What "blind" means here.** ReKV never passes a video at question time -- frames are
prefilled into the KV-Cache beforehand, and `question_answering` retrieves over that
cache. So this solver does not suppress an input; it skips ingestion. `clear_cache()` and
`encode_init_prompt()` run exactly as in a sighted run, leaving the 13 system-prompt
tokens in the cache, and no frame is ever decoded or handed to the vision tower. The
question then fires against that cache alone.

**Why not a blank/grey stream instead.** A constant stream still costs a full encode pass
and still puts 196 tokens per frame of "this is a grey rectangle" into memory -- real
evidence, merely uninformative. Worse for this repo: `model/token_pruning.py` keeps a
token iff its distance from the carried reference exceeds `threshold`, so a stream that
never changes has distance 0 everywhere and *nothing* survives past the first frame at any
threshold. A blank-stream baseline therefore cannot be run under the reduction stages this
codebase exists to measure. No-input has neither problem and costs nothing.

**The one implementation subtlety.** Simply not ingesting is not enough. Internal
retrieval scores the stored frame-blocks and picks the top-k; with an empty cache it trips
`assert self.global_remainder[0].size(-2) > self.n_init` in
model/attention/kv_cache_manager.py -- the code has no blocks to search. Passing
`retrieved_indices=[[]]` routes through the *external* retrieval branch instead, which
skips the search rather than performing an empty one. `frames_to_blocks` already
early-returns `[]` on an empty cache, and `get_retrieved_kv` then loads the init KV and
iterates an empty block list. No model-side change is needed.

Retrieval mode is still entered, so the question and its generated answer are never
written back into the cache (model/attention/rekv_attention.py sets
`updata_kv_cache = False`). Questions on the same video stay independent, as in a sighted
run.

Rows are written with the same columns as the sighted solver, keyed by the same
(video_id, question_id), so the two CSVs join directly. `n_frames_seen` is 0, which
satisfies the leak check in video_qa/eval/eval_odvbench.py rather than bypassing it.

Reads `gt_index` when the annotation has one and falls back to `choices.index(answer)`
otherwise, so this is not specific to ODV-Bench -- wiring it to another multiple-choice
dataset is a one-line change in run_eval.py.

Usage (normally via `run_eval.py --dataset odvbench --blind`):
    python video_qa/blind_vqa.py --model llava_ov_0.5b --anno_path data/odvbench/full_mc.json \
        --save_dir results/llava_ov_0.5b/odvbench/64-2.0-blind --sample_fps 2
"""

import torch
from logzero import logger

from video_qa.base import work
from video_qa.rekv_ovobench_vqa import ReKVOVOBenchVQA

# Tells the backend to retrieve nothing rather than to search for the best blocks. One
# inner list per batch unit; ReKV runs batch_size 1 throughout.
NO_BLOCKS = [[]]


class BlindVQA(ReKVOVOBenchVQA):
    """Inherits the MCQA prompt, letter extraction and record layout; overrides only the
    two places that would otherwise touch the video."""

    def video_close_qa(self, question, candidates, correct_choice):
        input_text = self.format_mcqa_prompt(question, candidates)
        pred_answer = self.qa_model.question_answering(
            input_text, max_new_tokens=16, retrieved_indices=NO_BLOCKS)
        pred_letter = self.extract_characters_regex(pred_answer)
        return {
            'pred_answer': pred_answer.replace('\n', ''),
            'pred_choice': pred_letter,
            'acc': float(pred_letter == correct_choice),
        }

    def correct_letter(self, sample):
        """Gold letter, preferring an explicit `gt_index`.

        OVO-Bench needs the index because its `answer` is often a paraphrase of the option
        rather than a copy; ODV-Bench's answers are verbatim, so either route works there.
        The fallback is what lets this solver run against a plain multiple-choice
        annotation that has no gt_index.
        """
        if 'gt_index' in sample:
            return self.choice_letters[int(sample['gt_index'])]
        return self.choice_letters[sample['choices'].index(sample['answer'])]

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        # No stream is opened and no frame is decoded. The cache is reset per video
        # anyway, so a blind row cannot inherit state from the video before it -- the run
        # is structurally identical to the sighted one, minus ingestion.
        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            choices = sample['choices']
            correct_choice = self.correct_letter(sample)
            qa_results = self.video_close_qa(sample['question'], choices, correct_choice)

            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question_id': sample.get('question_id'),
                'task': sample.get('question_type'),
                'question': sample['question'],
                'choices': choices,
                'answer': sample['answer'],
                'correct_choice': correct_choice,
                'pred_answer': qa_results['pred_answer'],
                'pred_choice': qa_results['pred_choice'],
                'qa_acc': qa_results['acc'] * 100,
                # Carried so a blind CSV joins to its sighted twin and so the leak check
                # has the columns it requires. Zero frames trivially satisfies the
                # invariant -- the check passes rather than being skipped.
                'realtime': sample.get('end_time'),
                'n_frames_seen': 0,
                'n_frames_loaded': 0,
                'sample_fps': self.sample_fps,
                'truncated': False,
                # Makes the CSV self-identifying: a blind run left in a mislabelled
                # directory can still never be read as a real score.
                'blind': True,
            })


if __name__ == '__main__':
    work(BlindVQA)
