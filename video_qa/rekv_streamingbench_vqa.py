"""StreamingBench solver: OVO-Bench's realtime loop, plus Sequential-QA's carried context.

StreamingBench's contract is the one video_qa/rekv_ovobench_vqa.py already implements --
a query at `time_stamp` t may only be answered from frames in [0, t] -- so the ingestion
loop, the frame accounting and the leak-audit columns are inherited verbatim rather than
copied. Confirmed against the reference runner (src/benchmark/StreamingBench.py), which
gets the same window by cutting a fresh clip `split_video(path, 0, timestamp)` per query:
same frames, 4250 encodes instead of 850.

The one behaviour that is genuinely StreamingBench's own is Sequential Question Answering.
Its five questions are a conversation -- "What colour was the jacket?" after "Who walked
in?" -- and the reference runner (src/benchmark/StreamingBenchSQA.py) prepends every
earlier question, its options, and its *gold* answer to the prompt. That is what
--sqa_context reproduces. Two consequences worth stating plainly:

* the context is built from the annotation, not from what the model predicted, so a wrong
  answer never poisons the later turns and the turns stay independently scorable;
* run SQA *without* it and you are measuring a harder, different task. The flag exists so
  that choice is explicit and recorded in the results directory name, not so it can be
  forgotten.

Not reproduced: the reference runner's `--context_time N`, which shows only the last N
seconds before the query (the paper's 60-second ablation). ReKV's cache holds the stream
from 0 by construction, which is that runner's `context_time 0` -- the main setting and
the one the leaderboard reports.
"""

import math

from video_qa.base import work
from video_qa.rekv_ovobench_vqa import ReKVOVOBenchVQA

# Mirrors src/benchmark/StreamingBenchSQA.py's wording, which is part of the task: the
# second sentence is what stops the model answering the *previous* question again.
CONTEXT_HEADER = ('Here are the contextual information related to the video. Please answer '
                  'the questions based on the contextual information: ')
CONTEXT_FOOTER = ('\n\nHere is the question. Answer it and don\'t confuse it with the '
                  'previous conversation.')


def hhmmss(seconds):
    """Back to the CSV's own 'HH:MM:SS' -- the reference runner quotes it into the prompt."""
    s = int(math.floor(float(seconds)))
    return f'{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}'


def sqa_contexts(conversations, letters):
    """One prompt prefix per turn: '' for the first, the earlier turns for the rest.

    Precomputable because the reference runner quotes the gold answer rather than the
    model's, so nothing here depends on generation -- which is also why the turns can
    still be scored independently.
    """
    prefixes, running = [], ''
    for sample in conversations:
        prefixes.append((running + CONTEXT_FOOTER) if running else '')
        if not running:
            running = CONTEXT_HEADER
        options = ', '.join(f'{letters[i]}. {c}' for i, c in enumerate(sample['choices']))
        running += (f"At timestamp {hhmmss(sample['realtime'])}, the following question and "
                    f"answer occurred: Question: {sample['question']}; Options: {options}; "
                    f"Answer: {letters[int(sample['gt_index'])]}; ")
    return prefixes


class ReKVStreamingBenchVQA(ReKVOVOBenchVQA):
    def __init__(self, *pos, args=None, **kw):
        super().__init__(*pos, **kw)
        self.sqa_context = bool(getattr(args, 'sqa_context', False))
        self._prefixes, self._turn = None, 0

    def analyze_a_video(self, video_sample):
        # Reset per video, before the inherited loop starts asking questions.
        self._prefixes = (sqa_contexts(video_sample['conversations'], self.choice_letters)
                          if self.sqa_context else None)
        self._turn = 0
        return super().analyze_a_video(video_sample)

    def format_mcqa_prompt(self, question, candidates):
        """The inherited prompt, with this turn's conversation history in front of it.

        Prepended to the *formatted* question rather than to the question text: the MC
        prompt's 'Question: ... Options: ... Only give the best option.' shape is what
        get_prompt(mc=True) primes 'Best option: (' against, and what
        extract_characters_regex parses back out. Sliding the history in ahead of it
        leaves both ends untouched.
        """
        out = super().format_mcqa_prompt(question, candidates)
        prefix = self._prefixes[self._turn] if self._prefixes else ''
        if prefix:
            formatted = f'{prefix}\n\n{out["formatted_question"]}'
            out = {'question': out['question'], 'formatted_question': formatted,
                   'prompt': self.qa_model.get_prompt(formatted, mc=True)}
        return out

    def video_close_qa(self, question, candidates, correct_choice):
        result = super().video_close_qa(question, candidates, correct_choice)
        self._turn += 1
        return result


def add_args(parser):
    parser.add_argument('--sqa_context', action='store_true',
                        help='Sequential Question Answering only: prepend the earlier '
                             "questions and their gold answers to each prompt, as "
                             'src/benchmark/StreamingBenchSQA.py does. Off for the other '
                             'three subsets, which have no carried context.')


if __name__ == '__main__':
    work(ReKVStreamingBenchVQA, add_args=add_args)
