"""Swappable local LLM judges for open-ended scoring.

`eval_open_ended_local.py` used to hard-code one prompt (the GPT-3.5 "reply with a
Python dict" prompt inherited from the API scorer) and one model (Qwen2.5-32B-Instruct).
That prompt is not a neutral harness: it assumes a general instruct model that will
follow a formatting instruction. Prometheus 2 is a *dedicated* evaluator model -- it is
trained to emit feedback followed by `[RESULT] <1-5>` against an explicit rubric, and
scores noticeably worse when forced into someone else's output format.

So a judge here is a pair -- prompt style + parser -- not just a checkpoint name. This
module holds one class per pair, and `resolve()` picks one:

    judge = resolve('prometheus')            # style default checkpoint
    judge = resolve('qwen', model='Qwen/Qwen2.5-7B-Instruct')
    judge = resolve('auto',  model='prometheus-eval/prometheus-8x7b-v2.0')

Everything downstream of `judge.messages()` / `judge.parse()` is style-agnostic, so
adding a third judge means adding a class and a registry entry, nothing else.

## What the verdict means

Both styles return the same dict -- `{'pred': 'yes'|'no', 'score': int}` -- because the
aggregation, the answer_type breakdown and `blind/compare_blind.py` all read those two
fields. But they are NOT on the same scale:

  * qwen       -- yes/no is the judge's own call; score is 0-5, also the judge's call.
  * prometheus -- the model emits only a 1-5 rubric score. yes/no is *derived* by
                  thresholding it (`yes_threshold`, default 4), because Prometheus has no
                  binary head to ask.

Accuracy from one judge is therefore not comparable to accuracy from the other, and mean
score is even less so (different floors, different rubrics). Compare arms scored by the
same judge; that is what a pruning sweep needs. The judge id is stamped into every cache
file and into `metrics.judge`, so a mixed directory is detectable rather than silently
averaged.
"""

import os
import re


class JudgeStyle:
    """Prompt + parser for one family of judge models.

    Subclasses set the class attributes and implement `messages` / `parse`.
    """

    name = None
    #: Checkpoint used when the caller names a style but no model.
    default_model = None
    #: Generation budget. A style that must write prose before its verdict needs more.
    default_max_new_tokens = 64
    #: Human-readable scale of `score`, for the printed summary only.
    score_range = '0-5'

    def __init__(self, model=None):
        self.model = model or self.default_model

    @property
    def id(self):
        """What gets stamped into caches -- style and checkpoint both matter."""
        return f'{self.name}:{self.model}'

    def messages(self, question, answer, pred):
        """Chat messages for one QA item."""
        raise NotImplementedError

    def parse(self, text):
        """Completion -> {'pred': 'yes'|'no', 'score': int}, or None if unusable.

        None means *missing data*, never "no": coercing an unparsable completion to a
        wrong answer deflates accuracy by exactly the judge's own failure rate, which is
        not a property of the model under test.
        """
        raise NotImplementedError


class QwenJudge(JudgeStyle):
    """General instruct model asked to reply with a `{'pred': ..., 'score': ...}` dict.

    The prompt is copied verbatim from `eval_open_ended.py` (the OpenAI scorer) and must
    stay byte-identical to it: judge-to-judge gaps already dominate the noise in these
    numbers, and a reworded prompt would add a second, unmeasurable one. Works with any
    chat model, not just Qwen -- the name is the family it was tuned against here.
    """

    name = 'qwen'
    default_model = 'Qwen/Qwen2.5-32B-Instruct'
    default_max_new_tokens = 64
    score_range = '0-5'

    SYSTEM = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )

    USER = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        "Question: {question}\n"
        "Correct Answer: {answer}\n"
        "Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
    )

    def messages(self, question, answer, pred):
        return [
            {'role': 'system', 'content': self.SYSTEM},
            {'role': 'user', 'content': self.USER.format(question=question, answer=answer, pred=pred)},
        ]

    def parse(self, text):
        import ast

        # Preferred path: the model emitted the dict it was asked for.
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                d = ast.literal_eval(match.group(0))
                if isinstance(d, dict) and 'pred' in d and 'score' in d:
                    pred = str(d['pred']).strip().lower()
                    if 'yes' in pred or 'no' in pred:
                        return {'pred': 'yes' if 'yes' in pred else 'no',
                                'score': int(round(float(d['score'])))}
            except (ValueError, SyntaxError, TypeError):
                pass

        # Fallback: the fields are in there, just not as a dict literal.
        pred_m = re.search(r"['\"]?pred['\"]?\s*[:=]\s*['\"]?(yes|no)\b", text, re.IGNORECASE)
        score_m = re.search(r"['\"]?score['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if pred_m and score_m:
            return {'pred': pred_m.group(1).lower(),
                    'score': int(round(float(score_m.group(1))))}
        return None


class PrometheusJudge(JudgeStyle):
    """Prometheus 2 (prometheus-eval/*-v2.0) in absolute-grading mode.

    Prometheus is trained on exactly one input layout -- task description, instruction,
    response, reference answer, 1-5 rubric -- and one output layout, `Feedback: ...
    [RESULT] n`. The templates below are the released ones; the only thing written for
    this repo is the rubric, which states the criterion the QA task actually has
    (does the prediction mean the same as the reference).

    Two consequences for the caller:

      * It writes feedback *before* the score, so it needs a real token budget
        (`default_max_new_tokens`). Truncate it and the `[RESULT]` never arrives and
        every item comes back unparsed.
      * It emits no yes/no. `yes_threshold` turns the rubric score into the binary the
        accuracy metric needs; 4 is the first rubric level that describes a match, so
        that is the default. Change it only with the rubric text in front of you.

    The v2.0 checkpoints are Mistral-based and their chat template rejects a system
    role, which is why the system text is folded into the user turn here -- that is also
    what the official prometheus-eval client does, so the model sees what it was tuned
    on.
    """

    name = 'prometheus'
    default_model = 'prometheus-eval/prometheus-7b-v2.0'
    # Feedback is a paragraph, then the score. 64 tokens would cut it off before [RESULT].
    default_max_new_tokens = 512
    score_range = '1-5'

    SYSTEM = (
        "You are a fair judge assistant tasked with providing clear, objective feedback based "
        "on specific criteria, ensuring each assessment reflects the absolute standards set "
        "for performance."
    )

    RUBRIC = (
        "[Does the response correctly answer the question, conveying the same meaning as the "
        "reference answer? Synonyms, paraphrases and differences in wording or level of detail "
        "are acceptable; only the information content matters.]\n"
        "Score 1: The response is wrong, contradicts the reference answer, or does not answer "
        "the question at all.\n"
        "Score 2: The response is mostly wrong, sharing only incidental details with the "
        "reference answer.\n"
        "Score 3: The response is partially correct: it gets some of the reference answer right "
        "but misses or misstates a substantive part of it.\n"
        "Score 4: The response conveys the same meaning as the reference answer, with minor "
        "omissions or imprecision that do not change the answer.\n"
        "Score 5: The response fully conveys the meaning of the reference answer."
    )

    TEMPLATE = (
        "###Task Description:\n"
        "An instruction (might include an Input inside it), a response to evaluate, a reference "
        "answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.\n"
        "1. Write a detailed feedback that assess the quality of the response strictly based on "
        "the given score rubric, not evaluating in general.\n"
        "2. After writing a feedback, write a score that is an integer between 1 and 5. You should "
        "refer to the score rubric.\n"
        "3. The output format should look as follows: \"Feedback: (write a feedback for criteria) "
        "[RESULT] (an integer number between 1 and 5)\"\n"
        "4. Please do not generate any other opening, closing, and explanations.\n\n"
        "###The instruction to evaluate:\n{question}\n\n"
        "###Response to evaluate:\n{pred}\n\n"
        "###Reference Answer (Score 5):\n{answer}\n\n"
        "###Score Rubrics:\n{rubric}\n\n"
        "###Feedback: "
    )

    def __init__(self, model=None, yes_threshold=4):
        super().__init__(model)
        self.yes_threshold = yes_threshold

    @property
    def id(self):
        return f'{self.name}:{self.model}:yes>={self.yes_threshold}'

    def messages(self, question, answer, pred):
        body = self.TEMPLATE.format(question=question, answer=answer, pred=pred, rubric=self.RUBRIC)
        # Single user turn: the v2.0 chat template has no system role.
        return [{'role': 'user', 'content': f'{self.SYSTEM}\n\n{body}'}]

    def parse(self, text):
        score = None
        # The format the model was trained to emit.
        m = re.search(r'\[RESULT\]\s*\(?\s*([1-5])', text)
        if m is None:
            # Seen occasionally when feedback runs long: the tag is dropped but the score
            # is still stated. Anything else is missing data.
            m = re.search(r'(?:^|\n)\s*(?:Score|Result)\s*[:=]\s*\(?\s*([1-5])\b', text, re.IGNORECASE)
        if m is not None:
            score = int(m.group(1))
        if score is None:
            return None
        return {'pred': 'yes' if score >= self.yes_threshold else 'no', 'score': score}


STYLES = {cls.name: cls for cls in (QwenJudge, PrometheusJudge)}

#: Substring of a checkpoint name -> style, for `--judge_style auto`.
_AUTODETECT = (
    ('prometheus', 'prometheus'),
)

DEFAULT_STYLE = 'prometheus'

#: Named judges, as (style, checkpoint, output suffix). One source of truth for both
#: entry points -- scripts/score_open_ended.sh resolves through `preset()` and
#: video_qa/run_eval.py imports this -- so a judge always writes to the same place
#: whichever one launched it.
#:
#: The suffix keeps judges from overwriting each other's verdicts inside one results dir.
#: It is per *preset*, not per style: prometheus-7b and prometheus-8x7b disagree often
#: enough that pooling them would be its own experiment. `qwen` keeps the empty suffix it
#: had when it was the only judge, so existing results_local.json / tmp_local/ caches stay
#: valid and blind/compare_blind.py keeps reading them.
PRESETS = {
    'prometheus':     ('prometheus', 'prometheus-eval/prometheus-7b-v2.0', '_prometheus'),
    'prometheus8x7b': ('prometheus', 'prometheus-eval/prometheus-8x7b-v2.0', '_prometheus8x7b'),
    'qwen':           ('qwen', 'Qwen/Qwen2.5-32B-Instruct', ''),
    'qwen7b':         ('qwen', 'Qwen/Qwen2.5-7B-Instruct', '_qwen7b'),
}

DEFAULT_PRESET = 'prometheus'


def preset(name):
    """(style, model, output_suffix) for a preset name, or for a bare checkpoint.

    Anything not in PRESETS is taken to be an HF id or local path: style 'auto', and a
    suffix slugged from the checkpoint name so two ad-hoc judges do not share a cache.
    """
    if name in PRESETS:
        return PRESETS[name]
    slug = re.sub(r'[^a-z0-9]+', '-', os.path.basename(name.rstrip('/')).lower()).strip('-')
    return 'auto', name, f'_{slug}'


def detect_style(model):
    """Style implied by a checkpoint name, or None if nothing matches.

    Only exact families are listed. Anything unrecognised falls back to the generic
    dict-prompt style, which is what an arbitrary instruct model can actually do.
    """
    if model:
        lowered = model.lower()
        for needle, style in _AUTODETECT:
            if needle in lowered:
                return style
    return None


def resolve(style='auto', model=None, yes_threshold=4):
    """Build a judge from a style name and an optional checkpoint.

    'auto' reads the style off the checkpoint name (so passing only
    `--judge_model prometheus-eval/prometheus-8x7b-v2.0` does the right thing) and falls
    back to DEFAULT_STYLE when no model is given.
    """
    if style in (None, 'auto'):
        style = detect_style(model) or (DEFAULT_STYLE if model is None else QwenJudge.name)
    if style not in STYLES:
        raise ValueError(f'unknown judge style {style!r}; known: {", ".join(sorted(STYLES))}')
    cls = STYLES[style]
    if cls is PrometheusJudge:
        return cls(model=model, yes_threshold=yes_threshold)
    return cls(model=model)


if __name__ == '__main__':
    # Preset lookup for shell callers: `python video_qa/eval/judges.py <preset-or-hf-id>`
    # prints style, checkpoint and output suffix as one tab-separated line, so
    # scripts/score_open_ended.sh does not carry a second copy of the table.
    import sys

    if len(sys.argv) != 2:
        sys.exit(f'usage: {sys.argv[0]} <preset|hf-id>   presets: {", ".join(PRESETS)}')
    print('\t'.join(preset(sys.argv[1])))
