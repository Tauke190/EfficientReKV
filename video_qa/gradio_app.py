"""Interactive Gradio app: stream a video into ReKV and ask questions as it plays.

A background thread feeds frames into the model one at a time (mirroring the
encode/QA interleave of video_qa/rekv_stream_vqa.py) while the chat box
answers questions against whatever has been ingested *so far*. Ask about
something that has not streamed in yet and the model genuinely cannot answer --
that is the point of the demo.

This works because question-answering is non-destructive to the video KV-Cache:
the retrieval branch sets `updata_kv_cache = False`
(model/attention/rekv_attention.py:71) and prefill/decode run on a throwaway
tuple, so `self.kv_cache` is never written to by a question. Encoding therefore
resumes unchanged after any number of questions.

Constraints this app has to respect (all verified against the source):
  - n_local >= the tokens fed in one pass (model/abstract_rekv.py, _forward_features).
    Frames are encoded one per pass, so this is just n_local >= 196.
  - clear_cache() must be followed by encode_init_prompt()
    (model/abstract_rekv.py:17-27); skipping the latter silently degrades
    answers rather than erroring.
  - The model is NOT thread-safe: GLOBAL_STREAM is process-wide
    (model/attention/kv_cache_manager.py:186) and `to_retrieve` is a mode flag
    on shared per-layer objects. A question firing during encode_frame would
    flip that call into the retrieval branch and silently drop those frames.
    Every model call therefore takes MODEL_LOCK.
  - reset_retrieval() is not in a try/finally upstream
    (model/llava_onevision_rekv.py:60-61), so a generation error would strand
    the cache in retrieval mode and stop ingestion. ask() forces the reset.
  - A question before the first frame trips an assert
    (model/attention/kv_cache_manager.py:432), so QA is refused until then.
  - Questions are independent; the model keeps no chat history.

Usage:
    python video_qa/gradio_app.py --share
"""

import os
import json
import time
import random
import argparse
import threading

import torch
import gradio as gr
import logging
import logzero
from logzero import logger

from video_qa.base import MODELS
from video_qa.frame_cache import open_frame_stream

# The encode path logs cache size at DEBUG per frame; far too chatty for a
# server that encodes continuously.
logzero.loglevel(logging.INFO)


# Serialises every call into the model. See module docstring.
# Reentrant: ask_stream() holds it across its yields, and the status readout it
# yields alongside each token would otherwise deadlock re-acquiring it.
MODEL_LOCK = threading.RLock()

MODEL = None
ANNO = None
ARGS = None


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

class StreamSession:
    """Holds the decoded video plus how much of it the model has ingested."""

    def __init__(self):
        self.video_sample = None
        self.frames = None          # FrameStream: one frame per read, never the whole clip
        self.playhead = 0           # next frame index not yet encoded
        self.running = False
        self.encode_seconds = 0.0
        self.capped = False
        # Wall-clock origin of the current play span, and the playhead at that
        # moment. The target is derived from these rather than accumulated per
        # tick, so a slow tick catches up instead of falling behind forever.
        self.play_started_at = None
        self.frames_at_play = 0

    @property
    def total_frames(self):
        return 0 if self.frames is None else len(self.frames)

    @property
    def video_seconds(self):
        """Seconds of video content ingested so far."""
        return self.playhead / ARGS.sample_fps if ARGS.sample_fps else 0.0

    @property
    def encode_fps(self):
        return self.playhead / self.encode_seconds if self.encode_seconds > 0 else 0.0

    def target_playhead(self):
        """Frame the browser's player should be at right now.

        The player runs at 1x in the browser; matching it means ingesting
        `sample_fps` frames per elapsed second from the moment play started.
        """
        if not self.running or self.play_started_at is None:
            return self.playhead
        elapsed = time.time() - self.play_started_at
        ahead = int(elapsed * ARGS.sample_fps * ARGS.speed)
        return min(self.frames_at_play + ahead, self.total_frames, ARGS.max_frames)

    def begin_play_span(self):
        self.play_started_at = time.time()
        self.frames_at_play = self.playhead


SESSION = StreamSession()


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------

def open_video(video_path, sample_fps, max_frames=None):
    """Open the clip as a frame stream -- nothing is decoded until the playhead asks.

    The demo's whole claim is that the model has seen only what has streamed in, so it
    must not hold the rest of the clip either. `max_frames` still caps the stream: the
    test set contains videos over two hours long and the KV-Cache, not the frames, is
    what bounds this app.
    """
    full = open_frame_stream(video_path, sample_fps=sample_fps,
                             cache_dir=os.environ.get('REKV_FRAME_CACHE'))
    truncated = max_frames is not None and len(full) > max_frames
    if not truncated:
        return full, False
    return open_frame_stream(video_path, sample_fps=sample_fps,
                             cache_dir=os.environ.get('REKV_FRAME_CACHE'),
                             num_frames=max_frames), True


def reset_cache():
    """clear_cache() + encode_init_prompt(), which must always go together."""
    with MODEL_LOCK:
        MODEL.clear_cache()
        MODEL.encode_init_prompt()


def kv_ram_gb():
    if MODEL is None or MODEL.kv_cache is None:
        return 0.0
    with MODEL_LOCK:
        try:
            return MODEL.calc_memory_usage() / (1024 ** 3)
        except Exception:
            return 0.0


@torch.inference_mode()
def encode_slice(start, end):
    """Ingest the frames that have arrived since the last tick, one forward pass each.

    One frame per pass because that is what a live stream can offer: at the playhead's
    rate the next frame does not exist yet when this one is encoded. `encode_video` would
    also flush its partial block at the end of every slice, padding a block per tick and
    undoing stage-2 pruning; `encode_frame` leaves the tail pending and
    `question_answering` flushes it when a question actually arrives.

    Caller must NOT already hold MODEL_LOCK.
    """
    t0 = time.perf_counter()
    with MODEL_LOCK:
        for i in range(start, end):
            MODEL.encode_frame(SESSION.frames[i])
        torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.inference_mode()
def ask(question, max_new_tokens=128):
    """Answer against the frames ingested so far.

    Wraps the call so `to_retrieve` is always cleared -- upstream only resets it
    on the happy path, and a stranded flag would silently stop frame ingestion.
    """
    input_text = {"question": question, "prompt": MODEL.get_prompt(question)}
    with MODEL_LOCK:
        try:
            return MODEL.question_answering(input_text, max_new_tokens=max_new_tokens)
        finally:
            for layer_kv in MODEL.kv_cache:
                layer_kv.reset_retrieval()


@torch.inference_mode()
def ask_stream(question, max_new_tokens=128):
    """Yield the answer token by token.

    Mirrors model/llava_onevision_rekv.py:37-106, but yields partial text
    instead of returning only at the end. Total time is unchanged -- decoding
    costs ~35 ms/token because every step re-concatenates the retrieved cache
    (model/attention/rekv_attention.py:78-90) -- but the first token lands in
    well under a second, so the answer reads as it is written.
    """
    device = MODEL.device
    tokenizer = MODEL.processor.tokenizer
    stop_token_ids = [tokenizer.eos_token_id]
    prompt = MODEL.get_prompt(question)

    with MODEL_LOCK:
        try:
            # Retrieval pass: the bare question selects the KV blocks. Must not
            # include the chat template, or retrieval is diluted by boilerplate.
            q_ids = torch.as_tensor([tokenizer(question).input_ids], device=device)
            for layer_kv in MODEL.kv_cache:
                layer_kv.set_retrieval()
            out = MODEL.language_model(input_ids=q_ids, use_cache=True,
                                       past_key_values=MODEL.kv_cache)
            past_key_values = out.past_key_values
            for layer_kv in MODEL.kv_cache:
                layer_kv.reset_retrieval()

            output_ids = []
            token = None
            for i in range(max_new_tokens):
                if i == 0:  # prefill the templated prompt
                    p_ids = torch.as_tensor([tokenizer(prompt).input_ids], device=device)
                    embeds = MODEL.get_input_embeddings()(p_ids)
                    out = MODEL.language_model(inputs_embeds=embeds, use_cache=True,
                                               past_key_values=past_key_values)
                else:
                    out = MODEL.language_model(
                        input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values

                token = int(torch.argmax(out.logits[0, -1, :]))
                if token in stop_token_ids:
                    break
                output_ids.append(token)
                yield tokenizer.decode(
                    output_ids, skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True)
        finally:
            # Never leave the cache stranded in retrieval mode; that would
            # silently stop frame ingestion.
            for layer_kv in MODEL.kv_cache:
                layer_kv.reset_retrieval()


# --------------------------------------------------------------------------
# Streaming loop
# --------------------------------------------------------------------------

def stream_worker():
    """Keep the ingested playhead level with the browser player's position.

    The player is the clock: it runs at 1x, and `target_playhead()` says which
    frame it is showing now. This loop closes the gap. Encoding measured ~19 FPS
    against a 0.5 fps requirement, so it has ~38x headroom and normally sits
    idle waiting for the next frame to "arrive".
    """
    while True:
        time.sleep(ARGS.tick_seconds)

        if not SESSION.running or SESSION.frames is None:
            continue
        if SESSION.playhead >= SESSION.total_frames:
            SESSION.running = False
            logger.info("Reached end of video.")
            continue
        if SESSION.playhead >= ARGS.max_frames:
            SESSION.running = False
            SESSION.capped = True
            logger.warning(f"Hit --max_frames cap ({ARGS.max_frames}); stream paused.")
            continue

        target = SESSION.target_playhead()
        # Bound work per tick so a large gap (e.g. after a long QA pause) is
        # closed over several ticks and the lock stays available for questions.
        budget = ARGS.catchup_budget

        while SESSION.running and SESSION.playhead < target and budget > 0:
            # Cap a single lock acquisition to --ingest_step frames. A question
            # arriving mid-catch-up waits at most that many frames, not the whole
            # backlog. Frames are still encoded one per forward pass inside.
            n = min(ARGS.ingest_step, target - SESSION.playhead, budget)
            try:
                SESSION.encode_seconds += encode_slice(SESSION.playhead,
                                                       SESSION.playhead + n)
                SESSION.playhead += n
                budget -= n
            except Exception as e:
                SESSION.running = False
                logger.error(f"Encoding failed, stream stopped: {e}")
                break


# --------------------------------------------------------------------------
# UI handlers
# --------------------------------------------------------------------------

def status_text():
    if SESSION.frames is None:
        return "No video loaded."

    pct = 100.0 * SESSION.playhead / max(1, SESSION.total_frames)
    state = "watching" if SESSION.running else "paused"
    if SESSION.playhead >= SESSION.total_frames:
        state = "finished"
    if SESSION.capped:
        state = "PAUSED - hit frame cap"

    # How far ingestion trails the player, if at all.
    lag = ""
    if SESSION.running:
        behind = (SESSION.target_playhead() - SESSION.playhead) / max(ARGS.sample_fps, 1e-6)
        if behind > 1.0:
            lag = f" &nbsp;|&nbsp; _catching up {behind:.0f}s_"

    return (
        f"**{state}** &nbsp;|&nbsp; "
        f"seen {SESSION.video_seconds:.0f}s of video "
        f"({SESSION.playhead}/{SESSION.total_frames} frames, {pct:.0f}%) &nbsp;|&nbsp; "
        f"KV RAM {kv_ram_gb():.2f} GB &nbsp;|&nbsp; "
        f"encode {SESSION.encode_fps:.1f} FPS{lag}"
    )


def pick_random_video():
    """Load a random clip from the annotation file and reset all state."""
    sample = random.choice(ANNO)
    SESSION.running = False
    time.sleep(ARGS.tick_seconds * 2)  # let the worker settle before we swap state

    logger.info(f"Loading {sample['video_id']} ({sample['duration']/60:.1f} min)")
    frames, truncated = open_video(sample['video_path'], ARGS.sample_fps, ARGS.max_frames)

    SESSION.video_sample = sample
    SESSION.frames = frames
    SESSION.playhead = 0
    SESSION.encode_seconds = 0.0
    SESSION.capped = False
    reset_cache()

    gt = [c['question'] for c in sample['conversations']]
    info = (
        f"### {sample['video_id']}\n"
        f"{sample['duration']/60:.1f} min &nbsp;|&nbsp; {len(frames)} frames "
        f"at {ARGS.sample_fps} fps &nbsp;|&nbsp; {len(gt)} ground-truth questions"
    )
    if truncated:
        info += (f"\n\n_Streaming only the first {ARGS.max_frames} frames "
                 f"(`--max_frames`) of this {sample['duration']/60:.0f} min video._")
    return (
        sample['video_path'],
        info,
        status_text(),
        gr.update(samples=[[q] for q in gt]),
        [],
    )


def start_stream():
    """Fired by the video player's `play` event (or the manual button)."""
    if SESSION.frames is None:
        return status_text()
    if SESSION.playhead >= SESSION.total_frames:
        return status_text()
    SESSION.capped = False
    SESSION.begin_play_span()
    SESSION.running = True
    return status_text()


def pause_stream():
    """Fired by the player's `pause`/`end` events (or the manual button)."""
    SESSION.running = False
    SESSION.play_started_at = None
    return status_text()


def reset_stream():
    """Wipe the cache and rewind, keeping the same video loaded."""
    SESSION.running = False
    SESSION.play_started_at = None
    time.sleep(ARGS.tick_seconds * 2)
    SESSION.playhead = 0
    SESSION.encode_seconds = 0.0
    SESSION.capped = False
    reset_cache()
    return status_text(), []


def on_ask(question, history):
    history = history or []
    question = (question or "").strip()
    # This is a generator (it streams tokens), so guards must yield, not return.
    if not question:
        yield history, "", status_text()
        return

    if SESSION.frames is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant",
                        "content": "_Load a video first._"})
        yield history, "", status_text()
        return

    if SESSION.playhead < 1:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant",
                        "content": "_Nothing watched yet - press play and give it a moment._"})
        yield history, "", status_text()
        return

    seen = SESSION.video_seconds
    frames_seen = SESSION.playhead
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": ""})

    # Snapshot the status once: ingestion is blocked while generating anyway, so
    # it cannot change, and recomputing it per token would re-read the cache size
    # on every single step.
    frozen_status = status_text()

    t0 = time.perf_counter()
    answer = ""
    try:
        for partial in ask_stream(question, max_new_tokens=ARGS.max_new_tokens):
            answer = partial
            history[-1]["content"] = answer
            yield history, "", frozen_status
    except Exception as e:
        answer = f"**Error:** {e}"

    latency = time.perf_counter() - t0
    history[-1]["content"] = (
        f"{answer}\n\n<sub>answered from the first {seen:.0f}s "
        f"({frames_seen} frames) &middot; {latency:.2f}s</sub>"
    )
    yield history, "", status_text()


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="ReKV live video QA") as demo:
        gr.Markdown(
            "# ReKV - streaming video question answering\n"
            "**The player drives the model.** Frames are ingested only as they play, so "
            "answers reflect exactly what has been watched so far - pause the video and the "
            "model stops seeing. Ask about something that has not played yet and it cannot "
            "answer; ask again after it plays and it can."
        )

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video(label="Source video", interactive=False, autoplay=True)
                info = gr.Markdown("_No video loaded._")
                gr.Markdown("<sub>Play/pause the video above to control ingestion.</sub>")
                with gr.Row():
                    btn_new = gr.Button("New random video", variant="secondary")
                    btn_start = gr.Button("Resume")
                    btn_pause = gr.Button("Pause")
                    btn_reset = gr.Button("Reset cache")

            with gr.Column(scale=4):
                status = gr.Markdown("No video loaded.")
                # gradio 6 dropped the `type` arg; messages format is the only one.
                chat = gr.Chatbot(label="Questions", height=420)
                question = gr.Textbox(
                    label="Ask about what has streamed so far",
                    placeholder="e.g. what is the person doing?",
                    lines=2,
                )
                btn_ask = gr.Button("Ask", variant="primary")
                gt_questions = gr.Dataset(
                    components=[gr.Textbox(visible=False)],
                    samples=[],
                    label="Ground-truth questions for this clip (click to fill)",
                )

        # Wiring. concurrency_limit=1 everywhere that touches the model.
        btn_new.click(pick_random_video, None,
                      [video, info, status, gt_questions, chat], concurrency_limit=1)
        btn_start.click(start_stream, None, status, concurrency_limit=1)
        btn_pause.click(pause_stream, None, status, concurrency_limit=1)
        btn_reset.click(reset_stream, None, [status, chat], concurrency_limit=1)

        # The player is the clock: hitting play/pause in it drives ingestion, so
        # the model only ever knows what has actually been watched.
        video.play(start_stream, None, status, concurrency_limit=1)
        video.pause(pause_stream, None, status, concurrency_limit=1)
        video.end(pause_stream, None, status, concurrency_limit=1)

        btn_ask.click(on_ask, [question, chat], [chat, question, status], concurrency_limit=1)
        question.submit(on_ask, [question, chat], [chat, question, status], concurrency_limit=1)
        gt_questions.click(lambda s: s[0], gt_questions, question)

        # Live status without touching the model beyond a locked RAM read.
        timer = gr.Timer(1.0)
        timer.tick(status_text, None, status)

    return demo


def main():
    global MODEL, ANNO, ARGS

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llava_ov_0.5b")
    parser.add_argument("--anno_path", type=str, default="data/mlvu/test_mc.json")
    parser.add_argument("--sample_fps", type=float, default=0.5)
    parser.add_argument("--n_local", type=int, default=15000)
    parser.add_argument("--retrieve_size", type=int, default=64)
    parser.add_argument("--retrieve_chunk_size", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=2000,
                        help="Ingestion stops here. The KV-Cache is never freed by the "
                             "library (~2.3 MB/frame for the 0.5b model), so this is the "
                             "only thing bounding RAM.")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Answer length cap. Latency is ~35 ms/token and almost entirely "
                             "generation, so this is the main latency dial: 128 tokens is "
                             "~4.5s worst case, 256 is ~9s.")
    parser.add_argument("--ingest_step", type=int, default=16,
                        help="Max frames per lock acquisition while catching up. Each is still "
                             "encoded one per forward pass; this only bounds how long a "
                             "question waits behind the encoder.")
    parser.add_argument("--catchup_budget", type=int, default=128,
                        help="Max frames ingested per tick. Bounds how fast a backlog (e.g. "
                             "after a long QA pause) is closed, so catch-up cannot starve "
                             "questions of the lock.")
    parser.add_argument("--tick_seconds", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Ingestion rate multiplier. Keep at 1.0 to stay level with the "
                             "browser player, which plays at 1x. Raising it makes the model "
                             "run AHEAD of what is on screen.")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Public link -- needed on a compute node without a tunnel.")
    ARGS = parser.parse_args()

    # One frame per forward pass means the only thing n_local has to cover is a single
    # frame's tokens, which is what `_forward_features` asserts.
    if ARGS.n_local < 196:
        raise SystemExit(
            f"--n_local {ARGS.n_local} is below one frame's 196 tokens "
            f"(abstract_rekv.py asserts n_local >= the tokens fed in one pass)."
        )

    ANNO = json.load(open(ARGS.anno_path))
    ANNO = [v for v in ANNO if os.path.exists(v['video_path'])]
    logger.info(f"{len(ANNO)} videos available from {ARGS.anno_path}")
    if not ANNO:
        raise SystemExit("No videos from the annotation file exist on disk.")

    model_path = MODELS[ARGS.model]['model_path']
    logger.info(f"Loading {model_path}")
    MODEL, _ = MODELS[ARGS.model]['load_func'](
        model_path=model_path,
        n_local=ARGS.n_local,
        topk=ARGS.retrieve_size,
        chunk_size=ARGS.retrieve_chunk_size,
    )
    reset_cache()

    threading.Thread(target=stream_worker, daemon=True).start()

    # The videos sit outside the CWD (typically ~/.cache/huggingface/mlvu), and
    # gradio refuses to serve files from unlisted directories.
    allowed = sorted({os.path.dirname(os.path.abspath(v['video_path'])) for v in ANNO})
    logger.info(f"Serving videos from: {allowed}")

    build_ui().queue().launch(
        server_name="0.0.0.0",
        server_port=ARGS.server_port,
        share=ARGS.share,
        allowed_paths=allowed,
    )


if __name__ == "__main__":
    main()
