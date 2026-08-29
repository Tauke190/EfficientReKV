"""Streaming solver: frames arrive one at a time, questions are asked mid-stream.

`FrameStream.frames` yields frame t and only frame t; `encode_frame` preprocesses it,
runs the vision tower on it, and prefills it into the KV-Cache before frame t+1 is read.
That is the whole point of the scenario -- a live stream cannot batch frames it has not
received -- and it is what this solver used to get wrong: it handed the model 64 frames
per forward pass.

The model side already worked this way; only the driver did not. `encode_frame` is the
same `_encode_video_chunk` with a chunk of one, ContextManager.append already cuts its
input into one-frame blocks internally (model/attention/kv_cache_manager.py), and both
reduction stages already decide frame by frame off a reference carried across calls. So
the answers are unchanged to within GPU non-determinism; what changes is that no stage of
the *encode* path depends on a frame that has not arrived.

Decoding is ReKV's own: one `vr.get_batch(frame_idx)` over the sampled index list, in
process, at the start of the video (video_qa/base.py `load_video`, and upstream's
rekv_stream_vqa.py). There is no frame cache and no pre-extraction step. The sampled
frames therefore sit in RAM for the length of the video -- ~11 GB for a 1-hour 1080p video
at 0.5 FPS -- which is the reference implementation's cost and the reason `num_frames`
exists below.
"""


import torch
from logzero import logger

from video_qa.base import BaseVQA, open_video_reader, work


class FrameStream:
    """The sampled frames of a video, decoded ahead and handed over one at a time.

    The decode is upstream ReKV's -- `vr.get_batch(frame_idx)`, over the whole video by
    default -- so the frames and the decode cost are the reference implementation's. Only
    what happens afterwards differs: frames leave this object one at a time, so the encode
    path still never batches a frame with its neighbours and never encodes one past the
    question it is answering.

    The arrival grid is BaseVQA.load_video's, so a streaming run and an offline run of the
    same file at the same --sample_fps see the same frames:

    * default (stride): source frames 0, stride, 2*stride, ... with stride =
      round(avg_fps)/sample_fps, floored to an integer and at least 1;
    * `exact`: slot k is the source frame most recently shown at t = k / sample_fps, which
      is the only grid that actually delivers the requested rate -- the integer stride
      quantizes it (at native 30 fps, 16 and 32 both come out as 30).

    `num_frames` caps the stream where the caller knows the tail will never be looked at
    (OVO-Bench stops at its last query's timestamp). Nothing past it is decoded or held,
    which is what keeps a 2-hour source video affordable; `len()` reports the capped
    length, and `n_available` the length before the cap, so a caller can tell that it
    truncated something.

    `window` decodes in blocks of that many slots instead of all of them up front, holding
    one block at a time. The default (None) keeps the whole-video decode every previous run
    used, and is right while the decoded video is small. It stops being right as
    --sample_fps rises: the pixels are held at source resolution, so a 600 s
    FPS-Bench-Stream video is ~4.6 MB/frame x 600 x fps -- 2.7 GB at 1 fps but 85 GB at
    32 fps, per worker, on top of the KV-Cache. Frames arrive in order and are never read
    again, so a block is decoded exactly once either way; windowing only bounds what is
    resident. Random access still works, but a caller that jumps between blocks re-decodes
    on every jump.
    """

    def __init__(self, video_path, sample_fps, exact=False, num_frames=None, window=None):
        vr = open_video_reader(video_path)
        self.sample_fps = sample_fps
        n_src = len(vr)
        src_fps = round(vr.get_avg_fps())
        if exact:
            n_slots = max(1, int(round(n_src / src_fps * sample_fps)))
            self._index = [min(n_src - 1, int(t * src_fps / sample_fps))
                           for t in range(n_slots)]
        else:
            # Clamped: asking for more frames per second than the file has would make the
            # stride zero and `range` raise.
            stride = max(1, int(src_fps / sample_fps))
            self._index = list(range(0, n_src, stride))
        self.n_available = len(self._index)
        if num_frames is not None:
            self._index = self._index[:max(1, num_frames)]
        self.window = int(window) if window else None
        self._block = None        # the decoded block, windowed mode only
        self._block_start = -1
        if self.window:
            # The reader has to outlive __init__ here, unlike the eager path where the
            # pixels are all copied out before it goes.
            self._vr = vr
            logger.debug(f'video: {len(self._index)} slots, decoded '
                         f'{self.window} at a time')
        else:
            self._video = torch.from_numpy(vr.get_batch(self._index).asnumpy())
            logger.debug(f'video shape: {tuple(self._video.shape)}')

    def __len__(self):
        return len(self._index)

    def _load_block(self, k):
        """Decode the block holding slot k, dropping the one before it.

        The old block is released before the new one is read so the two are never resident
        together -- the peak is one block, which is the whole point of windowing.
        """
        start = (k // self.window) * self.window
        self._block = None
        idx = self._index[start:start + self.window]
        self._block = torch.from_numpy(self._vr.get_batch(idx).asnumpy())
        self._block_start = start

    def get(self, k):
        """Slot k on its own, as a (1, H, W, 3) uint8 tensor.

        Random access, for a caller that assembles its own chunks
        (video_qa/measure_encoding_fps.py). The streaming solvers go through `frames`,
        which is the same read in arrival order.
        """
        if self.window:
            if self._block is None or not (
                    self._block_start <= k < self._block_start + len(self._block)):
                self._load_block(k)
            return self._block[k - self._block_start:k - self._block_start + 1]
        return self._video[k:k + 1]

    def frames(self, start, end):
        """Yield slots [start, end) as (1, H, W, 3) uint8 tensors, in arrival order."""
        for k in range(max(0, start), min(end, len(self._index))):
            yield self.get(k)


class ReKVStreamVQA(BaseVQA):
    def open_stream(self, video_sample, num_frames=None):
        """The video as a one-frame-at-a-time source. Reads no pixels yet."""
        return FrameStream(
            video_sample['video_path'],
            sample_fps=self.sample_fps,
            exact=self.exact_fps,
            num_frames=num_frames,
        )

    def ingest(self, stream, start, end):
        """Encode frames [start, end) as they arrive, one forward pass each.

        Returns the index one past the last frame encoded, which is `end` clamped to the
        length of the stream -- a video can end before a question's timestamp.
        """
        n = 0
        for frame in stream.frames(start, end):
            self.qa_model.encode_frame(frame)
            n += 1
        return start + n

    def video_open_qa(self, question, max_new_tokens=1024):
        input_text = {
            "question": question,
            "prompt": self.qa_model.get_prompt(question)
        }
        pred_answer = self.qa_model.question_answering(input_text, max_new_tokens=max_new_tokens)

        return {
            'pred_answer': pred_answer.replace('\n', ''),
        }

    @torch.inference_mode()
    def analyze_a_video(self, video_sample):
        stream = self.open_stream(video_sample)
        n_encoded = 0

        self.qa_model.clear_cache()
        self.qa_model.encode_init_prompt()

        for sample in video_sample['conversations']:
            logger.debug(f'sample: {sample}')
            question = sample['question']
            answer = sample['answer']

            # Encode up to the question's timestamp, then ask. The cursor only moves
            # forward, so each question sees exactly the frames the previous one saw plus
            # the ones that arrived since.
            n_visible = int(float(sample['end_time']) * self.sample_fps)
            if n_visible > n_encoded:
                n_encoded = self.ingest(stream, n_encoded, n_visible)

            # OpenQA. Reduction counters are read here rather than once per video: this
            # is a streaming run, so ingestion is interleaved with questions and the rate
            # at the time of the question is what that answer was produced against.
            qa_results = self.video_open_qa(question, max_new_tokens=256)
            self.record[(self.retrieve_size, self.chunk_size)].append({
                'video_id': video_sample['video_id'],
                'question': question,
                'answer': answer,
                'pred_answer': qa_results['pred_answer'],
                **self.reduction_stats(),
            })

        self.qa_model.log_reduction_summary(n_encoded)


if __name__ == "__main__":
    work(ReKVStreamVQA)
