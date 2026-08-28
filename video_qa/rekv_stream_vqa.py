"""Streaming solver: frames arrive one at a time, questions are asked mid-stream.

Nothing here ever holds more than the current frame. `open_frame_stream` yields frame t
and only frame t; `encode_frame` preprocesses it, runs the vision tower on it, and
prefills it into the KV-Cache before frame t+1 is read. That is the whole point of the
scenario -- a live stream cannot batch frames it has not received -- and it is what this
solver used to get wrong: it decoded the entire video into one array up front and then
handed the model 64 frames per forward pass.

The model side already worked this way; only the driver did not. `encode_frame` is the
same `_encode_video_chunk` with a chunk of one, ContextManager.append already cuts its
input into one-frame blocks internally (model/attention/kv_cache_manager.py), and both
reduction stages already decide frame by frame off a reference carried across calls. So
the answers are unchanged to within GPU non-determinism; what changes is that the pipeline
now never depends on a frame that has not arrived, and peak RAM is one frame instead of a
whole video (~11 GB for a 1-hour 1080p video at 0.5 FPS).

Set REKV_FRAME_CACHE to a directory to stream from pre-extracted frames instead of the
source video. Still strictly one frame per read, just ~17x faster: at a 0.5 FPS stride
every decord read seeks across the file (measured 1.9 frames/s against 33 for cached
JPEGs on an RVS-Ego video) -- see video_qa/frame_cache.py.
"""

import os

import torch
from logzero import logger

from video_qa.base import BaseVQA, work
from video_qa.frame_cache import open_frame_stream


class ReKVStreamVQA(BaseVQA):
    def open_stream(self, video_sample, num_frames=None):
        """The video as a one-frame-at-a-time source. Reads no pixels yet."""
        return open_frame_stream(
            video_sample['video_path'],
            video_id=video_sample.get('video_id'),
            sample_fps=self.sample_fps,
            cache_dir=os.environ.get('REKV_FRAME_CACHE'),
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
