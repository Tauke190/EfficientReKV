import numpy as np
import torch
from logzero import logger


class Abstract_ReKV:
    processor = None
    kv_cache = None
    # Decode steps paid for by the most recent question_answering call. Set by each
    # subclass's generation loop; None means that model does not report it.
    last_generated_tokens = None
    # Block indices retrieved for the most recent question, one list per layer. Set by
    # `_capture_retrieved_blocks`, which the backend calls while they are still live; None
    # means that backend does not report them.
    last_retrieved_blocks = None
    # Stage-1 encoder-side reduction (model/vision_reduction.py). Set by the subclass
    # when enabled -- it wraps that model's vision tower, so unlike the stage-2 pruner
    # it cannot be constructed here. None means the vision tower runs unmodified.
    vision_reducer = None

    def __init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                 token_pruner=None):
        self.processor = processor
        self.n_frame_tokens = n_frame_tokens
        self.init_prompt_ids = init_prompt_ids
        self.n_local = n_local
        self.topk = topk
        self.chunk_size = chunk_size

        # Stage-2 memory-side token pruning (see model/token_pruning.py). When set, a
        # frame no longer maps to a fixed number of KV-Cache tokens, so `n_frame_tokens`
        # stops being "tokens per frame" and means only "block size" -- the two were the
        # same number purely by convention. `frame_token_counts` is what restores the
        # frame<->block correspondence that convention used to provide for free.
        self.token_pruner = token_pruner
        self._reset_stream_state()

    @property
    def block_size(self):
        """KV-Cache block size. Blocks are cut by counting this many tokens off the
        stream (see ContextManager._append_global), with no notion of a frame. Without
        pruning one block happens to hold exactly one frame; with pruning a block spans
        a variable number of frames -- many for a static scene, one for a busy one."""
        return self.n_frame_tokens

    def _reset_stream_state(self):
        self.frame_token_counts = []   # tokens contributed to memory by each frame
        self._pending_embeds = None    # tokens not yet fed (< block_size, awaiting a full block)
        self._n_tokens_fed = 0         # video tokens handed to the LM so far
        self.last_retrieved_blocks = None  # blocks the last question retrieved (per layer)

    def clear_cache(self):
        self.kv_cache = None
        self._reset_stream_state()
        # Both reduction stages carry a per-position reference across chunks; leaving
        # either set would make the next video's first frames diff against the previous
        # video's content.
        if self.token_pruner is not None:
            self.token_pruner.reset()
        if self.vision_reducer is not None:
            self.vision_reducer.reset()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @torch.inference_mode()
    def encode_init_prompt(self):
        if not isinstance(self.init_prompt_ids, torch.Tensor):
            self.init_prompt_ids = torch.as_tensor([self.init_prompt_ids], device=self.device)
        output = self.language_model(input_ids=self.init_prompt_ids, use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values

    def _get_video_features(self, pixel_values_videos):
        pass

    def _forward_features(self, embeds):
        """Hand a whole number of blocks' worth of visual tokens to the LM."""
        assert self.n_local >= embeds.shape[1], f'n_local: {self.n_local}, video_features: {embeds.shape[1]}'
        output = self.language_model(inputs_embeds=embeds, past_key_values=self.kv_cache,
                                     use_cache=True, return_dict=True)
        self.kv_cache = output.past_key_values
        self._n_tokens_fed += embeds.shape[1]

    def _ingest_video_features(self, video_features, n_frames):
        """Prune, then feed to the LM in whole blocks.

        This is where the encode-time saving comes from: the LM call is ~84% of
        `_encode_video_chunk` and its cost is proportional to the number of tokens
        handed to it, so dropping tokens here is the only thing in the pipeline that
        shortens it.

        ContextManager requires every post-init append to be a whole number of blocks:
        _append_global asserts `global_remainder_len % block_size == 0`, append() asserts
        the remainder is fully drained, and the pre-offload retrieval path asserts the
        local window is block-aligned too. Pruned survivor counts are arbitrary, so the
        tail of each chunk is held back here until the next chunk completes a block.
        Buffering *embeddings* (not KV) is free -- it only delays the LM call.

        Args:
            video_features: (1, n_frames * n_frame_tokens, D) dense features.
            n_frames: number of frames in this chunk.
        """
        D = video_features.shape[-1]

        if self.token_pruner is not None:
            feats = video_features.view(n_frames, self.n_frame_tokens, D)
            kept, counts = self.token_pruner(feats)     # (N, D), list[int]
            video_features = kept.unsqueeze(0)          # (1, N, D)
        else:
            counts = [self.n_frame_tokens] * n_frames

        self.frame_token_counts.extend(counts)

        if self._pending_embeds is not None:
            video_features = torch.cat([self._pending_embeds, video_features], dim=1)

        n = video_features.shape[1]
        n_full = (n // self.block_size) * self.block_size
        if n_full > 0:
            self._forward_features(video_features[:, :n_full].contiguous())
        self._pending_embeds = video_features[:, n_full:].contiguous() if n_full < n else None

    def flush_stream(self):
        """Make everything ingested so far visible to a question, right now.

        A stream has no end to flush at, so this is called before each question instead
        (every `question_answering` starts with it) and the stream continues afterwards.
        It is a no-op unless stage-2 pruning left a partial block pending -- without
        pruning a frame is exactly one block, so nothing is ever held back.

        Not called per frame on purpose: that would pad every frame up to a full block and
        undo the pruning entirely.
        """
        self._flush_pending()

    def _flush_pending(self):
        """Feed the final partial block, padded up to block_size.

        Padding repeats the last surviving token. The alternative -- dropping the tail --
        would silently discard up to block_size-1 tokens from the end of every video,
        which is where the answer often is. The padding is charged to the last frame in
        `frame_token_counts` so the block<->frame table stays consistent with what was
        actually fed.
        """
        if self._pending_embeds is None or self._pending_embeds.shape[1] == 0:
            self._pending_embeds = None
            return
        n = self._pending_embeds.shape[1]
        n_pad = self.block_size - n
        pad = self._pending_embeds[:, -1:].expand(-1, n_pad, -1)
        self._forward_features(torch.cat([self._pending_embeds, pad], dim=1).contiguous())
        self._pending_embeds = None
        if self.frame_token_counts:
            self.frame_token_counts[-1] += n_pad
        logger.debug(f'flushed final block: {n} real + {n_pad} padding tokens')

    @torch.inference_mode()
    def _encode_video_chunk(self, video_chunk):
        pixel_values_videos = self.processor.video_processor(video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*196, D)
        self._ingest_video_features(video_features, n_frames=video_chunk.shape[0])

    @torch.inference_mode()
    def encode_frame(self, frame):
        """Ingest exactly one frame: preprocess it, encode it, append it to the KV-Cache.

        The streaming entry point. Everything from the pixel normalization to the LM
        prefill happens on this frame alone, so no stage of the pipeline ever holds a
        frame back waiting for the next one to arrive, and none of them can see a frame
        from the future. `video_qa/rekv_stream_vqa.py` and the OVO-Bench solver drive the
        whole video through here, one call per frame.

        This is not a different code path from `encode_video` -- it is the same
        `_encode_video_chunk` with a chunk of one, which is also why the two agree
        numerically. Both reduction stages carry their reference across calls and decide
        frame by frame (model/token_pruning.py, model/vision_reduction.py), and
        ContextManager.append already cuts its input into one-frame blocks internally, so
        arriving a frame at a time changes when the work happens, not what it computes.

        Args:
            frame: (1, H, W, 3) or (H, W, 3) uint8, the frame that just arrived.
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        assert frame.shape[0] == 1, \
            f'encode_frame takes one frame; got {frame.shape[0]}. Use encode_video for a clip.'
        self._encode_video_chunk(frame)

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=64):  # video: (Nv, H, W, 3)
        """Ingest a whole clip that is already in hand.

        The offline path: it batches frames into `encode_chunk_size` forward passes for
        throughput, which is legitimate only because the clip exists in full before the
        first frame is encoded. The streaming solvers call `encode_frame` instead -- see
        video_qa/rekv_stream_vqa.py.
        """
        # encode chunk by chunk
        num_frames = video.shape[0]
        num_chunks = num_frames // encode_chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * encode_chunk_size
            end_idx = start_idx + encode_chunk_size
            chunk_video = video[start_idx:end_idx]
            self._encode_video_chunk(chunk_video)
            logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

        # Handle remaining frames
        remaining_frames = num_frames % encode_chunk_size
        if remaining_frames > 0:
            start_idx = num_chunks * encode_chunk_size
            end_idx = start_idx + remaining_frames
            remaining_video = video[start_idx:end_idx]
            self._encode_video_chunk(remaining_video)

        # Nothing more is coming, so the held-back tail has to go in now.
        self._flush_pending()
        self.log_reduction_summary(num_frames)

    def log_reduction_summary(self, num_frames):
        """Report what the two reduction stages did, once the video is fully ingested.

        Its own method because the streaming path has no `encode_video` call to hang it
        off: it ingests frame by frame and reports when the stream ends.
        """
        # The two stages are reported separately because they are separate claims and
        # separate denominators: stage 1 is patches encoded (vision-tower cost, memory
        # untouched), stage 2 is tokens stored (LM prefill and KV, encoder untouched).
        if self.vision_reducer is not None:
            logger.info(f'video: {num_frames} frames, stage-1 encoded '
                        f'{100.0 * self.vision_reducer.keep_rate:.1f}% of patches')
        if self.token_pruner is not None:
            logger.info(
                f'video: {num_frames} frames, '
                f'{self._n_tokens_fed}/{num_frames * self.n_frame_tokens} tokens in memory '
                f'({100.0 * self.token_pruner.keep_rate:.1f}% keep rate), '
                f'{self.num_blocks} blocks spanning '
                f'{num_frames / max(self.num_blocks, 1):.1f} frames each on average'
            )
        logger.debug(f'KV-Cache RAM usage: {self.calc_memory_usage() / (1024**3):.1f} GB')

    # ---- block <-> frame mapping -------------------------------------------------
    # Without pruning these are the identity (block i == frame i). With pruning a block
    # covers a variable, content-dependent span of frames, so anything that reasons
    # about *when* something happened -- external retrieval, temporal grounding metrics
    # -- has to go through here instead of assuming the identity.

    @property
    def num_blocks(self):
        return self._n_tokens_fed // self.block_size

    def _frame_token_bounds(self):
        """(starts, ends) token offsets for each frame, in the video-token stream."""
        counts = np.asarray(self.frame_token_counts, dtype=np.int64)
        ends = np.cumsum(counts)
        return ends - counts, ends

    def _frame_blocks(self):
        """Per frame, the list of blocks holding the memory that represents it.

        A frame that contributed tokens maps to the blocks those tokens landed in. A
        frame that contributed *nothing* is not absent from memory -- it was dropped
        precisely because an earlier token already represents it -- so it maps to the
        block holding that carrier, i.e. the block of the last token emitted before it.
        Mapping such a frame to nothing would make it unretrievable, which is exactly
        backwards: a static stretch is the case where retrieval is most likely to be
        asked for a frame that emitted no tokens of its own.
        """
        starts, ends = self._frame_token_bounds()
        n_blocks = self.num_blocks
        out = []
        for f in range(len(self.frame_token_counts)):
            if ends[f] == starts[f]:  # contributed nothing: find its carrier
                carrier = max(int(starts[f]) - 1, 0) // self.block_size
                out.append([min(carrier, n_blocks - 1)] if n_blocks else [])
                continue
            b_st = int(starts[f]) // self.block_size
            b_ed = int(ends[f] - 1) // self.block_size
            out.append([b for b in range(b_st, b_ed + 1) if 0 <= b < n_blocks])
        return out

    def block_frame_ranges(self):
        """For each block, the (first_frame, last_frame) it represents, inclusive.

        The inverse of _frame_blocks, so the two are consistent by construction. Spans
        are monotonic and together cover every frame -- including the trailing frames of
        a static video, which emit no tokens but are still represented by the last block.
        """
        if not self.frame_token_counts or not self.num_blocks:
            return []
        ranges = [None] * self.num_blocks
        for f, blocks in enumerate(self._frame_blocks()):
            for b in blocks:
                lo, hi = ranges[b] if ranges[b] is not None else (f, f)
                ranges[b] = (min(lo, f), max(hi, f))
        # A block can hold only interior tokens of a single frame (a frame emits at most
        # block_size tokens, so it spans at most two blocks) and pick up no frame of its
        # own. Inherit from the neighbour rather than leaving a hole.
        last = (0, 0)
        for b in range(self.num_blocks):
            if ranges[b] is None:
                ranges[b] = last
            last = ranges[b]
        return ranges

    def frames_to_blocks(self, frame_indices):
        """Block indices holding the memory representing the given frames.

        Used by external retrieval, which speaks in frame indices.
        """
        if not self.frame_token_counts or not self.num_blocks:
            return []
        per_frame = self._frame_blocks()
        n_frames = len(per_frame)
        blocks = set()
        for f in frame_indices:
            f = int(f)
            if 0 <= f < n_frames:
                blocks.update(per_frame[f])
        return sorted(blocks)

    # --- what retrieval actually pulled in ------------------------------------------
    #
    # Each layer retrieves its own top-k blocks, and ContextManager.reset_retrieval()
    # clears the indices as soon as the question's forward pass is done -- by design, since
    # a stale set would be silently reused by the next question. So a solver that wants to
    # know *what was retrieved* has to read them in that window, which is what the capture
    # below is for: the backend calls it between the retrieval pass and the reset.

    def _capture_retrieved_blocks(self):
        """Stash the block indices every layer just retrieved, before they are reset.

        Shape: one sorted list per layer. Batch unit 0 only -- the eval path runs one video
        at a time, and a per-unit dimension nobody uses would make every reader index past
        it. Layers that did not retrieve (nothing offloaded yet, so the question attends
        the local window directly) contribute nothing, so an empty list means "retrieval
        never ran for this question", which is a different statement from "it ran and found
        nothing".
        """
        blocks = []
        for layer_kv in (self.kv_cache or []):
            idx = getattr(layer_kv, 'retrieved_block_indices', None)
            if not idx:
                continue
            blocks.append(sorted(int(b) for b in idx[0]))
        self.last_retrieved_blocks = blocks

    def tokens_after_frame(self, frame_index):
        """Video tokens fed to the LM after `frame_index`'s own tokens.

        The question a needle-in-a-haystack row has to answer before its retrieval columns
        mean anything: was the evidence still inside the local window when the question
        came? Below n_local it is attended directly, with true RoPE positions, and retrieval
        is irrelevant to that row -- reading a retrieval miss there as a failure would be
        exactly backwards. Counts tokens rather than frames because pruning makes those two
        different, and n_local is denominated in tokens.
        """
        counts = self.frame_token_counts
        if not counts:
            return 0
        i = min(max(int(frame_index) + 1, 0), len(counts))
        return int(sum(counts[i:]))

    def retrieval_hit_stats(self, first_frame, last_frame):
        """How much of the retrieved memory represents frames [first_frame, last_frame].

        The needle-in-a-haystack measurement: given the frames that hold the evidence, this
        says whether ReKV's retrieval actually reached them. Blocks are the unit because
        blocks are what retrieval selects; `block_frame_ranges()` is what turns them back
        into frames, and it is not the identity once stage-2 pruning is on -- a block then
        spans many frames of a static scene and one frame of a busy one.

        Reported per layer and then averaged, because retrieval is per layer: 28 layers each
        pick their own top-k, so "the needle was retrieved" is a fraction, not a boolean.
        `layers_hit_frac` is the headline -- the share of layers whose top-k contained at
        least one block overlapping the window -- and `blocks_hit_mean` says how many of the
        k slots that layer spent on it.

        Returns {} when nothing was captured, so a baseline row keeps its existing columns.
        """
        blocks = self.last_retrieved_blocks
        if not blocks:
            return {}
        ranges = self.block_frame_ranges()
        if not ranges:
            return {}
        # A block counts as the needle's if the frames it represents overlap the window at
        # all. Overlap rather than containment: with pruning a single block can hold the
        # whole needle plus its surroundings, and that block is exactly the one retrieval
        # had to find.
        needle_blocks = {b for b, (lo, hi) in enumerate(ranges)
                         if lo <= last_frame and hi >= first_frame}
        hits = [len(needle_blocks.intersection(layer)) for layer in blocks]
        n_layers = len(hits)
        return {
            'n_layers_retrieved': n_layers,
            'n_blocks_per_layer': len(blocks[0]),
            'n_needle_blocks': len(needle_blocks),
            'layers_hit_frac': round(sum(1 for h in hits if h) / n_layers, 4),
            'blocks_hit_mean': round(sum(hits) / n_layers, 4),
        }

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128):
        pass

    def calc_memory_usage(self):
        n_layers = len(self.kv_cache)
        memory = n_layers * self.kv_cache[0].calculate_cpu_memory()
        return memory
