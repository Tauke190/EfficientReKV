"""Temporal-redundancy pruning of visual tokens at the KV-Cache boundary (Stage 2).

This is the *memory-side* reduction: it drops redundant tokens right before they
enter the LLM, so they are never prefilled and never occupy a KV-Cache block.

It is also, on ReKV, the reduction that actually moves encoding throughput.
Measured breakdown of `_encode_video_chunk` (RVS-Ego @0.5fps, llava_ov_0.5b, GPU
preprocessing, local window full):

    preprocessing          1.40 ms/frame   ( 2.5%)
    vision tower           7.71 ms/frame   (13.6%)   <- what Stage 1 attacks
    language_model(...)   47.70 ms/frame   (84.0%)   <- what Stage 2 attacks

The LM's cost is proportional to the number of tokens fed, so feeding fewer is
the only lever on that 84%. Encoder-side reduction (model/vision_reduction.py)
cannot help there: `apply_pooling` bilinearly resamples the full 27x27 grid, so
the dense grid has to be rebuilt and all 196 tokens per frame reach the LM
regardless. Measured end to end, Stage 1 alone gave +1.6% frames/s, Stage 2 alone
+29%, and both together +58%.

The redundancy test is RLT's "ref" rule, applied to projected+pooled features
instead of SigLIP patch embeddings:

    keep(t, p)  iff  dist(feat[t, p], ref[p]) > threshold

where ``ref[p]`` is the feature that position ``p`` was last *kept* with. Diffing
against the carried reference rather than against frame t-1 is what bounds drift:
a token dropped at t is by construction within `threshold` of the value the model
will actually attend to for it, no matter how long the drop run gets. Diffing
against t-1 instead lets many individually-sub-threshold changes accumulate into
an arbitrarily large error.

Two properties matter for streaming:

  * **Stateful across calls.** The streaming path calls this once per arriving
    frame (T=1), so a pruner that reset between calls would force-keep every
    frame and do nothing at all; the offline path calls it per 64-frame chunk,
    where resetting would restart the drift bound 20+ times per video. ``ref``
    persists across ``__call__`` and is cleared only by ``reset()``, which
    ``Abstract_ReKV.clear_cache()`` calls between videos.
  * **Causal.** Nothing looks ahead. Under the default "cosine" metric the keep
    decision for frame t is bit-identical whether the video is ingested offline
    in one pass or arrives as a live stream, at any chunk size. This does NOT
    hold for "l2", whose normalizer is a running estimate of a whole-clip
    quantity -- see the `metric` argument.

Unlike encoder-side RLT there is no reconstruction step: a dropped token is
simply absent from the sequence. The LLM never sees a slot for it. That is what
makes both the prefill and the KV-Cache shrink, but it does mean the dropped
token contributes no attention mass at all, whereas a scattered-back duplicate
would have contributed a (redundant) copy at its own position. What it does *not*
do is fabricate a value: the content of a dropped token is still present in
memory, within `threshold`, in the token it was deduplicated against.
"""

import inspect

import torch
from logzero import logger


class StreamingTokenPruner:
    """Stateful temporal-redundancy filter over a stream of per-frame token grids.

    Args:
        threshold: distance above which a token is kept. **Must be calibrated on
            your own features** -- this operates on projected LLM-space
            activations, whose scale has nothing to do with the SigLIP-embedding
            thresholds used by Stage 1 or in the RLT literature. Enable
            ``log_percentiles`` and read the reported distance distribution rather
            than transplanting a number.
        metric: "cosine" (default) is 1 - cos_sim, i.e. direction only. It has no
            scale term, which makes it **exactly invariant to chunk size** --
            arriving one frame at a time gives the same keep decisions as
            ingesting the clip in one pass. That property is why it is the
            default here even though the offline RLT code defaults to l2.
            (Measured on RVS-Ego at threshold 0.25: per-frame and 64-frame
            ingestion of the same 285 frames both kept 66.64%, fed the same
            37,436 tokens into the same 191 blocks; the only differences were
            +-1 token at 15 frames whose distance sat on the threshold and moved
            with the vision tower's batch-shape fp noise.)
            "l2" is euclidean distance divided by the running mean token norm.
            The offline implementation normalizes by the mean norm of the *whole
            clip*, which a streaming pruner cannot know in advance; the running
            estimate here converges to it but is not equal to it, so l2 keep
            decisions **depend on how the video was chunked** (measured: only
            ~12-22% of frames match the one-shot result under realistic motion,
            though aggregate keep rate drifts by <2 points). Use l2 only for
            offline comparison against published RLT numbers, not for streaming.
        refresh_every: if > 0, force-keep every Nth frame regardless of distance.
            Bounds worst-case staleness; 0 disables.
        log_percentiles: log the per-chunk distance distribution at DEBUG. Use
            this to pick `threshold`.
    """

    def __init__(self, threshold, metric="cosine", refresh_every=0, log_percentiles=False):
        assert metric in ("l2", "cosine"), f"unknown metric {metric!r}"
        assert threshold > 0, f"threshold must be positive, got {threshold}"
        self.threshold = threshold
        self.metric = metric
        self.refresh_every = refresh_every
        self.log_percentiles = log_percentiles
        self.reset()

    def reset(self):
        """Clear all carried state. Must be called between videos."""
        self.ref = None            # (P, D) fp32: what each position currently carries
        self.frame_idx = 0         # frames seen since reset (for refresh_every)
        self._norm_sum = 0.0       # running mean token norm, for the "l2" scale
        self._norm_count = 0
        self.n_kept = 0
        self.n_seen = 0

    @property
    def keep_rate(self):
        return self.n_kept / self.n_seen if self.n_seen else 1.0

    def _scale(self, feats_f32):
        """Running mean token norm. Computed over every token seen so far, not
        just this chunk, so the threshold means the same thing at frame 10 and at
        frame 10000 -- a per-chunk scale would drift with content and silently
        retune the threshold mid-video."""
        self._norm_sum += float(torch.linalg.vector_norm(feats_f32, dim=-1).sum())
        self._norm_count += feats_f32.shape[0] * feats_f32.shape[1]
        return max(self._norm_sum / self._norm_count, 1e-6)

    @torch.inference_mode()
    def __call__(self, feats):
        """Prune one chunk.

        Args:
            feats: (T, P, D) per-frame token grids, in the model's dtype.

        Returns:
            kept: (N, D) surviving tokens in (t, p) raster order, original dtype.
            counts: list of T ints, survivors contributed by each frame. Frames
                may contribute 0 tokens (a completely static frame) -- callers
                must not assume a frame occupies any particular span.
        """
        assert feats.dim() == 3, f"expected (T, P, D), got {tuple(feats.shape)}"
        T, P, D = feats.shape

        # Distances between near-identical vectors cancel most of their significant
        # bits, and fp16 has none to spare -- the whole test lives in that
        # cancellation. Compare in fp32; return survivors in the original dtype so
        # nothing downstream sees a dtype change.
        feats_f32 = feats.float()
        scale = self._scale(feats_f32) if self.metric == "l2" else 1.0

        kept_rows = []
        counts = []
        all_dists = [] if self.log_percentiles else None

        for t in range(T):
            f = feats_f32[t]  # (P, D)

            if self.ref is None:
                # First frame ever: nothing to compare against, keep everything.
                mask = torch.ones(P, dtype=torch.bool, device=feats.device)
                self.ref = f.clone()
            else:
                if self.metric == "l2":
                    dist = torch.linalg.vector_norm(f - self.ref, dim=-1) / scale
                else:
                    dist = 1.0 - torch.nn.functional.cosine_similarity(f, self.ref, dim=-1)
                if all_dists is not None:
                    all_dists.append(dist)
                mask = dist > self.threshold
                if self.refresh_every > 0 and self.frame_idx % self.refresh_every == 0:
                    mask = torch.ones_like(mask)
                # A kept token becomes the new reference for its position; a dropped
                # one leaves the reference alone, which is what bounds its drift.
                self.ref[mask] = f[mask]

            kept_rows.append(feats[t][mask])
            counts.append(int(mask.sum()))
            self.frame_idx += 1

        if all_dists:
            d = torch.cat(all_dists).float()
            q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=d.device)
            p = torch.quantile(d, q).tolist()
            logger.debug(
                f"[prune] distance percentiles (10/25/50/75/90): "
                f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {p[3]:.4f} {p[4]:.4f} "
                f"| threshold={self.threshold}"
            )

        kept = torch.cat(kept_rows, dim=0) if kept_rows else feats.new_zeros((0, D))
        self.n_kept += int(kept.shape[0])
        self.n_seen += T * P
        logger.debug(
            f"[prune] chunk kept {kept.shape[0]}/{T * P} "
            f"({100.0 * kept.shape[0] / max(T * P, 1):.1f}%), "
            f"video-so-far {100.0 * self.keep_rate:.1f}%"
        )
        return kept, counts


# ---- method registry ---------------------------------------------------------------
#
# Anything registered here plugs in at the *KV-Cache boundary*: it sits between the
# projector and the LM in `Abstract_ReKV._ingest_video_features` and decides which
# projected tokens are allowed to reach the LM at all. Adding a method is a class plus
# one registry entry, provided it implements the same three members:
#
#   __call__(feats) -> (kept, counts)
#       feats is (T, P, D) in the model's dtype. Returns surviving tokens as (N, D) in
#       (t, p) raster order, plus a length-T list of per-frame survivor counts. Must be
#       causal and stateful across calls -- the streaming path passes one frame per call,
#       and a method that resets per call cannot deduplicate anything at all.
#   reset()
#       Clear all per-video state. `Abstract_ReKV.clear_cache()` calls this between
#       videos; without it, one video's first frames diff against the previous video's.
#   keep_rate
#       Property, n_kept / n_seen since the last reset. Read by the encode-path logging
#       and by video_qa/measure_encoding_fps.py.
#
# SCOPE -- worth being explicit about before adding APT/TPAT here:
# this hook can only remove tokens *after* the vision tower has already run, so it
# shrinks LM prefill and KV-Cache but saves zero encoder FLOPs. A method whose savings
# are inside the vision tower (APT, which prunes patches during encoding) belongs in
# VISION_REDUCERS in model/vision_reduction.py instead; registering it only here would
# silently measure the wrong thing, attaching a correct keep rate to an encoder cost
# that never went down. TPAT-style methods that select among already-projected tokens
# do fit this hook as-is.
PRUNERS = {
    'rlt': StreamingTokenPruner,
}


def build_pruner(method, **kwargs):
    """Construct a registered pruner, or fail with a message naming what exists.

    Hyperparameters are per-method by design: the caller passes what that method takes,
    and an argument the method does not accept is an error rather than a silent no-op --
    otherwise a stale `--prune_threshold` left in a script would look like it was
    applied.
    """
    if method not in PRUNERS:
        raise ValueError(
            f"unknown pruning method {method!r}; registered: {sorted(PRUNERS)}. "
            f"Add it to PRUNERS in model/token_pruning.py."
        )
    cls = PRUNERS[method]
    accepted = set(inspect.signature(cls.__init__).parameters) - {'self'}
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(
            f"pruning method {method!r} does not accept {sorted(unknown)}; "
            f"it takes {sorted(accepted)}"
        )
    return cls(**kwargs)
