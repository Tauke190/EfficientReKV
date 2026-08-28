"""Encoder-side temporal-redundancy reduction for a SigLIP vision tower.

Decides which patches are worth encoding, runs SigLIP on only those, and hands
back the dense grid the rest of the pipeline expects:

    frame -> 729 patches -> encode survivors only     <- the saving: encoder FLOPs
                         -> scatter back to 27x27
                         -> project -> pool -> LM

The saving is FLOPs inside the vision tower, and nothing else. The KV-Cache is
unchanged in size: `apply_pooling` bilinearly resamples the whole 27x27 grid, so
it cannot run on a ragged survivor set, and the dense grid has to be rebuilt
before it. Every frame therefore still contributes exactly 196 tokens and one
KV block is still one frame.

What it does change about the KV-Cache is its *contents*: dropped slots are
filled with the encoded state of the patch they carry, which is stale by
`threshold` at the encoder input and rather more than that at the output, since
a reused patch also misses every cross-patch context update its frame received.
Measured on one MLVU clip at 1 FPS, mean relative error in the encoder output:
65% patches encoded -> 0.34, 53% -> 0.52, 47% -> 0.60. That is a real accuracy
cost, so calibrate `threshold` against task accuracy rather than against keep
rate, and reach for `refresh_every` when you need fidelity back.

Relation to the reference implementation
----------------------------------------
The keep rule is RLT's "ref" rule, identical to videoxlpro's
`find_idxs_to_keep_embed(mask_mode="ref")`: diff each patch against the embedding
it will actually be *reused* from, refresh that reference only where the patch
survives. Two deliberate departures:

  * **Streaming state.** The reference implementation is offline: it takes a whole
    clip and always keeps all of frame 0. The streaming path calls this once per
    arriving frame, so that rule would force-keep *every* frame and save nothing
    at all (and even offline, at 64 frames a call, it would restart the drift
    bound ~29 times per hour). Here the reference, the per-layer hidden states,
    and the encoded output all carry across calls, so the call boundary is
    invisible to the result. See `_carry` for what that costs (~46 MB).
  * **Frames with no survivors are dropped from the attention batch**, which the
    offline version never has to handle because its frame 0 always keeps
    everything. With carried state a whole call can contribute nothing.

Attention uses xformers' block-diagonal `memory_efficient_attention`, as the
reference does: per-frame query sets are ragged, and packing them without padding
is what keeps the attention cost proportional to the keep rate rather than to the
dense grid.


Also dropped from the port: the paper's tubelet/consecutive-frame variants and the
run-length position embedding. Attention here is strictly within-frame (as in
per-image SigLIP, which is what these weights were trained for and what
LLaVA-OneVision expects), so no temporal position encoding is load-bearing.
"""

import inspect

import torch
import torch.nn.functional as F
from logzero import logger

try:
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "model/vision_reduction.py needs xformers for ragged per-frame attention "
        "(pip install xformers). The unmodified baseline path does not."
    ) from e


class SiglipRLT:
    """Streaming RLT over a SigLIP vision tower.

    Deliberately NOT an nn.Module: it holds references to submodules of a model
    that is already built and device-mapped. Registering it would duplicate every
    SigLIP weight in the parent's `state_dict` and `.to()` traversal for no gain.

    Args:
        vision_tower: an HF `SiglipVisionModel` (LLaVA-OneVision's `model.vision_tower`).
        threshold: distance above which a patch is re-encoded. **Calibrate on your
            own footage, against accuracy.** Redundancy is a direct function of
            frame spacing, so a value tuned at one sample_fps means something
            different at another, and it varies hugely by content (a fixed camera
            and a head-mounted one are not comparable).
        mask_space: where redundancy is tested.
            "embed" (default) compares SigLIP patch embeddings with the spatial
                position embedding removed. The patch-embed conv averages out sensor
                noise and codec ringing that a pixel test mistakes for content, and
                it is the quantity reuse actually incurs error in.
            "pixel" compares the preprocessed pixels directly (the paper's test).
                Cheaper to reason about, but blind to the distinction above.
                Its threshold is on the *normalized* pixel scale (roughly [-1, 1]),
                not 0-255.
        metric: "cosine" (default, 1 - cos_sim) or "l2" (euclidean over a running
            mean patch norm). Only meaningful for mask_space="embed"; the pixel test
            is always mean absolute difference. cosine has no scale term and so gives
            identical decisions at any chunk size; l2's normalizer is a running
            estimate of a whole-clip quantity and does not: the offline reference
            normalizes by the mean patch norm of the entire clip, which a streaming
            encoder cannot know in advance.
        refresh_every: force-encode every Nth frame in full regardless of distance.
            Bounds worst-case staleness. 0 disables.
    """

    def __init__(self, vision_tower, threshold, mask_space="embed", metric="cosine",
                 refresh_every=0):
        assert mask_space in ("embed", "pixel"), f"unknown mask_space {mask_space!r}"
        assert metric in ("l2", "cosine"), f"unknown metric {metric!r}"
        assert threshold > 0, f"threshold must be positive, got {threshold}"

        vm = vision_tower.vision_model
        self.embeddings = vm.embeddings
        self.encoder_layers = vm.encoder.layers
        self.embed_dim = self.embeddings.embed_dim
        self.patch_size = self.embeddings.patch_size

        self.threshold = threshold
        self.mask_space = mask_space
        self.metric = metric
        self.refresh_every = refresh_every
        self.reset()

    # ---- state -----------------------------------------------------------------

    def reset(self):
        """Clear all carried state. Must be called between videos."""
        self.ref = None            # (P, D) fp32 (embed) or (3, H', W') fp32 (pixel)
        self.prev_out = None       # (P, C) last encoder output each position carries
        self.prev_x = None         # per-layer (P, C) residual state each position carries
        self.frame_idx = 0
        self._norm_sum = 0.0       # running mean patch norm, for metric="l2"
        self._norm_count = 0
        self.n_kept = 0
        self.n_seen = 0

    @property
    def keep_rate(self):
        return self.n_kept / self.n_seen if self.n_seen else 1.0

    # ---- keep mask -------------------------------------------------------------

    def _embed_mask(self, emb):
        """(T, P) keep mask from patch embeddings. Updates `self.ref` in place.

        The spatial position embedding is subtracted first: it is constant per slot,
        so it carries no content, but it does not cancel under the metric's norms --
        leaving it in would make sensitivity vary by grid position and would inflate
        cosine similarity by a shared component. Read through the module forward, as
        HF does, so sharded-weight hooks fire.

        Frame-at-a-time rather than vectorised: the reference update is sequential by
        construction, and materializing an fp32 copy of the whole (T, P, C) chunk
        would cost ~215 MB at T=64 for no benefit.
        """
        T, P, _ = emb.shape
        sp = self.embeddings.position_embedding(self.embeddings.position_ids)[0]  # (P, C)

        if self.metric == "l2":
            # Running mean over every patch seen since reset, so the threshold means the
            # same thing at frame 10 and frame 10000. A per-chunk scale would drift with
            # content and silently retune the threshold mid-video.
            n = torch.linalg.vector_norm(emb.float() - sp.float(), dim=-1)
            self._norm_sum += float(n.sum())
            self._norm_count += n.numel()
            scale = max(self._norm_sum / self._norm_count, 1e-6)

        mask = torch.zeros(T, P, dtype=torch.bool, device=emb.device)
        for t in range(T):
            # Distances between near-identical vectors cancel most of their significant
            # bits, and fp16 has none to spare -- the whole test lives in that
            # cancellation. Compare in fp32.
            f = (emb[t] - sp).float()                                   # (P, C)
            if self.ref is None:
                k = torch.ones(P, dtype=torch.bool, device=emb.device)
                self.ref = f.clone()
            else:
                if self.metric == "l2":
                    d = torch.linalg.vector_norm(f - self.ref, dim=-1) / scale
                else:
                    d = 1.0 - F.cosine_similarity(f, self.ref, dim=-1)
                k = d > self.threshold
                if self.refresh_every > 0 and self.frame_idx % self.refresh_every == 0:
                    k = torch.ones_like(k)
                self.ref[k] = f[k]
            mask[t] = k
            self.frame_idx += 1
        return mask

    def _pixel_mask(self, pixel_values, P):
        """(T, P) keep mask from preprocessed pixels. Updates `self.ref` in place."""
        T, _, H, W = pixel_values.shape
        ps = self.patch_size
        h, w = H // ps, W // ps
        assert h * w == P, f"pixel grid {h}x{w}={h * w} != SigLIP grid {P}"
        # SigLIP's patch conv uses padding="valid", so it discards the ragged border.
        # Crop to match, or the reference update lines up with the wrong pixels.
        x = pixel_values[:, :, : h * ps, : w * ps].float()

        mask = torch.zeros(T, P, dtype=torch.bool, device=pixel_values.device)
        for t in range(T):
            cur = x[t]                                                  # (3, H', W')
            if self.ref is None:
                k = torch.ones(P, dtype=torch.bool, device=x.device)
                self.ref = cur.clone()
            else:
                d = F.avg_pool2d((cur - self.ref).abs().unsqueeze(0), ps)[0].mean(0)
                k = (d > self.threshold).reshape(-1)                    # (P,)
                if self.refresh_every > 0 and self.frame_idx % self.refresh_every == 0:
                    k = torch.ones_like(k)
                up = k.view(1, h, w).float()
                up = up.repeat_interleave(ps, -2).repeat_interleave(ps, -1)
                self.ref = self.ref * (1 - up) + cur * up
            mask[t] = k
            self.frame_idx += 1
        return mask

    # ---- index construction ----------------------------------------------------

    def _index(self, mask2d, N):
        """Gather maps from the packed survivor array to the dense grid.

        Survivors are packed in (t, p) raster order, so each frame's survivors form a
        contiguous run -- which is what lets the block-diagonal attention bias below be
        built from per-frame counts alone.

        `src_row[t, p]` is the row holding the state slot (t, p) carries: its own row
        if it survived, otherwise its last surviving copy. Positions with no survivor
        anywhere in this chunk ("orphans", possible only because the reference is
        carried across chunks) index into a virtual tail of P rows appended after the
        survivors, which `_encode` fills from the previous chunk's carried state.
        """
        T, P = mask2d.shape
        dev = mask2d.device
        pr = torch.arange(P, device=dev).view(1, P)

        surv_idx = mask2d.reshape(-1).nonzero(as_tuple=True)[0]         # (N,)
        rank = torch.zeros(T * P, dtype=torch.long, device=dev)
        rank[surv_idx] = torch.arange(N, device=dev)

        t_idx = torch.arange(T, device=dev).view(T, 1).expand(T, P)
        kept_t = torch.where(mask2d, t_idx, torch.full_like(t_idx, -1))
        carry = torch.cummax(kept_t, dim=0).values                      # (T, P), -1 = orphan
        src_row = rank[(carry.clamp(min=0) * P + pr).reshape(-1)].view(T, P)
        src_row = torch.where(carry < 0, N + pr.expand(T, P), src_row)

        # Frames with zero survivors contribute no queries; dropping their empty block
        # keeps xformers from ever seeing a zero-length sequence.
        counts = mask2d.sum(dim=1)                                      # (T,)
        active = counts > 0
        q_seqlen = [int(c) for c in counts[active].tolist()]

        return src_row, surv_idx, active, q_seqlen

    # ---- encoder ---------------------------------------------------------------

    def _encode(self, emb, mask2d):
        """Run SigLIP on survivors only; return the dense (T, P, C) grid.

        Attention is strictly within-frame, and every frame's query set attends over a
        FULL P-token key/value set: survivors contribute freshly computed k/v, dropped
        slots contribute the k/v they had when they last survived. That is exactly the
        state the dense model would compute for them, since they were dropped precisely
        because their content did not change, and layer_norm/k_proj/v_proj are
        position-wise. At 100% keep this reduces to dense SigLIP exactly.

        Cost: the running state stays in packed survivor form (N, C), so every linear op
        -- q/k/v/out projections and the MLP -- costs O(N) rather than O(T*P), and the
        keep-rate saving is preserved in full. The only dense-sized work is the two
        gathers that build each frame's attention key/value set.
        """
        T, P, C = emb.shape
        N = int(mask2d.sum())

        if N == 0:
            # A whole chunk with nothing to re-encode. Only reachable with carried state
            # (the first chunk keeps all of frame 0), so prev_out is always available.
            assert self.prev_out is not None
            return self.prev_out.unsqueeze(0).expand(T, P, C).clone()

        src_row, surv_idx, active, q_seqlen = self._index(mask2d, N)
        has_prev = bool((src_row >= N).any())
        assert not has_prev or self.prev_x is not None, \
            "orphan slots without carried state; this chunk cannot be reconstructed"

        n_act = len(q_seqlen)
        kv_rows = src_row[active].reshape(-1)                           # (n_act * P,)
        final_src = src_row[-1]                                         # (P,)
        attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlen, kv_seqlen=[P] * n_act)

        x = emb.reshape(T * P, C).index_select(0, surv_idx)             # (N, C)
        new_prev_x = []

        for i, layer in enumerate(self.encoder_layers):
            a = layer.self_attn
            H, d = a.num_heads, a.head_dim

            # Append the carried residual state so orphan slots have somewhere to gather
            # their k/v from. P extra rows per chunk -- one frame's worth.
            x_ext = torch.cat([x, self.prev_x[i]], 0) if has_prev else x
            new_prev_x.append(x_ext.index_select(0, final_src))          # (P, C)

            h_ext = layer.layer_norm1(x_ext)
            q = a.q_proj(h_ext[:N]).view(1, N, H, d)
            # Each frame's full key/value set: survivors' own k/v, plus the k/v carried
            # by every dropped slot from its last surviving copy.
            k = a.k_proj(h_ext).index_select(0, kv_rows).view(1, n_act * P, H, d)
            v = a.v_proj(h_ext).index_select(0, kv_rows).view(1, n_act * P, H, d)

            o = xops.memory_efficient_attention(
                q, k, v, attn_bias=attn_bias, scale=a.scale
            ).reshape(N, C)

            x = x + a.out_proj(o)
            x = x + layer.mlp(layer.layer_norm2(x))

        # Materialize the dense grid once, at the end. Video-XL-Pro and LLaVA-OneVision
        # both consume hidden_states[-1], i.e. BEFORE post_layernorm, so it is not applied
        # here either -- the projector was trained on pre-LN features.
        x_out = torch.cat([x, self.prev_out], 0) if has_prev else x
        dense = x_out.index_select(0, src_row.reshape(-1)).view(T, P, C)

        self.prev_x = new_prev_x
        self.prev_out = dense[-1].clone()
        return dense

    # ---- entry point -----------------------------------------------------------

    @torch.inference_mode()
    def __call__(self, pixel_values):
        """Encode one chunk.

        Args:
            pixel_values: (T, 3, H, W) preprocessed frames.

        Returns:
            (T, P, C) dense pre-post_layernorm features -- a drop-in for
            `vision_tower(...).hidden_states[-1]`.
        """
        assert pixel_values.dim() == 4, f"expected (T, 3, H, W), got {tuple(pixel_values.shape)}"
        T = pixel_values.shape[0]

        # Runs first either way: mask_space="embed" tests on these, and the encoder needs
        # them regardless, so nothing is computed twice.
        emb = self.embeddings(pixel_values)                             # (T, P, C)
        P = emb.shape[1]

        mask2d = self._embed_mask(emb) if self.mask_space == "embed" \
            else self._pixel_mask(pixel_values, P)

        n_kept = int(mask2d.sum())
        self.n_kept += n_kept
        self.n_seen += T * P
        logger.debug(
            f"[vision] chunk encoded {n_kept}/{T * P} patches "
            f"({100.0 * n_kept / max(T * P, 1):.1f}%), "
            f"video-so-far {100.0 * self.keep_rate:.1f}%"
        )
        return self._encode(emb, mask2d)


# ---- method registry ---------------------------------------------------------------
#
# Anything registered here replaces the vision tower's forward and must implement the
# same three members:
#
#   __call__(pixel_values) -> (T, P, C)
#       Drop-in for `vision_tower(...).hidden_states[vision_feature_layer]`. Must be
#       causal and stateful across calls: the streaming path passes one frame per call, so
#       a method that resets per call cannot skip any patch at all.
#   reset()
#       Clear per-video state. `Abstract_ReKV.clear_cache()` calls it between videos.
#   keep_rate
#       Property, patches encoded / patches seen since the last reset.
#
# The dense (T, P, C) return is not negotiable: `apply_pooling` bilinearly resamples the
# full 27x27 grid, so anything that returns a ragged set cannot be pooled. That is also
# why nothing registered here can shrink the KV-Cache, however aggressive it is.
VISION_REDUCERS = {
    'rlt': SiglipRLT,
}


def build_vision_reducer(method, vision_tower, **kwargs):
    """Construct a registered reducer, or fail with a message naming what exists.

    Hyperparameters are per-method by design: the caller passes what that method takes,
    and an argument the method does not accept is an error rather than a silent no-op --
    otherwise a stale flag left in a script would look like it was applied.
    """
    if method not in VISION_REDUCERS:
        raise ValueError(
            f"unknown vision reduction method {method!r}; registered: "
            f"{sorted(VISION_REDUCERS)}. Add it to VISION_REDUCERS in "
            f"model/vision_reduction.py."
        )
    cls = VISION_REDUCERS[method]
    accepted = set(inspect.signature(cls.__init__).parameters) - {'self', 'vision_tower'}
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(
            f"vision reduction method {method!r} does not accept {sorted(unknown)}; "
            f"it takes {sorted(accepted)}"
        )
    return cls(vision_tower, **kwargs)
