import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.token_pruning import build_pruner
# model.vision_reduction is imported lazily in load_model: it requires xformers, which
# the baseline and stage-2-only paths must not depend on.


class LlavaOneVision_ReKV(LlavaOnevisionForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                 token_pruner=None):
        LlavaOnevisionForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                               token_pruner=token_pruner)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|><|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape
        pixel_values_videos = pixel_values_videos.view(batch_size * frames, channels, height, width)

        if self.vision_reducer is not None:
            # Stage 1: SigLIP runs on non-redundant patches only, then rebuilds the
            # dense grid. Returns exactly what hidden_states[-1] would
            # (pre-post_layernorm), so everything downstream -- projector, pooling,
            # stage 2, the LM -- is untouched. load_model checks the two config fields
            # this equivalence depends on.
            selected_video_feature = self.vision_reducer(pixel_values_videos)
        else:
            video_features = self.vision_tower(pixel_values_videos, output_hidden_states=True)
            selected_video_feature = video_features.hidden_states[self.config.vision_feature_layer]

            if self.config.vision_feature_select_strategy == "default":
                selected_video_feature = selected_video_feature[:, 1:]
            elif self.config.vision_feature_select_strategy == "full":
                selected_video_feature = selected_video_feature

        video_features = self.multi_modal_projector(selected_video_feature)

        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*196, D)
        return video_features

    @torch.inference_mode()
    def _retrieve_and_prefill(self, input_text, retrieved_indices=None):
        """Retrieve on the question, prefill the prompt, return (logits, past_key_values).

        Factored out of `question_answering` so a caller that needs only the distribution
        over the first generated token can stop here. `video_qa/eval/eval_fpsbench_mba.py`
        scores a Yes/No decision that way: MBA asks K binaries per question, and paying a
        128-token decode for one bit of information K times over would dominate a sweep
        whose encoding cost is the thing being measured.

        Nothing here mutates `self.kv_cache`. In retrieval mode the attention path sets
        `updata_kv_cache = False` (model/attention/rekv_attention.py), so the question's
        own KV is used to select blocks and then dropped, and generation runs against the
        detached `(past_k, past_v)` this returns. That is what makes K binaries at one
        trigger mutually independent rather than a conversation.
        """
        # A question asked mid-stream must see every frame that has arrived, including one
        # whose tokens are still short of a full block. No-op unless stage-2 pruning is on.
        self.flush_stream()

        device = self.device

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # activate retrieval mode
            layer_kv.set_retrieval()

        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            # `retrieved_indices` is per-batch-unit lists of FRAME indices. Blocks are
            # cut by token count, so a frame maps to a block one-to-one only when no
            # stage-2 pruning happened; otherwise go through the frame->block table.
            block_indices = [self.frames_to_blocks(unit) for unit in retrieved_indices]
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.block_size, f'block_size: {layer_kv.block_size}, self.block_size: {self.block_size}'
                layer_kv.set_retrieved_block_indices(block_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)

        # Read the retrieved block indices while they are still live: reset_retrieval()
        # below clears them, and a needle-in-a-haystack run wants to know which blocks the
        # question actually pulled in (video_qa/rekv_fpsbench_stream_vqa.py). Storing a few
        # hundred ints per question, so it is unconditional rather than flag-gated.
        self._capture_retrieved_blocks()

        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        return out.logits, out.past_key_values

    @torch.inference_mode()
    def score_next_tokens(self, input_text, token_ids, retrieved_indices=None):
        """Logits for `token_ids` at the first generated position. One forward pass.

        `token_ids` is a flat list; the returned tensor is parallel to it. The caller
        groups them (see `video_qa/rekv_fpsbench_stream_small_vqa.py`, which takes the max
        over the surface forms of "Yes" and of "No"), because which spelling a tokenizer
        makes a single token of is a property of the backend, not of the metric.
        """
        logits, _ = self._retrieve_and_prefill(input_text, retrieved_indices)
        # No decode steps were paid for. Set explicitly so a stale count from a previous
        # question_answering call cannot be attributed to this one.
        self.last_generated_tokens = 0
        idx = torch.as_tensor(list(token_ids), device=logits.device)
        return logits[0, -1, :].index_select(0, idx).float().cpu()

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None,
                           min_new_tokens=0):
        """Answer one question against the retrieved KV-Cache.

        min_new_tokens suppresses the EOS stop until that many tokens have been emitted.
        Default 0 leaves generation exactly as it was; the speed benchmark sets it equal
        to max_new_tokens to hold the decode length fixed, since a latency averaged over
        answers of differing lengths measures the answers, not the system.
        """
        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                logits, past_key_values = self._retrieve_and_prefill(
                    input_text, retrieved_indices)
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0, -1, :]
            
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
            token = tokens[0]

            output_ids.append(token)

            if token in stop_token_ids and i + 1 >= min_new_tokens:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        # Counted here rather than re-tokenizing `output`: the decode drops the EOS and
        # detokenize->retokenize is not reliably round-trip, so the string is the wrong
        # place to recover the number of decode steps that were actually paid for.
        self.last_generated_tokens = len(output_ids)

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='model_zoo/LLaVA/llava-onevision-qwen2-7b-ov-hf',
               n_init=None, n_local=None, topk=64, chunk_size=1,
               prune_method=None, prune_threshold=None, prune_metric='cosine',
               prune_refresh_every=0, prune_log_percentiles=False,
               vision_method=None, vision_threshold=None, vision_mask_space='embed',
               vision_metric='cosine', vision_refresh_every=0):
    """Load LLaVA-OneVision with ReKV, and optionally either or both reduction stages.

    Independent switches attacking different parts of the encode cost. Measured split of
    `_encode_video_chunk` (RVS-Ego @0.5fps, llava_ov_0.5b, GPU preprocessing):
    preprocessing 2.5%, vision tower 13.6%, LM prefill 84.0%.

      vision_method  -- Stage 1, encoder-side. Skips SigLIP work on unchanged patches.
                        Attacks the 13.6%; KV-Cache size is UNCHANGED, because the dense
                        grid has to be rebuilt for apply_pooling.
      prune_method   -- Stage 2, memory-side. Drops redundant tokens before the LM.
                        Attacks the 84%, and shrinks KV RAM and retrieval cost with it.

    Measured end to end: stage 1 alone +1.6% frames/s, stage 2 alone +29%, both +58%.
    Both default to off, which is byte-for-byte the original model.
    """
    device = 'cuda'
    n_frame_tokens = 196

    # Stage 2. Off by default: with no method selected the encode path is byte-for-byte
    # the baseline (every frame contributes exactly n_frame_tokens, so one block is one
    # frame and nothing is ever buffered).
    #
    # Each method owns its own hyperparameters; the ones below belong to 'rlt'. A second
    # method with different knobs adds its own branch here rather than overloading these
    # -- build_pruner rejects arguments the method does not accept, so a mismatch fails
    # at load time instead of quietly doing nothing.
    if prune_method in (None, 'none') and prune_threshold is not None:
        prune_method = 'rlt'  # back-compat: --prune_threshold alone used to mean RLT

    token_pruner = None
    if prune_method not in (None, 'none'):
        if prune_method == 'rlt':
            assert prune_threshold is not None, "'rlt' pruning requires --prune_threshold"
            kwargs = dict(threshold=prune_threshold, metric=prune_metric,
                          refresh_every=prune_refresh_every,
                          log_percentiles=prune_log_percentiles)
        else:
            kwargs = {}
        token_pruner = build_pruner(prune_method, **kwargs)
        logger.info(f'token pruning: method={prune_method} '
                    + ' '.join(f'{k}={v}' for k, v in kwargs.items()))

    processor = LlavaOnevisionProcessor.from_pretrained(model_path)

    init_prompt = '<|im_start|>system \nYou are a helpful assistant.<|im_end|><|im_start|>user '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    inf_llm_config = {
        'n_init': init_prompt_ids.shape[1] if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LlavaOneVision_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        processor=processor,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
        token_pruner=token_pruner,
    )
    # The stage-1 reducer wraps the loaded vision tower, so it can only be built after
    # from_pretrained.
    if vision_method not in (None, 'none'):
        from model.vision_reduction import build_vision_reducer

        # _get_video_features substitutes the reducer's output for
        # hidden_states[vision_feature_layer] with no post-selection, which is only
        # equivalent under these two settings. Both hold for every llava-onevision-*-ov-hf
        # checkpoint; assert rather than silently encode against the wrong layer.
        assert model.config.vision_feature_layer == -1, \
            f'stage-1 reduction returns the last hidden state; config asks for ' \
            f'layer {model.config.vision_feature_layer}'
        assert model.config.vision_feature_select_strategy == 'full', \
            f'stage-1 reduction does not strip a CLS token; config asks for ' \
            f'{model.config.vision_feature_select_strategy!r}'
        if vision_method == 'rlt':
            assert vision_threshold is not None, "'rlt' stage-1 requires --vision_threshold"
            v_kwargs = dict(threshold=vision_threshold, mask_space=vision_mask_space,
                            metric=vision_metric, refresh_every=vision_refresh_every)
        else:
            v_kwargs = {}
        model.vision_reducer = build_vision_reducer(vision_method, model.vision_tower, **v_kwargs)
        logger.info(f'vision reduction: method={vision_method} '
                    + ' '.join(f'{k}={v}' for k, v in v_kwargs.items()))
        if token_pruner is None:
            logger.warning('stage-1 without stage-2: the KV-Cache stays baseline-sized '
                           'but now holds reused (approximate) features, and encode '
                           'throughput barely moves (the vision tower is ~14% of it).')

    model.language_model = patch_hf(model.language_model, **inf_llm_config)

    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
