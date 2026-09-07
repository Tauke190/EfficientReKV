import torch
from transformers import AutoConfig, VideoLlavaProcessor, VideoLlavaForConditionalGeneration
from logzero import logger

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.token_pruning import build_pruner


class VideoLlava_ReKV(VideoLlavaForConditionalGeneration, Abstract_ReKV):
    def __init__(self, config, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                 token_pruner=None):
        VideoLlavaForConditionalGeneration.__init__(self, config)
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                               token_pruner=token_pruner)
        self.processor.video_processor = self.processor.image_processor

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query} ASSISTANT:"
        if mc:
            prompt += ' Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):
        batch_size, frames, channels, height, width = pixel_values_videos.shape  # (B, Nv, 3, H, W)
        _, video_features, _ = self._get_vision_features(
            pixel_values_videos=pixel_values_videos,
            vision_feature_layer=self.config.vision_feature_layer,
            vision_feature_select_strategy=self.config.vision_feature_select_strategy
        )  # (Nv, 257, D)
        video_features = self.multi_modal_projector(video_features)  # (Nv, 257, D)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)  # (B, Nv*257, D)
        return video_features
    
    def _encode_video_chunk(self, video_chunk):  # (Nv, H, W, 3)
        pixel_values_videos = self.processor.video_processor(images=None, videos=video_chunk, return_tensors="pt").pixel_values_videos.to(self.device, self.dtype)  # (1, Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*257, D)
        # Goes through _ingest_video_features rather than calling the LM directly, which is
        # what puts stage-2 pruning and its block-alignment buffering on this path -- the
        # same route llava_ov and longva take. The n_local assertion that used to sit here
        # now lives in _forward_features, and fires on the same condition.
        self._ingest_video_features(video_features, n_frames=video_chunk.shape[0])

    @torch.inference_mode()
    def encode_video(self, video, encode_chunk_size=8):  # video: (Nv, H, W, 3)
        super().encode_video(video, encode_chunk_size)

    @torch.inference_mode()
    def question_answering(self, input_text, max_new_tokens=128, retrieved_indices=None):
        # A question asked mid-stream must see every frame that has arrived, including one
        # whose tokens are still short of a full block.
        self.flush_stream()

        device = self.device
        stop_token_ids = [self.processor.tokenizer.eos_token_id]

        output_ids = []
        stopped = False

        # NOTE: Only input the question to perform retrieval.
        input_ids = self.processor.tokenizer(input_text['question']).input_ids[1:]  # remove <s>
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # retrieval mode
            layer_kv.set_retrieval()
        
        if retrieved_indices is None:  # Internal retrieval
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        else:  # External retrieval
            for layer_kv in self.kv_cache:
                assert layer_kv.block_size == self.n_frame_tokens, f'block_size: {layer_kv.block_size}, n_frame_tokens: {self.n_frame_tokens}'
                layer_kv.set_retrieved_block_indices(retrieved_indices)
            out = self.language_model(input_ids=input_ids, use_cache=True, past_key_values=self.kv_cache)
            past_key_values = out.past_key_values  # Retrieved KV-Cache: L x 2 x (B, h, N, Dh)
        
        for layer_kv in self.kv_cache:  # reset to default
            layer_kv.reset_retrieval()

        for i in range(max_new_tokens):
            if i == 0:  # prefill
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids[1:]  # remove <s>
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = out.logits
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

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i == max_new_tokens - 1 or stopped:
                break

        self.last_generated_tokens = len(output_ids)

        output = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        
        return output


def load_model(model_path='model_zoo/Video-LLaVA-7B-hf', n_init=None, n_local=3084, topk=8, chunk_size=1,
               max_cached_block=128,
               prune_method=None, prune_threshold=None, prune_metric='cosine',
               prune_refresh_every=0, prune_log_percentiles=False):
    """Load Video-LLaVA with ReKV, and optionally stage-2 (memory-side) token pruning.

    Stage 2 only. Stage 1 (model/vision_reduction.py) is a SigLIP implementation and does
    not apply to Video-LLaVA's CLIP ViT-L/14 tower.

    DO NOT frame-match this backbone to the others. Its LM is Vicuna-7B, whose context is
    4096 tokens -- against 32768 for llava_ov and 224000 for longva. At 257 tokens a frame
    that is ~15 frames in total, so the 76.5-frame local window the other two run at
    (n_local 15000 / 11020) is 4.8x more context than this model has ever seen. Nothing
    raises: the question is attended at RoPE positions outside the trained range, the
    logits degenerate, the first sampled token is EOS, and the answer decodes to ''. It
    fails silently and progressively -- early questions in a video answer normally and
    later ones come back empty as the window fills, which reads like a model defect rather
    than a misconfiguration. Measured on one ovobench video: at n_local=6168 the last 5 of
    7 questions were empty and accuracy was 0%; at n_local=3084 none were empty and
    accuracy was 42.9%. The assertion below exists so this is an error, not a silence.

    So n_local defaults to 3084 (12 frames) and topk to 8 (2056 retrieved tokens), both
    inside 4096 with room for the question and the generation. These are this backbone's
    own defaults, not a shared budget -- a cross-backbone comparison has to either exclude
    Video-LLaVA or drop every backbone to a window this one can hold.

    `prune_threshold` is a distance in feature space, and this feature space is neither
    LLaVA-OneVision's nor LongVA's -- a different projector and no spatial pooling at all,
    so a frame here is 257 tokens against their 196 and 144. Measured on ovobench at 1 fps,
    llava_ov needs ~0.6 to reach a 16% keep rate while longva reaches 28% at 0.2, i.e. the
    useful range moves by more than a factor of two between two backbones that are closer
    to each other than either is to this one. Calibrate before trusting a number here.
    """
    device = 'cuda'
    n_frame_tokens = 257
    processor = VideoLlavaProcessor.from_pretrained(model_path)

    # Off by default: with no method selected the encode path is byte-for-byte the
    # baseline (every frame contributes exactly n_frame_tokens, so one block is one frame
    # and nothing is ever buffered).
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

    init_prompt = 'USER: '
    init_prompt_ids = processor.tokenizer(init_prompt, return_tensors="pt").input_ids.to(device)
    _n_init = init_prompt_ids.shape[1] if n_init is None else n_init

    # The failure this prevents is silent, which is why it is an assertion and not a note in
    # the docstring: over-running Vicuna's 4096 positions does not raise anywhere, it just
    # returns '' for the questions asked once the window has filled. `margin` is the
    # question, the formatted choices and up to max_new_tokens of generation, all of which
    # sit after the local window in the same position space.
    margin = 384
    ctx = AutoConfig.from_pretrained(model_path).text_config.max_position_embeddings
    for name, need in (('n_local', _n_init + n_local), ('topk*block_size', _n_init + topk * n_frame_tokens)):
        assert need + margin <= ctx, (
            f'{name} puts {need} tokens in context, and with ~{margin} for the question and '
            f'its answer that exceeds {model_path}\'s {ctx}-token limit. Vicuna-7B holds '
            f'~{(ctx - margin) // n_frame_tokens} frames at {n_frame_tokens} tokens each, so '
            f'do not carry n_local/retrieve_size over from llava_ov or longva -- their LMs '
            f'have 32k and 224k contexts. Use this backbone\'s own defaults.'
        )

    inf_llm_config = {
        'n_init': _n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        # Exposed because it is far more expensive here than on the other backbones, though
        # the default matches theirs. CudaCache allocates it eagerly and per layer --
        # max_cached_block x (n_kv_heads x block_size x dim_head x 2) x 2 bytes x n_layers
        # -- and Vicuna-7B is MHA (32 KV heads) where Qwen2-7B is GQA (4), on 257 tokens per
        # block against 196. One cached block is therefore 4.2 MB here versus 0.35 MB on
        # llava_ov, so 128 blocks costs 17.25 GB rather than 1.44 GB. At the defaults above
        # that still fits: weights 14.0 + cache 17.25 + global_buffer 1.08 + n_local window
        # 1.62 = 33.9 GB on a 47.4 GB A6000. It is only a problem if n_local or topk are
        # pushed up -- which for this backbone means past its 4096 context anyway.
        #
        # It is a cache, so lowering it costs retrieval speed (more host->device copies of
        # offloaded blocks) and never correctness. 16 gives 2.16 GB if you need the room.
        'max_cached_block': max_cached_block,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = VideoLlava_ReKV.from_pretrained(
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
    )
    # Attached after from_pretrained, not passed through it. Unrecognised kwargs are
    # forwarded to GenerationConfig.from_pretrained, and whether they get set as attributes
    # there depends on the checkpoint: a generation_config.json carrying
    # "_from_model_config": true makes GenerationConfig.__init__ drop them, one without it
    # does not -- and the pruner then lands on the generation config, whose __repr__
    # json.dumps() it, killing the load with "Object of type StreamingTokenPruner is not
    # JSON serializable". That is exactly what happened on LongVA (see model/longva_rekv.py).
    # Video-LLaVA-7B-hf is not in model_zoo here, so which way its generation_config falls
    # is unverified -- hence the safe route rather than the lucky one. Note `processor=`
    # above is non-serializable too and still goes through from_pretrained, so if a load
    # ever dies that way, that is the next thing to move down here.
    model.token_pruner = token_pruner
    model.language_model = patch_hf(model.language_model, **inf_llm_config)
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
