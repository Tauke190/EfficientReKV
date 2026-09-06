import torch
from logzero import logger

from transformers import AutoTokenizer
from longva.model import LlavaQwenForCausalLM

from model.patch import patch_hf
from model.abstract_rekv import Abstract_ReKV
from model.token_pruning import build_pruner


class LongVA_ReKV(LlavaQwenForCausalLM, Abstract_ReKV):
    def __init__(self, config, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                 token_pruner=None):
        LlavaQwenForCausalLM.__init__(self, config)
        processor = self.get_model().get_vision_tower().image_processor
        Abstract_ReKV.__init__(self, processor, n_frame_tokens, init_prompt_ids, n_local, topk, chunk_size,
                               token_pruner=token_pruner)

    def get_prompt(self, query, mc=False):
        prompt =  f"\n{query}<|im_end|>\n<|im_start|>assistant\n"
        if mc:
            prompt += 'Best option: ('
        return prompt

    def _get_video_features(self, pixel_values_videos):  # (Nv, 3, H, W)
        video_features = self.get_model().get_vision_tower()(pixel_values_videos)  # (Nv, 576, 1024)
        video_features = self.get_model().mm_projector(video_features)  # (Nv, 576, 3584)
        video_features = self.get_2dPool(video_features)  # (Nv, 144, 3584)
        video_features = video_features.flatten(0, 1).unsqueeze(0)  # (1, Nv*144, 3584)
        return video_features

    @torch.inference_mode()
    def _encode_video_chunk(self, video_chunk):  # (Nv, H, W, 3)
        # Overridden only for the preprocessing call: LongVA's vision tower carries a plain
        # image processor (`preprocess(...).pixel_values`), not the video processor the base
        # class uses. Ingestion itself goes through `_ingest_video_features` exactly as every
        # other backend does, which is what puts stage-2 pruning and its block-alignment
        # buffering on this path.
        pixel_values_videos = self.processor.preprocess(video_chunk, return_tensors="pt").pixel_values.to(self.device, self.dtype)  # (Nv, 3, H, W)
        video_features = self._get_video_features(pixel_values_videos)  # (1, Nv*144, D)
        self._ingest_video_features(video_features, n_frames=video_chunk.shape[0])

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
        input_ids = self.processor.tokenizer(input_text['question']).input_ids
        input_ids = torch.as_tensor([input_ids], device=device)
        for layer_kv in self.kv_cache:  # activate retrieval mode
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
                input_ids = self.processor.tokenizer(input_text['prompt']).input_ids
                input_ids = torch.as_tensor([input_ids], device=device)
                inputs_embeds = self.get_input_embeddings()(input_ids)
                out = self.language_model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                logits = self.lm_head(out['last_hidden_state'])
            else:  # decoding
                out = self.language_model(
                    input_ids=torch.as_tensor(
                        [[token]],
                        device=device,
                    ),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = self.lm_head(out['last_hidden_state'])
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


def load_model(model_path='model_zoo/LongVA-7B',
               n_init=None, n_local=8000, topk=32, chunk_size=1,
               prune_method=None, prune_threshold=None, prune_metric='cosine',
               prune_refresh_every=0, prune_log_percentiles=False):
    """Load LongVA with ReKV, and optionally stage-2 (memory-side) token pruning.

    Stage 2 only. Stage 1 (model/vision_reduction.py) is a SigLIP implementation and does
    not apply to LongVA's CLIP ViT-L tower.

    Note that `prune_threshold` is a distance in feature space, and this feature space is
    not LLaVA-OneVision's -- different tower, projector and pooling, and 144 tokens per
    frame rather than 196. A threshold carried over from an llava_ov run will not give the
    same keep rate here; calibrate with --prune_log_percentiles.
    """
    n_frame_tokens = 144
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

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

    init_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    init_prompt_ids = tokenizer(init_prompt).input_ids
    inf_llm_config = {
        'n_init': len(init_prompt_ids) if n_init is None else n_init,
        'n_local': n_local,
        'fattn': True,
        'block_size': n_frame_tokens,
        'topk': topk,
        'chunk_size': chunk_size,
        'max_cached_block': 128,
        'exc_block_size': n_frame_tokens,
        'pin_memory': True,
    }
    model = LongVA_ReKV.from_pretrained(
        model_path, 
        device_map="auto",
        low_cpu_mem_usage=True, 
        torch_dtype=torch.float16,
        n_frame_tokens=n_frame_tokens,
        init_prompt_ids=init_prompt_ids,
        n_local=n_local,
        topk=topk,
        chunk_size=chunk_size,
        token_pruner=token_pruner,
    )
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map="auto")
    processor = vision_tower.image_processor
    processor.tokenizer = tokenizer

    model = patch_hf(model, **inf_llm_config)
    model.language_model = model.model
    
    for k, v in inf_llm_config.items():
        logger.info(f'{k}: {v}')
    logger.info(f'n_frame_tokens: {n_frame_tokens}')

    model.eval()

    return model, processor
