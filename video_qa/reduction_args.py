"""CLI flags for the two token-reduction stages.

Its own module, importing nothing but argparse, because video_qa/run_eval.py is a
launcher that spawns workers and must not pull in torch or any model backend just to
parse a flag. video_qa/base.py re-exports `add_reduction_args`, so both the launcher
and the workers register exactly the same flags with exactly the same defaults -- a
sweep in which the two disagree silently evaluates a different model than it reports.

Two independent stages, both defaulting to 'none' (the untouched baseline). Measured
split of `_encode_video_chunk` (RVS-Ego @0.5fps, llava_ov_0.5b, GPU preprocessing):
preprocessing 2.5%, vision tower 13.6%, LM prefill 84.0%.

  --vision_method   Stage 1, inside the vision tower (model/vision_reduction.py).
                    Attacks the 13.6%. KV-Cache size is unchanged.
  --prune_method    Stage 2, at the KV-Cache boundary (model/token_pruning.py).
                    Attacks the 84%, and shrinks KV RAM and retrieval cost with it.

Measured end to end: stage 1 alone +1.6% frames/s, stage 2 alone +29%, both +58%.

Their thresholds live on different scales -- SigLIP patch embeddings vs projected,
pooled LLM-space tokens -- and must be calibrated separately. Do not copy one to the
other.
"""


def add_reduction_args(parser):
    """Register both stages' flags on an ArgumentParser and return it."""
    # ---- Stage 1: encoder-side ------------------------------------------------------
    parser.add_argument("--vision_method", type=str, default='none',
                        help="Reduction method applied inside the vision tower; must be "
                             "registered in model/vision_reduction.py (currently: rlt). "
                             "'none' = run the vision tower unmodified.")
    parser.add_argument("--vision_threshold", type=float, default=None,
                        help="Skip re-encoding a patch when its distance to the embedding "
                             "it would reuse is below this. On the SigLIP patch-embedding "
                             "scale; calibrate on your own footage.")
    parser.add_argument("--vision_mask_space", type=str, default='embed',
                        choices=['embed', 'pixel'],
                        help="Compare patch embeddings (default) or raw preprocessed "
                             "pixels. Thresholds are not comparable between the two.")
    parser.add_argument("--vision_metric", type=str, default='cosine',
                        choices=['l2', 'cosine'],
                        help="cosine is chunk-size invariant; l2 is not. embed space only.")
    parser.add_argument("--vision_refresh_every", type=int, default=0,
                        help="Force-encode every Nth frame in full. Bounds worst-case "
                             "staleness; 0 = off.")

    # ---- Stage 2: memory-side -------------------------------------------------------
    parser.add_argument("--prune_method", type=str, default='none',
                        help="Reduction method applied at the KV-Cache boundary; must be "
                             "registered in model/token_pruning.py (currently: rlt). The "
                             "--prune_* flags below are rlt's hyperparameters.")
    parser.add_argument("--prune_threshold", type=float, default=None,
                        help="Drop a visual token when its distance to the feature it "
                             "would reuse is below this. Calibrate with "
                             "--prune_log_percentiles; do not transplant a value.")
    parser.add_argument("--prune_metric", type=str, default='cosine',
                        choices=['l2', 'cosine'],
                        help="cosine is chunk-size invariant; l2 is not (see "
                             "model/token_pruning.py).")
    parser.add_argument("--prune_refresh_every", type=int, default=0,
                        help="Force-keep every Nth frame to bound staleness. 0 = off.")
    return parser
