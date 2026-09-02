"""The block draft head, and what it reads off the target.

The head is a small block-diffusion decoder in the style of DFlash
(https://arxiv.org/abs/2602.06036, MIT licensed). GLANCE does not change its
architecture. What changes is what it is fed and what is done with its output:
it reads the *fused* vision-language hidden states of an unmodified VLM, never
raw visual tokens and never a pruned or compressed copy of the image, and its
per-offset marginals are spent on width rather than on depth.

Because the head is an ordinary module, this file stays deliberately thin. Any
object exposing ``block_size``, ``mask_token_id``, ``target_layer_ids`` and a
forward over ``target_hidden`` and ``noise_embedding`` will decode.
"""

from __future__ import annotations

__all__ = ["context_feature", "kept_layer_ids", "load_block_head"]


def context_feature(hidden_states, layer_ids):
    """Concatenate the target's hidden states at the kept layers.

    ``hidden_states[0]`` is the embedding output, so layer ``i`` sits at index
    ``i + 1``. On a vision-language target these states are already fused, so
    the head inherits the image for free and the drafter pays nothing per step
    for having seen it.
    """
    import torch
    return torch.cat([hidden_states[i + 1] for i in layer_ids], dim=-1)


def kept_layer_ids(n_target_layers, n_head_layers):
    """Which target layers a head of this depth reads, spread over the stack.

    A single-layer head reads the middle of the target. Deeper heads spread
    their taps evenly from layer 1 to three layers below the top, the same
    placement the reference block-diffusion head uses.
    """
    if n_head_layers == 1:
        return [n_target_layers // 2]
    start, end = 1, n_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (n_head_layers - 1)))
            for i in range(n_head_layers)]


def load_block_head(path, dtype=None, device=None):
    """Load a trained block head from a checkpoint directory.

    Args:
        path: directory holding the head's weights and config.
        dtype: torch dtype, defaults to bfloat16.
        device: device string, defaults to ``"cuda"``.

    The head class comes from the ``dflash`` package, installed separately
    (https://z-lab.ai/projects/dflash/). Keeping it out of this repository
    avoids a vendored copy of someone else's model code drifting from the
    version you actually trained with.
    """
    import torch

    try:
        from dflash.model import DFlashDraftModel
    except ImportError as exc:
        raise ImportError(
            "the block head class lives in the dflash package: "
            "pip install 'glance-vlm[head]'") from exc

    head = DFlashDraftModel.from_pretrained(
        path, torch_dtype=dtype or torch.bfloat16)
    return head.to(device or "cuda").eval()
