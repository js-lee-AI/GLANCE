"""Vision-language targets.

A VLM prefills with the image and then decodes text, so the image lives in the
target's KV cache and the decode loop never touches pixels again. Two things
follow, and they are the whole of this file.

First, positions. Qwen-family VLMs use M-RoPE, whose prefill positions come
from the image grid rather than from a counter. Every position after the
prefill is a 1-D stream index shifted by the prefill's own delta and broadcast
over the three rope axes. The candidate tree needs positions stated explicitly,
since several packed tokens share one, which is exactly what
:class:`VisionPositions` supplies.

Second, the head. It never sees the image grid, only the fused hidden states,
so it stays on plain 1-D positions throughout.
"""

from __future__ import annotations

from .decode import StreamPositions

__all__ = ["VisionPositions", "target_class_for", "load_target"]

_FAMILIES = {
    "qwen3_vl": "Qwen3VLForConditionalGeneration",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "qwen2_vl": "Qwen2VLForConditionalGeneration",
}


class VisionPositions(StreamPositions):
    """M-RoPE positions for a Qwen-family VLM, plus the vision prefill inputs.

    Args:
        pixel_values: as returned by the processor.
        image_grid_thw: the image's temporal-height-width grid.

    Bind once per prompt. ``get_rope_index(ids, grid, None, None)`` is
    compatible across the three families, since image-only generation passes
    None for every video and timestamp argument.
    """

    def __init__(self, pixel_values, image_grid_thw):
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self._prefill_positions = None
        self._delta = 0

    def bind(self, target, input_ids):
        target.model.rope_deltas = None
        positions, deltas = target.model.get_rope_index(
            input_ids, self.image_grid_thw, None, None)
        self._prefill_positions = positions
        self._delta = int(deltas[0, 0])

    def prefill_inputs(self, input_ids, positions):
        return {
            "position_ids": self._prefill_positions,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }

    def ar_prefill_inputs(self, input_ids):
        import torch
        return {
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "cache_position": torch.arange(input_ids.shape[1],
                                           device=input_ids.device),
        }

    def __call__(self, positions):
        return (positions + self._delta).unsqueeze(0).expand(3, -1, -1)


def target_class_for(model_id):
    """The right conditional-generation class for a VL checkpoint.

    Resolved from the config's ``model_type``, so callers never pass a family
    flag. Qwen3-VL, Qwen2.5-VL and Qwen2-VL are supported; they differ for our
    purposes only in hidden size and layer count.
    """
    import transformers
    from transformers import AutoConfig

    model_type = getattr(AutoConfig.from_pretrained(model_id), "model_type", "")
    name = _FAMILIES.get(model_type)
    if name is None:
        raise ValueError(
            f"unsupported vision-language model_type {model_type!r} for {model_id}")
    return getattr(transformers, name)


def load_target(model_id, dtype=None, device=None, attn_implementation=None):
    """Load a frozen VL target and its processor.

    The target is never trained and never modified. It is loaded in eval mode
    and returned alongside the processor that prepares its inputs.
    """
    import torch
    from transformers import AutoProcessor

    cls = target_class_for(model_id)
    kwargs = {"torch_dtype": dtype or torch.bfloat16}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    target = cls.from_pretrained(model_id, **kwargs).to(device or "cuda").eval()
    return target, AutoProcessor.from_pretrained(model_id)
