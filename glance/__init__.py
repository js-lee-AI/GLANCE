"""GLANCE: grounded block drafting with one-pass candidate expansion.

Lossless speculative decoding for vision-language models. One draft pass fills
a whole block from the target's own fused vision-language state, the block's
per-offset marginals become a wide prefix-closed candidate tree at no extra
draft cost, and one ancestor-masked target pass verifies the tree and commits
the longest path the target itself would have taken.

    import glance

    ids, stats = glance.generate(head, target, prompt_ids, max_new_tokens=256,
                                 budget=63, return_stats=True)
    stats.tau               # accepted length per round
    stats.ms_per_token

The tree and the draftability law stand on their own and need neither a GPU
nor torch:

    nodes = glance.build_budget_tree(top_ids, top_logprobs, budget=63)
    law = glance.DraftabilityLaw.fit(entropy, accepted)
    law.expected(0.15)      # E[a | H]

Torch is imported only when a decoding entry point is first touched, so the
analysis half of the package installs and runs anywhere.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .law import DraftabilityLaw, decile_curve, expected_acceptance
from .metrics import (acceptance_length, drop_warmup, geomean, paired_bootstrap,
                      speedup)
from .tree import TreeNode, build_budget_tree, children_of, path_of

_LAZY = {
    "generate": "decode",
    "chain_generate": "decode",
    "greedy_generate": "decode",
    "gather_cache": "decode",
    "DecodeStats": "decode",
    "StreamPositions": "decode",
    "tree_mask": "tree",
    "context_feature": "head",
    "kept_layer_ids": "head",
    "load_block_head": "head",
    "VisionPositions": "vl",
    "target_class_for": "vl",
    "load_target": "vl",
}

__all__ = [
    "__version__",
    # decoding
    "generate", "chain_generate", "greedy_generate", "DecodeStats",
    "StreamPositions", "gather_cache",
    # the candidate tree
    "TreeNode", "build_budget_tree", "tree_mask", "children_of", "path_of",
    # the block head
    "context_feature", "kept_layer_ids", "load_block_head",
    # vision-language targets
    "VisionPositions", "target_class_for", "load_target",
    # the draftability law
    "DraftabilityLaw", "expected_acceptance", "decile_curve",
    # reading a run
    "acceptance_length", "geomean", "speedup", "paired_bootstrap", "drop_warmup",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
