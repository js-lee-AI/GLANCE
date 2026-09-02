"""Prefix-closed candidate trees built from one block-draft pass.

A block head fills every offset of a block in a single forward pass, so the
offsets are conditionally independent given the context and the committed
token. The score of a candidate path is therefore the product of its
offset-wise marginals, and the budget-N tree of highest-scoring paths is the
best-first expansion of that product. No draft pass is spent on width.

Nothing here needs a GPU, a model, or torch: the tree is built from top-k
token ids and log-probabilities alone, which makes it cheap to test and to
reason about.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

__all__ = ["TreeNode", "build_budget_tree", "tree_mask", "path_of", "children_of"]


@dataclass
class TreeNode:
    """One candidate token.

    Attributes:
        token: vocabulary id.
        depth: block offset, 1..B-1. Depth 0 is the committed token and is not
            a node; it is the implicit root, addressed as parent ``-1``.
        parent: index into the node list, or ``-1`` for a child of the root.
        rank: this token's rank among the top-k at its own depth.
    """

    token: int
    depth: int
    parent: int
    rank: int


def build_budget_tree(topk_ids, topk_logp, budget, branch_k=None):
    """Return the ``budget`` highest path-probability candidates, prefix-closed.

    Args:
        topk_ids: ``(B-1, K)`` top-k token ids per block offset. A torch
            tensor, a numpy array or a list of lists all work.
        topk_logp: ``(B-1, K)`` matching log-probabilities.
        budget: number of nodes to keep. This is the verifier's width, and the
            only knob that trades target compute for accepted length.
        branch_k: branch factor. Defaults to the full width of ``topk_ids``.

    Returns:
        Nodes in packed order, every parent before its children, so the list
        can be fed straight to :func:`tree_mask` and to the target pass.

    The expansion is best-first on cumulative log-probability, which under
    offset-wise independence is exactly the optimal prefix-closed set of that
    size. Ties are broken by insertion order, so the result is deterministic.
    """
    ids = [list(row) for row in topk_ids]
    logp = [[float(x) for x in row] for row in topk_logp]
    depth_max = len(ids)
    width = len(ids[0]) if depth_max else 0
    k = width if branch_k is None else min(branch_k, width)

    nodes: list[TreeNode] = []
    heap: list[tuple[float, int, int, int, int]] = []
    order = 0
    for r in range(k):
        heapq.heappush(heap, (-logp[0][r], 1, -1, r, order))
        order += 1

    while heap and len(nodes) < budget:
        neg_cum, depth, parent, rank, _ = heapq.heappop(heap)
        here = len(nodes)
        nodes.append(TreeNode(int(ids[depth - 1][rank]), depth, parent, rank))
        if depth < depth_max:
            for r in range(k):
                heapq.heappush(
                    heap, (neg_cum - logp[depth][r], depth + 1, here, r, order))
                order += 1
    return nodes


def children_of(nodes):
    """Adjacency map from node index (``-1`` = root) to its children."""
    children = {i: [] for i in range(-1, len(nodes))}
    for i, node in enumerate(nodes):
        children[node.parent].append(i)
    return children


def path_of(nodes, index):
    """Root-to-node path as a list of node indices, root first."""
    path = []
    while index != -1:
        path.append(index)
        index = nodes[index].parent
    path.reverse()
    return path


def tree_mask(n_prefix, nodes, device, dtype):
    """Additive ``(1, 1, q, k)`` attention mask for the packed verify pass.

    Packed position 0 is the committed token; packed position ``i + 1`` is
    ``nodes[i]``. Every packed token attends to the whole prefix KV and, inside
    the pack, to its own ancestor chain. One target pass therefore scores every
    candidate path at once, each under exactly the context it would have had.

    Visibility rows are built by inheritance on the CPU and moved to the device
    in a single copy, which keeps the per-round host-device traffic at one
    transfer regardless of the budget.
    """
    import numpy as np
    import torch

    n_pack = 1 + len(nodes)
    visible = np.zeros((n_pack, n_pack), dtype=bool)
    visible[0, 0] = True
    for i, node in enumerate(nodes):
        row = i + 1
        visible[row] = visible[node.parent + 1]
        visible[row, row] = True

    blocked = torch.from_numpy(~visible).to(device)
    mask = torch.zeros((1, 1, n_pack, n_prefix + n_pack), device=device, dtype=dtype)
    mask[0, 0, :, n_prefix:] = blocked * torch.finfo(dtype).min
    return mask
