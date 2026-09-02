"""One-pass block drafting with wide-tree verification.

A decoding round is five steps:

1. one draft pass fills every offset of the block at once, from the target's
   own fused hidden states at a few kept layers;
2. the per-offset marginals become a prefix-closed candidate tree
   (:mod:`glance.tree`), which costs no extra draft pass;
3. one ancestor-masked target pass verifies the whole tree;
4. the walk commits the longest path the target itself would have taken, plus
   the bonus token the target emits past the end of that path;
5. the target KV cache and the drafter's context features are gathered down to
   the committed path.

At temperature 0 the committed tokens are exactly the target's greedy output,
so the speedup is free of any change in what the model says. Above 0, each
committed token is drawn from the target's own conditional given the committed
prefix, so the output law is unchanged and the tree only decides how much of
the computation was reusable.

The same loop serves text and vision-language targets. The only difference is
how positions are laid out, which lives behind a small adapter
(:class:`StreamPositions` here, :class:`glance.vl.VisionPositions` for VLMs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .head import context_feature
from .tree import TreeNode, build_budget_tree, children_of, tree_mask

__all__ = ["generate", "chain_generate", "greedy_generate",
           "StreamPositions", "DecodeStats", "gather_cache"]


@dataclass
class DecodeStats:
    """What one prompt's decode cost and bought."""

    tau: float = 0.0
    """Mean accepted length per round, engine independent."""

    rounds: int = 0
    n_tokens: int = 0
    ms_per_token: float = 0.0
    tree_size: float = 0.0
    """Mean number of tokens sent through the verify pass, tree plus root."""

    timing: str = "legacy"
    ms_per_token_wall: float = 0.0
    prefill_ms: float = 0.0
    extended_round_frac: float = 0.0
    tau_extended: float = 0.0

    def to_dict(self):
        return asdict(self)


class StreamPositions:
    """Plain 1-D positions. The default, and what text targets want.

    Three hooks, because the speculative path and the autoregressive oracle
    place positions differently. The speculative path always states positions
    explicitly, since a tree pass has several tokens at the same one. The
    oracle leaves them to the model, which is what a stock decode would do.
    """

    def bind(self, target, input_ids):
        """Called once per prompt, before either prefill."""

    def prefill_inputs(self, input_ids, positions):
        """Extra target kwargs for the speculative prefill."""
        return {"position_ids": positions}

    def ar_prefill_inputs(self, input_ids):
        """Extra target kwargs for the autoregressive oracle's prefill."""
        n_in = input_ids.shape[1]
        return {"cache_position": torch.arange(n_in, device=input_ids.device)}

    def __call__(self, positions):
        """Map 1-D stream positions to whatever the target's rope expects."""
        return positions


def gather_cache(cache, n_prefix, keep):
    """Drop every packed KV entry except the committed path, in order.

    ``keep`` is packed indices, so ``0`` is the committed token and ``i + 1``
    is node ``i``. Both the current and the legacy ``DynamicCache`` layouts are
    handled, since which one is in play depends on the transformers version.
    """
    if hasattr(cache, "layers") and cache.layers and hasattr(cache.layers[0], "keys"):
        layers = cache.layers
        device = layers[0].keys.device
    else:
        layers = None
        device = cache.key_cache[0].device

    index = torch.arange(n_prefix, device=device)
    if keep:
        tail = torch.tensor([n_prefix + i for i in keep], device=device)
        index = torch.cat([index, tail])

    if layers is not None:
        for layer in layers:
            layer.keys = layer.keys.index_select(2, index)
            layer.values = layer.values.index_select(2, index)
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].index_select(2, index)
            cache.value_cache[i] = cache.value_cache[i].index_select(2, index)
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = index.numel()
    return cache


def _pick(logits, temperature):
    if temperature < 1e-5:
        return int(logits.argmax())
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, 1))


def _entropy_nats(logits):
    """Shannon entropy of the target's next-token distribution, in nats.

    This is the quantity the draftability law is indexed by, and it is read off
    the target rather than the drafter, so it is a property of the workload and
    not of how it is being drafted.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    return float(-(logp.exp() * logp).sum())


@torch.inference_mode()
def generate(head, target, input_ids, max_new_tokens, stop_token_ids=None,
             budget=63, branch_k=8, temperature=0.0, positions=None,
             return_stats=False, extend_budget=0, extend_branch=4,
             extend_threshold=0.30, timing="legacy", on_round=None):
    """Decode ``input_ids`` with one draft pass and one verify pass a round.

    Args:
        head: block draft head. Needs ``block_size``, ``mask_token_id``,
            ``target_layer_ids`` and a forward taking ``target_hidden`` and
            ``noise_embedding``. See :mod:`glance.head`.
        target: frozen causal LM, or VLM, that is being accelerated. It is
            never modified and never fine-tuned.
        input_ids: ``(1, n)`` prompt.
        max_new_tokens: stop after this many committed tokens.
        stop_token_ids: ids that end generation.
        budget: candidate tree size. Width is bought inside the verify pass,
            so this is the axis that is cheap here and expensive for a chain
            drafter.
        branch_k: branch factor per offset.
        temperature: 0 for greedy, which is the lossless setting.
        positions: position adapter. Defaults to :class:`StreamPositions`;
            pass :class:`glance.vl.VisionPositions` for a VLM.
        return_stats: also return a :class:`DecodeStats`.
        extend_budget: if positive, and the drafter's greedy spine is confident
            enough, spend one extra draft pass on offsets past the block to
            hang a subtree off the spine leaf. Verified in the same target
            pass, so accepted length can exceed the block size. Requires a head
            trained with a variable mask span.
        extend_branch: branch factor for that extension.
        extend_threshold: minimum spine path probability that triggers it.
        timing: ``"legacy"`` measures from before the prefill;
            ``"decode_only"`` opens the window after the prefill and after the
            first round's drafting, which is the convention block-diffusion
            drafters report, and synchronises before both reads.
        on_round: called once a round with the round's entropy, accepted
            length and tree size. This is how the round logs behind the
            draftability law are produced. The entropy is the target's own
            next-token distribution at the committed token, in nats, and is
            computed only when a callback is given.

    Returns:
        The committed ids, or ``(ids, stats)`` when ``return_stats``.
    """
    if timing not in ("legacy", "decode_only"):
        raise ValueError(f"timing must be 'legacy' or 'decode_only', got {timing!r}")
    positions = positions or StreamPositions()

    device = input_ids.device
    block = head.block_size
    mask_id = head.mask_token_id
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens
    dtype = next(target.parameters()).dtype
    embed = target.get_input_embeddings()

    out_ids = torch.full((1, max_len + 2 * block + 4), mask_id,
                         dtype=torch.long, device=device)
    stream = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    kv_target = DynamicCache()
    kv_head = DynamicCache()

    positions.bind(target, input_ids)
    want_window = timing == "decode_only"
    window_t0 = None
    window_elapsed = None

    wall_t0 = time.perf_counter()
    out = target(input_ids,
                 past_key_values=kv_target, use_cache=True, logits_to_keep=1,
                 output_hidden_states=True,
                 **positions.prefill_inputs(input_ids, stream[:, :n_in]))
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = _pick(out.logits[0, -1], temperature)
    fused = context_feature(out.hidden_states, head.target_layer_ids)

    start = n_in
    taus = []
    tree_sizes = []
    taus_extended = []
    n_extended = 0
    log_threshold = float(torch.log(torch.tensor(max(extend_threshold, 1e-9))))

    while start < max_len:
        # 1. one draft pass over the whole block
        block_ids = out_ids[:, start:start + block].clone()
        block_ids[:, 1:] = mask_id
        hidden = head(target_hidden=fused,
                      noise_embedding=embed(block_ids),
                      position_ids=stream[:, kv_head.get_seq_length():start + block],
                      past_key_values=kv_head, use_cache=True, is_causal=False)
        draft_logits = target.lm_head(hidden[:, 1 - block:, :])[0]
        kv_head.crop(start)

        # 2. the candidate tree, free of any further draft pass
        top = torch.topk(F.log_softmax(draft_logits.float(), dim=-1), branch_k, dim=-1)
        top_ids, top_logp = top.indices.cpu(), top.values.cpu()
        nodes = build_budget_tree(top_ids, top_logp, budget, branch_k)

        extended = (extend_budget > 0
                    and float(top_logp[:, 0].sum()) >= log_threshold
                    and start + 2 * block < out_ids.shape[1])
        if extended:
            nodes = _extend_past_block(
                nodes, head, target, embed, out_ids, stream, fused, kv_head,
                top_ids, start, block, mask_id, extend_budget, extend_branch,
                device)
            n_extended += 1

        if want_window and window_t0 is None:
            torch.cuda.synchronize()
            window_t0 = time.perf_counter()

        # 3. one ancestor-masked target pass over the whole tree
        packed = torch.tensor(
            [[int(out_ids[0, start])] + [n.token for n in nodes]], device=device)
        packed_pos = torch.tensor(
            [[start] + [start + n.depth for n in nodes]], device=device)
        n_prefix = kv_target.get_seq_length()
        out = target(packed,
                     position_ids=positions(packed_pos),
                     past_key_values=kv_target, use_cache=True,
                     attention_mask=tree_mask(n_prefix, nodes, device, dtype),
                     output_hidden_states=True)
        packed_logits = out.logits[0]

        # 4. walk the tree along what the target itself would have said
        children = children_of(nodes)
        path = []
        cursor = -1
        while True:
            token = _pick(packed_logits[0 if cursor == -1 else cursor + 1], temperature)
            hit = next((c for c in children.get(cursor, []) if nodes[c].token == token),
                       None)
            if hit is None:
                bonus = token
                break
            path.append(hit)
            cursor = hit

        accepted = len(path)
        taus.append(accepted + 1)
        tree_sizes.append(1 + len(nodes))
        if extended:
            taus_extended.append(accepted + 1)

        if on_round is not None:
            on_round({
                "round": len(taus) - 1,
                "entropy": _entropy_nats(packed_logits[0]),
                "accepted": accepted,
                "tau": accepted + 1,
                "tree_size": 1 + len(nodes),
                "extended": bool(extended),
            })

        # 5. commit, then gather both caches down to the committed path
        for offset, node_index in enumerate(path):
            out_ids[0, start + 1 + offset] = nodes[node_index].token
        out_ids[0, start + accepted + 1] = bonus

        keep = [0] + [i + 1 for i in path]
        gather_cache(kv_target, n_prefix, keep)
        fused = context_feature(out.hidden_states, head.target_layer_ids)[0][keep][None]

        start += accepted + 1
        if stop_token_ids and any(t in out_ids[0, n_in:start].tolist()
                                  for t in stop_token_ids):
            break

    wall = time.perf_counter() - wall_t0
    final = out_ids[:, :min(start + 1, max_len)]
    if stop_token_ids:
        stops = torch.tensor(stop_token_ids, device=device)
        hit = torch.isin(final[0, n_in:], stops).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            final = final[:, :n_in + hit[0] + 1]
    if window_t0 is not None:
        torch.cuda.synchronize()
        window_elapsed = time.perf_counter() - window_t0

    if not return_stats:
        return final

    n_out = final.shape[1] - n_in
    elapsed = wall if window_elapsed is None else window_elapsed
    stats = DecodeStats(
        tau=sum(taus) / max(len(taus), 1),
        rounds=len(taus),
        n_tokens=n_out,
        ms_per_token=elapsed * 1000.0 / max(n_out, 1),
        tree_size=sum(tree_sizes) / max(len(tree_sizes), 1),
        timing="decode_only" if window_elapsed is not None else "legacy",
        ms_per_token_wall=wall * 1000.0 / max(n_out, 1),
        prefill_ms=(window_t0 - wall_t0) * 1000.0 if window_t0 is not None else 0.0,
        extended_round_frac=n_extended / max(len(taus), 1),
        tau_extended=(sum(taus_extended) / len(taus_extended)) if taus_extended else 0.0,
    )
    return final, stats


def _extend_past_block(nodes, head, target, embed, out_ids, stream, fused,
                       kv_head, top_ids, start, block, mask_id,
                       extend_budget, extend_branch, device):
    """Hang a subtree past the block off the drafter's own greedy spine.

    One extra draft pass, with the spine revealed rather than masked, prices
    the offsets from ``block`` to ``2 * block``. The subtree is verified in the
    same target pass as everything else, and candidates only ever widen, so the
    acceptance walk and its guarantee are untouched.
    """
    by_parent = {(n.parent, n.rank): i for i, n in enumerate(nodes) if n.rank == 0}
    cursor = -1
    for depth in range(1, block):
        nxt = by_parent.get((cursor, 0))
        if nxt is None or nodes[nxt].depth != depth:
            nodes.append(TreeNode(int(top_ids[depth - 1][0]), depth, cursor, 0))
            nxt = len(nodes) - 1
            by_parent[(cursor, 0)] = nxt
        cursor = nxt
    spine_leaf = cursor

    revealed = torch.full((1, 2 * block), mask_id, dtype=torch.long, device=device)
    revealed[0, 0] = out_ids[0, start]
    revealed[0, 1:block] = top_ids[:, 0].to(device)
    hidden = head(target_hidden=fused[:, :0, :],
                  noise_embedding=embed(revealed),
                  position_ids=stream[:, start:start + 2 * block],
                  past_key_values=kv_head, use_cache=True, is_causal=False)
    kv_head.crop(start)
    logits = target.lm_head(hidden[:, block:2 * block, :])[0]

    top = torch.topk(F.log_softmax(logits.float(), dim=-1), extend_branch, dim=-1)
    base = len(nodes)
    for node in build_budget_tree(top.indices.cpu(), top.values.cpu(),
                                  extend_budget, extend_branch):
        nodes.append(TreeNode(
            node.token, block - 1 + node.depth,
            spine_leaf if node.parent == -1 else base + node.parent, node.rank))
    return nodes


@torch.inference_mode()
def chain_generate(head, target, input_ids, max_new_tokens, stop_token_ids=None,
                   temperature=0.0, positions=None, return_stats=False):
    """The same head, run as a width-1 chain instead of a tree.

    One draft pass still fills the block, but only the single most likely
    continuation is verified, so a round commits at most the block's greedy
    spine. This is the ablation that separates the head from the tree: it holds
    the drafter, the training and the draft budget fixed, and removes only the
    width that one-pass drafting makes affordable.
    """
    positions = positions or StreamPositions()
    positions.bind(target, input_ids)

    device = input_ids.device
    block = head.block_size
    mask_id = head.mask_token_id
    embed = target.get_input_embeddings()
    n_in = input_ids.shape[1]
    max_len = n_in + max_new_tokens

    out_ids = torch.full((1, max_len + block), mask_id, dtype=torch.long, device=device)
    stream = torch.arange(out_ids.shape[1], device=device).unsqueeze(0)
    kv_target = DynamicCache()
    kv_head = DynamicCache()

    t0 = time.perf_counter()
    out = target(input_ids, past_key_values=kv_target, use_cache=True,
                 logits_to_keep=1, output_hidden_states=True,
                 **positions.prefill_inputs(input_ids, stream[:, :n_in]))
    out_ids[:, :n_in] = input_ids
    out_ids[:, n_in] = _pick(out.logits[0, -1], temperature)
    fused = context_feature(out.hidden_states, head.target_layer_ids)

    start = n_in
    taus = []
    while start < max_len:
        block_ids = out_ids[:, start:start + block].clone()
        block_pos = stream[:, start:start + block]
        hidden = head(target_hidden=fused,
                      noise_embedding=embed(block_ids),
                      position_ids=stream[:, kv_head.get_seq_length():start + block],
                      past_key_values=kv_head, use_cache=True, is_causal=False)
        draft_logits = target.lm_head(hidden[:, 1 - block:, :])
        kv_head.crop(start)
        if temperature < 1e-5:
            block_ids[:, 1:] = draft_logits.argmax(dim=-1)
        else:
            block_ids[:, 1:] = torch.tensor(
                [[_pick(draft_logits[0, j], temperature)
                  for j in range(draft_logits.shape[1])]], device=device)

        out = target(block_ids, position_ids=positions(block_pos),
                     past_key_values=kv_target, use_cache=True,
                     output_hidden_states=True)
        if temperature < 1e-5:
            verified = out.logits.argmax(dim=-1)
        else:
            verified = torch.tensor(
                [[_pick(out.logits[0, j], temperature)
                  for j in range(out.logits.shape[1])]], device=device)

        accepted = int((block_ids[:, 1:] == verified[:, :-1]).cumprod(dim=1).sum())
        out_ids[:, start:start + accepted + 1] = block_ids[:, :accepted + 1]
        out_ids[0, start + accepted + 1] = verified[0, accepted]
        taus.append(accepted + 1)

        start += accepted + 1
        kv_target.crop(start)
        fused = context_feature(out.hidden_states,
                               head.target_layer_ids)[:, :accepted + 1, :]
        if stop_token_ids and any(t in out_ids[0, n_in:start].tolist()
                                  for t in stop_token_ids):
            break

    wall = time.perf_counter() - t0
    final = out_ids[:, :min(start + 1, max_len)]
    if stop_token_ids:
        stops = torch.tensor(stop_token_ids, device=device)
        hit = torch.isin(final[0, n_in:], stops).nonzero(as_tuple=True)[0]
        if hit.numel() > 0:
            final = final[:, :n_in + hit[0] + 1]
    if not return_stats:
        return final

    n_out = final.shape[1] - n_in
    return final, DecodeStats(
        tau=sum(taus) / max(len(taus), 1),
        rounds=len(taus),
        n_tokens=n_out,
        ms_per_token=wall * 1000.0 / max(n_out, 1),
        tree_size=float(block),
        ms_per_token_wall=wall * 1000.0 / max(n_out, 1),
    )


@torch.inference_mode()
def greedy_generate(target, input_ids, max_new_tokens, stop_token_ids=None,
                    positions=None, return_stats=False):
    """Plain autoregressive greedy decode.

    This is both the speed baseline and the correctness oracle: a lossless run
    has to reproduce this token for token.

    With ``return_stats`` the decode window opens after the prefill and the
    first token, and closes after the last step, with a synchronise before both
    reads. The denominator counts every output token including the one the
    prefill emitted, which is the convention the speculative path uses too.
    """
    positions = positions or StreamPositions()
    positions.bind(target, input_ids)
    device = input_ids.device
    cache = DynamicCache()
    n_in = input_ids.shape[1]

    if return_stats:
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()
    out = target(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 **positions.ar_prefill_inputs(input_ids))
    tokens = [int(out.logits[0, -1].argmax())]
    if return_stats:
        torch.cuda.synchronize()
        t_decode = time.perf_counter()

    for _ in range(max_new_tokens - 1):
        out = target(torch.tensor([[tokens[-1]]], device=device),
                     past_key_values=cache, use_cache=True,
                     cache_position=torch.tensor([cache.get_seq_length()],
                                                 device=device))
        tokens.append(int(out.logits[0, -1].argmax()))
        if stop_token_ids and tokens[-1] in stop_token_ids:
            break

    if return_stats:
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    final = torch.cat([input_ids, torch.tensor([tokens], device=device)], dim=1)
    if not return_stats:
        return final
    n_out = len(tokens)
    return final, DecodeStats(
        tau=1.0,
        rounds=n_out,
        n_tokens=n_out,
        ms_per_token=(t_end - t_decode) * 1000.0 / max(n_out, 1),
        tree_size=1.0,
        timing="decode_only",
        ms_per_token_wall=(t_end - t_prefill) * 1000.0 / max(n_out, 1),
        prefill_ms=(t_decode - t_prefill) * 1000.0,
    )
