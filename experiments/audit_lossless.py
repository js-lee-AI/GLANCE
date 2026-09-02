#!/usr/bin/env python3
"""Check that GLANCE reproduces the target's greedy output token for token.

Losslessness is gated, not assumed. For every prompt the audit decodes twice,
once autoregressively and once with GLANCE, and requires the two sequences to
be identical.

The audit runs at fp32 on purpose. In bf16 the target does not reproduce
*itself* on every prompt with no drafter in the loop at all, because a tree
pass and a single-token pass reduce in different orders and near-ties land on
either side. A bf16 mismatch would measure the arithmetic rather than the
method, so the audit first re-runs the plain autoregressive decode against
itself and reports that control alongside the result.

    python experiments/audit_lossless.py \
        --target Qwen/Qwen3-VL-8B-Instruct --head path/to/block-head \
        --data data/vl_eval --tasks docvqa,chartqa --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glance
from glance.vl import VisionPositions, load_target
from main_table import load_task


def first_divergence(reference, candidate):
    """Index of the first differing token, or None when the two agree."""
    n = min(reference.shape[1], candidate.shape[1])
    diff = (reference[0, :n] != candidate[0, :n]).nonzero(as_tuple=True)[0]
    if diff.numel() == 0:
        return None if reference.shape[1] == candidate.shape[1] else n
    return int(diff[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--head", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tasks", default="caption,textvqa,infovqa,docvqa,chartqa")
    parser.add_argument("--n", type=int, default=20, help="prompts per task")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--budget", type=int, default=63)
    parser.add_argument("--branch-k", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=896)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    target, processor = load_target(args.target, dtype=dtype, attn_implementation="sdpa")
    head = glance.load_block_head(args.head, dtype=dtype)
    stop_ids = [processor.tokenizer.eos_token_id]

    def encode(prompt, image):
        message = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(target.device)
        return (inputs["input_ids"], inputs.get("pixel_values"),
                inputs.get("image_grid_thw"))

    report = {"dtype": args.dtype, "budget": args.budget, "tasks": {}}
    audited = matched = self_consistent = 0

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        divergences = []
        control = []
        for image, prompt in load_task(task, args.data, args.n, args.max_pixels):
            input_ids, pixel_values, grid = encode(prompt, image)
            positions = VisionPositions(pixel_values, grid)

            reference = glance.greedy_generate(
                target, input_ids, args.max_new, stop_ids, positions=positions)
            again = glance.greedy_generate(
                target, input_ids, args.max_new, stop_ids, positions=positions)
            ours = glance.generate(
                head, target, input_ids, args.max_new, stop_ids,
                budget=args.budget, branch_k=args.branch_k, positions=positions)

            divergences.append(first_divergence(reference, ours))
            control.append(first_divergence(reference, again))

        n_match = sum(1 for d in divergences if d is None)
        n_control = sum(1 for d in control if d is None)
        audited += len(divergences)
        matched += n_match
        self_consistent += n_control
        report["tasks"][task] = {
            "n": len(divergences),
            "identical_to_greedy": n_match,
            "target_self_consistent": n_control,
            "first_divergence": divergences,
        }
        print(f"{task}: {n_match}/{len(divergences)} identical to greedy, "
              f"target reproduces itself on {n_control}/{len(control)}", flush=True)

    report["total"] = {"n": audited, "identical_to_greedy": matched,
                       "target_self_consistent": self_consistent}
    print(f"\naudited {audited} prompts: {matched} identical to greedy decoding, "
          f"target self-consistent on {self_consistent}")
    if matched < self_consistent:
        print("FAILED: GLANCE diverged where the target does not diverge from itself")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.out}")

    return 0 if matched >= self_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
