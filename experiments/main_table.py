#!/usr/bin/env python3
"""Accepted length and decode speed on the five vision-language tasks.

This is the main experiment. For each task it runs the autoregressive baseline
and GLANCE back to back on one card, in one process, so the speedup is a ratio
between two numbers measured under the same conditions. Accepted length is
pooled over rounds and is engine independent; wall-clock is reported next to
it and is not comparable across engines.

    python experiments/main_table.py \
        --target Qwen/Qwen3-VL-8B-Instruct \
        --head path/to/block-head \
        --data data/vl_eval --tasks caption,textvqa,infovqa,docvqa,chartqa \
        --out results/main_table.json

Each task file is JSONL with an ``image`` path relative to ``--data`` and a
``prompt`` string. The first sample of every run is a warm-up and is dropped,
so ``--n 100`` reads 101 prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glance
from glance.vl import VisionPositions, load_target

TASKS = ["caption", "textvqa", "infovqa", "docvqa", "chartqa"]


def load_task(name, data_dir, n, max_pixels):
    """Read one task's prompts and images, capped on the long edge."""
    path = os.path.join(data_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing task file: {path}")

    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = os.path.join(data_dir, row["image"])
            if os.path.exists(image_path):
                rows.append((image_path, row["prompt"]))
            if len(rows) >= n:
                break

    loaded = []
    for image_path, prompt in rows:
        image = Image.open(image_path).convert("RGB")
        if max_pixels > 0 and max(image.size) > max_pixels:
            w, h = image.size
            scale = max_pixels / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
        loaded.append((image, prompt))
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--head", required=True, help="trained block head")
    parser.add_argument("--data", required=True, help="directory of task JSONL")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100, help="prompts per task, after warm-up")
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--budget", type=int, default=63, help="candidate tree size")
    parser.add_argument("--branch-k", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=896, help="0 for native resolution")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chain", action="store_true",
                        help="also run the same head as a width-1 chain")
    parser.add_argument("--round-log", default=None,
                        help="write per-round entropy and accepted length as JSONL")
    args = parser.parse_args()

    started = time.time()

    def log(message):
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    target, processor = load_target(args.target, attn_implementation="sdpa")
    head = glance.load_block_head(args.head)
    stop_ids = [processor.tokenizer.eos_token_id]
    log(f"target {args.target}, block {head.block_size}, "
        f"layers read {list(head.target_layer_ids)}")

    def encode(prompt, image):
        message = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(
            message, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(target.device)
        return (inputs["input_ids"], inputs.get("pixel_values"),
                inputs.get("image_grid_thw"))

    round_log = open(args.round_log, "w") if args.round_log else None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = {
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "tasks": {},
    }

    for task in tasks:
        prompts = load_task(task, args.data, args.n + 1, args.max_pixels)
        if len(prompts) < 2:
            raise SystemExit(f"{task}: need at least two prompts, found {len(prompts)}")
        log(f"{task}: {len(prompts)} prompts, first is warm-up")

        arms = {"ar": [], "glance": []}
        if args.chain:
            arms["chain"] = []

        for index, (image, prompt) in enumerate(prompts):
            input_ids, pixel_values, grid = encode(prompt, image)
            positions = VisionPositions(pixel_values, grid)

            _, ar = glance.greedy_generate(
                target, input_ids, args.max_new, stop_ids,
                positions=positions, return_stats=True)
            arms["ar"].append(ar)

            sink = None
            if round_log is not None and index > 0:
                def sink(row, task=task, index=index):
                    row.update(task=task, prompt=index - 1)
                    round_log.write(json.dumps(row) + "\n")

            _, ours = glance.generate(
                head, target, input_ids, args.max_new, stop_ids,
                budget=args.budget, branch_k=args.branch_k,
                temperature=args.temperature, positions=positions,
                return_stats=True, timing="decode_only", on_round=sink)
            arms["glance"].append(ours)

            if args.chain:
                _, chain = glance.chain_generate(
                    head, target, input_ids, args.max_new, stop_ids,
                    temperature=args.temperature, positions=positions,
                    return_stats=True)
                arms["chain"].append(chain)

        measured = {name: glance.drop_warmup(runs) for name, runs in arms.items()}
        baseline = statistics.fmean(s.ms_per_token for s in measured["ar"])
        row = {"n_prompts": len(measured["ar"]),
               "ar_ms_per_token": baseline}

        for name, runs in measured.items():
            if name == "ar":
                continue
            ms = statistics.fmean(s.ms_per_token for s in runs)
            row[name] = {
                # rounds pooled, not prompts averaged
                "tau": glance.acceptance_length(
                    [s.tau for s in runs for _ in range(s.rounds)]),
                "ms_per_token": ms,
                "speedup": glance.speedup(baseline, ms),
                "tree_size": statistics.fmean(s.tree_size for s in runs),
                "per_prompt": {"tau": [s.tau for s in runs],
                               "ms_per_token": [s.ms_per_token for s in runs],
                               "rounds": [s.rounds for s in runs]},
            }
            log(f"{task} {name}: tau {row[name]['tau']:.2f}, "
                f"{ms:.2f} ms/tok, {row[name]['speedup']:.2f}x")

        results["tasks"][task] = row

    for name in ("glance", "chain"):
        rows = [t[name] for t in results["tasks"].values() if name in t]
        if rows:
            results.setdefault("geomean_speedup", {})[name] = glance.geomean(
                [r["speedup"] for r in rows])

    if round_log is not None:
        round_log.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
    log(f"wrote {args.out}")
    for name, value in results.get("geomean_speedup", {}).items():
        log(f"{name}: geomean speedup {value:.2f}x over {len(results['tasks'])} tasks")


if __name__ == "__main__":
    main()
