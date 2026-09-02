<div align="center">

# GLANCE

### Vision is not overhead

<em>Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models</em>

[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Paper: CC BY 4.0](https://img.shields.io/badge/Paper-CC%20BY%204.0-blue.svg)](#citation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/js-lee-AI/GLANCE?style=social)](https://github.com/js-lee-AI/GLANCE/stargazers)

<img src="assets/framework.png" width="96%" alt="One draft pass fills the block from the target's fused vision-language state, the block becomes a wide candidate tree, and one target pass verifies all of it" />

<em>One draft pass fills the whole block from the frozen target's own fused vision-language state. The block's per-offset marginals become a wide prefix-closed candidate tree at no extra draft cost, and a single ancestor-masked target pass verifies every path at once.</em>

<b><a href="#overview">Overview</a> · <a href="#install">Install</a> · <a href="#decode-with-it">Decode</a> · <a href="#the-candidate-tree-on-its-own">Candidate tree</a> · <a href="#the-draftability-law">Draftability law</a> · <a href="#results">Results</a> · <a href="#reproduce-the-main-experiment">Reproduce</a> · <a href="#citation">Citation</a></b>

</div>

---

## Overview

Speculative decoding makes generation faster without changing what the model says. On vision-language models it has been stuck in a cycle of its own premises. The drafter stays autoregressive, so a candidate `k` tokens deep costs `k` sequential draft passes, so the drafter has to stay small for those passes to be cheap. A small drafter cannot afford the image at every step, so image tokens get compressed, pruned, or hidden from it. A drafter cut off from the image is then least reliable about exactly the text the image already fixes.

GLANCE breaks the cycle at both ends.

* **Vision costs the drafter nothing.** A block-diffusion head reads the frozen target's already-fused vision-language hidden states at a few kept layers. It never sees raw visual tokens, and it never pays per step for having seen the image.
* **Depth costs no sequential passes.** One draft pass fills every offset of a block at once. Because the offsets are conditionally independent given the context, their marginals give a whole tree of candidate paths for free, and width becomes the cheap axis: it is bought inside the verify pass rather than with more draft passes.
* **The output does not change.** One ancestor-masked target pass verifies every path, and the walk commits the longest path the target itself would have taken. At temperature 0 that is the target's greedy output, token for token, and every reported run is gated on the equality rather than assumed to have it.

Grounded workloads reward this most. When a model reads a document or a chart, much of what it generates already exists in the image, so the next token is often near-deterministic and frequently an exact copy off the page. Those long verbatim runs cost an autoregressive drafter one pass per token and a block drafter one pass in total.

<div align="center">
<img src="assets/teaser.png" width="70%" alt="On a document page the image pins the answer and one draft pass commits the whole block; on open captioning the same pass commits only a short prefix" />
</div>

<div align="center"><em>On a document page the image pins the answer, entropy is low, and one draft pass commits the entire block. On open captioning many continuations are admissible, entropy is high, and the same pass commits only a short prefix.</em></div>

One relation organizes the results. Accepted length is set by the target's next-token entropy,

```
logit p = b0 - b1 * H        E[a | H] = p / (1 - p)
```

and the fitted slope steepens with grounding across all five tasks. The law transfers to a second target and to other modalities, and it names its own boundary: where entropy stays high, free-running text still favors a chain.

This repository is that decoder as a library, plus the code for the main experiment.

## Install

```bash
pip install -e .
```

The candidate tree and the draftability law run on numpy alone, so `import glance` works on a laptop with no GPU and no torch. Torch is imported only when a decoding entry point is first touched.

```bash
pip install -e ".[decode]"   # torch, transformers, pillow: needed to actually decode
```

The block head class itself comes from the `dflash` package, installed separately from [its project page](https://z-lab.ai/projects/dflash/). Keeping it out of this repository avoids a vendored copy drifting from the version you trained with.

## Decode with it

```python
import glance
from glance.vl import VisionPositions, load_target

target, processor = load_target("Qwen/Qwen3-VL-8B-Instruct")
head = glance.load_block_head("path/to/block-head")

inputs = processor.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": image},
                                  {"type": "text", "text": "What is the total?"}]}],
    add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt").to(target.device)

ids, stats = glance.generate(
    head, target, inputs["input_ids"], max_new_tokens=256,
    budget=63,                       # candidate tree size
    positions=VisionPositions(inputs["pixel_values"], inputs["image_grid_thw"]),
    return_stats=True)

print(stats.tau, stats.ms_per_token)
print(processor.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:]))
```

The target is frozen and unmodified throughout. Nothing is fine-tuned, no vision encoder is replaced, and the image is never pruned or re-tokenized.

Three baselines share the interface, which is what makes them comparable:

```python
glance.greedy_generate(target, ids, 256)              # plain autoregression, and the oracle
glance.chain_generate(head, target, ids, 256)         # the same head as a width-1 chain
glance.generate(head, target, ids, 256, budget=63)    # the wide tree
```

Text targets need no adapter at all, since `StreamPositions` is the default:

```python
ids, stats = glance.generate(head, text_model, prompt_ids, 256, return_stats=True)
```

## The candidate tree, on its own

The tree is the mechanism, and it is separable from everything else. Given one block draft's top-k per offset, it returns the budget-N highest path-probability candidates, prefix-closed and packed parents-first. No model, no GPU, no torch:

```python
from glance.tree import build_budget_tree, children_of

nodes = build_budget_tree(top_ids, top_logprobs, budget=63, branch_k=8)
len(nodes)                       # 63 candidates from a single draft pass
nodes[0].token, nodes[0].depth   # block offset this candidate sits at
```

`glance.tree_mask` turns those nodes into the additive attention mask that lets one target pass score every path under exactly the context it would have had.

## The draftability law

```python
from glance import DraftabilityLaw

law = DraftabilityLaw.fit(entropy, accepted)      # one pair per decoding round
law.b1                                            # slope: steeper where grounding is stronger
law.expected(0.05)                                # E[a | H] at near-certain next tokens
law.holds()                                       # (bool, the four conditions separately)
```

`holds()` is the bar the paper reports fits under, and it is deliberately unkind: the slope must be positive, the rank correlation negative and significant, the fit must clear either the decile curve or half of what entropy alone could explain, and the fit must be non-degenerate.

Single rounds are noisy, so `r2_round` is read against `r2_ceiling`, the share of round-level variance any function of entropy alone could reach. `DraftabilityLaw.fit` reports both.

## Results

Greedy decoding on Qwen3-VL-8B. Accepted length `τ` is the number of tokens committed per round and is engine independent. A speedup is always a within-system ratio against that system's own autoregressive baseline on the same card.

### Head to head in a production engine

Both drafters run inside SGLang 0.5.6 on one card. Engine, card, and round budget are held fixed, and both verify 32 draft tokens a round, so the only structural difference left is that EAGLE3-VL produces those tokens with eight sequential passes and GLANCE with one.

| task | AR ms/tok | EAGLE3-VL τ | EAGLE3-VL ms/tok | EAGLE3-VL speedup | GLANCE τ | GLANCE ms/tok | GLANCE speedup | GLANCE faster by |
|---|---|---|---|---|---|---|---|---|
| captioning | 24.25 | **3.50** | **11.21** | **2.16x** | 2.91 | 13.38 | 1.81x | -16.2% |
| TextVQA | 24.31 | **4.18** | **9.75** | **2.49x** | 3.32 | 12.04 | 2.02x | -19.1% |
| InfographicVQA | 24.37 | 3.57 | 11.50 | 2.12x | **3.68** | **10.69** | **2.28x** | **+7.6%** |
| DocVQA | 24.47 | 3.68 | 11.51 | 2.13x | **3.93** | **10.92** | **2.24x** | **+5.4%** |
| ChartQA | 24.23 | 4.52 | 8.75 | 2.77x | **4.62** | **8.26** | **2.93x** | **+6.0%** |

Where the answer is anchored in the image, GLANCE decodes up to 2.93x faster than autoregression and up to 7.6% faster than the production head, from one draft pass a round instead of eight, with every paired bootstrap interval excluding zero.

Where the output is free-running text the eight-pass chain leads instead, and the split is sharp rather than noisy. The head ranks candidates by a product of offset-wise marginals, which is exact only where the block's tokens are conditionally independent given the image. Near-determinism delivers that and open description does not. Note also that GLANCE's accepted length is ordered by grounding across all five tasks, `2.91 < 3.32 < 3.68 < 3.93 < 4.62`, while the production head's is not.

### Against everything shipped for this target

| method (draft passes a round) | params | caption | TextVQA | InfoVQA | DocVQA | ChartQA | geomean speedup | lossless |
|---|---|---|---|---|---|---|---|---|
| n-gram lookup (PLD, 0) | 0 | 1.29 | 2.49 | 2.53 | 3.30 | 2.57 | 0.92x | exact |
| Classic SD (Qwen3-VL-4B, 8) | 4.4B | **3.53** | **3.79** | **3.95** | **4.39** | 4.47 | 0.71x | exact |
| Classic SD (Qwen3-1.7B, text only, 8) | 2.0B | 1.49 | 1.42 | 1.90 | 1.68 | 2.12 | 0.42x | exact |
| EAGLE3-VL (production, 5) | 0.40B | 2.13 | 2.66 | 2.48 | 2.56 | 2.92 | 2.06x | exact |
| EAGLE-2 (ViSpec codebase, 3) | 0.23B | 1.72 | 2.00 | 1.39 | 1.86 | 1.61 | 1.05x | relaxed |
| ViSpec (official recipe, 3) | 0.31B | 1.87 | 1.96 | 2.07 | 2.37 | 2.31 | 1.20x | relaxed |
| Medusa (same codebase, 1) | 0.08B | 1.66 | 1.55 | 1.65 | 1.46 | 1.54 | 1.07x | relaxed |
| **GLANCE (1)** | 1.05B | 3.04 | 3.29 | 3.63 | 3.78 | **5.16** | **2.49x** | **audited** |

The two-model arms buy acceptance and lose on wall-clock: a 4.4B drafter accepts long blocks and still ends up slower than plain autoregression. `audited` means every prompt was checked to reproduce greedy decoding bitwise, not that losslessness was argued for. The three relaxed heads reproduce it on no prompt under any tree setting, which places their divergence in the acceptance rule rather than in arithmetic.

### Matched training

Both head architectures trained from scratch on one corpus, with the same frozen target, global batch, epochs, and framework, then scored on the same held-out prompts. The speedups here are timed in one harness and are only meaningful against each other.

| method (draft passes a round) | params | caption | TextVQA | InfoVQA | DocVQA | ChartQA | geomean speedup |
|---|---|---|---|---|---|---|---|
| EAGLE3 head, depth-3 chain (3) | 0.40B | 1.62 | 1.59 | 1.55 | 1.58 | 1.87 | 1.15x |
| **GLANCE, budget-63 tree (1)** | 1.05B | **4.05** | **3.97** | **3.90** | **4.13** | **7.44** | **2.48x** |

Pooled over the five tasks GLANCE accepts 2.7x longer blocks, and 4.0x on ChartQA. The gap replicates on a second training corpus, at 2.04 against 1.29.

### On ViSpec's own target

Qwen2.5-VL-7B, the released ViSpec head against ours trained on that target.

| method (draft passes a round) | params | caption | TextVQA | InfoVQA | DocVQA | ChartQA | geomean speedup | lossless |
|---|---|---|---|---|---|---|---|---|
| ViSpec (released head, 3) | 0.35B | 3.34 | 3.26 | 3.21 | 3.04 | 3.59 | 1.76x | relaxed |
| **GLANCE (1)** | 1.23B | **4.17** | **3.28** | **3.47** | **3.12** | **4.72** | **2.10x** | **audited** |

### Where the gains come from

<div align="center">
<img src="assets/results.png" width="100%" alt="Entropy law by decile, the tree against the identical head as a chain, margin against the production head at two context lengths, and speedup against the verifier budget" />
</div>

**(a)** Accepted length against *measured* next-token entropy, by decile. The curves separate by grounding, and DocVQA's near-certain end clears the pooled ceiling. **(b)** The wide tree against the identical head run as a width-1 chain, which holds the drafter, the training and the draft budget fixed and removes only width: about 1.45x to 1.49x on every task, so the gain is the tree and not just a bigger head. **(c)** Margin against the production head at two context lengths. **(d)** Speedup against the verifier budget `N`, which is the knob `--budget` sets.

## Reproduce the main experiment

Each task file is JSONL with an `image` path relative to the data directory and a `prompt` string:

```json
{"image": "docvqa/0001.png", "prompt": "What is the total amount of other expenses?"}
```

Run the five tasks. The autoregressive baseline and GLANCE run back to back on one card, in one process, so the speedup is a ratio between two numbers measured under the same conditions:

```bash
python experiments/main_table.py \
    --target Qwen/Qwen3-VL-8B-Instruct \
    --head path/to/block-head \
    --data data/vl_eval --tasks caption,textvqa,infovqa,docvqa,chartqa \
    --n 100 --budget 63 \
    --round-log results/rounds.jsonl \
    --out results/main_table.json
```

Add `--chain` to run the identical head as a width-1 chain in the same loop, which is the ablation that separates the head from the tree.

Gate the losslessness claim rather than assuming it:

```bash
python experiments/audit_lossless.py \
    --target Qwen/Qwen3-VL-8B-Instruct --head path/to/block-head \
    --data data/vl_eval --n 20 --out results/audit.json
```

The audit runs at fp32 on purpose. In bf16 the target does not reproduce *itself* on every prompt with no drafter in the loop at all, because a tree pass and a single-token pass reduce in different orders and near-ties land on either side. So the audit re-runs the plain autoregressive decode against itself and reports that control next to the result. A bf16 number would measure the arithmetic, not the method.

Fit the law on the rounds that run logged:

```bash
python experiments/fit_law.py results/rounds.jsonl --out results/law.json
```

Three conventions decide numbers here and are easy to get silently wrong, so they are stated in `glance/metrics.py` and applied throughout. The first sample of a run is a warm-up and is dropped, which is why a hundred reported samples are measured as a hundred and one. A task's accepted length pools rounds rather than averaging prompts, since prompts differ in how many rounds they take. Speedups are ratios, so tasks combine by geometric mean, always against each system's own baseline.

## What is in here

```
glance/tree.py        the prefix-closed candidate tree and its attention mask
glance/decode.py      the decoding loop: tree, chain, and the autoregressive oracle
glance/head.py        what the head reads off the target, and loading a trained one
glance/vl.py          M-RoPE positions and vision prefill for Qwen-family VLMs
glance/law.py         the draftability law, its fit, and its diagnostics
glance/metrics.py     acceptance, geometric-mean speedup, paired bootstrap

experiments/main_table.py       accepted length and speed on the five tasks
experiments/audit_lossless.py   bitwise equality against greedy decoding, at fp32
experiments/fit_law.py          fit the law on a run's round log
```

Two notes for anyone reading the decoder closely.

**The extension past the block.** `generate(..., extend_budget=N)` spends one extra draft pass, with the drafter's own greedy spine revealed rather than masked, to price offsets past the block and hang a subtree off the spine leaf. It is verified in the same target pass as everything else, and candidates only ever widen, so the acceptance walk and its guarantee are untouched. It needs a head trained with a variable mask span, and is off by default.

**Two timing conventions.** `timing="decode_only"` opens the window after the prefill and after the first round's drafting, and synchronizes before both reads. This is what block-diffusion drafters report, and it is what the tables above use. `timing="legacy"` measures from before the prefill. They are not comparable, so do not mix them in one table.

## Citation

```bibtex
@article{lee2026glance,
  title   = {Vision Is Not Overhead: One-Pass Block Drafting for Lossless
             Speculative Decoding in Vision-Language Models},
  author  = {Lee, Jungseob},
  year    = {2026}
}
```

The arXiv identifier is added here once it is assigned.

GLANCE builds a candidate tree on top of a block-diffusion draft head in the style of [DFlash](https://arxiv.org/abs/2602.06036), which is cited rather than re-claimed. What is new here is reading the target's fused vision-language state instead of a stripped-down copy of the image, spending one pass of block marginals on width, and gating the result on exact reproduction of greedy decoding.

## License

Code is MIT, see [LICENSE](LICENSE). The paper is CC BY 4.0.
