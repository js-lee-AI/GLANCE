#!/usr/bin/env python3
"""Fit the draftability law on the round log a main-table run produced.

    python experiments/main_table.py ... --round-log results/rounds.jsonl
    python experiments/fit_law.py results/rounds.jsonl --out results/law.json

Each line carries the round's measured next-token entropy and how much of the
block was accepted. The fit is per task and pooled, and reports the same
diagnostics the paper reads: the fit against the entropy-decile means, the
fit against single rounds, and how much of the round-level variance entropy
could explain at best.

A steeper slope means grounding buys more, so the slopes should order the way
the tasks do, with documents and charts above open captioning.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glance.law import DraftabilityLaw


def read_rounds(path):
    by_task = collections.defaultdict(list)
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                by_task[row.get("task", "all")].append(row)
    return by_task


def summarise(rows, cap):
    law = DraftabilityLaw.fit([r["entropy"] for r in rows],
                              [r["accepted"] for r in rows], cap=cap)
    holds, checks = law.holds()
    return {
        "n_rounds": law.n,
        "b0": law.b0, "b1": law.b1,
        "mean_accepted": law.mean_accepted,
        "r2_curve": law.r2_curve,
        "r2_round": law.r2_round,
        "frac_of_ceiling": law.frac_of_ceiling,
        "spearman": {"r": law.spearman_r, "p": law.spearman_p},
        "holds": holds, "checks": checks,
        "curve": law.curve,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rounds", help="JSONL from main_table.py --round-log")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cap", type=float, default=15.0,
                        help="reachable accepted length")
    args = parser.parse_args()

    by_task = read_rounds(args.rounds)
    pooled = [row for rows in by_task.values() for row in rows]
    report = {"pooled": summarise(pooled, args.cap), "per_task": {}}

    for task, rows in sorted(by_task.items()):
        if len(rows) < 30:
            print(f"{task}: {len(rows)} rounds, too few to fit")
            continue
        report["per_task"][task] = summarise(rows, args.cap)

    print(f"{'task':<12} {'rounds':>7} {'b1':>7} {'R2 curve':>9} "
          f"{'mean a':>7}  holds")
    for task, fit in list(report["per_task"].items()) + [("pooled", report["pooled"])]:
        print(f"{task:<12} {fit['n_rounds']:>7} {fit['b1']:>7.3f} "
              f"{fit['r2_curve']:>9.3f} {fit['mean_accepted']:>7.2f}  {fit['holds']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
