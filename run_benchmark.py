#!/usr/bin/env python3
"""CLI entry for the Metal kernel evolution benchmark.

Examples:
    # Just verify the seed kernels compile and pass correctness:
    python run_benchmark.py --task saxpy --evaluate-seed-only

    # Run 3 LLM iterations with Claude on heat2d:
    python run_benchmark.py --task heat2d --model claude-sonnet-4-6 --iterations 3

    # Run with Gemini:
    python run_benchmark.py --task nbody --model gemini-2.5-flash --iterations 5

    # Run with OpenAI:
    python run_benchmark.py --task heat2d --model gpt-5.5 --iterations 3
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from metal_kernels import tasks  # noqa: F401  (registers tasks)
from metal_kernels.evolve import evolve
from metal_kernels.harness import MetalHarness
from metal_kernels.hardware import detect_chip
from metal_kernels.task import get_task, list_tasks


def evaluate_seed(name: str) -> int:
    chip = detect_chip()
    harness = MetalHarness()
    task = get_task(name)
    src = task.spec.seed_path.read_text()
    result = task.evaluate_candidate(
        harness, chip, src, n_warmup=3, n_measure=10,
    )
    print(f"=== seed evaluation: task={name} ===")
    print(f"  chip: {chip.name}")
    print(f"  device: {harness.device_name()}")
    print(f"  compile_ok: {result.compile_ok}")
    print(f"  score (gmean fraction-of-ceiling): {result.score}")
    if result.fail_reason:
        print(f"  fail_reason: {result.fail_reason}")
    for s in result.size_results:
        ok = "OK" if s.correct else "FAIL"
        print(
            f"  {s.size_label:>16s} [{ok}]: err={s.error_value:.2e}, "
            f"{s.gpu_seconds*1e3:7.2f} ms, "
            f"{s.achieved:7.1f} {s.achieved_unit} "
            f"({s.fraction_of_ceiling*100:.1f}% of {s.ceiling:.0f} {s.ceiling_unit})"
        )
    return 0 if (result.score is not None) else 1


async def run_evolution(args) -> int:
    task = get_task(args.task)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.task}_{args.model}_{timestamp}"
    await evolve(
        task,
        model=args.model,
        n_iterations=args.iterations,
        output_dir=out_dir,
        n_warmup=args.warmup,
        n_measure=args.measure,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Metal kernel evolution benchmark CLI"
    )
    parser.add_argument(
        "--task", required=True,
        choices=list_tasks() or ["<no tasks registered>"],
        help="Which task to run.",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="LLM to use. Claude (e.g. claude-sonnet-4-6, claude-haiku-4-5), "
             "Gemini (e.g. gemini-2.5-flash), "
             "or OpenAI (e.g. gpt-5.5, gpt-5, o4-mini).",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of LLM iterations.",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Warmup dispatches per timing measurement.",
    )
    parser.add_argument(
        "--measure", type=int, default=10,
        help="Measured dispatches per timing measurement.",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory under which run outputs are written.",
    )
    parser.add_argument(
        "--evaluate-seed-only", action="store_true",
        help="Just compile + run + verify the seed kernel; skip the LLM loop.",
    )
    args = parser.parse_args()

    if args.evaluate_seed_only:
        return evaluate_seed(args.task)
    return asyncio.run(run_evolution(args))


if __name__ == "__main__":
    raise SystemExit(main())
