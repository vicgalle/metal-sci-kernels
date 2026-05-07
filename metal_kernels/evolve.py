"""Evolution loop: prompt LLM iteratively to improve a Metal kernel.

Mirrors the iterative-improvement structure from llm_self_play.py:
- evaluate the seed
- ask the LLM for an improvement, given previous attempt + incumbent best
- evaluate the candidate; if it beats the incumbent, promote it
- repeat for N iterations, persisting all artefacts to disk
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .harness import MetalHarness
from .hardware import ChipSpec, detect_chip
from .llm import call_llm, log
from .prompts import (
    SYSTEM_PROMPT,
    build_initial_prompt,
    build_iteration_prompt,
    extract_metal_source,
)
from .task import CandidateResult, Task


@dataclass
class IterationRecord:
    iteration: int
    role: str            # "seed", "candidate", "skipped"
    compile_ok: bool
    correct: bool
    score: float | None
    fail_reason: str | None
    source_path: str
    elapsed_s: float
    is_new_best: bool


def _result_summary(r: CandidateResult) -> dict:
    return {
        "compile_ok": r.compile_ok,
        "correct": r.score is not None,
        "score": r.score,
        "fail_reason": r.fail_reason,
        "sizes": [
            {
                "label": s.size_label,
                "correct": s.correct,
                "error_value": s.error_value,
                "error_kind": s.error_kind,
                "gpu_seconds": s.gpu_seconds,
                "achieved": s.achieved,
                "achieved_unit": s.achieved_unit,
                "ceiling": s.ceiling,
                "ceiling_unit": s.ceiling_unit,
                "fraction_of_ceiling": s.fraction_of_ceiling,
            }
            for s in r.size_results
        ],
    }


async def evolve(
    task: Task,
    *,
    model: str,
    n_iterations: int,
    output_dir: Path,
    n_warmup: int = 3,
    n_measure: int = 10,
    chip: ChipSpec | None = None,
) -> dict:
    """Run the evolution loop. Returns a summary dict; writes everything to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chip = chip or detect_chip()
    harness = MetalHarness()

    log(f"\n=== Metal kernel evolution: task={task.spec.name} ===")
    log(f"  model: {model}")
    log(f"  chip:  {chip.name} (peak {chip.peak_fp32_gflops:.0f} GFLOPS, "
        f"{chip.peak_bw_gb_s:.0f} GB/s)")
    log(f"  output: {output_dir}")

    history: list[IterationRecord] = []

    # --- Seed evaluation ----------------------------------------------------
    seed_src = task.spec.seed_path.read_text()
    log("\n[seed] Evaluating seed kernel...")
    t0 = time.time()
    seed_result = task.evaluate_candidate(
        harness, chip, seed_src, n_warmup=n_warmup, n_measure=n_measure,
    )
    seed_elapsed = time.time() - t0
    seed_path = output_dir / "00_seed.metal"
    seed_path.write_text(seed_src)
    history.append(IterationRecord(
        iteration=0, role="seed",
        compile_ok=seed_result.compile_ok,
        correct=seed_result.score is not None,
        score=seed_result.score,
        fail_reason=seed_result.fail_reason,
        source_path=str(seed_path),
        elapsed_s=seed_elapsed,
        is_new_best=True,
    ))
    if seed_result.score is None:
        raise RuntimeError(
            f"Seed kernel failed evaluation: {seed_result.fail_reason}"
        )
    log(f"  seed score: {seed_result.score:.4f}  ({seed_elapsed:.1f}s)")

    best_source = seed_src
    best_result = seed_result
    prev_source = seed_src
    prev_result = seed_result

    # --- Iteration loop -----------------------------------------------------
    for i in range(1, n_iterations + 1):
        log(f"\n--- iteration {i}/{n_iterations} ---")

        if i == 1:
            user_prompt = build_initial_prompt(task.spec, seed_src, seed_result)
        else:
            user_prompt = build_iteration_prompt(
                task.spec, prev_source, prev_result,
                best_source, best_result,
                history=[
                    {"iteration": h.iteration, "compile_ok": h.compile_ok,
                     "correct": h.correct, "score": h.score,
                     "is_new_best": h.is_new_best}
                    for h in history
                ],
            )

        # Save prompt for debuggability.
        (output_dir / f"{i:02d}_prompt.md").write_text(user_prompt)

        t0 = time.time()
        try:
            full_text, reasoning = await call_llm(
                SYSTEM_PROMPT, user_prompt, model,
            )
        except Exception as e:
            log(f"  LLM call failed: {e}")
            history.append(IterationRecord(
                iteration=i, role="skipped",
                compile_ok=False, correct=False, score=None,
                fail_reason=f"LLM call failed: {e}",
                source_path="",
                elapsed_s=time.time() - t0,
                is_new_best=False,
            ))
            continue
        llm_elapsed = time.time() - t0

        (output_dir / f"{i:02d}_response.md").write_text(full_text)
        if reasoning:
            (output_dir / f"{i:02d}_reasoning.md").write_text(reasoning)

        candidate_src = extract_metal_source(full_text)
        if candidate_src is None:
            log("  Could not extract a Metal source block from response.")
            history.append(IterationRecord(
                iteration=i, role="skipped",
                compile_ok=False, correct=False, score=None,
                fail_reason="no metal block in response",
                source_path=str(output_dir / f"{i:02d}_response.md"),
                elapsed_s=llm_elapsed,
                is_new_best=False,
            ))
            continue

        cand_path = output_dir / f"{i:02d}_candidate.metal"
        cand_path.write_text(candidate_src)

        log(f"  generated in {llm_elapsed:.1f}s; evaluating...")
        t1 = time.time()
        result = task.evaluate_candidate(
            harness, chip, candidate_src,
            n_warmup=n_warmup, n_measure=n_measure,
        )
        eval_elapsed = time.time() - t1
        elapsed = llm_elapsed + eval_elapsed

        is_new_best = (
            result.score is not None
            and (best_result.score is None
                 or result.score > best_result.score)
        )

        rec = IterationRecord(
            iteration=i, role="candidate",
            compile_ok=result.compile_ok,
            correct=result.score is not None,
            score=result.score,
            fail_reason=result.fail_reason,
            source_path=str(cand_path),
            elapsed_s=elapsed,
            is_new_best=is_new_best,
        )
        history.append(rec)

        if not result.compile_ok:
            log(f"  compile failed: {result.compile_error}")
        elif result.score is None:
            log(f"  incorrect: {result.fail_reason}")
        else:
            log(f"  score = {result.score:.4f} "
                f"(seed = {seed_result.score:.4f}, "
                f"best = {best_result.score:.4f})")
            for s in result.size_results:
                log(f"    {s.size_label:>16s}: "
                    f"{s.gpu_seconds*1e3:7.2f} ms, "
                    f"{s.achieved:7.1f} {s.achieved_unit} "
                    f"({s.fraction_of_ceiling*100:.1f}%)")

        if is_new_best:
            log("  NEW INCUMBENT")
            best_source = candidate_src
            best_result = result

        # Result-detail JSON for this iteration.
        (output_dir / f"{i:02d}_result.json").write_text(
            json.dumps(_result_summary(result), indent=2)
        )

        prev_source = candidate_src
        prev_result = result

    # --- Save history + best ------------------------------------------------
    (output_dir / "history.json").write_text(
        json.dumps([asdict(h) for h in history], indent=2)
    )
    (output_dir / "best.metal").write_text(best_source)
    (output_dir / "best_result.json").write_text(
        json.dumps(_result_summary(best_result), indent=2)
    )

    summary = {
        "task": task.spec.name,
        "model": model,
        "chip": chip.name,
        "n_iterations": n_iterations,
        "seed_score": seed_result.score,
        "best_score": best_result.score,
        "improvement": (
            best_result.score / seed_result.score
            if best_result.score and seed_result.score
            else None
        ),
        "history": [asdict(h) for h in history],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n=== done. seed={seed_result.score:.4f}, "
        f"best={best_result.score:.4f}, output={output_dir} ===")
    return summary
