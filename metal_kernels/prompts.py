"""Prompts for Metal kernel generation."""

from __future__ import annotations

import re
import textwrap

from .task import CandidateResult, TaskSpec


SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert Metal Shading Language (MSL) kernel engineer optimizing
compute kernels for Apple Silicon. You write `.metal` source code that
will be compiled at runtime by `MTLDevice.newLibraryWithSource`.

## Output format

Respond with a SINGLE fenced code block:

```metal
#include <metal_stdlib>
using namespace metal;

// ... your kernel(s) here ...
```

Before the code block, briefly describe (1) the optimization you are
applying, and (2) why you expect it to improve over the previous
version. Keep this under 150 words. Then provide the code block.

## Hard requirements

- The kernel signatures (function names, buffer indices, argument types)
  MUST match the spec exactly. The host binds buffers by index — getting
  this wrong produces incorrect output and will fail correctness.
- The kernel must be deterministic given the same inputs.
- Apple Silicon is unified memory; `device` and `constant` qualifiers
  behave as on discrete GPUs but with no PCIe transfer.
- Threadgroup memory is small (~32 KB); be conservative.
- Apple GPUs use SIMD width 32 (`thread_execution_width`).
- Common Metal optimizations:
  * threadgroup-memory tiling for stencils and dense matmuls
  * `simd_*` shuffle intrinsics for warp-level reductions
  * SIMDgroup matrix types for matmul-heavy kernels
  * `[[max_total_threads_per_threadgroup(N)]]` to hint the compiler
  * vectorized loads (`float4`) for memory-bound kernels
  * register blocking (have each thread compute multiple outputs)

## Correctness is non-negotiable

If the kernel produces output beyond the per-task tolerance, the
candidate is rejected and scores zero — even if it's faster. You will
get structured feedback on compile errors, correctness failures, and
performance, and may iterate on your previous attempt.
""")


def build_task_brief(spec: TaskSpec) -> str:
    """Render the task description block for the user prompt."""
    lines = [f"## Task: {spec.name}", "", spec.description, "",
             "## Required kernel signature(s)", "",
             "```", spec.kernel_signatures.strip(), "```", ""]
    return "\n".join(lines)


def build_initial_prompt(spec: TaskSpec, seed_source: str,
                         seed_result: CandidateResult) -> str:
    """First iteration: shows the seed kernel and its measured baseline."""
    parts = [build_task_brief(spec)]
    parts.append("## Baseline: naive seed kernel\n")
    parts.append("```metal\n" + seed_source.strip() + "\n```\n")
    parts.append("Measured baseline (seed):")
    parts.append(_format_result(seed_result))
    parts.append("")
    parts.append(textwrap.dedent("""\
    ## Your task

    Write an improved Metal kernel that produces correct results AND runs
    faster than the seed across all problem sizes. The fitness score is
    the geometric mean of `achieved / ceiling` across sizes; score 0 if
    any size fails correctness.

    Output ONE fenced ```metal``` code block containing the kernel(s).
    Preserve the kernel name(s) and buffer indices exactly.
    """))
    return "\n".join(parts)


def build_iteration_prompt(spec: TaskSpec, prev_source: str,
                           prev_result: CandidateResult,
                           best_source: str | None,
                           best_result: CandidateResult | None,
                           history: list[dict]) -> str:
    """Iteration prompt: prev attempt + best-so-far + summarized history."""
    parts = [build_task_brief(spec)]

    parts.append("## Your previous attempt\n")
    parts.append("```metal\n" + prev_source.strip() + "\n```\n")
    parts.append("Result of previous attempt:")
    parts.append(_format_result(prev_result))
    parts.append("")

    if best_source is not None and best_result is not None and \
            best_source.strip() != prev_source.strip():
        parts.append("## Current best (incumbent)\n")
        parts.append("```metal\n" + best_source.strip() + "\n```\n")
        parts.append("Incumbent result:")
        parts.append(_format_result(best_result))
        parts.append("")

    if history:
        parts.append("## History\n")
        for h in history[-8:]:
            parts.append(
                f"- iter {h['iteration']:>2d}: "
                f"compile={'OK' if h['compile_ok'] else 'FAIL'} | "
                f"correct={h['correct']} | "
                f"score={h['score'] if h['score'] is not None else 'N/A'}"
            )
        parts.append("")

    parts.append(textwrap.dedent("""\
    ## Instructions

    Write an improved Metal kernel. Address the failure mode in the
    previous attempt (if any), then push beyond the incumbent. Output ONE
    fenced ```metal``` code block. Preserve kernel name(s) and buffer
    indices.
    """))
    return "\n".join(parts)


def _format_result(r: CandidateResult) -> str:
    if not r.compile_ok:
        return f"  COMPILE FAILED: {r.compile_error}"
    if r.pipeline_error is not None:
        return f"  PIPELINE FAILED: {r.pipeline_error}"
    lines = []
    for s in r.size_results:
        if s.correct:
            lines.append(
                f"  {s.size_label:>16s}: correct, "
                f"{s.gpu_seconds*1e3:.2f} ms, "
                f"{s.achieved:.1f} {s.achieved_unit} "
                f"({s.fraction_of_ceiling*100:.1f}% of {s.ceiling:.0f} {s.ceiling_unit})"
            )
        else:
            lines.append(
                f"  {s.size_label:>16s}: INCORRECT "
                f"({s.error_kind}={s.error_value:.3e}, tol={s.extra.get('tol', 'n/a')})"
            )
    if r.score is not None:
        lines.append(f"  score (gmean of fraction): {r.score:.4f}")
    if r.fail_reason:
        lines.append(f"  fail_reason: {r.fail_reason}")
    return "\n".join(lines)


_FENCE_RE = re.compile(
    r"```(?:metal|cpp|c\+\+|c|msl)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_metal_source(text: str) -> str | None:
    """Pull the largest ```metal``` (or fallback) code block from LLM text."""
    matches = _FENCE_RE.findall(text)
    candidates = [m.strip() for m in matches
                  if "kernel " in m or "#include" in m]
    if not candidates:
        # Last resort: take the largest block of any kind.
        candidates = [m.strip() for m in matches]
    if not candidates:
        return None
    candidates.sort(key=len, reverse=True)
    return candidates[0]
