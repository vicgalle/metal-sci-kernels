# metal-kernels

Benchmark for LLM-generated Metal compute kernels on Apple Silicon. See
`PLAN.md` for the design rationale and the full eight-task roadmap.

This repo is a phase-1 cut: full evaluation harness end-to-end, with
three starter tasks across two optimization regimes.

## What's here

- **Harness** (`metal_kernels/harness.py`): runtime-compiles `.metal`
  source via `MTLDevice.newLibraryWithSource` (no offline `xcrun metal`
  toolchain needed), dispatches with `MTLCommandBuffer` GPU timestamps,
  reads back through unified-memory `MTLBuffer.contents()`.
- **Hardware** (`metal_kernels/hardware.py`): detects the chip
  (M1/M2/M3/M4 family) and looks up peak FP32 GFLOPS + DRAM bandwidth
  for the roofline ceiling.
- **Task abstraction** (`metal_kernels/task.py`): each task owns input
  generation, dispatch, CPU reference, tolerance, and roofline math.
  Score = geometric mean of `achieved / ceiling` across sizes; any
  correctness failure forces score = 0.
- **Three tasks**:
  - `saxpy` — BW-bound smoke task (12 B/elem)
  - `heat2d` — 5-point stencil, ping-pong over multiple steps
  - `nbody` — all-pairs gravity with leapfrog (compute-bound)
- **LLM bridge** (`metal_kernels/llm.py`): single `call_llm` entry that
  dispatches to Claude (via `claude_agent_sdk`) or Gemini (via
  `google-genai`) based on model name.
- **Evolution loop** (`metal_kernels/evolve.py`): seed → iterate; each
  iteration sees the previous attempt, the incumbent best, and a short
  history. Persists prompts, responses, sources, and JSON results.

## Install

```sh
pip install -r requirements.txt
```

## Quickstart

Verify the seed kernels compile, pass correctness, and time:

```sh
python3 run_benchmark.py --task saxpy  --evaluate-seed-only
python3 run_benchmark.py --task heat2d --evaluate-seed-only
python3 run_benchmark.py --task nbody  --evaluate-seed-only
```

Run an evolution loop with Claude or Gemini:

```sh
# Gemini (requires GEMINI_API_KEY or GOOGLE_API_KEY)
python3 run_benchmark.py --task nbody --model gemini-2.5-flash --iterations 5

# Claude via the Agent SDK
python3 run_benchmark.py --task heat2d --model claude-sonnet-4-6 --iterations 3
```

Per run, an output directory is created under `results/` containing:

```
00_seed.metal       # the unchanged seed
01_prompt.md        # the user prompt sent to the LLM at iteration 1
01_response.md      # raw LLM response
01_reasoning.md     # extended-thinking tokens (when available)
01_candidate.metal  # extracted Metal source
01_result.json      # per-size correctness + timing + fraction-of-ceiling
...
best.metal          # incumbent at end of run
best_result.json
history.json        # per-iteration record
summary.json
```

## Adding a new task

Drop a `seeds/<name>.metal` and a `metal_kernels/tasks/<name>.py`
implementing `Task.evaluate_size`. Decorate with `@register_task("<name>")`
and add the import to `metal_kernels/tasks/__init__.py`. The harness
plumbing (compile, multi-size loop, scoring, correctness gate) is
shared.
