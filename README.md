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

## Example runs (nbody, M1 Pro)

Two evolutionary runs on `nbody` (the most compute-bound seed, ~2% of
FP32 peak), 5 iterations each.

| Model | Seed score | Best score | Speedup | Best at N=2048 | Iter that won | Wall time |
|---|---|---|---|---|---|---|
| `gemini-3-flash-preview` | 0.0202 | 0.0605 | **3.0×** | 795 GFLOPS (18% peak) | 5 (monotone) | ~6.6 min |
| `gemini-3.1-pro-preview` | 0.0204 | 0.0446 | **2.2×** | 980 GFLOPS (22% peak) | 1 (one-shot) | ~16 min |

Different convergence styles emerged. **Flash** climbed monotonically,
recovering from two compile failures in iters 2-3 (mis-named Metal
attributes — fed back through the structured retry) and ending at a
SIMD-broadcast tile with `#pragma unroll(32)`, branch-free OOB masking,
and `mass` packed into `float4.w`. **Pro** landed the textbook
optimization in a single shot — threadgroup-memory tiling with
cooperative load + barrier, 4-way ILP unroll, G pre-multiplied into
mass — then regressed in every subsequent iteration by abandoning the
shared-memory path. The harness's incumbent tracking correctly held
iter 1 as `best.metal`.

Notable from Pro's iter 1 (`results/nbody_gemini-3.1-pro-preview_*/01_candidate.metal`):

```metal
threadgroup float4 shared_pos[1024];
// ...
for (uint j_start = 0; j_start < N; j_start += tsize) {
    uint j = j_start + tid;
    if (j < N) {
        // pre-multiply G into mass during the cooperative load
        shared_pos[tid] = float4(pos_in[j].xyz, G * mass[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4-way unrolled inner loop over the tile...
}
```

Notable from Flash's iter 5 (`results/nbody_gemini-3-flash-preview_*/05_candidate.metal`):

```metal
for (uint j_base = 0; j_base < N; j_base += 32) {
    float4 rj_mj = 0.0f;
    if (j_base + lane_id < N) {
        rj_mj.xyz = pos_in[j_base + lane_id].xyz;
        rj_mj.w   = mass[j_base + lane_id];   // pack mass into .w
    }
    #pragma unroll(32)
    for (ushort l = 0; l < 32; ++l) {
        const float4 other = simd_broadcast(rj_mj, l);
        const float3 d = other.xyz - ri;
        const float r2 = dot(d, d) + eps2;
        const float inv_r = rsqrt(r2);
        // branch-free OOB mask via mass=0
        const float force_mag = (j_base + l < N) ? other.w * inv_r * inv_r * inv_r : 0.0f;
        acc += d * force_mag;
    }
}
```

Pro wins at N=2048 (shared-memory tiling pays off), Flash wins at
N=256/1024 (no barrier overhead). Both runs are still well below the
plan's 70-85% target — register blocking (multiple bodies per thread)
is the obvious next lever; neither model has tried it.

## Adding a new task

Drop a `seeds/<name>.metal` and a `metal_kernels/tasks/<name>.py`
implementing `Task.evaluate_size`. Decorate with `@register_task("<name>")`
and add the import to `metal_kernels/tasks/__init__.py`. The harness
plumbing (compile, multi-size loop, scoring, correctness gate) is
shared.
