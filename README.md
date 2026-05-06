# metal-kernels

Benchmark for LLM-generated Metal compute kernels on Apple Silicon. See
`PLAN.md` for the design rationale and the full eight-task roadmap.

This repo is a phase-1 cut: full evaluation harness end-to-end, with
four starter tasks across three optimization regimes (regular stencils,
compute-bound, multi-field/exotic-memory).

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
- **Tasks**:
  - `saxpy` — BW-bound smoke task (12 B/elem)
  - `heat2d` — 5-point stencil, ping-pong over multiple steps
  - `nbody` — all-pairs gravity with leapfrog (compute-bound)
  - `lbm` — D2Q9 lattice Boltzmann, fused pull-stream + BGK (multi-field, periodic BCs)
  - `gradshaf` — Grad-Shafranov fixed-boundary Picard-Jacobi: max-reduction
    + variable-coefficient 5-point stencil with nonlinear source, two
    kernels per outer step (first multi-kernel + reduction task in the suite)
  - `fft3d` — 3D complex-to-complex forward FFT, fp32, power-of-two cube;
    three named kernels (one per axis) with shared-memory radix-2
    Cooley-Tukey butterflies. New "data-shuffle / butterfly" regime —
    optimization is dominated by twiddle caching, simdgroup shuffles,
    and bank-conflict avoidance, not stencil tiling.
- **LLM bridge** (`metal_kernels/llm.py`): single `call_llm` entry that
  dispatches to Claude (via `claude_agent_sdk`) or Gemini (via
  `google-genai`) based on model name.
- **Evolution loop** (`metal_kernels/evolve.py`): seed → iterate; each
  iteration sees the previous attempt, the incumbent best, and a short
  history. Persists prompts, responses, sources, and JSON results.


## Quickstart

Verify the seed kernels compile, pass correctness, and time:

```sh
uv run run_benchmark.py --task saxpy  --evaluate-seed-only
uv run run_benchmark.py --task heat2d --evaluate-seed-only
uv run run_benchmark.py --task nbody  --evaluate-seed-only
uv run run_benchmark.py --task lbm    --evaluate-seed-only
```

Run an evolution loop with Claude or Gemini:

```sh
# Gemini (requires GEMINI_API_KEY or GOOGLE_API_KEY)
uv run run_benchmark.py --task nbody --model gemini-2.5-flash --iterations 5

# Claude via the Agent SDK
uv run run_benchmark.py --task heat2d --model claude-sonnet-4-6 --iterations 3
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

## Example runs (lbm, M1 Pro)

Two 25-iteration runs on `lbm` (D2Q9, fused pull-stream + BGK collision,
periodic BCs). The seed is BW-bound and already hits peak DRAM at the
largest size, so the headroom lives in the small/mid-size regimes.

| Model | Seed | Best | Speedup | 64² peak | 128² peak | 256² peak | Compile fails |
|---|---|---|---|---|---|---|---|
| `claude-opus-4-6` | 0.392 | 0.545 | **1.39×** | 0.29 (1× iter) | 0.63 | 1.35 | 6 / 25 |
| `claude-opus-4-7` | 0.395 | 0.576 | **1.46×** | 0.34 (3× iters) | 0.62 | 1.30 | 0 / 25 |

Both models found the same algebraic win on 128². The BGK equilibrium
has pairwise symmetry along each velocity axis — for opposite directions
`k=1,3` (`±x`), `k=2,4` (`±y`), `k=5,7` (`±(x+y)`), `k=6,8` (`±(y−x)`),
the `f_eq` formulas share `w_k·ρ·(1 + 4.5·(c·u)² − 1.5·|u|²)` and differ
only by the sign of `3·(c·u)`. Folding that into two FMAs per opposite
pair, with `(1 − 1/τ)` and `(1/τ)` precomputed once, took 128²×100 from
35% → 60% of peak DRAM. The 256² case sits above 100% of nominal peak
on every iteration (working set is SLC-resident, so the 8 B/cell
roofline understates the achievable rate).

Notable from 4-6 iter 7 (`results/lbm_claude-opus-4-6_20260506_110442/07_candidate.metal`):

```metal
const float omtau   = 1.0f - inv_tau;          // f_out = omtau·f_in + (1/τ)·f_eq
const float base    = 1.0f - 1.5f * (ux*ux + uy*uy);
const float rw19    = (1.0f / 9.0f)  * rho;
const float rw136   = (1.0f / 36.0f) * rho;

// k=1,3: cu = ±ux  — share base + 4.5·ux², differ by ±3·ux
{
    const float sym  = base + 4.5f * ux * ux;
    const float rw   = rw19 * inv_tau;
    f_out[    N + idx] = fma(omtau, f1, rw * (sym + 3.0f * ux));
    f_out[3u* N + idx] = fma(omtau, f3, rw * (sym - 3.0f * ux));
}
// k=5,7: cu = ±(ux+uy)  — same trick on the diagonal axis
{
    const float cu   = ux + uy;
    const float sym  = fma(4.5f, cu * cu, base);
    const float rw   = rw136 * inv_tau;
    f_out[5u* N + idx] = fma(omtau, f5, rw * (sym + 3.0f * cu));
    f_out[7u* N + idx] = fma(omtau, f7, rw * (sym - 3.0f * cu));
}
```

The model differences are meaningful and concentrated on the small
grid. **4-7 cracked 64²×50** with a structural lever 4-6 missed:
attaching `[[max_total_threads_per_threadgroup(N)]]` to the kernel
declaration to pin a small group geometry (8×8, 16×16, 32×2 in iters
3/21/23 respectively). All three iters held 64² ≥ 0.28 — a real,
repeatable 2× speedup on that size, not measurement variance. **4-6's
single 64² spike at iter 7 was noise**: the source diff against iter
6 is purely cosmetic, no other correct 4-6 candidate hit that number,
and the workload is 0.5 ms total.

The compile-failure pattern explains why 4-6 missed the lever. Six of
its 25 iterations (4, 9, 15, 17, 19, 21) crashed with **the exact same
error** every time — `[[max_total_threads_per_threadgroup(N)]]` placed
before the kernel declaration as a standalone statement instead of as
a function attribute on the `kernel void ...` line. Every preamble
proposed threadgroup-memory cooperative tiling, but the model could
not get past the attribute syntax across six attempts. **4-7 placed
the attribute correctly on its first try at iter 3**, kept it across
the rest of the run, and had zero compile failures.

Notable from 4-7 iter 23 (`results/lbm_claude-opus-4-7_20260506_112341/23_candidate.metal`),
the eventual best:

```metal
// 32x2 = 64 threads/tg, matching SIMD width along x for clean
// 128-byte coalesced loads per simdgroup
[[max_total_threads_per_threadgroup(64)]]
kernel void lbm_step(device const float *f_in   [[buffer(0)]],
                     device       float *f_out  [[buffer(1)]],
                     constant uint        &NX   [[buffer(2)]],
                     constant uint        &NY   [[buffer(3)]],
                     constant float       &tau  [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) { ... }
```

Convergence shape also differed. 4-6 climbed early (peak at iter 7,
then 18 iterations of cosmetic permutations). 4-7 hit a strong iter 1
(0.49) and iter 3 (0.54), then 17 iterations of regressions in the
−27% to −44% band, then re-broke through at iter 23 (0.58). Same
budget, similar wall time, different exploration shapes — but only
4-7's late breakthrough actually beat its own iter 3.

## Example runs (hmc, M1 Pro)

A 10-iteration run on `hmc` (HMC sampler on an anisotropic Gaussian,
one thread per chain). Three sizes vary register pressure:
`d=8 K=16384`, `d=16 K=4096`, `d=32 K=1024`. The seed sits at 0.3–2%
of FP32 peak — the deepest hole among the compute-bound tasks.

| Model | Seed | Best | Speedup | d=8 peak | d=16 peak | d=32 peak | Iter that won | Compile fails |
|---|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` | 0.0088 | 0.0932 | **10.6×** | 970 GFLOPS (22%) | 551 GFLOPS (12%) | 138 GFLOPS (3.1%) | 6 | 0 / 10 |

The breakthrough is one structural change at iter 6: a
**`template <uint D>` worker dispatched on `d`**, on top of a
threadgroup-cached `A`. With `D` a compile-time constant, the matvec
fully unrolls, the per-thread `q/p/f/q_old` arrays are sized exactly,
and the entire 21-force-eval leapfrog becomes one statically scheduled
FMA chain.

Notable from iter 6 (`results/hmc_claude-opus-4-7_20260506_182733/06_candidate.metal`):

```metal
template <uint D>
inline void hmc_run(uint chain_idx, ...,
                    threadgroup const float *Atile, ...) {
    float q[D];  float p[D];  float f[D];  float qold[D];
    // ...
    #pragma unroll
    for (uint i = 0u; i < D; ++i) {
        threadgroup const float *Arow = Atile + i * D;
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0u; j < D; ++j) acc = fma(Arow[j], q[j], acc);
        f[i] = acc;
    }
    // ... leapfrog inner loop, unrolled per-D ...
}
if      (d == 8u)  hmc_run<8u> (chain_idx, ..., Atile, ...);
else if (d == 16u) hmc_run<16u>(chain_idx, ..., Atile, ...);
else               hmc_run<32u>(chain_idx, ..., Atile, ...);
```

The runtime-`d` variants in iters 1–5 had the same ingredients
(tg-cached A, float4 matvec) but never crossed 4% of peak — the
compiler couldn't unroll the inner reduction without `D` as a
constant. The template-D change alone took d=8 from 121 → 970 GFLOPS
in one iteration.

Iter 7 failed correctness only at d=32: the model moved
`(q, p, f, q_old)` into a threadgroup `[D][TG_W]` layout (right
intuition, botched implementation) and the d=32 sample covariance
landed 0.38 off target vs 0.18 for the iter-6 baseline — exactly the
discriminating behaviour the per-size K budget was tuned for. Iters
8–10 reverted to the iter-6 baseline plus float4 matvec, but the
`acc.x+y+z+w` horizontal sum cost more than it saved at d=8 and the
score never recovered.

## Example runs (gradshaf, M1 Pro)

A 10-iteration run on `gradshaf` (Grad-Shafranov fixed-boundary
Picard-Jacobi: max-reduction + variable-coefficient 5-point stencil
with nonlinear source, two kernels per outer step). The seed sits at
2% of peak BW with a deliberately-naive single-threadgroup reduction
and an untiled stencil — the headroom is in the reduction.

| Model | Seed | Best | Speedup | 65² peak | 257² peak | 513² peak | Iter that won | Compile fails |
|---|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` | 0.0217 | 0.0410 | **1.89×** | 1.1% | 5.8% | 10.7% | 4 | 1 / 10 |

The breakthrough at iter 4 is **simdgroup-tree reduction with 4-way
unrolled strided loads**, paired with an FMA-fused stencil that drops
divisions for reciprocal-multiplies and folds N+S into one FMA. The
stencil stays untiled (relying on L1/L2 for the 5 neighbour reads);
the model never tried threadgroup-memory tiling.

Notable from iter 4
(`results/gradshaf_claude-opus-4-7_20260506_204437/04_candidate.metal`):

```metal
// Reduction: 4-way unrolled strided sweep, then simd_max tree.
for (; k + stride4 <= total; k += stride4) {
    float v0 = psi[(j0 + 1u) * NR + (i0 + 1u)];
    float v1 = psi[(j1 + 1u) * NR + (i1 + 1u)];
    float v2 = psi[(j2 + 1u) * NR + (i2 + 1u)];
    float v3 = psi[(j3 + 1u) * NR + (i3 + 1u)];
    local_max = max(local_max, max(max(v0, v1), max(v2, v3)));
}
float sg_max = simd_max(local_max);
if ((tid & 31u) == 0u) simd_partials[tid >> 5] = sg_max;
// ... cross-simdgroup pass ...

// Stencil: FMA-chained Δ*ψ with N/S folded, reciprocal-multiply update.
float delta_psi = fma(a_W, psi_W,
                   fma(a_E, psi_E,
                    fma(a_NS, psi_N + psi_S,
                      a_C * psi_C)));
psi_out[idx] = fma(omega * r, inv_aC, psi_C);
```

The speedup scales with grid size (1.4× → 1.6× → **2.7×** at 513²),
exactly the shape expected when the lever is the reduction:
single-TG sweep cost grows with the field, simdgroup_max collapses
that to ~log32 levels.

A first 10-iter run (not shown) hit **6/10 NaN failures**, all with
the same root cause — the model attempted a threadgroup-memory
stencil tile with a hard-coded `TILE_X=32, TILE_Y=8`, while the
host pins the step kernel's threadgroup at 16×16. Tile origins
strided by 32 over a grid that strides by 16; tiles overlapped and
NaN propagated. Adding an explicit "dispatch geometry: TG = (16, 16),
tile dims must match" note to the task description steered the next
run cleanly past the trap with zero correctness failures.

## Adding a new task

Drop a `seeds/<name>.metal` and a `metal_kernels/tasks/<name>.py`
implementing `Task.evaluate_size`. Decorate with `@register_task("<name>")`
and add the import to `metal_kernels/tasks/__init__.py`. The harness
plumbing (compile, multi-size loop, scoring, correctness gate) is
shared.
