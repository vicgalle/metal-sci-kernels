# Metal Kernel Evolution Benchmark — Implementation Plan

A benchmark suite for evaluating and evolving LLM-generated Metal compute kernels on Apple Silicon, focused on scientific computing patterns. Designed for use with autoresearch / evolutionary code-search approaches (FunSearch, AlphaEvolve-style).

## Design principles

- **Correctness is non-negotiable.** Every task has a CPU reference implementation and a tolerance-based verification step. Incorrect kernels are filtered before they enter the fitness pool.
- **Every task has a known ceiling.** Performance is reported as a fraction of either the architectural roofline (peak FLOPS or DRAM bandwidth, whichever is binding) or a published hand-tuned reference, whichever is tighter.
- **Diverse optimization regimes.** The eight tasks were chosen to cover distinct optimization patterns rather than to maximize task count.
- **Fast inner loop.** Each task can be evaluated end-to-end (compile + run + verify + time) in seconds, so evolution can iterate.
- **Generalization-aware.** Each task runs at multiple problem sizes; fitness is the geometric mean across sizes to discourage overfitting.

## Scope

**In scope**

- Single-device Metal compute kernels (no graphics pipeline, no multi-GPU)
- Apple Silicon (M1 through M4) on macOS
- Scientific patterns: PDEs, particle methods, sparse linear algebra
- Correctness via CPU reference comparison

**Out of scope**

- ML-shaped kernels (GEMM, attention, conv) — covered by KernelBench and similar
- Graphics / rendering pipelines
- Cross-platform (CUDA, Vulkan, OpenCL)
- Distributed / multi-GPU

## Benchmark tasks

Eight tasks across five optimization regimes. Tasks marked ★ are inherited or adapted from `https://github.com/larsgeb/m1-gpu-cpp`; the rest are new.

### Regime 1 — Regular stencils (tile + halo + temporal blocking)

#### Task 1: 2D heat diffusion ★

- **Kernel**: `diffuse_steps` — multi-step 2D heat equation, 5-point stencil
- **Tests**: tile + halo handling, temporal blocking across timesteps
- **Sizes**: 512², 2048², 8192² grids × {100, 1000} steps
- **Correctness**: max-norm vs CPU reference, tolerance 1e-5 (fp32)
- **Ceiling**: roofline at peak DRAM BW for naive; 2–3× via temporal blocking
- **Reference**: Datta 2008 (Berkeley auto-tuning); Maruyama & Aoki 2014

#### Task 2: 2D elastic wave propagation ★

- **Kernel suite**: Virieux staggered grid — `stress_update`, `velocity_update`, `apply_damping`
- **Tests**: multi-kernel composition, kernel fusion, staggered-grid correctness
- **Sizes**: 1024², 4096² grids × {100, 1000} steps
- **Correctness**: seismogram comparison at probe points vs CPU reference, tolerance 1e-4
- **Ceiling**: larsgeb paper baseline; published seismic GPU codes
- **Reference**: Virieux 1986; Komatitsch GPU seismic literature

#### Task 3: 3D acoustic wave equation (NEW)

- **Kernel**: 3D wave equation, 25- or 27-point stencil
- **Tests**: register pressure, 2.5D blocking, choice of marching axis
- **Sizes**: 128³, 256³, 512³
- **Correctness**: max-norm vs CPU reference, tolerance 1e-5
- **Ceiling**: ~50–70% of peak BW via 2.5D blocking
- **Reference**: Micikevicius 2009; Datta thesis ch. 7

### Regime 2 — Compute-bound

#### Task 4: N-body all-pairs ★ (modified)

- **Kernel**: O(N²) all-pairs gravity with leapfrog integrator
- **Tests**: register blocking (bodies-per-thread), threadgroup-memory tiling
- **Sizes**: N ∈ {4096, 16384, 65536} × 100 steps
- **Correctness**: position drift vs CPU reference at fixed seed; energy conservation as secondary check
- **Ceiling**: ~70–85% of peak FP32 FLOPS on Apple Silicon
- **Reference**: Nyland/Harris/Prins, GPU Gems 3

### Regime 3 — Multi-field, exotic memory access

#### Task 5: Lattice Boltzmann D2Q9 (NEW)

- **Kernel**: streaming + BGK collision, 2D D2Q9 lattice
- **Tests**: AA vs AB pattern, SoA vs AoS layout, push vs pull streaming
- **Sizes**: 1024², 4096² grids × 1000 steps
- **Correctness**: lid-driven cavity flow; centerline velocity profile vs Ghia 1982
- **Ceiling**: ~60–80% of peak BW
- **Reference**: Schönherr 2011; Mawson & Revell 2014

### Regime 4 — Irregular memory, atomics, dynamic work

#### Task 6: Lennard-Jones MD with neighbor lists (NEW)

- **Kernel suite**: spatial hash → cell list → neighbor list → force computation → integration
- **Tests**: load balancing, atomics vs sorted reduction, irregular access patterns
- **Sizes**: 32K, 256K, 2M particles × 100 steps; cutoff 2.5σ
- **Correctness**: total energy drift < 1e-3 over 100 steps; pair energy vs CPU reference
- **Ceiling**: comparable to HOOMD-blue / LAMMPS-GPU on small systems
- **Reference**: Anderson et al. 2008 (HOOMD); Brown et al. 2011

### Regime 5 — Sparse linear algebra, kernel composition

#### Task 7: SpMV (NEW)

- **Kernel variants**: CSR baseline + at least one of {ELLPACK, sliced-ELL, hybrid}
- **Tests**: load balancing across irregular row lengths, format choice
- **Sizes**: SuiteSparse matrices — start with `bcsstk32`, `cant`, `pwtk`, `mc2depi` (mix of structured and unstructured)
- **Correctness**: relative residual vs CPU SpMV, tolerance 1e-6
- **Ceiling**: roofline at peak BW; Williams roofline per matrix
- **Reference**: Bell & Garland 2009; Williams 2007

#### Task 8: Conjugate gradient solver (NEW)

- **Kernel suite**: SpMV + dot product + AXPY composed in CG iteration
- **Tests**: kernel fusion across iterations, persistent threadgroup state, reduction strategy
- **Sizes**: SPD subset of Task 7 matrices; solve to relative residual 1e-8
- **Correctness**: convergence to tolerance; iteration count within 10% of reference
- **Ceiling**: bandwidth-bound; fusion can yield 1.5–2× over naive
- **Reference**: Naumov 2011 (NVIDIA TR); Bell & Garland sparse iterative

## Scaffolding requirements

The evaluation / evolution harness must provide:

### 1. Build pipeline

- Metal source (`.metal`) → `metallib` via `xcrun -sdk macosx metal` + `metallib`
- Compilation cache keyed by source hash
- Compilation errors surfaced back to the agent in structured form (file, line, message)
- Optional: AIR inspection for static analysis (e.g., register usage)

### 2. Execution

- C++ host using `metal-cpp` (larsgeb infrastructure is a viable starting point)
- Python bindings via pybind11 for evolution-loop integration
- Per-task harness: load metallib → bind buffers → dispatch → readback

### 3. Correctness verification

- CPU reference implementation per task (numpy + C++/OpenMP for the heavier ones)
- Per-task tolerance and norm choice (max, L2, energy conservation, convergence)
- Edge-case suite: smallest size, boundary inputs, degenerate configurations
- Determinism check: same seed → same output (catches race conditions)

### 4. Performance measurement

- `MTLCommandBuffer` GPU timestamps, not wall clock
- Warmup: ≥3 dispatches discarded
- Measurement: ≥10 dispatches, report median + IQR
- Per-task: achieved GFLOPS or GB/s + % of roofline

### 5. Hardware introspection

- Detect chip family (M1/M2/M3/M4 × base/Pro/Max/Ultra)
- Lookup table for peak FP32 FLOPS and peak DRAM bandwidth per chip
- Per-chip roofline computed for each task at registration time

### 6. Agent interface

- **Input**: task spec (problem description, kernel signature, correctness spec, size set)
- **Seed**: naive but correct implementation
- **Output**: candidate `.metal` source (single or multi-kernel)
- **Feedback** (structured): compile result, correctness result per size, perf result per size, error messages, comparison against incumbent best

### 7. Fitness function

- **Hard gate**: correctness must pass at all sizes
- **Score**: geometric mean across sizes of `achieved / ceiling`
- **Per-task** and **aggregate** (cross-task) scores
- **Novelty signal**: edit-distance or AST-distance to prior candidates, to discourage local convergence

## Evaluation methodology

### Per-candidate flow

1. Compile (fail-fast on syntax errors)
2. Correctness at smallest size (fail-fast on incorrectness)
3. Correctness at all sizes
4. Timing at all sizes with warmup
5. Score computation

### Suite-level reporting

- Per-task: best fraction of ceiling achieved, best wall time at largest size
- Suite: geometric mean across tasks
- Stratified by regime — reveals whether the agent has systematic gaps (e.g., always weak on irregular memory)

### Generalization sanity check

- Hold out one size per task during evolution
- Evaluate held-out sizes at end of run
- Report degradation: in-distribution vs held-out

## Implementation phases

**Phase 1 — Harness foundation (Week 1–2)**

- Adapt larsgeb build system; add metallib caching
- Python bindings for compile + dispatch + verify + time
- Hardware detection + roofline lookup table
- Bring up Tasks 1, 2, 4 (exist in larsgeb) as scaffolding shakedown

**Phase 2 — New stencil and exotic-memory tasks (Week 3–4)**

- Task 3 (3D wave): natural extension of larsgeb stencils
- Task 5 (LBM): cavity-flow validation harness
- Task 7 (SpMV): SuiteSparse ingestion, CSR baseline, ELL variant

**Phase 3 — Composite / irregular tasks (Week 5)**

- Task 6 (MD): spatial hash + neighbor list infrastructure
- Task 8 (CG): builds on Task 7 SpMV

**Phase 4 — Agent integration (Week 6)**

- Connect to LLM API for candidate generation
- Initial naive seeds for each task
- First end-to-end evolution run on Task 1 as smoke test
- Fitness / novelty signal tuning

**Phase 5 — Full evaluation runs**

- Multi-task, multi-seed, varied agents
- Comparative analysis vs hand-tuned references and naive seeds
- Cross-chip evaluation if multiple machines available

## Open questions

- **Precision policy.** fp32 only initially, or allow fp16 inner loops with fp32 accumulation? Affects correctness tolerances.
- **Multi-kernel candidates.** What's the submission format for tasks with multiple kernels (Tasks 2, 6, 8)? Single file with multiple `kernel` functions, or a tarball-equivalent?
- **Training-data contamination.** How do we detect verbatim retrieval of canonical implementations (Nyland n-body, etc.)? Worth distinguishing "agent recalled the answer" from "agent optimized."
- **Evolution algorithm.** AlphaEvolve-style island model? FunSearch? Simple `(μ + λ)` ES? Worth keeping the evaluation harness algorithm-agnostic and benchmarking multiple search strategies.
- **Cross-chip generalization.** Evolve on M2 Pro, evaluate on M3 Max — interesting signal or noise? Probably worth measuring once but not optimizing for.
- **Time / compute budget per evolution run.** Token budget per candidate? Wall-clock cap per task?

## References

- https://github.com/larsgeb/m1-gpu-cpp
- Datta et al., "Stencil Computation Optimization and Auto-tuning on State-of-the-Art Multicore Architectures" (SC 2008)
- Micikevicius, "3D Finite Difference Computation on GPUs using CUDA" (GPGPU 2009)
- Nyland, Harris, Prins, "Fast N-Body Simulation with CUDA" (GPU Gems 3, 2007)
- Schönherr et al., "Multi-thread implementations of the lattice Boltzmann method on non-uniform grids for CPUs and GPUs" (CMA 2011)
- Bell & Garland, "Implementing sparse matrix-vector multiplication on throughput-oriented processors" (SC 2009)
- Williams, Waterman, Patterson, "Roofline: An insightful visual performance model for multicore architectures" (CACM 2009)
- Anderson, Lorenz, Travesset, "General purpose molecular dynamics simulations fully implemented on graphics processing units" (JCP 2008)
- Ghia, Ghia, Shin, "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method" (JCP 1982)
- Geb et al., "Seamless GPU acceleration for C++ based physics using the M1's unified processing units" (arXiv:2206.01791)
