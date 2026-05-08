# Future Task Ideas — Exotic / Niche Memory Patterns

Brainstorm of new tasks for the **Metal-Sci** benchmark, organized by
*new optimization regimes* the current suite (R1–R6) does not stress.
Each task names the **lever** an LLM must reach for, why it is structurally
distinct from the existing 10 tasks, and a candidate held-out probe.

The goal is to extend the held-out gate $\Phi_\mathcal{T}$ to optimization
surfaces where canonical CUDA recall does not transfer cleanly — surfacing
silent overfit and silent regression in regimes the current paper does not
cover.

---

## R7 — Wavefront / strict-dependency sweeps

### `adi3d`
Alternating-Direction-Implicit Crank–Nicolson on a 3D grid. Three
tridiagonal solves per step, one along each axis, requiring serial sweeps
along the active axis with parallelism across the orthogonal plane.
- **Lever**: transposed-layout management between phases (the same data
  must be stride-1 along x, y, then z within one timestep). Cyclic-reduction
  vs. PCR vs. Thomas-with-tile-transpose is a real tradeoff.
- **Held-out**: a non-cubic prism (catches "I assumed N×N×N").

### `smithwaterman`
Antidiagonal sweep of a 2D dynamic-programming matrix (sequence alignment).
Each cell depends on N/W/NW.
- **Lever**: hyperplane skewing — launch one threadgroup per anti-diagonal.
  Tests scheduling and barrier-per-wavefront.
- **Held-out**: a long-thin matrix that breaks square-tile assumptions.

### `sptrsv`
Sparse triangular solve via level-set scheduling on a sparse DAG.
- **Lever**: dependency tracking with atomic ready-counters or
  self-scheduling. No existing R1–R6 task touches dynamic dependency
  structure.

---

## R8 — Bit-permutation memory (truly exotic)

### `qsim`
Quantum state-vector simulator. Apply a sequence of 1- and 2-qubit gates
to a $2^{20}$–$2^{25}$ complex amplitude vector. Each gate $G_k$ pairs
amplitude $|x\rangle$ with $|x \oplus 2^k\rangle$.
- **Lever**: the memory access pattern is "thread-paired-with-thread-XOR-(1<<k)",
  which doesn't appear anywhere else in the suite. Gates on low qubits →
  `simd_shuffle_xor`; gates on mid qubits → threadgroup-memory pairing;
  gates on high qubits → DRAM-strided. The same kernel must dispatch all
  three regimes by qubit index.
- **Verification**: inner product against CPU reference.
- **Roofline**: BW-bound at 16 B/amplitude.

### `morton`
Build a 3D field traversal in Z-order (Morton) curve layout via PDEP-style
bit interleaving and run a stencil on the reordered data; compare against
the same stencil on row-major.
- **Lever**: bitwise interleaving with `clz`/`ctz` and lookup-table
  tradeoffs, plus the cache-locality argument the reorder is supposed to
  deliver.

### `radix_sort`
Single-pass GPU radix sort with decoupled-lookback (Merrill 2016). One
persistent kernel; threadgroups communicate via a global flag array with
`simd_active_threads_mask` and atomic loads.
- **Lever**: inter-block synchronization without `MTLCommandBuffer`
  boundaries — a regime explicitly unique to persistent kernels.

---

## R9 — Persistent kernels and work-stealing

### `barneshut`
Barnes–Hut N-body. Build an octree (host or separate kernel), then
per-particle walk the tree with multipole acceptance. Threads in the same
simdgroup take divergent paths.
- **Levers**: (a) per-thread independent walks (warp-divergent), or
  (b) simdgroup-cooperative walks with a shared traversal queue via
  `simd_ballot`.
- **Held-out**: clustered distribution (Plummer model) where divergence
  is worse than uniform.

### `compact`
Stream compaction with single-pass exclusive prefix scan. Variable output
size per threadgroup; downstream kernels read with a global counter.
- **Lever**: tests scan + atomic counter compaction together — neither
  shows up in existing tasks.

---

## R10 — Top-k and rank-based reductions

### `knn`
Brute-force k-nearest-neighbors in 64-D with per-query top-k via a
simdgroup-cooperative bitonic heap.
- **Lever**: top-k is structurally distinct from `sum`/`max` reductions
  in `nbody`/`gradshaf` because the reduction operator is **non-commutative
  across truncation boundaries**. Choices: full sort then truncate, heap
  maintained across batches, or bitonic top-k merge.

### `spmv_ell_top1`
SpMV on power-law-degree graphs where the score combines a sum *and* the
argmax row index (PageRank-style "where did the dominant flow come from").
- **Lever**: combines reduction with index-tracking — uncommon in
  existing tasks.

---

## R11 — Multi-resolution / cross-scale data flow

### `mgvcycle`
Geometric multigrid V-cycle on the 2D Poisson equation: pre-smooth,
restrict, recurse, prolongate, post-smooth.
- **Lever**: cross-resolution data flow in a single-buffer storage
  scheme, plus the choice between Jacobi and red-black Gauss-Seidel as
  the smoother.
- **Held-out**: a non-power-of-two coarsening factor.

### `amr_flux`
Block-structured AMR with one level of refinement: coarse patches plus
fine patches with ghost-cell interpolation across patch boundaries.
- **Lever**: exotic indirection — patch-to-patch neighbor lookup tables
  — combined with stencil math.

---

## R12 — Mixed precision and determinism

### `kahan_reduce`
Compute $\sum_i x_i$ for an ill-conditioned vector ($x_i \sim
\mathcal{N}(0, 10^{12})$ with cancellation) and require bit-exact
agreement across two runs with different threadgroup counts.
- **Lever**: forces compensated summation (Kahan/Neumaier) or a fixed
  reduction tree. Tests *determinism as a fitness constraint* —
  orthogonal to anything in the current suite.

### `mp_gemv`
fp16-input, fp32-accumulate batched matvec with a verified-precision
tolerance the seed barely passes.
- **Lever**: stresses the precision/perf knife-edge.

---

## R13 — Small-batched dense kernels (DG-FEM regime)

### `dgfem`
Discontinuous-Galerkin volume integral: per-element $u \mathrel{+}= D\,u$
where $D$ is a fixed $(p+1)^3 \times (p+1)^3$ differentiation matrix and
elements are independent.
- **Lever**: register-resident small dense matvec with the matrix in
  constant memory or threadgroup. Distinct from `nbody` (one matrix vs.
  all-pairs) and from `hmc` (one matrix vs. tiny chain-resident matvecs).
- **Held-out**: vary polynomial order $p$.

---

## R14 — 4D lattice / non-trivial group structure

### `gauge_su3`
Lattice gauge update step on a 4D periodic lattice with complex 3×3
matrices per link; performs a Wilson plaquette computation: traces of
products of four SU(3) matrices around each elementary square.
- **Lever**: 4D memory linearisation + complex matmul fusion. Niche but
  a real HPC workload (lattice QCD).

---

## R15 — Variable-output and streaming compaction

### `rle`
Run-length encode a structured field. Output size is data-dependent.
Verified by round-trip decode.
- **Lever**: prefix-scan-driven layout-decision and per-thread variable
  writes.

### `cuckoo_build`
Build a cuckoo hashtable on the GPU: insertions resolve via atomic
CAS-loop with bounded eviction depth.
- **Lever**: the CAS-loop pattern, which `lj`'s atomic counter doesn't
  exercise (lj is monotonic-add only).
- **Held-out**: adversarial input whose load factor exceeds the eviction
  bound, forcing the kernel to detect and report failure.

---

## R16 — 3-body interactions and fourth-order operators

### `tersoff`
3-body bond-order MD potential. Each force depends on triplet
neighbours, not pairs.
- **Lever**: the inner loop is doubly nested over the neighbour list,
  breaking the assumption of independent pair iterations baked into `lj`.
  Real scientific code (LAMMPS Tersoff, SiC/Si simulations).

### `cahn_hilliard`
4th-order phase-field stencil (Laplacian of Laplacian, 9-point).
- **Lever**: long-stride neighbour reads vs. the compact 5-point in
  `heat2d`; fused two-level-stencil (compute $\nabla^2 u$ once into
  shared memory, then $\nabla^2$ again) vs. direct 9-point.

---

## Methodological extensions worth pairing with new tasks

- **A second held-out probe per task**: one held-out *size* (current) +
  one held-out *configuration* (e.g., HMC at $d{=}24$ catches Opus, but a
  non-square LBM domain or a non-power-of-two FFT would catch other
  classes of overfit). Cheap, mechanical, doubles the discriminating
  power of $\Phi_\mathcal{T}$.
- **Adversarial held-outs designed to defeat hardcoded constants** (qsim
  with 17 qubits, FFT3D at $96^3$, gradshaf at a prime grid size). The
  reported FFT3D collapse suggests this is a high-yield direction.
- **Determinism gate** alongside correctness: verify bit-exact output
  across two runs with different threadgroup counts. Useful on
  `kahan_reduce`, `compact`, `radix_sort`, `cuckoo_build`.
- **Cross-task transfer probe**: feed the LBM-winning candidate's
  prologue into `gauge_su3` as initial seed and measure score lift — a
  structural test of whether the LLM's "lever recognition" generalizes
  between R3 (lbm) and R14 (gauge).

---

## Priority picks for first implementation

The two whose optimization surfaces are most clearly disjoint from
anything in R1–R6:

1. **`qsim`** — the bit-XOR pairing pattern is genuinely without
   analogue in the existing suite. Stresses simdgroup intrinsics in a
   way fft3d does not (XOR-pairing rather than butterfly).
2. **`adi3d`** — transposed-layout management between phases is a
   distinct lever from anything in R5 multi-kernel; the per-axis
   stride asymmetry parallels fft3d but with serial-direction
   parallelism on top.

Strong runner-ups: `barneshut` (warp-divergent tree walks),
`kahan_reduce` (determinism gate), `cuckoo_build` (CAS-loop),
`dgfem` (small batched dense matvec).
