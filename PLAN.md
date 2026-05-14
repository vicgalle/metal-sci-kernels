# Metal-ZK — Plan

A sibling benchmark to **Metal-Sci** that lands the same methodological thesis
(canonical recipes don't transfer between regimes; the held-out gate
$\Phi_\mathcal{T}$ catches silent overfit) on a structurally distinct surface:
**zero-knowledge / lattice cryptography primitives on Apple Silicon Metal**.

The pitch is *ASIC-resistance / verifiable-computation primitives as an OOD
test*. ZK provers and FHE pipelines are the new HPC: bit-exact correctness,
rich parameter axes (curve, prime field, polynomial degree, hash arity), and
essentially zero Metal corpus in pretraining. The optimization surface decomposes
into six regimes whose canonical recipes (mostly CUDA + arkworks/halo2/icicle)
do not transfer cleanly to Metal.

---

## Why a sibling and not an extension

Reasons to ship Metal-ZK as a separate package rather than adding tasks to
Metal-Sci:

1. **Different roofline anchor.** ZK is integer-arithmetic-heavy. FP32 GFLOPS
   is the wrong ceiling; we need an empirical *peak int64 mul throughput* per
   chip family (see Methodology §1).
2. **Different correctness gate.** Bit-exact integer agreement, not
   `max_abs ≤ tol`. The harness already gates on correctness; the tolerance
   logic just collapses to `== 0`.
3. **Different audience.** Metal-Sci targets HPC + LLM-evaluation readers.
   Metal-ZK targets crypto + verifiable-computation readers; the regime
   taxonomy + held-out design speaks to different referees.
4. **Cleaner OOD claim.** Metal + ZK primitives is an essentially empty
   intersection in pretraining corpora. The Metal-Sci paper already argues
   Metal is underrepresented; ZK-on-Metal is one more strict layer of OOD.

The harness, evolution loop, LLM bridge, and scoring logic are reused unchanged
from `metal-kernels`. Only `task.py`'s tolerance behaviour and `hardware.py`'s
ceiling table need extension.

---

## Regime taxonomy (Z1–Z6)

Each regime stresses a structurally distinct bottleneck whose canonical recipe
does not transfer to its neighbours. Mirrors Metal-Sci §2's table 2.

| Regime | Task | Optimization lever | In-dist sizes | Held-out |
|---|---|---|---|---|
| Z1 modular | `montgomery_msm`  | 256-bit Montgomery mul, 4×64 limb registers, simdgroup-cooperative bucket accumulation | $N\in\{2^{16},2^{18},2^{20}\}$ scalars on BLS12-381 G1 | $N=2^{17}$ on BN254 G1 |
| Z2 NTT | `goldilocks_ntt`  | Stockham radix-2/4, simd_shuffle_xor stages 1–5, Mersenne-friendly modular reduction | length $N\in\{2^{14},2^{16},2^{18}\}$ | $N=2^{20}$ |
| Z3 sponge | `poseidon2_hash`  | (t=3) batched sponge, S-box ($x^5$) pipelining, MDS matvec in registers | batch $\in\{2^{12},2^{16},2^{20}\}$ | batch $=2^{18}$, **arity t=4** (different MDS) |
| Z4 tree | `merkle_build`    | level-by-level reduction, layout-aware boundary handling, in-place vs ping-pong | $2^{16},2^{18},2^{20}$ leaves, binary | $2^{19}$ leaves, **arity 4** |
| Z5 fold | `fri_round`       | coset evaluation + random LC + commitment, cross-kernel state, folding factor | degree $2^{16},2^{18},2^{20}$ over Goldilocks, fold=2 | degree $2^{17}$, **fold=4** |
| Z6 lattice | `kyber_ntt`     | negacyclic NTT mod $q=3329$, Barrett reduction, packed 16-bit vectorisation | $n=256$ (Kyber-768), batch $\in\{1,16,256\}$ | batch=$64$, **Dilithium parameters** ($q=8380417, n=256$) |
| (smoke) | `modmul_montgomery` | back-to-back Montgomery muls — saturates int-mul throughput | $N\in\{2^{20},2^{22},2^{24}\}$ ops | $N=2^{23}$ |

Each held-out probe is designed to flip on a specific overfit mode: a hardcoded
prime, a hardcoded NTT length, a hardcoded sponge arity, a hardcoded folding
factor. This is the direct ZK analog of Metal-Sci's `fft3d` GPT silent
regression and `hmc` Opus silent correctness fail.

---

## Per-task sketches

### `montgomery_msm` (Z1)

Multi-scalar multiplication on a short-Weierstrass elliptic curve: given
$N$ pairs $(s_i, P_i)$ with $s_i \in \mathbb{Z}_r$ and $P_i \in G$, compute
$\sum_i s_i P_i$. Default curve: BLS12-381 G1 (256-bit base field $\mathbb{F}_q$,
255-bit scalar field $\mathbb{F}_r$).

- **Lever**: 256-bit Montgomery multiplication in 4×64-bit limbs held in
  registers; Pippenger windowed bucketing; simdgroup-cooperative bucket
  reductions; affine vs Jacobian vs extended-Jacobian coordinate choice.
- **Held-out twist**: same algorithm at *different curve* (BN254). The
  Montgomery constants and modulus change but the structure is identical —
  a candidate that hardcodes BLS12-381's $q$ silently fails.
- **Roofline**: int64 mul throughput. Need microbenchmark (§Methodology 1).
- **Correctness**: compare to CPU `arkworks` or `blstrs` reference output
  bit-exactly; check the resulting curve point lies on the curve and has
  correct prime-order subgroup membership.

### `goldilocks_ntt` (Z2)

Forward NTT over the Goldilocks prime $p = 2^{64} - 2^{32} + 1$ (Plonky2,
Risc0). Mathematically the integer twin of `fft3d`. The Goldilocks reduction
is famously fast (one subtract + carry, no division), so this is *purely* a
modmul + butterfly + memory-pattern test.

- **Lever**: Stockham vs Cooley-Tukey, radix-2/4/8 mix; `simd_shuffle_xor` for
  the first 5 stages (Apple simd width = 32); modular reduction fusion
  (avoid full reductions between adjacent butterflies).
- **Held-out twist**: held-out length $2^{20}$ — *and* a parameter-axis
  held-out at the same in-dist length but in **BabyBear field**
  ($p = 2^{31} - 2^{27} + 1$). Different modulus, different reduction.
- **Roofline**: BW-bound at large $N$ (24 B/element: read + write + twiddle
  factor pull); compute-bound at small $N$ at int-mul peak.
- **Correctness**: bit-exact against a CPU reference (e.g. `plonky2`'s `goldilocks-field`).

### `poseidon2_hash` (Z3)

Batched Poseidon2 sponge hashing. Poseidon2 (Grassi-Khovratovich-Lüftenegger
2023, used in Risc0, Plonky3) compresses Poseidon's MDS matrix from full to
sparse, halving the cost of every full round.

- **Lever**: register-resident state vector, S-box ($x^5$) pipelining across
  independent sponges, MDS matvec via constant-memory matrix vs unrolled FMA
  chain, threadgroup batching to amortise round-constant fetch.
- **Held-out twist**: held-out arity $t=4$ (Poseidon2's MDS is structurally
  different for different $t$). A candidate that hardcodes $t=3$'s sparse
  MDS will produce wrong output, not just slow output.
- **Roofline**: compute-bound at int-mul peak (the $x^5$ S-box dominates).
- **Correctness**: bit-exact against a CPU Poseidon2 reference.

### `merkle_build` (Z4)

Level-by-level Merkle tree construction over $N$ leaves, with a configurable
hash (default Poseidon2-t=3 from Z3).

- **Lever**: in-place vs ping-pong layout; single-kernel-per-level vs fused
  multi-level kernels; boundary handling for non-power-of-two leaf counts.
  Mirrors `gradshaf`'s multi-kernel structure but with progressively shrinking
  data, not a fixed grid.
- **Held-out twist**: **4-ary tree** instead of binary (Plonky3 uses 8-ary).
  A candidate that hardcodes "two siblings per parent" silently fails.
- **Roofline**: BW-bound (8 B/leaf if Goldilocks; 32 B/leaf if BLS scalars).
- **Correctness**: bit-exact root + level-by-level intermediate digest agreement.

### `fri_round` (Z5)

One FRI folding round: given a polynomial committed via evaluations over a
coset of size $2^k$, fold to a polynomial committed over a coset of size
$2^{k-\log f}$ using a random challenge $\alpha$ from a transcript.

- **Lever**: cross-kernel state (transcript + commitment chain); fused
  evaluate-and-fold vs separate passes; folding factor (typically 2 or 4 in
  practice). Closest Metal-Sci analog: `gradshaf` (multi-kernel with state),
  but with state that grows across rounds rather than a fixed system.
- **Held-out twist**: **folding factor f=4** instead of f=2. A candidate that
  bakes f=2 into its kernel structure will fail at f=4.
- **Roofline**: BW-bound (16–32 B/eval pull + commit).
- **Correctness**: bit-exact folded-coset evaluations + Merkle root agreement.

### `kyber_ntt` (Z6)

Negacyclic NTT modulo $q=3329$ for length $n=256$ — the inner kernel of
ML-KEM (post-quantum KEM standardised by NIST 2024). Small modulus changes
the optimization surface entirely vs `goldilocks_ntt`: Barrett reduction,
packed `ushort` vectorisation, no need for 64-bit limbs.

- **Lever**: 4-way `ushort` packing into 64-bit registers; Barrett vs
  Montgomery reduction; in-place vs out-of-place butterfly with the simdgroup
  permutation tricks from `fft3d`.
- **Held-out twist**: same kernel run on **Dilithium parameters**
  ($q = 8380417$, requires 32-bit limbs, Barrett constants change). A candidate
  hardcoding `q=3329` or 16-bit packing produces garbage.
- **Roofline**: int-mul peak.
- **Correctness**: bit-exact against `liboqs` or `pqclean`.

### `modmul_montgomery` (smoke)

A back-to-back chain of Montgomery muls over BLS12-381's $\mathbb{F}_q$.
Same role as `saxpy` in Metal-Sci: validates the harness and the int-mul
throughput ceiling on each chip family without touching the regime-specific
optimization surface.

---

## Methodology extensions

### 1. Empirical int-mul roofline

`hardware.py` currently looks up `peak_fp32_gflops` per chip. ZK tasks need
`peak_int64_mul_gops` and `peak_int32_mul_gops`, which Apple does not publish.

**Plan**: ship a one-time microbenchmark per chip family that measures sustained
`uint64 × uint64 → uint64 lo/hi` throughput in a tight loop with disabled
loop-carried dependencies. Cache the result in `~/.cache/metal-zk/` keyed by
chip-id + Metal-driver-version. Document the microbenchmark methodology in
the README and the paper appendix.

### 2. Parameter-axis held-out gate $\Phi^\mathrm{cfg}_\mathcal{T}$

Metal-Sci has *size*-axis held-out: train on $\{32,64,128\}^3$, test on $256^3$.
Metal-ZK should additionally have **configuration-axis** held-out: train on
BLS12-381, test on BN254; train on Poseidon2-t=3, test on t=4; train on
fold=2, test on fold=4. This catches a class of overfit (hardcoded constants,
hardcoded shape parameters) that pure size-axis held-out cannot. The
suggestion is already in `FUTURE_TASKS.md §Methodological extensions`; ZK
is where it lives most naturally.

### 3. Determinism gate

ZK proofs are bit-exact by construction. Adding a determinism gate (two runs
with different threadgroup counts must produce byte-identical output) for
*free* — every ZK task gets it. Useful as a foil to Metal-Sci's fp32-tolerance
gate.

### 4. Cross-task transfer probe

Test whether a Poseidon2 (Z3) winning candidate's S-box pipelining transfers
when seeded into Merkle-build (Z4, which uses Poseidon2 as a sub-component).
Direct test of "lever recognition" generalising across structurally-related
tasks. No analog in Metal-Sci (no two tasks share sub-components).

---

## Implementation order

Priority by *(structural distinctness × pretraining-corpus emptiness)*:

1. **`goldilocks_ntt`** — most direct port of an existing Metal-Sci task
   (`fft3d`) to integer arithmetic. Validates the harness change for
   tolerance → bit-exact and roofline change for FP → int-mul.
   Risk: low. Reuses the simd_shuffle_xor lever cleanly.
2. **`poseidon2_hash`** — clean compute-bound task with rich held-out
   (arity). No Metal corpus exists. Highest *yield-per-engineering-hour* of
   the suite.
3. **`merkle_build`** — depends on (2) for the inner hash. Clean multi-kernel
   reduction.
4. **`montgomery_msm`** — the *flagship* task but also the most engineering.
   Requires a 256-bit limb math library in `.metal`. Tackle after the harness
   changes are validated by (1)–(3).
5. **`fri_round`** — depends on (1)–(3). Tests cross-kernel state.
6. **`kyber_ntt`** — closes the lattice-vs-pairing-vs-hash story. Lowest
   priority of the six because its lever (packed ushort + Barrett) is the
   most ASIC-amenable and least likely to surface LLM-distinguishing
   behaviour.

CPU references — pin to:
- `plonky2 ≥ 0.2.2` (goldilocks, poseidon2-via-plonky2)
- `arkworks ≥ 0.4` (BLS12-381, BN254)
- `pqclean` (Kyber, Dilithium NTT)
- `plonky3` (Poseidon2 reference vectors)

All have stable test-vector outputs we can vendor as `.npy` / `.bin` for
offline verification (avoid making the harness link against Rust at runtime).

---

## Open questions / risks

1. **Int-mul peak measurement reproducibility.** Apple GPU integer throughput
   varies with thermal state and is undocumented. Need to validate that the
   microbenchmark gives stable readings across cold/warm runs; otherwise the
   roofline anchor is noise.
2. **Pretraining contamination via `icicle` / `cuZK`.** NVIDIA's `icicle`
   library has high-quality CUDA implementations of MSM, NTT, Poseidon2.
   Frontier LLMs may have ingested it. Need to evaluate whether the Metal
   translation is mechanical (corpus contamination defeats the OOD claim)
   or genuinely non-trivial (it isn't — `cudaShfl_xor` ↔ `simd_shuffle_xor`
   isn't a one-line rewrite once threadgroup memory and unified-memory
   buffer aliasing enter). Worth a dedicated paragraph in the paper.
3. **Curve-arithmetic correctness boundaries.** Curve points must be on the
   curve *and* in the prime-order subgroup. Naive correctness checks miss
   the subgroup check. Mirror `arkworks`'s `is_on_curve()` +
   `is_in_correct_subgroup_assuming_on_curve()` exactly.
4. **License footprint.** Vendoring test vectors from arkworks (Apache+MIT),
   plonky2 (MIT), pqclean (CC0 / public-domain) is clean. Vendoring code
   from those projects into our seeds would force a license review. Keep
   seeds independent implementations.
5. **"Just use MLX/icicle/halo2" reviewer pushback.** Pre-empt with a
   §Related Work paragraph that names every existing GPU ZK library and
   states explicitly that our value is the *evolutionary-search OOD test*,
   not the hand-written kernels.

---

## Pre-work checklist (before the first task lands)

- [ ] Vendor a minimal Goldilocks + BLS12-381 CPU reference (Python via
      `py_arkworks_bls12_381` / `py_plonky2` if available; else a small
      `cffi` wrapper) under `metal_zk/reference/`.
- [ ] Write the int-mul microbenchmark + cache layer in
      `metal_zk/hardware.py`. Validate stability across 3 cold runs on the
      author's M1 Pro.
- [ ] Subclass `Task` to support `bit_exact=True` correctness mode in
      `metal_zk/task.py` (alternative: parameterise tolerance kind in
      `SizeResult.error_kind`).
- [ ] Repository skeleton: `metal-zk-kernels/` mirroring `metal-kernels/`,
      with `seeds/`, `metal_zk/tasks/`, `results/`. Either a sibling repo
      or a `metal_zk/` subdirectory in this one — decide before (1) lands.
- [ ] Draft a "regime overview" figure analogous to `figures/overview.png`
      with Z1–Z6 panels.

---

## Out of scope (for now)

- FHE bootstrapping kernels (CKKS, BFV). Different optimization regime
  (huge ciphertexts, multi-level key switching). Worth a follow-up paper,
  not a first task.
- Custom-gate constraint synthesis (Halo2-style circuit-compilation
  kernels). Not really a kernel — it's a graph traversal with codegen.
- Sumcheck / GKR rounds (`gkr_sumcheck`). Interesting, but its lever
  (per-round polynomial evaluation with a *growing* transcript) duplicates
  `fri_round` for benchmark-discrimination purposes. Add as a Z5b variant
  if Z5 turns out to be ambiguous.
- Pairing computation (Miller loop + final exponentiation). Important for
  Groth16/KZG verification but verification is single-pair so the
  bandwidth/compute roofline argument is weak. Skip.
