#!/usr/bin/env python3
"""Re-evaluate fft3d seed vs best on the new held-out 256³ config."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/Users/victorgallego/metal-kernels")
from metal_kernels import tasks  # noqa: F401
from metal_kernels.harness import MetalHarness
from metal_kernels.hardware import detect_chip
from metal_kernels.task import get_task

ROOT = Path("/Users/victorgallego/metal-kernels/results")
MODEL = "claude-opus-4-7"

# Locate the canonical fft3d run dir.
cands = sorted(ROOT.glob(f"fft3d_{MODEL}_*"))
cands = [c for c in cands if (c/"summary.json").exists()
         and (c/"best.metal").exists() and (c/"00_seed.metal").exists()]
ten = [c for c in cands if json.loads((c/"summary.json").read_text()).get("n_iterations")==10]
d = (ten[-1] if ten else cands[-1])
print(f"# fft3d held-out re-evaluation, run_dir={d.name}")

chip = detect_chip()
harness = MetalHarness()
task = get_task("fft3d")
held = task.spec.held_out_sizes
print(f"# new held-out sizes: {[(h.label, h.params) for h in held]}")
print(f"# chip: {chip.name}\n")

seed_src = (d/"00_seed.metal").read_text()
best_src = (d/"best.metal").read_text()

seed_res = task.evaluate_candidate(harness, chip, seed_src, sizes=held)
best_res = task.evaluate_candidate(harness, chip, best_src, sizes=held)

print("| Variant | compile | correct | frac of ceiling | achieved | fail_reason |")
print("|---|---|---|---|---|---|")
for name, res in [("seed", seed_res), ("best", best_res)]:
    if not res.compile_ok:
        print(f"| {name} | FAIL | — | — | — | {res.fail_reason} |"); continue
    if not res.size_results:
        print(f"| {name} | OK | — | — | — | {res.fail_reason} |"); continue
    s = res.size_results[0]
    if not s.correct:
        print(f"| {name} | OK | FAIL | — | — | {res.fail_reason} |"); continue
    print(f"| {name} | OK | OK | {s.fraction_of_ceiling*100:.2f}% | {s.achieved:.1f} {s.achieved_unit} | — |")

# In-dist refresher
print("\n# In-distribution sizes for reference:")
for s in task.spec.sizes:
    print(f"#   {s.label}: {s.params}")
