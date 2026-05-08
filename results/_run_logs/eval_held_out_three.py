#!/usr/bin/env python3
"""Held-out evaluation for all three sweep models in one session, so the
absolute fraction-of-ceiling numbers are measured under consistent thermal
state and the cross-model held-out comparison stays apples-to-apples.

Outputs `three_held_out.json` next to this script."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/Users/victorgallego/metal-kernels")
from metal_kernels import tasks  # noqa: F401  (registers tasks)
from metal_kernels.harness import MetalHarness
from metal_kernels.hardware import detect_chip
from metal_kernels.task import get_task

ROOT = Path("/Users/victorgallego/metal-kernels/results")
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]
MODELS = ["claude-opus-4-7", "gemini-3.1-pro-preview", "gpt-5.5"]

def pick_dir(task: str, model: str) -> Path | None:
    cands = sorted(ROOT.glob(f"{task}_{model}_*"))
    cands = [c for c in cands if (c/"summary.json").exists()
             and (c/"best.metal").exists() and (c/"00_seed.metal").exists()]
    if not cands: return None
    def n_iters(c):
        return json.loads((c/"summary.json").read_text()).get("n_iterations", -1)
    return max(cands, key=n_iters)

chip = detect_chip()
harness = MetalHarness()
print(f"# Three-way held-out, chip={chip.name}\n")

results = []
for tname in TASKS:
    task = get_task(tname)
    held = task.spec.held_out_sizes
    if not held:
        print(f"## {tname}: no held-out sizes — skipped"); continue
    h = held[0]

    # All three models share the same seed (00_seed.metal is identical across
    # runs). Use whichever exists.
    dirs = {m: pick_dir(tname, m) for m in MODELS}
    seed_d = next((d for d in dirs.values() if d is not None), None)
    if seed_d is None:
        print(f"## {tname}: no run dirs at all — skipped"); continue
    seed_src = (seed_d/"00_seed.metal").read_text()
    seed_res = task.evaluate_candidate(harness, chip, seed_src, sizes=held)
    if not (seed_res.compile_ok and seed_res.size_results
            and seed_res.size_results[0].correct):
        sf = None
    else:
        sf = seed_res.size_results[0].fraction_of_ceiling

    row = {"task": tname, "held": h.label, "seed_frac": sf,
           "ceiling": seed_res.size_results[0].ceiling if seed_res.size_results else None,
           "ceiling_unit": seed_res.size_results[0].ceiling_unit if seed_res.size_results else "",
           "achieved_unit": seed_res.size_results[0].achieved_unit if seed_res.size_results else "",
           "models": {}}

    for m in MODELS:
        d = dirs[m]
        if d is None:
            row["models"][m] = {"frac": None, "achieved": None, "fail": "no run dir", "dir": None}
            continue
        best_src = (d/"best.metal").read_text()
        r = task.evaluate_candidate(harness, chip, best_src, sizes=held)
        if not r.compile_ok:
            row["models"][m] = {"frac": None, "achieved": None,
                                "fail": f"compile: {r.fail_reason}", "dir": d.name}
        elif not r.size_results or not r.size_results[0].correct:
            row["models"][m] = {"frac": None, "achieved": None,
                                "fail": f"correctness: {r.fail_reason}", "dir": d.name}
        else:
            s = r.size_results[0]
            row["models"][m] = {"frac": s.fraction_of_ceiling,
                                "achieved": s.achieved, "fail": None,
                                "dir": d.name}
    results.append(row)

# Markdown table
print("| Task | Held-out | Seed frac | O frac | O × | G frac | G × | GPT5 frac | GPT5 × |")
print("|---|---|---|---|---|---|---|---|---|")
for r in results:
    sf = r["seed_frac"]
    sf_s = f"{sf*100:.2f}%" if sf is not None else "FAIL"
    cells = []
    for m in MODELS:
        info = r["models"][m]
        if info["frac"] is None:
            cells.append("FAIL"); cells.append("—")
        else:
            cells.append(f"{info['frac']*100:.2f}%")
            cells.append(f"{info['frac']/sf:.2f}×" if sf else "—")
    print(f"| {r['task']} | {r['held']} | {sf_s} | " + " | ".join(cells) + " |")

out = ROOT/"_run_logs"/"three_held_out.json"
out.write_text(json.dumps(results, indent=2, default=str))
print(f"\nRaw: {out}")
