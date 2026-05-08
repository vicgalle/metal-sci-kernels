#!/usr/bin/env python3
"""Evaluate seed vs best (gpt-5.5) on each task's held-out size."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/Users/victorgallego/metal-kernels")
from metal_kernels import tasks  # noqa: F401  (registers tasks)
from metal_kernels.harness import MetalHarness
from metal_kernels.hardware import detect_chip
from metal_kernels.task import get_task

ROOT = Path("/Users/victorgallego/metal-kernels/results")
MODEL = "gpt-5.5"
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]

def pick_dir(task: str) -> Path | None:
    cands = sorted(ROOT.glob(f"{task}_{MODEL}_*"))
    cands = [c for c in cands if (c / "summary.json").exists()
             and (c / "best.metal").exists()
             and (c / "00_seed.metal").exists()]
    if not cands: return None
    # Prefer the run with the highest n_iterations (one canonical pick per task).
    def n_iters(c):
        return json.loads((c/"summary.json").read_text()).get("n_iterations", -1)
    return max(cands, key=n_iters)

chip = detect_chip()
harness = MetalHarness()
print(f"# Held-out evaluation, model={MODEL}, chip={chip.name}\n")

rows = []
for tname in TASKS:
    d = pick_dir(tname)
    if d is None:
        print(f"## {tname}: no run dir — skipped"); continue
    task = get_task(tname)
    held = task.spec.held_out_sizes
    if not held:
        print(f"## {tname}: no held-out sizes defined — skipped"); continue

    seed_src = (d / "00_seed.metal").read_text()
    best_src = (d / "best.metal").read_text()

    seed_res = task.evaluate_candidate(harness, chip, seed_src, sizes=held)
    best_res = task.evaluate_candidate(harness, chip, best_src, sizes=held)

    for hsize in held:
        s_row = next((r for r in seed_res.size_results if r.size_label == hsize.label), None)
        b_row = next((r for r in best_res.size_results if r.size_label == hsize.label), None)
        rows.append({
            "task": tname,
            "held_label": hsize.label,
            "seed_compile_ok": seed_res.compile_ok,
            "seed_correct": s_row.correct if s_row else False,
            "seed_frac": (s_row.fraction_of_ceiling if (s_row and s_row.correct) else None),
            "seed_achieved": (s_row.achieved if (s_row and s_row.correct) else None),
            "seed_unit": (s_row.achieved_unit if s_row else ""),
            "seed_fail": seed_res.fail_reason,
            "best_compile_ok": best_res.compile_ok,
            "best_correct": b_row.correct if b_row else False,
            "best_frac": (b_row.fraction_of_ceiling if (b_row and b_row.correct) else None),
            "best_achieved": (b_row.achieved if (b_row and b_row.correct) else None),
            "best_unit": (b_row.achieved_unit if b_row else ""),
            "best_fail": best_res.fail_reason,
            "ceiling": (s_row.ceiling if s_row else None),
            "ceiling_unit": (s_row.ceiling_unit if s_row else ""),
            "run_dir": d.name,
        })

# Markdown table
print("| Task | Held-out config | Seed (frac) | Best (frac) | Seed (abs) | Best (abs) | Speedup | Notes |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    seed_f  = f"{r['seed_frac']*100:.2f}%" if r['seed_frac']  is not None else "FAIL"
    best_f  = f"{r['best_frac']*100:.2f}%" if r['best_frac']  is not None else "FAIL"
    seed_a  = f"{r['seed_achieved']:.1f} {r['seed_unit']}" if r['seed_achieved'] is not None else "—"
    best_a  = f"{r['best_achieved']:.1f} {r['best_unit']}" if r['best_achieved'] is not None else "—"
    if r['seed_frac'] and r['best_frac']:
        sp = f"{r['best_frac']/r['seed_frac']:.2f}×"
    else:
        sp = "—"
    notes = []
    if not r['seed_compile_ok']: notes.append(f"seed COMPILE FAIL: {r['seed_fail']}")
    elif not r['seed_correct']:  notes.append(f"seed CORRECTNESS FAIL: {r['seed_fail']}")
    if not r['best_compile_ok']: notes.append(f"best COMPILE FAIL: {r['best_fail']}")
    elif not r['best_correct']:  notes.append(f"best CORRECTNESS FAIL: {r['best_fail']}")
    note_str = "; ".join(notes) if notes else ""
    print(f"| {r['task']} | {r['held_label']} | {seed_f} | {best_f} | {seed_a} | {best_a} | **{sp}** | {note_str} |")

# Persist raw JSON
out = ROOT / "_run_logs" / f"held_out_results_{MODEL}.json"
out.write_text(json.dumps(rows, indent=2, default=str))
print(f"\nRaw results: {out}")
