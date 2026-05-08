#!/usr/bin/env python3
"""Compare claude-opus-4-7 vs gemini-3.1-pro-preview across all 10 tasks."""
from __future__ import annotations
import json, sys
from pathlib import Path

sys.path.insert(0, "/Users/victorgallego/metal-kernels")
from metal_kernels import tasks  # noqa: F401
from metal_kernels.harness import MetalHarness
from metal_kernels.hardware import detect_chip
from metal_kernels.task import get_task

ROOT = Path("/Users/victorgallego/metal-kernels/results")
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]
MODELS = ["claude-opus-4-7", "gemini-3.1-pro-preview"]

def pick_dir(task: str, model: str) -> Path | None:
    cands = sorted(ROOT.glob(f"{task}_{model}_*"))
    cands = [c for c in cands if (c/"summary.json").exists()
             and (c/"best.metal").exists() and (c/"00_seed.metal").exists()]
    if not cands: return None
    # Prefer the run with the highest n_iterations (one canonical pick per
    # (task, model) pair).
    def n_iters(c):
        return json.loads((c/"summary.json").read_text()).get("n_iterations", -1)
    return max(cands, key=n_iters)

def load_summary(task: str, model: str):
    d = pick_dir(task, model)
    if d is None:
        return None
    s = json.loads((d/"summary.json").read_text())
    hist = s["history"]
    seed_score = s["seed_score"]
    best_score = s["best_score"]
    n_iter = s["n_iterations"]
    best_iter = 0
    for h in hist:
        if h["role"]=="candidate" and h.get("score")==best_score:
            best_iter = h["iteration"]; break
    cands = [h for h in hist if h["role"]=="candidate"]
    cf = sum(1 for h in cands if not h["compile_ok"])
    kf = sum(1 for h in cands if h["compile_ok"] and not h["correct"])
    elapsed = sum(h.get("elapsed_s",0) for h in hist)
    return {
        "dir": d.name, "n_iter": n_iter, "seed": seed_score, "best": best_score,
        "speedup": best_score/seed_score if seed_score and best_score else None,
        "best_iter": best_iter, "compile_fails": cf, "correct_fails": kf,
        "n_candidates": len(cands), "wall_min": elapsed/60,
    }

# In-distribution table
print("# In-distribution: claude-opus-4-7 vs gemini-3.1-pro-preview\n")
print("| Task | iters (O / G) | Seed (≈ same) | O Best | G Best | O speedup | G speedup | O fails (cmp/corr) | G fails (cmp/corr) |")
print("|---|---|---|---|---|---|---|---|---|")
data = {}
for t in TASKS:
    o = load_summary(t, "claude-opus-4-7")
    g = load_summary(t, "gemini-3.1-pro-preview")
    data[t] = {"opus": o, "gemini": g}
    seed = (o["seed"] if o else g["seed"]) if (o or g) else None
    seed_s = f"{seed:.4f}" if seed is not None else "—"
    o_iter = str(o["n_iter"]) if o else "—"
    g_iter = str(g["n_iter"]) if g else "—"
    o_best = f"{o['best']:.4f}" if o else "—"
    g_best = f"{g['best']:.4f}" if g else "—"
    o_sp = f"{o['speedup']:.2f}×" if o else "—"
    g_sp = f"{g['speedup']:.2f}×" if g else "—"
    o_f = f"{o['compile_fails']}/{o['correct_fails']}" if o else "—"
    g_f = f"{g['compile_fails']}/{g['correct_fails']}" if g else "—"
    print(f"| {t} | {o_iter} / {g_iter} | {seed_s} | {o_best} | {g_best} | **{o_sp}** | **{g_sp}** | {o_f} | {g_f} |")

# Wall-time table
print("\n# Wall time (LLM + harness)\n")
print("| Task | iters (O / G) | O wall (min) | G wall (min) |")
print("|---|---|---|---|")
for t in TASKS:
    o, g = data[t]["opus"], data[t]["gemini"]
    print(f"| {t} | {o['n_iter'] if o else '—'} / {g['n_iter'] if g else '—'} | "
          f"{o['wall_min']:.1f} | {g['wall_min']:.1f} |"
          if o and g else
          f"| {t} | {o['n_iter'] if o else '—'} / {g['n_iter'] if g else '—'} | "
          f"{o['wall_min']:.1f if o else '—'} | {g['wall_min']:.1f if g else '—'} |")

# Held-out
print("\n# Held-out config: seed vs best by model\n")
chip = detect_chip()
harness = MetalHarness()
print(f"chip: {chip.name}\n")
print("| Task | Held-out | Seed frac | O best frac | O speedup | G best frac | G speedup |")
print("|---|---|---|---|---|---|---|")
held_rows = []
for t in TASKS:
    task = get_task(t)
    held = task.spec.held_out_sizes
    if not held: continue
    h = held[0]
    # Use the seed source from either run dir (they share the same seed file).
    o_dir = pick_dir(t, "claude-opus-4-7")
    g_dir = pick_dir(t, "gemini-3.1-pro-preview")
    seed_dir = o_dir or g_dir
    if seed_dir is None: continue
    seed_src = (seed_dir/"00_seed.metal").read_text()
    seed_res = task.evaluate_candidate(harness, chip, seed_src, sizes=held)
    sf = None
    if seed_res.compile_ok and seed_res.size_results and seed_res.size_results[0].correct:
        sf = seed_res.size_results[0].fraction_of_ceiling
    sf_s = f"{sf*100:.2f}%" if sf is not None else "FAIL"

    def best_frac(d):
        if d is None: return None, None
        src = (d/"best.metal").read_text()
        r = task.evaluate_candidate(harness, chip, src, sizes=held)
        if not r.compile_ok or not r.size_results or not r.size_results[0].correct:
            return None, r.fail_reason
        return r.size_results[0].fraction_of_ceiling, None
    of, oerr = best_frac(o_dir)
    gf, gerr = best_frac(g_dir)
    of_s = f"{of*100:.2f}%" if of is not None else "FAIL"
    gf_s = f"{gf*100:.2f}%" if gf is not None else "FAIL"
    osp = f"{of/sf:.2f}×" if (of and sf) else "—"
    gsp = f"{gf/sf:.2f}×" if (gf and sf) else "—"
    print(f"| {t} | {h.label} | {sf_s} | {of_s} | {osp} | {gf_s} | {gsp} |")
    held_rows.append({"task":t,"held":h.label,"seed_frac":sf,
                      "opus_frac":of,"opus_err":oerr,
                      "gem_frac":gf,"gem_err":gerr})

(ROOT/"_run_logs"/"both_held_out.json").write_text(json.dumps(held_rows, indent=2, default=str))
print(f"\nRaw held-out: results/_run_logs/both_held_out.json")
