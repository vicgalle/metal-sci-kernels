#!/usr/bin/env python3
"""Summarize claude-opus-4-7 runs across all 10 tasks."""
from __future__ import annotations
import json, os, glob
from pathlib import Path

ROOT = Path("/Users/victorgallego/metal-kernels/results")
MODEL = "claude-opus-4-7"

# Pick one canonical run per task. For tasks with multiple opus-4-7 dirs,
# prefer the most recent run with summary.json.
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]

def pick_dir(task: str) -> Path | None:
    cands = sorted(ROOT.glob(f"{task}_{MODEL}_*"))
    cands = [c for c in cands if (c / "summary.json").exists()]
    if not cands:
        return None
    # If a 10-iter run exists, prefer it; otherwise take the latest.
    ten = [c for c in cands
           if json.loads((c/"summary.json").read_text()).get("n_iterations") == 10]
    return ten[-1] if ten else cands[-1]

rows = []
for t in TASKS:
    d = pick_dir(t)
    if not d:
        rows.append((t, None, None, None, None, None, None, None, None, "no run"))
        continue
    s = json.loads((d/"summary.json").read_text())
    hist = s["history"]
    seed_score = s["seed_score"]
    best_score = s["best_score"]
    n_iter = s["n_iterations"]

    # Iter index that produced best_score (skip seed if a candidate matched).
    best_iter = None
    for h in hist:
        if h["role"] == "candidate" and h.get("score") == best_score:
            best_iter = h["iteration"]
            break
    if best_iter is None:
        # Best is the seed itself.
        best_iter = 0

    compile_fails = sum(1 for h in hist if h["role"]=="candidate" and not h["compile_ok"])
    correct_fails = sum(
        1 for h in hist
        if h["role"]=="candidate" and h["compile_ok"] and not h["correct"]
    )
    candidates = sum(1 for h in hist if h["role"]=="candidate")
    speedup = (best_score / seed_score) if seed_score and best_score else None
    elapsed = sum(h.get("elapsed_s",0) for h in hist)

    rows.append((t, n_iter, seed_score, best_score, speedup,
                 best_iter, compile_fails, correct_fails, candidates,
                 d.name, elapsed))

# Print markdown table
print(f"## {MODEL} sweep over 10 tasks (M1 Pro)\n")
print("| Task | iters | Seed | Best | Speedup | Best iter | Compile fails | Correctness fails | Wall time |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r[1] is None:
        t, *_, note = r
        print(f"| {t} | — | — | — | — | — | — | — | _{note}_ |")
        continue
    t, n, seed, best, sp, bi, cf, kf, cands, name, el = r
    print(f"| {t} | {n} | {seed:.4f} | {best:.4f} | **{sp:.2f}×** | {bi} | {cf}/{cands} | {kf}/{cands} | {el/60:.1f} min |")

# Detail: per-size best for each task
print("\n## Best-iteration size breakdown (fraction-of-ceiling)\n")
for r in rows:
    if r[1] is None:
        continue
    t, *_, name, el = r
    d = ROOT / name
    br = json.loads((d/"best_result.json").read_text())
    parts = [f"{s['label']}: {s['fraction_of_ceiling']*100:.1f}%" for s in br["sizes"]]
    print(f"- **{t}** — " + " | ".join(parts))

print()
