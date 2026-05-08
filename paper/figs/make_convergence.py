#!/usr/bin/env python3
"""Convergence curves: best-so-far self-speedup vs iteration, per task per model."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path("/Users/victorgallego/metal-kernels/results")
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]
MODELS = [
    ("claude-opus-4-7",        "Opus 4.7",          "#5B2D8A"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro",    "#0F8A4F"),
    ("gpt-5.5",                "GPT-5.5",           "#C04A2A"),
]

def pick_dir(task: str, model: str) -> Path | None:
    cands = sorted(ROOT.glob(f"{task}_{model}_*"))
    cands = [c for c in cands if (c/"summary.json").exists()]
    if not cands: return None
    def n(c): return json.loads((c/"summary.json").read_text()).get("n_iterations",-1)
    return max(cands, key=n)

def best_so_far(task, model):
    d = pick_dir(task, model)
    if d is None: return None
    s = json.loads((d/"summary.json").read_text())
    seed = s["seed_score"]
    if not seed: return None
    iters, ratio, fails = [0], [1.0], []  # iter 0 = seed = 1.0× self
    cur = seed
    for h in s["history"]:
        if h["role"] != "candidate": continue
        it = h["iteration"]
        sc = h.get("score")
        if sc is None:
            fails.append(it)
            ratio.append(cur/seed)
        else:
            cur = max(cur, sc)
            ratio.append(cur/seed)
        iters.append(it)
    return iters, ratio, fails, s["n_iterations"]

fig, axes = plt.subplots(2, 5, figsize=(11.0, 4.6),
                         sharex=False, sharey=False)
plt.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.10,
                    wspace=0.35, hspace=0.45)

for ax, t in zip(axes.flat, TASKS):
    max_iter = 0
    ymin, ymax = 1.0, 1.0
    for (m, label, col) in MODELS:
        out = best_so_far(t, m)
        if out is None: continue
        iters, ratio, fails, niter = out
        max_iter = max(max_iter, niter)
        ymin = min(ymin, min(ratio))
        ymax = max(ymax, max(ratio))
        # Step plot to emphasise non-decreasing best-so-far.
        ax.step(iters, ratio, where="post", color=col, lw=1.3, label=label)
        # Mark the iteration that achieved the final best.
        final_best = ratio[-1]
        for i, r in enumerate(ratio):
            if r == final_best:
                ax.plot(iters[i], r, "o", color=col, ms=4); break
        # Tick the failed candidates along the seed line.
        for f in fails:
            ax.plot(f, 1.0, "x", color=col, ms=3.5, mew=0.9, alpha=0.55)
    ax.set_title(t, fontsize=9, pad=2)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":", alpha=0.7)
    ax.set_xlim(-0.5, max_iter + 0.5)
    span = max(ymax - 1.0, 0.05)
    ax.set_ylim(1.0 - 0.04*span, ymax + 0.10*span)
    ax.tick_params(labelsize=7, length=2)
    ax.grid(True, lw=0.3, alpha=0.4)
    # Tight x ticks.
    if max_iter <= 10: ax.set_xticks([0,2,4,6,8,10])
    elif max_iter <= 15: ax.set_xticks([0,3,6,9,12,15])
    else: ax.set_xticks([0,5,10,15,20,25])

# Y/X axis labels on shared edges.
for ax in axes[:,0]:
    ax.set_ylabel("best/seed", fontsize=8)
for ax in axes[1,:]:
    ax.set_xlabel("iteration", fontsize=8)

# Legend: top centered, single row.
handles = [plt.Line2D([0],[0], color=c, lw=1.5, label=l) for _,l,c in MODELS]
handles += [plt.Line2D([0],[0], marker="x", color="gray", lw=0,
                       ms=4, mew=0.9, label="compile/correctness fail")]
fig.legend(handles=handles, loc="upper center", ncol=4,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.00))

out = Path("/Users/victorgallego/metal-kernels/paper/figs/convergence.png")
fig.savefig(out, bbox_inches="tight", dpi=240)
print(f"wrote {out}")
