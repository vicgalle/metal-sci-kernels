#!/usr/bin/env python3
"""Convergence curves: best-so-far self-speedup vs iteration, per task per model."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path("/Users/victorgallego/metal-kernels/results")
TASKS = ["saxpy","heat2d","nbody","lbm","gradshaf","fft3d","hmc","ising","lj","wave3d"]
MODELS = [
    ("claude-opus-4-7",        "Opus 4.7",         "#C04A2A"),
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro",    "#5B2D8A"),
    ("gpt-5.5",                "GPT-5.5",          "#0F8A4F"),
]

# Typography & global style.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 9.5,
    "axes.titleweight": "regular",
    "axes.labelsize": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

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

fig, axes = plt.subplots(2, 5, figsize=(11.0, 4.7),
                         sharex=False, sharey=False)
plt.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.11,
                    wspace=0.42, hspace=0.55)

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
        ax.step(iters, ratio, where="post", color=col, lw=1.6,
                solid_joinstyle="round", solid_capstyle="round",
                label=label, zorder=3)
        # Mark the iteration that achieved the final best.
        final_best = ratio[-1]
        for i, r in enumerate(ratio):
            if r == final_best:
                ax.plot(iters[i], r, "o", color=col, ms=4.2,
                        mec="white", mew=0.8, zorder=4)
                break
        # Tick the failed candidates along the seed line.
        for f in fails:
            ax.plot(f, 1.0, "x", color=col, ms=3.8, mew=1.0,
                    alpha=0.55, zorder=2)
    ax.set_title(t, fontsize=9.5, pad=4, color="#222222")
    # Reference: y = 1 (seed).
    ax.axhline(1.0, color="#888888", lw=0.6, ls=(0, (2, 2)), alpha=0.7, zorder=1)
    ax.set_xlim(-0.5, max_iter + 0.5)
    span = max(ymax - 1.0, 0.05)
    ax.set_ylim(1.0 - 0.05*span, ymax + 0.12*span)
    # Horizontal-only grid, very subtle.
    ax.grid(False)
    ax.yaxis.grid(True, lw=0.4, color="#D8D8D8", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=2)
    # Tight x ticks.
    if max_iter <= 10: ax.set_xticks([0,2,4,6,8,10])
    elif max_iter <= 15: ax.set_xticks([0,3,6,9,12,15])
    else: ax.set_xticks([0,5,10,15,20,25])
    # Trim left/bottom spines slightly inward feel by not extending beyond data.
    for s in ("left","bottom"):
        ax.spines[s].set_color("#666666")

# Y/X axis labels on shared edges.
for ax in axes[:,0]:
    ax.set_ylabel("speedup", fontsize=8.5, color="#222222", labelpad=2)
for ax in axes[1,:]:
    ax.set_xlabel("iteration", fontsize=8.5, color="#222222", labelpad=2)

# Legend: top centered, single row, refined.
handles = [plt.Line2D([0],[0], color=c, lw=1.8, label=l,
                      solid_capstyle="round") for _,l,c in MODELS]
handles += [plt.Line2D([0],[0], marker="x", color="#888888", lw=0,
                       ms=4.5, mew=1.1, label="compile / correctness fail")]
fig.legend(handles=handles, loc="upper center", ncol=4,
           frameon=False, fontsize=8.8, handlelength=1.8,
           handletextpad=0.6, columnspacing=2.4,
           bbox_to_anchor=(0.5, 1.005))

out = Path("/Users/victorgallego/metal-kernels/paper/figs/convergence.png")
fig.savefig(out, bbox_inches="tight", dpi=260)
print(f"wrote {out}")
