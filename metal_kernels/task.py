"""Task abstraction for the Metal kernel benchmark.

A ``Task`` packages everything needed to (a) describe a problem to an LLM,
(b) compile a candidate ``.metal`` source, (c) dispatch the kernels at
multiple problem sizes, (d) verify correctness against a CPU reference,
(e) compute achieved throughput, and (f) score the candidate as a
fraction of the architectural roofline.

Each concrete task subclasses :class:`Task` and implements
``evaluate_size``. All other plumbing (compile, multi-size loop, scoring)
lives in :func:`evaluate_candidate` below.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .harness import MetalHarness
from .hardware import ChipSpec


@dataclass
class TaskSize:
    """One problem size for a task."""
    label: str
    params: dict[str, Any]


@dataclass
class TaskSpec:
    name: str
    description: str
    kernel_signatures: str          # text block shown to the LLM
    kernel_names: list[str]         # functions to extract from compiled lib
    seed_path: Path
    sizes: list[TaskSize]
    held_out_sizes: list[TaskSize] = field(default_factory=list)


@dataclass
class SizeResult:
    size_label: str
    correct: bool
    error_value: float              # task-specific error metric
    error_kind: str                 # e.g. "max_abs", "rel_residual"
    gpu_seconds: float
    achieved: float                 # GFLOPS or GB/s
    achieved_unit: str
    ceiling: float
    ceiling_unit: str
    fraction_of_ceiling: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateResult:
    compile_ok: bool
    compile_error: str | None
    pipeline_error: str | None
    size_results: list[SizeResult]
    score: float | None             # None if any size failed correctness
    fail_reason: str | None         # human-readable hint for the LLM


class Task(ABC):
    """Base class for benchmark tasks."""

    spec: TaskSpec

    @abstractmethod
    def evaluate_size(
        self,
        harness: MetalHarness,
        pipelines: dict[str, object],
        size: TaskSize,
        chip: ChipSpec,
        n_warmup: int,
        n_measure: int,
    ) -> SizeResult:
        ...

    def evaluate_candidate(
        self,
        harness: MetalHarness,
        chip: ChipSpec,
        source: str,
        n_warmup: int = 3,
        n_measure: int = 10,
        sizes: list[TaskSize] | None = None,
    ) -> CandidateResult:
        """Compile + run + verify + score a candidate ``.metal`` source."""
        sizes = sizes if sizes is not None else self.spec.sizes
        cr = harness.compile(source)
        if cr.error is not None:
            return CandidateResult(
                compile_ok=False, compile_error=cr.error,
                pipeline_error=None, size_results=[], score=None,
                fail_reason=f"compile error: {cr.error}",
            )
        pipelines, perr = harness.make_pipelines(cr.library, self.spec.kernel_names)
        if perr is not None:
            return CandidateResult(
                compile_ok=True, compile_error=None,
                pipeline_error=perr, size_results=[], score=None,
                fail_reason=f"pipeline error: {perr}",
            )

        size_results: list[SizeResult] = []
        for size in sizes:
            try:
                res = self.evaluate_size(
                    harness, pipelines, size, chip, n_warmup, n_measure,
                )
            except Exception as e:
                size_results.append(SizeResult(
                    size_label=size.label, correct=False,
                    error_value=float("inf"), error_kind="exception",
                    gpu_seconds=0.0, achieved=0.0, achieved_unit="",
                    ceiling=0.0, ceiling_unit="",
                    fraction_of_ceiling=0.0,
                    extra={"exception": str(e)},
                ))
                return CandidateResult(
                    compile_ok=True, compile_error=None,
                    pipeline_error=None,
                    size_results=size_results,
                    score=None,
                    fail_reason=f"runtime error at size {size.label}: {e}",
                )
            size_results.append(res)
            if not res.correct:
                return CandidateResult(
                    compile_ok=True, compile_error=None,
                    pipeline_error=None,
                    size_results=size_results,
                    score=None,
                    fail_reason=(
                        f"correctness failed at size {size.label}: "
                        f"{res.error_kind}={res.error_value:.3e}"
                    ),
                )

        # All sizes correct → gmean of fraction_of_ceiling.
        fractions = [r.fraction_of_ceiling for r in size_results]
        # Guard against zero (would NaN the gmean).
        log_sum = sum(math.log(max(f, 1e-12)) for f in fractions)
        score = math.exp(log_sum / max(len(fractions), 1))
        return CandidateResult(
            compile_ok=True, compile_error=None, pipeline_error=None,
            size_results=size_results, score=score, fail_reason=None,
        )


# Convenient throughput formulas
def gb_per_s(bytes_moved: float, seconds: float) -> float:
    return bytes_moved / seconds / 1e9


def gflops(flops: float, seconds: float) -> float:
    return flops / seconds / 1e9


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_TASK_REGISTRY: dict[str, type[Task]] = {}


def register_task(name: str):
    """Decorator: register a Task subclass under ``name``."""
    def decorator(cls):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator


def get_task(name: str) -> Task:
    if name not in _TASK_REGISTRY:
        raise KeyError(
            f"unknown task: {name!r}. Available: {sorted(_TASK_REGISTRY)}"
        )
    return _TASK_REGISTRY[name]()


def list_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY)
