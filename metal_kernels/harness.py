"""Metal compile + dispatch + timing harness.

Wraps PyObjC's Metal bindings into a small, opinionated API:

- ``MetalHarness.compile(source)`` runtime-compiles a ``.metal`` source string
  via ``MTLDevice.newLibraryWithSource``. No ``xcrun metal`` toolchain needed.
- ``make_pipelines`` returns a dict of ``MTLComputePipelineState`` keyed by
  kernel name.
- Buffer helpers (``buf_from_np``, ``np_view``) move data in/out of unified
  memory with a single copy.
- ``time_dispatch`` runs the user-supplied ``dispatch_fn`` once per command
  buffer and returns ``GPUEndTime - GPUStartTime``.
- ``time_repeated`` does warmup + measured iterations and reports median.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np
import Metal


@dataclass
class CompileResult:
    library: object | None
    error: str | None


@dataclass
class TimingResult:
    median_s: float
    min_s: float
    max_s: float
    iqr_s: float
    samples: list[float]


class MetalHarness:
    """Small, opinionated wrapper around PyObjC's Metal framework."""

    def __init__(self):
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise RuntimeError("No Metal device found")
        self.device = device
        self.queue = device.newCommandQueue()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def compile(self, source: str) -> CompileResult:
        """Compile a ``.metal`` source string. Returns (library, error_str)."""
        opts = Metal.MTLCompileOptions.new()
        # Fast-math is the Metal default; leave on. Future: surface as a knob.
        lib, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
        if lib is None:
            return CompileResult(None, str(err) if err else "unknown compile error")
        return CompileResult(lib, None)

    def make_pipelines(
        self, library, kernel_names: list[str]
    ) -> tuple[dict[str, object] | None, str | None]:
        """Build compute pipeline states for the given kernel names."""
        pipelines = {}
        for name in kernel_names:
            fn = library.newFunctionWithName_(name)
            if fn is None:
                return None, f"kernel function '{name}' not found in compiled library"
            pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
            if pso is None:
                return None, f"pipeline build failed for '{name}': {err}"
            pipelines[name] = pso
        return pipelines, None

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def buf_from_np(self, arr: np.ndarray):
        """Copy a numpy array into a new shared MTLBuffer."""
        arr = np.ascontiguousarray(arr)
        buf = self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared,
        )
        return buf

    def buf_zeros(self, nbytes: int):
        return self.device.newBufferWithLength_options_(
            nbytes, Metal.MTLResourceStorageModeShared,
        )

    def buf_scalar(self, value, dtype):
        """One-shot buffer holding a scalar (uint, int, float, ...)."""
        arr = np.array([value], dtype=dtype)
        return self.buf_from_np(arr)

    def np_view(self, buf, dtype, count: int) -> np.ndarray:
        """Return a numpy array view aliasing the buffer's contents.

        Caller should ``.copy()`` before mutating the buffer if a snapshot is
        needed.
        """
        nbytes = int(np.dtype(dtype).itemsize) * count
        mv = buf.contents().as_buffer(nbytes)
        return np.frombuffer(mv, dtype=dtype, count=count)

    # ------------------------------------------------------------------
    # Dispatch + timing
    # ------------------------------------------------------------------

    def time_dispatch(self, dispatch_fn) -> float:
        """Run ``dispatch_fn(encoder)`` inside one command buffer; return GPU seconds.

        ``dispatch_fn`` is a closure that takes a ``MTLComputeCommandEncoder``,
        sets pipeline + buffers, and emits dispatchThreads/dispatchThreadgroups
        calls. It may issue multiple dispatches (e.g. multi-kernel tasks); they
        will share one command buffer and one timing window.
        """
        cmdbuf = self.queue.commandBuffer()
        enc = cmdbuf.computeCommandEncoder()
        dispatch_fn(enc)
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()
        if cmdbuf.status() != Metal.MTLCommandBufferStatusCompleted:
            err = cmdbuf.error()
            raise RuntimeError(f"command buffer failed: {err}")
        return float(cmdbuf.GPUEndTime() - cmdbuf.GPUStartTime())

    def time_repeated(
        self, dispatch_fn, n_warmup: int = 3, n_measure: int = 10,
    ) -> TimingResult:
        """Warm up then measure. Returns median + IQR of GPU times."""
        for _ in range(n_warmup):
            self.time_dispatch(dispatch_fn)
        samples = [self.time_dispatch(dispatch_fn) for _ in range(n_measure)]
        arr = np.array(samples)
        q1, q3 = np.percentile(arr, [25, 75])
        return TimingResult(
            median_s=float(np.median(arr)),
            min_s=float(arr.min()),
            max_s=float(arr.max()),
            iqr_s=float(q3 - q1),
            samples=samples,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def device_name(self) -> str:
        return str(self.device.name())

    def max_threads_per_threadgroup(self) -> int:
        # The MTLSize struct returned has width/height/depth; for compute the
        # relevant cap is the product, but PyObjC exposes the struct directly.
        sz = self.device.maxThreadsPerThreadgroup()
        return int(sz.width) * int(sz.height) * int(sz.depth)


def threadgroup_1d(width: int) -> object:
    return Metal.MTLSizeMake(int(width), 1, 1)


def threadgroup_2d(w: int, h: int) -> object:
    return Metal.MTLSizeMake(int(w), int(h), 1)


def grid_1d(n: int) -> object:
    return Metal.MTLSizeMake(int(n), 1, 1)


def grid_2d(w: int, h: int) -> object:
    return Metal.MTLSizeMake(int(w), int(h), 1)
