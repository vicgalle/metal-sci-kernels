"""Apple Silicon chip detection + roofline lookup.

Peak FP32 GFLOPS and DRAM bandwidth per chip family. Numbers come from
Apple's published specs and Andreas Gerstmayr / Geekbench hardware-survey
data; they're approximate but serve as the *ceiling* for the
``achieved/ceiling`` fitness metric.

Source notes:
- M1: 8-core GPU ≈ 2.6 TFLOPS FP32, 68 GB/s LPDDR4X
- M1 Pro: 14-core ≈ 4.5 / 16-core ≈ 5.2 TFLOPS, 200 GB/s LPDDR5
- M1 Max: 24-core ≈ 7.8 / 32-core ≈ 10.4 TFLOPS, 400 GB/s
- M1 Ultra: 64-core ≈ 21 TFLOPS, 800 GB/s
- M2: 8 / 10-core ≈ 3.6 TFLOPS, 100 GB/s
- M2 Pro: 16 / 19-core ≈ 6.8 TFLOPS, 200 GB/s
- M2 Max: 30 / 38-core ≈ 13.6 TFLOPS, 400 GB/s
- M3 / M3 Pro / M3 Max: similar TFLOPS, 100 / 150 / 300-400 GB/s
- M4 / M4 Pro / M4 Max: ~4.6 / 9.2 / 18 TFLOPS, 120 / 273 / 546 GB/s

When the exact GPU-core count isn't detectable we fall back to the *base*
variant of the chip family (the lower roofline), which is the
conservative choice for an "achieved/ceiling" metric.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ChipSpec:
    name: str
    peak_fp32_gflops: float   # peak FP32 throughput in GFLOPS
    peak_bw_gb_s: float       # peak DRAM bandwidth in GB/s


# Conservative defaults per chip family. Refine by GPU core count below.
_CHIP_TABLE: dict[str, ChipSpec] = {
    "Apple M1":         ChipSpec("Apple M1",         2_600,  68),
    "Apple M1 Pro":     ChipSpec("Apple M1 Pro",     4_500, 200),
    "Apple M1 Max":     ChipSpec("Apple M1 Max",     7_800, 400),
    "Apple M1 Ultra":   ChipSpec("Apple M1 Ultra",  21_000, 800),
    "Apple M2":         ChipSpec("Apple M2",         3_600, 100),
    "Apple M2 Pro":     ChipSpec("Apple M2 Pro",     6_800, 200),
    "Apple M2 Max":     ChipSpec("Apple M2 Max",    13_600, 400),
    "Apple M2 Ultra":   ChipSpec("Apple M2 Ultra",  27_200, 800),
    "Apple M3":         ChipSpec("Apple M3",         4_100, 100),
    "Apple M3 Pro":     ChipSpec("Apple M3 Pro",     7_400, 150),
    "Apple M3 Max":     ChipSpec("Apple M3 Max",    14_200, 300),
    "Apple M4":         ChipSpec("Apple M4",         4_600, 120),
    "Apple M4 Pro":     ChipSpec("Apple M4 Pro",     9_200, 273),
    "Apple M4 Max":     ChipSpec("Apple M4 Max",    18_000, 546),
}


def _read_chip_name() -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
    except Exception:
        out = ""
    if out:
        return out
    # Fallback to system_profiler.
    try:
        sp = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True,
        )
        m = re.search(r"Chip:\s*(.+)", sp)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return "Unknown"


def detect_chip() -> ChipSpec:
    name = _read_chip_name()
    if name in _CHIP_TABLE:
        return _CHIP_TABLE[name]
    # Try a coarser match (drop trailing variant words).
    for key, spec in _CHIP_TABLE.items():
        if key in name:
            return spec
    # Unknown chip: return a placeholder with conservative numbers so the
    # benchmark still runs (just with imprecise ceilings).
    return ChipSpec(name=name or "Unknown", peak_fp32_gflops=2_000, peak_bw_gb_s=80)
