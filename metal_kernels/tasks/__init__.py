"""Concrete benchmark tasks.

Importing this package registers every task in the global task registry.
"""

from . import saxpy   # noqa: F401

# heat2d and nbody are imported lazily to keep the smoke test minimal;
# importing this package registers whatever concrete tasks are available.
try:
    from . import heat2d  # noqa: F401
except ImportError:
    pass
try:
    from . import nbody   # noqa: F401
except ImportError:
    pass
try:
    from . import lbm     # noqa: F401
except ImportError:
    pass
try:
    from . import lj      # noqa: F401
except ImportError:
    pass
try:
    from . import wave3d  # noqa: F401
except ImportError:
    pass
try:
    from . import ising   # noqa: F401
except ImportError:
    pass
try:
    from . import hmc     # noqa: F401
except ImportError:
    pass
try:
    from . import gradshaf  # noqa: F401
except ImportError:
    pass
try:
    from . import fft3d   # noqa: F401
except ImportError:
    pass
