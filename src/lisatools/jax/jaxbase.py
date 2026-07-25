"""
jaxbase.py
==========
Base class and utilities for JAX-based subclasses. It includes a mixin for
JAX-backed waveform/computation classes that (a) bridges host/cupy inputs
into JAX arrays on the caller's target GPU via DLPack, and (b) implements a threading-safety protocol by tracking a per-(callable, device)
compile-shape registry.

The mixin exists so `DomainComputationGroupArray` (DCGA) can dispatch a
JAX-tracing callable concurrently across multiple GPUs without tripping
XLA's per-device compile lock. Callers consult the protocol methods
(`supports_threaded`, `record_completion`) on the callable's owner before
every threaded dispatch and auto-falls-back to serial on novel shapes.
Non-JAX callables (cupy-only, C++, pure NumPy) do not need this mixin;
they are treated as default-safe.
The main consumer of the protocol is the :class:`DomainComputationGroupArray` in ``domaincomputation.py``.

Preconditions for correct JAX placement via DLPack:
  - JAX and cupy must enumerate CUDA devices in the same order. Defaults
    do (both follow ``CUDA_VISIBLE_DEVICES`` / PCI order). Breaks if
    ``JAX_PLATFORMS`` or ``XLA_FLAGS`` restricts JAX's visible devices
    differently from cupy's.
  - DLPack handoff assumes the producer stream has synchronized. For
    host -> cupy via ``cp.asarray(numpy_array)``, this is effectively
    synchronous (pageable memory). For device-resident cupy arrays
    produced on non-default streams, ``_jax_from_host`` syncs the array's
    device before the bridge as a defensive measure.
"""
from __future__ import annotations

from typing import Any, Tuple, Optional

import numpy as np

try:
    import cupy as cp

    cupy_available = True
except (ImportError, ModuleNotFoundError):
    import numpy as cp  # type: ignore
    cupy_available = False

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    jax_available = True
except (ImportError, ModuleNotFoundError):
    jax = None  # type: ignore
    import numpy as jnp # type: ignore
    jax_available = False


class JaxBase:
    """Base class for JAX-backed waveforms and computations.

    This provides common logic for JAX-based subclasses and their interaction with array libraries.
    """

    def __init__(self) -> None:
        """Initialize the JaxBase instance."""
        if not jax_available:
            raise ImportError("JAX is required for JaxBase but is not available.")
        
    def _to_jax(self, 
                x: float | np.ndarray | cp.ndarray, 
                device_id: Optional[int] = None, 
                do_synchronize: bool = False) -> jnp.ndarray:
        """Convert input to a JAX array, optionally synchronizing if it's a CuPy array."""
        if isinstance(x, cp.ndarray) and cupy_available:
            if do_synchronize:
                x.device.synchronize()
            return jax.dlpack.from_dlpack(x)
    
        elif isinstance(x, (float, int, np.ndarray)):
            platform = jax.default_backend()
            device = jax.devices(platform)[device_id] if device_id is not None else None
            return jax.device_put(x, device=device)
           
        else:
            raise TypeError(f"Unsupported type for _to_jax: {type(x)}")
        
    def _from_jax(self, x: jnp.ndarray, do_synchronize: bool = False, to_host: bool = False) -> np.ndarray | cp.ndarray:
        """Convert a JAX array back to a NumPy or CuPy array matching the consumer's backend.

        The target is ``self.xp`` (numpy for a CPU backend, cupy for a GPU
        backend), NOT merely "cupy if importable" -- otherwise a CPU-backend
        waveform on a GPU box (cupy installed) would be handed device arrays
        that its numpy response then rejects. Callers rely on this: they index
        the result against other ``_from_jax`` outputs and pass it straight into
        ``xp=self.xp`` helpers.
        """
        if do_synchronize:
            x.block_until_ready()

        xp = getattr(self, "xp", np)
        want_cupy = (
            cupy_available
            and not to_host
            and getattr(xp, "__name__", "numpy") != "numpy"
        )
        if not want_cupy:
            return np.asarray(x)  # JAX (any device) -> host numpy

        # GPU backend: zero-copy when x is already on the device; otherwise
        # (JAX resolved onto CPU) bounce through host so the result is cupy.
        try:
            return cp.from_dlpack(x)
        except Exception:
            return cp.asarray(np.asarray(x))
        