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

from typing import Any, Tuple

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

    jax_available = True
except (ImportError, ModuleNotFoundError):
    jax = None  # type: ignore
    import numpy as jnp # type: ignore
    jax_available = False


class JaxThreadingMixin:
    """Mixin for JAX-tracing waveform / computation classes driven by DCGA.

    Provides two orthogonal capabilities:

    1. **Device-preserving host -> JAX bridge** via :meth:`_to_jax`.
       Under the caller's cupy TLS device context (set by DCGA's
       ``_threaded_device_context``), the bridge routes host NumPy input
       through ``cp.asarray`` -> ``jax.dlpack.from_dlpack``, producing a
       JAX array on the same CUDA device as the cupy allocation. This
       avoids ``jax.default_device`` (process-global, thread-unsafe)
       entirely. Subclasses should call ``self._to_jax(x)`` in
       place of ``jnp.asarray(x)`` for every host input that would
       otherwise trace on the default device.

    2. **Compile-shape registry** implementing the DCGA threading-safety
       protocol. The registry maps ``(callable_id, device_id)`` to the
       set of shape keys already compiled on that device. DCGA consults
       :meth:`supports_threaded` before threaded dispatch and only goes
       threaded when every non-empty split's ``(callable, device, shape)``
       triple is in the registry; otherwise it falls back to the serial
       dispatcher (which is safe under XLA's per-device compile lock).
       After a successful threaded or serial dispatch, DCGA calls
       :meth:`record_completion` so subsequent calls with the same shape
       take the threaded fast path.

    The registry is owned per waveform *instance*. Multiple DCGAs sharing
    the same waveform share the registry, so compiles paid by one DCGA
    help the others.

    IMPORTANT: JAX-tracing waveforms MUST mix this in AND route host
    inputs through :meth:`_to_jax`. Without the DLPack bridge,
    threaded dispatch will place JAX arrays on the wrong device (or
    deadlock on first compile) even when the registry logic is correct.
    """

    # No ``__init__`` — the mixin is intentionally init-free so it can be
    # plugged into any class regardless of how that class wires its base
    # initializers (cooperative ``super()`` vs. the explicit per-base-init
    # convention used elsewhere in this codebase). The compile registry is
    # lazily created on first access via :meth:`_get_compile_registry`.

    def _get_compile_registry(self) -> dict[Tuple[str, int], set[tuple]]:
        """Return the per-instance registry, creating it on first access.

        Stored in ``self.__dict__`` rather than as a class attribute so that
        sibling instances do not share state.
        """
        reg = self.__dict__.get("_compile_registry")
        if reg is None:
            reg = {}
            self.__dict__["_compile_registry"] = reg
        return reg

    # ------------------------------------------------------------------ #
    # Host -> JAX bridge                                                 #
    # ------------------------------------------------------------------ #

    @property
    def _on_gpu(self) -> bool:
        """Best-effort "am I on a CUDA backend?" check.

        Resolves via ``self.xp is not np`` when the subclass inherits from
        ``LISAToolsParallelModule`` (which populates ``self.xp``). Falls
        back to ``False`` for pure-CPU / test subclasses that lack ``xp``.
        Subclasses may override if they expose a different flag.
        """
        xp = getattr(self, "xp", None)
        return xp is not None and xp is not np

    def _to_jax(self, array: np.ndarray | cp.ndarray) -> jnp.ndarray:
        """Place a host (or cupy) array as a JAX array on the current
        cupy TLS device via the DLPack bridge.

        - Host numpy array on GPU backend: ``cp.asarray(array)`` places on
          the current cupy device (``cupy.cuda.Device()`` TLS context set
          by the worker's ``_threaded_device_context``); the H2D copy is
          stream-ordered on that device's default stream, so DLPack sees
          a ready buffer.
        - Device-resident cupy array: defensively sync the array's device
          (covers the case of an upstream non-default-stream producer),
          then DLPack-bridge zero-copy.
        - CPU backend (no ``cupy``, or ``_on_gpu == False``): fall back to
          ``jnp.asarray``; placement is irrelevant on CPU.

        The result is a JAX array whose ``.device()`` is
        ``jax.devices("gpu")[N]`` when the cupy TLS device was N, without
        any ``jax.default_device`` involvement.
        """
        if not jax_available:
            raise RuntimeError(
                "_to_jax called but JAX is not importable. "
                "Install jax to use JAX-backed waveforms."
            )

        if not self._on_gpu or not cupy_available:
            return jnp.asarray(array)

        if isinstance(array, cp.ndarray):
            # Defensive sync: future callers may produce arrays on a
            # non-default stream. For arrays produced synchronously on
            # the default stream (the common case), this is a no-op.
            array.device.synchronize()
            return jax.dlpack.from_dlpack(array)

        cp_arr = cp.asarray(array)
        return jax.dlpack.from_dlpack(cp_arr)
    
    def _to_cupy(self, array: jnp.ndarray) -> cp.ndarray:
        """Convert a JAX array to cupy via DLPack.

        The result is a cupy array on the same device as the input JAX
        array, without any ``jax.default_device`` involvement.
        """
        if not jax_available:
            raise RuntimeError(
                "_to_cupy called but JAX is not importable. "
                "Install jax to use JAX-backed waveforms."
            )
        if not cupy_available:
            raise RuntimeError(
                "_to_cupy called but cupy is not importable. "
                "Install cupy to use GPU-backed waveforms."
            )
        
        # make sure the array is fully on the device and not a lazy view
        array.block_until_ready()

        return cp.from_dlpack(array)
    
    def _to_host(self, array: jnp.ndarray) -> np.ndarray:
        """Convert a JAX array to host numpy.

        The result is a host numpy array with the same contents as the input
        JAX array. Placement is irrelevant on CPU.
        """
        if not jax_available:
            raise RuntimeError(
                "_to_host called but JAX is not importable. "
                "Install jax to use JAX-backed waveforms."
            )

        return np.asarray(array)
    
    def _from_jax(self, array: jnp.ndarray) -> np.ndarray | cp.ndarray:
        """Convert a JAX array to either cupy or host numpy, depending on
        the backend.

        - GPU backend: convert to cupy via DLPack; the result is on the
          same device as the input JAX array.
        - CPU backend: convert to host numpy via ``np.asarray``; placement
          is irrelevant on CPU.
        """
        if not jax_available:
            raise RuntimeError(
                "_from_jax called but JAX is not importable. "
                "Install jax to use JAX-backed waveforms."
            )

        if self._on_gpu and cupy_available:
            return self._to_cupy(array)
        else:
            return self._to_host(array)

    # ------------------------------------------------------------------ #
    # Threading-safety protocol                                          #
    # ------------------------------------------------------------------ #

    def _shape_key(self, *args: Any, **kwargs: Any) -> tuple:
        """Derive a hashable key capturing the per-trace-varying shape of
        a call.

        Conventions:
        - The first positional argument is assumed to be an array-like
          whose leading dimension ``N`` is the trace-varying axis
          (batch of binaries). This matches all JAX waveforms in this
          codebase.
        - Static kwargs (Python scalars and strings) contribute to the
          key, because they typically map to JIT ``static_argnums``.
        - Array-valued kwargs are ignored; subclasses with array kwargs
          that affect tracing should override this method.
        """
        n = 0
        if args and hasattr(args[0], "shape") and len(args[0].shape) > 0:
            n = int(args[0].shape[0])
        kw_items = tuple(
            sorted(
                (k, v)
                for k, v in kwargs.items()
                if isinstance(v, (int, float, bool, str, type(None)))
            )
        )
        return (n, kw_items)

    def supports_threaded(
        self, callable_id: str, device_id: int, *args: Any, **kwargs: Any
    ) -> bool:
        """Return True iff ``(callable_id, device_id, shape(args))`` has
        already been recorded as compiled.

        Empty-input short-circuit: a call whose leading array has
        ``shape[0] == 0`` traces nothing and is always safe to dispatch.
        """
        if (
            args
            and hasattr(args[0], "shape")
            and len(args[0].shape) > 0
            and args[0].shape[0] == 0
        ):
            return True
        key = self._shape_key(*args, **kwargs)
        return key in self._get_compile_registry().get((callable_id, device_id), set())

    def record_completion(
        self, callable_id: str, device_id: int, *args: Any, **kwargs: Any
    ) -> None:
        """Register ``(callable_id, device_id, shape(args))`` as compiled.

        DCGA calls this after a successful dispatch (serial or threaded);
        subsequent calls with the same key take the threaded fast path.
        """
        if (
            args
            and hasattr(args[0], "shape")
            and len(args[0].shape) > 0
            and args[0].shape[0] == 0
        ):
            return
        key = self._shape_key(*args, **kwargs)
        self._get_compile_registry().setdefault((callable_id, device_id), set()).add(key)

    def clear_compile_registry(self) -> None:
        """Evict every recorded shape.

        Call after invalidating JAX's own compile caches (e.g.
        ``jax.clear_caches()``) so the registry doesn't incorrectly
        claim a shape is still compiled.
        """
        self._get_compile_registry().clear()
