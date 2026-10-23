"""JAX array-bridge mixin for JAX-backed waveform / computation classes.

Holds :class:`JaxBase`, whose two helpers move inputs into JAX
(:meth:`JaxBase._to_jax`: a cupy array zero-copy through DLPack, a host
scalar / numpy array with ``jax.device_put`` onto a chosen device) and
outputs back to the consumer's array module (:meth:`JaxBase._from_jax`:
numpy, or cupy through DLPack when the consumer's ``xp`` is cupy). Its
global-fit user is the phentax MBH generator
:class:`~lisatools.sources.bbh.waveform.PhenomTHMWaveformBase` (production
MBH waveforms). :mod:`lisatools.utils.jaxbase` re-exports it. Importing the
module turns on ``jax_enable_x64`` when jax is available.

(An earlier version also carried a per-(callable, device) compile-shape
registry for threaded multi-GPU dispatch; those protocol methods are no
longer in this class.)

Preconditions for correct JAX placement via DLPack:
  - JAX and cupy must enumerate CUDA devices in the same order. Defaults
    do (both follow ``CUDA_VISIBLE_DEVICES`` / PCI order). Breaks if
    ``JAX_PLATFORMS`` or ``XLA_FLAGS`` restricts JAX's visible devices
    differently from cupy's.
  - DLPack handoff assumes the producer stream has synchronized. For
    host -> cupy via ``cp.asarray(numpy_array)``, this is effectively
    synchronous (pageable memory). For device-resident cupy arrays
    produced on non-default streams, pass ``do_synchronize=True`` to
    :meth:`JaxBase._to_jax` to sync the array's device before the bridge.
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
    """Mixin for JAX-backed waveforms and computations: array bridging to and from JAX.

    Subclasses call ``JaxBase.__init__(self)`` (which only checks that jax
    imports) and use :meth:`_to_jax` / :meth:`_from_jax`. :meth:`_from_jax`
    reads the subclass's ``xp`` attribute (numpy when it has none) to pick
    the output array module.

    Raises:
        ImportError: jax is not available (at construction).

    Example:
        .. code-block:: python

            from lisatools.jax.jaxbase import JaxBase

            class MyJaxWaveform(JaxBase):
                xp = cupy                       # the consumer's array module

                def __init__(self):
                    JaxBase.__init__(self)

                def __call__(self, params):
                    p = self._to_jax(params)    # cupy -> jax via DLPack
                    out = my_jax_fn(p)
                    return self._from_jax(out)  # jax -> cupy
    """

    def __init__(self) -> None:
        """Check that jax is available (raises ``ImportError`` otherwise)."""
        if not jax_available:
            raise ImportError("JAX is required for JaxBase but is not available.")
        
    def _to_jax(self, 
                x: float | np.ndarray | cp.ndarray, 
                device_id: Optional[int] = None, 
                do_synchronize: bool = False) -> jnp.ndarray:
        """Convert input to a JAX array, optionally synchronizing if it's a CuPy array.

        Args:
            x: A cupy array (zero-copy through DLPack, on the array's own
                device), or a Python ``float`` / ``int`` / numpy array
                (``jax.device_put``).
            device_id: Index into ``jax.devices()`` of the default platform for
                host inputs; ``None`` (default) leaves placement to JAX. Ignored
                for cupy inputs.
            do_synchronize: Synchronize a cupy array's device first. Default
                ``False``.

        Returns:
            jnp.ndarray: The JAX array.

        Raises:
            TypeError: Any other input type (including a JAX array or a numpy
                integer scalar, which is not an ``int``).
        """
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

        Args:
            x: The JAX array.
            do_synchronize: ``block_until_ready`` first. Default ``False``.
            to_host: Force a numpy result. Default ``False``.

        Returns:
            np.ndarray or cupy.ndarray: numpy unless cupy is importable,
            ``to_host`` is ``False`` and ``self.xp`` is not numpy; then cupy
            (zero-copy through DLPack when possible, else through the host).
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
        