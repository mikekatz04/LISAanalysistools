"""Device-pinning helpers: the canonical multi-GPU discipline for LAT.

The rule (see ``docs/conventions.md``): pin the run's *main* device once at
operation entry (:func:`pin_main_device`); every multi-shard structure
(``AnalysisContainerArray``, ``DomainComputationGroupArray``, sub-band
buffers) enters and restores its own per-shard device contexts
(:func:`device_context`); kernels always launch from inside the owning
context. CuPy's current device is thread-local, so per-split worker threads
must enter their shard's context themselves.

Everything here is a stateless function on purpose: device identity must
never live on ``Backend`` singletons (they re-resolve by registry name on
deepcopy/pickle) nor on pre-build settings objects.

All helpers are CPU-safe no-ops when ``xp`` is numpy (or any module without
a ``cuda`` attribute) so call sites need no ``uses_cupy`` guards.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence, Union

__all__ = [
    "device_context",
    "pin_main_device",
    "current_device",
    "jax_device_context",
]


def _first_gpu(gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    if gpus is None:
        return None
    if isinstance(gpus, int):
        return gpus
    return int(gpus[0])


def device_context(xp, device: Optional[int]):
    """Context manager placing work on ``device`` under ``xp``.

    Returns ``xp.cuda.Device(device)`` on the CuPy path and a
    ``nullcontext()`` when ``device`` is None or ``xp`` has no ``cuda``
    (numpy / CPU path).
    """
    if device is None or not hasattr(xp, "cuda"):
        return nullcontext()
    return xp.cuda.Device(int(device))


def pin_main_device(xp, gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    """Pin the process-current device to the run's main GPU (``gpus[0]``).

    No-op on CPU (``gpus is None`` or numpy ``xp``). Returns the previous
    device id (None when nothing was pinned) so callers that need to restore
    can ``xp.cuda.runtime.setDevice(prev)`` afterwards.
    """
    first = _first_gpu(gpus)
    if first is None or not hasattr(xp, "cuda"):
        return None
    prev = int(xp.cuda.runtime.getDevice())
    xp.cuda.runtime.setDevice(first)
    return prev


def current_device(xp) -> Optional[int]:
    """The process-current device id under ``xp`` (None on the CPU path)."""
    if not hasattr(xp, "cuda"):
        return None
    return int(xp.cuda.runtime.getDevice())


def jax_device_context(device: Optional[int], *, kind: str = "gpu"):
    """Pin JAX's default device to match cupy ``device`` (JAX-backed gens).

    CuPy's :func:`device_context` does NOT move JAX ops: JAX placement is
    governed by ``jax.default_device`` / ``jax.device_put``. So any
    per-device JAX generation (the phentax MBH waveform) must ALSO enter this
    context, otherwise scalar ``device_put(x, device=None)`` inputs and the
    traced kernels land on JAX's *default* device (gpu0) regardless of the
    surrounding cupy :func:`device_context`. This is the JAX twin of
    :func:`device_context`; the two are entered together at each per-shard
    source-generation site.

    Assumes JAX and cupy enumerate CUDA devices in the same order (the DLPack
    precondition already documented in ``lisatools/jax/jaxbase.py``): the
    cupy device id indexes ``jax.devices(kind)`` directly.

    Returns ``nullcontext()`` when ``device`` is None (CPU / single-device /
    the run's primary device), when JAX is unavailable, or when JAX cannot
    see a device at ``jax.devices(kind)[device]`` -- so call sites need no
    guards and single-GPU/CPU behaviour is unchanged.
    """
    if device is None:
        return nullcontext()
    try:
        import jax
    except (ImportError, ModuleNotFoundError):
        return nullcontext()
    try:
        return jax.default_device(jax.devices(kind)[int(device)])
    except (RuntimeError, IndexError):
        # JAX on CPU while cupy on GPU, or fewer JAX devices than cupy sees.
        return nullcontext()
