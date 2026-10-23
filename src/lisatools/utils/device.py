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
    "order_after_array",
    "pin_main_device",
    "current_device",
    "to_current_device",
    "sensitivity_to_current_device",
    "to_current_device_cached",
    "synchronize",
    "jax_device_context",
]


def _first_gpu(gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    """Return the main device id: ``gpus`` itself for an int, ``gpus[0]`` for a sequence."""
    if gpus is None:
        return None
    if isinstance(gpus, int):
        return gpus
    return int(gpus[0])


def device_context(xp, device: Optional[int]):
    """Context manager placing work on ``device`` under ``xp``.

    Entering it makes ``device`` CuPy's current device for this thread; leaving restores the
    previous one.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        device: CUDA device id, or ``None``.

    Returns:
        contextlib.AbstractContextManager: ``xp.cuda.Device(device)`` on the CuPy path, a
        ``nullcontext()`` when ``device`` is None or ``xp`` has no ``cuda`` (numpy / CPU path).

    Example:
        >>> import numpy as np
        >>> from lisatools.utils.device import device_context
        >>> with device_context(np, 1):
        ...     x = np.zeros(3)
    """
    if device is None or not hasattr(xp, "cuda"):
        return nullcontext()
    return xp.cuda.Device(int(device))


def order_after_array(xp, arr) -> None:
    """Order the CURRENT device's stream behind the device that owns ``arr``.

    THE CROSS-DEVICE COPY EDGE (2026-09-16). A direct peer copy --
    ``xp.asarray(foreign)`` issued while another device is current, the
    shape every :class:`~lisatools.analysiscontainer.BandView` per-shard
    loop and every routed worker's input move desugars to -- reads memory
    a kernel on the SOURCE device may still be writing. CUDA orders
    nothing between two devices' streams, so the copy can read stale
    bytes: silent, deterministic (fixed kernel sequences replay
    identically) and invisible to any single-GPU test.

    ``beaa60f5`` / ``e3dde390`` closed the dispatcher half of this --
    shard worker -> the caller's current stream, on the caller's CURRENT
    device. This closes the rest: it records an event on the source
    device's current stream and makes the current device's stream wait on
    it. Same thread throughout, so the source stream really is the one
    that enqueued the producing kernel (cupy's current stream is
    thread-local, and per-thread-default-stream builds are covered for
    the same reason).

    A stream edge, not a fence: nothing drains and both devices keep
    running. No-op when ``arr`` is already on the current device, when it
    is a host array, and for NumPy / any ``xp`` without a CuPy-like
    ``cuda.get_current_stream``.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        arr (array): The array about to be read from another device.
    """
    cuda = getattr(xp, "cuda", None)
    get_stream = getattr(cuda, "get_current_stream", None)
    if get_stream is None:
        return
    src = getattr(getattr(arr, "device", None), "id", None)
    if src is None:
        return  # host array, or a backend without device tags
    cur = current_device(xp)
    if cur is None or int(src) == int(cur):
        return  # same device == same stream in this thread: already ordered
    with device_context(xp, int(src)):
        event = get_stream().record()
    get_stream().wait_event(event)


def pin_main_device(xp, gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    """Pin the process-current device to the run's main GPU (``gpus[0]``).

    No-op on CPU (``gpus is None`` or numpy ``xp``). Unlike :func:`device_context` this is
    not scoped: the device stays current until something else sets it.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        gpus: The run's GPU list (an int is treated as a one-element list), or ``None``.

    Returns:
        Optional[int]: The previous device id (None when nothing was pinned), so callers that
        need to restore can ``xp.cuda.runtime.setDevice(prev)`` afterwards.
    """
    first = _first_gpu(gpus)
    if first is None or not hasattr(xp, "cuda"):
        return None
    prev = int(xp.cuda.runtime.getDevice())
    xp.cuda.runtime.setDevice(first)
    return prev


def current_device(xp) -> Optional[int]:
    """Return the current device id under ``xp`` (None on the CPU path).

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).

    Returns:
        Optional[int]: ``xp.cuda.runtime.getDevice()``, or ``None`` when ``xp`` has no
        ``cuda``.
    """
    if not hasattr(xp, "cuda"):
        return None
    return int(xp.cuda.runtime.getDevice())


def to_current_device(xp, arr):
    """Return ``arr`` resident on the process-current device under ``xp``.

    On the CuPy path, if ``arr`` lives on a DIFFERENT GPU than the current
    device -- the multi-GPU per-shard build, where a shared settings object's
    cached arrays live on its home device while a walker builds inside its own
    owning device's context (run.setup_acs) -- copy ``arr`` to the current
    device VIA HOST so the move needs no peer access (P2P is unavailable
    between GPUs on some nodes; a same-device ufunc against it then raises
    "device where the array resides differs from the current device").

    No-op for NumPy/CPU and for arrays already on the current device. A host array under a
    CuPy ``xp`` is uploaded with ``xp.asarray``.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        arr (array): Array to place.

    Returns:
        array: ``arr`` itself, or a copy on the current device.
    """
    if not hasattr(xp, "cuda"):
        return arr
    dev = getattr(arr, "device", None)
    if dev is None:
        return arr
    dev_id = getattr(dev, "id", None)
    if dev_id is None:
        # A HOST array under a CuPy ``xp`` (numpy >= 2 exposes
        # ``ndarray.device == "cpu"``, a plain string): upload it to the
        # current device -- that IS "resident on the current device".
        return xp.asarray(arr)
    if int(dev_id) == int(xp.cuda.runtime.getDevice()):
        return arr
    return xp.asarray(xp.asnumpy(arr))


def to_current_device_cached(xp, owner, tag, arrays):
    """``arrays`` (a tuple) resident on the current device, the copies cached
    on ``owner`` per ``(tag, device)``.

    For arrays a shared object caches on its HOME device and every GPU of a
    multi-device run reads (the WDM settings' window and fold map): the
    first read on another device pays one :func:`to_current_device` host
    hop, later reads reuse it for as long as ``owner`` hands out the SAME
    source objects (an ``is`` check, so a rebuilt source re-copies). The
    cache lives in ``owner._device_copy_cache``; an owner that is pickled
    or deep-copied must drop that key in ``__getstate__``. A no-op on CPU.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        owner (object): Object whose ``__dict__`` holds the cache.
        tag (hashable): Cache key for this group of arrays.
        arrays (Iterable): The source arrays.

    Returns:
        tuple: The arrays on the current device, in input order.
    """
    arrays = tuple(arrays)
    if not hasattr(xp, "cuda"):
        return arrays
    dev = int(xp.cuda.runtime.getDevice())
    cache = owner.__dict__.setdefault("_device_copy_cache", {})
    hit = cache.get((tag, dev))
    if hit is not None and len(hit[0]) == len(arrays) and all(
            a is b for a, b in zip(hit[0], arrays)):
        return hit[1]
    moved = tuple(to_current_device(xp, a) for a in arrays)
    cache[(tag, dev)] = (arrays, moved)
    return moved


def sensitivity_to_current_device(xp, sm):
    """``sm`` itself, or a shallow clone (same class) whose ``invC`` / ``detC``
    are on the current device, copied once via the host, when either lives
    on another GPU.

    For an inner product whose signals were built on one device against a
    walker container on another (2026-10-09: the SOBBH Gram eigen table,
    templates on the comp's device, PSD on the walker's). ``inner_product``
    reads only ``invC`` as an array (``detC`` too in a likelihood), so the
    decision is made on THOSE arrays: a production walker matrix can hold
    ``sens_mat`` on the settings' home device while its ``invC`` / ``detC``
    are refreshed on the walker's device (the first version keyed on
    ``sens_mat`` and let exactly that case through, 6mo job 748). The clone
    shares every other attribute, ``sens_mat`` included (only its shape is
    read). A dirty (lazily inverted) matrix is inverted on its ``sens_mat``'s
    device first. The clone is built with ``object.__new__`` + the instance
    dict, not ``copy.copy``, so no ``__getstate__`` / ``__deepcopy__`` hook
    runs. A no-op on CPU and when both arrays are already local.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``).
        sm (SensitivityMatrix): The sensitivity matrix
            (:class:`~lisatools.sensitivity.SensitivityMatrix` or a subclass).

    Returns:
        SensitivityMatrix: ``sm`` itself, or the shallow clone with local ``invC`` /
        ``detC``.
    """
    if not hasattr(xp, "cuda"):
        return sm

    def _dev(a):
        return getattr(getattr(a, "device", None), "id", None)

    cur = int(xp.cuda.runtime.getDevice())
    if getattr(sm, "_inv_det_dirty", False) and getattr(sm, "do_inv_det", True):
        home = _dev(getattr(sm, "sens_mat", None))
        with device_context(xp, home):
            invC, detC = sm.invC, sm.detC
    else:
        invC, detC = sm.invC, sm.detC          # plain reads, nothing computed
    if all(_dev(a) is None or int(_dev(a)) == cur for a in (invC, detC)):
        return sm
    out = object.__new__(type(sm))
    out.__dict__.update(sm.__dict__)
    out.invC = to_current_device(xp, invC)   # setters clear the dirty flag
    out.detC = to_current_device(xp, detC)
    return out


def synchronize(xp) -> None:
    """Block until the current device has finished its queued work.

    FOR TIMING ATTRIBUTION ONLY. CUDA kernel launches return immediately,
    so an un-synced ``launch`` span reads as microseconds and the whole
    wall silently lands on whatever device-to-host pull blocks next --
    which is exactly how job 508's telemetry ended up reporting "kernel =
    100.0%" with no way to say WHICH part of the call that was. Placing an
    explicit sync at a span boundary that is immediately followed by a
    blocking D2H costs nothing: the wait happens either way, this just
    labels it.

    No-op for NumPy / any ``xp`` without a ``cuda`` attribute, and for
    ``xp is None``.

    Args:
        xp (module): Array module (``numpy`` or ``cupy``), or ``None``.
    """
    cuda = getattr(xp, "cuda", None)
    if cuda is None:
        return
    cuda.runtime.deviceSynchronize()


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

    Args:
        device: CuPy device id, or ``None``.
        kind: JAX platform passed to ``jax.devices``. Default ``"gpu"``.

    Returns:
        contextlib.AbstractContextManager: ``jax.default_device(...)`` or ``nullcontext()``.
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
