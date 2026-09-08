"""Shared infrastructure helpers for the Erebor stock family.

The small, deduplicated versions of the blocks every settings file used to
copy-paste: compute-backend detection, WDM-grid derivation, and TDI-channel
bookkeeping. All pure functions over plain values — nothing here touches
data files or GPUs beyond an import probe.
"""

from __future__ import annotations

import math
import typing

from lisatools.domains import WDMSettings
from lisatools.sensitivity import tdi_generation_from_channel


def cupy_available() -> bool:
    """True when cupy imports — the same probe the settings files used."""
    try:
        import cupy  # noqa: F401

        return True
    except (ModuleNotFoundError, ImportError):
        return False


# GPU_BACKEND values that mean "pick the CUDA wheel that matches this
# machine" rather than a deliberate, authoritative choice. An explicit
# ``"cudaXXx"`` is NEVER in this set, so a real ``force_backend`` is always
# honored verbatim (see :func:`resolve_gpu_backend`).
_AUTO_BACKEND_SENTINELS = frozenset({None, "", "auto", "cuda", "gpu"})


def resolve_gpu_backend(gpu_backend: typing.Optional[str]) -> str:
    """Resolve a GPU-backend request to a concrete installed wheel name.

    An **explicit** request (``"cuda11x"`` / ``"cuda12x"`` / ``"cuda13x"``)
    is returned unchanged — the caller's choice is authoritative and is
    never overridden. A deliberate ``force_backend`` therefore always wins,
    and a mismatched explicit choice fails loudly downstream instead of
    being silently swapped underneath the user.

    Only the *auto* sentinels (``None`` / ``""`` / ``"auto"`` / ``"cuda"`` /
    ``"gpu"``) trigger detection: the CUDA wheel whose major version matches
    the running cupy runtime, confirmed present via an ``importlib`` spec
    probe (no CUDA context is created here — the device is pinned later).
    """
    if gpu_backend not in _AUTO_BACKEND_SENTINELS:
        return gpu_backend

    import importlib.util

    order = ["cuda12x", "cuda11x", "cuda13x"]
    try:
        import cupy

        major = cupy.cuda.runtime.runtimeGetVersion() // 1000  # 12040 -> 12
        preferred = f"cuda{major}x"
        order = [preferred] + [name for name in order if name != preferred]
    except Exception:
        pass
    for name in order:
        if importlib.util.find_spec(f"lisatools_backend_{name}") is not None:
            return name
    # No plugin wheel found under the expected names — defer to the
    # lisatools registry alias, which loads the first CUDA backend that
    # actually imports; hard-fall to cuda12x only if even that fails.
    try:
        import lisatools

        return lisatools.get_backend("cuda").backend_base
    except Exception:
        return "cuda12x"


def resolve_compute(
    use_gpu: typing.Optional[bool],
    gpu_backend: str,
    gpus: typing.Optional[typing.Sequence[int]],
) -> typing.Tuple[typing.Optional[typing.List[int]], str]:
    """Resolve the (gpus, gpu_backend) pair the engine consumes.

    ``use_gpu=None`` auto-detects via :func:`cupy_available`. On the CPU
    path this returns ``(None, "cpu")`` — matching the legacy settings
    files' ``GPU_BACKEND = "cpu"`` fallback. That equality
    (``force_backend == gpu_backend``) is what makes the engine clone the
    data processor's orbits into ``gpu_orbits`` on CPU-only runs.

    On the GPU path ``gpu_backend`` is passed through
    :func:`resolve_gpu_backend`: an explicit ``"cudaXXx"`` is honored
    verbatim, while an auto sentinel (the default when ``GPU_BACKEND`` is
    unset) is resolved to the CUDA wheel matching this machine — so a
    non-propagated ``GPU_BACKEND`` env var can no longer silently select
    the wrong plugin wheel.
    """
    if use_gpu is None:
        use_gpu = cupy_available()
    if not use_gpu:
        return None, "cpu"
    gpus = list(gpus) if gpus is not None else [0]
    return gpus, resolve_gpu_backend(gpu_backend)


def derive_wdm_grid(
    tobs_target: float,
    dt: float,
    wavelet_duration_bounds: typing.Tuple[float, float],
) -> typing.Tuple[int, int, float, float]:
    """``(Nf, Nt, wavelet_duration, Tobs)`` via ``WDMSettings.adjust_to_even_bins``."""
    Nf, Nt, wavelet_duration = WDMSettings.adjust_to_even_bins(
        t_min=wavelet_duration_bounds[0],
        t_max=wavelet_duration_bounds[1],
        dt=dt,
        Tobs=tobs_target,
    )
    return Nf, Nt, wavelet_duration, Nf * Nt * dt


def default_edge_crop_wavelets(window_tukey_alpha: float, Nt: int) -> int:
    """WDM time-edge crop covering boundary wavelets AND the Tukey taper.

    The chunked-het Tukey/edge-leak rule: edge-cut >= alpha*Nt/2 (+ margin).
    """
    return max(20, int(math.ceil(window_tukey_alpha * Nt / 2)) + 4)


def tdi_generation_info(tdi_chan: str) -> typing.Tuple[int, str]:
    """``(tdi_gen, tdi_gen_str)`` derived from the TDI channel string."""
    tdi_gen = tdi_generation_from_channel(tdi_chan)
    return tdi_gen, f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"


def make_default_inner_moves(kind: str = "eigen") -> list:
    """Default single-source PE inner-move stack for an addremove branch.

    ``"eigen"`` — the eryn :class:`~eryn.moves.EigenAxisMove`
    (information-matrix eigen-axis jump; its per-leaf tables are fed by
    ``ResidualAddOneRemoveOneMove.refresh_inner_move_tables``).
    ``"stretch"`` — the legacy affine-invariant stretch.

    The single implementation behind every branch's ``inner_moves``
    fallback; the knob is the branch Settings field ``inner_move_kind``
    (env ``{BRANCH}_INNER_MOVE_KIND``).
    """
    kind_in = kind
    kind = str(kind).strip().lower()
    if kind == "eigen":
        from eryn.moves import EigenAxisMove

        return [(EigenAxisMove(mode="axis"), 1.0)]
    if kind == "stretch":
        from eryn.moves import StretchMove

        return [(StretchMove(), 1.0)]
    raise ValueError(
        f"unknown inner_move_kind {kind_in!r} (use 'eigen' or 'stretch')"
    )


def resolve_inner_moves(settings) -> list:
    """Fill ``settings.inner_moves`` from ``inner_move_kind`` when unset.

    An explicitly provided ``inner_moves`` list always wins; a missing
    ``inner_move_kind`` attribute defaults to ``"eigen"``.
    """
    if settings.inner_moves is None:
        settings.inner_moves = make_default_inner_moves(
            getattr(settings, "inner_move_kind", "eigen")
        )
    return settings.inner_moves
