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


def resolve_compute(
    use_gpu: typing.Optional[bool],
    gpu_backend: str,
    gpus: typing.Optional[typing.Sequence[int]],
) -> typing.Tuple[typing.Optional[typing.List[int]], str]:
    """Resolve the (gpus, gpu_backend) pair the engine consumes.

    ``use_gpu=None`` auto-detects via :func:`cupy_available`. The engine
    derives ``force_backend = gpu_backend if gpus is not None else "cpu"``,
    so returning ``gpus=None`` selects the CPU path regardless of
    ``gpu_backend``.
    """
    if use_gpu is None:
        use_gpu = cupy_available()
    if not use_gpu:
        return None, gpu_backend
    return list(gpus) if gpus is not None else [0], gpu_backend


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
