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


def default_source_ladder(ndim, input_basis, ntemps, tmax=None):
    """The default per-leaf ladder of an MBH / EMRI / SOBBH branch.

    ``eryn.moves.tempering.make_ladder(ndim, ntemps=ntemps)`` with ``Tmax``
    unset: the geometric spacing that gives a ~25 % swap acceptance between
    neighbouring rungs of an ``ndim``-dimensional Gaussian (user ruling
    2026-10-13: "the default ladders for non-RJ sources should be from
    make_ladder. not from the 1/1.2^k setup"). ``ndim`` is the branch's
    sampling dimension (the settings field: EMRI 12, MBH 11, SOBBH 11); when
    it is unset the length of the branch's sampling ``input_basis`` stands in.
    ``make_ladder`` insists on plain ``int`` arguments, hence the casts.

    ``tmax`` (the settings field ``ladder_tmax``, env ``{BRANCH}_LADDER_TMAX``;
    ``None`` = unset) caps the ladder: ``make_ladder(ndim, ntemps, Tmax=tmax)``
    is the unique geometric spacing from 1 to ``tmax`` over ``ntemps`` rungs.
    Why it exists: with 8 rungs the uncapped 25 % spacing reaches T ~ 120
    (EMRI, ndim 12) / ~150 (MBH, SOBBH, ndim 11), and at those temperatures a
    loud source (SNR ~50, likelihood peak ~1,200 nats) is a ~10-nat bump
    over the prior volume -- the top rungs cannot hold it, the lost-rung
    guard then re-anchors them every visit, and the shared eigen tables
    (built at the cold walker) step ~sqrt(T) too narrow there. A cap of
    ~10 keeps every rung on the source (hot-rung width ~3 sigma, enough to
    bridge sigma-spaced islands) at the price of a lower per-rung swap
    acceptance. Left unset by default so the ruling's ladder is the default.
    """
    from eryn.moves.tempering import make_ladder

    nd = int(ndim) if ndim is not None else int(len(input_basis))
    if tmax is None:
        return make_ladder(nd, ntemps=int(ntemps))
    tmax = float(tmax)
    if not tmax > 1.0:
        raise ValueError(f"ladder_tmax must be > 1 (a temperature), got {tmax!r}")
    return make_ladder(nd, ntemps=int(ntemps), Tmax=tmax)


def make_default_inner_moves(
    kind: str = "eigen", jump_mix: float = 0.0, jump_factor: float = 3.0
) -> list:
    """Default single-source PE inner-move stack for an addremove branch.

    ``"eigen"`` — the eryn :class:`~eryn.moves.EigenAxisMove`
    (information-matrix eigen-axis jump; its per-leaf tables are fed by
    ``ResidualAddOneRemoveOneMove.refresh_inner_move_tables``). With
    ``jump_mix > 0`` it is the
    :class:`~lisatools.sampling.moves.eigenjumpmix.EigenAxisJumpMixMove`
    instead: that fraction of the draws takes ``jump_factor`` x the table's
    step (user ruling 2026-10-13). ``jump_mix == 0`` builds the plain eryn
    move, so every draw is bit-identical to the pre-knob stack.
    ``"stretch"`` — the legacy affine-invariant stretch (the jump knobs do
    not apply).

    The single implementation behind every branch's ``inner_moves``
    fallback; the knobs are the branch Settings fields ``inner_move_kind``,
    ``eigen_jump_mix`` and ``eigen_jump_factor`` (env
    ``{BRANCH}_INNER_MOVE_KIND``, ``{BRANCH}_EIGEN_JUMP_MIX``,
    ``{BRANCH}_EIGEN_JUMP_FACTOR``).
    """
    kind_in = kind
    kind = str(kind).strip().lower()
    if kind == "eigen":
        jump_mix = float(jump_mix or 0.0)
        if jump_mix > 0.0:
            from lisatools.sampling.moves.eigenjumpmix import EigenAxisJumpMixMove

            return [
                (
                    EigenAxisJumpMixMove(
                        mode="axis", jump_mix=jump_mix, jump_mix_factor=float(jump_factor)
                    ),
                    1.0,
                )
            ]
        from eryn.moves import EigenAxisMove

        return [(EigenAxisMove(mode="axis"), 1.0)]
    if kind == "stretch":
        from eryn.moves import StretchMove

        return [(StretchMove(), 1.0)]
    raise ValueError(
        f"unknown inner_move_kind {kind_in!r} (use 'eigen' or 'stretch')"
    )


def inner_move_kwargs(settings) -> dict:
    """The :func:`make_default_inner_moves` arguments a branch block carries."""
    return dict(
        kind=getattr(settings, "inner_move_kind", "eigen"),
        jump_mix=float(getattr(settings, "eigen_jump_mix", 0.0) or 0.0),
        jump_factor=float(getattr(settings, "eigen_jump_factor", 3.0)),
    )


def resolve_inner_moves(settings) -> list:
    """Fill ``settings.inner_moves`` from ``inner_move_kind`` when unset.

    An explicitly provided ``inner_moves`` list always wins; a missing
    ``inner_move_kind`` attribute defaults to ``"eigen"``, missing
    ``eigen_jump_mix`` / ``eigen_jump_factor`` to 0 / 3.
    """
    if settings.inner_moves is None:
        settings.inner_moves = make_default_inner_moves(**inner_move_kwargs(settings))
    return settings.inner_moves
