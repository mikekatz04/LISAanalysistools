"""Shared infrastructure helpers for the Erebor stock family.

The small, deduplicated versions of the blocks every settings file used to
copy-paste: compute-backend detection, WDM-grid derivation, and TDI-channel
bookkeeping, plus the default per-leaf tempering ladder and inner-move stack
of the single-source (MBH / EMRI / SOBBH) branches. All pure functions over
plain values — nothing here touches data files or GPUs beyond an import
probe.

Pipeline position: :meth:`EreborFit.make_general_settings
<lisatools.globalfit.stock.erebor.fit.EreborFit.make_general_settings>` calls
:func:`resolve_compute`, :func:`derive_wdm_grid` and
:func:`default_edge_crop_wavelets` at build; the MBH / EMRI / SOBBH Setups
and branch preps call :func:`default_source_ladder` and
:func:`resolve_inner_moves`. Knobs: ``USE_GPU``, ``GPU_BACKEND``,
``TOBS_TARGET``, ``EDGE_CROP_WAVELETS`` and the per-branch
``{BRANCH}_NTEMPS`` / ``{BRANCH}_LADDER_TMAX`` / ``{BRANCH}_INNER_MOVE_KIND``
/ ``{BRANCH}_EIGEN_JUMP_MIX`` / ``{BRANCH}_DE_JUMP_MIX`` fields.
"""

from __future__ import annotations

import math
import typing

from lisatools.domains import WDMSettings
from lisatools.sensitivity import tdi_generation_from_channel


def cupy_available() -> bool:
    """True when cupy imports — the same probe the settings files used.

    Used for ``use_gpu=None`` (auto) in :func:`resolve_compute`; the launchers
    set ``USE_GPU=1``, so it is not consulted in production.
    """
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
    The probe order is the cupy-matching wheel, then ``cuda12x``,
    ``cuda11x``, ``cuda13x``; without any installed plugin wheel it defers
    to ``lisatools.get_backend("cuda")`` and finally returns ``"cuda12x"``.

    Args:
        gpu_backend: The request (``GPU_BACKEND`` / ``general.gpu_backend``).

    Returns:
        str: A backend name such as ``"cuda13x"``.
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

    Args:
        use_gpu: ``True`` / ``False``, or ``None`` = auto-detect
            (``USE_GPU``; launchers 1).
        gpu_backend: Backend request (``GPU_BACKEND``; launchers ``cuda13x``).
        gpus: Device indices (``GPUS``); ``None`` -> ``[0]`` on the GPU path.

    Returns:
        tuple[list[int] | None, str]: ``(gpus, gpu_backend)``; ``(None, "cpu")``
        on the CPU path.
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
    """``(Nf, Nt, wavelet_duration, Tobs)`` via ``WDMSettings.adjust_to_even_bins``.

    Args:
        tobs_target: Target span (s); ``TOBS_TARGET``.
        dt: Sample spacing (s).
        wavelet_duration_bounds: ``(t_min, t_max)`` (s) allowed for the
            wavelet duration ``Nf * dt``.

    Returns:
        tuple[int, int, float, float]: ``Nf`` (layers), ``Nt`` (time columns),
        the wavelet duration (s) and the exact ``Tobs = Nf * Nt * dt`` (s).
    """
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
    Used only when ``edge_crop_wavelets`` is ``None`` (the launchers pin
    ``EDGE_CROP_WAVELETS=60``).

    Args:
        window_tukey_alpha: Data-window Tukey alpha (fraction of ``Tobs``).
        Nt: WDM time columns.

    Returns:
        int: ``max(20, ceil(alpha * Nt / 2) + 4)`` columns per side.

    Example:
        >>> from lisatools.globalfit.stock.erebor.common import default_edge_crop_wavelets
        >>> default_edge_crop_wavelets(0.05, 2160)
        58
        >>> default_edge_crop_wavelets(0.0, 2160)
        20
    """
    return max(20, int(math.ceil(window_tukey_alpha * Nt / 2)) + 4)


def tdi_generation_info(tdi_chan: str) -> typing.Tuple[int, str]:
    """``(tdi_gen, tdi_gen_str)`` derived from the TDI channel string.

    Args:
        tdi_chan: Channel string (``"XYZ"``, ...), parsed by
            :func:`~lisatools.sensitivity.tdi_generation_from_channel`.

    Returns:
        tuple[int, str]: e.g. ``(2, "2nd generation")``; generation 1 gives
        ``"1st generation"``.
    """
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

    With ``{BRANCH}_LADDER_TOP_SNR`` > 0 (ruling 2026-10-18, default off)
    this uniform ladder is only the fallback of a leaf whose SNR is not yet
    known: each leaf gets ``make_ladder(ndim, ntemps, Tmax=clip((SNR /
    top)^2, {BRANCH}_LADDER_TMAX_MIN, tmax))``, ``tmax`` its CEILING
    (``recipe.resolve_snr_keyed_ladders``, :mod:`lisatools.sampling.snrladder`).

    Production launchers: ``MBH_NTEMPS=8`` / ``MBH_LADDER_TMAX=10``,
    ``EMRI_NTEMPS=12`` / ``EMRI_LADDER_TMAX=625``, ``SOBBH_NTEMPS=8`` /
    ``SOBBH_LADDER_TMAX=6``; ``*_LADDER_TOP_SNR`` unset (off).

    Args:
        ndim (int | None): Sampling dimension; ``None`` -> ``len(input_basis)``.
        input_basis (sequence): The branch's sampling-basis names.
        ntemps (int): Number of rungs.
        tmax (float, optional): Top temperature (must be > 1). Default
            ``None`` = eryn's uncapped ~25 % spacing.

    Returns:
        np.ndarray: ``(ntemps,)`` inverse temperatures ``betas``, ``betas[0] == 1``.

    Raises:
        ValueError: When ``tmax`` is set and not > 1.
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
    kind: str = "eigen", jump_mix: float = 0.0, jump_factor: float = 3.0,
    de_jump_mix: float = 0.0, de_jump_mode_frac: float = 0.5,
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

    ``de_jump_mix > 0`` (eigen only; user ruling 2026-10-18, "we do not have
    enough walkers for the stretch move, so we need to improve what we
    have") adds the cross-walker difference jump
    :class:`~lisatools.sampling.moves.walkerdiffjump.WalkerDifferenceJumpMove`
    with that stack weight (the eigen move keeps ``1 - de_jump_mix``), its
    fallback the eigen move itself, ``de_jump_mode_frac`` its share of
    ``gamma = 1`` mode jumps. ``de_jump_mix == 0`` leaves the stack as above
    (bit-identical).

    The single implementation behind every branch's ``inner_moves``
    fallback; the knobs are the branch Settings fields ``inner_move_kind``,
    ``eigen_jump_mix``, ``eigen_jump_factor``, ``de_jump_mix`` and
    ``de_jump_mode_frac`` (env ``{BRANCH}_INNER_MOVE_KIND``,
    ``{BRANCH}_EIGEN_JUMP_MIX``, ``{BRANCH}_EIGEN_JUMP_FACTOR``,
    ``{BRANCH}_DE_JUMP_MIX``, ``{BRANCH}_DE_JUMP_MODE_FRAC``).

    Production launchers: ``*_INNER_MOVE_KIND=eigen``,
    ``*_EIGEN_JUMP_MIX=0.1`` / ``*_EIGEN_JUMP_FACTOR=3.0``, and
    ``EMRI_DE_JUMP_MIX=0.2`` (``EMRI_DE_JUMP_MODE_FRAC=0.5``),
    ``MBH_DE_JUMP_MIX=0.1``, ``SOBBH_DE_JUMP_MIX=0.2``.

    Args:
        kind: ``"eigen"`` or ``"stretch"`` (case-insensitive).
        jump_mix: Fraction of eigen draws taken at ``jump_factor`` x the step.
        jump_factor: Step multiplier of those draws.
        de_jump_mix: Stack weight of the cross-walker difference jump, in
            ``[0, 1)``.
        de_jump_mode_frac: Share of ``gamma = 1`` mode jumps in that move.

    Returns:
        list: ``[(move, weight), ...]`` eryn inner-move stack.

    Raises:
        ValueError: Unknown ``kind``, or ``de_jump_mix`` outside ``[0, 1)``.
    """
    kind_in = kind
    kind = str(kind).strip().lower()
    if kind == "eigen":
        jump_mix = float(jump_mix or 0.0)
        if jump_mix > 0.0:
            from lisatools.sampling.moves.eigenjumpmix import EigenAxisJumpMixMove

            eigen = EigenAxisJumpMixMove(
                mode="axis", jump_mix=jump_mix, jump_mix_factor=float(jump_factor)
            )
        else:
            from eryn.moves import EigenAxisMove

            eigen = EigenAxisMove(mode="axis")
        de = float(de_jump_mix or 0.0)
        if not (0.0 <= de < 1.0):
            raise ValueError(f"de_jump_mix must be in [0, 1), got {de_jump_mix!r}")
        if de > 0.0:
            from lisatools.sampling.moves.walkerdiffjump import WalkerDifferenceJumpMove

            return [
                (eigen, 1.0 - de),
                (WalkerDifferenceJumpMove(eigen, mode_frac=float(de_jump_mode_frac)), de),
            ]
        return [(eigen, 1.0)]
    if kind == "stretch":
        from eryn.moves import StretchMove

        return [(StretchMove(), 1.0)]
    raise ValueError(
        f"unknown inner_move_kind {kind_in!r} (use 'eigen' or 'stretch')"
    )


def inner_move_kwargs(settings) -> dict:
    """The :func:`make_default_inner_moves` arguments a branch block carries.

    Args:
        settings (Settings): A branch Settings / Setup; missing fields fall back to
            ``"eigen"``, 0.0, 3.0, 0.0, 0.5.

    Returns:
        dict: ``kind``, ``jump_mix``, ``jump_factor``, ``de_jump_mix``,
        ``de_jump_mode_frac``.
    """
    return dict(
        kind=getattr(settings, "inner_move_kind", "eigen"),
        jump_mix=float(getattr(settings, "eigen_jump_mix", 0.0) or 0.0),
        jump_factor=float(getattr(settings, "eigen_jump_factor", 3.0)),
        de_jump_mix=float(getattr(settings, "de_jump_mix", 0.0) or 0.0),
        de_jump_mode_frac=float(getattr(settings, "de_jump_mode_frac", 0.5)),
    )


def resolve_inner_moves(settings) -> list:
    """Fill ``settings.inner_moves`` from ``inner_move_kind`` when unset.

    An explicitly provided ``inner_moves`` list always wins; a missing
    ``inner_move_kind`` attribute defaults to ``"eigen"``, missing
    ``eigen_jump_mix`` / ``eigen_jump_factor`` to 0 / 3.

    Args:
        settings (Settings): Branch Settings / Setup with an ``inner_moves`` attribute
            (mutated in place).

    Returns:
        list: ``settings.inner_moves``.
    """
    if settings.inner_moves is None:
        settings.inner_moves = make_default_inner_moves(**inner_move_kwargs(settings))
    return settings.inner_moves
