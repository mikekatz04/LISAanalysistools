"""GB (galactic-binary) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
import warnings
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.domains import DomainSettingsBase, FDSettings, WDMSettings
from lisatools.utils.constants import YRSID_SI

from ..base import env_default
from ...engine import Settings, Setup
from ...hdfbackend import GBHDFBackend
from ...loginfo import init_logger
from ...priors.gbpriors import get_fdot_mojito
from ...state import GBState
from .transforms import make_gb_transform_container


def get_n_based_band_edges(
    start_freq: float,
    end_freq: float,
    Tobs: float,
    layer_df: float,
    *,
    subband_divisor: int = 1,
    oversample: int = 4,
    extra_buffer: int = 5,
    target_count: int = 0,
    min_band_layers: typing.Optional[float] = None,
    unit_stride: int = 2,
    sep_factor: typing.Optional[float] = None,
    amp: float = 1e-30,
) -> np.ndarray:
    """FREE-FLOATING GB band edges at the ``2 * get_N`` minimum width.

    THE WIDTH RULE (user ruling, 2026-08-15): each sub-band's width is
    the MINIMUM the chunked-heterodyne bookkeeping allows,

        ``w_band = 2 * get_N(f_max_band) / Tobs``  (Hz),

    where :func:`gbgpu.utils.utility.get_N` is the installed production
    function that sizes the initial frequency-domain heterodyned
    waveform of a single GB chunked-het source (the same call the
    FD/chunked-het paths use for per-source windows -- reused, never
    reimplemented), evaluated at the maximum frequency IN the band. The
    builder MAXIMIZES the number of sub-bands: it packs minimum-width
    bands left-to-right across ``[start_freq, end_freq]`` and never
    widens a band beyond its own floor, with exactly two documented
    exceptions:

    * the FINAL band is clamped to ``end_freq`` and absorbs the
      remainder (merged into its neighbor when the remainder is below
      the top band's floor), and
    * the fixed-point resolution of the self-consistent width (below).

    Self-consistency: the width depends on the band's own upper edge,
    ``w = 2 * get_N(f_lo + w) / Tobs``. ``get_N`` is monotone
    non-decreasing in ``f`` and returns discrete values, so iterating
    ``w_{n+1} = 2 * get_N(f_lo + w_n) / Tobs`` from
    ``w_0 = 2 * get_N(f_lo) / Tobs`` is a monotone non-decreasing
    sequence over a finite value set: it converges EXACTLY to the least
    fixed point in a few steps (typically 1-2; hard-capped with a loud
    RuntimeError -- never a silent approximation).

    FREE-FLOATING EDGES (2026-08-15 user ruling): edges are NOT snapped
    to WDM layer (or fractional-layer) boundaries -- the frequency-based
    parallel scheduling is fully independent of the WDM pixelization.
    The outermost edges are ``start_freq`` / ``end_freq`` verbatim
    (callers clamp those to the WDM active band first). Only the STORAGE
    slabs are layer-derived (see ``SubBandBuffer.recommend_band_slab_layers``
    / ``_compute_slab_min_f``: whole-layer origins/spans covering each
    band's touched range plus max(leakage, FD-support) margins).
    ``layer_df`` is used here only for the summary log's
    layers-equivalent numbers.

    LEAKAGE / SEPARATION GUARD ("strong checks" ruling): a source at a
    band's top edge has FD support reaching ``get_N(f_hi)/Tobs`` Hz past
    the edge, so same-unit concurrency is guarded in SUPPORT terms by
    :func:`lisatools.globalfit.moves.gbbands.check_band_support_separation`
    (gap between same-unit bands >= ``GB_ORTHO_SEP_FACTOR`` x the sum of
    edge half-supports; the width rule guarantees stride 2 passes at
    factor 1.0, stride 3 adds a full band of clearance -- see that
    docstring for the derivation). The builder runs it on the final
    edges and raises loudly on failure, then logs a one-line summary
    (band count, width stats in FD bins and layer-widths, worst
    overhang, minimum safe stride).

    Knob semantics in this mode:

    * ``target_count`` (``GB_BAND_TARGET_COUNT``) > 0 -- optional
      COARSENING: adjacent minimum-width bands are merged (evenly
      spread) down to ~``target_count`` bands. Merged bands are wider
      than each member's floor, so the width rule's guarantees are
      preserved. The count saturates at the maximal packing (a target
      above it is a no-op); the default 0 IS the maximal packing.
    * ``min_band_layers`` (``GB_BAND_MIN_LAYERS``) has NO role here (the
      2*get_N rule is the authority) -- warned and ignored when set;
      it belongs to the legacy uniform mode's era only.
    * ``subband_divisor`` (``GB_SUBBAND_DIVISOR``) -- warned and
      ignored; it only defines the snap grid of the legacy ``uniform``
      mode.
    * ``extra_buffer`` -- unused by this rule (the legacy FD walk keeps
      it); accepted for signature compatibility.

    Returns:
        np.ndarray: ascending band edges (Hz), ``len(edges) - 1`` bands.
    """
    if int(subband_divisor) not in (0, 1):
        warnings.warn(
            "subband_divisor is ignored by the get_n band construction "
            "(free-floating edges, 2026-08-15 user ruling: scheduling is "
            "independent of the WDM pixelization); it only defines the "
            "snap grid of the legacy 'uniform' mode.",
            DeprecationWarning,
            stacklevel=2,
        )
    if min_band_layers is not None and float(min_band_layers) != 1.0:
        warnings.warn(
            "min_band_layers has no role in the get_n band construction "
            "(2026-08-15 user ruling: band widths are set by "
            "2*get_N(f_max_band)/Tobs, the minimum the chunked-het "
            "bookkeeping allows); the value is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    start_freq = float(start_freq)
    end_freq = float(end_freq)
    if end_freq <= start_freq:
        raise ValueError(
            f"empty GB frequency range [{start_freq:.4e}, {end_freq:.4e}]."
        )
    df = 1.0 / float(Tobs)

    def _n_at(f: float) -> float:
        return float(get_N(amp, f, Tobs, oversample=oversample).item())

    def _min_width(f_lo: float) -> float:
        # Least fixed point of w = 2 * get_N(f_lo + w) * df, iterated
        # from below (monotone: get_N non-decreasing => the sequence is
        # non-decreasing over get_N's finite value set => exact
        # convergence). Cap + loud failure rather than silent drift.
        w = 2.0 * _n_at(f_lo) * df
        for _ in range(32):
            w_next = 2.0 * _n_at(f_lo + w) * df
            if w_next == w:
                return w
            if w_next < w:  # get_N must be non-decreasing; never trust silently
                raise RuntimeError(
                    f"get_N not monotone at f={f_lo:.6e} Hz "
                    f"(w {w:.6e} -> {w_next:.6e}); cannot size bands."
                )
            w = w_next
        raise RuntimeError(
            f"band-width fixed point did not converge at f={f_lo:.6e} Hz "
            f"(last w={w:.6e} Hz); get_N is expected to be a step "
            "function with a handful of levels across one band."
        )

    # Maximal packing at the floor: every band exactly its own minimum
    # width, left-to-right.
    fs = [start_freq]
    f = start_freq
    while f < end_freq:
        f = min(f + _min_width(f), end_freq)
        fs.append(f)
    # The final band absorbs the remainder at the top edge: if the
    # clamped last band is below ITS OWN floor (2*get_N(end_freq)*df),
    # merge it into its neighbor (the one documented widening).
    if len(fs) >= 3 and fs[-1] - fs[-2] < 2.0 * _n_at(end_freq) * df:
        del fs[-2]
    edges = np.asarray(fs, dtype=float)

    if int(target_count) > 0 and int(target_count) < len(edges) - 1:
        # Optional coarsening: MERGE adjacent minimum-width bands, evenly
        # spread, down to ~target_count. Keeps the outer edges and the
        # relative get_N profile; merged widths exceed each member's
        # floor, so every guarantee of the width rule still holds.
        n = len(edges) - 1
        keep = np.unique(np.round(
            np.linspace(0, n, int(target_count) + 1)
        ).astype(int))
        edges = edges[keep]

    assert np.all(np.diff(edges) > 0)

    # SUPPORT-based separation guard on the final edges (raises loudly)
    # + the one-line bookkeeping summary (user "strong checks" ruling).
    from ...moves.gbbands import (
        band_support_halfwidths,
        check_band_support_separation,
    )

    sep = check_band_support_separation(
        edges, Tobs, int(unit_stride),
        sep_factor=sep_factor, oversample=oversample, amp=amp,
        context="get_n_based_band_edges", enforce=True,
    )
    widths = np.diff(edges)
    support = band_support_halfwidths(
        edges, Tobs, oversample=oversample, amp=amp)
    ldf = float(layer_df)
    logging.getLogger(__name__).info(
        "[GB_BAND_EDGES get_n] %d bands over [%.4e, %.4e] Hz; widths "
        "min/median/max = %.0f/%.0f/%.0f FD bins (%.3g/%.3g/%.3g "
        "layer-widths); worst edge-source overhang %.3g layer-widths "
        "(vs default slab margin max(leakage=2, support)+guard); "
        "min safe stride %s at sep_factor %.3g (running stride %d).",
        len(edges) - 1, edges[0], edges[-1],
        float(np.min(widths) / df), float(np.median(widths) / df),
        float(np.max(widths) / df),
        float(np.min(widths) / ldf), float(np.median(widths) / ldf),
        float(np.max(widths) / ldf),
        float(np.max(support) / ldf),
        str(sep["min_safe_stride"]), float(sep["sep_factor"]),
        int(unit_stride),
    )
    return edges


@dataclasses.dataclass
class GBSettings(Settings):
    """Settings dataclass describing the GB branch in an Erebor-style recipe.

    Mostly holds parameter limits, frequency-band oversampling, and the
    starting frequency / TDI configuration consumed by :class:`GBSetup`.
    """

    A_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    # GB's own tempering ladder size (the engine runs cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("GB_NTEMPS", 24, int))
    # Ceiling on the DYNAMIC nleaves_max sizing (2x catalogue sources in
    # band). The fresh-start prior draw allocates
    # (ntemps, nwalkers, nleaves_max, ndim) float64 several times over, so
    # an uncapped full-band catalogue count OOMs the host at init (observed:
    # 83.8 GB peak against an 80 GB allocation on the 3-month combined run).
    # An explicitly set ``nleaves_max`` bypasses the resolver and this cap.
    nleaves_max_cap: int = dataclasses.field(
        default_factory=env_default("GB_NLEAVES_MAX_CAP", 20000, int)
    )
    m_chirp_lims: typing.List[float] = dataclasses.field(default_factory=list)
    fdot_lims: typing.List[float] = dataclasses.field(default_factory=list)
    phi0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    iota_lims: typing.List[float] = dataclasses.field(default_factory=list)
    psi_lims: typing.List[float] = dataclasses.field(default_factory=list)
    alpha_lims: typing.List[float] = dataclasses.field(default_factory=list)
    delta_lims: typing.List[float] = dataclasses.field(default_factory=list)
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    # Explicit band_edges override (Hz, ascending, one value per band
    # boundary). If set (e.g. ``fit.gb.band_edges_override = np.array([...])``)
    # it REPLACES the per-WDM-layer edges derived in ``compute_band_edges``.
    band_edges_override: typing.Optional[typing.Any] = None
    # Subdivide each WDM layer into ``subband_divisor`` sub-bands when deriving
    # band_edges in the legacy ``uniform`` mode ONLY (2 -> half-layer_df
    # edges). Ignored if ``band_edges_override`` is set, and IGNORED (with a
    # DeprecationWarning) by the ``get_n`` mode -- free-floating edges have
    # no snap grid (2026-08-15 user ruling: the scheduling division is
    # independent of the WDM pixelization).
    subband_divisor: int = dataclasses.field(
        default_factory=env_default("GB_SUBBAND_DIVISOR", 1, int)
    )
    # WDM band-edge construction mode (ignored if ``band_edges_override`` set):
    #   "uniform" [default] -> today's per-layer edges (one band per
    #           layer_df/subband_divisor step), bit-for-bit unchanged.
    #   "get_n" -> FREE-FLOATING bands at the 2*get_N minimum width
    #           (:func:`get_n_based_band_edges`; 2026-08-15 user rulings):
    #           each band's width is its own MINIMUM
    #           ``2 * get_N(f_max_band) / Tobs`` (the installed chunked-het
    #           FD sizing function, self-consistent in the band's upper
    #           edge), packed left-to-right for the MAXIMAL band count; no
    #           snapping to layer boundaries -- scheduling is independent
    #           of the WDM pixelization (storage slabs stay layer-derived).
    #           Low-f bands (small N) come out narrow, high-f bands (large
    #           N) wide -> more parallel (temp, walker, band) cells where
    #           sources are dense. Env: ``GB_BAND_EDGES_MODE``.
    band_edges_mode: str = dataclasses.field(
        default_factory=env_default("GB_BAND_EDGES_MODE", "uniform", str)
    )
    # OPTIONAL COARSENING for ``band_edges_mode="get_n"``. 0 [default] ->
    # maximal packing at the 2*get_N floor (the authority; most bands, each
    # at its own minimum width). >0 -> adjacent minimum-width bands are
    # MERGED (evenly spread) down to ~this many bands -- merged bands are
    # wider than each member's floor, so the width rule's separation
    # guarantees are preserved. Saturates at the maximal packing (a target
    # above it is a no-op). Env: ``GB_BAND_TARGET_COUNT``.
    band_target_count: int = dataclasses.field(
        default_factory=env_default("GB_BAND_TARGET_COUNT", 0, int)
    )
    # DEPRECATED (2026-08-15 user ruling): band widths in ``get_n`` mode
    # are set by ``2 * get_N(f_max_band)/Tobs`` -- the minimum the
    # chunked-het bookkeeping allows -- so a layer-based minimum width has
    # NO role; a non-default value is warned about and ignored. The knob
    # never affected the ``uniform`` mode. Env: ``GB_BAND_MIN_LAYERS``.
    band_min_layers: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_BAND_MIN_LAYERS", None, float)
    )
    # Band-unit stride for the GB move's concurrent sub-band scheduling
    # (flows to the moves as ``band_units`` via ``group_proposal_kwargs``).
    # Stride k partitions bands into k units by ``band_index % k``; each
    # unit's bands are opened/scored CONCURRENTLY (independent buffer
    # cells), so same-unit bands always have k - 1 closed bands between
    # them. Default 2 = the legacy odd/even parity scheduling,
    # bit-identical to historical behavior. Honored by BOTH the proposal
    # unit loop and run_tempering. The orthogonality physics ruling
    # (FD inner product ~0 => WDM inner product ~0, even within one
    # wavelet layer) makes frequency separation -- not wavelet-pixel
    # support -- the concurrency constraint; the guard is SUPPORT-based
    # (gap between same-unit bands >= GB_ORTHO_SEP_FACTOR x the sum of
    # edge-source half-supports get_N(f_edge)/Tobs -- see
    # ``check_band_support_separation``). Under the 2*get_N width rule
    # stride 2 always passes at factor 1.0; stride 3 adds a full band of
    # clearance. Env: ``GB_BAND_UNIT_STRIDE``.
    band_unit_stride: int = dataclasses.field(
        default_factory=env_default("GB_BAND_UNIT_STRIDE", 2, int)
    )
    oversample: int = 4. # FD
    extra_buffer: int = 5
    start_resample_iter: Optional[typing.Tuple[int]] = (-1,)  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    num_repeat_proposals: int = 100
    search_kwargs: Optional[dict] = None
    group_proposal_kwargs: Optional[dict] = None
    # Peak GPU-memory cap for the GB special move. ``n_subbands`` is the
    # maximum number of (temp, walker, band) sub-band buffer cells resident
    # at once (the ``BandScheduler`` slot count / the move's
    # ``num_band_preload``). The move preloads at most this many cells and
    # swaps finished cells out for pending ones, so PEAK sub-band-buffer
    # memory scales with ``n_subbands``, not the total cell count
    # (ntemps*nwalkers*n_bands). Lower it to bound peak GPU memory on large
    # runs; the historical move default is 20000 (effectively "load every
    # cell"). Env: ``GB_N_SUBBANDS``. Flows to the move via
    # ``group_proposal_kwargs["num_band_preload"]`` (see ``__post_init__``).
    # Peak sub-band-buffer memory is ``n_subbands * per_slot`` once the cap
    # binds (total cells > n_subbands); ntemps/nwalkers/n_bands only decide
    # whether it binds and how many scheduler rounds run, NOT the peak. Default
    # 1024 (was an effectively-uncapped 20000): bounds peak on large multi-band
    # runs while leaving small runs -- whose cell count is already < 1024 --
    # untouched. Env: ``GB_N_SUBBANDS``.
    n_subbands: int = dataclasses.field(
        default_factory=env_default("GB_N_SUBBANDS", 1024, int)
    )
    # Sampling-basis knob: sample **chirp mass ``Mc``** at slot 2 (Msol)
    # instead of ``fdot``. The physical basis is unchanged (fdot); the
    # sampling->physical :class:`TransformContainer` gets a multi-parameter
    # ``(f0, Mc) -> (f0, fdot)`` transform via
    # :func:`gbgpu.utils.utility.get_fdot`, and ``key_map={"Mc": "fdot"}``
    # places the sampled Mc at the fdot slot before the transform runs.
    # **Default: True** -- Mc is the physically meaningful axis (fdot varies
    # by ~10 orders of magnitude across the GB band, while Mc is ~O(1) Msol
    # everywhere), and the F-stat proposal / astrophysical joint prior both
    # live natively in the Mc axis. Set ``GB_USE_CHIRP_MASS=0`` to fall back
    # to the legacy fdot sampling basis (requires ``fdot_lims`` populated).
    use_chirp_mass: bool = dataclasses.field(
        default_factory=env_default("GB_USE_CHIRP_MASS", True, bool)
    )
    # When ``use_chirp_mass`` is on, swap the separate uniform priors on
    # ``f0`` and ``Mc`` for the 6-component astrophysical GMM fit from
    # ``heatmap_GMMs.ipynb`` (combined heatmap of 102 population-synthesis
    # models; :class:`lisatools.sampling.f0_mchirp_prior.F0McGMMSampling`,
    # truncated to the run's (f0, Mc) box and renormalized). **Default:
    # True** -- every GB-carrying stock fit now samples the astrophysical
    # joint (f0, Mc) prior; set ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR=0`` for
    # the legacy separate uniforms. Ignored (with the legacy fdot basis)
    # when ``use_chirp_mass=False``.
    use_astrophysical_f0_mc_prior: bool = dataclasses.field(
        default_factory=env_default("GB_USE_ASTROPHYSICAL_F0_MC_PRIOR", True, bool)
    )
    # Half-width M of the uniform prior on the 9th sampled parameter
    # ``fdot_astro_ratio`` (see :attr:`use_fdot_astro`): ``r = fdot_astro /
    # fdot_gr ~ U[-M, M]``, physical ``fdot = fdot_gr(f0, Mc) * (1 + r)``,
    # so ``r < -1`` represents interacting ``fdot < 0`` systems and the
    # total fdot is bounded to ``(1 +/- M) * fdot_gr`` by construction.
    # Catalogue facts (scripts/gb/fdot_astro_prior_study.py, full wdwd
    # catalogue): at truth Mc the whole population lies in ``r`` in
    # ``[-1.5, +5e-4]``; the mirror-convention seeds land at ``r = 0`` /
    # ``r = -2``; the most negative sources need ``r`` down to ``-4.83``
    # even at Mc = 1.0 -- so M >= 5 keeps every catalogue source
    # representable (M = 3 would exclude the loudest accretors). Env:
    # ``GB_FDOT_ASTRO_RATIO_MAX``.
    fdot_astro_ratio_max: float = dataclasses.field(
        default_factory=env_default("GB_FDOT_ASTRO_RATIO_MAX", 5.0, float)
    )
    # Distance box [d_min, d_max] in kpc for slot 0 of the distance basis
    # (see :attr:`use_distance`): the amplitude is DERIVED from the sampled
    # (dist, f0, Mc), so distance -- not lnA -- is the free parameter. The
    # 1 pc floor keeps A finite (A propto 1/d). PLACEHOLDER uniform box per
    # the incoming Galaxy 3-D (dist, sky) distribution; tighten when that
    # lands. Env: ``GB_DIST_LIMS`` (comma pair) is not wired -- set the
    # field directly. Units: kpc (gbgpu ``get_amplitude`` convention).
    dist_lims: typing.List[float] = dataclasses.field(
        default_factory=lambda: [0.001, 40.0]
    )
    # Incoming 3-D joint prior over ``(dist, alpha, sin_delta)`` (the
    # Galaxy-shaped sky+distance distribution). An eryn duck-typed
    # distribution: ``rvs(size) -> (size, 3)`` columns (dist[kpc],
    # alpha[rad], sin_delta), ``logpdf((n, 3)) -> (n,)``; must pickle
    # (sprint rule). None (default) -> a placeholder nested
    # ``ProbDistContainer`` of independent uniforms (dist over ``dist_lims``,
    # alpha over ``alpha_lims``, sin_delta over ``sin(delta_lims)``) is built
    # in :meth:`GBSetup.init_sampling_info`. Swap this in for the real
    # distribution; the birth container's independent U(dist) then needs a
    # p(dist | sky) draw (documented follow-up).
    sky_dist_distribution: typing.Optional[typing.Any] = None
    # Build the 3-D Milky Way prior over (dist, alpha, sin_delta) instead of
    # the independent-uniform placeholder. Ignored when
    # ``sky_dist_distribution`` is set explicitly (that always wins).
    # See lisatools.globalfit.priors.galaxy_sky_dist; validate with
    # scripts/gb/check_galaxy_sky_dist.py.
    use_galaxy_prior: bool = dataclasses.field(
        # Default ON (2026-08-12 user ruling): the 3-D Milky Way joint is the
        # production prior; GB_USE_GALAXY_PRIOR=0 restores the independent
        # uniforms (dist x sky placeholder).
        default_factory=env_default("GB_USE_GALAXY_PRIOR", True, bool)
    )

    @property
    def use_fdot_astro(self) -> bool:
        """9-parameter basis switch (deliberately NOT a separate knob).

        The ``fdot_astro_ratio`` column is part of the astrophysical-prior
        package: active iff ``use_chirp_mass and
        use_astrophysical_f0_mc_prior``. ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR
        =0`` (or ``GB_USE_CHIRP_MASS=0``) reverts to the 8-parameter basis.
        """
        return bool(self.use_chirp_mass and self.use_astrophysical_f0_mc_prior)

    @property
    def use_distance(self) -> bool:
        """Distance-basis switch (slot 0 = ``dist`` instead of ``lnA``).

        Part of the SAME astrophysical package as :attr:`use_fdot_astro`
        (also NOT a separate knob): once Mc is sampled the amplitude is
        derived, so distance is the natural free parameter and the prior can
        carry a 3-D (dist, sky) joint. Active exactly when
        :attr:`use_fdot_astro` is.
        """
        return self.use_fdot_astro
    # Custom RJ-birth distribution for the prior RJ moves. An eryn
    # duck-typed distribution over the FULL 8-column GB sampling basis
    # (``rvs(size) -> (size, 8)`` with f0 in mHz at column 1 and Mc at
    # column 2; ``logpdf((n, 8)) -> (n,)``). Build one from a 4-D intrinsic
    # proposal (e.g. :class:`lisatools.sampling.fstat_proposal
    # .FStatProposal4D`) with :func:`lisatools.sampling.fstat_proposal
    # .make_gb_rj_birth_container`; wrap narrow proposals in a
    # ``UniformFloorMixture`` so death factors stay finite wherever
    # refined leaves can drift. Must pickle/deepcopy (sprint rule) --
    # FStatProposal4D-backed containers do. None (default) keeps the
    # stock global-prior births.
    rj_birth_distribution: typing.Optional[typing.Any] = None
    # Warm-start components npz path (workstream B, user ruling 2026-08-24):
    # the clustered previous-run posterior components file produced by
    # scripts/gb/warmstart_fit_from_store.py. Non-empty -> build_gb_moves
    # constructs the ``rj_warm_search`` RJ birth move (the prior-RJ move
    # class drawing full 9-column births from
    # ``lisatools.sampling.warmstart_proposal.WarmStartComponents``: mixture
    # weights ~ inclusion probability p, f0 windows re-derived against the
    # NEW run's 1/Tobs, no phase maximization); the staged search recipes
    # order it IMMEDIATELY BEFORE ``rj_fstat_search``. Empty/unset (default)
    # -> the feature is entirely absent and runs are bit-identical. A plain
    # path string -> deepcopy/pickle safe. Env: GB_WARM_START_COMPONENTS.
    warm_start_components: str = dataclasses.field(
        default_factory=env_default("GB_WARM_START_COMPONENTS", "", str)
    )
    # RESERVED (v1 parse-only; documented in submit_gf_6mo_v1.sh's staged
    # warm-start block): the future MIXTURE WEIGHT of the warm-start
    # component against the F-stat proposal once both feed ONE birth move
    # (v2). v1 runs them as separate sequential moves (rj_warm_search then
    # rj_fstat_search), so this parses but is unused. Env:
    # GB_WARM_START_WEIGHT.
    warm_start_weight: typing.Optional[float] = dataclasses.field(
        default_factory=env_default("GB_WARM_START_WEIGHT", None, float)
    )
    # PE-stage extrinsic DRAW (user design ruling 2026-08-25). STAGE
    # SPLIT: the SEARCH RJ stages (rj_fstat_search / rj_prior_removal /
    # rj_replace) PIN phi0/cos_iota/psi deterministically at the F-stat
    # maximizers and charge them as uniform constants — unchanged,
    # bit-identical. The PE stages (rj_fstat_pe / rj_prior_pe, when their
    # F-stat distance-birth path is active) instead DRAW each extrinsic
    # from a genuine distribution centered on its maximizer — von Mises
    # for phi0 (period 2 pi), von Mises on the DOUBLED angle for psi
    # (period pi), truncated Gaussian on [-1, 1] for cos iota — each
    # eps-floor-mixed with the uniform law on its domain, and charge the
    # real forward AND reverse densities in the RJ factors (exact
    # detailed balance; deaths evaluate the reverse density around the
    # dead row's OWN maximizers). Widths sigma = geom/SNR from the F-stat
    # curvature; the (phi0 + pi, psi + pi/2) F-stat identity is handled
    # by summing the density over both representatives. See
    # ``lisatools.sampling.fstat_proposal.pe_extrinsic_rvs`` /
    # ``pe_extrinsic_logpdf`` and ``GBSpecialBase._pe_or_pin_extrinsics``.
    # Default ON; =0 restores today's PE pin + uniform-wash behavior
    # bit-identically. Companion env-only knobs:
    # GB_PE_EXTRINSIC_FLOOR_EPS (default 0.05) and
    # GB_PE_EXTRINSIC_SIGMA_GEOM (default 2.0).
    # Env: GB_PE_EXTRINSIC_DRAW.
    pe_extrinsic_draw: bool = dataclasses.field(
        default_factory=env_default("GB_PE_EXTRINSIC_DRAW", True, bool)
    )
    # -- search RJ cycle knobs (2026-08-01; search mode ONLY, i.e.
    # GB_MODE=search) -- when on, the variant's recipe setup inserts the
    # matching stock move(s) right after the fstat-birth prior move
    # (``rj_prior``) in its stage, so the stage's GFCombineMove runs the
    # per-iteration cycle fstat-birth -> fstat-REPLACE -> prior-REMOVAL in
    # guaranteed order (eryn CombineMove proposes its moves sequentially in
    # list order). PE recipes are untouched. Plain bools -> deepcopy/pickle
    # safe.
    #
    # search_rj_replace: build + install the fixed-dimension REPLACEMENT
    # move ``rj_replace`` directly AFTER the fstat-birth move in the
    # gb_search cycle (2026-08-24 exact-MH reinstatement, USER directive):
    # full 9-column candidates from the F-stat grid container + epoch
    # center table (stored maximizing extrinsics), scored at their EXACT
    # likelihood -- never phase-maximized -- so a higher-SNR table draw
    # can replace a mis-seated leaf in one accepted move. See
    # GBSpecialBase._run_replace_step for the acceptance derivation.
    # Default ON (search recipes only; full_pe is untouched -- PE
    # installation is future work). Env: GB_SEARCH_RJ_REPLACE.
    search_rj_replace: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_RJ_REPLACE", True, bool)
    )
    # search_prior_removal: build + install the removal-only pruning move
    # ``rj_prior_removal`` (GBSpecialRJPriorMove with the PRIOR container +
    # rj_removal_only=True: births force-rejected, deaths untouched). Env:
    # GB_SEARCH_PRIOR_REMOVAL.
    search_prior_removal: bool = dataclasses.field(
        default_factory=env_default("GB_SEARCH_PRIOR_REMOVAL", False, bool)
    )
    # leaf_cap_iter_only: per-band progressive leaf caps advance on a FIXED
    # schedule -- the only gate is ``iters >= GB_LEAF_CAP_MIN_ITERS`` (the
    # lnL-plateau and occupancy tests are skipped; the cap < nleaves_max
    # guard stays). Env: GB_LEAF_CAP_ITER_ONLY.
    leaf_cap_iter_only: bool = dataclasses.field(
        default_factory=env_default("GB_LEAF_CAP_ITER_ONLY", False, bool)
    )
    # cap_divisor: the LEAF-CAP CELL grid (user design 2026-08-15). Each
    # sub-band is split into this many equal pieces and the per-cell leaf
    # caps are enforced on THOSE, not on the sub-band. Sub-band widths are
    # set by what the likelihood engine can run concurrently; the scale at
    # which two GB sources actually get confused is the posterior width,
    # which is much narrower. NOTHING ELSE MOVES -- scheduling, units,
    # buffers, tempering, band shutoff and the cell-ll credit all stay on
    # the band grid, and every cap cell is contained in exactly one band.
    #
    # DEFAULT 8, from the snap11 3-month store (154 uniform 138.9 uHz bands
    # = 1080 FD bins each; 1147 coexisting same-band cold pairs): the pair
    # separation distribution is bimodal -- ~10% of pairs sit within 8 FD
    # bins (genuinely degenerate, no subdivision separates them) and the
    # rest spread near-uniformly across the band. Directly assigning cells,
    # the fraction of pairs SHARING a cap cell is 100% at K=1, 48.9% at
    # K=2, 23.1% at K=4, 16.5% at K=8 and 12.8% at K=16: 4 -> 8 removes 76
    # shared cells, 8 -> 16 only another 42 for double the bookkeeping. A
    # K=8 cell is 17.4 uHz / 135 FD bins, still far wider than a source's
    # own 3-month frequency extent (~+-2 uHz of Doppler sidebands), so cell
    # membership stays stable under in-model f0 moves; K=16 starts to
    # approach that scale. 8 is the knee.
    #
    # ``GB_CAP_DIVISOR=1`` reproduces the pre-2026-08-15 per-band caps
    # BIT-IDENTICALLY (the cap grid IS the band grid). Changing this on an
    # existing run requires scripts/fstat_proposal/migrate_gb_cap_grid.py --
    # the resume guard refuses a mismatched store. Env: GB_CAP_DIVISOR.
    cap_divisor: int = dataclasses.field(
        default_factory=env_default("GB_CAP_DIVISOR", 8, int)
    )
    # cap_stagger: STAGGER the cap-cell grid against the sub-band grid
    # (user design 2026-08-20, the v5 grid). Interior cap edges shift by
    # half a cell width so NO cap edge coincides with a band edge -- the
    # leaf-cap seams and the band seams (scheduling / serial-within-band /
    # F-stat fit interior) decouple, and no source can sit on both at
    # once. Band b still owns cells b*K..b*K+K-1 (sizes, reshapes and
    # storage unchanged); the cell at each band seam physically straddles
    # it. Requires cap_divisor >= 2 (ignored at 1). Changing it on an
    # existing store is refused by the resume guard -- fresh store only.
    # Env: GB_CAP_STAGGER.
    cap_stagger: bool = dataclasses.field(
        default_factory=env_default("GB_CAP_STAGGER", False, bool)
    )
    # cap_overlap_frac: OVERLAP the leaf-cap cells (user design 2026-08-23).
    # Each cap cell's enforcement SPAN widens symmetrically so it shares
    # this fraction of its own width with EACH neighbour: 0.25 gives the
    # 1/4-overlap / 1/2-alone / 1/4-overlap layout (cell width w = s/(1-p)
    # for stride s; extension x = (w-s)/2 = s/6 at p=0.25). The stored EDGE
    # grid never changes -- same count, same stride, same stagger -- so
    # resume guards and every stored cap array keep their shapes, and a
    # rewound store picks the new semantics up with no migration. A leaf in
    # an overlap zone is a member of BOTH covering cells: the occupancy
    # census counts it in both, and a location is AT CAP when ANY covering
    # cell is at its cap (AND-headroom for births / cap-drift entries).
    # Purpose: prevents FORMATION of split sources at cap-cell edges (the
    # flagship 20.38 mHz double-count straddled a cell edge -- each
    # fragment lived under its own cell's cap while the pair shared one
    # posterior mode). Only acts while caps are armed (search stages).
    # Valid range [0, 0.5); 0 (default) is bit-identical to the exact
    # partition. Meaningless at cap_divisor == 1 (forced off there).
    # Env: GB_CAP_OVERLAP_FRAC.
    cap_overlap_frac: float = dataclasses.field(
        default_factory=env_default("GB_CAP_OVERLAP_FRAC", 0.0, float)
    )
    # Task-b: narrow per-band WDM slabs. Each per-band sub-band-buffer slab
    # spans a few WDM layers centered on the band instead of the full analysis
    # band ``Nf_active``, cutting the dominant buffer memory term by
    # ~Nf_active/slab_Nf. WDM path only (FD is already per-band narrow).
    # Requires the chunked-het backend built with the task-b per-slab origin
    # (the default CUDA/CPU build). Env: GB_WDM_BAND_SLAB_LAYERS.
    #   0    -> AUTO: band layer span + 2*(leakage + wdm_slab_guard_layers),
    #           leakage=2 (recommended-Tukey estimate) -> ~band_span + 6. [default]
    #   N>0  -> EXPLICIT N layers.
    #   None -> OFF (full active band; bit-identical to pre-task-b). Not
    #           reachable from the env knob (which casts to int); set it
    #           programmatically (``fit.gb.wdm_band_slab_layers = None``) to
    #           restore the pre-task-b full-band buffers.
    #
    # AUTO is the default because the full-band slab is the dominant sub-band
    # buffer term: a cell spans the whole analysis band (e.g. 180 WDM layers)
    # when the GB band itself occupies only a few, so AUTO cuts per-cell
    # memory by ~Nf_active/slab_Nf (~20x at Nf_active=180) on both the device
    # buffers and the host staging copies in the ACA repack.
    wdm_band_slab_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_WDM_BAND_SLAB_LAYERS", 0, int)
    )
    # Adjustable guard (extra WDM layers each side) used by the AUTO slab size
    # (wdm_band_slab_layers=0). Env: GB_WDM_SLAB_GUARD_LAYERS.
    wdm_slab_guard_layers: int = dataclasses.field(
        default_factory=env_default("GB_WDM_SLAB_GUARD_LAYERS", 1, int)
    )
    start_freq_ind: Optional[int] = 0  # goes into GPU for start of data stream
    t0: Optional[float] = 0.0
    tdi_setup: Optional[str] = "XYZ" # other options are AET and AE.
    use_tdi2: Optional[bool] = True
    waveform_kwargs: dict = dataclasses.field(default_factory=dict)
    # Domain selection for this branch. The user passes the same
    # ``DomainSettingsBase`` that lives on the parent ``GeneralSetup``
    # (FDSettings, STFTSettings, WDMSettings, ...). The band structure /
    # waveform-engine selection branches on ``isinstance(domain_settings, ...)``
    # — no string mode flag.
    domain_settings: Optional[DomainSettingsBase] = None
    # Optional WDM-domain likelihood object (a
    # ``gbgpu.gbcomps.GBWDMComputations`` instance). Required when
    # ``domain_settings`` is a :class:`WDMSettings`; ignored otherwise. The
    # user builds this once their WDM grid + lookup table are known
    # (see global-fit input scripts).
    gb_wdm_comp: typing.Any = None



class GBSetup(Setup, GBSettings):
    """:class:`Setup` for galactic binaries in the Erebor recipe.

    Args:
        gb_settings: Settings dataclass holding GB parameter ranges and
            domain configuration.
    """

    def __init__(self, gb_settings: GBSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the GB :class:`TransformContainer`, prior, periodicity, and waveform kwargs.

        Plain parameter names (same convention as
        :func:`make_gb_transform_container`) with ICRS sky angles
        (``alpha`` / ``sin_delta``) -- the run frame is ICRS. ``A`` is
        sampled in ``ln A`` (the ``np.exp`` transform maps to the physical
        amplitude), ``f0`` in mHz.

        When :attr:`GBSettings.use_chirp_mass` is on, slot 2 of the sampling
        basis carries ``Mc`` (chirp mass, Msol) instead of ``fdot``. A
        multi-parameter ``(f0, Mc) -> (f0, fdot)`` transform via
        :func:`gbgpu.utils.utility.get_fdot` recovers the physical fdot after
        ``fill_values`` moves the sampled ``Mc`` to the fdot slot (via
        ``key_map={"Mc": "fdot"}``). When
        :attr:`GBSettings.use_astrophysical_f0_mc_prior` is also on, the
        separate ``f0`` and ``Mc`` uniforms are replaced by the 6-component
        heatmap GMM from :func:`~lisatools.sampling.f0_mchirp_prior
        .F0McGMMSampling.from_heatmap` under a tuple key ``("f0", "Mc")``,
        AND the basis gains a 9th sampled column ``fdot_astro_ratio``
        (:attr:`GBSettings.use_fdot_astro`): ``r = fdot_astro / fdot_gr ~
        U[-M, M]`` (M = :attr:`GBSettings.fdot_astro_ratio_max`), physical
        ``fdot = fdot_gr(f0, Mc) * (1 + r)`` with ``fddot`` exactly 0 --
        interacting ``fdot < 0`` systems live at ``r < -1``.
        """
        third_name = "Mc" if self.use_chirp_mass else "fdot"
        # Slot 0 is the luminosity DISTANCE (kpc) in the distance basis --
        # amplitude is derived from the sampled (dist, f0, Mc) -- else lnA.
        slot0_name = "dist" if self.use_distance else "A"
        input_basis = [slot0_name, "f0", third_name, "phi0",
                       "cos_iota", "psi", "alpha", "sin_delta"]
        if self.use_fdot_astro:
            # 9th sampled column: r = fdot_astro / fdot_gr (dimensionless;
            # physical fdot = fdot_gr(f0, Mc) * (1 + r), fddot stays 0).
            input_basis.append("fdot_astro_ratio")

        if self.transform is None:
            # THE single GB transform factory (phi0 sign convention lives
            # there, nowhere else).
            self.transform = make_gb_transform_container(
                use_chirp_mass=self.use_chirp_mass,
                use_fdot_astro=self.use_fdot_astro,
                use_distance=self.use_distance,
                mc_lims=(
                    tuple(self.m_chirp_lims) if self.m_chirp_lims
                    else (0.001, 1.0)
                ),
            )

        if self.periodic is None:
            self.periodic = {"gb": {"phi0": 2*np.pi, "psi": np.pi, "alpha": 2 * np.pi}}

        if self.priors is None:
            if self.use_chirp_mass and self.use_astrophysical_f0_mc_prior:
                # Joint tuple-key prior on (f0[mHz], Mc[Msol]) from the
                # heatmap 6-component GMM fit.
                from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

                # Truncate the heatmap GMM to the run's sampled box and
                # renormalize (RJ birth/death compares prior masses across
                # leaf counts, so the truncated prior must integrate to 1
                # over exactly the space the sampler covers). Gaussian
                # tails keep the density finite anywhere in the box, even
                # outside the original heatmap support.
                _f0_mc_prior = F0McGMMSampling.from_heatmap(
                    f0_lims_mHz=tuple(np.asarray(self.f0_lims) * 1e3),
                    mc_lims=tuple(self.m_chirp_lims) if self.m_chirp_lims
                    else None,
                )
                # Slot 0 is DISTANCE (kpc) in this branch (use_distance): the
                # sky and distance share a 3-D JOINT prior over
                # (dist, alpha, sin_delta) -- the Galaxy-shaped distribution
                # is incoming, so this is a tuple-key slot. For now the joint
                # is a placeholder nested ProbDistContainer of independent
                # uniforms (dist over dist_lims, sky uniform); swap
                # sky_dist_distribution for the real 3-D distribution.
                sky_dist = self.sky_dist_distribution
                if sky_dist is None and getattr(
                    self, "use_galaxy_prior", False
                ):
                    from ...priors.galaxy_sky_dist import (
                        build_gb_galaxy_sky_dist,
                    )

                    sky_dist = build_gb_galaxy_sky_dist(
                        dist_lims=self.dist_lims
                    )
                    self.logger.info(
                        "GB sky/distance prior: 3-D Milky Way "
                        "(GB_USE_GALAXY_PRIOR=1), dist_lims=%s kpc.",
                        list(self.dist_lims),
                    )
                if sky_dist is None:
                    sky_dist = ProbDistContainer({
                        0: uniform_dist(self.dist_lims[0], self.dist_lims[1]),
                        1: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                        2: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                    })
                priors_gb = {
                    # tuple members get CONSECUTIVE columns by insertion
                    # order; reset_key_order below remaps them to the real
                    # basis columns (dist->0, alpha->6, sin_delta->7).
                    ("dist", "alpha", "sin_delta"): sky_dist,
                    ("f0", "Mc"): _f0_mc_prior,
                    "phi0": uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    "cos_iota": uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    "psi": uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    # 9th column: r ~ U[-M, M]. Prior defined in the SAMPLING
                    # basis (no in-sampler Jacobian); induced physical prior
                    # at fixed (f0, Mc) is uniform in fdot -- see
                    # McDistFdotAstroQuad.
                    "fdot_astro_ratio": uniform_dist(
                        -self.fdot_astro_ratio_max, self.fdot_astro_ratio_max
                    ),
                }
            elif self.use_chirp_mass:
                if not self.m_chirp_lims:
                    raise ValueError(
                        "use_chirp_mass=True requires GBSettings.m_chirp_lims "
                        "to be set (a two-element [Mc_min, Mc_max] in Msol)."
                    )
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),
                    input_basis[2]: uniform_dist(self.m_chirp_lims[0], self.m_chirp_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: sort defensively.
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }
            else:
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),  # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                    input_basis[2]: uniform_dist(self.fdot_lims[0], self.fdot_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: cos(iota_lims) comes out
                    # (max, min), so sort into increasing order before handing
                    # it to uniform_dist -- never rely on the dist silently
                    # swapping reversed bounds. (Same defensive sort on the
                    # sin(delta) line even though sin is increasing there.)
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }

            _gb_prior = ProbDistContainer(priors_gb)
            # The distance branch inserts the ("dist","alpha","sin_delta")
            # joint FIRST, so its members grabbed consecutive columns 0/1/2;
            # remap by name to the real basis columns (dist->0, alpha->6,
            # sin_delta->7). A no-op for the other branches (already in
            # column order), so it is safe to apply whenever the input basis
            # is available.
            if self.use_distance:
                _gb_prior.reset_key_order(list(input_basis))
            self.priors = {"gb": _gb_prior}

        if self.betas is None:
            # snrs_ladder = np.array(
            #     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
            #      15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2]
            # )
            ntemps_pe = 24 # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        # GBGPU FD waveform kwargs. The WDM-domain engine in
        # ``gb_likelihood.WDMBandLikelihoodEngine`` ignores
        # ``start_freq_ind`` / ``oversample`` (it dispatches to the WDM C
        # kernel via ``GBWDMComputations``), but we keep the same dict so
        # the down-stream Buffer's ``tdi_channel_setup`` read still works
        # on either path. The settings file may pre-populate
        # ``waveform_kwargs`` (stft_tof field); this default is built from
        # the other fields only when it is left empty.
        if not self.waveform_kwargs:
            self.waveform_kwargs = dict(
                dt=self.dt,
                T=self.Tobs,
                use_c_implementation=True,
                oversample=self.oversample,
                start_freq_ind=self.start_freq_ind,
                tdi_channel_setup=self.tdi_setup,
                tdi2=self.use_tdi2
            )

        if self.group_proposal_kwargs is None:
            self.group_proposal_kwargs: typing.Dict[str, Any] = dict(
                n_iter_update=1, live_dangerously=True, a=1.75, num_repeat_proposals=200
            )
        # Peak sub-band-buffer memory cap. Injected here (not only in the
        # None-default dict above) so an explicit ``group_proposal_kwargs``
        # still gets the ``n_subbands`` knob unless it names one itself.
        self.group_proposal_kwargs.setdefault("num_band_preload", int(self.n_subbands))
        # Task-b narrow per-band WDM slabs (None -> full active band).
        self.group_proposal_kwargs.setdefault(
            "wdm_band_slab_layers", self.wdm_band_slab_layers
        )
        self.group_proposal_kwargs.setdefault(
            "wdm_slab_guard_layers", self.wdm_slab_guard_layers
        )
        # Band-unit stride for the concurrent sub-band scheduling
        # (GB_BAND_UNIT_STRIDE; default 2 = legacy parity, bit-identical).
        # Flows to every GBSpecial* move as the ``band_units`` ctor kwarg.
        self.group_proposal_kwargs.setdefault(
            "band_units", int(getattr(self, "band_unit_stride", 2))
        )

        if self.search_kwargs is None:
            # Configuration for the stft_tof per-band serial GB search
            # (consumed by ``GBSpecialRJSerialSearchMCMC.setup`` and
            # ``GBSpecialRJRefitMove.setup``):
            #   nwalkers / ntemps          — per-band ParaEnsembleSampler size
            #   burn_1 / nsteps_1          — stage 1: F-stat (phase-maximized)
            #                                MCMC, Gibbs on the 4 intrinsic-
            #                                like params
            #   snr_threshold              — band "found a source" gate on the
            #                                stage-1 chain's min optimal SNR
            #   burn_2 / nsteps_2          — stage 2: full-likelihood refine
            #                                from stage-1's last sample; its
            #                                samples feed the GMM fit that
            #                                becomes the RJ proposal
            #   shutoff_band_iteration     — shut a band off after this many
            #                                consecutive source-less search
            #                                iterations (all-off reverts the
            #                                RJ proposal to the global prior)
            #   shutoff_frequency_threshold— only bands above this frequency
            #                                are shutoff-eligible (None = all)
            #   refit_start_iteration      — samples kept per leaf for the
            #                                GMM refit in GBSpecialRJRefitMove
            self.search_kwargs: typing.Dict[str, Any] = dict(
                nwalkers = 32,
                ntemps = 24,
                shutoff_band_iteration = 5,
                shutoff_frequency_threshold = None, # 4e-3
                burn_1 = 200,
                nsteps_1 = 200,
                snr_threshold = 8.0,
                burn_2 = 500,
                nsteps_2 = 500,
                refit_start_iteration = 5
            )

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        """Run the band-structure, sampling-info, and state-backend init helpers."""
        self.init_band_structure()
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default :attr:`branch_state` to :class:`GBState` and :attr:`branch_backend` to :class:`GBHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = GBState

        if self.branch_backend is None:
            self.branch_backend = GBHDFBackend

    def init_band_structure(self):
        """Compute :attr:`band_edges` and :attr:`band_N_vals` from the GB frequency range.

        Both FD and WDM domains use a frequency-banded layout. The edges
        are derived from ``df = 1/Tobs`` (always meaningful for either
        basis) and the FD-oversampled per-band N. For WDM runs the
        per-band ``band_N_vals`` is unused by the WDM likelihood engine
        (which sizes its buffers from ``WDMSettings.Nf_active`` /
        ``Nt_active``) but is preserved for shape parity.
        """
        # band separation setup
        if self.oversample is None and self.Tobs < YRSID_SI / 2.0:
            self.oversample = 2
        elif self.oversample is None:
            self.oversample = 4

        assert self.oversample >= 1

        # Clamp the GB band to the WDM active band when running on a WDM
        # grid. The WDM kernel only carries layers in [ind_min_f, ind_max_f];
        # putting band edges outside that range silently produces zero-fill
        # behaviour, so we trim to stay inside.
        if isinstance(self.domain_settings, WDMSettings):
            wdm = self.domain_settings
            wdm_min = float(wdm.ind_min_f * wdm.layer_df)
            wdm_max = float(wdm.ind_max_f * wdm.layer_df)
            self.start_freq = max(self.start_freq, wdm_min)
            self.end_freq = min(self.end_freq, wdm_max)
            if self.start_freq >= self.end_freq:
                raise ValueError(
                    "WDM active band [{:.4e}, {:.4e}] does not overlap GB "
                    "frequency range; widen WDMSettings.min_freq/max_freq "
                    "or the GB start/end_freq.".format(wdm_min, wdm_max)
                )

        _edges_override = getattr(self, "band_edges_override", None)
        if _edges_override is not None and np.size(_edges_override) >= 2:
            # Explicit user override: use the supplied band_edges verbatim
            # (Hz, ascending). ``fit.gb.band_edges_override = np.array([...])``.
            self.band_edges = np.asarray(_edges_override, dtype=float)
            self.band_N_vals = np.asarray(
                [
                    get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
                    for edge in self.band_edges[:-1]
                ]
            )
        elif isinstance(self.domain_settings, WDMSettings):
            wdm = self.domain_settings
            layer_df = float(wdm.layer_df)
            div = max(1, int(getattr(self, "subband_divisor", 1)))
            _mode = str(getattr(self, "band_edges_mode", "uniform")).lower()
            if _mode == "get_n":
                # FREE-FLOATING bands at the 2*get_N minimum width
                # (GB_BAND_EDGES_MODE=get_n; 2026-08-15 user rulings): each
                # band's width is 2*get_N(f_max_band)/Tobs, maximal
                # packing, no layer snapping. See
                # :func:`get_n_based_band_edges` for the knob semantics.
                # subband_divisor / band_min_layers are passed only so the
                # builder can warn-and-ignore when a user set them: they
                # have no role in the 2*get_N width rule (min_band_layers)
                # or the free-floating construction (subband_divisor).
                self.band_edges = get_n_based_band_edges(
                    self.start_freq,
                    self.end_freq,
                    self.Tobs,
                    layer_df,
                    subband_divisor=div,
                    oversample=int(self.oversample),
                    target_count=int(getattr(self, "band_target_count", 0)),
                    min_band_layers=getattr(self, "band_min_layers", None),
                    unit_stride=int(getattr(self, "band_unit_stride", 2)),
                )
            elif _mode != "uniform":
                raise ValueError(
                    "GB_BAND_EDGES_MODE / GBSettings.band_edges_mode must be "
                    f"'uniform' or 'get_n', got {_mode!r}."
                )
            else:
                # WDM path (default): one band per WDM frequency layer, or per
                # 1/``subband_divisor`` layer (GB_SUBBAND_DIVISOR=2 -> half-layer
                # edges). Edges land on ``k * layer_df / div`` boundaries so the
                # band grid aligns with the WDM grid; ``band_N_vals`` is unused by
                # the WDM likelihood engine but is preserved for shape parity.
                k_lo = int(np.ceil(self.start_freq / layer_df)) * div
                k_hi = int(np.floor(self.end_freq / layer_df)) * div
                if k_hi <= k_lo:
                    raise ValueError(
                        "GB frequency range [{:.4e}, {:.4e}] spans fewer than "
                        "one WDM sub-band (layer_df/div={:.4e}).".format(
                            self.start_freq, self.end_freq, layer_df / div
                        )
                    )
                self.band_edges = np.asarray(
                    [k * layer_df / div for k in range(k_lo, k_hi + 1)]
                )
            self.band_N_vals = np.asarray(
                [
                    get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
                    for edge in self.band_edges[:-1]
                ]
            )
            # SUPPORT-based separation diagnostic on the ACTUAL edges
            # (2026-08-15 rulings: the guard is expressed in FD-SUPPORT
            # terms -- gap between same-unit bands vs the sum of the
            # edge-source half-supports get_N(f_edge)/Tobs -- replacing
            # the earlier layer-width formulas). For get_n grids the
            # BUILDER already enforced it (raise); the legacy uniform /
            # override grids are NOT refused (uniform stride-2 is the
            # measured-safe production default even though high-f
            # edge-source FD supports overlap by this conservative
            # envelope -- the WDM likelihood windows there are +-1 layer
            # and the [GB_ORTHO_LL] runtime monitor, default ON, is the
            # operative accuracy check), so the diagnostic only LOGS the
            # minimum safe stride for them.
            if _mode != "get_n":
                from ...moves.gbbands import check_band_support_separation

                _sep = check_band_support_separation(
                    self.band_edges,
                    self.Tobs,
                    int(getattr(self, "band_unit_stride", 2)),
                    oversample=int(self.oversample),
                    context="GBSetup.init_band_structure",
                    enforce=False,
                )
                self.logger.info(
                    "GB band grid (mode=%s): support-based same-unit "
                    "separation at stride %d passes=%s; min safe stride "
                    "%s at sep_factor %.3g.",
                    _mode, int(getattr(self, "band_unit_stride", 2)),
                    _sep["passes"], str(_sep["min_safe_stride"]),
                    float(_sep["sep_factor"]),
                )
        else:
            # FD path: bands sized in multiples of ``df = 1/Tobs`` using the
            # FD-oversampled per-band N. Walks down from ``end_freq``
            # (stft_tof refinements: half-bin start, min_N stop guard, and
            # edge trim against out-of-bound indexing).
            # TODO: assign to binned f or leave general? probably better to be general
            band_edges_in_reverse_order = [self.end_freq]
            current_N = get_N(1e-30, self.end_freq, self.Tobs, oversample=self.oversample).item()
            min_N = get_N(1e-30, self.start_freq, self.Tobs, oversample=self.oversample).item()
            band_N_vals_reverse_order = [current_N]

            current_freq = self.end_freq - self.df / 2
            last_freq = self.end_freq
            while current_freq > self.start_freq + min_N * self.df:
                current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
                band_edges_in_reverse_order.append(current_freq)
                current_N = get_N(1e-30, current_freq, self.Tobs, oversample=self.oversample).item()
                band_N_vals_reverse_order.append(current_N)
                last_freq = current_freq

            band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
            band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]

            # trim edges to avoid out of bound indexing
            self.band_edges = band_edges[2:-1]
            self.band_N_vals = band_N_vals[2:-1]

        self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]

        self.fdot_lims = [
            get_fdot_mojito(self.f0_lims[1], sign="-"), 
            get_fdot_mojito(self.f0_lims[1], sign="+"), 
        ]

        self.num_sub_bands = len(self.band_edges) - 1
        
        self.logger.info(
            f"GB f0 prior range is set from {round(self.f0_lims[0],7)} to {round(self.f0_lims[1],7)}"
        )
        self.logger.info(f"The number of subbands is {self.num_sub_bands}")
        self.logger.info(f"Min freq of subbands is {self.band_edges.min()}")
        self.logger.info(f"Max freq of subbands is {self.band_edges.max()}")

