r"""F-statistic proposal distributions over the GB intrinsic parameters.

The GB reversible-jump F-statistic BIRTH machinery: a structured tensor grid over
4 intrinsic parameters, filled with batched F-statistic evaluations (the fit
itself lives in :mod:`lisatools.sampling.fstat_gridfit`), turned into an
exact-sampling distribution via the flattened cell inverse-CDF, and wrapped into
the 8/9-column RJ birth container. Originally "Family A" of
``scripts/fstat_proposal/PLAN_scattered_field_distributions.md``.

The target density on the grid box is

.. math::

    p(\theta) \propto \exp\left[\beta\,\mathcal{F}(\theta)\right], \qquad
    \mathcal{F} = \tfrac{1}{2}\,N^{\mathsf T} M^{-1} N, \qquad
    N_i = \langle d | A_i \rangle, \quad M_{ij} = \langle A_i | A_j \rangle,

the Cornish-Crowder F-statistic of the 4 basis filters ``A_i``, returned per
binary as the ``(N, M)`` pair by ``get_fstat_ll_wdm``
(:meth:`lisatools.chunked_het.WDMComputationsBase.get_fstat_ll_wdm`, or the
sig-het scorer ``GBSignalHetComputations.get_fstat_ll_wdm`` that the production
fit routes through :func:`~lisatools.sampling.fstat_gridfit.build_sighet_call_fstat`).
``F`` is a function of the intrinsic parameters only -- the 4 extrinsic
amplitude parameters ``(A, iota, psi, phi0)`` are analytically maximized inside
the statistic. ``F`` is the maximized log-likelihood, ``2F`` the squared SNR of
the maximizing template (so SNR thresholds are ``F = SNR^2 / 2``), and under the
null ``2F ~ chi^2_4``. The fitted grids use ``beta = 1``.

**Grid bases.** Axis 1 of every grid has two meanings, chosen by
``FSTAT_FDOT_AXIS`` (:func:`fdot_axis_on`):

* ``FSTAT_FDOT_AXIS=1`` (default since 2026-09-01; the 9mo/12mo launcher value):
  ``(f_mid [mHz], fdot [Hz/s], alpha [rad], sin_delta)`` with
  ``f0 = f_mid - (T/2) fdot``. The linear fdot axis reaches ``fdot < 0`` (the
  interacting DWDs the old basis could not birth), and :class:`FdotAxisBirth`
  converts a grid draw into the sampling basis ``(f0, Mc, r, alpha, sin_delta)``.
* ``FSTAT_FDOT_AXIS=0`` (legacy): ``(f0 [mHz], Mc [Msun], alpha, sin_delta)``
  with ``fdot = fdot_gr(f0, Mc)`` (:func:`gbgpu.utils.utility.get_fdot`), i.e.
  the ``r = 0`` manifold, which can only propose ``fdot > 0``.

The class attribute ``param_names = ("f0", "Mc", "alpha", "sin_delta")`` of the
4-D classes names the legacy layout; under the fdot basis column 0 holds
``f_mid`` and column 1 ``fdot``. Sky angles are ICRS (``alpha`` = RA,
``sin_delta``). The physical 9-column rows handed to the scorer are
``[A, f0 (Hz), fdot (Hz/s), fddot, phi0, iota, psi, alpha, delta]`` with a
placeholder amplitude and ``iota = pi/2`` (ignored by the statistic).

**Production birth container** (built per F-stat epoch by
:func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution` for the
``rj_fstat_search``, ``rj_fstat_pe`` and ``rj_fstat_pe_residual`` moves of
``GBSpecialRJFStatGridMove``): an eryn ``ProbDistContainer`` whose slot 0
(``dist``, uniform in ``dist_lims``) and ``phi0`` / ``cos_iota`` / ``psi`` are
uniform priors, and whose ``(f0, Mc, fdot_astro_ratio, alpha, sin_delta)`` block
is ``FdotAxisBirth(UniformFloorMixture(MixtureProposal([peak stack, comb])))``
-- the peak stack a :class:`StackedFStatProposal4D` or
:class:`GroupedStackedFStatProposal`, the comb a :class:`CombIntrinsicProposal`.

**Entry points.** :func:`compute_fstat` / :func:`fstat_maximized_extrinsics`
(the statistic and its maximizers); :class:`StackedFStatProposal4D`,
:class:`GroupedStackedFStatProposal`, :func:`stacked_from_cache` (the peak
grids); :class:`UniformFloorMixture`, :class:`MixtureProposal`,
:class:`CombIntrinsicProposal`, :class:`FdotAxisBirth`,
:class:`RatioTightenedBirth`, :func:`make_gb_rj_birth_container`,
:func:`reseed_birth_tree` (the container); the knob resolvers below; the GMM
pack/unpack helpers used by the warm start. :class:`FStatProposal4D` /
:class:`GridSpec` are the standalone single-box version used by the
``scripts/fstat_proposal`` tools.

**Knobs read here** (an explicit environment value always wins):

* through :func:`fstat_knob` / :data:`FSTAT_KNOB_DEFAULTS`: ``FSTAT_BATCH``
  (4096 rows per scorer call), ``FSTAT_PEAKS_PER_BAND`` (200; launchers 200),
  ``FSTAT_PEAK_HALF_MHZ`` (2.5e-3 mHz peak-box half width), ``FSTAT_MC_MIN``
  (0.01), ``FSTAT_MC_ETA`` (1.0), ``FSTAT_N_ALPHA`` / ``FSTAT_N_SINDELTA`` (8;
  setting either pins the stage-B sky), ``FSTAT_FLOOR_EPS`` (0.1),
  ``FSTAT_COMB_WEIGHT`` (0.3), and the offline-prep keys ``FSTAT_PEAK_SAMPLING``,
  ``FSTAT_FIT_GMM``, ``FSTAT_GMM_SAMPLES``, ``FSTAT_GMM_MAX_COMP``,
  ``FSTAT_PLOT_PEAKS``;
* peak floor: the in-code override (:func:`set_peak_min_F_override`, set per
  recipe stage: SNR 8 in the seed / ``gb_search_1`` stages, 6.25 in
  ``gb_search_2`` and ``full_pe``; :func:`peak_min_snr_scope` gives
  ``rj_fstat_pe`` its own 6.5 from ``GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE``), then
  ``FSTAT_PEAK_MIN_SNR`` (launchers 8.0), then ``FSTAT_PEAK_MIN_F``;
  ``FSTAT_PEAK_MIN_SNR_COARSE`` / ``FSTAT_PEAK_MIN_SNR_FINE`` (6.25) for the
  per-walker stage schedule (off in the launchers);
* node counts: ``FSTAT_N_F0``, ``FSTAT_N_MC``, ``FSTAT_N_PER_AXIS``;
* ``FSTAT_FDOT_AXIS`` (default ``"1"``; launchers 1) and ``GB_FSTAT_KEEP_NODES``
  (unset in the launchers; the recipe drops the node grids because no replace
  move is composed).

The distributions are eryn duck-typed -- ``rvs(size) -> size + (ndim,)`` and
``logpdf((n, ndim)) -> (n,)`` -- so they register directly in a
``ProbDistContainer`` under a tuple key, e.g.
``{("f0", "Mc", "alpha", "sin_delta"): proposal}``. Each owns a private
``np.random.Generator`` that :func:`reseed_birth_tree` re-derives from one
integer per rank and epoch.

Everything is ``xp``-agnostic: the array module is taken from the inputs (or
from the supplied ``gb_wdm_comp``; numpy on CPU, cupy on CUDA backends). Per the
sprint deepcopy/pickle rule no array module is stored as an attribute.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Optional, Tuple

import numpy as np

# Scalar physics shared with the in-model observable-basis proposal, so the
# grid rows and the chain-side move cannot drift apart. The column-resolution
# half of that module is deliberately NOT used here: grid rows are the
# physical 9-column waveform layout and carry no transform_container.
from .gb_observable_basis import (
    fdot_gr,
    mc_floor_for_fdot,
    r_from_fdot,
)

__all__ = [
    "GridSpec",
    "FdotAxisBirth",
    "FStatProposal4D",
    "StackedFStatProposal4D",
    "GroupedStackedFStatProposal",
    "stacked_from_cache",
    "iter_stacked_components",
    "stacked_in_cell_mode",
    "UniformFloorMixture",
    "MixtureProposal",
    "CombIntrinsicProposal",
    "ColumnPermutedProposal",
    "compute_fstat",
    "fstat_maximized_extrinsics",
    "pe_extrinsic_sigma",
    "pe_extrinsic_logpdf",
    "pe_extrinsic_rvs",
    "make_gb_rj_birth_container",
    "fit_gmm_to_stacked",
    "pack_gmm_components",
    "unpack_gmm_components",
    "build_peak_gmm",
    "FSTAT_KNOB_DEFAULTS",
    "fstat_knob",
    "fstat_peak_min_F",
    "set_peak_min_F_override",
    "peak_min_F_override",
    "peak_min_snr_scope",
    "set_stacked_keep_nodes",
    "stacked_keep_nodes",
    "fstat_peak_min_F_stages",
    "fstat_band_min_F",
    "fstat_n_f0",
    "fstat_n_mc",
    "fdot_axis_on",
    "fstat_n_axis",
]


# ======================================================================
# FSTAT_* environment knobs -- the single source of truth for defaults.
# The grid-prep script, the search runner, and the selection helpers all
# resolve knobs through here; an explicit environment value always wins.
# ======================================================================
FSTAT_KNOB_DEFAULTS = {
    "FSTAT_BATCH": 4096,           # kernel rows per get_fstat_ll_wdm call
    # SELECTION FLOOR, F = SNR^2 / 2. Raised 5.0 -> 8.0 (user ruling
    # 2026-08-17) because 5.0 is a PER-TRIAL cut with no look-elsewhere
    # correction. Under the null 2F ~ chi2_4, so F >= 12.5 has p = 5.0e-5 per
    # evaluation -- and a comb is millions of evaluations. Measured on the
    # high-f probe: 4,321 f0 nodes x 256 sky = 1,106,176 evaluations predicts
    # 56 false peaks; the fit returned 81, of which 58 sat >50 bins from the
    # ONLY real source in the window. The birth proposal was drawing from a
    # peak list that was ~98% noise, which is what populated 2-7 leaves
    # against one source. At SNR 8 (F = 32) the expected false count is
    # 5e-7 on that comb and 4e-5 on a full-band production comb -- i.e. zero
    # at any size we run, and the threshold no longer needs to scale with the
    # window. COST: peak boxes are no longer fitted for injections whose
    # node F-stat SNR is below 8; the 0.3-weight comb component still
    # proposes births there, so those sources are reachable, just not via a
    # refined peak box.
    "FSTAT_PEAK_MIN_SNR": 8.0,
    "FSTAT_PEAKS_PER_BAND": 200,   # per-sub-band peak cap (35 -> 200,
                                   #   2026-08-12: 35 truncated real-data
                                   #   bands; sig-het comb+stage-B is fast
                                   #   enough that the cap should not bind)
    "FSTAT_PEAK_HALF_MHZ": 2.5e-3, # peak-box f0 half width [mHz]
    "FSTAT_MC_MIN": 0.01,          # Mc grid-box floor
    "FSTAT_N_MC": 3,               # Mc floor; the AUTO path scales it with
                                   #   the box's fdot span (fstat_n_mc)
    # STAGE-B SKY. These two are now a FLOOR-and-override pair, not the
    # grid: setting EITHER pins the sky to a fixed n_alpha x n_sd and turns
    # the f0-adaptive sky off for the whole fit (a line is logged saying
    # so). Left unset, stage B sizes its sky per box from
    # ``fstat_gridfit.sky_nodes_required`` -- the same law stage A uses --
    # and factorizes it with ``stage_b_sky_axes``. The old fixed 8x8 was up
    # to 8x COARSER than the stage-A scan that produced its own input, and
    # spent 64 evaluations on only 44 distinct sky directions.
    "FSTAT_N_ALPHA": 8,
    "FSTAT_N_SINDELTA": 8,
    # INERT, and kept only so this table still documents the name. The comb
    # reads FSTAT_COMB_NSKY straight from os.environ (run_comb_scan), NOT
    # through fstat_knob, so this value has never been used -- and it is 2.7
    # to 85x below what the adaptive path actually picks. Routing it through
    # fstat_knob "for consistency" would silently drop the comb to 6 sky
    # points at every frequency. Set the ENV var to pin it; leave this be.
    "FSTAT_COMB_NSKY": 6,
    "FSTAT_FLOOR_EPS": 0.1,        # uniform-floor mixture weight
    "FSTAT_COMB_WEIGHT": 0.3,      # comb component weight. 2026-07 study
                                   #   (band75): unboxed-source coverage
                                   #   scales ~linearly with w while boxed
                                   #   mass drops only ~9% at 0.3 (the
                                   #   linear-in-F comb concentrates on the
                                   #   same loud sources); off-source waste
                                   #   flat ~5.5% (floor-only) at any w.
    "FSTAT_MC_ETA": 1.0,           # AUTO Mc density: node spacing in
                                   #   fdot-coherence widths 1/(pi Tobs^2)
                                   #   (see fstat_n_mc)
    "FSTAT_PEAK_SAMPLING": "grid", # production draw layer (gmm = option)
    "FSTAT_FIT_GMM": 0,            # prep-time GMM fit is OPT-IN
    "FSTAT_GMM_SAMPLES": 4096,     # GMM fit: draws per box
    "FSTAT_GMM_MAX_COMP": 12,      # GMM fit: max components per box
    "FSTAT_PLOT_PEAKS": 2,         # prep: corner plots for the top boxes
}


def fstat_knob(name: str, cast=float):
    """Resolve an ``FSTAT_*`` env knob against :data:`FSTAT_KNOB_DEFAULTS`.

    A non-empty (after ``strip``) environment value wins; otherwise the table
    default is used. Read at every call (no caching), so a knob change takes
    effect at the next fit. Used throughout this module and
    :mod:`lisatools.sampling.fstat_gridfit` (batch size, peak cap, box half
    width, floor and comb weights, ...).

    Args:
        name: Knob name; must be a key of :data:`FSTAT_KNOB_DEFAULTS`.
        cast (callable, optional): Converter applied to the raw string or the
            default (``float``, ``int``, ``str``). Default ``float``.

    Returns:
        object: ``cast(value)``.

    Raises:
        KeyError: ``name`` is unset in the environment and has no default.
        ValueError: ``cast`` rejects the value.

    Example:
        >>> from lisatools.sampling.fstat_proposal import fstat_knob
        >>> batch = fstat_knob("FSTAT_BATCH", int)
    """
    raw = os.environ.get(name, "").strip()
    if raw:
        return cast(raw)
    return cast(FSTAT_KNOB_DEFAULTS[name])


#: Live, in-code override of the peak-selection floor, in F units.
#:
#: The v9 three-stage GB search runs stage 1 at peak SNR 8 and stages 2-3 at
#: 6.25, and it CANNOT do that with an environment variable: the floor is
#: read when the grid is FITTED, the stage-B cache stamps the value it
#: selected at, and the loader refuses a mismatch (see
#: ``fstat_gridfit._check_cached_peak_threshold``) -- so changing the env at a
#: stage boundary either silently reuses the old peak list or hard-refuses.
#:
#: USER RESOLUTION (2026-09-24), in place of deleting cache files: make the
#: threshold an updateable value in code and force an F-stat refit on the
#: first call inside the new stage. A refit opens a FRESH epoch directory, so
#: the new floor is what selects its peaks and what its own stamp records --
#: the old epoch stays on disk, valid, and unread.
#:
#: ``None`` (the default) means "no override": every resolution below behaves
#: exactly as it did before this existed.
_PEAK_MIN_F_OVERRIDE: Optional[float] = None


def set_peak_min_F_override(value) -> Optional[float]:
    """Install (or clear, with ``None``) the live peak-floor override.

    ``value`` is an SNR, not an F -- the same unit the knob and every log
    line use -- and is stored as ``F = SNR^2 / 2``. Returns the PREVIOUS
    override in F units so a caller can tell whether it actually changed
    anything and restore it if it must.

    Called at stage entry by the recipe: ``SearchStageProfileStep._apply_profile``
    (the v9 profile's ``peak_min_snr``: 8 for ``gb_search_seed`` /
    ``gb_search_1``, 6.25 for ``gb_search_2``) and
    ``PERecipeStep.note_recipe_step`` (``full_pe``: 6.25, or ``None`` to clear).
    Process-global: it changes what every later :func:`fstat_peak_min_F` returns.

    Args:
        value (float or None): Peak floor as an SNR; ``None`` clears the
            override.

    Returns:
        float or None: The previous override in F units (``None`` when none was
        set).
    """
    global _PEAK_MIN_F_OVERRIDE
    prev = _PEAK_MIN_F_OVERRIDE
    _PEAK_MIN_F_OVERRIDE = (None if value is None
                            else 0.5 * float(value) ** 2)
    return prev


def peak_min_F_override() -> Optional[float]:
    """The live override in F units, or ``None`` when unset.

    Read by ``SearchStageProfileStep._apply_profile`` to decide whether a stage
    entry changed the floor (and must force an F-stat refit).

    Returns:
        float or None: ``SNR^2 / 2`` of the installed override, or ``None``.
    """
    return _PEAK_MIN_F_OVERRIDE


#: Whether a :class:`StackedFStatProposal4D` built with ``keep_nodes=None``
#: (every production path: the in-move fit, ``from_cache``,
#: ``stacked_from_cache``) RETAINS its node grids after construction.
#:
#: The node grids are read after construction by ONE thing: the "trilinear"
#: in-cell mode, which only the F-stat REPLACE moves (``rj_replace`` /
#: ``rj_replace_pe``, ``_replace_incell_mode``) switch on. The RJ birth
#: machinery draws and scores in the uniform mode from the cell log-weights
#: alone. So a run without a replace move carries the node stack on every
#: card for nothing: on the 9mo PE galaxy grid that is 24.2 GB of the ~43 GB
#: resident (the cell weights are the other ~20 GB; the CDF is on the host).
#: The recipe sets this to ``False`` when it composes no replace move
#: (:func:`set_stacked_keep_nodes`, user ruling 2026-10-12: "Make the
#: necessary changes including 1)"); ``GB_FSTAT_KEEP_NODES=1`` / ``0`` on the
#: environment overrides either way (``1`` restores the pre-2026-10-12
#: footprint; ``0`` forces the drop and makes a trilinear request raise).
#: ``True`` by default so standalone scripts and tests keep today's
#: behaviour.
_STACKED_KEEP_NODES: bool = True


def set_stacked_keep_nodes(flag) -> bool:
    """Install the process-wide default for ``keep_nodes=None`` containers.

    The recipe (:func:`lisatools.globalfit.recipe.build_gb_moves`) calls this
    once per process with "a replace move is composed" (see
    :data:`_STACKED_KEEP_NODES`); the 9mo/12mo runs compose none
    (``GB_SEARCH_RJ_REPLACE=0``, ``GB_PE_RJ_REPLACE=0``), so they pass ``False``
    and every stacked grid drops its node stack after construction.
    ``GB_FSTAT_KEEP_NODES`` still overrides the result (:func:`stacked_keep_nodes`).

    Args:
        flag (bool): Keep the node grids (``True``) or drop them (``False``).

    Returns:
        bool: The previous default.
    """
    global _STACKED_KEEP_NODES
    prev = bool(_STACKED_KEEP_NODES)
    _STACKED_KEEP_NODES = bool(flag)
    return prev


def stacked_keep_nodes() -> bool:
    """The resolved ``keep_nodes`` default for :class:`StackedFStatProposal4D`.

    ``GB_FSTAT_KEEP_NODES`` (``"1"`` / ``"0"``) when set to one of those, else
    the recipe's :func:`set_stacked_keep_nodes` value (``True`` until the recipe
    says otherwise). Read once per stacked-grid construction.

    Returns:
        bool: Whether a ``keep_nodes=None`` container retains its node grids.
    """
    env = os.environ.get("GB_FSTAT_KEEP_NODES", "").strip()
    if env in ("0", "1"):
        return env == "1"
    return bool(_STACKED_KEEP_NODES)


@contextlib.contextmanager
def peak_min_snr_scope(snr):
    """A MOVE's own peak floor for the duration of the block, then the
    previous override back, exactly (the raw F value, no SNR round trip).

    ``snr=None`` changes nothing: the block runs at whatever floor is in
    force (the stage's declared override, else ``FSTAT_PEAK_MIN_SNR``).

    Why a scope and not a second global (user ruling 2026-10-12): ``full_pe``
    carries TWO F-stat birth moves with different floors -- the GB-free
    galaxy fit at SNR 6.5 (``rj_fstat_pe``) and the live-residual fit at the
    stage's 6.25 (``rj_fstat_pe_residual``). The floor is read while a grid
    is FITTED (peak selection, the stage-B ``peak_min_F`` stamp, the forced-
    refit decline), all of it on the head inside one move's ``setup()``, so
    scoping the process-global override around that call gives each move its
    own floor while every other reader keeps seeing the stage's.

    Used by ``GBSpecialRJFStatGridMove.setup`` with the move's
    ``fstat_peak_min_snr`` (6.5 on ``rj_fstat_pe`` from
    ``GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE``; ``None`` on the other F-stat moves).

    Args:
        snr (float or None): The move's floor as an SNR; ``None`` is a no-op.

    Yields:
        None: Control, with the override set to ``snr**2 / 2`` for the block;
        the previous raw value is restored on exit, even on an exception.

    Example:
        >>> from lisatools.sampling.fstat_proposal import (
        ...     fstat_peak_min_F, peak_min_snr_scope)
        >>> with peak_min_snr_scope(6.5):
        ...     floor = fstat_peak_min_F()
        >>> floor
        21.125
    """
    global _PEAK_MIN_F_OVERRIDE
    if snr is None:
        yield
        return
    prev = _PEAK_MIN_F_OVERRIDE
    _PEAK_MIN_F_OVERRIDE = 0.5 * float(snr) ** 2
    try:
        yield
    finally:
        _PEAK_MIN_F_OVERRIDE = prev


def fstat_peak_min_F() -> float:
    """Peak-selection floor in F units (``F = SNR^2 / 2``).

    Precedence: the in-code override (:func:`set_peak_min_F_override`, the
    v9 per-stage profile) > explicit ``FSTAT_PEAK_MIN_SNR`` > explicit
    ``FSTAT_PEAK_MIN_F`` > the default SNR
    (``FSTAT_KNOB_DEFAULTS['FSTAT_PEAK_MIN_SNR']`` = 8, i.e. F = 32).

    The override wins over the environment deliberately: the env value is
    the RUN's floor, set once in the submit script, while the override is
    the STAGE's floor and the stage is the narrower, later statement. A run
    that sets no override is bit-identical to before.

    In the 9mo/12mo runs ``FSTAT_PEAK_MIN_SNR=8.0`` and the recipe overrides it
    per stage (see :func:`set_peak_min_F_override`). Read by peak selection
    (:func:`~lisatools.sampling.fstat_gridfit.select_comb_peaks`), the stage-B
    cache stamp and its load check, and the GB moves and recipe.

    Returns:
        float: The floor in F units.
    """
    if _PEAK_MIN_F_OVERRIDE is not None:
        return float(_PEAK_MIN_F_OVERRIDE)
    snr = os.environ.get("FSTAT_PEAK_MIN_SNR", "").strip()
    if snr:
        return 0.5 * float(snr) ** 2
    f = os.environ.get("FSTAT_PEAK_MIN_F", "").strip()
    if f:
        return float(f)
    return 0.5 * float(FSTAT_KNOB_DEFAULTS["FSTAT_PEAK_MIN_SNR"]) ** 2


def fstat_peak_min_F_stages() -> tuple:
    """``(coarse_F, fine_F)`` peak-selection floors, in F units.

    The F-stat half of the per-(walker, band) search-stage schedule (user
    request 2026-09-23). COARSE defaults to whatever :func:`fstat_peak_min_F`
    already resolves to, so a run that sets neither stage knob behaves
    exactly as it does today; FINE defaults to
    ``FSTAT_PEAK_MIN_SNR_FINE`` (6.25, the landed 2026-09-23 knee value).

    ⚠ THE CATALOG IS SHARED ACROSS WALKERS and always will be: the comb is
    swept ONCE against ONE reference walker's residual, so there is no
    walker axis for a peak list to have. A band's catalog floor is
    therefore the LOOSER of the floors any walker holds there, and the
    per-walker half of the schedule is carried by the opt-SNR boundary
    (``GB_OPT_SNR_LIMIT_SEARCH_{COARSE,FINE}``), which IS per row.

    That direction is deliberate under the standing recall-over-precision
    ruling: a peak a walker is not ready for costs some proposal mass and
    dies in that walker's own SNR gate, while a peak missing from the
    catalog is invisible to every walker.

    Only reached through :func:`fstat_band_min_F`, i.e. with the per-walker
    search-stage table bound (``GB_SEARCH_STAGE_PER_WALKER=1``); the 9mo/12mo
    launchers export 0, so production fits never call it.

    Returns:
        tuple[float, float]: ``(coarse_F, fine_F)``; ``coarse_F`` from
        ``FSTAT_PEAK_MIN_SNR_COARSE`` (else :func:`fstat_peak_min_F`), ``fine_F``
        from ``FSTAT_PEAK_MIN_SNR_FINE`` (default SNR 6.25, F = 19.53).

    Raises:
        ValueError: The fine floor is above the coarse floor.
    """
    coarse_env = os.environ.get("FSTAT_PEAK_MIN_SNR_COARSE", "").strip()
    coarse = (0.5 * float(coarse_env) ** 2 if coarse_env
              else fstat_peak_min_F())
    fine_env = os.environ.get("FSTAT_PEAK_MIN_SNR_FINE", "").strip()
    fine = 0.5 * float(fine_env if fine_env else 6.25) ** 2
    if fine > coarse:
        raise ValueError(
            f"FSTAT_PEAK_MIN_SNR_FINE resolves to F={fine:.3f} which is "
            f"ABOVE the coarse floor F={coarse:.3f}. The stage schedule "
            f"promotes COARSE -> FINE one way, on the premise that FINE "
            f"admits strictly MORE peaks; inverted, promoting a band would "
            f"shrink its catalog and the search would lose candidates it "
            f"had already earned the right to see."
        )
    return (coarse, fine)


def fstat_band_min_F(band_stage, num_bands: int):
    """Per-band peak floor from a ``(nwalkers, nbands)`` stage table.

    Returns a ``(num_bands,)`` array of F floors: a band gets the FINE
    floor once ANY walker has promoted there (``stage.max(axis=0)``), for
    the reason spelled out in :func:`fstat_peak_min_F_stages` -- one shared
    catalog, so it must cover the loosest floor in force anywhere.

    ``band_stage`` of ``None`` (feature off) returns ``None``, and every
    caller then keeps the scalar floor it has always used.

    Called by ``gbspecialstretch.fstat_band_min_F_for``, which passes
    ``None`` unless ``GB_SEARCH_STAGE_PER_WALKER=1`` (0 in the 9mo/12mo
    launchers).

    Args:
        band_stage (array_like or None): ``(nwalkers, nbands)`` or ``(nbands,)``
            integer stage table (0 = coarse, >= 1 = fine), or ``None``.
        num_bands: Number of sub-bands of the peak grid.

    Returns:
        np.ndarray or None: ``(num_bands,)`` float F floors, or ``None``.

    Raises:
        ValueError: The table does not cover ``num_bands`` bands, or (from
            :func:`fstat_peak_min_F_stages`) the floors are inverted.
    """
    if band_stage is None:
        return None
    coarse, fine = fstat_peak_min_F_stages()
    stage = np.asarray(band_stage)
    if stage.ndim == 2:
        stage = stage.max(axis=0)
    stage = stage.astype(int).reshape(-1)
    if stage.shape[0] != int(num_bands):
        raise ValueError(
            f"band stage table covers {stage.shape[0]} bands but the peak "
            f"grid has {int(num_bands)}. Per-peak ``band_idx`` labels index "
            f"the band grid, so a mismatched stage table would loosen the "
            f"floor in the wrong bands."
        )
    return np.where(stage >= 1, fine, coarse)


def fstat_band_skip(shut_w, num_bands: int):
    """Bands the F-stat fit may SKIP, from a ``(nwalkers, nbands)`` valve.

    Returns a ``(num_bands,)`` bool array, True where the level-3 valve
    is shut on EVERY walker -- the mirror of :func:`fstat_band_min_F`'s
    ``max(axis=0)``, and for the same reason. There is ONE peak catalog
    shared by every walker, so a band may only be skipped when NO walker
    could still use a peak there; ``any`` would starve the walkers whose
    pair is still open.

    ``shut_w`` of ``None`` (valve off, or a move that never bound the
    table) returns ``None``, and the fit scans everything as before.

    This is the PER-WALKER half of the skip set only. The fit's caller,
    ``gbspecialstretch.fstat_band_skip_for``, ORs in the per-band
    empty-band valve (births OFF there on every walker by construction)
    and applies the empty-catalog guard to the union. Live in the 9mo/12mo
    search stages (``GB_SEARCH_BAND_SHUTOFF_PER_WALKER=1`` with the all-walker
    rule); PE fits pass no valve.

    Args:
        shut_w (array_like or None): ``(nwalkers, nbands)`` or ``(nbands,)`` bool
            valve table (True = shut), or ``None``.
        num_bands: Number of sub-bands of the peak grid.

    Returns:
        np.ndarray or None: ``(num_bands,)`` bool, True where every walker is
        shut; ``None`` when ``shut_w`` is ``None``.

    Raises:
        ValueError: The table does not cover ``num_bands`` bands.
    """
    if shut_w is None:
        return None
    shut = np.asarray(shut_w, dtype=bool)
    if shut.ndim == 2:
        shut = shut.all(axis=0)
    shut = shut.reshape(-1)
    if shut.shape[0] != int(num_bands):
        raise ValueError(
            f"band shutoff table covers {shut.shape[0]} bands but the peak "
            f"grid has {int(num_bands)}. Per-peak ``band_idx`` labels index "
            f"the band grid, so a mismatched table would skip the WRONG "
            f"bands -- silently, and the symptom would be missing sources."
        )
    return shut


def fstat_n_f0(box_width_mHz: float, Tobs_s: float) -> int:
    """f0 node count for a peak box.

    Precedence: explicit ``FSTAT_N_F0`` > explicit ``FSTAT_N_PER_AXIS`` >
    AUTO -- one cell per ~1/Tobs (the matched-filter peak width), clamped to
    ``[12, 96]`` (~40 for the default +-2.5e-3 mHz box at 90 d). The f0
    cell width is the proposal's sharpest efficiency lever: cells much wider
    than the peak spread birth mass off-source. Neither knob is set by the
    9mo/12mo launchers, so the AUTO rule applies there; with the default box it
    hits the 96 clamp beyond ~7 months of data, i.e. on both runs. Called once
    per stage-B fit by
    :func:`~lisatools.sampling.fstat_gridfit.run_stacked_stage_b` with
    ``box_width_mHz = 2 * FSTAT_PEAK_HALF_MHZ``.

    Args:
        box_width_mHz: Full f0 width of a peak box (mHz).
        Tobs_s: Observation span (s).

    Returns:
        int: ``clip(round(box_width_mHz / (1e3 / Tobs_s)) + 1, 12, 96)`` under
        AUTO, else the explicit knob value.
    """
    raw = os.environ.get("FSTAT_N_F0", "").strip()
    if raw:
        return int(raw)
    per = os.environ.get("FSTAT_N_PER_AXIS", "").strip()
    if per:
        return int(per)
    cells = round(float(box_width_mHz) / (1e3 / float(Tobs_s)))
    return int(np.clip(cells + 1, 12, 96))


_MSUN_S = 4.925490947641267e-6  # GM_sun / c^3 [s]


def _fdot_gr(mc, f0_hz):
    """GR chirp fdot [Hz/s] at chirp mass ``mc`` [Msun], ``f0_hz`` [Hz].

    Scalar helper of :func:`fstat_n_mc` (note the ``(mc, f0)`` argument order,
    the reverse of :func:`lisatools.sampling.gb_observable_basis.fdot_gr`). Uses
    the hard-coded ``_MSUN_S = GM_sun / c^3`` above rather than lisaconstants.
    """
    return (
        (96.0 / 5.0) * np.pi ** (8.0 / 3.0)
        * (_MSUN_S * float(mc)) ** (5.0 / 3.0)
        * float(f0_hz) ** (11.0 / 3.0)
    )


def fdot_axis_on() -> bool:
    """``FSTAT_FDOT_AXIS`` -- fdot as a first-class grid axis. Default ON.

    ONE reader for a flag with four consumers (stage-B assembly, the cache
    loader's basis guard, the birth container's key, and the uniform
    floor's box). They must agree: axis 2 is a chirp mass in one basis and
    Hz/s in the other, and any pair of them disagreeing produces births at
    absurd parameters with no error anywhere. A single helper is cheaper
    than four literals that can drift.

    Default flipped to ON 2026-09-01. Set ``FSTAT_FDOT_AXIS=0`` to restore
    the r = 0 grid bit-identically -- but note that an existing
    ``*_peaks_stacked.npz`` fitted in the other basis is REFUSED on load
    (by design), so flipping this on an in-flight run means refitting.
    The 9mo/12mo launchers export ``FSTAT_FDOT_AXIS=1``.

    Returns:
        bool: ``True`` unless ``FSTAT_FDOT_AXIS`` (stripped) is something other
        than ``"1"``.
    """
    return os.environ.get("FSTAT_FDOT_AXIS", "1").strip() == "1"


def fstat_n_mc(f0_mHz: float, mc_lo: float, mc_hi: float,
               Tobs_s: float) -> int:
    """AUTO Mc node count for a peak box (user rule 2026-08-26).

    Precedence: explicit ``FSTAT_N_MC`` > explicit ``FSTAT_N_PER_AXIS`` >
    AUTO -- one node per ``FSTAT_MC_ETA`` (default 1.0) fdot-coherence
    widths ``1/(pi * Tobs^2)`` across the box's GR-fdot span at the box's
    f0, clamped to ``[3, 96]``::

        n = span / (eta / (pi Tobs^2)) + 1,
        span = fdot_gr(mc_hi, f0) - fdot_gr(mc_lo, f0)

    The span scales as ``f0**(11/3)``, which is why the band75 study's 3
    nodes (7.5 mHz, 90 d: span ~2 widths -- fdot per-band unmeasurable)
    and the 20.4 mHz flagship's ~70 (span ~69 widths) are the SAME
    criterion evaluated at different frequencies; the old fixed default
    of 3 under-resolved high f, capping replace/birth candidate match at
    ~0.6 against the flagship's 0.87-match ridge mode. Cost is linear in
    n (stage B only); the 1-yr clamp at 96 is a documented limitation --
    the comb's Mc-fixed detection stage decoheres there anyway and needs
    its own design (fdot-resolved or warm-started comb) first.

    Legacy Mc-axis path only: :func:`~lisatools.sampling.fstat_gridfit.run_stacked_stage_b`
    calls it per box when ``FSTAT_FDOT_AXIS=0``; the production fdot axis sizes
    its nodes with :func:`lisatools.sampling.gb_observable_basis.n_fdot_nodes`
    instead.

    Args:
        f0_mHz: Box frequency (mHz).
        mc_lo: Lower Mc of the grid box (Msun).
        mc_hi: Upper Mc of the grid box (Msun).
        Tobs_s: Observation span (s).

    Returns:
        int: Mc node count.
    """
    raw = os.environ.get("FSTAT_N_MC", "").strip()
    if raw:
        return int(raw)
    per = os.environ.get("FSTAT_N_PER_AXIS", "").strip()
    if per:
        return int(per)
    eta = fstat_knob("FSTAT_MC_ETA", float)
    f0_hz = float(f0_mHz) * 1e-3
    span = _fdot_gr(mc_hi, f0_hz) - _fdot_gr(mc_lo, f0_hz)
    width = eta / (np.pi * float(Tobs_s) ** 2)
    return int(np.clip(round(span / width) + 1, 3, 96))


def fstat_n_axis(name: str) -> int:
    """Node count for the Mc / alpha / sin_delta axes.

    Precedence: explicit ``FSTAT_N_MC``/``_ALPHA``/``_SINDELTA`` > explicit
    ``FSTAT_N_PER_AXIS`` > the anisotropic defaults (3 / 8 / 8).
    (For the Mc axis prefer :func:`fstat_n_mc`, which is AUTO by default;
    this fixed-default path remains for alpha / sin_delta and legacy
    callers.)

    For the SKY axes in stage B this is now an OVERRIDE, not the grid: if
    any of the three env vars above is set, ``run_stacked_stage_b`` takes it
    as an instruction to pin the sky and skips the f0-adaptive sizing
    entirely. Reading the return value does not by itself mean the adaptive
    path is off -- the caller checks the environment for that. The 9mo/12mo
    launchers set none of them (adaptive stage-B sky, ``FSTAT_STAGEB_SKY_ADAPT=1``).

    Args:
        name: ``"FSTAT_N_MC"``, ``"FSTAT_N_ALPHA"`` or ``"FSTAT_N_SINDELTA"``
            (any key of :data:`FSTAT_KNOB_DEFAULTS` works).

    Returns:
        int: The node count.
    """
    raw = os.environ.get(name, "").strip()
    if raw:
        return int(raw)
    per = os.environ.get("FSTAT_N_PER_AXIS", "").strip()
    if per:
        return int(per)
    return int(FSTAT_KNOB_DEFAULTS[name])


def _host(x):
    """cupy/numpy array -> host numpy (no-op on numpy).

    Also imported by :mod:`lisatools.sampling.fstat_gridfit` as ``_to_host``.

    Args:
        x (array_like): A cupy array (``.get()`` is called) or anything
            ``np.asarray`` accepts.

    Returns:
        np.ndarray: Host array.
    """
    return x.get() if hasattr(x, "get") else np.asarray(x)


# ======================================================================
# RJ birth-container reseeding
# ======================================================================
# Every proposal class below builds ``np.random.default_rng(seed)`` at
# construction, and the RJ birth assembly
# (``fstat_gridfit.build_gb_birth_distribution``) constructs all of them with
# ``seed=None`` -- i.e. OS entropy. Under the multi-rank fan-out that makes the
# birth stream irreproducible even for a legacy-vs-legacy rerun of the same run
# seed. ``reseed(seed)`` re-derives the whole tree's generators from ONE
# integer AFTER construction, so:
#
# * ``general.random_seed`` set   -> births are seed-determined per rank and
#                                    per F-stat epoch (the move derives the
#                                    integer, see ``_birth_seed``);
# * ``general.random_seed`` unset -> ``seed is None``, every ``reseed`` is a
#                                    no-op and every stream stays on entropy,
#                                    exactly as before.
#
# Each node gets a DISTINCT generator: the node's own ``_rng`` comes from
# ``SeedSequence(seed)`` and its children from that sequence's ``spawn``
# children, in the fixed attribute order each class documents.


def _child_seeds(ss, n: int):
    """``n`` independent integer seeds spawned off ``ss`` (a SeedSequence).

    Args:
        ss (np.random.SeedSequence): Parent sequence (``spawn`` advances it).
        n: Number of children.

    Returns:
        list[int]: One uint32-derived integer seed per child.
    """
    return [int(kid.generate_state(1, dtype=np.uint32)[0])
            for kid in ss.spawn(int(n))]


def _reseed_node(node, seed):
    """Reseed one child node when it owns a private ``Generator``.

    Nodes with no ``reseed`` are left alone ON PURPOSE: eryn's analytic
    distributions (``UniformDistribution``, ...) draw from the MODULE-level
    ``np.random`` state, which the run seeds per rank in
    ``run.py::_seed_rank_streams`` -- there is no per-object stream to set.

    Anything exposing a ``priors_in`` MAPPING is recursed into first
    (insertion order, which is the container's construction order): an eryn
    ``ProbDistContainer``, and also
    :class:`~lisatools.sampling.prior.FullGaussianMixtureModel`, whose
    ``priors_in`` is a *property* forwarding to its ``base_prior`` -- so that
    class is WALKED, not skipped. It is not in the GB RJ birth tree today, so
    nothing in production depends on either reading.

    The skip is NOT unconditional: a node that owns a private ``_rng`` or
    carries a wrapped child (``base`` / ``components`` / ``grid4``) but
    defines no ``reseed`` raises :class:`TypeError` here, exactly as a bad
    ROOT does in :func:`reseed_birth_tree`. Without that a new wrapper
    inserted INSIDE the tree would leave its whole subtree on OS entropy with
    no error anywhere -- the failure mode this helper exists to prevent. None
    of the eryn analytic leaves define any of those four attributes, and
    ``FullGaussianMixtureModel`` returns through the ``priors_in`` branch
    before the check is reached (its wrapped child is ``base_prior``, not
    ``base``), so the legitimate skips are unaffected.

    Args:
        node (object or None): The node; ``None`` is a no-op.
        seed (int or None): Integer seed; ``None`` is a no-op.

    Raises:
        TypeError: ``node`` owns ``_rng`` / ``base`` / ``components`` / ``grid4``
            but has no ``reseed`` method.
    """
    if node is None or seed is None:
        return
    fn = getattr(node, "reseed", None)
    if callable(fn):
        fn(int(seed))
        return
    priors_in = getattr(node, "priors_in", None)
    if isinstance(priors_in, dict) and priors_in:
        kids = _child_seeds(np.random.SeedSequence(int(seed)), len(priors_in))
        for child, kid in zip(priors_in.values(), kids):
            _reseed_node(child, kid)
        return
    owned = [name for name in ("_rng", "base", "components", "grid4")
             if getattr(node, name, None) is not None]
    if owned:
        raise TypeError(
            f"{type(node).__name__} sits inside an RJ birth tree, owns "
            f"{', '.join(owned)} and defines no reseed(seed): skipping it "
            "would leave that subtree on OS entropy. Give the class a "
            "reseed(seed) method (see reseed_birth_tree)."
        )


def reseed_birth_tree(obj, seed):
    """Reseed a whole RJ birth container from one integer; returns ``obj``.

    The strict entry point: ``obj`` must be either a proposal class of this
    module (any ``reseed`` owner) or an eryn ``ProbDistContainer``. Anything
    else raises :class:`TypeError` naming the class -- a new wrapper class
    that forgot ``reseed`` then fails loudly in the birth-seed unit test
    instead of silently leaving a birth stream on entropy in a production run.

    ``seed=None`` is a no-op (the type check still runs). Called once per
    container by
    :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution` with
    the move's per-rank, per-epoch ``_birth_seed(k)`` (``None`` when
    ``general.random_seed`` is unset).

    Args:
        obj (object): The container root (a ``reseed`` owner or a
            ``ProbDistContainer``).
        seed (int or None): One integer for the whole tree.

    Returns:
        object: ``obj`` itself (reseeded in place).

    Raises:
        TypeError: ``obj`` (or a node inside it) cannot be reseeded.
    """
    if not (callable(getattr(obj, "reseed", None))
            or isinstance(getattr(obj, "priors_in", None), dict)):
        raise TypeError(
            f"reseed_birth_tree: {type(obj).__name__} has no reseed() and is "
            "not a ProbDistContainer, so its RNG stream cannot be made "
            "reproducible. Give the class a reseed(seed) method (see "
            "fstat_proposal's proposal classes) before putting it in the RJ "
            "birth container."
        )
    _reseed_node(obj, seed)
    return obj


# Row-major upper-triangle layout of the symmetric (4, 4) filter Gram matrix
# as returned by ``get_fstat_ll_wdm``: [M00, M01, M02, M03, M11, M12, M13,
# M22, M23, M33].
_TRIU_ROWS = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
_TRIU_COLS = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)


def compute_fstat(N_arr, M_upper, ridge: float = 1e-12):
    r"""``F = 0.5 * N^T M^{-1} N`` from the batched ``(N, M)`` F-stat pieces.

    .. math::

        \mathcal{F} = \tfrac{1}{2}\, N^{\mathsf T}
            \left(M + \epsilon\,\bar m\,\mathbb{1}\right)^{-1} N,
        \qquad \bar m = \max\!\left(\tfrac{1}{4}\textstyle\sum_i |M_{ii}|,
            10^{-300}\right),

    with ``epsilon = ridge``, solved per binary with ``xp.linalg.solve``. This is
    the scorer of every comb, stage-B and center sweep in
    :mod:`lisatools.sampling.fstat_gridfit` (and of the para-search tool).

    Args:
        N_arr (xp.ndarray): ``(num_bin, 4)`` data-filter inner products
            ``<d | A_i>`` (1-D input is promoted to one row).
        M_upper (xp.ndarray): ``(num_bin, 10)`` row-major upper triangle of the
            symmetric filter Gram matrix ``<A_i | A_j>``, ordered
            ``[M00, M01, M02, M03, M11, M12, M13, M22, M23, M33]``.
        ridge: Fractional Tikhonov regularization added to the diagonal
            (scaled by the mean diagonal per binary) so grid cells with a
            near-singular Gram matrix (e.g. dead sky/frequency corners)
            return a finite, tiny F instead of raising. Default ``1e-12``.

    Returns:
        xp.ndarray: ``(num_bin,)`` float64 F-statistic values (same array module
        as the inputs; non-finite results are mapped to ``-inf`` so downstream
        ``exp(beta * F)`` weights vanish cleanly).

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.fstat_proposal import compute_fstat
        >>> N = np.array([[1.0, 0.0, 0.0, 0.0]])
        >>> M = np.array([[2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0]])
        >>> float(np.round(compute_fstat(N, M)[0], 6))
        0.25
    """
    from ..utils.utility import get_array_module

    xp = get_array_module(N_arr)
    N_arr = xp.atleast_2d(xp.asarray(N_arr))
    M_upper = xp.atleast_2d(xp.asarray(M_upper))
    num_bin = N_arr.shape[0]

    M4 = xp.empty((num_bin, 4, 4), dtype=xp.float64)
    for k, (i, j) in enumerate(zip(_TRIU_ROWS, _TRIU_COLS)):
        M4[:, i, j] = M_upper[:, k]
        M4[:, j, i] = M_upper[:, k]

    # Per-binary diagonal ridge keeps xp.linalg.solve away from exactly
    # singular Gram matrices without perturbing healthy ones.
    diag_scale = xp.clip(
        xp.mean(xp.abs(M4[:, (0, 1, 2, 3), (0, 1, 2, 3)]), axis=-1), 1e-300, None
    )
    M4 = M4 + (ridge * diag_scale)[:, None, None] * xp.eye(4)[None]

    sol = xp.linalg.solve(M4, N_arr[..., None])[..., 0]
    F = 0.5 * xp.sum(N_arr * sol, axis=-1)
    return xp.where(xp.isfinite(F), F, -xp.inf)


def fstat_maximized_extrinsics(N_arr, M_upper, ridge: float = 1e-12):
    r"""``(A_max, phi0_max, iota_max, psi_max, F)`` from the F-stat ``(N, M)``.

    Ports the Jaranowski-Krol amplitude-vector inversion from
    ``gbgpu.GBGPU.get_fstat_ll`` to the ``(N, M)`` pieces returned by the new
    ``GBFDComputations.get_fstat_ll_fd`` / ``GBWDMComputations.get_fstat_ll_wdm``
    (and the sig-het ``GBSignalHetComputations.get_fstat_ll_wdm``)
    -- which build the SAME 4 basis filters at the SAME fixed reference
    ``(A, iota, psi, phi0) = (2, pi/2, {0,pi/4,0,pi/4}, {0,pi,3pi/2,pi/2})`` and
    return the SAME 10-element upper-triangle ``M`` layout, so the inversion is
    identical. ``a = M^-1 N`` is the ML amplitude vector; ``A_max`` is the
    physical maximized amplitude (reference A=2 baked into the filters, matching
    the legacy formula) and ``F = 0.5 * a . N``. With ``a = (a1, a2, a3, a4)``:

    .. math::

        r_{1,2} &= \sqrt{(a_1 \mp a_4)^2 + (a_2 \pm a_3)^2}, \qquad
        A_+ = r_1 + r_2, \quad A_\times = r_1 - r_2, \\
        A_{\max} &= \tfrac{1}{2}\left(A_+ + \sqrt{A_+^2 - A_\times^2}\right), \qquad
        \iota_{\max} = \arccos\!\frac{A_\times}{A_+ + \sqrt{A_+^2 - A_\times^2}}
            \bmod \pi, \\
        \alpha &= \operatorname{atan2}(-(a_2 + a_3),\, a_1 - a_4), \qquad
        \beta = \operatorname{atan2}(a_2 - a_3,\, a_1 + a_4), \\
        \phi_{0,\max} &= -\tfrac{1}{2}(\alpha + \beta) \bmod 2\pi, \qquad
        \psi_{\max} = \tfrac{1}{4}(\alpha - \beta) \bmod \pi.

    Here ``alpha`` and ``beta`` are auxiliary angles, not sky coordinates.
    Numerically guarded (ridge on ``M``, non-negative discriminant, clipped
    ``arccos``) so near-singular / off-peak births return finite maxima instead
    of NaN. Same array module as the inputs.

    2026-08-24: ``A_max`` verified EXACT (ratio 1.000 to the injected
    amplitude on a noise-free truth residual through the GBGPU CPU basis
    filters), retiring the old "known adjustment" TODO here; the actual
    long-standing defect was the psi/phi0 angle convention, fixed below
    (see the inline calibration note). ``phi0_max`` is the sampling-basis
    ``phi0`` and ``psi_max`` the sampling-basis ``psi``.

    Callers: :func:`~lisatools.sampling.fstat_gridfit.run_center_sweep` (the
    per-epoch center table, built every epoch in the 9mo/12mo runs under
    ``GB_FSTAT_CTR_MODE=epoch``), ``GBSpecialBase._fstat_dist_centers`` (the
    F-stat distance birth / replace path) and the para-search tool. The table's
    entries are consumed only by the distance birth (``GB_RJ_FSTAT_DIST_BIRTH=0``
    in the launchers) and the replace moves (not composed), so in production
    the table is built but none of these outputs reaches a birth.

    Args:
        N_arr (xp.ndarray): ``(num_bin, 4)`` data-filter inner products.
        M_upper (xp.ndarray): ``(num_bin, 10)`` upper-triangle Gram matrix (the
            :func:`compute_fstat` layout).
        ridge: Fractional diagonal ridge, as in :func:`compute_fstat`. Default
            ``1e-12``.

    Returns:
        tuple: ``(A_max, phi0_max, iota_max, psi_max, F)``, each a
        ``(num_bin,)`` float64 array on the input module: ``A_max`` the
        maximized strain amplitude (0 where non-finite), ``phi0_max`` in
        ``[0, 2 pi)``, ``iota_max`` in ``[0, pi)`` (see the warning),
        ``psi_max`` in ``[0, pi)``, and ``F`` (``-inf`` where non-finite).

    Warning:
        DORMANT cos(iota) SIGN DEFECT (found 2026-10-16; left unfixed by
        ruling). The returned ``iota_max`` is ``pi - iota`` of the source on
        BOTH F-stat paths (the sig-het shared-reference scorer and the chunked
        ``get_fstat_ll_wdm``): ``cos(iota_max) = -cos(iota)`` in the GB
        sampling convention, so a consumer that pins or centers
        ``cos_iota = cos(iota_max)`` gets the wrong sign (28-60 nats lost at
        ``|cos iota|`` 0.3-0.6 on SNR-10 noise-free injections), while
        ``-cos(iota_max)`` reproduces ``F`` on all four scorers. ``A_max``,
        ``phi0_max``, ``psi_max`` and ``F`` are correct. The affected
        consumers -- the F-stat distance birth's pin and the PE extrinsic draw
        centres (:func:`pe_extrinsic_rvs` / :func:`pe_extrinsic_logpdf` use
        ``cos(iota_c % pi)``) and the replace pins -- are all off in the
        9mo/12mo runs (``GB_RJ_FSTAT_DIST_BIRTH=0``, no replace move); the
        center table stores the uncorrected ``iota``. Repro:
        ``scripts/diagnostics/xcheck_fstat_cos_iota.py``.
    """
    from ..utils.utility import get_array_module

    xp = get_array_module(N_arr)
    N_arr = xp.atleast_2d(xp.asarray(N_arr))
    M_upper = xp.atleast_2d(xp.asarray(M_upper))
    num_bin = N_arr.shape[0]

    M4 = xp.empty((num_bin, 4, 4), dtype=xp.float64)
    for k, (i, j) in enumerate(zip(_TRIU_ROWS, _TRIU_COLS)):
        M4[:, i, j] = M_upper[:, k]
        M4[:, j, i] = M_upper[:, k]
    diag_scale = xp.clip(
        xp.mean(xp.abs(M4[:, (0, 1, 2, 3), (0, 1, 2, 3)]), axis=-1), 1e-300, None
    )
    M4 = M4 + (ridge * diag_scale)[:, None, None] * xp.eye(4)[None]

    a = xp.linalg.solve(M4, N_arr[..., None])[..., 0]
    F = 0.5 * xp.sum(N_arr * a, axis=-1)
    a1, a2, a3, a4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    # 2026-08-24 psi/phi0 CONVENTION FIX (rj_replace candidate-quality
    # forensics). Calibrated against the actual GBGPU basis filters on a
    # noise-free truth residual: the ML amplitude vector ``a`` relates to
    # the textbook JKS map through ``a = 2 * (t1, -t2, t3, -t4)`` — the
    # two psi = pi/4 filters carry the OPPOSITE sign to what the legacy
    # angle formulas assumed. Working the textbook identities through
    # that basis:
    #
    #   r1 = A+ + Ax  = sqrt((a1 - a4)^2 + (a2 + a3)^2)
    #   r2 = A+ - Ax  = sqrt((a1 + a4)^2 + (a2 - a3)^2)
    #   alpha = atan2(-(a2 + a3), a1 - a4)
    #   beta  = atan2(  a2 - a3,  a1 + a4)
    #
    # with the GBGPU carrier-sign convention giving
    # phi0 = -(alpha + beta) / 2 and psi = (alpha - beta) / 4 (the sign
    # calibrated by concrete-template score: -(...)/2 recovers the FULL
    # truth delta bit-exactly, +(...)/2 leaves an O(0.6 rad) carrier
    # error). The branch pair is automatically consistent through the
    # physical (phi0 + pi, psi + pi/2) identity, and psi may be reduced
    # mod pi independently (2 psi + 2 pi is the same polarization). ``A_max``,
    # ``iota_max`` and ``F`` are BIT-IDENTICAL to the legacy formulas
    # (r1 + r2 and the arccos argument are invariant under the basis
    # sign; the legacy A_max/iota always verified against truth) — only
    # psi_max/phi0_max change. The legacy formulas returned a
    # concrete-template carrier phase off by an O(1)-rad,
    # source-dependent angle, which cost ``cos(err)`` of every pinned
    # candidate's <r|h> (root cause of rj_replace's delta shortfall even
    # at exact intrinsics; verified: corrected pins recover the full
    # truth delta, match 1.0000).
    #
    # 2026-10-16 CORRECTION: ``iota_max`` is NOT right in the sampling
    # convention -- it comes out as ``pi - iota`` (cos flipped) on both F-stat
    # paths; see the docstring Warning and
    # scripts/diagnostics/xcheck_fstat_cos_iota.py. Dormant in production, left
    # unfixed by ruling.
    r1 = xp.sqrt((a1 - a4) ** 2 + (a2 + a3) ** 2)
    r2 = xp.sqrt((a1 + a4) ** 2 + (a2 - a3) ** 2)
    A_plus = r1 + r2       # = 2 A+  (legacy scale convention retained)
    A_cross = r1 - r2      # = 2 Ax
    disc = xp.sqrt(xp.clip(A_plus ** 2 - A_cross ** 2, 0.0, None))

    A_max = (A_plus + disc) / 2.0
    iota_max = xp.arccos(
        xp.clip(A_cross / xp.clip(A_plus + disc, 1e-300, None), -1.0, 1.0)) % np.pi
    alpha = xp.arctan2(-(a2 + a3), a1 - a4)
    beta = xp.arctan2(a2 - a3, a1 + a4)
    phi0_max = (-(alpha + beta) / 2.0) % (2.0 * np.pi)
    psi_max = ((alpha - beta) / 4.0) % np.pi

    A_max = xp.where(xp.isfinite(A_max), A_max, 0.0)
    F = xp.where(xp.isfinite(F), F, -xp.inf)
    return A_max, phi0_max, iota_max, psi_max, F


# ======================================================================
# PE-mode extrinsic proposal (user design ruling 2026-08-25)
# ======================================================================
#
# The GB RJ F-stat distance-birth path historically PINNED (phi0,
# cos_iota, psi) at the F-stat maximizers and charged them as uniform
# constants ("uniform wash"). The SEARCH stages (rj_fstat_search /
# rj_prior_removal / rj_replace) keep that convention bit-identically.
# The PE stages (rj_fstat_pe / rj_prior_pe, when the F-stat distance-
# birth path is active there) instead DRAW each extrinsic from a genuine
# distribution centered on its maximizer and charge the real forward AND
# reverse densities in the RJ factors, so the pair is directly reversible
# (exact detailed balance):
#
#   phi0     ~ von Mises on the circle (period 2 pi), center phi0_max;
#   psi      ~ von Mises on the DOUBLED angle (the von Mises lives on
#              2 psi, so the period-pi wrap is correct), center psi_max;
#   cos_iota ~ truncated Gaussian on [-1, 1], center cos(iota_max).
#
# WIDTHS come from the F-stat curvature through the same 1/SNR scaling
# the slot-0 lognormal uses: ``sigma = geom / snr`` with ``snr =
# exp(ln_snr) = sqrt(max(2 F, 1))`` — the ``max(., 1)`` clip is the floor
# that keeps weak-F rows on a BROAD (sigma = geom rad), never degenerate,
# proposal (the ``_dist_center_and_width`` pattern). ``geom`` is ONE
# shared O(1) geometric factor, default 2.0: the diagonal Fisher widths
# of a quasi-monochromatic GB are ~1/snr in the carrier phase and the
# doubled polarization angle and ~2/snr in cos iota, and near face-on
# the phi0/psi degeneracy widens the marginals well past the
# conditionals — one conservative 2x inflation on all three coordinates
# covers that without a per-coordinate table (a genuinely per-row 3x3
# curvature projection of the Gram matrix through the Jaranowski-Krol
# Jacobian was considered and NOT done: the default epoch center table
# stores only ``ln_snr``, not ``M``, so the closed form could not feed
# the production death side). ``sigma`` is clipped below at 1e-3 rad as
# a pure numerical guard on the von Mises concentration
# ``kappa = 1/sigma^2 <= 1e6``.
#
# MEASURED (2026-08-25, flagship 20.38 mHz truth band, SNR 63.9,
# noise-free e2e, 200 draws): at geom = 1 the concentrated draws lose a
# median 1.20 in delta vs the pinned maximizers — exactly the
# ideal-matched-proposal bound (median of 0.5 chi^2_3 = 1.18), i.e. the
# local curvature per coordinate IS 1/snr there; at geom = 2 the median
# loss is 4.85 (the predicted 4x). The default stays at the conservative
# 2.0 because the production birth centers come from the epoch table's
# nearest-f0-node lookup (extrinsics can sit off the row's own maximum),
# where the extra width buys coverage; with exact per-row centers,
# GB_PE_EXTRINSIC_SIGMA_GEOM=1 is the efficiency-optimal setting.
#
# EPS-MIXTURE: each component is ``(1 - eps) * concentrated + eps *
# Uniform(domain)`` (the ``UniformFloorMixture`` device), so a polished
# or drifted leaf pays a BOUNDED reverse bill (~ ``log eps - log
# domain``), never ``-inf``.
#
# THE (phi0 + pi, psi + pi/2) IDENTITY: the F-stat maximum is defined
# only up to this joint shift, so a density treating phi0 and psi as
# independent circles would not be well-defined on the identified space.
# The (phi0, psi) proposal therefore SUMS its mixture over the two
# representatives:
#
#   g(phi0, psi | c) = 1/2 * sum_{b in {0, 1}}
#       g_phi(phi0 | c_phi + b pi) * g_psi(psi | c_psi + b pi/2)
#
# which is invariant under the identity applied to EITHER the evaluated
# point or the center (the von Mises branch terms swap; the uniform
# floor terms are shift-invariant). ``pe_extrinsic_rvs`` draws the
# branch b with probability 1/2 and then draws each component from its
# in-branch eps-mixture — exactly this density. cos_iota does not
# participate in the identity and factors out.


def _pe_std_norm_cdf(x, xp):
    """Standard normal CDF on ``xp`` arrays (scipy/cupyx erf)."""
    if xp is np:
        from scipy.special import erf
    else:
        from cupyx.scipy.special import erf
    return 0.5 * (1.0 + erf(xp.asarray(x) / np.sqrt(2.0)))


def _pe_std_norm_ppf(p, xp):
    """Standard normal inverse CDF on ``xp`` arrays (scipy/cupyx erfinv)."""
    if xp is np:
        from scipy.special import erfinv
    else:
        from cupyx.scipy.special import erfinv
    return np.sqrt(2.0) * erfinv(2.0 * xp.asarray(p) - 1.0)


def _pe_log_i0(x, xp):
    """``log I0(x)`` through the exponentially scaled Bessel ``i0e``
    (``I0(x) = exp(|x|) i0e(x)``), stable at the large concentrations a
    loud source's ``kappa = 1/sigma^2`` reaches."""
    if xp is np:
        from scipy.special import i0e
    else:
        from cupyx.scipy.special import i0e
    x = xp.asarray(x)
    return xp.log(i0e(x)) + xp.abs(x)


# TODO check: keep/remove/document -- only reachable through the F-stat distance birth / replace centre path (GB_RJ_FSTAT_DIST_BIRTH=1), off in the 9mo/12mo runs and by default
def pe_extrinsic_sigma(ln_snr, geom: float = 2.0):
    """Angle-coordinate proposal width from the F-stat curvature.

    ``sigma = geom / snr`` with ``snr = exp(ln_snr) = sqrt(max(2 F, 1))``
    (see the section comment above for the geometric factor and the
    weak-F broad floor; the 1e-3 rad lower clip is a numerical guard on
    ``kappa = 1/sigma^2``). Applied identically to the phi0 circle, the
    DOUBLED psi angle, and the cos-iota coordinate.
    """
    from ..utils.utility import get_array_module

    xp = get_array_module(ln_snr)
    return xp.clip(float(geom) * xp.exp(-xp.asarray(ln_snr)), 1e-3, None)


def _pe_logaddexp(a, b, xp):
    """Row-wise ``log(exp(a) + exp(b))`` with ``-inf`` propagation."""
    m = xp.maximum(a, b)
    m_safe = xp.where(xp.isfinite(m), m, 0.0)
    out = m_safe + xp.log(xp.exp(a - m_safe) + xp.exp(b - m_safe))
    return xp.where(xp.isfinite(m), out, -xp.inf)


# TODO check: keep/remove/document -- only caller GBSpecialBase._pe_extr_logg (via _pe_or_pin_extrinsics: the F-stat distance birth, GB_RJ_FSTAT_DIST_BIRTH=1, and the PE replace step); both off in the 9mo/12mo runs
def pe_extrinsic_logpdf(phi0, cos_iota, psi, phi0_c, iota_c, psi_c, ln_snr,
                        eps: float = 0.05, geom: float = 2.0):
    """Joint log density of the PE-mode extrinsic proposal.

    Exactly the density :func:`pe_extrinsic_rvs` draws from — the
    identity-summed (phi0, psi) mixture times the eps-floored truncated
    Gaussian in cos iota (see the section comment). Evaluates the
    forward side on drawn births and the reverse side on death rows
    (each around that row's OWN maximizers). All inputs are per-row
    arrays on one module; ``iota_c`` is the maximizer ANGLE (the center
    used is ``cos(iota_c % pi)``, mirroring the pin convention).
    Returns ``(n,)`` log densities.
    """
    from ..utils.utility import get_array_module

    xp = get_array_module(phi0)
    phi0 = xp.asarray(phi0, dtype=xp.float64)
    cos_iota = xp.asarray(cos_iota, dtype=xp.float64)
    psi = xp.asarray(psi, dtype=xp.float64)
    phi0_c = xp.asarray(phi0_c, dtype=xp.float64)
    psi_c = xp.asarray(psi_c, dtype=xp.float64)
    ci_c = xp.cos(xp.asarray(iota_c, dtype=xp.float64) % np.pi)
    sigma = pe_extrinsic_sigma(ln_snr, geom=geom)
    kappa = 1.0 / sigma**2
    log_i0 = _pe_log_i0(kappa, xp)
    eps = float(eps)
    log_eps = float(np.log(eps)) if eps > 0.0 else -np.inf
    log_1meps = float(np.log1p(-eps)) if eps < 1.0 else -np.inf

    # branch-b component log densities; the kappa*(cos - 1) + kappa form
    # keeps huge concentrations finite (kappa - log I0 ~ 0.5 log(2 pi
    # kappa) asymptotically).
    def _lg_phi(b):
        lv = (kappa * (xp.cos(phi0 - (phi0_c + b * np.pi)) - 1.0)
              + kappa - log_i0 - np.log(2.0 * np.pi))
        return _pe_logaddexp(
            log_1meps + lv,
            xp.full_like(lv, log_eps - np.log(2.0 * np.pi)), xp)

    def _lg_psi(b):
        # p(psi) = 2 * vonMises(2 psi; 2 (psi_c + b pi/2), kappa): the
        # factor 2 is the doubled-angle Jacobian, so the density
        # integrates to 1 over one period [0, pi).
        lv = (np.log(2.0)
              + kappa * (xp.cos(2.0 * psi - 2.0 * (psi_c + b * np.pi / 2.0))
                         - 1.0)
              + kappa - log_i0 - np.log(2.0 * np.pi))
        return _pe_logaddexp(
            log_1meps + lv, xp.full_like(lv, log_eps - np.log(np.pi)), xp)

    lg_joint = _pe_logaddexp(
        _lg_phi(0.0) + _lg_psi(0.0), _lg_phi(1.0) + _lg_psi(1.0), xp
    ) - np.log(2.0)

    # cos iota: eps-floored truncated Gaussian on [-1, 1]
    a_std = (-1.0 - ci_c) / sigma
    b_std = (1.0 - ci_c) / sigma
    logZ = xp.log(xp.clip(
        _pe_std_norm_cdf(b_std, xp) - _pe_std_norm_cdf(a_std, xp),
        1e-300, None))
    inside = (cos_iota >= -1.0) & (cos_iota <= 1.0)
    lg_tn = (-0.5 * ((cos_iota - ci_c) / sigma) ** 2 - xp.log(sigma)
             - 0.5 * np.log(2.0 * np.pi) - logZ)
    lg_tn = xp.where(inside, lg_tn, -xp.inf)
    lg_ci = _pe_logaddexp(
        log_1meps + lg_tn,
        xp.where(inside, log_eps - np.log(2.0), -xp.inf), xp)
    return lg_joint + lg_ci


def _pe_vonmises_rvs(kappa, rand, xp):
    """Zero-centered von Mises deviates (Best & Fisher 1979 rejection).

    ``rand(m)`` supplies uniforms on the same module (the caller owns the
    RNG stream). Acceptance is bounded below (~66% at worst), so the
    rejection loop finishes in a handful of rounds; a 512-round cap turns
    a theoretically-impossible stall into a loud error instead of a hang.
    ``kappa < 1e-8`` rows fall back to the uniform circle.
    """
    kappa = xp.asarray(kappa, dtype=xp.float64)
    n = int(kappa.shape[0])
    tiny = kappa < 1e-8
    k_safe = xp.maximum(kappa, 1e-8)
    tau = 1.0 + xp.sqrt(1.0 + 4.0 * k_safe**2)
    rho = (tau - xp.sqrt(2.0 * tau)) / (2.0 * k_safe)
    r = (1.0 + rho**2) / (2.0 * xp.maximum(rho, 1e-300))
    f_out = xp.zeros(n)
    todo = ~tiny
    rounds = 0
    while bool(todo.any()):
        rounds += 1
        if rounds > 512:
            raise RuntimeError(
                "pe_extrinsic_rvs: von Mises rejection sampler stalled")
        idx = xp.where(todo)[0]
        m = int(idx.shape[0])
        u1 = rand(m)
        u2 = xp.clip(rand(m), 1e-300, None)
        z = xp.cos(np.pi * u1)
        rr = r[idx]
        f = (1.0 + rr * z) / xp.maximum(rr + z, 1e-300)
        c = xp.clip(kappa[idx] * (rr - f), 1e-300, None)
        acc = (c * (2.0 - c) - u2 > 0.0) | (xp.log(c / u2) + 1.0 - c >= 0.0)
        keep = idx[acc]
        f_out[keep] = f[acc]
        todo[keep] = False
    u3 = rand(n)
    theta = xp.sign(u3 - 0.5) * xp.arccos(xp.clip(f_out, -1.0, 1.0))
    return xp.where(tiny, 2.0 * np.pi * rand(n) - np.pi, theta)


# TODO check: keep/remove/document -- only caller GBSpecialBase._pe_extr_draw (via _pe_or_pin_extrinsics: the F-stat distance birth, GB_RJ_FSTAT_DIST_BIRTH=1, and the PE replace step); both off in the 9mo/12mo runs
def pe_extrinsic_rvs(phi0_c, iota_c, psi_c, ln_snr, eps: float = 0.05,
                     geom: float = 2.0, rand=None, seed: Optional[int] = None):
    """Draw ``(phi0, cos_iota, psi)`` from the PE-mode extrinsic proposal.

    Samples exactly the density :func:`pe_extrinsic_logpdf` evaluates:
    one shared identity branch ``b ~ Bernoulli(1/2)`` shifts the
    (phi0, psi) centers jointly by ``(pi, pi/2)``, then each component is
    drawn from its in-branch eps-mixture (uniform floor with probability
    ``eps``, else the concentrated law). ``rand(m) -> (m,)`` uniforms on
    the target module is the caller-owned RNG stream; with none supplied
    this falls back to ``np.random.default_rng(seed)``, so ``seed=None``
    (the default) is OS entropy exactly as before and a direct caller that
    needs reproducibility passes ``rand`` or ``seed``. The GB move always
    supplies ``rand``, backed by the global cupy/numpy stream that
    ``run.py``'s ``_seed_rank_streams`` already seeds PER RANK. Outputs are
    wrapped/clipped into the sampling domains
    ``[0, 2 pi) x [-1, 1] x [0, pi)``.
    """
    from ..utils.utility import get_array_module

    xp = get_array_module(phi0_c)
    phi0_c = xp.asarray(phi0_c, dtype=xp.float64)
    psi_c = xp.asarray(psi_c, dtype=xp.float64)
    ci_c = xp.cos(xp.asarray(iota_c, dtype=xp.float64) % np.pi)
    n = int(phi0_c.shape[0])
    if rand is None:
        _rng = np.random.default_rng(seed)

        def rand(m):
            return xp.asarray(_rng.random(m))

    sigma = pe_extrinsic_sigma(ln_snr, geom=geom)
    kappa = 1.0 / sigma**2
    eps = float(eps)

    b = rand(n) < 0.5
    c_phi = phi0_c + xp.where(b, np.pi, 0.0)
    c_psi = psi_c + xp.where(b, np.pi / 2.0, 0.0)

    v_phi = c_phi + _pe_vonmises_rvs(kappa, rand, xp)
    u_phi = 2.0 * np.pi * rand(n)
    phi0 = xp.where(rand(n) < eps, u_phi, v_phi) % (2.0 * np.pi)

    v_psi = c_psi + 0.5 * _pe_vonmises_rvs(kappa, rand, xp)
    u_psi = np.pi * rand(n)
    psi = xp.where(rand(n) < eps, u_psi, v_psi) % np.pi

    a_std = (-1.0 - ci_c) / sigma
    b_std = (1.0 - ci_c) / sigma
    Fa = _pe_std_norm_cdf(a_std, xp)
    Fb = _pe_std_norm_cdf(b_std, xp)
    p = xp.clip(Fa + rand(n) * (Fb - Fa), 1e-15, 1.0 - 1e-16)
    tn = xp.clip(ci_c + sigma * _pe_std_norm_ppf(p, xp), -1.0, 1.0)
    u_ci = 2.0 * rand(n) - 1.0
    cos_iota = xp.where(rand(n) < eps, u_ci, tn)
    return phi0, cos_iota, psi


@dataclasses.dataclass
class GridSpec:
    """Tensor-grid design over the 4 intrinsic sampling parameters.

    Ranges are the closed box edges; ``n_*`` are the node counts per axis
    (``n`` nodes -> ``n - 1`` cells). Units match the legacy sampling basis:
    ``f0`` in mHz, ``Mc`` in solar masses, ``alpha`` in radians (ICRS RA),
    ``sin_delta`` dimensionless in ``[-1, 1]``. The design of a single
    :class:`FStatProposal4D` box (offline / script use; the in-run fit builds
    its boxes in :mod:`lisatools.sampling.fstat_gridfit`).

    Args:
        f0_range: ``(lo, hi)`` f0 box edges (mHz).
        Mc_range: ``(lo, hi)`` chirp-mass box edges (Msun).
        alpha_range: ``(lo, hi)`` right-ascension box edges (rad).
        sin_delta_range: ``(lo, hi)`` sin(declination) box edges.
        n_f0: f0 node count. Default ``24``.
        n_Mc: Mc node count. Default ``24``.
        n_alpha: alpha node count. Default ``24``.
        n_sin_delta: sin_delta node count. Default ``24``.

    Example:
        >>> from lisatools.sampling.fstat_proposal import GridSpec
        >>> spec = GridSpec((3.0, 3.01), (0.1, 1.0), (0.0, 6.28), (-1.0, 1.0),
        ...                 n_f0=12, n_Mc=4, n_alpha=8, n_sin_delta=8)
        >>> spec.shape
        (12, 4, 8, 8)
    """

    f0_range: Tuple[float, float]
    Mc_range: Tuple[float, float]
    alpha_range: Tuple[float, float]
    sin_delta_range: Tuple[float, float]
    n_f0: int = 24
    n_Mc: int = 24
    n_alpha: int = 24
    n_sin_delta: int = 24

    def axes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four uniformly spaced node axes (``np.linspace`` over each range).

        Returns:
            tuple: ``(f0_ax [mHz], Mc_ax [Msun], alpha_ax [rad], sin_delta_ax)``,
            host float arrays of lengths ``n_f0, n_Mc, n_alpha, n_sin_delta``.
        """
        return (
            np.linspace(*map(float, self.f0_range), int(self.n_f0)),
            np.linspace(*map(float, self.Mc_range), int(self.n_Mc)),
            np.linspace(*map(float, self.alpha_range), int(self.n_alpha)),
            np.linspace(*map(float, self.sin_delta_range), int(self.n_sin_delta)),
        )

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Node-grid shape ``(n_f0, n_Mc, n_alpha, n_sin_delta)``."""
        return (int(self.n_f0), int(self.n_Mc), int(self.n_alpha), int(self.n_sin_delta))


class FStatProposal4D:
    """Grid + inverse-CDF proposal over ``(f0 [mHz], Mc, alpha, sin_delta)``.

    Construction sweeps the F-statistic over the :class:`GridSpec` nodes in
    batched ``get_fstat_ll_wdm`` calls, stores ``g = beta * F`` on the node
    grid, and precomputes the flattened cell CDF. The represented density is
    piecewise-constant on the ``(n - 1)^4`` cells (each cell carries the
    average of ``exp(g)`` over its 2^4 corner nodes -- see :meth:`_build_cdf`),
    which makes ``rvs`` and ``logpdf`` *exactly* consistent with each other.

    The standalone single-box proposal (the in-run birth grid is the stacked
    :class:`StackedFStatProposal4D`). Used by the ``scripts/fstat_proposal``
    plotting / mock tools; not on the 9mo/12mo run path.

    Args:
        gb_wdm_comp (object): Object providing ``get_fstat_ll_wdm(params,
            wdm_holder, **kwargs) -> (N, M_upper)`` -- normally a
            ``gbgpu.gbcomps.GBWDMComputations`` (any object with that method
            works, e.g. the mock in
            ``scripts/fstat_proposal/plot_fstat_proposal_mock_highest_gb.py``);
            its ``xp`` attribute (if any) sets the array module.
        wdm_holder (object): Passed through untouched to ``get_fstat_ll_wdm`` (an
            ``AnalysisContainerArray`` holding the WDM residual + invC).
        grid_spec: The tensor-grid design box + resolution.
        beta: Inverse temperature; ``beta < 1`` broadens the proposal for
            healthier MCMC acceptance, ``beta > 1`` sharpens it. Default ``1.0``.
        amp_ref: Amplitude packed into the physical parameter rows. The
            F-stat maximizes over the extrinsics analytically, so this only
            has to be a sane waveform amplitude, not the source's. Default
            ``1e-22``.
        batch_size: Grid nodes per ``get_fstat_ll_wdm`` launch. Default ``16384``.
        seed: RNG seed for ``rvs`` (fresh ``default_rng`` when None).
        fstat_kwargs: Extra keyword arguments forwarded to
            ``get_fstat_ll_wdm`` (e.g. ``m_band_half_width``). Default ``None``.

    Raises:
        ValueError: Every grid cell has zero target mass.

    Attributes:
        ndim (int): 4.

    Example:
        .. code-block:: python

            from lisatools.sampling.fstat_proposal import FStatProposal4D, GridSpec

            spec = GridSpec((3.0, 3.01), (0.1, 1.0), (0.0, 2 * np.pi), (-1.0, 1.0),
                            n_f0=40, n_Mc=3, n_alpha=8, n_sin_delta=8)
            prop = FStatProposal4D(gb_wdm_comp, wdm_holder, spec, beta=1.0)
            theta = prop.rvs(1000)          # (1000, 4)
            lp = prop.logpdf(theta)         # (1000,)
    """

    #: sampling-basis column names, in order: ``("f0", "Mc", "alpha", "sin_delta")``
    param_names = ("f0", "Mc", "alpha", "sin_delta")
    ndim = 4

    def __init__(
        self,
        gb_wdm_comp,
        wdm_holder,
        grid_spec: GridSpec,
        beta: float = 1.0,
        amp_ref: float = 1e-22,
        batch_size: int = 16384,
        seed: Optional[int] = None,
        fstat_kwargs: Optional[dict] = None,
    ):
        self.gb_wdm_comp = gb_wdm_comp
        self.wdm_holder = wdm_holder
        self.grid_spec = grid_spec
        self.beta = float(beta)
        self.amp_ref = float(amp_ref)
        self.batch_size = int(batch_size)
        self.fstat_kwargs = dict(fstat_kwargs) if fstat_kwargs else {}
        self._rng = np.random.default_rng(seed)

        # Node axes are kept as host numpy for plotting/interp bookkeeping;
        # heavy per-sample work runs on self.xp.
        self._axes = grid_spec.axes()
        self._lo = np.array([ax[0] for ax in self._axes])
        self._hi = np.array([ax[-1] for ax in self._axes])
        self._dx = np.array([ax[1] - ax[0] for ax in self._axes])
        self._cell_shape = tuple(n - 1 for n in grid_spec.shape)

        self._F_grid = self._sweep_fstat()          # (n0, n1, n2, n3) node F
        self._logp_grid = self.beta * self._F_grid  # node log-target
        self._build_cdf()

    @classmethod
    def from_grid(cls, axes, logp_grid, beta: float = 1.0,
                  seed: Optional[int] = None):
        """Rebuild a proposal from a cached node grid (no F-stat sweep).

        The result has no scorer (``gb_wdm_comp is None``), so it runs on numpy.

        Args:
            axes (sequence of np.ndarray): 4 node arrays ``(f0_mHz, Mc, alpha,
                sin_delta)`` -- uniformly spaced, as produced by :class:`GridSpec`.
            logp_grid (np.ndarray): node-shaped ``beta * F`` array (what a prior
                instance stored in ``_logp_grid`` / a grid cache).
            beta: the inverse temperature already applied to ``logp_grid``
                (recorded for bookkeeping; the grid is used as-is). Default
                ``1.0``.
            seed: RNG seed for :meth:`rvs`.

        Returns:
            FStatProposal4D: The rebuilt proposal.
        """
        self = cls.__new__(cls)
        f0_ax, mc_ax, al_ax, sd_ax = [np.asarray(a, dtype=float) for a in axes]
        self.gb_wdm_comp = None
        self.wdm_holder = None
        self.grid_spec = GridSpec(
            (f0_ax[0], f0_ax[-1]), (mc_ax[0], mc_ax[-1]),
            (al_ax[0], al_ax[-1]), (sd_ax[0], sd_ax[-1]),
            len(f0_ax), len(mc_ax), len(al_ax), len(sd_ax),
        )
        self.beta = float(beta)
        self.amp_ref = 1e-22
        self.batch_size = 0
        self.fstat_kwargs = {}
        self._rng = np.random.default_rng(seed)
        self._axes = (f0_ax, mc_ax, al_ax, sd_ax)
        self._lo = np.array([ax[0] for ax in self._axes])
        self._hi = np.array([ax[-1] for ax in self._axes])
        self._dx = np.array([ax[1] - ax[0] for ax in self._axes])
        self._cell_shape = tuple(n - 1 for n in self.grid_spec.shape)
        self._logp_grid = np.asarray(logp_grid, dtype=float)
        self._F_grid = self._logp_grid / self.beta if self.beta else self._logp_grid
        self._build_cdf()
        return self

    # ------------------------------------------------------------------
    # backend plumbing (no module stored as attribute -- sprint pickle rule)
    # ------------------------------------------------------------------
    @property
    def xp(self):
        """Array module of the scorer (``gb_wdm_comp.xp``), else numpy."""
        return getattr(self.gb_wdm_comp, "xp", np)

    def __getstate__(self):
        # The proposal is a runtime object built around live kernel/data
        # handles; strip them so a pickled copy keeps the fitted grid and
        # stays usable for rvs/logpdf (rebuild to re-sweep the F-stat).
        state = self.__dict__.copy()
        state["gb_wdm_comp"] = None
        state["wdm_holder"] = None
        return state

    def reseed(self, seed):
        """Re-derive :attr:`_rng` from ``seed``; LEAF (no child proposals).

        ``seed is None`` returns immediately and touches nothing, so the
        entropy stream built at construction survives (the
        ``general.random_seed is None`` path). See the module's
        "RJ birth-container reseeding" block.
        """
        if seed is None:
            return
        self._rng = np.random.default_rng(np.random.SeedSequence(int(seed)))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _pack_physical(self, theta):
        """Sampling-basis ``(n, 4)`` -> physical 9-parameter GB rows ``(n, 9)``.

        Physical order: ``[A, f0(Hz), fdot, fddot, phi0, iota, psi, alpha,
        delta]`` (ICRS). Extrinsic slots carry fixed reference values
        (``A = amp_ref``, ``iota = pi/2``, ``phi0 = psi = fddot = 0``) -- the
        F-stat kernel builds its own 4 basis-filter extrinsic combinations.
        ``fdot = get_fdot(f0, Mc)`` (the ``r = 0`` manifold).

        Args:
            theta (array_like): ``(n, 4)`` rows ``(f0 [mHz], Mc, alpha, sin_delta)``.

        Returns:
            xp.ndarray: ``(n, 9)`` float64 physical rows.
        """
        from gbgpu.utils.utility import get_fdot

        xp = self.xp
        theta = xp.atleast_2d(xp.asarray(theta, dtype=xp.float64))
        n = theta.shape[0]
        f0_Hz = theta[:, 0] * 1e-3
        Mc = theta[:, 1]
        alpha = theta[:, 2]
        sin_delta = xp.clip(theta[:, 3], -1.0, 1.0)

        params = xp.zeros((n, 9), dtype=xp.float64)
        params[:, 0] = self.amp_ref
        params[:, 1] = f0_Hz
        params[:, 2] = xp.asarray(get_fdot(f=f0_Hz, Mc=Mc))
        # fddot = 0, phi0 = 0, psi = 0
        params[:, 5] = 0.5 * np.pi  # iota
        params[:, 7] = alpha
        params[:, 8] = xp.arcsin(sin_delta)
        return params

    def _sweep_fstat(self):
        """Batched F-stat over every grid node -> node-shaped array.

        Returns:
            xp.ndarray: ``grid_spec.shape`` float64 F values (``batch_size``
            nodes per scorer call).
        """
        xp = self.xp
        shape = self.grid_spec.shape
        n_total = int(np.prod(shape))
        axes_xp = [xp.asarray(ax) for ax in self._axes]

        F_flat = xp.empty(n_total, dtype=xp.float64)
        for start in range(0, n_total, self.batch_size):
            stop = min(start + self.batch_size, n_total)
            flat_idx = xp.arange(start, stop)
            multi = xp.unravel_index(flat_idx, shape)
            theta = xp.stack(
                [axes_xp[j][multi[j]] for j in range(4)], axis=1
            )
            params = self._pack_physical(theta)
            N_arr, M_upper = self.gb_wdm_comp.get_fstat_ll_wdm(
                params, self.wdm_holder, **self.fstat_kwargs
            )
            F_flat[start:stop] = compute_fstat(xp.asarray(N_arr), xp.asarray(M_upper))
        return F_flat.reshape(shape)

    def _build_cdf(self):
        """Cell CDF + normalization from the node grid.

        Each cell's weight is the *corner-averaged* (trapezoid) target over
        its 2^4 nodes rather than the lower-corner value -- this removes the
        systematic +dx/2 mean shift a lower-corner histogram density carries,
        while keeping the density piecewise-constant (rvs/logpdf stay exactly
        consistent). Sets ``_log_wcell`` (cell log weights), ``_cdf`` (flat
        normalized CDF) and ``_log_norm`` (``log sum_cells w_cell * cell_vol``).

        Raises:
            ValueError: Every cell has zero target mass.
        """
        xp = self.xp
        g_max = float(xp.max(self._logp_grid))
        p = xp.exp(self._logp_grid - g_max)
        for ax in range(4):
            lo = [slice(None)] * 4
            hi = [slice(None)] * 4
            lo[ax] = slice(None, -1)
            hi[ax] = slice(1, None)
            p = 0.5 * (p[tuple(lo)] + p[tuple(hi)])
        # p is now the (n-1)^4 corner-averaged cell weight (relative).
        with np.errstate(divide="ignore"):
            self._log_wcell = xp.log(p) + g_max
        cdf = xp.cumsum(p.reshape(-1))
        total = float(cdf[-1])
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(
                "FStatProposal4D: all grid cells have zero target mass "
                "(every F-stat evaluation was -inf?)"
            )
        self._cdf = cdf / total
        # log Z = log( sum_cells w_cell * cell_vol )   [uniform cells]
        cell_vol = float(np.prod(self._dx))
        self._log_norm = g_max + float(np.log(total)) + float(np.log(cell_vol))

    # ------------------------------------------------------------------
    # eryn duck-typed distribution interface
    # ------------------------------------------------------------------
    def rvs(self, size=1):
        """Exact draws from the piecewise-constant grid density.

        A cell is drawn from the CDF, then the position is uniform in it.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            xp.ndarray: ``size + (4,)`` float64 draws in the sampling basis
            ``(f0 [mHz], Mc, alpha, sin_delta)``.
        """
        xp = self.xp
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))

        u = xp.asarray(self._rng.random(n))
        flat_idx = xp.searchsorted(self._cdf, u, side="right")
        flat_idx = xp.clip(flat_idx, 0, self._cdf.shape[0] - 1)
        multi = xp.unravel_index(flat_idx, self._cell_shape)

        lo = xp.asarray(self._lo)
        dx = xp.asarray(self._dx)
        corners = xp.stack(
            [lo[j] + multi[j] * dx[j] for j in range(4)], axis=1
        )
        jitter = xp.asarray(self._rng.random((n, 4))) * dx[None, :]
        return (corners + jitter).reshape(size + (4,))

    def logpdf(self, x):
        """Normalized log density at ``x``, exactly consistent with :meth:`rvs`.

        Args:
            x (array_like): ``(n, 4)`` points (1-D input is one row).

        Returns:
            xp.ndarray: ``(n,)`` log density; ``-inf`` outside the grid box.
        """
        xp = self.xp
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))

        lo = xp.asarray(self._lo)
        hi = xp.asarray(self._hi)
        dx = xp.asarray(self._dx)
        inside = xp.all((x >= lo[None, :]) & (x <= hi[None, :]), axis=1)

        idx = xp.floor((x - lo[None, :]) / dx[None, :]).astype(xp.int64)
        n_cells = xp.asarray(self._cell_shape)
        idx = xp.clip(idx, 0, (n_cells - 1)[None, :])

        # Same corner-averaged cell weight rvs samples from -- exact
        # rvs/logpdf consistency.
        g_here = self._log_wcell[tuple(idx[:, j] for j in range(4))]
        out = g_here - self._log_norm
        return xp.where(inside, out, -xp.inf)


class StackedFStatProposal4D:
    """Vectorized flat mixture of K same-shape 4-D grid proposals.

    The per-sub-band peak refactor produces one local ``(n_f0, n_Mc,
    n_alpha, n_sin_delta)`` grid box per selected comb peak -- ~35 per
    sub-band, hundreds to thousands across a full-band run. A
    :class:`MixtureProposal` over that many :class:`FStatProposal4D`
    components evaluates ``rvs``/``logpdf`` in a per-component Python loop;
    this class instead stacks all K node grids into one ``(K, n0, n1, n2,
    n3)`` array so both operations are single vectorized array pipelines.
    It is an *implementation* of the flat mixture, not a per-band router --
    the consumer still sees one global 4-D distribution.

    Requirements (true by construction in the grid-gen pipeline):

    * every box shares the same node shape (the global ``FSTAT_N_*`` design);
    * the Mc / alpha / sin_delta node axes are IDENTICAL across boxes (the
      global Mc box + full sky); only the f0 axis differs per box (peak
      position, clamped to its sub-band, so per-box ``lo``/``dx``).

    Cell semantics match :class:`FStatProposal4D` exactly: piecewise-
    constant density on the ``(n-1)^4`` cells with corner-averaged
    (trapezoid) cell weights, so per-box ``rvs``/``logpdf`` are mutually
    exact; the mixture weighting is exact on top of that.

    Memory: the per-chunk cell log-weights (and, with ``keep_nodes``, the
    node grids) stay on the input's array module -- the device for a cupy
    stack. The flattened sampling CDF is built and kept on the HOST: it is
    only ever ``searchsorted`` by :meth:`rvs`, and a device copy cost about
    a third of the resident grid (6mo job 738 OOM in the device cumsum,
    2026-10-06). Same values, same draws.

    THE F-STAT GRID MEMORY PLAN (user ruling 2026-10-12, after the 9mo PE
    galaxy grid came to ~43 GB resident per card: node grids 24.2 GB + cell
    log-weights ~20 GB, both float64):

    1. DONE: ``keep_nodes=None`` resolves through
       :func:`stacked_keep_nodes`; the recipe drops the node grids when no
       replace move is composed (the only reader of ``log_node`` after
       construction is the trilinear in-cell mode). Bit-identical draws and
       densities in the uniform mode.
    2. TODO: store ``log_wcell`` (and ``log_node`` when kept) as float32 --
       another ~10 GB back on the 9mo. The proposal density then carries a
       ~1e-7 relative rounding; re-run the rvs/logpdf mutual-exactness tests
       at float32 and keep the CDF (host) and ``_log_norm`` in float64.
    3. TODO: host-resident grids with per-band upload. The RJ births are
       scheduled per band unit, so a propose only needs the boxes of the
       bands in the current unit (~1/9 of the grid per pass); keep the cell
       weights in pinned host memory, upload the needed boxes per pass
       through a small cache, and gather ``logpdf`` cells from the host.
       Device footprint ~1-2 GB; a rewrite of the chunk routing in
       :meth:`rvs` / :meth:`logpdf`; GPU-only validation.

    AXIS 1 IS THE GRID BASIS'S SECOND AXIS: chirp mass (Msun) on a legacy
    ``grid_basis="Mc"`` grid, ``fdot`` (Hz/s) on the production
    ``grid_basis="fdot"`` grid (``FSTAT_FDOT_AXIS=1``), where axis 0 then holds
    ``f_mid`` (mHz) rather than ``f0``. This class is agnostic: it only sees
    uniform axes. Built by :func:`~lisatools.sampling.fstat_gridfit.run_stacked_stage_b`
    (one per Mc group) and by :meth:`from_cache` / :func:`stacked_from_cache`
    on every epoch load; drawn from and scored by ``BandSorter`` through the RJ
    birth container on every F-stat RJ propose.

    Args:
        logp_grids (xp.ndarray): ``(K, n0, n1, n2, n3)`` node grids of
            ``beta * F`` (numpy or cupy; computation stays on the input's
            module, except the host-resident CDF).
        f0_los (array_like): ``(K,)`` per-box axis-0 start [mHz].
        f0_dxs (array_like): ``(K,)`` per-box axis-0 node spacing [mHz].
        mc_ax (array_like): shared axis-1 nodes (uniform spacing), length ``n1``.
        alpha_ax (array_like): shared alpha nodes [rad], length ``n2``.
        sin_delta_ax (array_like): shared sin(delta) nodes, length ``n3``.
        weights (array_like, optional): ``(K,)`` non-negative mixture weights
            (normalized internally); ``None`` -> equal.
        seed: RNG seed for :meth:`rvs`.
        mem_budget_mb: When set and the stacked cell arrays exceed this
            budget, the K axis is held/evaluated in chunks that fit it
            (``FSTAT_GRID_MEM_MB``; the 9mo/12mo launchers set 512).
            ``rvs``/``logpdf`` stay fully vectorized within each chunk.
        keep_nodes: Retain the raw node grids after construction (needed only
            by the ``"trilinear"`` in-cell mode). ``None`` -> the process
            default :func:`stacked_keep_nodes` (``False`` in the 9mo/12mo runs,
            which compose no replace move).

    Raises:
        ValueError: ``logp_grids`` is not 5-D, the axes' lengths disagree with
            the grid, or the total mixture mass is zero / non-finite.
        AssertionError: ``f0_los`` / ``f0_dxs`` / ``weights`` have the wrong
            length or the weights are negative or all zero.

    Attributes:
        K (int): Number of peak boxes.
        weights (np.ndarray): ``(K,)`` normalized box weights (host).
        keep_nodes (bool): Whether the node grids were retained.

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.sampling.fstat_proposal import StackedFStatProposal4D

            d = np.load("epoch_0003/fstat_grid_peaks_stacked.npz")  # legacy 1-group
            prop = StackedFStatProposal4D.from_cache(d, use_cupy=True, device=0)
            x = prop.rvs(512)               # (512, 4) on the device
            lp = prop.logpdf(x)             # (512,)
    """

    param_names = ("f0", "Mc", "alpha", "sin_delta")
    ndim = 4

    #: In-cell density mode. ``"uniform"`` (default) is the historical
    #: piecewise-constant density: cells are drawn by corner-averaged
    #: weight, the position within a cell is uniform. ``"trilinear"``
    #: keeps the cell selection and per-box normalizer BIT-IDENTICAL and
    #: replaces only the within-cell law with the multilinear (4-D
    #: "trilinear") interpolant of the node weights: since the integral
    #: of a multilinear function over a cell equals the average of its
    #: 2^4 corner values times the cell volume — exactly the
    #: corner-averaged cell weight the CDF and ``_log_norm`` already use
    #: — ``rvs`` and ``logpdf`` remain mutually exact in both modes.
    #: Motivation (2026-08-24 rj_replace candidate quality, root cause
    #: (b)): with 3 Mc nodes over the prior box a cell spans ~0.5 in Mc,
    #: and uniform in-cell jitter almost never lands on the thin
    #: fdot/sky ridge; the trilinear law concentrates draws toward the
    #: high-F corner (up to 2x per axis). Flip via
    #: :func:`stacked_in_cell_mode`, which restores the previous mode on
    #: exit — the RJ birth machinery keeps the uniform default.
    in_cell = "uniform"

    def __init__(self, logp_grids, f0_los, f0_dxs, mc_ax, alpha_ax,
                 sin_delta_ax, weights=None, seed: Optional[int] = None,
                 mem_budget_mb: Optional[float] = None,
                 keep_nodes: Optional[bool] = None):
        from ..utils.utility import get_array_module

        # ``None`` -> the process default (the recipe's "a replace move is
        # composed", env GB_FSTAT_KEEP_NODES on top); see stacked_keep_nodes.
        keep_nodes = stacked_keep_nodes() if keep_nodes is None else bool(keep_nodes)
        self.keep_nodes = keep_nodes
        xp = get_array_module(logp_grids)
        K = int(logp_grids.shape[0])
        node_shape = tuple(int(n) for n in logp_grids.shape[1:])
        if len(node_shape) != 4:
            raise ValueError("logp_grids must be (K, n_f0, n_Mc, n_alpha, n_sd)")
        self._node_shape = node_shape
        self._cell_shape = tuple(n - 1 for n in node_shape)
        self._ncells = int(np.prod(self._cell_shape))
        self.K = K
        self._rng = np.random.default_rng(seed)
        # per-peak draw census (host, always on -- one bincount per rvs)
        self._draw_counts = np.zeros(K, dtype=np.int64)

        # host metadata (small)
        self._f0_lo = np.asarray(_host(f0_los), dtype=float).ravel()
        self._f0_dx = np.asarray(_host(f0_dxs), dtype=float).ravel()
        assert self._f0_lo.shape == (K,) and self._f0_dx.shape == (K,)
        self._f0_hi = self._f0_lo + (node_shape[0] - 1) * self._f0_dx
        axes3 = [np.asarray(_host(a), dtype=float) for a in
                 (mc_ax, alpha_ax, sin_delta_ax)]
        # Only a[0], a[-1] and a[1]-a[0] are ever read from an axis, so an
        # axis of the WRONG LENGTH raises nothing here and instead rescales
        # every draw on that axis for the life of the proposal -- births at
        # sky angles and chirp masses the grid was never scored at, with no
        # error anywhere. Since the sky axes became per-group (f0-adaptive
        # sky, 2026-09-24) a loader that reaches for the wrong group's key
        # produces exactly that, so check the lengths against the grid.
        _want3 = tuple(node_shape[1:])
        _got3 = tuple(len(a) for a in axes3)
        if _got3 != _want3:
            raise ValueError(
                f"StackedFStatProposal4D: grid has (n_Mc, n_alpha, n_sd) = "
                f"{_want3} but the axes given are {_got3} "
                f"(Mc/alpha/sin_delta) -- the axes must describe the grid "
                f"they were swept on")
        self._lo3 = np.array([a[0] for a in axes3])
        self._hi3 = np.array([a[-1] for a in axes3])
        self._dx3 = np.array([a[1] - a[0] for a in axes3])
        self._axes3 = axes3

        w = (np.ones(K) if weights is None
             else np.asarray(_host(weights), dtype=float).ravel())
        assert w.shape == (K,) and np.all(w >= 0) and w.sum() > 0
        self.weights = w / w.sum()

        # K-chunking: per-box cell working set ~ log_wcell (device) + cdf
        # (host since 2026-10-06; the budget still counts both).
        per_box_bytes = 2 * self._ncells * 8
        if mem_budget_mb:
            k_chunk = max(1, int(float(mem_budget_mb) * 1e6 / per_box_bytes))
        else:
            k_chunk = K
        self._k_chunk = k_chunk

        # Build per-chunk cell weights + a GLOBAL-cumulative flat CDF.
        self._chunks = []
        running_mass = 0.0
        log_norm = np.empty(K)
        for k0 in range(0, K, k_chunk):
            k1 = min(k0 + k_chunk, K)
            g = xp.asarray(logp_grids[k0:k1])            # (Kc, n0..n3)
            gmax = g.reshape(g.shape[0], -1).max(axis=1)  # (Kc,)
            p = xp.exp(g - gmax[:, None, None, None, None])
            for ax in range(1, 5):
                lo_sl = [slice(None)] * 5
                hi_sl = [slice(None)] * 5
                lo_sl[ax] = slice(None, -1)
                hi_sl[ax] = slice(1, None)
                p = 0.5 * (p[tuple(lo_sl)] + p[tuple(hi_sl)])
            # p: (Kc,) + cell_shape corner-averaged relative cell weights
            with np.errstate(divide="ignore"):
                log_wcell = xp.log(p)
            log_wcell += gmax[:, None, None, None, None]   # in place: no 2nd cell array
            total = p.reshape(p.shape[0], -1).sum(axis=1)  # (Kc,)
            total_h = _host(total)
            cell_vol = self._f0_dx[k0:k1] * float(np.prod(self._dx3))
            with np.errstate(divide="ignore"):
                log_norm[k0:k1] = (_host(gmax) + np.log(total_h)
                                   + np.log(cell_vol))
            # sampling mass per cell: w_k * (cell weight / box total)
            scale = xp.asarray(self.weights[k0:k1] / np.clip(total_h, 1e-300, None))
            mass = (p.reshape(p.shape[0], -1) * scale[:, None]).ravel()
            del p
            # The CDF lives on the HOST (2026-10-06, 6mo job 738 PE refit
            # OOM at 88.7 GB in this cumsum): it is only ever searchsorted
            # by ``rvs``, so a GPU copy was a third of the resident grid for
            # nothing. Same values, same draws.
            cdf = np.cumsum(np.asarray(_host(mass), dtype=np.float64))
            del mass
            cdf += running_mass
            running_mass = float(cdf[-1])
            # ``log_node`` retains the raw node grids for the trilinear
            # in-cell mode. On numpy/cupy alike the slice-asarray above is
            # a view into the caller's stack, so keeping it costs no new
            # allocation — it only pins the stack alive for the container's
            # lifetime (``keep_nodes=False`` drops it; trilinear then
            # raises).
            self._chunks.append(dict(
                k0=k0, k1=k1, log_wcell=log_wcell, cdf=cdf,
                log_node=(g if keep_nodes else None),
            ))
        if not np.isfinite(running_mass) or running_mass <= 0.0:
            raise ValueError(
                "StackedFStatProposal4D: zero/non-finite total mixture mass "
                "(every F-stat grid cell was -inf?)"
            )
        for ch in self._chunks:
            ch["cdf"] /= running_mass
        self._chunk_cum = np.array([float(ch["cdf"][-1]) for ch in self._chunks])
        self._log_norm = log_norm  # per-box log Z (host)

        # f0-interval overlap structure for logpdf: boxes sorted by lo, plus
        # the ACTUAL max overlap depth D (several kept peaks can share one
        # box width -- comparable-F rescue -- so never assume a fixed depth).
        self._order = np.argsort(self._f0_lo, kind="stable")
        self._lo_sorted = self._f0_lo[self._order]
        his_sorted_all = np.sort(self._f0_hi)
        depth = np.arange(1, K + 1) - np.searchsorted(
            his_sorted_all, self._lo_sorted, side="left"
        )
        self._overlap_depth = int(max(1, depth.max()))

    # ------------------------------------------------------------------
    @property
    def xp(self):
        """Array module of the cell log-weights (numpy, or cupy on device)."""
        from ..utils.utility import get_array_module

        return get_array_module(self._chunks[0]["log_wcell"])

    def reseed(self, seed):
        """Re-derive :attr:`_rng` from ``seed``; LEAF (no child proposals).

        ``seed is None`` returns immediately and touches nothing.

        Args:
            seed (int or None): Integer seed.
        """
        if seed is None:
            return
        self._rng = np.random.default_rng(np.random.SeedSequence(int(seed)))

    @classmethod
    def from_cache(cls, d, weights=None, seed: Optional[int] = None,
                   mem_budget_mb: Optional[float] = None,
                   use_cupy: bool = False, device: Optional[int] = None,
                   keep_nodes: Optional[bool] = None):
        """Rebuild from a stacked-cache mapping (the ``*_peaks_stacked.npz``
        contents): keys ``logp_grids``, ``f0_los``, ``f0_dxs``, ``mc_ax``,
        ``alpha_ax``, ``sin_delta_ax``. ``use_cupy`` moves the stack to the
        GPU (rvs/logpdf then run on-device; numpy query inputs still work
        via ``cupy.asarray``).

        ``device`` pins WHICH GPU the grid stack lands on. Without it the
        upload takes cupy's process-current device, which for a container
        built BEFORE the run pins its main device
        (``lisatools.utils.device.pin_main_device``, reached from
        ``GlobalFit.setup_acs``) is device 0 regardless of ``GPUS`` -- every
        subsequent ``rvs`` / ``logpdf`` would then mix a device-0 grid with
        the run's coordinates on another device. Passing the run's GPU makes
        the container safe to build anywhere in the setup order.

        Args:
            d (Mapping): The cache (an ``np.load`` NpzFile or any mapping with
                those keys).
            weights (array_like, optional): ``(K,)`` box weights; ``None`` ->
                equal.
            seed: RNG seed for :meth:`rvs`.
            mem_budget_mb: K-chunking budget (MB), as in the constructor.
            use_cupy: Upload the grid stack to the GPU. Default ``False``.
            device: GPU id for the upload (``None`` = cupy's current device).
            keep_nodes: As in the constructor.

        Returns:
            StackedFStatProposal4D: The rebuilt proposal.
        """
        if use_cupy:
            import cupy as _cp

            from ..utils.device import device_context

            with device_context(_cp, device):
                grids = _cp.asarray(np.asarray(d["logp_grids"], dtype=float))
        else:
            grids = np.asarray(d["logp_grids"], dtype=float)
        return cls(
            grids,
            np.asarray(d["f0_los"], dtype=float),
            np.asarray(d["f0_dxs"], dtype=float),
            np.asarray(d["mc_ax"], dtype=float),
            np.asarray(d["alpha_ax"], dtype=float),
            np.asarray(d["sin_delta_ax"], dtype=float),
            weights=weights, seed=seed, mem_budget_mb=mem_budget_mb,
            keep_nodes=keep_nodes,
        )

    # ------------------------------------------------------------------
    def _corners_from_cells(self, kk, cell_flat, xp):
        """(box index, flat cell index) -> cell corner coords ``(n, 4)``.

        Args:
            kk (xp.ndarray): ``(n,)`` global box indices.
            cell_flat (xp.ndarray): ``(n,)`` flat cell indices within a box.
            xp (module): Array module.

        Returns:
            tuple[xp.ndarray, xp.ndarray]: ``(corners (n, 4), f0_dx (n,))`` --
            the lower cell corners and each row's axis-0 spacing.
        """
        multi = xp.unravel_index(cell_flat, self._cell_shape)
        f0_lo = xp.asarray(self._f0_lo)[kk]
        f0_dx = xp.asarray(self._f0_dx)[kk]
        lo3 = xp.asarray(self._lo3)
        dx3 = xp.asarray(self._dx3)
        out = xp.empty((cell_flat.shape[0], 4), dtype=xp.float64)
        out[:, 0] = f0_lo + multi[0] * f0_dx
        for j in range(3):
            out[:, j + 1] = lo3[j] + multi[j + 1] * dx3[j]
        return out, f0_dx

    @staticmethod
    def _lin_inv_cdf(u, a, b, xp):
        """Inverse CDF of the linear density ``f(t) ~ a*(1-t) + b*t`` on
        [0, 1] (endpoint values ``a``, ``b`` >= 0). Degenerate rows
        (``a + b == 0`` or ``a == b``) reduce to the uniform ``t = u``."""
        s = a + b
        d = b - a
        degen = (s <= 0.0) | (xp.abs(d) <= 1e-12 * xp.maximum(s, 1e-300))
        disc = xp.clip(a * a + d * s * u, 0.0, None)
        t = (xp.sqrt(disc) - a) / xp.where(degen, 1.0, d)
        t = xp.where(degen, u, t)
        return xp.clip(t, 0.0, 1.0)

    def _corner_cube(self, ch, k_local, multi, xp):
        """Gather the 2^4 corner node log-weights of the given cells:
        ``(m,)`` chunk-local box indices + 4-tuple of ``(m,)`` cell
        multi-indices -> ``(m, 2, 2, 2, 2)``."""
        g = ch.get("log_node")
        if g is None:
            raise RuntimeError(
                "StackedFStatProposal4D: trilinear in-cell mode requires "
                "keep_nodes=True (node grids were dropped at construction). "
                "The recipe keeps them only when a replace move is composed "
                "(set_stacked_keep_nodes); GB_FSTAT_KEEP_NODES=1 forces it."
            )
        di = xp.arange(2)
        m0 = int(k_local.shape[0])
        idx = [k_local.reshape(m0, 1, 1, 1, 1)]
        for ax in range(4):
            sh = [1, 1, 1, 1, 1]
            sh[ax + 1] = 2
            idx.append(multi[ax].reshape(m0, 1, 1, 1, 1) + di.reshape(sh))
        return g[tuple(idx)]

    @staticmethod
    def _interp_log_cube(cube, tt, xp):
        """``log`` of the multilinear interpolant of ``exp(cube)`` at the
        fractional in-cell position ``tt`` (4-tuple of ``(m,)`` in
        [0, 1]). The interpolation weights are the multilinear basis
        functions ``prod_ax((1-t_ax) or t_ax)``; rows whose contributing
        corners are all ``-inf`` return ``-inf``."""
        m0 = cube.shape[0]
        logw = xp.zeros_like(cube)
        with np.errstate(divide="ignore"):
            for ax, t in enumerate(tt):
                sh = [m0, 1, 1, 1, 1]
                sh[ax + 1] = 2
                logw = logw + xp.log(xp.stack([1.0 - t, t], axis=1).reshape(sh))
        flat = (cube + logw).reshape(m0, -1)
        mx = flat.max(axis=1)
        mx_safe = xp.where(xp.isfinite(mx), mx, 0.0)
        with np.errstate(invalid="ignore"):
            out = xp.log(xp.sum(xp.exp(flat - mx_safe[:, None]), axis=1)) + mx_safe
        return xp.where(xp.isfinite(mx), out, -xp.inf)

    def _in_cell_offsets(self, ch, k_local, cell, u4, xp):
        """Fractional in-cell positions for :meth:`rvs`.

        Uniform mode passes the 4 uniforms straight through (the
        historical jitter). Trilinear mode inverse-CDF samples the
        multilinear in-cell density one axis at a time: the marginal of a
        multilinear function along an axis is LINEAR (endpoints = the
        means of the corner values on each face), so each conditional is
        an exact linear-density inverse CDF; after each axis the cube is
        collapsed at the drawn ``t``. Exactly the same 4 uniforms per row
        are consumed as in uniform mode. Rows with no finite corner fall
        back to uniform."""
        if self.in_cell != "trilinear":
            return u4
        multi = xp.unravel_index(cell, self._cell_shape)
        cube = self._corner_cube(ch, k_local, multi, xp)
        m0 = cube.shape[0]
        mx = cube.reshape(m0, -1).max(axis=1)
        ok = xp.isfinite(mx)
        w = xp.exp(cube - xp.where(ok, mx, 0.0).reshape(m0, 1, 1, 1, 1))
        out = xp.empty_like(u4)
        cur = w
        for ax in range(4):
            a = cur[:, 0].reshape(m0, -1).mean(axis=1)
            b = cur[:, 1].reshape(m0, -1).mean(axis=1)
            t = self._lin_inv_cdf(u4[:, ax], a, b, xp)
            out[:, ax] = t
            if ax < 3:
                sh = (m0,) + (1,) * (cur.ndim - 2)
                cur = cur[:, 0] * (1.0 - t).reshape(sh) + cur[:, 1] * t.reshape(sh)
        return xp.where(ok[:, None], out, u4)

    def rvs(self, size=1):
        """Exact draws from the stacked mixture.

        Cell selection is always by the corner-averaged cell weights (the
        host CDF, one ``searchsorted`` per chunk); the within-cell law follows
        ``self.in_cell`` (see the class attribute). Each draw also increments
        the per-box census returned by :meth:`pop_draw_counts`.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            xp.ndarray: ``size + (4,)`` float64 draws on :attr:`xp` in the grid
            basis (``(f0 [mHz], Mc, alpha, sin_delta)``, or
            ``(f_mid [mHz], fdot [Hz/s], alpha, sin_delta)`` on an fdot grid).
        """
        xp = self.xp
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        u = self._rng.random(n)
        jit = self._rng.random((n, 4))
        out = xp.empty((n, 4), dtype=xp.float64)
        chunk_of = np.searchsorted(self._chunk_cum, u, side="right")
        chunk_of = np.clip(chunk_of, 0, len(self._chunks) - 1)
        for ci, ch in enumerate(self._chunks):
            m = chunk_of == ci
            if not m.any():
                continue
            flat = np.searchsorted(ch["cdf"], u[m], side="right")   # host cdf
            flat = xp.asarray(np.clip(flat, 0, ch["cdf"].shape[0] - 1))
            k_local = flat // self._ncells
            cell = flat - k_local * self._ncells
            kk = k_local + ch["k0"]
            _kkh = kk.get() if hasattr(kk, "get") else np.asarray(kk)
            np.add.at(self._draw_counts, _kkh.astype(np.int64), 1)
            corners, f0_dx = self._corners_from_cells(kk, cell, xp)
            j = self._in_cell_offsets(ch, k_local, cell, xp.asarray(jit[m]), xp)
            corners[:, 0] += j[:, 0] * f0_dx
            corners[:, 1:] += j[:, 1:] * xp.asarray(self._dx3)[None, :]
            out[xp.asarray(np.where(m)[0])] = corners
        return out.reshape(size + (4,))

    def pop_draw_counts(self):
        """Per-peak ``rvs`` draw counts since the last call, then reset.

        Read by the GB move's ``[FSTAT_PEAKS]`` census
        (``GBSpecialBase._stacked_for_census``). That walker does not descend
        into :class:`FdotAxisBirth` (``grid4``), so under the production
        ``FSTAT_FDOT_AXIS=1`` container it never reaches this method and the
        census line is not logged (none in the 9mo/12mo run logs).

        Returns:
            np.ndarray: ``(K,)`` int64 counts, in box order.
        """
        c = self._draw_counts
        self._draw_counts = np.zeros_like(c)
        return c

    # TODO check: keep/remove/document -- only caller is fit_gmm_to_stacked (the offline GMM layer of scripts/fstat_proposal); not reached by the 9mo/12mo runs
    def rvs_per_box(self, n_per_box: int):
        """``(K, n, 4)`` draws, ``n`` from EACH box's own grid density
        (ignoring mixture weights) -- the GMM-fitting sample source."""
        xp = self.xp
        out = xp.empty((self.K, int(n_per_box), 4), dtype=xp.float64)
        for ch in self._chunks:
            Kc = ch["k1"] - ch["k0"]
            lw = ch["log_wcell"].reshape(Kc, -1)
            m = xp.exp(lw - lw.max(axis=1, keepdims=True))
            cdf = xp.cumsum(m, axis=1)
            cdf = cdf / cdf[:, -1:]
            u = xp.asarray(self._rng.random((Kc, int(n_per_box))))
            offs = xp.arange(Kc, dtype=xp.float64)[:, None]
            idx = xp.searchsorted((cdf + offs).ravel(), (u + offs).ravel(),
                                  side="right")
            k_local = xp.repeat(xp.arange(Kc), int(n_per_box))
            cell = xp.clip(idx - k_local * self._ncells, 0, self._ncells - 1)
            kk = k_local + ch["k0"]
            corners, f0_dx = self._corners_from_cells(kk, cell, xp)
            j = xp.asarray(self._rng.random((corners.shape[0], 4)))
            corners[:, 0] += j[:, 0] * f0_dx
            corners[:, 1:] += j[:, 1:] * xp.asarray(self._dx3)[None, :]
            out[ch["k0"]:ch["k1"]] = corners.reshape(Kc, int(n_per_box), 4)
        return out

    def logpdf(self, x):
        """Normalized mixture log density, exactly consistent with :meth:`rvs`.

        Vectorized gather over the (actual) max f0-overlap depth D: every box
        whose axis-0 interval contains the point contributes
        ``g + log w_k - log Z_k`` (``g`` = the cell log weight, or the
        multilinear interpolant in trilinear mode), log-summed over boxes.

        Args:
            x (array_like): ``(n, 4)`` points in the grid basis (1-D input is
                one row).

        Returns:
            xp.ndarray: ``(n,)`` log density on :attr:`xp`; ``-inf`` outside every
            box or outside the shared axes 1-3.
        """
        xp = self.xp
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        n = x.shape[0]
        f0 = x[:, 0]

        lo3 = xp.asarray(self._lo3)
        hi3 = xp.asarray(self._hi3)
        dx3 = xp.asarray(self._dx3)
        inside3 = xp.all((x[:, 1:] >= lo3[None, :]) & (x[:, 1:] <= hi3[None, :]),
                         axis=1)
        idx3 = xp.floor((x[:, 1:] - lo3[None, :]) / dx3[None, :]).astype(xp.int64)
        idx3 = xp.clip(idx3, 0, xp.asarray(
            np.array(self._cell_shape[1:]) - 1)[None, :])
        # fractional in-cell positions (trilinear mode; exact 1.0 at the
        # top edge where the index clip pinned the cell)
        t3 = xp.clip((x[:, 1:] - lo3[None, :]) / dx3[None, :] - idx3, 0.0, 1.0)

        D = self._overlap_depth
        j = xp.searchsorted(xp.asarray(self._lo_sorted), f0, side="right")
        cand_pos = j[:, None] - 1 - xp.arange(D)[None, :]        # (n, D)
        valid = cand_pos >= 0
        cand_pos = xp.clip(cand_pos, 0, self.K - 1)
        kk = xp.asarray(self._order)[cand_pos]                   # (n, D)
        f0_lo_k = xp.asarray(self._f0_lo)[kk]
        f0_hi_k = xp.asarray(self._f0_hi)[kk]
        valid &= (f0[:, None] >= f0_lo_k) & (f0[:, None] <= f0_hi_k)
        i0 = xp.floor((f0[:, None] - f0_lo_k)
                      / xp.asarray(self._f0_dx)[kk]).astype(xp.int64)
        i0 = xp.clip(i0, 0, self._cell_shape[0] - 1)
        t0 = xp.clip(
            (f0[:, None] - f0_lo_k) / xp.asarray(self._f0_dx)[kk] - i0,
            0.0, 1.0)

        lp = xp.full((n, D), -xp.inf, dtype=xp.float64)
        log_w = xp.asarray(np.log(np.clip(self.weights, 1e-300, None)))
        log_norm = xp.asarray(self._log_norm)
        for ch in self._chunks:
            sel = valid & (kk >= ch["k0"]) & (kk < ch["k1"])
            if not bool(sel.any()):
                continue
            rows, cols = xp.where(sel)
            k_sel = kk[rows, cols]
            if self.in_cell == "trilinear":
                cube = self._corner_cube(
                    ch, k_sel - ch["k0"],
                    (i0[rows, cols], idx3[rows, 0], idx3[rows, 1],
                     idx3[rows, 2]), xp)
                g = self._interp_log_cube(
                    cube,
                    (t0[rows, cols], t3[rows, 0], t3[rows, 1], t3[rows, 2]),
                    xp)
            else:
                g = ch["log_wcell"][
                    k_sel - ch["k0"], i0[rows, cols],
                    idx3[rows, 0], idx3[rows, 1], idx3[rows, 2],
                ]
            lp[rows, cols] = g + log_w[k_sel] - log_norm[k_sel]

        m = xp.max(lp, axis=1)
        m_safe = xp.where(xp.isfinite(m), m, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = xp.log(xp.sum(xp.exp(lp - m_safe[:, None]), axis=1)) + m_safe
        out = xp.where(xp.isfinite(m), out, -xp.inf)
        return xp.where(inside3, out, -xp.inf)


class GroupedStackedFStatProposal:
    """Contiguous f0-groups of :class:`StackedFStatProposal4D`, one Mc axis
    density per group (the banded-Mc stage-B refactor, user ruling
    2026-08-26).

    At full band the auto Mc density is set by the MAX peak f0, so a single
    rectangular stack carries the high-frequency node count on every
    low-frequency box (~30x waste). Stage B instead groups the (f0-sorted)
    peak boxes by Mc-node requirement and builds one rectangular stack per
    group; this wrapper recombines them into the EXACT same mixture:

        p(x) = sum_g (W_g / W) * p_g(x),   W_g = sum of group g's box
                                            weights, p_g the group stack's
                                            internally-normalized mixture

    which equals ``sum_k (w_k / W) p_k(x)`` -- the single-stack density --
    identically. Groups are contiguous in the f0-sorted global box order,
    so per-box arrays (draw counts, ``rvs_per_box``) concatenate in group
    order to the global order the census expects.

    Duck-typed surfaces kept from the single stack: ``rvs`` / ``logpdf``
    (birth draws + BandSorter factors), ``pop_draw_counts`` (the census
    walker matches on this attribute BEFORE descending into components),
    ``rvs_per_box`` (GMM layer), ``components`` (walked by
    :func:`iter_stacked_components`, e.g. the trilinear in-cell switch).

    This is the peak stack of every multi-group stage-B fit -- in practice
    every full-band 9mo/12mo epoch (the banded node ladder and the f0-adaptive
    sky split a full-band fit into several groups). Each group may carry its own axis-1
    (fdot) axis and sky axes.

    Args:
        components (sequence of StackedFStatProposal4D): The group stacks, in
            f0-sorted global box order.
        box_weights (array_like, optional): ``(K,)`` GLOBAL per-box weights
            (``K`` = total boxes over all groups); the group mixture weight is
            each group's summed share. ``None`` -> equal per box.
        seed: RNG seed for the group draw.

    Raises:
        ValueError: No groups, ``box_weights`` of the wrong length, or weights
            that are negative or sum to zero.

    Attributes:
        components (list[StackedFStatProposal4D]): The group stacks.
        group_sizes (np.ndarray): ``(n_groups,)`` boxes per group.
        K (int): Total box count.
        weights (np.ndarray): ``(n_groups,)`` normalized group weights.
    """

    param_names = ("f0", "Mc", "alpha", "sin_delta")
    ndim = 4

    def __init__(self, components, box_weights=None, seed: Optional[int] = None):
        self.components = list(components)
        if not self.components:
            raise ValueError("GroupedStackedFStatProposal needs >= 1 group.")
        sizes = np.array([int(c.K) for c in self.components])
        self.group_sizes = sizes
        self.K = int(sizes.sum())
        self._bounds = np.concatenate([[0], np.cumsum(sizes)])
        if box_weights is None:
            w = np.ones(self.K)
        else:
            w = np.asarray(_host(box_weights), dtype=float).ravel()
        if w.shape != (self.K,):
            raise ValueError(
                f"box_weights has shape {w.shape}; expected ({self.K},) to "
                "match the total box count over all groups.")
        Wg = np.array([w[a:b].sum()
                       for a, b in zip(self._bounds[:-1], self._bounds[1:])])
        if not (np.all(Wg >= 0) and Wg.sum() > 0):
            raise ValueError("box_weights must be non-negative with positive "
                             "total mass.")
        self.weights = Wg / Wg.sum()
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Re-derive :attr:`_rng` and every group stack's, from ``seed``.

        CHILD ORDER: ``self.components`` (the group stacks, in group order).
        ``seed is None`` returns immediately and touches nothing.
        """
        if seed is None:
            return
        ss = np.random.SeedSequence(int(seed))
        self._rng = np.random.default_rng(ss)
        for child, kid in zip(self.components,
                              _child_seeds(ss, len(self.components))):
            _reseed_node(child, kid)

    # global per-box metadata (group order == f0-sorted global order)
    @property
    def f0_los(self):
        """``(K,)`` host per-box axis-0 starts [mHz], in global box order."""
        return np.concatenate([c._f0_lo for c in self.components])

    @property
    def f0_dxs(self):
        """``(K,)`` host per-box axis-0 node spacings [mHz], in global box order."""
        return np.concatenate([c._f0_dx for c in self.components])

    def rvs(self, size=1):
        """Draw a group by its weight, then from that group's stack.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            np.ndarray: ``size + (4,)`` HOST float64 draws in the grid basis
            (each group's device output is copied back).
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        which = self._rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty((n, 4))
        for g, comp in enumerate(self.components):
            m = which == g
            if m.any():
                out[m] = _host(comp.rvs(size=(int(m.sum()),)))
        return out.reshape(size + (4,))

    def logpdf(self, x):
        """Mixture log density ``log sum_g W_g p_g(x)``, exact w.r.t. :meth:`rvs`.

        Args:
            x (array_like): ``(n, 4)`` points in the grid basis.

        Returns:
            xp.ndarray: ``(n,)`` log density on the array module of ``x``
            (``-inf`` outside every group).
        """
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        with np.errstate(divide="ignore"):
            lps = xp.stack(
                [np.log(max(w, 1e-300)) + xp.asarray(c.logpdf(x))
                 for c, w in zip(self.components, self.weights)]
            )
        m = xp.max(lps, axis=0)
        m_safe = xp.where(xp.isfinite(m), m, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = xp.log(xp.sum(xp.exp(lps - m_safe), axis=0)) + m_safe
        return xp.where(xp.isfinite(m), out, -xp.inf)

    def pop_draw_counts(self):
        """Per-peak draw counts in the GLOBAL (f0-sorted) box order, then reset.

        Returns:
            np.ndarray: ``(K,)`` int64 counts.
        """
        return np.concatenate(
            [c.pop_draw_counts() for c in self.components])

    # TODO check: keep/remove/document -- only caller is fit_gmm_to_stacked (offline GMM layer); not reached by the 9mo/12mo runs
    def rvs_per_box(self, n_per_box: int):
        xp = self.components[0].xp
        return xp.concatenate(
            [xp.asarray(c.rvs_per_box(n_per_box)) for c in self.components],
            axis=0)


def stacked_from_cache(d, weights=None, seed: Optional[int] = None,
                       mem_budget_mb: Optional[float] = None,
                       use_cupy: bool = False, device: Optional[int] = None,
                       keep_nodes: Optional[bool] = None):
    """Rebuild the stage-B peak proposal from a ``*_peaks_stacked.npz``.

    Dispatches on the cache format: the legacy single-stack keys
    (``logp_grids`` / ``mc_ax``) rebuild a plain
    :class:`StackedFStatProposal4D` exactly as before; the grouped format
    (``group_sizes`` + per-group ``logp_grids_g{i}`` / ``mc_ax_g{i}``,
    with GLOBAL ``f0_los`` / ``f0_dxs`` / ``weights``) rebuilds a
    :class:`GroupedStackedFStatProposal`. ``weights`` is always the GLOBAL
    per-box weight vector; each group's stack receives its slice (a
    zero-mass group falls back to equal weights internally and simply
    never gets drawn).

    Refuses a cache whose ``grid_basis`` disagrees with ``FSTAT_FDOT_AXIS``.
    Axis 2 means chirp mass in one basis and Hz/s in the other; a consumer
    that reads one as the other gets NO error, just births at absurd
    parameters. An absent stamp is the legacy ``"Mc"`` basis.

    Called on every epoch load by
    :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution` and on
    the completed-epoch short circuit of
    :func:`~lisatools.sampling.fstat_gridfit.run_fstat_grid_fit`.

    Args:
        d (Mapping): The cache (``np.load`` NpzFile, or the
            :func:`~lisatools.sampling.fstat_gridfit.stacked_cache_at_floor`
            view).
        weights (array_like, optional): ``(K,)`` GLOBAL per-box weights;
            ``None`` -> equal.
        seed: RNG seed for every stack and the group draw.
        mem_budget_mb: K-chunking budget per stack (MB).
        use_cupy: Upload the grids to the GPU. Default ``False``.
        device: GPU id for the upload (``None`` = cupy's current device).
        keep_nodes: As in :class:`StackedFStatProposal4D`.

    Returns:
        StackedFStatProposal4D or GroupedStackedFStatProposal: The rebuilt peak
        proposal (single-stack cache -> the former).

    Raises:
        ValueError: ``grid_basis`` disagrees with :func:`fdot_axis_on`.

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.sampling.fstat_proposal import stacked_from_cache

            d = np.load("epoch_0003/fstat_grid_peaks_stacked.npz")
            prop = stacked_from_cache(d, use_cupy=True, device=0)
            x = prop.rvs(256)
    """
    keys = getattr(d, "files", None) or list(d)
    want = "fdot" if fdot_axis_on() else "Mc"
    got = str(np.asarray(d["grid_basis"]).item()) if "grid_basis" in keys \
        else "Mc"
    if got != want:
        raise ValueError(
            f"F-stat grid cache was fitted in the {got!r} basis but "
            f"FSTAT_FDOT_AXIS asks for {want!r}. Axis 2 is a chirp mass in "
            f"one and Hz/s in the other -- reusing it would place births at "
            f"absurd parameters with no error. Delete the "
            f"*_peaks_stacked.npz cache to refit, or restore the flag.")
    if "logp_grids" in keys:
        return StackedFStatProposal4D.from_cache(
            d, weights=weights, seed=seed, mem_budget_mb=mem_budget_mb,
            use_cupy=use_cupy, device=device, keep_nodes=keep_nodes)
    sizes = np.asarray(d["group_sizes"], dtype=int)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    K = int(sizes.sum())
    w = (None if weights is None
         else np.asarray(_host(weights), dtype=float).ravel())
    f0_los = np.asarray(d["f0_los"], dtype=float)
    f0_dxs = np.asarray(d["f0_dxs"], dtype=float)
    comps = []
    for gi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        sub_w = None
        if w is not None:
            sub_w = w[a:b]
            if not np.any(sub_w > 0):
                sub_w = None  # zero-mass group: equal inside, never drawn
        # PER-GROUP sky axes (f0-adaptive sky, 2026-09-24). The flat
        # ``alpha_ax`` / ``sin_delta_ax`` keys describe group 0 only in a
        # cache written since then, so preferring them would hand groups
        # 1..n an axis of the wrong LENGTH -- StackedFStatProposal4D reads
        # only ``a[0]``, ``a[-1]`` and ``a[1]-a[0]`` from an axis, so a
        # wrong-length alpha axis raises nothing and silently rescales
        # every sky draw in that group. Legacy caches have no suffixed key
        # and fall back to the flat one, which is correct for them because
        # their sky grid really was shared.
        _keys = d.files if hasattr(d, "files") else d
        comps.append(StackedFStatProposal4D.from_cache(
            dict(logp_grids=d[f"logp_grids_g{gi}"],
                 f0_los=f0_los[a:b], f0_dxs=f0_dxs[a:b],
                 mc_ax=d[f"mc_ax_g{gi}"],
                 alpha_ax=(d[f"alpha_ax_g{gi}"]
                           if f"alpha_ax_g{gi}" in _keys else d["alpha_ax"]),
                 sin_delta_ax=(d[f"sin_delta_ax_g{gi}"]
                               if f"sin_delta_ax_g{gi}" in _keys
                               else d["sin_delta_ax"])),
            weights=sub_w, seed=seed, mem_budget_mb=mem_budget_mb,
            use_cupy=use_cupy, device=device, keep_nodes=keep_nodes))
    return GroupedStackedFStatProposal(comps, box_weights=w, seed=seed)


def iter_stacked_components(dist, _seen=None):
    """Yield every :class:`StackedFStatProposal4D` reachable inside a
    proposal wrapper chain.

    Walks the containers the RJ birth assembly actually builds
    (:func:`make_gb_rj_birth_container` /
    ``fstat_gridfit.build_gb_birth_distribution``):
    ``RatioTightenedBirth.base`` -> eryn ``ProbDistContainer.priors_in``
    values -> ``UniformFloorMixture.base`` ->
    ``MixtureProposal.components`` -> stacked. Purely attribute-duck-typed
    (``base`` / ``components`` / ``priors_in`` / ``priors``), cycle-safe.

    It does NOT descend into :class:`FdotAxisBirth`, whose child is
    ``grid4`` -- so on a production ``FSTAT_FDOT_AXIS=1`` container it yields
    nothing (only consumer: :func:`stacked_in_cell_mode`, i.e. the replace
    moves, which the 9mo/12mo runs do not compose).

    Args:
        dist (object): Root of the wrapper chain.
        _seen (set, optional): Internal ``id`` set for cycle safety.

    Yields:
        StackedFStatProposal4D: Each stacked component found (depth-first).
    """
    if _seen is None:
        _seen = set()
    if dist is None or id(dist) in _seen:
        return
    _seen.add(id(dist))
    if isinstance(dist, StackedFStatProposal4D):
        yield dist
        return
    for child in getattr(dist, "components", None) or []:
        yield from iter_stacked_components(child, _seen)
    base = getattr(dist, "base", None)
    if base is not None:
        yield from iter_stacked_components(base, _seen)
    priors_in = getattr(dist, "priors_in", None)
    if isinstance(priors_in, dict):
        for child in priors_in.values():
            yield from iter_stacked_components(child, _seen)
    priors = getattr(dist, "priors", None)
    if isinstance(priors, (list, tuple)):
        for item in priors:
            child = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
            yield from iter_stacked_components(child, _seen)


@contextlib.contextmanager
def stacked_in_cell_mode(dist, mode: str):
    """Temporarily set the in-cell density mode of every stacked component
    inside ``dist``; restores the previous modes on exit (exception-safe).

    Yields ``True`` when ``mode == "trilinear"`` AND at least one stacked
    component was found — i.e. the trilinear density is actually in force
    for both ``rvs`` and ``logpdf`` — so the caller can keep its
    forward/reverse density bookkeeping consistent with what was drawn
    (the rj_replace usage: draw AND both logpdf sides inside ONE ``with``
    block). ``mode == "uniform"`` is a no-op that yields ``False``.

    Used by the replace step (``GBSpecialBase._run_replace_step``, stock
    default ``rj_replace`` / ``rj_replace_pe``; not composed in the 9mo/12mo
    runs). Trilinear mode needs the node grids (``keep_nodes``).

    Args:
        dist (object): The birth container (walked by
            :func:`iter_stacked_components`).
        mode: ``"uniform"`` or ``"trilinear"``.

    Yields:
        bool: Whether the trilinear density is in force inside the block.

    Raises:
        ValueError: ``mode`` is not one of the two names.
    """
    if mode not in ("uniform", "trilinear"):
        raise ValueError(
            f"in_cell mode must be 'uniform' or 'trilinear', got {mode!r}")
    stacked = list(iter_stacked_components(dist)) if mode == "trilinear" else []
    old = [s.in_cell for s in stacked]
    try:
        for s in stacked:
            s.in_cell = mode
        yield bool(stacked)
    finally:
        for s, o in zip(stacked, old):
            s.in_cell = o


class UniformFloorMixture:
    r"""``(1 - eps) * base + eps * Uniform(box)`` over the 4 intrinsics.

    RJ detailed-balance factors evaluate ``+logpdf(current leaf)`` for a
    death: a leaf whose in-model refinement drifted outside the base
    proposal's support would get ``-inf`` there and could never die. Mixing
    in a small uniform floor over the full per-band prior box keeps
    ``logpdf`` finite everywhere the sampler can go (the plan's
    "mixture + background floor").

    .. math::

        q(x) = (1 - \epsilon)\, q_{\rm base}(x)
            + \epsilon\, \frac{\mathbb{1}[x \in \text{box}]}{V_{\rm box}}.

    In the production container it wraps ``MixtureProposal([peak stack, comb])``
    with ``eps = FSTAT_FLOOR_EPS`` (default 0.1), built by
    :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution`.

    Args:
        base (object): Intrinsic 4-D distribution (``rvs``/``logpdf``
            duck-typed), e.g. :class:`StackedFStatProposal4D` or
            :class:`MixtureProposal`.
        box_lo (array_like): ``(4,)`` lower floor-box bounds in the grid basis.
        box_hi (array_like): ``(4,)`` upper floor-box bounds. Normally the
            band's interior f0 edges plus the full ranges of the other three
            axes: ``(f0 [mHz], Mc, alpha, sin_delta)`` on a legacy grid, or
            ``(f_mid [mHz], fdot [Hz/s], alpha, sin_delta)`` with the fitted
            fdot axis range on an fdot grid.
        eps: Floor mixture weight. Default ``0.05`` (the production caller passes
            ``FSTAT_FLOOR_EPS``).
        seed: RNG seed for ``rvs``.
    """

    ndim = 4

    def __init__(self, base, box_lo, box_hi, eps: float = 0.05,
                 seed: Optional[int] = None):
        self.base = base
        self.lo = np.asarray(box_lo, dtype=float)
        self.hi = np.asarray(box_hi, dtype=float)
        self.eps = float(eps)
        self._rng = np.random.default_rng(seed)
        self._log_vol = float(np.sum(np.log(self.hi - self.lo)))

    def reseed(self, seed):
        """Re-derive :attr:`_rng` and the wrapped base's, from ``seed``.

        CHILD ORDER: ``self.base``. ``seed is None`` touches nothing.
        """
        if seed is None:
            return
        ss = np.random.SeedSequence(int(seed))
        self._rng = np.random.default_rng(ss)
        _reseed_node(self.base, _child_seeds(ss, 1)[0])

    def rvs(self, size=1):
        """Draw from the floor with probability ``eps``, else from ``base``.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            np.ndarray: ``size + (4,)`` HOST float64 draws.
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        out = np.empty((n, 4))
        floor = self._rng.random(n) < self.eps
        n_floor = int(floor.sum())
        if n_floor:
            out[floor] = self._rng.uniform(self.lo, self.hi, size=(n_floor, 4))
        if n - n_floor:
            out[~floor] = _host(self.base.rvs(size=(n - n_floor,)))
        return out.reshape(size + (4,))

    def logpdf(self, x):
        """``log[(1 - eps) q_base(x) + eps / V_box]`` (floor term only inside the box).

        Args:
            x (array_like): ``(n, 4)`` points in the grid basis.

        Returns:
            xp.ndarray: ``(n,)`` log density on the array module of ``x``; NaN
            results are mapped to ``-inf``.
        """
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        lo = xp.asarray(self.lo)
        hi = xp.asarray(self.hi)
        inside = xp.all((x >= lo) & (x <= hi), axis=1)
        lp_base = xp.asarray(self.base.logpdf(x), dtype=xp.float64)
        lp_floor = xp.where(inside, -self._log_vol, -xp.inf)
        with np.errstate(invalid="ignore"):
            out = xp.logaddexp(np.log1p(-self.eps) + lp_base,
                               np.log(self.eps) + lp_floor)
        return xp.where(xp.isnan(out), -xp.inf, out)


def make_gb_rj_birth_container(intrinsic_dist, A_lims, use_cupy: bool = False,
                               fdot_astro_ratio_max=None, dist_lims=None,
                               ratio_tight=None, tobs=None, mc_lims=None):
    """Wrap a 4-D intrinsic proposal into the 8/9-column GB RJ birth container.

    Mirrors the stock GMM birth container
    (``gbspecialstretch.GBSpecialRJRefitMove.setup``): the intrinsics come from
    ``intrinsic_dist`` under a tuple key; slot 0 (``lnA`` or ``dist``) and the
    remaining extrinsics come from the stock prior uniforms
    (``phi0 ~ U[0, 2 pi)``, ``cos_iota ~ U[-1, 1]``, ``psi ~ U[0, pi)``; the band
    engines re-maximize phi0 analytically when ``phase_maximize`` is on).
    Returns an eryn ``ProbDistContainer`` (or its :class:`RatioTightenedBirth`
    wrapper) whose ``rvs(size) -> (size, N)`` / ``logpdf((n, N)) -> (n,)``
    match what ``BandSorter`` expects of ``rj_proposal_distribution["gb"]``
    (``N = 8``, or ``9`` with ``fdot_astro_ratio_max``), in the sampler's
    column order ``[A|dist, f0, Mc, phi0, cos_iota, psi, alpha, sin_delta(,
    fdot_astro_ratio)]``.

    Three layouts of the intrinsic block:

    * ``fdot_astro_ratio_max`` set and ``FSTAT_FDOT_AXIS=1`` (the 9mo/12mo
      runs): ``intrinsic_dist`` lives on the fdot grid and is wrapped in
      :class:`FdotAxisBirth` under the 5-D key
      ``("f0", "Mc", "fdot_astro_ratio", "alpha", "sin_delta")``; ``ratio_tight``
      is ignored.
    * ``fdot_astro_ratio_max`` set, legacy axis: the 4-D key
      ``("f0", "Mc", "alpha", "sin_delta")`` plus ``fdot_astro_ratio ~ U[-M, M]``,
      wrapped in :class:`RatioTightenedBirth` when ``ratio_tight`` is given.
    * ``fdot_astro_ratio_max=None``: the 8-column container, 4-D key only.

    Called by :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution`.

    Args:
        intrinsic_dist (object): 4-D distribution over the grid basis
            (e.g. :class:`UniformFloorMixture`).
        A_lims (sequence of float): ``[A_min, A_max]`` physical amplitude limits
            (``GBSettings.A_lims``); lnA is drawn uniform in ``log(A_lims)``
            when ``dist_lims`` is None.
        use_cupy: Match the run backend (False for CPU runs). Default ``False``.
        fdot_astro_ratio_max (float, optional): When not ``None``, the 9-column
            container with the ratio column (see above). ``None`` -> 8-column
            container.
        dist_lims (sequence of float, optional): When not ``None``, slot 0 is the
            luminosity DISTANCE (kpc) drawn uniform in ``dist_lims`` (linear)
            instead of lnA -- the distance basis (the 9mo/12mo runs). Follow-up:
            when the real 3-D (dist, sky) distribution lands, births should
            draw ``dist | sky`` from it.
        ratio_tight (dict, optional): Keyword arguments of
            :class:`RatioTightenedBirth` (``tobs``, ``phase_rad``, ``eps``,
            ``w_min``); used only on the legacy axis. Default ``None``.
        tobs (float, optional): Observation span (s); REQUIRED on the fdot axis
            (the ``f_mid`` shear), pass ``1.0 / df``.
        mc_lims (sequence of float, optional): Chirp-mass prior box for
            :class:`FdotAxisBirth`; ``None`` -> ``[0.001, 1.0]``.

    Returns:
        eryn.priors.ProbDistContainer or RatioTightenedBirth: The birth container.

    Raises:
        ValueError: The fdot axis is on and ``tobs`` is falsy.
    """
    from eryn.priors import ProbDistContainer, UniformDistribution

    if dist_lims is not None:
        slot0_name = "dist"
        slot0_prior = UniformDistribution(float(dist_lims[0]), float(dist_lims[1]))
    else:
        slot0_name = "A"
        slot0_prior = UniformDistribution(*np.log(np.asarray(A_lims, dtype=float)))
    priors_in = {
        slot0_name: slot0_prior,
        "phi0": UniformDistribution(0.0, 2.0 * np.pi),
        "cos_iota": UniformDistribution(-1.0, 1.0),
        "psi": UniformDistribution(0.0, np.pi),
    }
    key_order = [slot0_name, "f0", "Mc", "phi0", "cos_iota", "psi",
                 "alpha", "sin_delta"]
    M = None if fdot_astro_ratio_max is None else float(fdot_astro_ratio_max)
    if M is not None:
        key_order.append("fdot_astro_ratio")

    # FSTAT_FDOT_AXIS: the grid lives in (f_mid, fdot, alpha, sin_delta), so
    # the intrinsic block must own ``r`` as well -- inverting
    # ``fdot = fdot_gr(f0, Mc)(1 + r)`` needs both Mc and r, and the 4-D key
    # below has neither r nor any way to acquire it. The 5-D block does the
    # conversion and carries its own measure; it REPLACES both the separate
    # U[-M, M] ratio column and RatioTightenedBirth, whose whole job was to
    # patch up a draw the grid never scored.
    _fdot_axis = M is not None and fdot_axis_on()
    if _fdot_axis:
        if not tobs:
            raise ValueError(
                "FSTAT_FDOT_AXIS=1 needs tobs for the f_mid shear; got "
                f"{tobs!r}. Pass 1.0/df, never basis_settings.Tobs -- the "
                "latter is absent on FDSettings and a getattr default of 0.0 "
                "would silently disable the shear.")
        _mc = list(mc_lims or [0.001, 1.0])
        priors_in[("f0", "Mc", "fdot_astro_ratio", "alpha", "sin_delta")] = (
            FdotAxisBirth(intrinsic_dist, tobs=float(tobs),
                          mc_lo=float(_mc[0]), mc_hi=float(_mc[-1]),
                          ratio_max=M, use_cupy=use_cupy))
    else:
        priors_in[("f0", "Mc", "alpha", "sin_delta")] = intrinsic_dist
        if M is not None:
            priors_in["fdot_astro_ratio"] = UniformDistribution(-M, M)

    dist = ProbDistContainer(priors_in, use_cupy=use_cupy)
    # reset_key_order re-maps rvs/logpdf columns to the sampler layout
    # (a bare ``key_order = [...]`` assignment would NOT re-map).
    dist.reset_key_order(key_order)
    if M is not None and ratio_tight is not None and not _fdot_axis:
        return RatioTightenedBirth(dist, M, use_cupy=use_cupy, **ratio_tight)
    return dist


# GR chirp constant: fdot_gr = _FDOT_K * Mc[Msun]^{5/3} * f[Hz]^{11/3}
# (96/5) * pi^{8/3} * (GM_sun/c^3)^{5/3} with lisaconstants' GM_sun (2026-10-08;
# was G * an old M_sun, 3.6e-5 off); matches gbgpu.utils.utility.get_fdot.
from ..utils.constants import MTSUN_SI as _MTSUN_SI  # noqa: E402
_FDOT_K = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * _MTSUN_SI ** (5.0 / 3.0)


class RatioTightenedBirth:
    """Birth container whose ``fdot_astro_ratio`` PROPOSAL is tightened.

    THE PRIOR IS UNCHANGED (user ruling 2026-08-20): the run still samples
    ``r ~ U[-M, M]``. Only the RJ-birth proposal changes: the wrapped
    9-column container's independent ``r ~ U[-M, M]`` draw scattered the
    physical ``fdot = fdot_gr(f0, Mc) * (1 + r)`` over ``[-(M-1), M+1] x
    fdot_gr`` -- at 20 mHz that is +-5e-13 against a Fisher width of
    ~2e-15, so <1% of births carried a usable fdot even when the F-stat
    grid supplied the right (f0, Mc). Low f never noticed (fdot_gr tiny).

    New draw, CONDITIONAL on the candidate's own (f0, Mc):

        r | f0, Mc ~ (1 - eps) * U[-w, +w]  +  eps * U[-M, +M]
        w = clip( phase_rad / (pi * Tobs^2 * fdot_gr(f0, Mc)), w_min, M )

    i.e. tight around r = 0 -- which IS the grid-informed value: the
    F-stat kernel scores its templates at exactly ``fdot = fdot_gr(f0,
    Mc_node)`` (r = 0), so the tight component proposes what the grid
    actually measured. ``phase_rad`` is the allowed carrier-phase drift
    error over the window (default one cycle); at 20 mHz w ~ 0.27, below
    ~5 mHz the clip at M makes the draw identical to the old one. The
    ``eps`` floor keeps FULL support over the prior box: BandSorter
    evaluates ``logpdf`` at EXISTING sources for RJ death factors, and a
    hard-truncated proposal would assign them -inf (deaths unproposable).

    ``rvs``/``logpdf`` are exactly consistent (the same mixture), so the
    RJ Metropolis-Hastings factors remain correct; ndim and the prior are
    untouched, so stores resume cleanly and the change is proposal-only.

    Legacy-axis only: :func:`make_gb_rj_birth_container` applies it when
    ``GB_FSTAT_BIRTH_RATIO_TIGHT=1`` supplies ``ratio_tight`` AND
    ``FSTAT_FDOT_AXIS=0``. Under the production fdot axis :class:`FdotAxisBirth`
    owns ``r`` and this wrapper is not built.

    Args:
        base (object): The 9-column birth ``ProbDistContainer`` (its ratio column
            drawn ``U[-M, M]``).
        ratio_max (float): ``M``.
        tobs (float): Observation span (s); keyword-only.
        phase_rad (float, optional): Allowed carrier-phase drift (rad). Default
            ``2 pi`` (``GB_FSTAT_BIRTH_RATIO_PHASE``).
        eps (float, optional): Weight of the full-prior component. Default
            ``0.1`` (``GB_FSTAT_BIRTH_RATIO_EPS``).
        w_min (float, optional): Floor on the tight half-width ``w``. Default
            ``0.05`` (``GB_FSTAT_BIRTH_RATIO_WMIN``).
        f0_col (int, optional): Column of ``f0`` (mHz). Default ``1``.
        mc_col (int, optional): Column of ``Mc``. Default ``2``.
        r_col (int, optional): Column of ``fdot_astro_ratio``. Default ``8``.
        use_cupy (bool, optional): Array module when no input fixes it. Default
            ``False``.
        seed (int, optional): RNG seed. Default ``None``.
    """

    ndim = 9

    def __init__(self, base, ratio_max, *, tobs, phase_rad=2.0 * np.pi,
                 eps=0.1, w_min=0.05, f0_col=1, mc_col=2, r_col=8,
                 use_cupy=False, seed=None):
        self.base = base
        self.M = float(ratio_max)
        self.tobs = float(tobs)
        self.phase_rad = float(phase_rad)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.f0_col, self.mc_col, self.r_col = int(f0_col), int(mc_col), int(r_col)
        self.use_cupy = bool(use_cupy)
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Re-derive :attr:`_rng` and the wrapped container's, from ``seed``.

        CHILD ORDER: ``self.base`` (the 8/9-column ``ProbDistContainer``).
        ``seed is None`` touches nothing.
        """
        if seed is None:
            return
        ss = np.random.SeedSequence(int(seed))
        self._rng = np.random.default_rng(ss)
        _reseed_node(self.base, _child_seeds(ss, 1)[0])

    def _xp(self, x=None):
        """Array module of ``x``, else cupy when ``use_cupy``, else numpy."""
        if x is not None:
            from ..utils.utility import get_array_module
            return get_array_module(x)
        if self.use_cupy:
            import cupy as cp
            return cp
        return np

    def _width(self, f0_mHz, mc, xp):
        """Tight half-width ``w = clip(phase_rad / (pi T^2 fdot_gr), w_min, M)`` per row."""
        fdot_gr = _FDOT_K * xp.maximum(mc, 1e-6) ** (5.0 / 3.0) * \
            xp.maximum(f0_mHz * 1e-3, 1e-6) ** (11.0 / 3.0)
        dfdot = self.phase_rad / (np.pi * self.tobs ** 2)
        return xp.clip(dfdot / fdot_gr, self.w_min, self.M)

    def rvs(self, size=1):
        """Draw from ``base`` and redraw the ratio column from the tight mixture.

        Args:
            size (int or tuple[int, ...], optional): Number of rows (a tuple is
                flattened). Default ``1``.

        Returns:
            xp.ndarray: ``(n, 9)`` rows.
        """
        n = int(np.prod(size)) if not isinstance(size, int) else int(size)
        x = self.base.rvs(size=n)
        xp = self._xp(x)
        x = xp.atleast_2d(x)
        w = self._width(x[:, self.f0_col], x[:, self.mc_col], xp)
        u = xp.asarray(self._rng.random(n))
        wide = u < self.eps
        draw = xp.asarray(self._rng.uniform(-1.0, 1.0, n))
        x[:, self.r_col] = xp.where(wide, draw * self.M, draw * w)
        return x

    def logpdf(self, x):
        """Base log density with its ``U[-M, M]`` ratio term replaced by the mixture.

        Args:
            x (array_like): ``(n, 9)`` rows.

        Returns:
            xp.ndarray: ``(n,)`` log density (NaN mapped to ``-inf``).
        """
        xp = self._xp(x)
        x = xp.atleast_2d(xp.asarray(x))
        lp = xp.asarray(self.base.logpdf(x))
        r = x[:, self.r_col]
        w = self._width(x[:, self.f0_col], x[:, self.mc_col], xp)
        q = ((1.0 - self.eps) * (xp.abs(r) <= w) / (2.0 * w)
             + self.eps * (xp.abs(r) <= self.M) / (2.0 * self.M))
        # replace the base's independent U[-M, M] term with the mixture
        with np.errstate(divide="ignore"):
            lp = lp + np.log(2.0 * self.M) + xp.log(q)
        return xp.where(xp.isnan(lp), -xp.inf, lp)


class FdotAxisBirth:
    """5-D birth block ``(f0, Mc, r, alpha, sin_delta)`` over an fdot grid.

    THE CHANGE. The F-stat grid's ``Mc`` axis is already an ``fdot`` axis,
    just a bad one: rows are assembled as ``fdot = fdot_gr(f0, Mc_node)``,
    i.e. the ``r = 0`` MANIFOLD. So the grid searches only GR-driven
    chirps, ``fdot ~ Mc^(5/3)`` makes uniform-in-Mc nodes non-uniform in
    the coordinate that matters (while ``StackedFStatProposal4D``
    hard-assumes uniform axes), and negative ``fdot`` is UNREPRESENTABLE
    -- against 40% of low-f and 21% of high-f v7 leaves that carry
    ``fdot < 0``. Under this class the grid lives in
    ``(f_mid, fdot, alpha, sin_delta)`` with ``fdot`` a first-class linear
    axis, and the conversion to the sampling basis happens here.

    WHY 5-D AND NOT 4-D. Inverting ``fdot = fdot_gr(f0, Mc)(1 + r)``
    needs BOTH ``Mc`` and ``r``, so the conversion cannot live in the
    4-D ``(f0, Mc, alpha, sin_delta)`` block. Owning ``r`` here RETIRES
    :class:`RatioTightenedBirth`'s blind draw -- which is the point: today
    ``r`` is drawn by something the grid never scored, and it shears the
    candidate back off the ridge the grid just put it on.

    THE MEASURE. ``Mc`` is the fiber coordinate (the likelihood is flat
    along it), drawn uniformly on the FEASIBLE interval
    ``[mc_floor(fdot, f0), mc_hi]`` -- the set on which ``|r| <= M``. With
    ``g = fdot_gr(f0, Mc)``, the map ``(f_mid, fdot, Mc) -> (f0, Mc, r)``
    has determinant ``1/g``::

        det = dr/dfdot + c_t * dr/df_mid
            = [1/g + c_t*d*g_f0/g**2] - [c_t*d*g_f0/g**2]
            = 1/g

    The two shear terms cancel EXACTLY even though ``g`` depends on
    ``f0`` -- so the shear contributes nothing and a wrong ``c_t`` costs
    efficiency only, never correctness. Hence::

        log q = log p_grid(f_mid, fdot, alpha, sd)
                + log g                       # dfdot/dr
                - log(mc_hi - mc_floor)       # the uniform Mc draw

    ``rvs`` and ``logpdf`` are exactly consistent, so the RJ
    Metropolis-Hastings factors stay correct. That consistency is the one
    thing here that can be silently wrong -- a mismatch biases the
    acceptance ratio in both directions and raises no error -- so it is
    pinned by an independence-proposal invariance test with five negative
    controls (``tests/test_fstat_fdot_birth.py``).

    ``_defect`` injects a deliberate error for those controls. It is a
    TEST-ONLY hook and it validates its argument: a silently-ignored
    defect name would make a control pass while testing nothing.

    This is the intrinsic birth block of every 9mo/12mo F-stat RJ birth
    (``FSTAT_FDOT_AXIS=1``), built by :func:`make_gb_rj_birth_container` with
    ``tobs = 1/df``, ``mc_lims`` from the GB settings and
    ``ratio_max = fdot_astro_ratio_max``. Rows that fall outside the feasible
    set are counted in ``n_infeasible`` (see :meth:`rvs`).

    Args:
        grid4 (object): 4-D distribution over the fdot grid basis
            ``(f_mid [mHz], fdot [Hz/s], alpha, sin_delta)`` (the floored
            peak/comb mixture).
        tobs (float): Observation span (s); keyword-only. Sets
            ``c_t = shear * tobs``.
        mc_lo (float): Chirp-mass prior floor (Msun).
        mc_hi (float): Chirp-mass prior ceiling (Msun).
        ratio_max (float): Ratio half-width ``M``.
        shear (float, optional): Shear fraction. Default ``0.5`` (``f_mid``);
            must match the grid's ``grid_c_t``.
        seed (int, optional): RNG seed for the ``Mc`` draw. Default ``None``.
        use_cupy (bool, optional): Array module when no input fixes it.
            Default ``False``.
        _defect (str, optional): Test-only defect name from :attr:`DEFECTS`.
            Default ``None``.

    Raises:
        ValueError: ``_defect`` is not a recognised name.

    Attributes:
        param_names (tuple[str, ...]): ``("f0", "Mc", "fdot_astro_ratio",
            "alpha", "sin_delta")``.
        ndim (int): 5.
    """

    param_names = ("f0", "Mc", "fdot_astro_ratio", "alpha", "sin_delta")
    ndim = 5

    #: recognised ``_defect`` values (test-only; see the class docstring)
    DEFECTS = (None, "omit_jac", "flip_jac", "omit_mcwidth",
               "shear_rvs_only", "mc_full_box_rvs")

    def __init__(self, grid4, *, tobs, mc_lo, mc_hi, ratio_max,
                 shear=0.5, seed=None, use_cupy=False, _defect=None):
        if _defect not in self.DEFECTS:
            raise ValueError(
                f"unknown _defect {_defect!r}; expected one of {self.DEFECTS}")
        self.grid4 = grid4
        self.tobs = float(tobs)
        self.mc_lo = float(mc_lo)
        self.mc_hi = float(mc_hi)
        self.M = float(ratio_max)
        self.shear = float(shear)
        self.use_cupy = bool(use_cupy)
        self._defect = _defect
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Re-derive :attr:`_rng` and the wrapped 4-D grid's, from ``seed``.

        CHILD ORDER: ``self.grid4``. ``seed is None`` touches nothing.
        """
        if seed is None:
            return
        ss = np.random.SeedSequence(int(seed))
        self._rng = np.random.default_rng(ss)
        _reseed_node(self.grid4, _child_seeds(ss, 1)[0])

    @property
    def _c_t(self):
        """Shear coefficient ``c_t = shear * tobs`` in seconds."""
        return self.shear * self.tobs

    def _xp(self, x=None):
        """Array module of ``x``, else cupy when ``use_cupy``, else numpy."""
        if x is not None:
            from ..utils.utility import get_array_module
            return get_array_module(x)
        if self.use_cupy:
            import cupy as cp
            return cp
        return np

    def _floor(self, fdot, f0_hz, *, side):
        """Feasible ``Mc`` floor. ``side`` is ``"rvs"`` or ``"logpdf"``.

        ``side`` exists ONLY for the ``mc_full_box_rvs`` control, and the
        asymmetry is the point. An earlier version applied that defect
        here unconditionally -- so both paths saw the same widened
        interval, the proposal stayed self-consistent, and the control
        correctly failed to skew (KS p = 0.005). A defect injected into a
        shared helper is not a defect; it has to break rvs against logpdf.
        """
        if self._defect == "mc_full_box_rvs" and side == "rvs":
            xp = self._xp(fdot)
            return xp.full(xp.shape(fdot), self.mc_lo)
        return mc_floor_for_fdot(fdot, f0_hz, self.M, self.mc_lo)

    def rvs(self, size=1):
        """Draw ``(f_mid, fdot, alpha, sd)`` from the grid, then ``Mc`` on its fiber.

        ``f0 = f_mid - c_t fdot``; ``Mc ~ U[mc_floor(fdot, f0), mc_hi]``;
        ``r = fdot / fdot_gr(f0, Mc) - 1``. A row whose floor exceeds ``mc_hi``
        (a mis-sized grid) is clamped and counted in ``self.n_infeasible``.

        Args:
            size (int or tuple[int, ...], optional): Number of rows (a tuple is
                flattened). Default ``1``.

        Returns:
            np.ndarray: ``(n, 5)`` float64 rows ``(f0 [mHz], Mc, r, alpha,
            sin_delta)`` (host: the grid draw is converted with ``np.asarray``).
        """
        n = int(np.prod(size)) if not isinstance(size, int) else int(size)
        z = np.atleast_2d(np.asarray(self.grid4.rvs(size=n), dtype=float))
        xp = self._xp(z)
        f_mid_hz = z[:, 0] * 1e-3
        fdot = z[:, 1]
        f0_hz = f_mid_hz - self._c_t * fdot
        floor = self._floor(fdot, f0_hz, side="rvs")
        # INFEASIBLE ROWS. The grid's fdot axis is sized at ONE reference
        # f0, but each row's own f0 comes off the shear, and the reachable
        # |fdot| goes as f0^(11/3) -- so a row whose f0 falls below the
        # reference can be handed an fdot no Mc in the box can carry, i.e.
        # mc_floor > mc_hi. That is a MIS-SIZED GRID (fix it by sizing the
        # axis at the group's lowest f0 -- run_stacked_stage_b intersects
        # the per-box bounds for exactly this reason), not something to
        # paper over: such a row is priced -inf and rejected forever, which
        # reads as inefficiency rather than as a bug. Clamp so the draw
        # stays in the box, and COUNT it so a mis-sized grid is visible.
        bad = floor > self.mc_hi
        n_bad = int(xp.count_nonzero(bad))
        if n_bad:
            self.n_infeasible = getattr(self, "n_infeasible", 0) + n_bad
            floor = xp.minimum(floor, self.mc_hi)
        # Mc is the FIBER coordinate: the likelihood is flat along it, so
        # it is prior-set rather than grid-informed. Uniform on the
        # feasible interval, which is exactly the set where |r| <= M.
        mc = floor + xp.asarray(self._rng.random(n)) * (self.mc_hi - floor)
        r = r_from_fdot(fdot, f0_hz, mc)
        out = xp.zeros((n, 5), dtype=xp.float64)
        out[:, 0] = f0_hz * 1e3
        out[:, 1] = mc
        out[:, 2] = r
        out[:, 3] = z[:, 2]
        out[:, 4] = z[:, 3]
        return out

    def logpdf(self, x):
        """``log p_grid(f_mid, fdot, alpha, sd) + log fdot_gr - log(mc_hi - mc_floor)``.

        Exactly the density :meth:`rvs` samples (see the class docstring for the
        measure). Boundary rows are judged with a ``1e-9`` relative tolerance.

        Args:
            x (array_like): ``(n, 5)`` rows ``(f0 [mHz], Mc, r, alpha, sin_delta)``.

        Returns:
            xp.ndarray: ``(n,)`` log density; ``-inf`` outside the feasible set
            (``|r| > M``, ``Mc`` outside ``[mc_floor, mc_hi]``, empty interval, or
            a non-finite grid density).
        """
        xp = self._xp(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        f0_hz = x[:, 0] * 1e-3
        mc = x[:, 1]
        r = x[:, 2]
        g = fdot_gr(f0_hz, mc)
        fdot = g * (1.0 + r)
        # The shear must be applied on BOTH sides. Applying it in rvs only
        # (or logpdf only) silently biases the RJ ratio in both directions
        # with no error anywhere -- the control that catches it is the
        # single most important test in this file.
        c_t = 0.0 if self._defect == "shear_rvs_only" else self._c_t
        f_mid_mHz = (f0_hz + c_t * fdot) * 1e3
        q = xp.stack([f_mid_mHz, fdot, x[:, 3], x[:, 4]], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = xp.asarray(self.grid4.logpdf(q), dtype=xp.float64)
            floor = self._floor(fdot, f0_hz, side="logpdf")
            width = self.mc_hi - floor
            if self._defect != "omit_mcwidth":
                lp = lp - xp.log(xp.where(width > 0, width, 1.0))
            if self._defect == "flip_jac":
                lp = lp - xp.log(g)
            elif self._defect != "omit_jac":
                lp = lp + xp.log(g)
        # Tolerant at the boundary, deliberately. At the top of the fdot
        # axis ``floor -> mc_hi``, the feasible Mc interval collapses and
        # ``|r| = M`` EXACTLY; ``logpdf`` then re-derives fdot from
        # ``(f0, Mc, r)`` and the floor from that, so a round trip of a
        # boundary row lands a few ulp outside. A hard comparison rejects
        # ~0.4% of the block's own draws -- the classic silent RJ bug where
        # a birth is made, priced at -inf and rejected forever, so the move
        # looks merely inefficient. The excluded set has measure zero, so a
        # relative epsilon changes no density, only the boundary verdict.
        eps = 1e-9
        ok = ((width > 0) & (mc >= floor * (1.0 - eps) - eps)
              & (mc <= self.mc_hi * (1.0 + eps) + eps)
              & (xp.abs(r) <= self.M * (1.0 + eps) + eps)
              & xp.isfinite(lp))
        return xp.where(ok, lp, -xp.inf)


class MixtureProposal:
    """Weighted mixture of 4-D intrinsic proposals (``rvs``/``logpdf``).

    In the production birth container this is ``[peak stack, comb]`` with
    weights ``[1 - FSTAT_COMB_WEIGHT, FSTAT_COMB_WEIGHT]`` (0.7 / 0.3 by
    default), built by
    :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution`
    whenever the epoch has a comb cache.

    Args:
        components (sequence): 4-D distributions (``rvs``/``logpdf`` duck-typed).
        weights (array_like, optional): Component weights (normalized
            internally); ``None`` -> equal.
        seed: RNG seed for the component draw.
    """

    ndim = 4

    def __init__(self, components, weights=None, seed: Optional[int] = None):
        self.components = list(components)
        w = (np.ones(len(self.components)) if weights is None
             else np.asarray(weights, dtype=float))
        self.weights = w / w.sum()
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Re-derive :attr:`_rng` and every component's, from ``seed``.

        CHILD ORDER: ``self.components`` (construction order -- peak stack
        first, comb second in the RJ birth assembly). ``seed is None``
        touches nothing.
        """
        if seed is None:
            return
        ss = np.random.SeedSequence(int(seed))
        self._rng = np.random.default_rng(ss)
        for child, kid in zip(self.components,
                              _child_seeds(ss, len(self.components))):
            _reseed_node(child, kid)

    def rvs(self, size=1):
        """Draw a component by weight, then from it.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            np.ndarray: ``size + (4,)`` HOST float64 draws.
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        which = self._rng.choice(len(self.components), size=n, p=self.weights)
        out = np.empty((n, 4))
        for k, comp in enumerate(self.components):
            m = which == k
            if m.any():
                out[m] = _host(comp.rvs(size=(int(m.sum()),)))
        return out.reshape(size + (4,))

    def logpdf(self, x):
        """Mixture log density ``log sum_k w_k q_k(x)``.

        Args:
            x (array_like): ``(n, 4)`` points.

        Returns:
            xp.ndarray: ``(n,)`` log density on the array module of ``x``; NaN
            results are mapped to ``-inf``.
        """
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        lps = xp.stack(
            [np.log(w) + xp.asarray(c.logpdf(x))
             for c, w in zip(self.components, self.weights)]
        )
        m = xp.max(lps, axis=0)
        m = xp.where(xp.isfinite(m), m, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = xp.log(xp.sum(xp.exp(lps - m), axis=0)) + m
        return xp.where(xp.isnan(out), -xp.inf, out)


class CombIntrinsicProposal:
    """Comb-scan f0 density x uniform (Mc, alpha, sin_delta).

    Built from a cached comb scan (``f0_nodes``, ``F_max``): the f0 marginal
    is piecewise-constant with cell weights ``clip(F, 0) ** power``
    (corner-averaged). ``power=1`` (linear-in-F) keeps proportional mass on
    *every* comb peak -- the right successive-birth behavior: once the
    sampler births and subtracts the loudest source, births there stop
    gaining likelihood while the next peaks still carry proposal mass
    (an ``exp(F)`` weighting would collapse onto the single loudest peak).
    The other three axes are uniform over their prior boxes (the band
    engines re-maximize phi0; sky/Mc refine in-model after birth).

    In the 9mo/12mo runs (``FSTAT_FDOT_AXIS=1``) it is built with
    ``fdot_conditional=True`` by
    :func:`~lisatools.sampling.fstat_gridfit.build_gb_birth_distribution` from
    the epoch's ``*_comb.npz`` (weight ``FSTAT_COMB_WEIGHT``, default 0.3):
    column 0 is then read downstream as ``f_mid`` and column 1 is ``fdot``
    drawn uniformly over the feasible range at the row's own frequency.

    Args:
        f0_nodes_mHz (array_like): ``(n_nodes,)`` uniformly spaced comb nodes
            (mHz).
        F_max (array_like): ``(n_nodes,)`` per-node sky-maximized F.
        mc_lims (sequence of float): ``(lo, hi)`` box for column 1 when it is
            ``Mc`` (Msun).
        alpha_lims (tuple[float, float], optional): RA box (rad). Default
            ``(0, 2 pi)``.
        sin_delta_lims (tuple[float, float], optional): sin(dec) box. Default
            ``(-1, 1)``.
        power: Exponent of the cell weight ``clip(F, 0) ** power``. Default
            ``1.0``.
        seed: RNG seed.
        fdot_conditional: Draw column 1 as ``fdot`` [Hz/s] uniform on
            ``fdot_axis_bounds(f0, mc_hi, ratio_max)`` (see the attribute
            comment below). Default ``False``.
        ratio_max: ``M``; required when ``fdot_conditional``.
        mc_hi: Upper chirp-mass bound (Msun); required when
            ``fdot_conditional``.

    Raises:
        ValueError: The comb weights are empty / non-finite, or
            ``fdot_conditional`` is set while ``ratio_max`` or ``mc_hi`` is
            ``None``.
    """

    ndim = 4

    #: When set, column 1 is FDOT [Hz/s] drawn CONDITIONALLY on the row's
    #: own f0 -- uniform over ``fdot_axis_bounds(f0, mc_hi, ratio_max)``,
    #: the range a physical source at that f0 can actually carry. Without
    #: it, under FSTAT_FDOT_AXIS this component drew "Mc uniform on
    #: [0.001, 1]" that the 5-D birth block then read as Hz/s: every comb
    #: birth became f0 ~ -1e6 mHz, was priced -inf and silently rejected --
    #: 30% of the proposal mass wasted and the comb's whole job (births
    #: where no peak box exists) dead, with nothing louder than a slightly
    #: low birth acceptance to show for it. The conditional width enters
    #: rvs and logpdf through the SAME function, so the pair stays
    #: mutually exact and the RJ factors stay correct.
    def __init__(self, f0_nodes_mHz, F_max, mc_lims, alpha_lims=(0.0, 2 * np.pi),
                 sin_delta_lims=(-1.0, 1.0), power: float = 1.0,
                 seed: Optional[int] = None, fdot_conditional: bool = False,
                 ratio_max: Optional[float] = None,
                 mc_hi: Optional[float] = None):
        self.f0_nodes = np.asarray(f0_nodes_mHz, dtype=float)
        F = np.clip(np.asarray(F_max, dtype=float), 0.0, None) ** float(power)
        w = 0.5 * (F[:-1] + F[1:])
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("comb weights are empty/non-finite")
        self._w = w
        self._cdf = np.cumsum(w) / total
        spacing = float(self.f0_nodes[1] - self.f0_nodes[0])
        # per-cell f0 density = w / (total * spacing)
        with np.errstate(divide="ignore"):
            self._log_f0_pdf = np.log(w) - np.log(total) - np.log(spacing)
        self.mc_lims = tuple(map(float, mc_lims))
        self.alpha_lims = tuple(map(float, alpha_lims))
        self.sin_delta_lims = tuple(map(float, sin_delta_lims))
        self.fdot_conditional = bool(fdot_conditional)
        if self.fdot_conditional:
            if ratio_max is None or mc_hi is None:
                raise ValueError(
                    "fdot_conditional=True needs ratio_max and mc_hi to "
                    "size the per-f0 feasible range; guessing them would "
                    "silently mis-scale every comb birth.")
            self._ratio_max = float(ratio_max)
            self._mc_hi = float(mc_hi)
        self._log_uni = -sum(
            np.log(hi - lo) for lo, hi in
            (self.mc_lims, self.alpha_lims, self.sin_delta_lims)
        )
        # sky-only part, for the conditional mode (the fdot width is per-row)
        self._log_uni_sky = -sum(
            np.log(hi - lo) for lo, hi in
            (self.alpha_lims, self.sin_delta_lims)
        )
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Re-derive :attr:`_rng` from ``seed``; LEAF (no child proposals).

        ``seed is None`` returns immediately and touches nothing.
        """
        if seed is None:
            return
        self._rng = np.random.default_rng(np.random.SeedSequence(int(seed)))

    def _fdot_lo_width(self, f0_mHz, xp=np):
        """Per-row feasible-fdot (lo, width) at ``mc_hi`` -- ONE function
        for rvs and logpdf, so the pair cannot drift apart."""
        g = fdot_gr(xp.asarray(f0_mHz, dtype=xp.float64) * 1e-3, self._mc_hi)
        return (1.0 - self._ratio_max) * g, 2.0 * self._ratio_max * g

    def rvs(self, size=1):
        """Draw a comb cell by weight (uniform inside it), then the other axes.

        Args:
            size (int or tuple[int, ...], optional): Batch shape. Default ``1``.

        Returns:
            np.ndarray: ``size + (4,)`` HOST float64 draws.
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        idx = np.searchsorted(self._cdf, self._rng.random(n), side="right")
        idx = np.clip(idx, 0, len(self._w) - 1)
        u = self._rng.random(n)
        out = np.empty((n, 4))
        out[:, 0] = (self.f0_nodes[idx]
                     + u * (self.f0_nodes[idx + 1] - self.f0_nodes[idx]))
        if self.fdot_conditional:
            lo1, w1 = self._fdot_lo_width(out[:, 0])
            out[:, 1] = lo1 + self._rng.random(n) * w1
        else:
            out[:, 1] = self._rng.uniform(*self.mc_lims, n)
        out[:, 2] = self._rng.uniform(*self.alpha_lims, n)
        out[:, 3] = self._rng.uniform(*self.sin_delta_lims, n)
        return out.reshape(size + (4,))

    def logpdf(self, x):
        """Comb f0 cell density times the uniform densities of the other axes.

        Args:
            x (array_like): ``(n, 4)`` points.

        Returns:
            xp.ndarray: ``(n,)`` log density on the array module of ``x``;
            ``-inf`` outside the comb span or the boxes (the per-row fdot
            interval in conditional mode).
        """
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        f0 = x[:, 0]
        f0_nodes = xp.asarray(self.f0_nodes)
        idx = xp.searchsorted(f0_nodes, f0, side="right") - 1
        in_f0 = (f0 >= f0_nodes[0]) & (f0 <= f0_nodes[-1])
        idx = xp.clip(idx, 0, len(self._w) - 1)
        in_sky = (
            (x[:, 2] >= self.alpha_lims[0]) & (x[:, 2] <= self.alpha_lims[1])
            & (x[:, 3] >= self.sin_delta_lims[0])
            & (x[:, 3] <= self.sin_delta_lims[1])
        )
        if self.fdot_conditional:
            lo1, w1 = self._fdot_lo_width(f0, xp=xp)
            inside = (in_f0 & in_sky
                      & (x[:, 1] >= lo1) & (x[:, 1] <= lo1 + w1))
            with np.errstate(divide="ignore"):
                out = (xp.asarray(self._log_f0_pdf)[idx] + self._log_uni_sky
                       - xp.log(xp.maximum(w1, 1e-300)))
        else:
            inside = (in_f0 & in_sky
                      & (x[:, 1] >= self.mc_lims[0])
                      & (x[:, 1] <= self.mc_lims[1]))
            out = xp.asarray(self._log_f0_pdf)[idx] + self._log_uni
        return xp.where(inside, out, -xp.inf)


# ======================================================================
# GMM sampling layer, built FROM the stacked grids (reuse-only wrappers
# around lisatools.sampling.gmm.vec_fit_gmm_min_bic /
# lisatools.sampling.prior.FullGaussianMixtureModel)
# ======================================================================

# ``FullGaussianMixtureModel.logpdf`` culls components by frequency-sorted
# windowing on COLUMN 1 of its native basis (its stock consumer is the 8-col
# GB basis where f0 is column 1). The intrinsic sampling basis here is
# (f0, Mc, alpha, sin_delta) with f0 at column 0, so the fitted components
# are permuted to a native (Mc, f0, alpha, sin_delta) order for the model
# and wrapped back through :class:`ColumnPermutedProposal` -- without this
# the culling would window on Mc (correct but a no-op: every component gets
# evaluated for every query, O(n x n_components) per RJ step).
_GMM_COLUMN_PERM = (1, 0, 2, 3)


# TODO check: keep/remove/document -- only built by build_peak_gmm (offline GMM layer, FSTAT_PEAK_SAMPLING=gmm in scripts/fstat_proposal/run_fstat_rj_search.py); not reached by the 9mo/12mo runs
class ColumnPermutedProposal:
    """View of a distribution whose native columns are a permutation of the
    canonical ``(f0, Mc, alpha, sin_delta)`` basis (Jacobian = 1).

    ``perm`` maps canonical -> native: ``x_native = x[..., perm]``.
    """

    ndim = 4

    def __init__(self, base, perm):
        self.base = base
        self._perm = tuple(int(p) for p in perm)
        inv = [0] * len(self._perm)
        for i, p in enumerate(self._perm):
            inv[p] = i
        self._inv = tuple(inv)

    def reseed(self, seed):
        """Forward ``seed`` to the wrapped distribution; owns no ``_rng``.

        CHILD ORDER: ``self.base``. The stock base here is
        :class:`~lisatools.sampling.prior.FullGaussianMixtureModel`, which
        draws from the module-level ``np.random`` state rather than a private
        ``Generator`` -- :func:`_reseed_node` leaves such a node alone, so
        this is a no-op on the GMM path (its stream is the one
        ``run.py::_seed_rank_streams`` seeds per rank). The forward exists so
        a future reseedable base is covered without another edit here.
        """
        if seed is None:
            return
        _reseed_node(self.base, int(seed))

    def rvs(self, size=1):
        out = self.base.rvs(size=size)
        return out[..., list(self._inv)]

    def logpdf(self, x):
        return self.base.logpdf(x[..., list(self._perm)])


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only callers scripts/fstat_proposal/run_fstat_rj_search.py and plot_fstat_proposal_mojito.py (FSTAT_FIT_GMM=1); already flagged TODO(fstat-gmm-deprecation)
def fit_gmm_to_stacked(stacked, n_samples_per_box: int = 4096, gpu=None,
                       min_comp: int = 1, max_comp: int = 12,
                       verbose: bool = False):
    """Fit per-box GMMs to a :class:`StackedFStatProposal4D`: draw
    ``n_samples_per_box`` from each box's grid density and hand the
    ``(K, n, 4)`` block to the existing GPU-batched min-BIC fitter
    :func:`lisatools.sampling.gmm.vec_fit_gmm_min_bic`.

    The GMM is the OPTIONAL memory-light sampling layer for the F-stat
    birth proposal (the stacked grids are the production layer).
    TODO(fstat-gmm-deprecation): may be deprecated for that use once the
    grid path proves out at full-band GPU scale; the serial-search / refit
    container path is a separate consumer of the GMM machinery.

    Returns the raw component lists ``[weights, means, covs, invcovs, dets,
    mins, maxs]`` (canonical column order) for caching via
    :func:`pack_gmm_components`.
    """
    from .gmm import vec_fit_gmm_min_bic

    samples = _host(stacked.rvs_per_box(int(n_samples_per_box)))
    return vec_fit_gmm_min_bic(
        samples, min_comp=int(min_comp), max_comp=int(max_comp),
        gpu=gpu, verbose=verbose, return_components=True,
    )


def pack_gmm_components(comps) -> dict:
    """Flatten ragged per-box GMM component lists into npz-storable arrays.

    The storage format of the GB warm-start components
    (``GB_WARM_START_COMPONENTS``, the ``GB_WARM_SELF`` refit): written by
    :mod:`lisatools.globalfit.warmstart.fit_from_store` (and the para-search
    tool), read back with :func:`unpack_gmm_components`.

    Args:
        comps (sequence): ``[weights, means, covs, invcovs, dets, mins, maxs]``,
            each a length-``K`` list of per-box arrays (numpy or cupy):
            ``weights`` ``(n_k,)``, ``means`` ``(n_k, d)``, ``covs`` /
            ``invcovs`` ``(n_k, d, d)``, ``dets`` ``(n_k,)``, ``mins`` /
            ``maxs`` ``(d,)``.

    Returns:
        dict[str, np.ndarray]: Host arrays ``gmm_ncomp`` ``(K,)`` int,
        ``gmm_weights`` ``(sum n_k,)``, ``gmm_means`` ``(sum n_k, d)``,
        ``gmm_covs`` / ``gmm_invcovs`` ``(sum n_k, d, d)``, ``gmm_dets``
        ``(sum n_k,)``, ``gmm_mins`` / ``gmm_maxs`` ``(K, d)``.
    """
    weights, means, covs, invcovs, dets, mins, maxs = comps
    return dict(
        gmm_ncomp=np.array([len(_host(w)) for w in weights], dtype=int),
        gmm_weights=np.concatenate([_host(w) for w in weights]),
        gmm_means=np.concatenate([_host(m) for m in means], axis=0),
        gmm_covs=np.concatenate([_host(c) for c in covs], axis=0),
        gmm_invcovs=np.concatenate([_host(c) for c in invcovs], axis=0),
        gmm_dets=np.concatenate([_host(d) for d in dets]),
        gmm_mins=np.vstack([_host(m) for m in mins]),
        gmm_maxs=np.vstack([_host(m) for m in maxs]),
    )


def unpack_gmm_components(d):
    """Inverse of :func:`pack_gmm_components` (accepts an npz mapping).

    Called when the warm-start proposal loads its components
    (:mod:`lisatools.globalfit.warmstart.proposal`).

    Args:
        d (Mapping): Mapping with the ``gmm_*`` keys written by
            :func:`pack_gmm_components`.

    Returns:
        list: ``[weights, means, covs, invcovs, dets, mins, maxs]``, each a
        length-``K`` list of host float arrays (the per-box split).
    """
    ncomp = np.asarray(d["gmm_ncomp"], dtype=int)
    splits = np.cumsum(ncomp)[:-1]
    return [
        np.split(np.asarray(d["gmm_weights"], dtype=float), splits),
        np.split(np.asarray(d["gmm_means"], dtype=float), splits, axis=0),
        np.split(np.asarray(d["gmm_covs"], dtype=float), splits, axis=0),
        np.split(np.asarray(d["gmm_invcovs"], dtype=float), splits, axis=0),
        np.split(np.asarray(d["gmm_dets"], dtype=float), splits),
        [row for row in np.asarray(d["gmm_mins"], dtype=float)],
        [row for row in np.asarray(d["gmm_maxs"], dtype=float)],
    ]


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; only caller scripts/fstat_proposal/run_fstat_rj_search.py (FSTAT_PEAK_SAMPLING=gmm); already flagged TODO(fstat-gmm-deprecation)
def build_peak_gmm(comps, box_weights=None, use_cupy: bool = False,
                   limit: float = 10.0):
    """Assemble the batched peak-GMM sampling object from fitted components.

    TODO(fstat-gmm-deprecation): optional layer (``FSTAT_PEAK_SAMPLING=gmm``)
    -- may be deprecated once the grid path proves out at full-band GPU
    scale.

    Wraps :class:`lisatools.sampling.prior.FullGaussianMixtureModel` (the
    existing batched mixture with single-array-op ``rvs``/``logpdf`` over
    thousands of Gaussians). ``box_weights`` re-weights the per-box shares
    (default equal per box, the model's native convention): pass the peak F
    values for the ``w ~ F`` default, or ``None``/uniform for equal weights.
    Component arrays are permuted so f0 is the model's column 1 (its
    frequency-culling axis); the returned object speaks the canonical
    ``(f0, Mc, alpha, sin_delta)`` basis.
    """
    from .prior import FullGaussianMixtureModel

    perm = np.asarray(_GMM_COLUMN_PERM)
    weights, means, covs, invcovs, dets, mins, maxs = comps
    K = len(weights)
    weights = [np.asarray(_host(w), dtype=float) for w in weights]
    if box_weights is not None:
        bw = np.asarray(_host(box_weights), dtype=float).ravel()
        assert bw.shape == (K,) and np.all(bw >= 0) and bw.sum() > 0
        bw = bw / bw.sum()
        # The model gives each box an equal 1/K share
        # (``concatenate(weights) / K``); pre-scaling each box's (unit-sum)
        # weight vector by K * bw_k turns that into the bw_k share while the
        # grand total stays exactly 1.
        weights = [w * (float(b) * K) for w, b in zip(weights, bw)]
    means_p = [np.asarray(_host(m), dtype=float)[:, perm] for m in means]
    covs_p = [np.asarray(_host(c), dtype=float)[:, perm][:, :, perm]
              for c in covs]
    invcovs_p = [np.asarray(_host(c), dtype=float)[:, perm][:, :, perm]
                 for c in invcovs]
    dets = [np.asarray(_host(d), dtype=float) for d in dets]
    mins_p = [np.asarray(_host(m), dtype=float)[perm] for m in mins]
    maxs_p = [np.asarray(_host(m), dtype=float)[perm] for m in maxs]
    model = FullGaussianMixtureModel(
        weights, means_p, covs_p, invcovs_p, dets, mins_p, maxs_p,
        limit=limit, use_cupy=use_cupy,
    )
    return ColumnPermutedProposal(model, _GMM_COLUMN_PERM)
