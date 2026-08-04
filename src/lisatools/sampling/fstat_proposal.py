"""F-statistic proposal distributions over the GB intrinsic parameters.

Implements "Family A" of ``scripts/fstat_proposal/PLAN_scattered_field_
distributions.md``: a structured tensor grid over the 4 intrinsic sampling
parameters, filled with batched F-statistic evaluations, turned into an
exact-sampling distribution via the (flattened) conditional inverse-CDF.

The target density is

    p(theta) ∝ exp( beta * F(theta) )        (masked to the grid box)

with ``F = 0.5 * N^T M^{-1} N`` the Cornish-Crowder F-statistic returned by
:meth:`lisatools.chunked_het.WDMComputationsBase.get_fstat_ll_wdm` as the
per-binary ``(N, M)`` pair. F is a function of the 4 *intrinsic* parameters
only -- the 4 extrinsic amplitude parameters ``(A, iota, psi, phi0)`` are
analytically maximized inside the statistic.

Sampling basis (matches the stock erebor GB chirp-mass basis):

    theta = (f0 [mHz], Mc [Msol], alpha [rad], sin_delta)

``fdot`` is recovered from ``(f0, Mc)`` via the monochromatic-GB relation
(:func:`gbgpu.utils.utility.get_fdot`) when packing the physical 9-parameter
waveform rows for the kernel, so the proposal lives natively in the chirp-mass
sampling basis used by :class:`lisatools.globalfit.stock.erebor.gb.GBSetup`.

TODO (fdot proposal / interacting DWDs): the ``(f0, Mc)`` basis can only
propose ``fdot > 0`` (``Mc > 0`` -> ``fdot > 0`` through the GW relation), so
the F-stat proposal structurally cannot birth the interacting DWDs that have
``fdot < 0`` (e.g. mass transfer). We should examine doing the F-stat proposal
directly in ``(f0, fdot)`` instead of ``(f0, Mc)`` so negative fdot is
representable -- making sure the fdot grid range is SCALED to each sub-band's
frequency (the physical fdot bounds are f0-dependent; cf.
``lisatools.globalfit.priors.gbpriors.get_fdot_mojito(f, sign=+/-)`` which
gives the +/- fdot envelope per f0). Pairs with the legacy-basis sampling
route and the ``fdot < 0`` seeding TODO in
``recipe.setup_state_for_injection``. For now the workaround is to run
sub-bands that contain no ``fdot < 0`` catalogue sources (or subtract them as
known signals).

The distribution object is eryn duck-typed -- ``rvs(size) -> size + (4,)``
and ``logpdf((n, 4)) -> (n,)`` -- so it registers directly in a
``ProbDistContainer`` under a tuple key, e.g.
``{("f0", "Mc", "alpha", "sin_delta"): proposal}``.

Everything is ``xp``-agnostic: the array module is taken from the supplied
``gb_wdm_comp`` (numpy on CPU, cupy on CUDA backends). Per the sprint
deepcopy/pickle rule no array module is stored as an attribute.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "GridSpec",
    "FStatProposal4D",
    "StackedFStatProposal4D",
    "UniformFloorMixture",
    "MixtureProposal",
    "CombIntrinsicProposal",
    "ColumnPermutedProposal",
    "compute_fstat",
    "fstat_maximized_extrinsics",
    "make_gb_rj_birth_container",
    "fit_gmm_to_stacked",
    "pack_gmm_components",
    "unpack_gmm_components",
    "build_peak_gmm",
    "FSTAT_KNOB_DEFAULTS",
    "fstat_knob",
    "fstat_peak_min_F",
    "fstat_n_f0",
    "fstat_n_axis",
]


# ======================================================================
# FSTAT_* environment knobs -- the single source of truth for defaults.
# The grid-prep script, the search runner, and the selection helpers all
# resolve knobs through here; an explicit environment value always wins.
# ======================================================================
FSTAT_KNOB_DEFAULTS = {
    "FSTAT_BATCH": 4096,           # kernel rows per get_fstat_ll_wdm call
    "FSTAT_PEAK_MIN_SNR": 5.0,     # selection floor (F = SNR^2 / 2)
    "FSTAT_PEAKS_PER_BAND": 35,    # per-sub-band peak cap
    "FSTAT_PEAK_HALF_MHZ": 2.5e-3, # peak-box f0 half width [mHz]
    "FSTAT_MC_MIN": 0.01,          # Mc grid-box floor
    "FSTAT_N_MC": 3,               # anisotropic node counts: Mc / sky are
    "FSTAT_N_ALPHA": 8,            #   per-band unmeasurable, so coarse
    "FSTAT_N_SINDELTA": 8,
    "FSTAT_COMB_NSKY": 6,          # comb sky points
    "FSTAT_FLOOR_EPS": 0.1,        # uniform-floor mixture weight
    "FSTAT_COMB_WEIGHT": 0.3,      # comb component weight. 2026-07 study
                                   #   (band75): unboxed-source coverage
                                   #   scales ~linearly with w while boxed
                                   #   mass drops only ~9% at 0.3 (the
                                   #   linear-in-F comb concentrates on the
                                   #   same loud sources); off-source waste
                                   #   flat ~5.5% (floor-only) at any w.
    "FSTAT_PEAK_SAMPLING": "grid", # production draw layer (gmm = option)
    "FSTAT_FIT_GMM": 0,            # prep-time GMM fit is OPT-IN
    "FSTAT_GMM_SAMPLES": 4096,     # GMM fit: draws per box
    "FSTAT_GMM_MAX_COMP": 12,      # GMM fit: max components per box
    "FSTAT_PLOT_PEAKS": 2,         # prep: corner plots for the top boxes
}


def fstat_knob(name: str, cast=float):
    """Resolve an ``FSTAT_*`` env knob against :data:`FSTAT_KNOB_DEFAULTS`
    (explicit environment value wins)."""
    raw = os.environ.get(name, "").strip()
    if raw:
        return cast(raw)
    return cast(FSTAT_KNOB_DEFAULTS[name])


def fstat_peak_min_F() -> float:
    """Peak-selection floor in F units (``F = SNR^2 / 2``).

    Precedence: explicit ``FSTAT_PEAK_MIN_SNR`` > explicit
    ``FSTAT_PEAK_MIN_F`` > the default SNR
    (``FSTAT_KNOB_DEFAULTS['FSTAT_PEAK_MIN_SNR']`` = 5, i.e. F = 12.5).
    """
    snr = os.environ.get("FSTAT_PEAK_MIN_SNR", "").strip()
    if snr:
        return 0.5 * float(snr) ** 2
    f = os.environ.get("FSTAT_PEAK_MIN_F", "").strip()
    if f:
        return float(f)
    return 0.5 * float(FSTAT_KNOB_DEFAULTS["FSTAT_PEAK_MIN_SNR"]) ** 2


def fstat_n_f0(box_width_mHz: float, Tobs_s: float) -> int:
    """f0 node count for a peak box.

    Precedence: explicit ``FSTAT_N_F0`` > explicit ``FSTAT_N_PER_AXIS`` >
    AUTO -- one cell per ~1/Tobs (the matched-filter peak width), clamped to
    ``[12, 96]`` (~40 for the default +-2.5e-3 mHz box at 90 d). The f0
    cell width is the proposal's sharpest efficiency lever: cells much wider
    than the peak spread birth mass off-source.
    """
    raw = os.environ.get("FSTAT_N_F0", "").strip()
    if raw:
        return int(raw)
    per = os.environ.get("FSTAT_N_PER_AXIS", "").strip()
    if per:
        return int(per)
    cells = round(float(box_width_mHz) / (1e3 / float(Tobs_s)))
    return int(np.clip(cells + 1, 12, 96))


def fstat_n_axis(name: str) -> int:
    """Node count for the Mc / alpha / sin_delta axes.

    Precedence: explicit ``FSTAT_N_MC``/``_ALPHA``/``_SINDELTA`` > explicit
    ``FSTAT_N_PER_AXIS`` > the anisotropic defaults (3 / 8 / 8).
    """
    raw = os.environ.get(name, "").strip()
    if raw:
        return int(raw)
    per = os.environ.get("FSTAT_N_PER_AXIS", "").strip()
    if per:
        return int(per)
    return int(FSTAT_KNOB_DEFAULTS[name])


def _host(x):
    """cupy/numpy array -> host numpy (no-op on numpy)."""
    return x.get() if hasattr(x, "get") else np.asarray(x)


# Row-major upper-triangle layout of the symmetric (4, 4) filter Gram matrix
# as returned by ``get_fstat_ll_wdm``: [M00, M01, M02, M03, M11, M12, M13,
# M22, M23, M33].
_TRIU_ROWS = (0, 0, 0, 0, 1, 1, 1, 2, 2, 3)
_TRIU_COLS = (0, 1, 2, 3, 1, 2, 3, 2, 3, 3)


def compute_fstat(N_arr, M_upper, ridge: float = 1e-12):
    """``F = 0.5 * N^T M^{-1} N`` from the batched ``(N, M)`` F-stat pieces.

    Args:
        N_arr: ``(num_bin, 4)`` data-filter inner products ``<d | A_i>``.
        M_upper: ``(num_bin, 10)`` row-major upper triangle of the symmetric
            filter Gram matrix ``<A_i | A_j>``.
        ridge: Fractional Tikhonov regularization added to the diagonal
            (scaled by the mean diagonal per binary) so grid cells with a
            near-singular Gram matrix (e.g. dead sky/frequency corners)
            return a finite, tiny F instead of raising.

    Returns:
        ``(num_bin,)`` F-statistic values (same array module as the inputs;
        non-finite results are mapped to ``-inf`` so downstream
        ``exp(beta * F)`` weights vanish cleanly).
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
    """``(A_max, phi0_max, iota_max, psi_max, F)`` from the F-stat ``(N, M)``.

    Ports the Jaranowski-Krol amplitude-vector inversion from
    ``gbgpu.GBGPU.get_fstat_ll`` to the ``(N, M)`` pieces returned by the new
    ``GBFDComputations.get_fstat_ll_fd`` / ``GBWDMComputations.get_fstat_ll_wdm``
    -- which build the SAME 4 basis filters at the SAME fixed reference
    ``(A, iota, psi, phi0) = (2, pi/2, {0,pi/4,0,pi/4}, {0,pi,3pi/2,pi/2})`` and
    return the SAME 10-element upper-triangle ``M`` layout, so the inversion is
    identical. ``a = M^-1 N`` is the ML amplitude vector; ``A_max`` is the
    physical maximized amplitude (reference A=2 baked into the filters, matching
    the legacy formula) and ``F = 0.5 * a . N``.

    Numerically guarded (ridge on ``M``, non-negative discriminant, clipped
    ``arccos``) so near-singular / off-peak births return finite maxima instead
    of NaN. Same array module as the inputs.

    TODO (known F-stat adjustment): the F-stat computation needs a slight
    adjustment (a normalization/scale correction we already know about) before
    ``A_max`` here is exactly the physical maximized amplitude. It is not yet
    applied; add it in this one place once finalized so both the FD and WDM
    paths (which share this routine) inherit it. Until then treat ``A_max`` /
    ``dist*`` as approximate -- fine as a proposal center, but verify the
    absolute distance scale after the adjustment lands.
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

    r1 = xp.sqrt((a1 + a4) ** 2 + (a2 - a3) ** 2)
    r2 = xp.sqrt((a1 - a4) ** 2 + (a2 + a3) ** 2)
    A_plus = r1 + r2
    A_cross = r1 - r2
    disc = xp.sqrt(xp.clip(A_plus ** 2 - A_cross ** 2, 0.0, None))

    A_max = (A_plus + disc) / 2.0
    psi_max = (0.5 * xp.arctan2(
        A_plus * a4 - A_cross * a1, -(A_cross * a2 + A_plus * a3))) % np.pi
    iota_max = xp.arccos(
        xp.clip(-A_cross / xp.clip(A_plus + disc, 1e-300, None), -1.0, 1.0)) % np.pi
    c = xp.sign(xp.sin(2.0 * psi_max))
    phi0_max = xp.arctan2(
        c * (A_plus * a4 - A_cross * a1),
        -c * (A_cross * a2 + A_plus * a3)) % (2.0 * np.pi)

    A_max = xp.where(xp.isfinite(A_max), A_max, 0.0)
    F = xp.where(xp.isfinite(F), F, -xp.inf)
    return A_max, phi0_max, iota_max, psi_max, F


@dataclasses.dataclass
class GridSpec:
    """Tensor-grid design over the 4 intrinsic sampling parameters.

    Ranges are the closed box edges; ``n_*`` are the node counts per axis
    (``n`` nodes -> ``n - 1`` cells). Units match the sampling basis:
    ``f0`` in mHz, ``Mc`` in solar masses, ``alpha`` in radians (ICRS RA),
    ``sin_delta`` dimensionless in ``[-1, 1]``.
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
        return (
            np.linspace(*map(float, self.f0_range), int(self.n_f0)),
            np.linspace(*map(float, self.Mc_range), int(self.n_Mc)),
            np.linspace(*map(float, self.alpha_range), int(self.n_alpha)),
            np.linspace(*map(float, self.sin_delta_range), int(self.n_sin_delta)),
        )

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (int(self.n_f0), int(self.n_Mc), int(self.n_alpha), int(self.n_sin_delta))


class FStatProposal4D:
    """Grid + inverse-CDF proposal over ``(f0 [mHz], Mc, alpha, sin_delta)``.

    Construction sweeps the F-statistic over the :class:`GridSpec` nodes in
    batched ``get_fstat_ll_wdm`` calls, stores ``g = beta * F`` on the node
    grid, and precomputes the flattened cell CDF. The represented density is
    piecewise-constant on the ``(n - 1)^4`` cells (each cell carries its
    lower-corner node's ``g``), which makes ``rvs`` and ``logpdf`` *exactly*
    consistent with each other.

    Args:
        gb_wdm_comp: Object providing ``get_fstat_ll_wdm(params, wdm_holder)
            -> (N, M_upper)`` -- normally a ``gbgpu.gbcomps.GBWDMComputations``
            (any object with that method works, e.g. the mock in
            ``scripts/fstat_proposal/plot_fstat_proposal_mock_highest_gb.py``).
        wdm_holder: Passed through untouched to ``get_fstat_ll_wdm`` (an
            ``AnalysisContainerArray`` holding the WDM residual + invC).
        grid_spec: The tensor-grid design box + resolution.
        beta: Inverse temperature; ``beta < 1`` broadens the proposal for
            healthier MCMC acceptance, ``beta > 1`` sharpens it.
        amp_ref: Amplitude packed into the physical parameter rows. The
            F-stat maximizes over the extrinsics analytically, so this only
            has to be a sane waveform amplitude, not the source's.
        batch_size: Grid nodes per ``get_fstat_ll_wdm`` launch.
        seed: RNG seed for ``rvs`` (fresh ``default_rng`` when None).
        fstat_kwargs: Extra keyword arguments forwarded to
            ``get_fstat_ll_wdm`` (e.g. ``m_band_half_width``).
    """

    #: sampling-basis column names, in order
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

        Args:
            axes: 4 node arrays ``(f0_mHz, Mc, alpha, sin_delta)`` --
                uniformly spaced, as produced by :class:`GridSpec`.
            logp_grid: node-shaped ``beta * F`` array (what a prior
                instance stored in ``_logp_grid`` / a grid cache).
            beta: the inverse temperature already applied to ``logp_grid``
                (recorded for bookkeeping; the grid is used as-is).
            seed: RNG seed for :meth:`rvs`.
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
        return getattr(self.gb_wdm_comp, "xp", np)

    def __getstate__(self):
        # The proposal is a runtime object built around live kernel/data
        # handles; strip them so a pickled copy keeps the fitted grid and
        # stays usable for rvs/logpdf (rebuild to re-sweep the F-stat).
        state = self.__dict__.copy()
        state["gb_wdm_comp"] = None
        state["wdm_holder"] = None
        return state

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _pack_physical(self, theta):
        """Sampling-basis ``(n, 4)`` -> physical 9-parameter GB rows ``(n, 9)``.

        Physical order: ``[A, f0(Hz), fdot, fddot, phi0, iota, psi, lam,
        beta_sky]``. Extrinsic slots carry fixed reference values -- the
        F-stat kernel builds its own 4 basis-filter extrinsic combinations.
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
        """Batched F-stat over every grid node -> node-shaped array."""
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
        consistent).
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

        Returns ``size + (4,)`` in the sampling basis
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
        """Normalized log density at ``x`` of shape ``(n, 4)`` (``-inf``
        outside the grid box), exactly consistent with :meth:`rvs`."""
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

    Args:
        logp_grids: ``(K, n0, n1, n2, n3)`` node grids of ``beta * F``
            (numpy or cupy; computation stays on the input's module).
        f0_los: ``(K,)`` per-box f0 axis start [mHz].
        f0_dxs: ``(K,)`` per-box f0 node spacing [mHz].
        mc_ax, alpha_ax, sin_delta_ax: shared node axes (uniform spacing).
        weights: ``(K,)`` mixture weights (normalized internally);
            ``None`` -> equal.
        seed: RNG seed for :meth:`rvs`.
        mem_budget_mb: When set and the stacked cell arrays exceed this
            budget, the K axis is held/evaluated in chunks that fit it
            (``FSTAT_GRID_MEM_MB`` in the scripts). ``rvs``/``logpdf``
            stay fully vectorized within each chunk.
    """

    param_names = ("f0", "Mc", "alpha", "sin_delta")
    ndim = 4

    def __init__(self, logp_grids, f0_los, f0_dxs, mc_ax, alpha_ax,
                 sin_delta_ax, weights=None, seed: Optional[int] = None,
                 mem_budget_mb: Optional[float] = None):
        from ..utils.utility import get_array_module

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

        # host metadata (small)
        self._f0_lo = np.asarray(_host(f0_los), dtype=float).ravel()
        self._f0_dx = np.asarray(_host(f0_dxs), dtype=float).ravel()
        assert self._f0_lo.shape == (K,) and self._f0_dx.shape == (K,)
        self._f0_hi = self._f0_lo + (node_shape[0] - 1) * self._f0_dx
        axes3 = [np.asarray(_host(a), dtype=float) for a in
                 (mc_ax, alpha_ax, sin_delta_ax)]
        self._lo3 = np.array([a[0] for a in axes3])
        self._hi3 = np.array([a[-1] for a in axes3])
        self._dx3 = np.array([a[1] - a[0] for a in axes3])
        self._axes3 = axes3

        w = (np.ones(K) if weights is None
             else np.asarray(_host(weights), dtype=float).ravel())
        assert w.shape == (K,) and np.all(w >= 0) and w.sum() > 0
        self.weights = w / w.sum()

        # K-chunking: per-box cell working set ~ log_wcell + cdf.
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
                log_wcell = xp.log(p) + gmax[:, None, None, None, None]
            total = p.reshape(p.shape[0], -1).sum(axis=1)  # (Kc,)
            total_h = _host(total)
            cell_vol = self._f0_dx[k0:k1] * float(np.prod(self._dx3))
            with np.errstate(divide="ignore"):
                log_norm[k0:k1] = (_host(gmax) + np.log(total_h)
                                   + np.log(cell_vol))
            # sampling mass per cell: w_k * (cell weight / box total)
            scale = xp.asarray(self.weights[k0:k1] / np.clip(total_h, 1e-300, None))
            mass = (p.reshape(p.shape[0], -1) * scale[:, None]).ravel()
            cdf = xp.cumsum(mass) + running_mass
            running_mass = float(_host(cdf[-1]))
            self._chunks.append(dict(k0=k0, k1=k1, log_wcell=log_wcell, cdf=cdf))
        if not np.isfinite(running_mass) or running_mass <= 0.0:
            raise ValueError(
                "StackedFStatProposal4D: zero/non-finite total mixture mass "
                "(every F-stat grid cell was -inf?)"
            )
        for ch in self._chunks:
            ch["cdf"] = ch["cdf"] / running_mass
        self._chunk_cum = np.array(
            [float(_host(ch["cdf"][-1])) for ch in self._chunks]
        )
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
        from ..utils.utility import get_array_module

        return get_array_module(self._chunks[0]["log_wcell"])

    @classmethod
    def from_cache(cls, d, weights=None, seed: Optional[int] = None,
                   mem_budget_mb: Optional[float] = None,
                   use_cupy: bool = False, device: Optional[int] = None):
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
        )

    # ------------------------------------------------------------------
    def _corners_from_cells(self, kk, cell_flat, xp):
        """(box index, flat cell index) -> cell corner coords ``(n, 4)``."""
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

    def rvs(self, size=1):
        """Exact draws from the stacked piecewise-constant mixture; returns
        ``size + (4,)`` in ``(f0 [mHz], Mc, alpha, sin_delta)``."""
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
            uu = xp.asarray(u[m])
            flat = xp.searchsorted(ch["cdf"], uu, side="right")
            flat = xp.clip(flat, 0, ch["cdf"].shape[0] - 1)
            k_local = flat // self._ncells
            cell = flat - k_local * self._ncells
            kk = k_local + ch["k0"]
            corners, f0_dx = self._corners_from_cells(kk, cell, xp)
            j = xp.asarray(jit[m])
            corners[:, 0] += j[:, 0] * f0_dx
            corners[:, 1:] += j[:, 1:] * xp.asarray(self._dx3)[None, :]
            out[xp.asarray(np.where(m)[0])] = corners
        return out.reshape(size + (4,))

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
        """Normalized mixture log density at ``x`` of shape ``(n, 4)``;
        vectorized gather over the (actual) max f0-overlap depth D."""
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

        lp = xp.full((n, D), -xp.inf, dtype=xp.float64)
        log_w = xp.asarray(np.log(np.clip(self.weights, 1e-300, None)))
        log_norm = xp.asarray(self._log_norm)
        for ch in self._chunks:
            sel = valid & (kk >= ch["k0"]) & (kk < ch["k1"])
            if not bool(sel.any()):
                continue
            rows, cols = xp.where(sel)
            k_sel = kk[rows, cols]
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


class UniformFloorMixture:
    """``(1 - eps) * base + eps * Uniform(box)`` over the 4 intrinsics.

    RJ detailed-balance factors evaluate ``+logpdf(current leaf)`` for a
    death: a leaf whose in-model refinement drifted outside the base
    proposal's support would get ``-inf`` there and could never die. Mixing
    in a small uniform floor over the full per-band prior box keeps
    ``logpdf`` finite everywhere the sampler can go (the plan's
    "mixture + background floor").

    Args:
        base: Intrinsic 4-D distribution (``rvs``/``logpdf`` duck-typed),
            e.g. :class:`FStatProposal4D`.
        box_lo, box_hi: Per-axis floor-box bounds in the sampling basis
            ``(f0 [mHz], Mc, alpha, sin_delta)`` -- normally the band's f0
            edges plus the full prior ranges of the other three axes.
        eps: Floor mixture weight.
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

    def rvs(self, size=1):
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
                               fdot_astro_ratio_max=None, dist_lims=None):
    """Wrap a 4-D intrinsic proposal into the 8/9-column GB RJ birth container.

    Mirrors the stock GMM birth container
    (``gbspecialstretch.GBSpecialRJRefitMove.setup``): the intrinsics
    ``(f0 [mHz], Mc, alpha, sin_delta)`` come from ``intrinsic_dist`` under a
    tuple key; slot 0 (``lnA`` or ``dist``) and the remaining extrinsics come
    from the stock prior uniforms (the band engines re-maximize phi0
    analytically when ``phase_maximize`` is on). Returns an eryn
    ``ProbDistContainer`` whose ``rvs(size) -> (size, N)`` /
    ``logpdf((n, N)) -> (n,)`` match what ``BandSorter`` expects of
    ``rj_proposal_distribution["gb"]`` (``N = 8``, or ``9`` with
    ``fdot_astro_ratio_max``).

    Args:
        intrinsic_dist: 4-D distribution over the intrinsic sampling basis
            (e.g. :class:`FStatProposal4D` or :class:`UniformFloorMixture`).
        A_lims: ``[A_min, A_max]`` physical amplitude limits
            (``GBSettings.A_lims``); lnA is drawn uniform in ``log(A_lims)``
            when ``dist_lims`` is None.
        use_cupy: Match the run backend (False for CPU runs).
        fdot_astro_ratio_max: When not ``None``, append a 9th
            ``fdot_astro_ratio`` column drawn from ``U[-M, M]`` (births draw
            the ratio from its prior; it is degenerate with Mc under the
            F-stat). ``None`` -> 8-column container.
        dist_lims: When not ``None``, slot 0 is the luminosity DISTANCE
            (kpc) drawn uniform in ``dist_lims`` (linear) instead of lnA --
            the distance basis. Follow-up: when the real 3-D (dist, sky)
            distribution lands, births should draw ``dist | sky`` from it.
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
        ("f0", "Mc", "alpha", "sin_delta"): intrinsic_dist,
        "phi0": UniformDistribution(0.0, 2.0 * np.pi),
        "cos_iota": UniformDistribution(-1.0, 1.0),
        "psi": UniformDistribution(0.0, np.pi),
    }
    key_order = [slot0_name, "f0", "Mc", "phi0", "cos_iota", "psi",
                 "alpha", "sin_delta"]
    if fdot_astro_ratio_max is not None:
        M = float(fdot_astro_ratio_max)
        priors_in["fdot_astro_ratio"] = UniformDistribution(-M, M)
        key_order.append("fdot_astro_ratio")
    dist = ProbDistContainer(priors_in, use_cupy=use_cupy)
    # reset_key_order re-maps rvs/logpdf columns to the sampler layout
    # (a bare ``key_order = [...]`` assignment would NOT re-map).
    dist.reset_key_order(key_order)
    return dist


class MixtureProposal:
    """Weighted mixture of 4-D intrinsic proposals (``rvs``/``logpdf``)."""

    ndim = 4

    def __init__(self, components, weights=None, seed: Optional[int] = None):
        self.components = list(components)
        w = (np.ones(len(self.components)) if weights is None
             else np.asarray(weights, dtype=float))
        self.weights = w / w.sum()
        self._rng = np.random.default_rng(seed)

    def rvs(self, size=1):
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
    """

    ndim = 4

    def __init__(self, f0_nodes_mHz, F_max, mc_lims, alpha_lims=(0.0, 2 * np.pi),
                 sin_delta_lims=(-1.0, 1.0), power: float = 1.0,
                 seed: Optional[int] = None):
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
        self._log_uni = -sum(
            np.log(hi - lo) for lo, hi in
            (self.mc_lims, self.alpha_lims, self.sin_delta_lims)
        )
        self._rng = np.random.default_rng(seed)

    def rvs(self, size=1):
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        idx = np.searchsorted(self._cdf, self._rng.random(n), side="right")
        idx = np.clip(idx, 0, len(self._w) - 1)
        u = self._rng.random(n)
        out = np.empty((n, 4))
        out[:, 0] = (self.f0_nodes[idx]
                     + u * (self.f0_nodes[idx + 1] - self.f0_nodes[idx]))
        out[:, 1] = self._rng.uniform(*self.mc_lims, n)
        out[:, 2] = self._rng.uniform(*self.alpha_lims, n)
        out[:, 3] = self._rng.uniform(*self.sin_delta_lims, n)
        return out.reshape(size + (4,))

    def logpdf(self, x):
        from ..utils.utility import get_array_module

        xp = get_array_module(x)
        x = xp.atleast_2d(xp.asarray(x, dtype=xp.float64))
        f0 = x[:, 0]
        f0_nodes = xp.asarray(self.f0_nodes)
        idx = xp.searchsorted(f0_nodes, f0, side="right") - 1
        in_f0 = (f0 >= f0_nodes[0]) & (f0 <= f0_nodes[-1])
        idx = xp.clip(idx, 0, len(self._w) - 1)
        inside = (
            in_f0
            & (x[:, 1] >= self.mc_lims[0]) & (x[:, 1] <= self.mc_lims[1])
            & (x[:, 2] >= self.alpha_lims[0]) & (x[:, 2] <= self.alpha_lims[1])
            & (x[:, 3] >= self.sin_delta_lims[0])
            & (x[:, 3] <= self.sin_delta_lims[1])
        )
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

    def rvs(self, size=1):
        out = self.base.rvs(size=size)
        return out[..., list(self._inv)]

    def logpdf(self, x):
        return self.base.logpdf(x[..., list(self._perm)])


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
    """Flatten ragged per-box GMM component lists into npz-storable arrays."""
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
    """Inverse of :func:`pack_gmm_components` (accepts an npz mapping)."""
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
