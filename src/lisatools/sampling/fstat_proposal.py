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
from typing import Optional, Tuple

import numpy as np

__all__ = ["GridSpec", "FStatProposal4D", "compute_fstat"]


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
