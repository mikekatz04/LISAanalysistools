"""Astrophysical joint (f0, Mc) prior for galactic binaries.

Wraps the 6-component Gaussian-mixture fit to the combined DWD population
heatmap (102 population-synthesis models; fit in
``scripts/fstat_proposal/heatmap_GMMs.ipynb``, R^2 = 0.9904 against the
combined heatmap) as an eryn duck-typed 2-D distribution over the GB
*sampling basis* columns ``(f0 [mHz], Mc [Msun])``.

The fit lives in ``x = log10(f0 / Hz)`` vs ``Mc`` space; this class handles

* the change of variables to the linear-mHz sampling axis
  (``p(f0_mHz, Mc) = p_gmm(x, Mc) / (f0_mHz * ln 10)``),
* truncation to the run's ``(f0_lims, m_chirp_lims)`` box with exact
  renormalization (a grid quadrature at construction) -- RJ birth/death
  acceptance compares prior masses across leaf counts, so the truncated
  prior must integrate to 1 over the sampled box,
* eryn ``ProbDistContainer`` conventions (``rvs(size) -> size + (2,)``,
  ``logpdf((n, 2)) -> (n,)``, ``use_cupy`` / ``return_gpu`` attributes the
  container toggles).

Register under a tuple key in the GB prior container (done by
:meth:`~lisatools.globalfit.stock.erebor.gb.GBSetup.init_sampling_info` when
``use_chirp_mass`` and ``use_astrophysical_f0_mc_prior`` are both on)::

    ProbDistContainer({..., ("f0", "Mc"): F0McGMMSampling.from_heatmap(
        f0_lims_mHz=(f0_lo_mHz, f0_hi_mHz), mc_lims=(mc_lo, mc_hi)), ...})

Production: both knobs (``GB_USE_CHIRP_MASS``,
``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR``) default on and neither launcher sets
them, so the 9mo / 12mo GB prior on ``(f0, Mc)`` is this class, truncated to
the run's GB ``f0_lims`` (converted to mHz) and ``m_chirp_lims``.

Everything is picklable / deepcopy-safe (plain float arrays only).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["F0McGMMSampling", "HEATMAP_GMM_PARAMS"]

_LN10 = float(np.log(10.0))

# 6-component GMM fit to the combined heatmap (heatmap_GMMs.ipynb, cell
# "Coding the prior"; R^2 = 0.9904). Space: [log10(f0/Hz), Mc/Msun];
# heatmap fit box: log10(f0) in [-4, -1], Mc in [0.1, 1.4].
HEATMAP_GMM_PARAMS = dict(
    weights=[0.2294, 0.1444, 0.2534, 0.0620, 0.1833, 0.1275],
    means=[
        [-3.4941, 0.5846],
        [-2.6715, 0.4930],
        [-3.2362, 0.5526],
        [-2.0510, 0.4993],
        [-3.5154, 0.6689],
        [-2.9433, 0.5609],
    ],
    covs=[
        [[0.010790, -0.001486], [-0.001486, 0.002662]],
        [[0.057879, -0.003847], [-0.003847, 0.003256]],
        [[0.025530, -0.002133], [-0.002133, 0.004554]],
        [[0.101730, -0.009939], [-0.009939, 0.009511]],
        [[0.019447, 0.000336], [0.000336, 0.011663]],
        [[0.035272, 0.000360], [0.000360, 0.013418]],
    ],
    fit_box_logf=(-4.0, -1.0),
    fit_box_mc=(0.1, 1.4),
)


class F0McGMMSampling:
    """Truncated-GMM joint prior over ``(f0 [mHz], Mc [Msun])``.

    The truncation mass ``Z`` over the box is computed once at construction
    by an 801 x 801 trapezoid quadrature in ``(log10(f0/Hz), Mc)`` space, and
    :meth:`logpdf` subtracts ``log Z``, so the density integrates to 1 over
    the box. Evaluation is on the host (numpy).

    Args:
        weights: Mixture weights, shape ``(K,)``; renormalized to sum to 1.
        means: Component means in ``(log10(f0/Hz), Mc)`` space, shape ``(K, 2)``
            (see :data:`HEATMAP_GMM_PARAMS`).
        covs: Component covariances in the same space, shape ``(K, 2, 2)``.
        f0_lims_mHz: Truncation box in the sampling f0 axis (mHz). Default:
            the heatmap fit box (``10**[-4, -1]`` Hz -> ``[0.1, 100]`` mHz).
            Pass the run's band limits so the prior matches the sampled
            space exactly; the Gaussian tails keep the density finite
            anywhere inside the box, including outside the original
            heatmap support.
        mc_lims: Truncation box in Mc (Msun). Default: the heatmap fit box
            ``(0.1, 1.4)``.
        seed: Seed of the :func:`numpy.random.default_rng` stream used by
            :meth:`rvs`. Default ``None`` (OS entropy).

    Raises:
        ValueError: ``f0_lims_mHz`` is not ``0 < lo < hi``, ``mc_lims`` is not
            ``lo < hi``, or the truncation mass is not finite and positive
            (box outside the mixture support).

    Attributes:
        ndim (int): ``2``.
        weights (np.ndarray): Normalized weights, ``(K,)``.
        means (np.ndarray): ``(K, 2)``.
        covs (np.ndarray): ``(K, 2, 2)``.
        f0_lims_mHz (tuple[float, float]): The f0 box (mHz).
        mc_lims (tuple[float, float]): The Mc box (Msun).
        use_cupy (bool): Toggled by eryn's
            :class:`~eryn.priors.ProbDistContainer`; selects :attr:`xp`.
        return_gpu (bool): Toggled by the container; with ``use_cupy``,
            :meth:`rvs` and :meth:`logpdf` return cupy arrays.

    Example:
        >>> from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling
        >>> prior = F0McGMMSampling.from_heatmap(f0_lims_mHz=(1.0, 4.0), mc_lims=(0.1, 1.4), seed=1)
        >>> draws = prior.rvs(5)
        >>> draws.shape, prior.logpdf(draws).shape
        ((5, 2), (5,))
    """

    ndim = 2

    def __init__(
        self,
        weights: Sequence[float],
        means: Sequence[Sequence[float]],
        covs: Sequence[Sequence[Sequence[float]]],
        f0_lims_mHz: Optional[Tuple[float, float]] = None,
        mc_lims: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
    ):
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()
        self.means = np.asarray(means, dtype=float)          # (K, 2)
        self.covs = np.asarray(covs, dtype=float)            # (K, 2, 2)
        self._chol = np.linalg.cholesky(self.covs)
        self._inv = np.linalg.inv(self.covs)
        self._logdet = np.linalg.slogdet(self.covs)[1]

        box_logf = HEATMAP_GMM_PARAMS["fit_box_logf"]
        box_mc = HEATMAP_GMM_PARAMS["fit_box_mc"]
        if f0_lims_mHz is None:
            f0_lims_mHz = (10.0 ** box_logf[0] * 1e3, 10.0 ** box_logf[1] * 1e3)
        if mc_lims is None:
            mc_lims = box_mc
        self.f0_lims_mHz = (float(f0_lims_mHz[0]), float(f0_lims_mHz[1]))
        self.mc_lims = (float(mc_lims[0]), float(mc_lims[1]))
        if not (0 < self.f0_lims_mHz[0] < self.f0_lims_mHz[1]):
            raise ValueError(f"bad f0_lims_mHz {self.f0_lims_mHz}")
        if not (self.mc_lims[0] < self.mc_lims[1]):
            raise ValueError(f"bad mc_lims {self.mc_lims}")

        # exact-enough truncation mass over the box (in (x, Mc) space) by
        # trapezoid quadrature; one-time cost at construction.
        x_lo = np.log10(self.f0_lims_mHz[0] * 1e-3)
        x_hi = np.log10(self.f0_lims_mHz[1] * 1e-3)
        xs = np.linspace(x_lo, x_hi, 801)
        ms = np.linspace(self.mc_lims[0], self.mc_lims[1], 801)
        xx, mm = np.meshgrid(xs, ms, indexing="ij")
        dens = np.exp(self._log_gmm(np.column_stack([xx.ravel(), mm.ravel()])))
        Z = np.trapezoid(
            np.trapezoid(dens.reshape(xx.shape), ms, axis=1), xs, axis=0
        )
        if not np.isfinite(Z) or Z <= 0:
            raise ValueError(
                f"GMM truncation mass is {Z} over f0={self.f0_lims_mHz} mHz, "
                f"Mc={self.mc_lims} -- box outside the mixture support?"
            )
        self._log_trunc = float(np.log(Z))
        self._box_mass = float(Z)

        self._rng = np.random.default_rng(seed)
        # toggled by eryn's ProbDistContainer
        self.use_cupy = False
        self.return_gpu = False

    @classmethod
    def from_heatmap(cls, f0_lims_mHz=None, mc_lims=None, seed=None):
        """Build the fitted combined-heatmap mixture, optionally truncated to a run's box.

        Uses :data:`HEATMAP_GMM_PARAMS`.

        Args:
            f0_lims_mHz (tuple[float, float], optional): f0 box (mHz); default
                the heatmap fit box ``[0.1, 100]`` mHz.
            mc_lims (tuple[float, float], optional): Mc box (Msun); default
                ``(0.1, 1.4)``.
            seed (int, optional): Seed of the :meth:`rvs` stream.

        Returns:
            F0McGMMSampling: The truncated prior.
        """
        return cls(
            HEATMAP_GMM_PARAMS["weights"],
            HEATMAP_GMM_PARAMS["means"],
            HEATMAP_GMM_PARAMS["covs"],
            f0_lims_mHz=f0_lims_mHz,
            mc_lims=mc_lims,
            seed=seed,
        )

    # ------------------------------------------------------------------
    @property
    def xp(self):
        """module: ``cupy`` when :attr:`use_cupy` is set, else ``numpy``."""
        if self.use_cupy:
            import cupy

            return cupy
        return np

    def _log_gmm(self, pts):
        """Return the un-truncated mixture log-density in ``(x=log10 f0_Hz, Mc)``.

        Args:
            pts (np.ndarray): ``(n, 2)`` points in ``(log10(f0/Hz), Mc)``.

        Returns:
            np.ndarray: ``(n,)`` log-densities (numpy in, numpy out).
        """
        diff = pts[:, None, :] - self.means[None]            # (n, K, 2)
        maha = np.einsum("nki,kij,nkj->nk", diff, self._inv, diff)
        log_comp = (
            np.log(self.weights)[None]
            - 0.5 * (2.0 * np.log(2.0 * np.pi) + self._logdet)[None]
            - 0.5 * maha
        )
        m = log_comp.max(axis=1)
        return np.log(np.exp(log_comp - m[:, None]).sum(axis=1)) + m

    def rvs(self, size=1):
        """Draw from the truncated prior in the sampling basis.

        Draws from the mixture in ``(log10(f0/Hz), Mc)`` space, maps f0 to mHz,
        and keeps only the draws inside the box (rejection), batching by the
        box mass until ``size`` draws are filled.

        Args:
            size (int or tuple[int, ...], optional): Output batch shape.
                Default ``1``.

        Returns:
            np.ndarray: Shape ``size + (2,)``, columns ``(f0 [mHz], Mc [Msun])``;
            a cupy array when ``use_cupy`` and ``return_gpu`` are both set.
        """
        if isinstance(size, int):
            size = (size,)
        n = int(np.prod(size))
        out = np.empty((n, 2))
        filled = 0
        # batch size adapted to the box acceptance rate
        batch = max(64, int(1.5 * n / max(self._box_mass, 1e-6)))
        batch = min(batch, 2_000_000)
        while filled < n:
            comp = self._rng.choice(len(self.weights), size=batch,
                                    p=self.weights)
            z = self._rng.standard_normal((batch, 2))
            pts = self.means[comp] + np.einsum(
                "nij,nj->ni", self._chol[comp], z
            )
            f0_mHz = 10.0 ** pts[:, 0] * 1e3
            keep = (
                (f0_mHz >= self.f0_lims_mHz[0])
                & (f0_mHz <= self.f0_lims_mHz[1])
                & (pts[:, 1] >= self.mc_lims[0])
                & (pts[:, 1] <= self.mc_lims[1])
            )
            k = int(keep.sum())
            take = min(k, n - filled)
            if take:
                sel = np.flatnonzero(keep)[:take]
                out[filled:filled + take, 0] = f0_mHz[sel]
                out[filled:filled + take, 1] = pts[sel, 1]
                filled += take
        out = out.reshape(size + (2,))
        if self.use_cupy and self.return_gpu:
            return self.xp.asarray(out)
        return out

    def logpdf(self, x):
        r"""Return the normalized log prior at ``(f0 [mHz], Mc)`` points.

        .. math::

            \ln p(f_{0,\mathrm{mHz}}, M_c) = \ln p_\mathrm{gmm}(\log_{10} f_{0,\mathrm{Hz}}, M_c)
            - \ln(f_{0,\mathrm{mHz}} \ln 10) - \ln Z

        Computed on the host. Accepts numpy or cupy input.

        Args:
            x (np.ndarray or cupy.ndarray): ``(n, 2)`` points (a single
                ``(2,)`` point is promoted to ``(1, 2)``).

        Returns:
            np.ndarray or cupy.ndarray: ``(n,)`` log prior, ``-inf`` outside
            the truncation box. The result is converted with :attr:`xp` when
            the input is cupy or when ``use_cupy`` and ``return_gpu`` are set,
            so a cupy input comes back as numpy unless ``use_cupy`` is on.
        """
        xp_in = None
        if type(x).__module__.startswith("cupy"):
            xp_in = self.xp
            x = x.get()
        x = np.atleast_2d(np.asarray(x, dtype=float))
        f0_mHz = x[:, 0]
        mc = x[:, 1]
        inside = (
            (f0_mHz >= self.f0_lims_mHz[0])
            & (f0_mHz <= self.f0_lims_mHz[1])
            & (mc >= self.mc_lims[0])
            & (mc <= self.mc_lims[1])
            & (f0_mHz > 0)
        )
        safe_f0 = np.where(f0_mHz > 0, f0_mHz, 1.0)
        pts = np.column_stack([np.log10(safe_f0 * 1e-3), mc])
        out = (
            self._log_gmm(pts)
            - np.log(safe_f0 * _LN10)   # d log10(f0_Hz) / d f0_mHz
            - self._log_trunc
        )
        out = np.where(inside, out, -np.inf)
        if xp_in is not None or (self.use_cupy and self.return_gpu):
            return self.xp.asarray(out)
        return out
