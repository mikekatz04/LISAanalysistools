"""Prior distributions and density helpers for LISA sampling.

Global-fit (9mo / 12mo run path) classes:

- :class:`LogUniformLinear`: the log-uniform prior on a LINEAR column; the
  stock MBH mass-ratio column ``Q = m1/m2`` on ``[1, 10]``.
- :class:`EMRIKerrDomainPrior`: the EMRI prior container with the FEW Kerr
  eccentric-equatorial domain-of-validity cut.

General use: :class:`FullGaussianMixtureModel` (the concatenated per-band
Gaussian mixture built from GB posterior fits; see
:mod:`lisatools.sampling.gmm`; its global-fit builders, the GB GMM refit move
and :func:`~lisatools.sampling.gmm.fit_gb_gmm_rj_container`, are not staged
in the 9mo / 12mo runs) and :class:`SNRPrior` (the analytic SNR prior, LATW
tutorial 07).
The amplitude-from-SNR family (:class:`AmplitudeFromSNR`,
:class:`AmplitudeFrequencySNRPrior`, :class:`GBPriorWrap`) is legacy with no
live callers.
"""

import numpy as np
from eryn.moves.multipletry import logsumexp
from eryn.prior import ProbDistContainer
from eryn.priors import UniformDistribution
from scipy import stats

from ..sensitivity import get_sensitivity
from ..utils.constants import *
from ..utils.utility import asnumpy, get_array_module

try:
    from ..cutils.psd_gpu import compute_logpdf

except (ModuleNotFoundError, ImportError) as e:
    pass

import sys
from typing import List, Optional, Tuple, Union

sys.path.append("/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/gf_search/")
# from galaxy_ffdot import GalaxyFFdot
# from galaxy import Galaxy

try:
    import cupy as cp

except (ModuleNotFoundError, ImportError) as e:
    pass


class LogUniformLinear(UniformDistribution):
    """Log-uniform prior that lives on the LINEAR column: ``p(x) = 1 / (x ln(max/min))``.

    Why this exists (2026-09-16). Eryn's prior rewrite (eryn commit
    ``20ff728``, 2026-07-05) redefined :class:`eryn.priors.analytical.LogUniform`
    as a *sampling-space* distribution: it stores ``minimum = ln(min)`` /
    ``maximum = ln(max)`` and its ``rvs`` / ``pdf`` / ``logpdf`` all work in
    ``u = ln x``; the physical (linear) column is reached only through the
    ``*_physical`` variants. :class:`~eryn.priors.probdist.ProbDistContainer`
    calls the plain ``rvs`` / ``logpdf``, so a column spelled
    ``log_uniform(a, b)`` is drawn and scored in ``ln`` units while every
    consumer downstream still reads it as a linear value.

    That silently broke the stock MBH mass-ratio column ``Q = m1/m2 >= 1``,
    whose transform (:func:`lisatools.globalfit.stock.erebor.transforms.mT_Q`)
    asserts ``m1 >= m2``: with ``log_uniform(1., 10.)`` the support became
    ``[0, ln 10] = [0, 2.30]``, 43 % of prior draws (``Q < 1``) raised inside
    the transform, and every ``Q`` in ``(2.30, 10]`` scored ``-inf``.

    This class is the linear-column log-uniform those call sites always
    meant: ``minimum`` / ``maximum`` / ``width`` report the PHYSICAL bounds
    (so :func:`lisatools.globalfit.moves.eigen_refresh.prior_box_widths`
    whitens the eigen tables with the real box width, 9.0 for ``Q``), ``rvs``
    returns ``exp(U(ln min, ln max))`` in ``[min, max]``, and ``logpdf`` is
    ``-ln x - ln(ln(max/min))`` inside the box, ``-inf`` outside.

    Call sites:

    * ``lisatools.globalfit.stock.erebor.mbh.MBHSetup.init_sampling_info``
      (``"Q": LogUniformLinear(1.0, 10.0)``)
    * ``lisatools.globalfit.stock.erebor.source_runtime.prepare_mbh_branch``
      (same column)

    Do NOT re-implement this by subclassing eryn's ``LogUniform``: its
    ``minimum`` / ``maximum`` are the ln-space bounds, and the base
    ``logpdf_physical`` calls ``self.logpdf`` (overriding ``logpdf`` there
    would recurse through the forward transform).

    Args:
        minimum: Lower bound of the physical column; must be ``> 0``.
        maximum: Upper bound of the physical column; must be ``> minimum``.
        name: Optional parameter name, passed to the base class as ``name``.
            Default ``None``.
        **kwargs: Forwarded to :class:`eryn.priors.analytical.UniformDistribution`
            (``use_cupy``, ``return_gpu``, ...).

    Raises:
        ValueError: If ``minimum <= 0`` or ``maximum <= minimum``.

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.prior import LogUniformLinear
        >>> prior = LogUniformLinear(1.0, 10.0)
        >>> prior.minimum, prior.maximum
        (1.0, 10.0)
        >>> bool(np.isneginf(prior.logpdf(np.array([0.5]))[0]))
        True
    """

    def __init__(self, minimum: float, maximum: float, name: str | None = None, **kwargs):
        minimum = float(minimum)
        maximum = float(maximum)
        kwargs["name"] = name  # keep the base's positional (minimum, maximum, name) order
        if minimum <= 0:
            raise ValueError(
                f"LogUniformLinear minimum must be strictly positive (got {minimum})."
            )
        if maximum <= minimum:
            raise ValueError(
                f"LogUniformLinear maximum {maximum} must exceed minimum {minimum}."
            )

        # PHYSICAL bounds on purpose: ``minimum``/``maximum``/``width`` are the
        # linear box. The uniform ``pdf_val``/``logpdf_val`` the base sets from
        # them are unused here -- ``pdf``/``logpdf`` are overridden below.
        super().__init__(minimum=minimum, maximum=maximum, **kwargs)

        self._log_min = float(np.log(minimum))
        self._log_max = float(np.log(maximum))
        # ln of the normalization ln(max/min) = ln(ln max - ln min)
        self._log_norm = float(np.log(self._log_max - self._log_min))

    def _logpdf_raw(self, x):
        """``logpdf`` without the host/device round trip (array module native)."""
        x_arr = self.xp.asarray(x, dtype=self.xp.float64)
        mask = (x_arr >= self.minimum) & (x_arr <= self.maximum)
        # never take the log of a non-positive input: mask first, fill -inf
        safe = self.xp.where(mask, x_arr, 1.0)
        return self.xp.where(mask, -self.xp.log(safe) - self._log_norm, -self.xp.inf)

    def rvs(self, size: int | tuple = (1,), **kwargs):
        """Draw ``exp(U(ln min, ln max))`` -- values in ``[minimum, maximum]``.

        Args:
            size: Output shape; an ``int`` means ``(size,)``.
                Default ``(1,)``.
            **kwargs: Ignored (eryn prior API).

        Returns:
            xp.ndarray: Draws of shape ``size`` (host array unless ``use_cupy``
            and ``return_gpu``).
        """
        if isinstance(size, int):
            size = (size,)
        u = self.xp.random.uniform(self._log_min, self._log_max, size=size)
        return self._to_device(self.xp.exp(u))

    def logpdf(self, x, **kwargs):
        """Return ``-ln x - ln(ln(max/min))`` inside the box, ``-inf`` outside.

        Args:
            x (array-like): Physical (linear) values.
            **kwargs: Ignored (eryn prior API).

        Returns:
            xp.ndarray: float64 log-density, same shape as ``x``.
        """
        return self._to_device(self._logpdf_raw(x))

    def pdf(self, x, **kwargs):
        """Return ``exp(logpdf(x))`` -- ``1 / (x ln(max/min))`` inside the box, 0 outside.

        Args:
            x (array-like): Physical (linear) values.
            **kwargs: Ignored (eryn prior API).

        Returns:
            xp.ndarray: float64 density, same shape as ``x``.
        """
        return self._to_device(self.xp.exp(self._logpdf_raw(x)))


class EMRIKerrDomainPrior(ProbDistContainer):
    """:class:`ProbDistContainer` with the FEW Kerr ecc-eq domain-of-validity cut.

    The per-parameter EMRI priors are independent boxes, but the FEW
    Kerr eccentric-equatorial waveform is only valid on a joint ``(a, p0, e0)``
    region (interpolation grid + separatrix buffer). On top of the standard
    container behavior this class:

    - returns ``-inf`` from :meth:`logpdf` for rows outside that region
      (matching the ``-1e300`` log-likelihood sentinel used when a waveform
      call itself fails on a domain error), and
    - rejection-resamples :meth:`rvs` so drawn points are always in-domain.

    The validity test is :func:`lisatools.sources.emri.domain.emri_kerr_domain_mask`
    (cheap, no waveform generated). The cut is skipped when ``keys`` restricts
    evaluation to a parameter subset, since the joint constraint is undefined
    there.

    Built by :meth:`~lisatools.globalfit.stock.erebor.emri.EMRISetup.init_sampling_info`
    as the ``"emri"`` prior, with the column indices of ``a`` / ``p0`` / ``e0``
    read off the stock EMRI sampling basis and ``xI_fill`` from the leaves'
    ``xI0`` (prograde ``1.0`` when they mix).

    Args:
        priors_in (dict): As for :class:`~eryn.priors.ProbDistContainer`.
        a_index: Column of the spin ``a`` in the sampling basis. Default ``2``.
        p0_index: Column of the initial semi-latus rectum ``p0``. Default ``3``.
        e0_index: Column of the initial eccentricity ``e0``. Default ``4``.
        xI_fill: Fill value used for ``x0`` (``+1`` or ``-1``) in the waveform
            basis (the stock EMRI transform fills ``x0``; it is not sampled).
            Default ``1.0``.
        **kwargs: Forwarded to :class:`~eryn.priors.ProbDistContainer`
            (``use_cupy``, ``return_gpu``).

    Example:
        .. code-block:: python

            from eryn.prior import uniform_dist
            from lisatools.sampling.prior import EMRIKerrDomainPrior

            # columns (logm1, m2, a, p0, e0) as in the stock basis; the cut reads 2, 3, 4
            prior = EMRIKerrDomainPrior(
                {i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(boxes)},
                a_index=2, p0_index=3, e0_index=4, xI_fill=1.0,
            )
            x = prior.rvs(size=100)  # (100, 5), every row inside the FEW domain
    """

    def __init__(
        self,
        priors_in,
        a_index: int = 2,
        p0_index: int = 3,
        e0_index: int = 4,
        xI_fill: float = 1.0,
        **kwargs,
    ):
        super().__init__(priors_in, **kwargs)
        self.a_index = int(a_index)
        self.p0_index = int(p0_index)
        self.e0_index = int(e0_index)
        self.xI_fill = float(xI_fill)

    def _in_domain(self, x) -> np.ndarray:
        """Return the boolean ``(N,)`` mask of sampling-basis rows inside the FEW domain."""
        from ..sources.emri.domain import emri_kerr_domain_mask

        x_np = np.atleast_2d(asnumpy(x))
        return emri_kerr_domain_mask(
            x_np[:, self.a_index],
            x_np[:, self.p0_index],
            x_np[:, self.e0_index],
            xI=self.xI_fill,
        )

    def logpdf(self, x, keys=None, **kwargs):
        """Return the container log-prior with out-of-domain rows set to ``-inf``.

        Args:
            x (xp.ndarray): ``(ndim,)`` or ``(N, ndim)`` sampling-basis points.
            keys (list, optional): Component subset (see
                :meth:`eryn.priors.ProbDistContainer.logpdf`); when given, the
                domain cut is skipped.
            **kwargs: Forwarded to the base :meth:`logpdf`.

        Returns:
            float or xp.ndarray: As the base class (a float for 1-D ``x``), with
            ``-inf`` where :func:`~lisatools.sources.emri.domain.emri_kerr_domain_mask`
            is ``False``.
        """
        lp = super().logpdf(x, keys=keys, **kwargs)
        if keys is not None:
            return lp
        ok = self._in_domain(x)
        if np.ndim(lp) == 0:
            # 1D input squeezes to a scalar in the base class.
            return lp if bool(ok[0]) else -np.inf
        xp_here = get_array_module(lp)
        return xp_here.where(xp_here.asarray(ok), lp, -np.inf)

    def rvs(self, size=1, keys=None, max_tries: int = 1000, **kwargs):
        """Draw prior samples, redrawing out-of-domain rows until all are valid.

        Args:
            size (int or tuple): Leading output shape. Default ``1``.
            keys (list, optional): Component subset; when given, the base draw
                is returned unchanged (no domain cut).
            max_tries: Maximum resampling rounds. Default ``1000``.
            **kwargs: Forwarded to the base :meth:`rvs`.

        Returns:
            np.ndarray: Host array of shape ``size + (ndim,)`` (always numpy on
            the cut path).

        Raises:
            RuntimeError: Some rows are still out of domain after ``max_tries``
                rounds.
        """
        out = super().rvs(size=size, keys=keys, **kwargs)
        if keys is not None:
            return out
        out_np = asnumpy(out)
        shape = out_np.shape
        flat = out_np.reshape(-1, self.ndim)
        bad = ~self._in_domain(flat)
        tries = 0
        while np.any(bad):
            tries += 1
            if tries > max_tries:
                raise RuntimeError(
                    "EMRIKerrDomainPrior.rvs: could not draw in-domain samples "
                    f"after {max_tries} resampling rounds; check the prior limits."
                )
            n_bad = int(bad.sum())
            repl = asnumpy(super().rvs(size=n_bad, **kwargs)).reshape(n_bad, self.ndim)
            flat[bad] = repl
            bad = ~self._in_domain(flat)
        return flat.reshape(shape)


# TODO check: keep/remove/document -- no callers (only a commented-out line in stock/erebor/gb.py)
class AmplitudeFrequencySNRPrior:
    """Joint prior on GB amplitude and frequency expressed via SNR.

    Internally the amplitude prior is the SNR prior of :class:`SNRPrior`
    transformed to amplitude using :class:`AmplitudeFromSNR` (which depends
    on the frequency through the LISA noise PSD). The frequency prior is
    user-supplied.

    Args:
        rho_star: Characteristic SNR scale used by :class:`SNRPrior`.
        frequency_prior: Prior on the frequency parameter (in mHz). Must
            expose ``logpdf`` and ``rvs`` and accept a ``use_cupy`` flag.
        L: LISA arm length in meters; passed to :class:`AmplitudeFromSNR`.
        Tobs: Observation time in seconds; passed to
            :class:`AmplitudeFromSNR`.
        use_cupy: If ``True``, use ``cupy`` arrays internally.
        **noise_kwargs: Default keyword arguments for
            :func:`get_sensitivity` when evaluating the LISA noise PSD.
    """

    def __init__(self, rho_star, frequency_prior, L, Tobs, use_cupy=False, **noise_kwargs):
        self.rho_star = rho_star
        self.frequency_prior = frequency_prior

        self.transform = AmplitudeFromSNR(L, Tobs, use_cupy=use_cupy, **noise_kwargs)
        self.snr_prior = SNRPrior(rho_star, use_cupy=use_cupy)

        # must be after transform and snr_prior due to setter
        self.use_cupy = use_cupy

    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy
        self.transform.use_cupy = use_cupy
        self.snr_prior.use_cupy = use_cupy
        self.frequency_prior.use_cupy = use_cupy

    def pdf(self, *args, **noise_kwargs):
        """Return the joint amplitude-frequency PDF, ``exp(logpdf(...))``."""
        return np.exp(self.logpdf(*args, **noise_kwargs))

    def logpdf(self, amp, f0_ms, **noise_kwargs):
        """Evaluate the joint log-PDF on amplitude and frequency.

        Args:
            amp: GB amplitude(s).
            f0_ms: GB frequency(ies) in mHz.
            **noise_kwargs: Forwarded to the SNR / PSD calculation.

        Returns:
            Log-probability density evaluated at the input points.
        """

        xp = np if not self.use_cupy else cp

        f0 = f0_ms / 1e3
        rho, f0 = self.transform.forward(amp, f0, **noise_kwargs)

        rho_pdf = self.snr_prior.pdf(rho)

        Jac = xp.abs(rho / amp)

        logpdf_amp = np.log(np.abs(Jac * rho_pdf))

        # TODO: fix this?
        logpdf_f = xp.asarray(self.frequency_prior.logpdf(f0_ms))
        return logpdf_amp + logpdf_f

    def rvs(self, size=1, f0_input=None, **noise_kwargs):
        """Draw samples ``(amp, f0_ms)`` from the joint prior.

        Args:
            size: Sample shape; an integer is interpreted as a 1-tuple.
            f0_input: Optional pre-drawn frequency samples (in mHz) used
                in place of drawing from ``self.frequency_prior``.
            **noise_kwargs: Forwarded to the SNR-to-amplitude transform.

        Returns:
            Tuple ``(amp, f0_ms)`` with shape ``size`` each.
        """
        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        if f0_input is None:
            f0_ms = self.frequency_prior.rvs(size=size)
        else:
            f0_ms = f0_input
            assert f0_input.shape[:-1] == size

        f0 = f0_ms / 1e3

        rho = self.snr_prior.rvs(size=size)

        amp, _ = self.transform(rho, f0, **noise_kwargs)

        return (amp, f0_ms)


class SNRPrior:
    """Prior on signal-to-noise ratio with characteristic scale :math:`\\rho_*`.

    Implements the analytic SNR prior

    .. math::

        p(\\rho) = \\frac{3 \\rho}{4 \\rho_*^2 (1 + \\rho / (4 \\rho_*))^5} \\quad \\rho > 0,

    along with its CDF and an inverse-CDF sampler (closed-form root of the
    quartic ``cdf(rho) = u``). The density peaks at :math:`\\rho = \\rho_*`.

    Args:
        rho_star (float): Characteristic SNR scale :math:`\\rho_*`.
        use_cupy (bool, optional): If ``True``, evaluate with ``cupy`` arrays
            (inputs must then be cupy arrays). Default ``False``.

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.prior import SNRPrior
        >>> prior = SNRPrior(rho_star=10.0)
        >>> prior.rvs(size=1000).shape
        (1000,)
        >>> float(prior.cdf(np.array([0.0]))[0])
        0.0
    """

    def __init__(self, rho_star, use_cupy=False):
        self.rho_star = rho_star
        self.use_cupy = use_cupy

    @property
    def use_cupy(self):
        """bool: Evaluate with ``cupy`` (inputs must then be cupy arrays)."""
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy

    def pdf(self, rho):
        """Evaluate the SNR PDF, returning zero for non-positive SNR.

        Args:
            rho (xp.ndarray): SNR values (an array; boolean indexing is used).

        Returns:
            xp.ndarray: Density, same shape as ``rho``.
        """

        xp = np if not self.use_cupy else cp

        p = xp.zeros_like(rho)
        good = rho > 0.0
        p[good] = (
            3 * rho[good] / (4 * self.rho_star**2 * (1 + rho[good] / (4 * self.rho_star)) ** 5)
        )
        return p

    def logpdf(self, rho):
        """Return ``log(pdf(rho))`` (``-inf`` for non-positive SNR).

        Args:
            rho (xp.ndarray): SNR values.

        Returns:
            xp.ndarray: Log-density, same shape as ``rho``.
        """
        xp = np if not self.use_cupy else cp
        return xp.log(self.pdf(rho))

    def cdf(self, rho):
        r"""Evaluate the analytic CDF of the SNR prior.

        .. math::

            F(\rho) = 1 - \frac{256 \rho_*^3 (\rho + \rho_*)}{(\rho + 4 \rho_*)^4},
            \quad \rho > 0,

        and 0 for non-positive SNR.

        Args:
            rho (xp.ndarray): SNR values.

        Returns:
            xp.ndarray: CDF values, same shape as ``rho``.
        """
        xp = np if not self.use_cupy else cp
        c = xp.zeros_like(rho)
        good = rho > 0.0
        c[good] = (
            768
            * self.rho_star**3
            * (
                1 / (768.0 * self.rho_star**3)
                - (rho[good] + self.rho_star) / (3.0 * (rho[good] + 4 * self.rho_star) ** 4)
            )
        )
        return c

    def rvs(self, size=1):
        """Draw SNR samples via inverse-CDF sampling.

        Uses the global ``numpy.random`` (or ``cupy.random``) stream.

        Args:
            size (int or tuple, optional): Output shape; an integer is
                interpreted as a 1-tuple. Default ``1``.

        Returns:
            xp.ndarray: SNR samples with shape ``size``.
        """
        if isinstance(size, int):
            size = (size,)

        xp = np if not self.use_cupy else cp

        u = xp.random.rand(*size)

        rho = (
            -4 * self.rho_star
            + xp.sqrt(
                -32 * self.rho_star**2
                - (32 * (-self.rho_star**2 + u * self.rho_star**2)) / (1 - u)
                + (
                    3072
                    * 2**0.3333333333333333
                    * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3)
                    * (self.rho_star**4 - u * self.rho_star**4)
                )
                / (
                    (-1 + u) ** 2
                    * xp.cbrt(
                        -1769472 * self.rho_star**6
                        + 1769472 * u * self.rho_star**6
                        - xp.sqrt(
                            3131031158784 * u * self.rho_star**12
                            - 6262062317568 * u**2 * self.rho_star**12
                            + 3131031158784 * u**3 * self.rho_star**12
                        )
                    )
                )
                + xp.cbrt(
                    -1769472 * self.rho_star**6
                    + 1769472 * u * self.rho_star**6
                    - xp.sqrt(
                        3131031158784 * u * self.rho_star**12
                        - 6262062317568 * u**2 * self.rho_star**12
                        + 3131031158784 * u**3 * self.rho_star**12
                    )
                )
                / (3.0 * 2**0.3333333333333333 * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3))
            )
            / 2.0
            + xp.sqrt(
                32 * self.rho_star**2
                + (32 * (-self.rho_star**2 + u * self.rho_star**2)) / (1 - u)
                - (
                    3072
                    * 2**0.3333333333333333
                    * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3)
                    * (self.rho_star**4 - u * self.rho_star**4)
                )
                / (
                    (-1 + u) ** 2
                    * xp.cbrt(
                        -1769472 * self.rho_star**6
                        + 1769472 * u * self.rho_star**6
                        - xp.sqrt(
                            3131031158784 * u * self.rho_star**12
                            - 6262062317568 * u**2 * self.rho_star**12
                            + 3131031158784 * u**3 * self.rho_star**12
                        )
                    )
                )
                - xp.cbrt(
                    -1769472 * self.rho_star**6
                    + 1769472 * u * self.rho_star**6
                    - xp.sqrt(
                        3131031158784 * u * self.rho_star**12
                        - 6262062317568 * u**2 * self.rho_star**12
                        + 3131031158784 * u**3 * self.rho_star**12
                    )
                )
                / (3.0 * 2**0.3333333333333333 * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3))
                + (2048 * self.rho_star**3 - (2048 * u * self.rho_star**3) / (-1 + u))
                / (
                    4.0
                    * xp.sqrt(
                        -32 * self.rho_star**2
                        - (32 * (-self.rho_star**2 + u * self.rho_star**2)) / (1 - u)
                        + (
                            3072
                            * 2**0.3333333333333333
                            * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3)
                            * (self.rho_star**4 - u * self.rho_star**4)
                        )
                        / (
                            (-1 + u) ** 2
                            * xp.cbrt(
                                -1769472 * self.rho_star**6
                                + 1769472 * u * self.rho_star**6
                                - xp.sqrt(
                                    3131031158784 * u * self.rho_star**12
                                    - 6262062317568 * u**2 * self.rho_star**12
                                    + 3131031158784 * u**3 * self.rho_star**12
                                )
                            )
                        )
                        + xp.cbrt(
                            -1769472 * self.rho_star**6
                            + 1769472 * u * self.rho_star**6
                            - xp.sqrt(
                                3131031158784 * u * self.rho_star**12
                                - 6262062317568 * u**2 * self.rho_star**12
                                + 3131031158784 * u**3 * self.rho_star**12
                            )
                        )
                        / (3.0 * 2**0.3333333333333333 * xp.cbrt(-1 + 3 * u - 3 * u**2 + u**3))
                    )
                )
            )
            / 2.0
        )

        return rho


# TODO check: keep/remove/document -- only caller is AmplitudeFrequencySNRPrior (no callers)
class AmplitudeFromSNR:
    """Convert between GB amplitude and optimal SNR for a given LISA noise model.

    The amplitude / SNR relation used here is

    .. math::

        \\rho = \\frac{1}{2} \\, A \\, \\sqrt{\\frac{T_\\text{obs} \\, \\sin^2(f / f_*)}{S_n(f)}},

    where :math:`f_* = c / (2\\pi L)` is the LISA transfer frequency. Either
    direction (amplitude → SNR or SNR → amplitude) can be evaluated, and the
    PSD :math:`S_n(f)` can be supplied directly, interpolated from a sampled
    PSD, or computed from :func:`get_sensitivity`.

    Args:
        L: LISA arm length in meters.
        Tobs: Observation time in seconds.
        fd: Optional frequency grid (Hz) used to interpolate sampled PSDs in
            :meth:`interp_psd`.
        use_cupy: If ``True``, store ``fd`` and operate on ``cupy`` arrays.
        **noise_kwargs: Default kwargs passed to :func:`get_sensitivity` when
            no PSD is supplied at call time.
    """

    def __init__(self, L, Tobs, fd=None, use_cupy=False, **noise_kwargs):
        self.f_star = 1 / (2.0 * np.pi * L) * C_SI
        self.Tobs = Tobs
        self.noise_kwargs = noise_kwargs

        xp = np if not use_cupy else cp
        if fd is not None:
            self.fd = xp.asarray(fd)
        else:
            self.fd = fd

        # got to be after fd
        self.use_cupy = use_cupy

    @property
    def use_cupy(self):
        return self._use_cupy

    @use_cupy.setter
    def use_cupy(self, use_cupy):
        self._use_cupy = use_cupy
        if use_cupy and not isinstance(self.fd, cp.ndarray):
            self.fd = cp.asarray(self.fd)
        elif not use_cupy and isinstance(self.fd, cp.ndarray):
            self.fd = asnumpy(self.fd)

    def interp_psd(self, f0, psds, walker_inds=None):
        """Linearly interpolate sampled PSDs at frequencies ``f0``.

        Args:
            f0: Target frequencies (Hz) at which to evaluate the PSD.
            psds: Array of shape ``(nwalkers, len(self.fd))`` (or
                broadcastable to that) holding the sampled PSDs.
            walker_inds: Per-frequency walker indices selecting which row of
                ``psds`` to use. If ``None``, walker index 0 is used for all.

        Returns:
            Array of interpolated PSD values at ``f0``.
        """
        assert self.fd is not None
        xp = np if not self.use_cupy else cp
        psds = xp.atleast_2d(psds)

        if xp == cp and not isinstance(self.fd, cp.ndarray):
            self.fd = xp.asarray(self.fd)
        try:
            inds_fd = xp.searchsorted(self.fd, f0, side="right") - 1
        except:
            breakpoint()
        if walker_inds is None:
            walker_inds = xp.zeros_like(f0, dtype=int)

        new_psds = (psds[(walker_inds, inds_fd + 1)] - psds[(walker_inds, inds_fd)]) / (
            self.fd[inds_fd + 1] - self.fd[inds_fd]
        ) * (f0 - self.fd[inds_fd]) + psds[(walker_inds, inds_fd)]
        return new_psds

    def __call__(self, rho, f0, **noise_kwargs):
        """Convert SNR to amplitude at the given frequency.

        Args:
            rho: SNR value(s).
            f0: Frequency value(s) in Hz.
            **noise_kwargs: Forwarded to :meth:`get_Sn_f`.

        Returns:
            Tuple ``(amp, f0)``.
        """

        xp = np if not self.use_cupy else cp

        if noise_kwargs == {}:
            noise_kwargs = self.noise_kwargs

        Sn_f = self.get_Sn_f(f0, **noise_kwargs)

        factor = 1.0 / 2.0 * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / Sn_f)
        amp = rho / factor
        return (amp, f0)

    def get_Sn_f(self, f0, psds=None, walker_inds=None, Sn_f=None, **noise_kwargs):
        """Resolve the noise PSD at frequencies ``f0``.

        The PSD is, in priority order: a user-supplied ``Sn_f`` (matched to
        ``f0`` and converted between numpy/cupy as needed), an interpolation
        of a sampled ``psds`` array, or a fresh evaluation of
        :func:`get_sensitivity` with ``noise_kwargs``.

        Args:
            f0: Frequencies (Hz) at which to obtain the PSD.
            psds: Optional sampled PSD array; see :meth:`interp_psd`.
            walker_inds: Walker indices used by :meth:`interp_psd`.
            Sn_f: Optional precomputed PSD values, same length as ``f0``.
            **noise_kwargs: Forwarded to :func:`get_sensitivity` when
                neither ``Sn_f`` nor ``psds`` is given.

        Returns:
            PSD values at ``f0``.
        """
        if Sn_f is not None:
            assert len(f0) == len(Sn_f)
            if not isinstance(f0, type(Sn_f)):
                if isinstance(f0, np.ndarray):
                    assert isinstance(Sn_f, cp.ndarray)
                    Sn_f = asnumpy(Sn_f)
                else:
                    assert isinstance(Sn_f, np.ndarray)
                    f0 = cp.asarray(f0)

        elif psds is not None:
            Sn_f = self.interp_psd(f0, psds, walker_inds=walker_inds)
        else:
            Sn_f = get_sensitivity(f0, **noise_kwargs)

        return Sn_f

    def forward(self, amp, f0, **noise_kwargs):
        """Convert amplitude to SNR at the given frequency.

        Args:
            amp: GB amplitude(s).
            f0: Frequency value(s) in Hz.
            **noise_kwargs: Forwarded to :meth:`get_Sn_f`.

        Returns:
            Tuple ``(rho, f0)``.
        """

        if noise_kwargs == {}:
            noise_kwargs = self.noise_kwargs

        Sn_f = self.get_Sn_f(f0, **noise_kwargs)

        factor = 1.0 / 2.0 * np.sqrt((self.Tobs * np.sin(f0 / self.f_star) ** 2) / Sn_f)
        rho = amp * factor
        return (rho, f0)


# TODO check: keep/remove/document -- imported by moves/gbspecialstretch.py, only named in comments
class GBPriorWrap:
    """Wrap a GB parameter prior so amplitude is sampled jointly with frequency.

    Re-routes the amplitude / frequency components through an
    :class:`AmplitudeFrequencySNRPrior` (stored under key ``(0, 1)`` of the
    underlying ``priors_in`` mapping) while delegating the remaining
    parameters to ``full_prior_container``.

    Args:
        ndim: Total parameter dimensionality of the GB model.
        full_prior_container: ``eryn``-style prior container exposing
            ``priors_in``, ``logpdf``, and ``rvs``, with the joint
            amplitude-frequency prior stored at key ``(0, 1)``.
        gen_frequency_alone: If ``True``, draw frequency from
            ``full_prior_container`` and reuse it when sampling amplitude;
            otherwise the joint amplitude-frequency prior generates both.
    """

    # TODO/DOCS: ``keys_sep`` is the list of indices not handled by the joint
    # amplitude-frequency prior; precise eryn semantics of these keys are not
    # documented here.
    def __init__(self, ndim, full_prior_container, gen_frequency_alone=False):
        self.base_prior = full_prior_container
        self.use_cupy = full_prior_container.use_cupy
        self.ndim = ndim
        self.gen_frequency_alone = gen_frequency_alone

        if gen_frequency_alone:
            self.keys_sep = [1, 2, 3, 4, 5, 6, 7]
        else:
            self.keys_sep = [2, 3, 4, 5, 6, 7]

    @property
    def priors_in(self):
        return self.base_prior.priors_in

    def logpdf(self, x, **noise_kwargs):
        """Evaluate the log-prior, summing the joint A-f piece and the rest.

        Args:
            x: Parameter array of shape ``(n, ndim)``.
            **noise_kwargs: Forwarded to the joint amplitude-frequency prior.

        Returns:
            Log-prior values of shape ``(n,)``.
        """
        xp = np if not self.use_cupy else cp
        assert x.shape[1] == self.ndim and x.ndim == 2

        logpdf_everything_else = xp.asarray(self.base_prior.logpdf(x, keys=self.keys_sep))

        f0 = xp.asarray(x[:, 1])
        amp = xp.asarray(x[:, 0])
        logpdf_A_f = self.base_prior.priors_in[(0, 1)].logpdf(amp, f0, **noise_kwargs)

        return logpdf_A_f + logpdf_everything_else

    def rvs(self, size=1, ignore_amp=False, **kwargs):
        """Draw samples from the wrapped prior.

        Args:
            size: Sample shape; an integer is interpreted as a 1-tuple.
            ignore_amp: If ``True``, leave the amplitude (and frequency, if
                ``gen_frequency_alone``) entries at zero rather than drawing
                from the joint amplitude-frequency prior.
            **kwargs: Forwarded to the joint amplitude-frequency ``rvs``.

        Returns:
            Array of shape ``size + (self.ndim,)``.
        """
        xp = np if not self.use_cupy else cp
        if isinstance(size, int):
            size = (size,)

        arr = xp.zeros(size + (self.ndim,)).reshape(-1, self.ndim)

        diff = self.ndim - len(self.keys_sep)
        assert diff >= 0

        arr[:, :] = self.base_prior.rvs(size, keys=self.keys_sep).reshape(-1, self.ndim)

        if not ignore_amp:
            f0_input = arr[:, 1] if self.gen_frequency_alone else None
            arr[:, :diff] = (
                xp.asarray(self.base_prior.priors_in[(0, 1)].rvs(size, f0_input=f0_input, **kwargs))
                .reshape(diff, -1)
                .T
            )

        arr = arr.reshape(size + (self.ndim,))
        return arr


from ..utils.parallelbase import LISAToolsParallelModule


class FullGaussianMixtureModel(LISAToolsParallelModule):
    """Concatenated Gaussian mixture distribution over many sub-domains.

    Each input list provides the parameters of an independently-fit GMM in
    its own ``(mins, maxs)``-bounded box (typically one per GB frequency
    band; the output of :func:`~lisatools.sampling.gmm.vec_fit_gmm_min_bic`
    with ``return_components=True``). Each sub-mixture lives in its box's
    mapped coordinates ``2 (x - mins) / (maxs - mins) - 1`` (the
    :math:`[-1, 1]^d` cube); the density in physical coordinates adds the
    map's log-Jacobian. The component lists are concatenated into one flat
    mixture and every sub-mixture's weights (which must each sum to 1) are
    divided by the number of sub-domains, so each sub-domain carries equal
    total weight and the global PDF integrates to one.

    Column 1 is the frequency column: :meth:`logpdf` sums, per point, only
    the components whose frequency window (mapped coordinate within
    ``[-limit, limit]`` of their box) contains the point's column 1. For
    ``ndim == 6`` (the 6 non-phase GB columns) it uses the backend
    ``compute_logpdf`` C/CUDA kernel; any other ``ndim`` uses the numpy/cupy
    mirror :meth:`_logpdf_xp` (same math).

    Built by the GB GMM refit move
    (:class:`~lisatools.globalfit.moves.gbspecialstretch.GBSpecialRJRefitMove`,
    ``use_cupy=True``) and by :func:`~lisatools.sampling.gmm.vec_fit_gmm_min_bic`
    (without ``return_components``), and used as the joint component of the
    6 non-phase GB columns in an eryn :class:`~eryn.priors.ProbDistContainer`.
    Neither builder runs in the 9mo / 12mo runs (``include_refit=False``; the
    serial-search move is never staged).

    Args:
        weights (list[xp.ndarray]): Per-sub-domain weight arrays ``(K_i,)``,
            each summing to 1.
        means (list[xp.ndarray]): Per-sub-domain means ``(K_i, ndim)`` in
            mapped coordinates.
        covs (list[xp.ndarray]): Per-sub-domain covariances
            ``(K_i, ndim, ndim)`` in mapped coordinates.
        invcovs (list[xp.ndarray]): Per-sub-domain inverse covariances.
        dets (list[xp.ndarray]): Per-sub-domain covariance determinants
            ``(K_i,)``.
        mins (list[xp.ndarray]): Per-sub-domain lower bounds ``(ndim,)``
            (physical units).
        maxs (list[xp.ndarray]): Per-sub-domain upper bounds ``(ndim,)``.
        limit (float, optional): Half-width, in mapped units of column 1, of
            each component's frequency window in :meth:`logpdf`. Default
            ``10.0``.
        use_cupy (bool, optional): If ``True``, store the arrays on the GPU
            and construct the backend with ``force_backend="gpu"``; otherwise
            ``"cpu"``. Default ``False``.

    Raises:
        AssertionError: The concatenated, rescaled weights do not sum to 1.

    Attributes:
        ndim (int): Dimension of the mixture.
        weights (xp.ndarray): Flat, rescaled weights ``(C,)``.
        means (xp.ndarray): ``(C, ndim)``.
        covs (xp.ndarray): ``(C, ndim, ndim)``.
        chols (xp.ndarray): Cholesky factors of ``covs`` (used by :meth:`rvs`).
        indexing (xp.ndarray): Sub-domain index of each flat component ``(C,)``.
        mins (xp.ndarray): ``(n_sub, ndim)`` box lower bounds.
        maxs (xp.ndarray): ``(n_sub, ndim)`` box upper bounds.
        log_det_J (xp.ndarray): Per-component log-Jacobian of the box map.

    Example:
        >>> import numpy as np
        >>> from lisatools.sampling.prior import FullGaussianMixtureModel
        >>> eye = np.eye(2)[None]
        >>> gmm = FullGaussianMixtureModel(
        ...     [np.array([1.0])], [np.zeros((1, 2))], [eye], [eye], [np.ones(1)],
        ...     [np.array([0.0, 1.0])], [np.array([1.0, 2.0])],
        ... )
        >>> gmm.rvs(size=4).shape, gmm.logpdf(np.array([[0.5, 1.5]])).shape
        ((4, 2), (1,))
    """

    def __init__(
        self,
        weights,
        means,
        covs,
        invcovs,
        dets,
        mins,
        maxs,
        limit=10.0,
        use_cupy=False,
    ):

        force_backend = "gpu" if use_cupy else "cpu"
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        self.use_cupy = use_cupy
        if use_cupy:
            xp = cp
        else:
            xp = np

        indexing = []
        for i, weight in enumerate(weights):
            index_base = np.full_like(weight, i, dtype=int)
            indexing.append(index_base)

        self.indexing = xp.asarray(np.concatenate(indexing))
        # invidivual weights / total number of components to uniformly choose from them
        self.weights = xp.asarray(np.concatenate(weights, axis=0) * 1 / len(weights))

        assert xp.allclose(self.weights.sum(), 1.0)
        self.means = xp.asarray(np.concatenate(means, axis=0))
        self.covs = xp.asarray(np.concatenate(covs, axis=0))
        self.invcovs = xp.asarray(np.concatenate(invcovs, axis=0))
        self.dets = xp.asarray(np.concatenate(dets, axis=0))
        # Cholesky factors for rvs: draws are ``mean + L z`` with
        # ``L L^T = cov`` -- multiplying by the covariance itself (the old
        # code) produces draws with covariance cov^2, silently inconsistent
        # with logpdf (and therefore with RJ detailed balance).
        self.chols = xp.linalg.cholesky(self.covs)
        self.ndim = self.means.shape[1]

        self.mins = xp.asarray(np.vstack(mins))
        self.maxs = xp.asarray(np.vstack(maxs))

        self.mins_in_pdf = self.mins[self.indexing].T.flatten().copy()
        self.maxs_in_pdf = self.maxs[self.indexing].T.flatten().copy()
        self.means_in_pdf = self.means.T.flatten().copy()
        self.invcovs_in_pdf = self.invcovs.transpose(1, 2, 0).flatten().copy()

        self.cumulative_weights = xp.concatenate([xp.array([0.0]), xp.cumsum(self.weights)])

        self.min_limit_f = self.map_back_frequency(
            -1.0 * limit, self.mins[self.indexing, 1], self.maxs[self.indexing, 1]
        )
        self.max_limit_f = self.map_back_frequency(
            +1.0 * limit, self.mins[self.indexing, 1], self.maxs[self.indexing, 1]
        )

        # compute the jacobian
        self.log_det_J = (self.ndim * np.log(2) - xp.sum(xp.log(self.maxs - self.mins), axis=-1))[
            self.indexing
        ].copy()

        """self.inds_sort_min_limit_f = xp.argsort(self.min_limit_f)
        self.inds_sort_max_limit_f = xp.argsort(self.max_limit_f)
        self.sorted_min_limit_f = self.min_limit_f[self.inds_sort_min_limit_f]
        self.sorted_max_limit_f = self.max_limit_f[self.inds_sort_max_limit_f]
        """

    @property
    def compute_logpdf(self) -> callable:
        """Backend C/CUDA kernel that fills the per-point log-PDF buffer."""
        return self.backend.compute_logpdf

    def logpdf(self, x):
        """Evaluate the log-PDF at the input points.

        For ``ndim == 6`` the points are sorted by column 1, matched to the
        components whose frequency window contains them, and summed by the
        backend ``compute_logpdf`` kernel; a ``ValueError`` from a stale CPU
        binding falls back (with one ``RuntimeWarning``) to :meth:`_logpdf_xp`.
        Other ``ndim`` go straight to :meth:`_logpdf_xp`.

        Args:
            x (xp.ndarray): Array of shape ``(n, ndim)`` in physical
                coordinates (cupy when ``use_cupy``).

        Returns:
            xp.ndarray: Log-density values of shape ``(n,)``. Points outside
            every component's frequency window are ``-inf``.
        """

        if self.use_cupy:
            xp = cp
        else:
            xp = np

        assert len(x.shape) == 2
        assert x.shape[1] == self.ndim

        k = self.ndim

        if k != 6:
            # The compiled kernel is PDF_NDIM == 6 (the 6 non-phase columns
            # of the GB refit basis) and hard-rejects any other dimension;
            # other bases (e.g. the 4-D F-stat intrinsic proposal) evaluate
            # through the xp mirror of the same math instead.
            return self._logpdf_xp(x)

        inds_sort = xp.argsort(x[:, 1])
        f_sort = x[:, 1][inds_sort]
        points_sorted = x[inds_sort]

        ind_min_limit = xp.searchsorted(f_sort, self.min_limit_f, side="left")
        ind_max_limit = xp.searchsorted(f_sort, self.max_limit_f, side="right")

        diff = ind_max_limit - ind_min_limit
        cs = xp.concatenate([xp.array([0]), xp.cumsum(diff)])
        tmp = xp.arange(cs[-1])
        keep_component_map = xp.searchsorted(cs, tmp, side="right") - 1
        keep_point_map = tmp - cs[keep_component_map] + ind_min_limit[keep_component_map]
        max_components = diff.max().item()

        int_check = int(1e6)
        assert int_check > self.min_limit_f.shape[0]
        special_point_component_map = int_check * keep_point_map + keep_component_map

        sorted_special = xp.sort(special_point_component_map)

        points_keep_in = (sorted_special / float(int_check)).astype(int)
        components_keep_in = sorted_special - points_keep_in * int_check

        unique_points, unique_starts = xp.unique(points_keep_in, return_index=True)
        start_index_in_pdf = xp.concatenate(
            [unique_starts, xp.array([len(points_keep_in)])]
        ).astype(xp.int32)
        assert xp.all(xp.diff(unique_starts) > 0)

        points_sorted_in = points_sorted[unique_points]

        logpdf_out_tmp = xp.zeros(points_sorted_in.shape[0])

        try:
            self.compute_logpdf(
                logpdf_out_tmp,
                components_keep_in.astype(xp.int32),
                points_sorted_in,
                self.weights,
                self.mins_in_pdf,
                self.maxs_in_pdf,
                self.means_in_pdf,
                self.invcovs_in_pdf,
                self.dets,
                self.log_det_J,
                points_sorted_in.shape[0],
                start_index_in_pdf,
                self.weights.shape[0],
                x.shape[1],
            )
        except ValueError:
            # CPU binaries built before the binding_detector.cxx length-check
            # fix reject the variable-length component_index / per-point
            # start_index (CUDA builds compile the checks out, so the GPU
            # path never threw). The xp mirror is the exact same math --
            # fall back until the CPU backend is rebuilt.
            if not getattr(self, "_warned_stale_logpdf_binding", False):
                self._warned_stale_logpdf_binding = True
                import warnings

                warnings.warn(
                    "compute_logpdf CPU binding rejected its inputs (stale "
                    "pre-fix binary); using the xp logpdf mirror. Rebuild "
                    "lisatools to restore the compiled path.",
                    RuntimeWarning,
                )
            return self._logpdf_xp(x)

        # need to reverse the sort
        logpdf_out = xp.full(x.shape[0], -xp.inf)
        logpdf_out[xp.sort(inds_sort[unique_points])] = logpdf_out_tmp[
            xp.argsort(inds_sort[unique_points])
        ]
        return logpdf_out
        """# breakpoint()

        # map to reduced domain
        x_mapped = (self.map_input(points_sorted[:, None, :], self.mins[None, :, :], self.maxs[None, :, :]))[:, self.indexing]
        
        diff = x_mapped - self.means[None, :, :]
        log_main_part = -1./2. * xp.einsum("...k,...k", diff, xp.einsum("...jk,...k->...j", self.invcovs, diff))
        log_norm_factor = (k / 2) * xp.log(2 * np.pi) + (1 / 2) * xp.log(self.dets)
        log_weighted_pdf = (xp.log(self.weights) + log_norm_factor)[None, :] + log_main_part
        
        logpdf_full_dist_tmp = logsumexp(log_weighted_pdf, axis=-1, xp=xp)
        logpdf_full_dist = logpdf_full_dist_tmp[xp.argsort(inds_sort)]

        breakpoint()
        assert xp.allclose(logpdf_full_dist, logpdf_out)

        return logpdf_full_dist"""

    def _logpdf_xp(self, x, batch_size: int = 4096):
        """xp mirror of the compiled ``compute_logpdf`` kernel.

        Same math per (point, component): map into the component's box,
        Gaussian in mapped coordinates, ``+ log_det_J`` jacobian, with the
        same column-1 ``limit`` window culling; log-sum-exp over the kept
        components (``-inf`` when none). Vectorized in single array ops per
        point batch (numpy or cupy) -- used for every ``ndim != 6`` mixture,
        where the compiled kernel (``PDF_NDIM == 6``) cannot run.

        Args:
            x (xp.ndarray): ``(n, ndim)`` physical coordinates.
            batch_size: Points per vectorized batch. Default ``4096``.

        Returns:
            xp.ndarray: ``(n,)`` log-density.
        """
        xp = cp if self.use_cupy else np
        x = xp.asarray(x)
        n, k = x.shape
        mins_c = self.mins[self.indexing]      # (C, d)
        maxs_c = self.maxs[self.indexing]
        out = xp.full(n, -np.inf)
        log_w = xp.log(self.weights)
        log_norm = -0.5 * k * np.log(2.0 * np.pi) - 0.5 * xp.log(self.dets)
        for s in range(0, n, int(batch_size)):
            e = min(s + int(batch_size), n)
            xb = x[s:e]
            keep = ((xb[:, None, 1] >= self.min_limit_f[None, :])
                    & (xb[:, None, 1] <= self.max_limit_f[None, :]))
            xm = ((xb[:, None, :] - mins_c[None]) / (maxs_c - mins_c)[None]
                  * 2.0 - 1.0)
            diff = xm - self.means[None]
            maha = xp.einsum("bcd,cde,bce->bc", diff, self.invcovs, diff)
            lw = (log_w[None] + log_norm[None] - 0.5 * maha
                  + self.log_det_J[None])
            lw = xp.where(keep, lw, -np.inf)
            m = xp.max(lw, axis=1)
            m_safe = xp.where(xp.isfinite(m), m, 0.0)
            with np.errstate(divide="ignore"):
                val = (xp.log(xp.sum(xp.exp(lw - m_safe[:, None]), axis=1))
                       + m_safe)
            out[s:e] = xp.where(xp.isfinite(m), val, -np.inf)
        return out

    # TODO check: keep/remove/document -- no callers (logpdf paths inline the same map)
    def map_input(self, x, mins, maxs):
        """Map physical coordinates into the :math:`[-1, 1]^d` box."""
        return ((x - mins) / (maxs - mins)) * 2.0 - 1.0

    def map_back_frequency(self, x, mins, maxs):
        """Map :math:`[-1, 1]^d` coordinates back into physical units.

        Despite the name, it maps any column (``(x + 1) / 2 (maxs - mins) + mins``,
        broadcasting); it is used for the column-1 frequency windows and for
        :meth:`rvs`.

        Args:
            x (xp.ndarray or float): Mapped coordinates.
            mins (xp.ndarray): Box lower bounds (broadcastable to ``x``).
            maxs (xp.ndarray): Box upper bounds.

        Returns:
            xp.ndarray: Physical coordinates.
        """
        return (x + 1.0) * 1.0 / 2.0 * (maxs - mins) + mins

    def rvs(self, size=(1,)):
        """Draw samples from the mixture distribution.

        Picks a flat component by its weight, draws ``mean + L z`` with
        ``L L^T = cov`` in that component's mapped coordinates, and maps the
        draw back through its sub-domain box. Uses the global
        ``numpy.random`` / ``cupy.random`` stream.

        Args:
            size (int or tuple, optional): Sample shape; an integer is
                interpreted as a 1-tuple. Default ``(1,)``.

        Returns:
            xp.ndarray: Array of shape ``size + (ndim,)`` of samples in physical
            coordinates (not clipped to the box).
        """

        if isinstance(size, int):
            size = (size,)

        if self.use_cupy:
            xp = cp
        else:
            xp = np

        # choose which component
        draw = xp.random.rand(*size)
        component = (
            xp.searchsorted(self.cumulative_weights, draw.flatten(), side="right") - 1
        ).reshape(draw.shape)

        mean_here = self.means[component]
        chol_here = self.chols[component]

        # mean + L z with L L^T = cov (the covariance matrix itself is NOT a
        # valid mixing matrix -- it would give cov^2 draws, inconsistent with
        # logpdf and hence with RJ detailed balance).
        new_points = mean_here + xp.einsum(
            "...kj,...j->...k",
            chol_here,
            xp.random.randn(*(component.shape + (self.ndim,))),
        )

        index_here = self.indexing[component]
        mins_here = self.mins[index_here]
        maxs_here = self.maxs[index_here]
        new_points_mapped = self.map_back_frequency(new_points, mins_here, maxs_here)

        return new_points_mapped


# class FlowDist:
#     def __init__(self, config: dict, model: Union[Galaxy, GalaxyFFdot], fit: str, ndim: int):

#         self.dist = model(config)
#         self.dist.load_fit()

#         param_min, param_max = np.loadtxt(fit)
#         self.dist.set_min(param_min)
#         self.dist.set_max(param_max)

#         self.config = config
#         self.fit = fit
#         self.ndim = ndim

#     def rvs(self, size: Optional[Union[int, tuple]]=(1,)) -> cp.ndarray:
#         if isinstance(size, int):
#             size = (size,)

#         total_samp = int(np.prod(size))
#         samples = self.dist.sample(total_samp).reshape(size + (self.ndim,))
#         return samples

#     def logpdf(self, x: cp.ndarray) -> cp.ndarray:
#         assert x.shape[-1] == self.ndim
#         log_prob = self.dist.log_prob(x.reshape(-1, self.ndim)).reshape(x.shape[:-1])
#         return log_prob

# class GalaxyFlowDist(FlowDist):
#     def __init__(self):
#         config = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/configs/gbs/density_galaxy.yaml'
#         model = Galaxy
#         fit = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/minmax_galaxy_sangria.txt'
#         ndim = 3
#         super().__init__(config, model, fit, ndim)

#     def logpdf(self, x: cp.ndarray) -> cp.ndarray:
#         # adjust amplitudes to exp
#         x[:, 0] = np.log(x[:, 0])
#         return super().logpdf(x)

#     def rvs(self, size: Optional[Union[int, tuple]]=(1,)) -> cp.ndarray:
#         if isinstance(size, int):
#             size = (size,)

#         samples = super().rvs(size=size)
#         samples = samples.reshape(-1, samples.shape[-1])
#         samples[:, 0] = np.exp(samples[:, 0])
#         samples = samples.reshape(size + (samples.shape[-1],))
#         return samples

# class FFdotFlowDist(FlowDist):
#     def __init__(self):
#         config = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/configs/gbs/density_f.yaml'
#         model = GalaxyFFdot
#         fit = '/data/mkatz/LISAanalysistools/lisaflow/flow/experiments/rvs/minmax_ffdot_sangria.txt'
#         ndim = 2
#         super().__init__(config, model, fit, ndim)
