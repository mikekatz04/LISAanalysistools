"""Sky-frame conversions and calculation controllers for the LISA source classes.

Two groups of helpers live here:

- Frame conversions between ICRS and the ecliptic (LISA-DDPC-SEG-TN-007 sec. 5.4.2, fixed
  obliquity ``EPS_RAD``): :func:`icrs_to_ecliptic` and :func:`ecliptic_to_icrs`. These are on the
  global-fit run path: the run frame is ICRS, and the recipe, the EMRI catalogue glue
  (:func:`~lisatools.sources.emri.waveform.emri_catalogue_to_waveform_basis`) and the MBH
  LISA-frame proposal map (``sampling.moves.skymodehop.LISAFrameMap``) call them.
- Calculation controllers for quick SNR / covariance studies:
  :class:`CalculationController` and its :class:`BBHCalculationController`,
  :class:`GBCalculationController` and :class:`EMRICalculationController` subclasses (re-exported
  by :mod:`lisatools.sources`). They predate the domain-based
  :func:`~lisatools.diagnostic.inner_product` API; see the Warning on
  :class:`CalculationController`.

Also: the BBH parameter helpers (:func:`mT_q_to_m1_m2`, :func:`dist_convert`,
:func:`time_convert`) and a few unused helpers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from eryn.utils import TransformContainer

from ..diagnostic import covariance, plot_covariance_contour, plot_covariance_corner

from ..detector import LISAModel
from ..diagnostic import snr as snr_func
from ..sensitivity import A1TDISens, Sensitivity
from ..utils.constants import *
from ..utils.utility import get_array_module
from .waveformbase import AETTDIWaveform, SNRWaveform

if TYPE_CHECKING:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        import numpy as cp

class CalculationController:
    """Wrapper class to control investigative computations.

    Provides a common interface for computing SNRs (and, in subclasses, information /
    covariance matrices) for a particular source class given a template waveform
    generator and a LISA noise model. The constructor only stores its arguments.

    Args:
        aet_template_gen: Template waveform generator. Called as
            ``aet_template_gen(*params, **kwargs)`` and expected to return three channels; the
            first two are scored (treated as A and E), the third is ignored. Its ``dt``,
            ``f_arr`` and ``df`` attributes are forwarded to the SNR call.
        model: LISA noise model, added as ``model`` to ``psd_kwargs`` for the sensitivity.
        psd_kwargs: Keyword arguments for the sensitivity evaluation.
        Tobs: Observation time in **years** (:class:`GBCalculationController` converts it to
            seconds in place).
        dt: Timestep in seconds.
        psd: Sensitivity curve type to use. Default is
            :class:`~lisatools.sensitivity.A1TDISens` because these simplified calculations
            ignore ``T`` and the ``A`` and ``E`` sensitivities are equivalent.

    Attributes:
        f_arr (np.ndarray): Frequency array of the last :meth:`get_snr` call.
        last_output (tuple): ``(a_chan, e_chan)`` of the last :meth:`get_snr` call.

    Warning:
        :meth:`get_snr` (and the subclasses' ``get_cov``) pass ``dt=``, ``f_arr=`` and ``df=``
        through :func:`~lisatools.diagnostic.snr` / :func:`~lisatools.diagnostic.covariance`
        to :func:`~lisatools.diagnostic.inner_product`, whose current signature
        (``basis_settings``, ``psd``, ``psd_args``, ``psd_kwargs``, ``normalize``,
        ``complex``) does not accept them. With the current diagnostic API these calls raise
        ``TypeError``; the controllers are kept for reference until they are ported.
    """

    def __init__(
        self,
        aet_template_gen: SNRWaveform | AETTDIWaveform,
        model: LISAModel,
        psd_kwargs: dict,
        Tobs: float,
        dt: float,
        psd: Sensitivity = A1TDISens,
    ) -> None:

        # Store everything.
        self.aet_template_gen = aet_template_gen
        self.psd_kwargs = psd_kwargs
        self.model = model
        self.psd = psd
        self.Tobs = Tobs
        self.dt = dt

    @property
    def parameter_transforms(self) -> TransformContainer:
        """:class:`~eryn.utils.TransformContainer` from the sampling basis to the waveform basis.

        Not set by the constructor (reading it before the setter raises ``AttributeError``);
        the subclasses keep their container in ``transform_fn`` instead.
        """
        return self._parameter_transforms

    @parameter_transforms.setter
    def parameter_transforms(self, parameter_transforms: TransformContainer) -> None:
        """Set the parameter transform container (must be a ``TransformContainer``)."""
        assert isinstance(parameter_transforms, TransformContainer)
        self._parameter_transforms = parameter_transforms

    def get_snr(self, *params: Any, **kwargs: Any) -> float:
        """Compute the optimal SNR of the first two template channels.

        Generates ``(a, e, t) = aet_template_gen(*params, **kwargs)`` and evaluates
        :func:`~lisatools.diagnostic.snr` on ``[a, e]`` with ``psd``, ``psd_kwargs`` (plus
        ``model``) and the generator's ``dt`` / ``f_arr`` / ``df``. Stores :attr:`f_arr` and
        :attr:`last_output` as side effects. See the class Warning on the current
        ``inner_product`` signature.

        Args:
            *params: Parameters to go into waveform generator.
            **kwargs: Kwargs for waveform generator.

        Returns:
            float: Optimal SNR.
        """
        # generate waveform
        a_chan, e_chan, t_chan = self.aet_template_gen(*params, **kwargs)

        # ignore t channel for snr computation
        # compute SNR
        opt_snr = snr_func(
            [a_chan, e_chan],
            psd=self.psd,
            psd_kwargs={**self.psd_kwargs, "model": self.model},
            dt=self.aet_template_gen.dt,
            f_arr=self.aet_template_gen.f_arr,
            df=self.aet_template_gen.df,
        )

        # prepare outputs
        self.f_arr = self.aet_template_gen.f_arr
        self.last_output = (a_chan, e_chan)

        return opt_snr


def mT_q_to_m1_m2(mT: float, q: float) -> Tuple[float, float]:
    r"""Convert total mass and mass ratio to component masses.

    Computes :math:`m_1 = M / (1 + q)` and :math:`m_2 = q M / (1 + q)`, so ``q`` is the mass
    ratio :math:`m_2 / m_1`. :class:`BBHCalculationController` feeds it
    :math:`q = m_2 / m_1 \le 1` (``m1`` the heavier). This is the inverse of the stock MBH
    global-fit convention :math:`Q = m_1 / m_2 \ge 1`
    (:func:`~lisatools.globalfit.stock.erebor.transforms.make_mbh_transform_container`).

    Args:
        mT: Total mass :math:`M = m_1 + m_2`.
        q: Mass ratio :math:`q = m_2 / m_1` (``<= 1`` for ``m1 >= m2``).

    Returns:
        tuple[float, float]: ``(m1, m2)`` component masses in the same units as ``mT``.
    """
    return (mT / (1 + q), (q * mT) / (1 + q))


def dist_convert(x: float) -> float:
    """Convert a distance from Gpc to meters (``x * 1e9 * PC_SI``)."""
    return x * 1e9 * PC_SI


def time_convert(x: float) -> float:
    """Convert a time from years to seconds (``x * YRSID_SI``, the sidereal year)."""
    return x * YRSID_SI


class BBHCalculationController(CalculationController):
    """Calculation controller for BBHs.

    Meant for a :class:`~lisatools.sources.bbh.waveform.BBHSNRWaveform` template generator
    (bbhx ``BBHWaveformFD``), whose 12-parameter waveform basis is ``(m1, m2, chi1z, chi2z,
    distance [m], phi_ref, f_ref, inc, lam, beta, psi, t_ref [s])``. The information matrix is
    computed in the basis ``(ln mT, q = m2/m1, chi1z, chi2z, dist [Gpc], phi_ref, f_ref,
    cos inc, lam, sin beta, psi, t_ref [yr])``; ``transform_fn`` maps it back.

    Args:
        *args: Args for :class:`CalculationController`.
        **kwargs: Kwargs for :class:`CalculationController`.

    Attributes:
        transform_fn (TransformContainer): Information-matrix basis -> waveform basis.
    """

    def __init__(self, *args: Any, **kwargs: Any):

        # transforms from information matrix basis
        parameter_transforms = {
            0: np.exp,
            4: dist_convert,
            7: np.arccos,
            9: np.arcsin,
            11: time_convert,
            (0, 1): mT_q_to_m1_m2,
        }

        input_basis = list(range(12))
        output_basis = list(range(12))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(BBHCalculationController, self).__init__(*args, **kwargs)

    def get_snr(self, *args: Any, **kwargs: Any) -> float:
        """Compute the SNR.

        Unless ``t_obs_start`` is given, sets ``shift_t_limits=True``, ``t_obs_start=0.0`` and
        ``t_obs_end=Tobs`` (years) before calling :meth:`CalculationController.get_snr`.

        Args:
            *args: Parameters to go into waveform generator (the 12-parameter waveform basis).
            **kwargs: Kwargs for waveform generator.

        Returns:
            float: Optimal SNR.
        """
        # adjust kwargs to simplify calculation
        if "t_obs_start" not in kwargs:
            kwargs["shift_t_limits"] = True
            kwargs["t_obs_start"] = 0.0
            kwargs["t_obs_end"] = self.Tobs
        # compute snr
        return super(BBHCalculationController, self).get_snr(*args, **kwargs)

    def get_cov(
        self,
        *params: Any,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        The 12 waveform-basis parameters are converted to the information-matrix basis (masses
        are swapped first if ``m2 > m1``, so ``q = m2 / m1 <= 1``), ``return_array=True`` and
        (unless ``t_obs_start`` is given) the observation limits are added to the waveform
        kwargs, and :func:`~lisatools.diagnostic.covariance` is evaluated. See the Warning on
        :class:`CalculationController`.

        Args:
            *params: The 12 BBH waveform-basis parameters (including ``f_ref`` at index 6).
            more_accurate: If ``True``, run a more accurate derivative requiring 2x more
                waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the
                information matrix, by index. If ``None``, every index except 6 (``f_ref``);
                index 6 is always removed.
            precision: If ``True``, uses 500-dps precision to compute the information matrix
                inverse (requires `mpmath <https://mpmath.org>`_). This is typically a good idea
                as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            tuple[np.ndarray, np.ndarray]: The parameters in the information-matrix basis at
            ``deriv_inds`` and their covariance matrix.

        Raises:
            AssertionError: If ``len(params) != 12``.
        """

        # setup all bbh specific quantities.
        assert len(params) == 12

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        m1 = params[0]
        m2 = params[1]
        mT = m1 + m2

        if m2 > m1:
            tmp = m2
            m2 = m1
            m1 = tmp

        q = m2 / m1

        params[0] = mT
        params[1] = q

        params[0] = np.log(params[0])
        params[4] = params[4] / 1e9 / PC_SI
        params[7] = np.cos(params[7])
        params[9] = np.sin(params[9])
        params[11] = params[11] / YRSID_SI

        # default deriv inds
        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(12), 6)

        # remove f_ref derivative
        if 6 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 6)[0])

        kwargs["return_array"] = True

        if "t_obs_start" not in kwargs:
            kwargs["shift_t_limits"] = True
            kwargs["t_obs_start"] = 0.0
            kwargs["t_obs_end"] = self.Tobs

        # compute covariance
        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        # return parameters and their covariance
        return params[deriv_inds], cov


class GBCalculationController(CalculationController):
    """Calculation controller for GBs.

    Meant for a :class:`~lisatools.sources.gb.waveform.GBAETWaveform` template generator
    (``GBGPU.run_wave``) with the 9-parameter waveform basis ``(A, f0 [Hz], fdot [Hz/s], fddot,
    phi0, iota, psi, lam, beta)``. The information matrix is computed in the scaled basis
    ``(A / 1e-23, f0 [mHz], fdot / 1e-18, fddot, phi0, cos iota, psi, lam, sin beta)``;
    ``transform_fn`` maps it back. ``Tobs`` (years at construction) is converted to seconds
    in place.

    Args:
        *args: Args for :class:`CalculationController`.
        **kwargs: Kwargs for :class:`CalculationController`.

    Attributes:
        transform_fn (TransformContainer): Information-matrix basis -> waveform basis.
    """

    def __init__(self, *args: Any, **kwargs: Any):

        # parameter transforms from sampling basis to waveform basis
        parameter_transforms = {
            0: lambda x: x * 1e-23,
            1: lambda x: x / 1e3,
            2: lambda x: x * 1e-18,
            5: np.arccos,
            8: np.arcsin,
            # (1, 2, 3): lambda x, y, z: (x, y, 11.0 / 3.0 * y**2 / x),
        }

        input_basis = list(range(9))
        output_basis = list(range(9))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(GBCalculationController, self).__init__(*args, **kwargs)
        # convert back to seconds
        self.Tobs *= YRSID_SI

    def get_cov(
        self,
        *params: np.ndarray | list,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        Converts the 9 waveform-basis parameters to the scaled information-matrix basis, adds
        ``return_array=True``, ``dt`` and ``T`` (seconds) to the waveform kwargs and evaluates
        :func:`~lisatools.diagnostic.covariance`. See the Warning on
        :class:`CalculationController`.

        Args:
            *params: The 9 GB waveform-basis parameters (including ``fddot`` at index 3, which
                must be 0).
            more_accurate: If ``True``, run a more accurate derivative requiring 2x more
                waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the
                information matrix, by index. If ``None``, every index except 3 (``fddot``);
                index 3 is always removed.
            precision: If ``True``, uses 500-dps precision to compute the information matrix
                inverse (requires `mpmath <https://mpmath.org>`_). This is typically a good idea
                as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            tuple[np.ndarray, np.ndarray]: The parameters in the information-matrix basis at
            ``deriv_inds`` and their covariance matrix.

        Raises:
            AssertionError: If ``len(params) != 9``.
            NotImplementedError: If ``fddot != 0``.
        """
        assert len(params) == 9

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        # params[0] = np.log(params[0])
        params[0] = params[0] / 1e-23
        params[1] = params[1] * 1e3
        params[2] = params[2] / 1e-18

        if params[3] != 0.0:
            raise NotImplementedError("This class has not been implemented for fddot != 0 yet.")

        params[5] = np.cos(params[5])
        params[8] = np.sin(params[8])

        if deriv_inds is None:
            deriv_inds = np.delete(np.arange(9), 3)

        # remove fddot for now
        if 3 in deriv_inds:
            deriv_inds = np.delete(deriv_inds, np.where(deriv_inds == 3)[0])

        kwargs["return_array"] = True

        kwargs["dt"] = self.dt
        kwargs["T"] = self.Tobs

        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        return params[deriv_inds], cov

    def get_snr(self, *args: Any, **kwargs: Any) -> float:
        """Compute the SNR.

        Sets ``tdi2=True`` (unless given), ``dt`` and ``T`` (seconds) in the waveform kwargs
        before calling :meth:`CalculationController.get_snr`.

        Args:
            *args: Parameters to go into waveform generator (the 9-parameter GB basis).
            **kwargs: Kwargs for waveform generator.

        Returns:
            float: Optimal SNR.
        """
        # make sure it is TDI 2
        if "tdi2" not in kwargs:
            kwargs["tdi2"] = True

        kwargs["dt"] = self.dt
        kwargs["T"] = self.Tobs

        # ensures tdi2 is added correctly for GBGPU
        return super(GBCalculationController, self).get_snr(*args, **kwargs)


class EMRICalculationController(CalculationController):
    """Calculation controller for EMRIs.

    Meant for an :class:`~lisatools.sources.emri.waveform.EMRITDIWaveform` template generator
    with the 14-parameter FEW basis ``(M, mu, a, p0, e0, xI0, dist [Gpc], qS, phiS, qK, phiK,
    Phi_phi0, Phi_theta0, Phi_r0)``. The information matrix is computed with ``ln M`` and
    ``cos`` of ``xI0``, ``qS`` and ``qK`` (indices 0, 5, 7, 9); ``transform_fn`` maps it back.

    Args:
        *args: Args for :class:`CalculationController`.
        **kwargs: Kwargs for :class:`CalculationController`.

    Attributes:
        transform_fn (TransformContainer): Information-matrix basis -> waveform basis.
    """

    def __init__(self, *args: Any, **kwargs: Any):

        # parameter transforms for EMRIs
        parameter_transforms = {
            0: np.exp,
            5: np.arccos,
            7: np.arccos,
            9: np.arccos,
            # (1, 2, 3): lambda x, y, z: (x, y, 11.0 / 3.0 * y**2 / x),
        }

        input_basis = list(range(14))
        output_basis = list(range(14))
        self.transform_fn = TransformContainer(
            input_basis,
            output_basis,
            parameter_transforms=parameter_transforms,
            fill_dict=None,  # fill_dict
        )

        super(EMRICalculationController, self).__init__(*args, **kwargs)

    def get_cov(
        self,
        *params: np.ndarray | list,
        more_accurate: bool = False,
        eps: float = 1e-9,
        deriv_inds: np.ndarray = None,
        precision: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get covariance matrix.

        Converts the 14 waveform-basis parameters to the information-matrix basis, adds
        ``return_array=True`` to the waveform kwargs and evaluates
        :func:`~lisatools.diagnostic.covariance`. See the Warning on
        :class:`CalculationController`.

        Args:
            *params: The 14 EMRI waveform-basis parameters.
            more_accurate: If ``True``, run a more accurate derivative requiring 2x more
                waveform generations.
            eps: Absolute **derivative** step size. See :func:`lisatools.diagnostic.info_matrix`.
            deriv_inds: Subset of parameters of interest for which to calculate the
                information matrix, by index. If ``None``, ``np.arange(len(params))`` (inside
                :func:`~lisatools.diagnostic.info_matrix`).
            precision: If ``True``, uses 500-dps precision to compute the information matrix
                inverse (requires `mpmath <https://mpmath.org>`_). This is typically a good idea
                as the information matrix can be highly ill-conditioned.
            **kwargs: Kwargs for waveform generation.

        Returns:
            tuple[np.ndarray, np.ndarray]: The parameters in the information-matrix basis at
            ``deriv_inds`` and their covariance matrix.

        Raises:
            AssertionError: If ``len(params) != 14``, or if the generator's ``response.dt`` is
                not ``dt``.
            AttributeError: The next check reads ``aet_template_gen.response.T``, which the
                current :class:`~lisatools.response.directresponse.ResponseWrapper` does not
                define (it stores ``Tobs``).
        """
        assert len(params) == 14

        if isinstance(params, tuple):
            params = list(params)

        params = np.asarray(params)

        params[0] = np.log(params[0])
        params[5] = np.cos(params[5])
        params[7] = np.cos(params[7])
        params[9] = np.cos(params[9])

        kwargs["return_array"] = True

        assert self.aet_template_gen.response.dt == self.dt
        assert self.aet_template_gen.response.T == self.Tobs

        cov = covariance(
            eps,
            self.aet_template_gen,
            params,
            parameter_transforms=self.transform_fn,
            inner_product_kwargs=dict(
                psd=self.psd,
                psd_kwargs={**self.psd_kwargs, "model": self.model},
                dt=self.aet_template_gen.dt,
                f_arr=self.aet_template_gen.f_arr,
                df=self.aet_template_gen.df,
            ),
            waveform_kwargs=kwargs,
            more_accurate=more_accurate,
            deriv_inds=deriv_inds,
            precision=precision,
        )

        return params[deriv_inds], cov


def return_input(x: Any) -> Any:
    """Return the input unchanged (the array-output path of the frame conversions)."""
    return x


def return_float(x: np.ndarray | float) -> float:
    """Return the input as a Python ``float`` (the scalar-output path of the frame conversions)."""
    return float(x)


def icrs_to_ecliptic(
    ra: float | np.ndarray | cp.ndarray, dec: float | np.ndarray | cp.ndarray, psi: Optional[float | np.ndarray | cp.ndarray] = None
) -> (
    Tuple[float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray]
    | Tuple[float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray, float | np.ndarray | cp.ndarray]
):
    r"""Convert ICRS angles (ra, dec, optional psi) to ecliptic (lambda, beta, optional psi).

    The sky angles are converted with the fixed-obliquity rotation of the convention document
    LISA-DDPC-SEG-TN-007, sec. 5.4.2 (:math:`\epsilon` = ``EPS_RAD``):

    .. math::

        \sin\beta = \sin\delta \cos\epsilon - \cos\delta \sin\epsilon \sin\alpha,

        \cos\lambda = \cos\delta \cos\alpha / \cos\beta,

        \sin\lambda = (\sin\delta \sin\epsilon + \cos\delta \cos\epsilon \sin\alpha) / \cos\beta.

    The polarization angle is converted with

    .. math::

        \psi_\mathrm{ecliptic} = \psi - \Delta\psi,

        \cos\Delta\psi = (\sin\epsilon \sin\delta \sin\alpha + \cos\epsilon \cos\delta) / \cos\beta,

        \sin\Delta\psi = -\sin\epsilon \cos\alpha / \cos\beta.

    ``lambda`` is wrapped to :math:`[0, 2\pi)` and ``psi_ecliptic`` to :math:`[0, \pi)`;
    :math:`\cos\beta` is floored at machine epsilon (keeping its sign) at the poles. This is the
    mojito / global-fit convention (no astropy). Inputs are broadcast following NumPy rules.

    Args:
        ra: Right ascension in radians.
        dec: Declination in radians.
        psi: Optional ICRS polarization angle in radians. If given, the converted ecliptic
            polarization angle is returned as a third element.

    Returns:
        tuple: ``(lambda, beta)`` or, with ``psi``, ``(lambda, beta, psi_ecliptic)``, in
        radians. When ``ra`` and ``dec`` (and ``psi``, if given) are ``float`` instances (Python
        ``float`` or ``np.float64``) the entries are Python ``float``; otherwise (including
        ints and 0-d arrays) they are ``xp.ndarray`` on the array module of ``ra`` (numpy or
        cupy).

    Example:
        >>> from lisatools.sources.utils import ecliptic_to_icrs, icrs_to_ecliptic
        >>> lam, beta = icrs_to_ecliptic(1.0, 0.3)
        >>> ra, dec = ecliptic_to_icrs(lam, beta)
        >>> round(ra, 12), round(dec, 12)
        (1.0, 0.3)
    """
    scalar_output = (
        isinstance(ra, float) and isinstance(dec, float) and (psi is None or isinstance(psi, float))
    )
    out_fun = return_float if scalar_output else return_input

    xp = np if scalar_output else get_array_module(ra)

    ra = xp.asarray(ra, dtype=float)
    dec = xp.asarray(dec, dtype=float)

    cos_dec = xp.cos(dec)
    sin_dec = xp.sin(dec)
    cos_ra = xp.cos(ra)
    sin_ra = xp.sin(ra)
    cos_eps = xp.cos(EPS_RAD)
    sin_eps = xp.sin(EPS_RAD)

    sin_beta = sin_dec * cos_eps - cos_dec * sin_eps * sin_ra
    beta = xp.arcsin(sin_beta)

    cos_beta = xp.cos(beta)
    eps_cos_beta = xp.finfo(float).eps
    safe_cos_beta = xp.where(
        xp.abs(cos_beta) < eps_cos_beta,
        xp.copysign(eps_cos_beta, cos_beta),
        cos_beta,
    )
    inv_cos_beta = 1.0 / safe_cos_beta

    cos_lambda = cos_dec * cos_ra * inv_cos_beta
    sin_lambda = (sin_dec * sin_eps + cos_dec * cos_eps * sin_ra) * inv_cos_beta
    lambd = xp.arctan2(sin_lambda, cos_lambda) % (2 * xp.pi)

    if psi is not None:
        psi = xp.asarray(psi, dtype=float)

        cosdeltapsi = inv_cos_beta * (sin_eps * sin_dec * sin_ra + cos_eps * cos_dec)
        sindeltapsi = -inv_cos_beta * sin_eps * cos_ra

        deltapsi = xp.arctan2(sindeltapsi, cosdeltapsi)
        psi_ecliptic = (psi - deltapsi) % xp.pi

        return out_fun(lambd), out_fun(beta), out_fun(psi_ecliptic)

    return out_fun(lambd), out_fun(beta)


def ecliptic_to_icrs(
    lambd: float | np.ndarray | cp.ndarray,
    beta: float | np.ndarray | cp.ndarray,
    psi_ecliptic: Optional[float | np.ndarray | cp.ndarray] = None,
) -> (
    Tuple[float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray]
    | Tuple[float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray, float | np.ndarray |cp.ndarray]
):
    r"""Convert ecliptic (lambda, beta, optional psi) to ICRS angles (ra, dec, optional psi).

    The inverse of :func:`icrs_to_ecliptic` (LISA-DDPC-SEG-TN-007, sec. 5.4.2, fixed obliquity
    :math:`\epsilon` = ``EPS_RAD``):

    .. math::

        \sin\delta = \sin\beta \cos\epsilon + \cos\beta \sin\epsilon \sin\lambda,

        \cos\alpha = \cos\beta \cos\lambda / \cos\delta,

        \sin\alpha = (-\sin\beta \sin\epsilon + \cos\beta \cos\epsilon \sin\lambda) / \cos\delta.

    The polarization angle is converted with

    .. math::

        \psi = \psi_\mathrm{ecliptic} + \Delta\psi,

        \cos\Delta\psi = (\sin\epsilon \sin\delta \sin\alpha + \cos\epsilon \cos\delta) / \cos\beta,

        \sin\Delta\psi = -\sin\epsilon \cos\alpha / \cos\beta.

    ``ra`` is wrapped to :math:`[0, 2\pi)` and ``psi`` to :math:`[0, \pi)`;
    :math:`\cos\delta` and :math:`\cos\beta` are floored at machine epsilon (keeping their sign).
    Inputs are broadcast following NumPy rules.

    Args:
        lambd: Ecliptic longitude in radians.
        beta: Ecliptic latitude in radians.
        psi_ecliptic: Optional ecliptic polarization angle in radians. If given, the converted
            ICRS polarization angle is returned as a third element.

    Returns:
        tuple: ``(ra, dec)`` or, with ``psi_ecliptic``, ``(ra, dec, psi)``, in radians. When
        ``lambd`` and ``beta`` (and ``psi_ecliptic``, if given) are ``float`` instances (Python
        ``float`` or ``np.float64``) the entries are Python ``float``; otherwise (including
        ints and 0-d arrays) they are ``xp.ndarray`` on the array module of ``lambd`` (numpy or
        cupy).

    Example:
        >>> from lisatools.sources.utils import ecliptic_to_icrs
        >>> ra, dec = ecliptic_to_icrs(0.0, 0.0)
        >>> round(ra, 12), round(dec, 12)
        (0.0, 0.0)
    """
    scalar_output = (
        isinstance(lambd, float)
        and isinstance(beta, float)
        and (psi_ecliptic is None or isinstance(psi_ecliptic, float))
    )
    out_fun = return_float if scalar_output else return_input
    xp = np if scalar_output else get_array_module(lambd)

    lambd = xp.asarray(lambd, dtype=float)
    beta = xp.asarray(beta, dtype=float)

    cos_beta = xp.cos(beta)
    sin_beta = xp.sin(beta)
    cos_lambda = xp.cos(lambd)
    sin_lambda = xp.sin(lambd)
    cos_eps = xp.cos(EPS_RAD)
    sin_eps = xp.sin(EPS_RAD)

    sin_dec = sin_beta * cos_eps + cos_beta * sin_eps * sin_lambda
    dec = xp.arcsin(sin_dec)
    cos_dec = xp.cos(dec)
    eps_float = xp.finfo(float).eps
    safe_cos_dec = xp.where(
        xp.abs(cos_dec) < eps_float,
        xp.copysign(eps_float, cos_dec),
        cos_dec,
    )
    inv_cos_dec = 1.0 / safe_cos_dec

    cos_ra = cos_beta * cos_lambda * inv_cos_dec
    sin_ra = (-sin_beta * sin_eps + cos_beta * cos_eps * sin_lambda) * inv_cos_dec
    ra = xp.arctan2(sin_ra, cos_ra) % (2 * xp.pi)

    if psi_ecliptic is not None:
        psi_ecliptic = xp.asarray(psi_ecliptic, dtype=float)

        cos_beta = xp.cos(beta)
        safe_cos_beta = xp.where(
            xp.abs(cos_beta) < eps_float,
            xp.copysign(eps_float, cos_beta),
            cos_beta,
        )
        inv_cos_beta = 1.0 / safe_cos_beta

        cosdeltapsi = inv_cos_beta * (sin_eps * sin_dec * sin_ra + cos_eps * cos_dec)
        sindeltapsi = -inv_cos_beta * sin_eps * cos_ra

        deltapsi = xp.arctan2(sindeltapsi, cosdeltapsi)
        psi = (psi_ecliptic + deltapsi) % xp.pi

        return out_fun(ra), out_fun(dec), out_fun(psi)

    return out_fun(ra), out_fun(dec)


# TODO check: keep/remove/document -- not reached by the 9mo/12mo runs; imported by gf.recipe but never called there; only caller is scripts/gb_chunked_het/gb_mojito_mcmc_three_ways.py
def evolve_galactic_binary(
    t_start: float, 
    t_end: float, 
    f_start: float | np.ndarray, 
    phi_start: float | np.ndarray, 
    fdot_start: float | np.ndarray, 
    fddot: float | np.ndarray = 0.0,
    phase_sign: int = 1,
    Mc: Optional[float | np.ndarray] = None
) -> (
    Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]
):
    """Evolves one or more galactic binaries from its initial starting parameters to some set of final parameters for a given starting and end time.
    If Mc is provided, the binary is evolved using gravitationally driven quadrupolar radiation. Otherwise, a linear approximation is used.

    Args:
        t_start (float): Start/initial time of the binaries before evolving.
        t_end (float): End/final time of the binaries after evolving.
        f_start (float | np.ndarray): Start/initial frequency of the binaries before evolving.
        phi_start (float | np.ndarray): Start/initial phase of the binaries before evolving.
        fdot_start (float | np.ndarray): Start/initial frequency derivative of the binary before evolving. Only evolves when chirp mass Mc is provided or fddot is not zero.
        fddot (float | np.ndarray): Second order frequency derivative of the binary. Standard value is 0.0
        phase_sign (int): Sign of the phase evolution.  JaxGB assumes -1, while the GBGPU
            implementation assumes +1. This enters as: :math:`\\phi(t) = \\phi_0 + \\text{sign}\\,
            2\\pi (f_0 \\Delta t + \\dot f_0 \\Delta t^2 / 2 + \\ddot f \\Delta t^3 / 6)`, wrapped
            to :math:`[0, 2\\pi)`.
        Mc (float | np.ndarray | None): The chrip mass of the binary to evolve the binary using gravitationally driven quadrupolar radiation. Standard is None.
        
    Returns:
        Tuple of (f_end, phi_end, fdot_end).
        (Tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray])
    """
    assert phase_sign in [1, -1], "phase_sign must be either +1 or -1"
    
    if Mc is None:
       delta_t = t_end - t_start
       fdot_end = fdot_start + fddot * delta_t
       f_end = f_start + fdot_start * delta_t + 1/2 * fddot * delta_t ** 2
       phi_end = phi_start + phase_sign * 2 * np.pi * (f_start * delta_t + 1/2 * fdot_start * delta_t ** 2 + 1/6 * fddot * delta_t ** 3) 
       phi_end = phi_end % (2 * np.pi)
       return f_end, phi_end, fdot_end
       
    else:
        raise NotImplementedError("Currently, the galactic binaries can only be evolved using the linear approximation")


# TODO check: keep/remove/document -- no callers in src, scripts, tests or the sibling repos
def psi_rotation_icrs_to_ecliptic(lam_ecl, beta_ecl):
    """Polarization-basis rotation ``chi`` such that ``psi_ecl = psi_icrs - chi``.

    Both polarization angles are measured from the respective frame's local
    north (direction of increasing latitude) toward local east, with the same
    rotational sense about the line of sight, so only the relative rotation of
    the two local-north directions matters.

    Computed with the analytic LISA-DDPC-SEG-TN-007 (sec 5.4.2) convention at
    fixed obliquity ``EPS_RAD`` -- exactly the ``deltapsi`` used inside
    :func:`icrs_to_ecliptic` / :func:`ecliptic_to_icrs`, so the three
    functions share one frame convention. (The previous astropy
    ``barycentrictrueecliptic`` vector construction was replaced at the
    2026-06 stft_tof merge to keep the module astropy-free and xp-aware.)

    Args:
        lam_ecl: Ecliptic longitude(s) in radians.
        beta_ecl: Ecliptic latitude(s) in radians.

    Returns:
        ``chi`` in radians with the same shape as the inputs.
    """
    scalar_output = isinstance(lam_ecl, float) and isinstance(beta_ecl, float)
    out_fun = return_float if scalar_output else return_input
    xp = np if scalar_output else get_array_module(lam_ecl)

    lambd = xp.asarray(lam_ecl, dtype=float)
    beta = xp.asarray(beta_ecl, dtype=float)

    cos_beta = xp.cos(beta)
    sin_beta = xp.sin(beta)
    cos_lambda = xp.cos(lambd)
    sin_lambda = xp.sin(lambd)
    cos_eps = xp.cos(EPS_RAD)
    sin_eps = xp.sin(EPS_RAD)

    # ICRS image of the position (same equations as ecliptic_to_icrs).
    sin_dec = sin_beta * cos_eps + cos_beta * sin_eps * sin_lambda
    dec = xp.arcsin(sin_dec)
    cos_dec = xp.cos(dec)
    eps_float = xp.finfo(float).eps
    safe_cos_dec = xp.where(
        xp.abs(cos_dec) < eps_float,
        xp.copysign(eps_float, cos_dec),
        cos_dec,
    )
    inv_cos_dec = 1.0 / safe_cos_dec
    cos_ra = cos_beta * cos_lambda * inv_cos_dec
    sin_ra = (-sin_beta * sin_eps + cos_beta * cos_eps * sin_lambda) * inv_cos_dec

    safe_cos_beta = xp.where(
        xp.abs(cos_beta) < eps_float,
        xp.copysign(eps_float, cos_beta),
        cos_beta,
    )
    inv_cos_beta = 1.0 / safe_cos_beta
    cosdeltapsi = inv_cos_beta * (sin_eps * sin_dec * sin_ra + cos_eps * cos_dec)
    sindeltapsi = -inv_cos_beta * sin_eps * cos_ra

    return out_fun(xp.arctan2(sindeltapsi, cosdeltapsi))


# TODO check: keep/remove/document -- no callers; the stock erebor code keeps its own module-level _WAVE_WRAP_CACHE dict
class WaveWrapCache:
    """Tiny keyed cache for expensive per-run wave-generator / wave-wrap objects.

    Replaces the ad-hoc module-level ``_WAVE_WRAP_CACHE`` / ``_MBH_PHENOM_GEN_CACHE``
    dicts that the global-fit settings files used to memoize per-run response
    wrappers and domain-projection adapters. The caller owns the cache key (e.g.
    ``(id(general_info), nchannels, use_tdionfly)``) so the run-scoped semantics
    are preserved.

    Example::

        _CACHE = WaveWrapCache()
        wrap = _CACHE.get_or_build(
            (id(general_info), nchannels), lambda: build_emri_wave_wrap(...)
        )
    """

    def __init__(self):
        self._store: dict = {}

    def get_or_build(self, key, factory):
        """Return the cached object for ``key``, building it via ``factory()`` once."""
        if key not in self._store:
            self._store[key] = factory()
        return self._store[key]

    def clear(self):
        """Drop all cached entries."""
        self._store.clear()

    def __contains__(self, key):
        return key in self._store

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value
