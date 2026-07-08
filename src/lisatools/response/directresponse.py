from multiprocessing.sharedctypes import Value
import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy

import time
import h5py

from scipy.interpolate import CubicSpline

from ..detector import EqualArmlengthOrbits, Orbits, copy_orbits
from ..utils.constants import C_SI, YRSID_SI
from ..utils.utility import AET

from .parallelbase import FastLISAResponseParallelModule
from .tdiconfig import TDIConfig

def get_factorial(n):
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

C_inv = 1.0 / C_SI  # 3.3356409519815204e-09


from astropy.coordinates import SkyCoord
import astropy.units as u

# ...existing code...
def ecliptic_to_icrs(lambda_ecl, beta_ecl):
    """
    Convert ecliptic coordinates (lambda, beta) in radians to ICRS (RA, Dec) in radians.

    Parameters
    ----------
    lambda_ecl, beta_ecl : float or array-like
        Ecliptic longitude (lambda) and latitude (beta) in radians.

    Returns
    -------
    ra, dec : tuple
        Right ascension and declination in radians (same shape as inputs).
    """
    ecl = SkyCoord(lon=lambda_ecl * u.rad, lat=beta_ecl * u.rad, frame='barycentrictrueecliptic')
    icrs = ecl.transform_to('icrs')
    return icrs.ra.rad, icrs.dec.rad


def icrs_to_ecliptic(ra, dec):
    """Convert ICRS coordinates (ra, dec) to ecliptic coordinates (lambda, beta)."""

    icrs_coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
    ecliptic_coord = icrs_coord.barycentrictrueecliptic

    lambda_ecl = ecliptic_coord.lon.rad
    beta_ecl = ecliptic_coord.lat.rad

    return lambda_ecl, beta_ecl


def warn_deprecated_frame_conversion(stacklevel: int = 3) -> None:
    """Warn that per-call ``convert_to_ra_dec=True`` is deprecated.

    Sky coordinates are consumed in the orbits frame directly (matching
    the TDI-on-the-fly handling); the sprint runs everything in the SSB
    ecliptic frame with orbits loaded as ``frame="ecliptic"``. Callers
    that still sample in the ecliptic frame against ICRS-frame orbits can
    keep passing ``convert_to_ra_dec=True`` for now, but should migrate
    to orbit-frame coordinates (see ``Orbits.frame``).
    """
    warnings.warn(
        "convert_to_ra_dec=True is deprecated: sky coordinates are now "
        "consumed in the orbits frame directly (matching the "
        "TDI-on-the-fly handling). Load orbits in the frame you sample "
        "in (e.g. frame='ecliptic') and drop this kwarg.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class pyResponseTDI(FastLISAResponseParallelModule):
    """Class container for fast LISA response function generation.

    The class computes the generic time-domain response function for LISA.
    It takes LISA constellation orbital information as input and properly determines
    the response for these orbits numerically. This includes both the projection
    of the gravitational waves onto the LISA constellation arms and combinations \
    of projections into TDI observables. The methods and maths used can be found
    [here](https://arxiv.org/abs/2204.06633).

    This class is also GPU-accelerated, which is very helpful for Bayesian inference
    methods.

    Args:
        sampling_frequency (double): The sampling rate in Hz.
        num_pts (int): Number of points to produce for the final output template.
        order (int, optional): Order of Lagrangian interpolation technique. Lower orders
            will be faster. The user must make sure the order is sufficient for the
            waveform being used. (default: 25)
        tdi (str or list, optional): TDI setup. Currently, the stock options are
            :code:`'1st generation'` and :code:`'2nd generation'`. Or the user can provide
            a list of tdi_combinations of the form
            :code:`{"link": 12, "links_for_delay": [21, 13, 31], "sign": 1, "type": "delay"}`.
            :code:`'link'` (`int`) the link index (12, 21, 13, 31, 23, 32) for the projection (:math:`y_{ij}`).
            :code:`'links_for_delay'` (`list`) are the link indexes as a list used for delays
            applied to the link projections.
            ``'sign'`` is the sign in front of the contribution to the TDI observable. It takes the value of `+1` or `-1`.
            ``type`` is either ``"delay"`` or ``"advance"``. It is optional and defaults to ``"delay"``.
            (default: ``"1st generation"``)
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits 
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        tdi_chan (str, optional): Which TDI channel combination to return. Choices are :code:`'XYZ'`,
            :code:`AET`, or :code:`AE`. (default: :code:`'XYZ'`)
        tdi_orbits (:class:`Orbits`, optional): Set if different orbits from projection.
            Orbits class from LISA Analysis Tools. Works with LISA Orbits
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        use_spline (bool, optional): Default interpolation scheme for the *projection*
            waveform evaluation. If ``True``, evaluate the delayed waveform with a
            degree-5 (quintic) spline; if ``False``, use the order-``order`` Lagrangian
            fractional-delay filter. The TDI computation always stays on the Lagrangian
            path regardless of this setting. Can be overridden per call via
            ``get_projections(..., use_spline=...)``. (default: ``False``)
        quintic_chunk (int, optional): SPIKE band-solve chunk length (rows per partition)
            for the quintic projection fit, forwarded to
            :class:`~gpubackendtools.interpolate.QuinticSplineInterpolant`. ``0`` lets GBT
            auto-size it. Only a tuning lever for the quintic path; ignored when
            ``use_spline`` is ``False``. Raising the chunk count increases the GPU
            reduced-solve parallelism in the low-spline-count regime (e.g. a single long
            source, ``ninterps == 2``). (default: ``0``)
        force_backend (str, optional): If given, run this class on the requested backend.
            Options are ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``. (default: ``None``)

    Attributes:
        A_in (xp.ndarray): Array containing y values for linear spline of A
            during Lagrangian interpolation.
        buffer_integer (int): Self-determined buffer necesary for the given
            value for :code:`order`.
        channels_no_delays (2D np.ndarray): Carrier of link index and sign information
            for arms that do not get delayed during TDI computation.
        deps (double): The spacing between Epsilon values in the interpolant
            for the A quantity in Lagrangian interpolation. Hard coded to
            1/(:code:`num_A` - 1).
        dt (double): Inverse of the sampling_frequency.
        E_in (xp.ndarray): Array containing y values for linear spline of E
            during Lagrangian interpolation.
        half_order (int): Half of :code:`order` adjusted to be :code:`int`.
        link_inds (xp.ndarray): Link indexes for delays in TDI.
        link_space_craft_0_in (xp.ndarray): Link indexes for receiver on each
            arm of the LISA constellation.
        link_space_craft_1_in (xp.ndarray): Link indexes for emitter on each
            arm of the LISA constellation.
        nlinks (int): The number of links in the constellation. Typically 6.
        num_A (int): Number of points to use for A spline values used in the Lagrangian
            interpolation. This is hard coded to 1001.
        num_channels (int): 3.
        num_pts (int): Number of points to produce for the final output template.
        order (int): Order of Lagrangian interpolation technique.
        quintic_chunk (int): SPIKE band-solve chunk length forwarded to the quintic
            projection fit (``0`` = GBT auto-sizes). Tuning lever for the quintic path.
        sampling_frequency (double): The sampling rate in Hz.
        tdi (str or list): TDI setup.
        tdi_buffer (int): The buffer necessary for all information needed at early times
            for the TDI computation. This is set to 200.
        use_spline (bool): Default projection interpolation scheme (``True`` = quintic
            spline, ``False`` = Lagrangian). Overridable per call in ``get_projections``.
        xp (obj): Either Numpy or Cupy.

    """

    def __init__(
        self,
        sampling_frequency,
        num_pts,
        order=25,
        tdi="1st generation",
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_orbits: Optional[Orbits] = None,
        tdi_chan="XYZ",
        use_spline=False,
        quintic_chunk=0,
        force_backend=None,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        self.tdi_buffer = 200

        self.num_pts = num_pts

        # Lagrangian interpolation setup
        self.order = order
        self.buffer_integer = self.order * 2 + 1
        self.half_order = int((order + 1) / 2)

        # setup TDI information. ``_init_TDI_delays`` (and the C++ binding
        # below) expects ``self.tdi`` to be a :class:`TDIConfig` instance,
        # so promote string / list inputs here for backwards compatibility
        # with callers that still pass ``tdi="1st generation"``.
        if not isinstance(tdi, TDIConfig):
            tdi = TDIConfig(tdi, force_backend=force_backend)
        self.tdi = tdi
        self.tdi_chan = tdi_chan

        # interpolation scheme for the *projection* waveform evaluation. The TDI
        # computation always uses the Lagrangian path regardless of this setting.
        assert isinstance(use_spline, bool), "use_spline must be a bool."
        self.use_spline = use_spline

        # SPIKE band-solve chunk length for the quintic projection fit (0 = let
        # GBT auto-size). Tuning lever for the low-spline-count regime (e.g. a
        # single long source -> ninterps == 2): a larger chunk count raises the
        # GPU reduced-solve parallelism. Forwarded to QuinticSplineInterpolant.
        assert isinstance(quintic_chunk, int) and quintic_chunk >= 0, (
            "quintic_chunk must be a non-negative int (0 = auto)."
        )
        self.quintic_chunk = quintic_chunk

        super().__init__(force_backend=force_backend)

        # prepare the interpolation of A and E in the Lagrangian interpolation
        self._fill_A_E()

        # setup orbits
        self.response_orbits = orbits

        if tdi_orbits is None:
            tdi_orbits = self.response_orbits

        self.tdi_orbits = tdi_orbits

        if self.num_pts * self.dt > self.response_orbits.t_base.max():
            warnings.warn(
                "Input number of points is longer in time than available orbital information. Trimming to fit orbital information."
            )
            self.num_pts = int(self.response_orbits.t_base.max() / self.dt)

        # setup spacecraft links indexes

        self.tdi_config = self.tdi
        
        # setup TDI info
        self._init_TDI_delays()

        # initialize the cpp holders of orbit and other information
        # self.cpp_response.add_orbit_information(*self.check_add_orbit_args(*self.response_orbits.pycppdetector_args))
        # self.cpp_response.add_tdi_config(*self.tdi_config.pytdiconfig_args)

        self.cpp_orbits = self.backend.OrbitsWrap(*self.response_orbits.pycppdetector_args)
        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self.tdi_config.pytdiconfig_args)
        self.cpp_response = self.backend.LISAResponseWrap(self.cpp_orbits, self.cpp_tdi_config)

        # batched-response state (feat-batching, 2026-06). ``batch_size``
        # defaults to a single source; ``get_projections`` updates it (and
        # ``t0_arr``) per call. ``y_gw_flat``/``t_arr_proj`` start unset so
        # ``get_tdi_delays`` can detect "projections not yet computed".
        self.y_gw_flat = None
        self.t_arr_proj = None
        self.batch_size = 1
        self.t0_arr = self.xp.zeros(1, dtype=self.xp.float64)
        # Per-source sub-sample data-grid shift (paired with t0_arr); updated per
        # call in get_projections. Carried into the kernel eval time so a batch of
        # sources with different sub-sample offsets each land on the data grid.
        self.t0_shift_arr = self.xp.zeros(1, dtype=self.xp.float64)

    def check_add_orbit_args(self, *args):
        """Check orbit arguments for adherence to cpp Orbits class.
        
        # TODO: make this an automatic version based check?
        
        Args are supposed to be [dt, N,  n_arr, L_arr,  x_arr, links, sc_r,  sc_e, armlength].

        Args:
            *args (tuple): Arguments for cpp Orbits class.
            
        """
        try:
            assert len(args) == 9
            assert isinstance(args[0], float)
            assert isinstance(args[1], int)
            assert isinstance(args[2], self.xp.ndarray)
            assert isinstance(args[3], self.xp.ndarray)
            assert isinstance(args[4], self.xp.ndarray)
            assert args[2].dtype == args[3].dtype == args[4].dtype == float
            # assert len(args[2]) == 9 * len(args[3]) == len(args[4])
            assert len(args[5]) == len(self.response_orbits.LINKS)
            assert len(args[6]) == len(self.response_orbits.LINKS)
            assert len(args[7]) == len(self.response_orbits.LINKS)
            assert isinstance(args[8], float)
        except AssertionError:
            raise ValueError("Arguments for cpp class are not correct.")
        
        return args
    
    @property
    def cpp_response(self):
        if self._cpp_response is None:
            raise ValueError("Must add cpp_response and add orbit information.")
        return self._cpp_response

    @cpp_response.setter
    def cpp_response(self, cpp_response):
        self._cpp_response = cpp_response

    @property
    def response_gen(self) -> callable:
        """CPU/GPU function for generating the projections."""
        return self.cpp_response.get_response_wrap

    @property
    def response_quintic_gen(self) -> callable:
        """CPU/GPU function for generating the projections with the quintic spline."""
        return self.cpp_response.get_response_quintic_wrap

    @property
    def tdi_gen(self) -> callable:
        """CPU/GPU function for generating tdi."""
        return self.cpp_response.get_tdi_delays_wrap
    
    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def response_orbits(self) -> Orbits:
        """Response function orbits."""
        return self._response_orbits

    @response_orbits.setter
    def response_orbits(self, orbits: Orbits) -> None:
        """Set response orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)

        self._response_orbits = copy_orbits(orbits)

        if not self._response_orbits.configured:
            self._response_orbits.configure(linear_interp_setup=True)

    @property
    def tdi_orbits(self) -> Orbits:
        """TDI function orbits."""
        return self._tdi_orbits

    @tdi_orbits.setter
    def tdi_orbits(self, orbits: Orbits) -> None:
        """Set TDI orbits."""

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)
        assert orbits.backend.name.split("_")[-1] == self.backend.name.split("_")[-1]

        self._tdi_orbits = copy_orbits(orbits)

        if not self._tdi_orbits.configured:
            self._tdi_orbits.configure(linear_interp_setup=True)

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        # Phase 3L.7l holdout (fixed 2026-06-05): switch from the stale
        # `fastlisaresponse_<flavor>` prefix to the canonical
        # `_BACKEND_PREFIX` ("lisatools" via FastLISAResponseParallelModule).
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def _fill_A_E(self):
        """Set up A and E terms inside the Lagrangian interpolant"""

        factorials = np.asarray([float(get_factorial(n)) for n in range(40)])

        # base quantities for linear interpolant over A
        self.num_A = 1001
        self.deps = 1.0 / (self.num_A - 1)

        eps = np.arange(self.num_A) * self.deps

        h = self.half_order

        denominator = factorials[h - 1] * factorials[h]

        # prepare A
        A_in = np.zeros_like(eps)
        for j, eps_i in enumerate(eps):
            A = 1.0
            for i in range(1, h):
                A *= (i + eps_i) * (i + 1 - eps_i)

            A /= denominator
            A_in[j] = A

        self.A_in = self.xp.asarray(A_in)

        # prepare E
        E_in = self.xp.zeros((self.half_order,))

        for j in range(1, self.half_order):
            first_term = factorials[h - 1] / factorials[h - 1 - j]
            second_term = factorials[h] / factorials[h + j]
            value = first_term * second_term
            value = value * (-1.0) ** j
            E_in[j - 1] = value

        self.E_in = self.xp.asarray(E_in)

    def _init_TDI_delays(self):
        """Initialize TDI specific information"""

        # setup the actual TDI combination
        # if self.tdi in ["1st generation", "2nd generation"]:
        #     # tdi 1.0
        #     tdi_combinations = [
        #         {"link": 13, "links_for_delay": [], "sign": +1},
        #         {"link": 31, "links_for_delay": [13], "sign": +1},
        #         {"link": 12, "links_for_delay": [13, 31], "sign": +1},
        #         {"link": 21, "links_for_delay": [13, 31, 12], "sign": +1},
        #         {"link": 12, "links_for_delay": [], "sign": -1},
        #         {"link": 21, "links_for_delay": [12], "sign": -1},
        #         {"link": 13, "links_for_delay": [12, 21], "sign": -1},
        #         {"link": 31, "links_for_delay": [12, 21, 13], "sign": -1},
        #     ]

        #     if self.tdi == "2nd generation":
        #         # tdi 2.0 is tdi 1.0 + additional terms
        #         tdi_combinations += [
        #             {"link": 12, "links_for_delay": [13, 31, 12, 21], "sign": +1},
        #             {"link": 21, "links_for_delay": [13, 31, 12, 21, 12], "sign": +1},
        #             {
        #                 "link": 13,
        #                 "links_for_delay": [13, 31, 12, 21, 12, 21],
        #                 "sign": +1,
        #             },
        #             {
        #                 "link": 31,
        #                 "links_for_delay": [13, 31, 12, 21, 12, 21, 13],
        #                 "sign": +1,
        #             },
        #             {"link": 13, "links_for_delay": [12, 21, 13, 31], "sign": -1},
        #             {"link": 31, "links_for_delay": [12, 21, 13, 31, 13], "sign": -1},
        #             {
        #                 "link": 12,
        #                 "links_for_delay": [12, 21, 13, 31, 13, 31],
        #                 "sign": -1,
        #             },
        #             {
        #                 "link": 21,
        #                 "links_for_delay": [12, 21, 13, 31, 13, 31, 12],
        #                 "sign": -1,
        #             },
        #         ]

        # elif isinstance(self.tdi, list):
        #     tdi_combinations = self.tdi

        # else:
        #     raise ValueError(
        #         "tdi kwarg should be '1st generation', '2nd generation', or a list with a specific tdi combination."
        #     )
        # self.tdi_combinations = tdi_combinations

        assert isinstance(self.tdi, TDIConfig)

    @property
    def y_gw(self):
        """Projections along the arms"""
        raw = self.y_gw_flat.reshape(self.batch_size, self.nlinks, -1)
        return raw[0] if self.batch_size == 1 else raw

    def _data_time_check(
        self, t_data: np.ndarray, input_in: np.ndarray, t0_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # remove input data that goes beyond orbital information. ``t_data`` is
        # the shared relative time array (starts at 0); the per-source absolute
        # time is ``t_data + t0_arr``. Trim to the worst-case across the batch.
        t_orbit_max = float(self.response_orbits.t.max())
        if bool(
            self.xp.any((t_data + t0_arr.reshape(-1, 1)).max(axis=-1) > t_orbit_max)
        ):
            warnings.warn(
                "Input waveform is longer than available orbital information. Trimming to fit orbital information."
            )

            # ``where(...)[1][-1]`` returned the last in-range column of the LAST
            # source (row-major order), not the batch worst case -- a heterogeneous
            # -t0 batch could then keep eval times > t_orbit_max for an earlier
            # source. Count in-range columns per source (the condition is a prefix
            # since t_data increases) and trim to the shortest-reaching source.
            in_range = (t_data.reshape(1, -1) + t0_arr.reshape(-1, 1)) <= t_orbit_max
            max_ind = int(in_range.sum(axis=-1).min())

            t_data = t_data[:max_ind]
            input_in = input_in[:, :max_ind]
        return (t_data, input_in)

    def get_projections(self, input_in, lam, beta, t0_shift_to_data=0.0, t0=0.0, t_buffer=10000.0, use_spline: Optional[bool] = None, run_async=False):
        """Compute projections of GW signal on to LISA constellation.

        Supports a single source or a batch (feat-batching, 2026-06): a true
        batched CUDA kernel processes all sources in parallel along the grid
        ``z`` dimension. A single source is just ``batch_size == 1``.

        Args:
            input_in (xp.ndarray): Input complex time-domain signal,
                :math:`h_+ + ih_x`. Shape ``(num_pts,)`` for a single source or
                ``(batch_size, num_pts)`` for a batch. CuPy array on the GPU.
            lam (double or array): Ecliptic Longitude in radians. Scalar, or an
                array of length ``batch_size`` for a batch.
            beta (double or array): Ecliptic Latitude in radians. Scalar or
                length-``batch_size`` array.
            t0_shift_to_data (double, optional): Shift to apply to ``t0`` to align
                the input strain with the datastream. Default: 0.0 seconds. 
            t0 (double or array): Absolute start time(s) in seconds. Scalar or
                length-``batch_size`` array.
            t_buffer (double, optional): Buffer time from ``t0``. The start of
                the waveform is garbage because of the delays and interpolation
                towards earlier times; ``t_buffer`` says where to start. Default: 10000.0 seconds.
            use_spline (bool, optional): Per-call override of the instance default
                ``self.use_spline``. If ``None`` (the default), use the instance
                default. If ``True`` / ``False``, force the quintic-spline / Lagrangian
                projection for this call, overriding ``self.use_spline`` in either
                direction. Default: ``None``.
            run_async (bool, optional): If True, use async device allocs/streams.
                (Default: ``False``)

        Raises:
            ValueError: If ``t_buffer`` is not large enough.
        """
        # None -> fall back to the instance default; an explicit bool overrides it
        # in either direction (force the spline on or off for this call).
        assert use_spline is None or isinstance(use_spline, bool), (
            "use_spline must be None or a bool."
        )
        use_spline = self.use_spline if use_spline is None else use_spline

        # --- batch detection (scalar lam/beta -> batch_size 1) ---
        lam = self.xp.ascontiguousarray(
            self.xp.atleast_1d(self.xp.asarray(lam, dtype=self.xp.float64))
        )
        beta = self.xp.ascontiguousarray(
            self.xp.atleast_1d(self.xp.asarray(beta, dtype=self.xp.float64))
        )
        batch_size = len(lam)

        assert np.all(np.abs(t0_shift_to_data) < self.dt), (
            "t0_shift_to_data should be less than the data time step (dt)."
        )
        # t0_arr is the waveform/parameter reference time = the EXACT start of the input
        # waveform. t0_shift_to_data (the sub-sample data-grid alignment) must NOT be baked
        # in here; it belongs only on the evaluation grid (t_arr, below). The kernels index
        # the waveform array via ``delay - t0_arr``, so a shift baked into t0_arr cancels
        # against the same shift in the eval time and the waveform ends up sampled off the
        # data grid by t0_shift_to_data.
        t0_arr = self.xp.ascontiguousarray(
                self.xp.atleast_1d(self.xp.asarray(t0, dtype=np.float64))
        )
        if t0_arr.ndim == 1 and t0_arr.shape[0] == 1:
            # broadcast a shared t0 across the batch
            t0_arr = t0_arr.repeat(batch_size)
        self.batch_size = batch_size

        self.tdi_start_ind = int(t_buffer / self.dt)
        # get necessary buffer for TDI
        self.check_tdi_buffer = int(100.0 * self.sampling_frequency) + 4 * self.order

        tmp_orbits = deepcopy(self.response_orbits.x_base)
        self.projection_buffer = (
            int(
                (
                    np.sum(
                        tmp_orbits.copy() * tmp_orbits.copy(),
                        axis=-1,
                    )
                    ** (1 / 2)
                ).max()
                * C_inv
            )
            + 4 * self.order
        )
        self.projections_start_ind = self.tdi_start_ind - 2 * self.check_tdi_buffer

        if self.projections_start_ind < self.projection_buffer:
            raise ValueError(
                "Need to increase t_buffer. The initial buffer is not large enough."
            )

        # --- promote input_in to (batch_size, num_pts) ---
        input_in = self.xp.asarray(input_in)
        if input_in.ndim == 1:
            input_in = input_in.reshape(1, -1)
        assert input_in.shape[0] == batch_size, (
            f"input_in batch dim {input_in.shape[0]} != batch_size {batch_size}."
        )
        num_inputs_per_source = input_in.shape[1]
        self.num_total_points = num_inputs_per_source

        # Shared relative evaluation grid, starting at 0. The per-source absolute
        # eval time is ``t_arr + t0_arr + t0_shift_arr`` inside the kernel: t0_arr is
        # the waveform's exact start, t0_shift_arr the sub-sample data-grid shift.
        # The shift rides the kernel eval time (NOT this shared grid) so each source
        # in a batch can carry its own sub-sample offset; t0_arr still anchors the
        # waveform-array index, so the shift survives in clipped_delay exactly as in
        # the single-source path. (Pre-2026-06 the shift was baked into t_arr here,
        # which a per-source batch cannot represent -> broadcast crash for B>=2.)
        t_arr = self.xp.arange(num_inputs_per_source, dtype=self.xp.float64) * self.dt
        t0_shift_arr = self.xp.ascontiguousarray(
            self.xp.atleast_1d(
                self.xp.asarray(t0_shift_to_data, dtype=self.xp.float64)
            )
        )
        if t0_shift_arr.shape[0] == 1:
            t0_shift_arr = t0_shift_arr.repeat(batch_size)
        assert t0_shift_arr.shape[0] == batch_size, (
            f"t0_shift_to_data length {t0_shift_arr.shape[0]} != batch_size {batch_size}."
        )
        t_arr, input_in = self._data_time_check(t_arr, input_in, t0_arr)
        num_inputs_per_source = input_in.shape[1]

        assert num_inputs_per_source >= self.num_pts

        # --- batched sky vectors (flat: batch_size * 3) ---
        k_in = self.xp.zeros(batch_size * 3, dtype=self.xp.float64)
        u_in = self.xp.zeros(batch_size * 3, dtype=self.xp.float64)
        v_in = self.xp.zeros(batch_size * 3, dtype=self.xp.float64)

        cb = self.xp.cos(beta)
        sb = self.xp.sin(beta)
        cl = self.xp.cos(lam)
        sl = self.xp.sin(lam)

        v_in[0::3] = -sb * cl
        v_in[1::3] = -sb * sl
        v_in[2::3] = cb
        u_in[0::3] = sl
        u_in[1::3] = -cl
        u_in[2::3] = 0.0
        k_in[0::3] = -cb * cl
        k_in[1::3] = -cb * sl
        k_in[2::3] = -sb

        self.nlinks = 6
        input_flat = input_in.reshape(-1)  # (batch_size * num_inputs_per_source,)

        y_gw = self.xp.zeros(batch_size * self.nlinks * self.num_pts, dtype=self.xp.float64)
        if not use_spline: # Lagrangian interpolation path. 
            self.response_gen(
                y_gw,
                t_arr,
                k_in,
                u_in,
                v_in,
                self.dt,
                num_inputs_per_source,
                input_flat,
                num_inputs_per_source,
                self.order,
                self.sampling_frequency,
                self.buffer_integer,
                self.A_in,
                self.deps,
                len(self.A_in),
                self.E_in,
                self.projections_start_ind,
                t0_arr,
                t0_shift_arr,
                batch_size,
                run_async,
            )
        else:
            # Quintic-spline projection path. Fit the real & imag parts of the
            # batched waveform on the shared relative-time grid in ONE combined
            # QuinticSplineInterpolant (ninterps = 2*batch_size: the batch_size
            # real splines first, then the batch_size imag splines), then hand the
            # two contiguous coefficient halves to the quintic kernel. The single
            # combined fit maximizes the spline count for the GBT band-solve
            # coalescing (see plan "Interaction with GBT GPU-opt #4").
            from gpubackendtools.interpolate import (
                QuinticSplineInterpolant,
                CUBIC_SPLINE_LINEAR_SPACING,
            )

            assert num_inputs_per_source >= 6, "Quintic projection requires at least 6 input samples."

            flavor = self.backend.name.split("_")[-1]
            # interp-major flat layout: [real_0..real_{B-1}, imag_0..imag_{B-1}],
            # each block ``num_inputs_per_source`` long, on the shared uniform grid t_arr.
            x_grid = self.xp.tile(t_arr, 2 * batch_size)
            y_stack = self.xp.concatenate(
                [input_in.real, input_in.imag], axis=0
            ).reshape(-1)

            spl = QuinticSplineInterpolant(
                x_grid, y_stack, ninterps=2 * batch_size, length=num_inputs_per_source,
                force_backend=flavor, _chunk=self.quintic_chunk,
            )
            assert spl.spline_type == CUBIC_SPLINE_LINEAR_SPACING, (
                "Quintic projection assumes a uniform time grid."
            )
            assert spl.xp is self.xp, "Quintic fit backend does not match the response backend."

            # Retain only the five coefficient buffers, then drop everything else.
            # The quintic eval reads y0 from ``input_flat`` and computes x0
            # analytically on the uniform grid, so it never touches the fit's x/y
            # arrays -- those (~2 device arrays of 2*batch_size*num_inputs_per_source
            # doubles; e.g. ~1.7 GB for 100x 1-month sources @ 5 s) are dead weight
            # after the solve. ``c1_flat..c5_flat`` are standalone allocations, so
            # holding just them (and dropping the interpolant plus the x_grid/y_stack
            # temporaries it aliases) frees x/y before the kernel launches, while
            # keeping the coeff data alive across the (possibly async) call and until
            # the next get_projections.
            self._quintic_coeffs = (
                spl.c1_flat, spl.c2_flat, spl.c3_flat, spl.c4_flat, spl.c5_flat
            )
            del spl, x_grid, y_stack
            c1f, c2f, c3f, c4f, c5f = self._quintic_coeffs

            half = batch_size * num_inputs_per_source  # real coeffs in [:half], imag coeffs in [half:]
            self.response_quintic_gen(
                y_gw,
                t_arr,
                k_in,
                u_in,
                v_in,
                self.dt,
                num_inputs_per_source,
                input_flat,
                num_inputs_per_source,
                self.sampling_frequency,
                c1f[:half], c2f[:half], c3f[:half], c4f[:half], c5f[:half],
                c1f[half:], c2f[half:], c3f[half:], c4f[half:], c5f[half:],
                self.projections_start_ind,
                int(CUBIC_SPLINE_LINEAR_SPACING),
                t0_arr,
                t0_shift_arr,
                batch_size,
                run_async,
            )

        self.t_arr_proj = t_arr
        self.t0_arr = t0_arr
        self.t0_shift_arr = t0_shift_arr
        self.y_gw_flat = y_gw
        self.y_gw_length = self.num_pts

    @property
    def XYZ(self):
        """Return links as an array"""
        raw = self.delayed_links_flat.reshape(self.batch_size, 3, -1)
        return raw[0] if self.batch_size == 1 else raw

    def get_tdi_delays(self, t_arr=None, y_gw=None, run_async=False):
        """Get TDI combinations from projections.

        This functions generates the TDI combinations from the projections
        computed with ``get_projections``. It can return XYZ, AET, or AE depending
        on what was input for ``tdi_chan`` into ``__init__``. Batched
        (feat-batching, 2026-06): when ``get_projections`` was run on a batch,
        each returned channel has shape ``(batch_size, num_pts)``.

        Args:
            t_arr (xp.ndarray, optional): Time array. Only provide when entering
                ``y_gw`` directly; otherwise the projection time array is reused.
            y_gw (xp.ndarray, optional): Projections along the arms (single
                source), 2D with shape ``(nlinks, num_pts)``. The link order is
                ``orbits.LINKS``. (Default: ``None``)
            run_async (bool, optional): If True, use async device allocs/streams.
                (Default: ``False``)

        Returns:
            tuple: (X,Y,Z) or (A,E,T) or (A,E). Each entry is ``(num_pts,)`` for
            a single source or ``(batch_size, num_pts)`` for a batch.

        Raises:
            ValueError: If ``tdi_chan`` is not one of the options.
        """
        # y_gw entered directly -> always a single source.
        if y_gw is not None:
            assert y_gw.shape == (len(self.orbits.LINKS), self.num_pts)
            self.batch_size = 1
            self.t0_arr = self.xp.zeros(1, dtype=self.xp.float64)
            self.t0_shift_arr = self.xp.zeros(1, dtype=self.xp.float64)
            self.y_gw_flat = y_gw.flatten().copy()
            self.y_gw_length = self.num_pts
            if t_arr is None:
                raise ValueError("If entering y_gw directly, also need to enter t_arr directly.")
            assert t_arr.shape == (self.num_pts,)

        elif self.y_gw_flat is None:
            raise ValueError(
                "Need to either enter projection array or have this code determine projections."
            )
        else:
            assert self.t_arr_proj is not None
            t_arr = self.t_arr_proj

        self.delayed_links_flat = self.xp.zeros(
            self.batch_size * 3 * self.num_pts, dtype=self.xp.float64
        )

        # NOTE: unit_starts / unit_lengths now live on the TDIConfig object
        # (passed to LISAResponse at construction), so they are no longer
        # threaded through tdi_gen here.

        # The projections (input_links / y_gw) were evaluated at
        # ``t_arr[i] + t0_arr[bin] + t0_shift_arr[bin]`` in get_projections, so y_gw
        # BEGINS at ``t_arr[0] + t0_arr[bin] + t0_shift_arr[bin]``. The TDI kernel
        # indexes y_gw via ``delay - t0_arr`` (it subtracts only its own t0_arr arg),
        # so that arg must equal the projection start. Re-split the grid so the
        # relative axis starts at 0 and the per-source absolute offset carries the
        # full start: ``t_arr_start`` covers the direct-entry path (user t_arr[0] != 0),
        # and ``t0_shift_arr`` carries the per-source sub-sample data-grid shift (on the
        # standard path t_arr[0] == 0 and the shift lives entirely in t0_shift_arr).
        # The kernel eval time (t_arr_tdi + t0_arr_tdi) is invariant; only the y_gw
        # array reference is corrected.
        t_arr_start = t_arr[0]
        t_arr_tdi = t_arr - t_arr_start
        t0_arr_tdi = self.t0_arr + t_arr_start + self.t0_shift_arr

        self.tdi_gen(
            self.delayed_links_flat,
            self.y_gw_flat,
            self.y_gw_length,
            self.num_pts,
            t_arr_tdi,
            self.order,
            self.sampling_frequency,
            self.buffer_integer,
            self.A_in,
            self.deps,
            len(self.A_in),
            self.E_in,
            self.tdi_start_ind,
            t0_arr_tdi,
            self.batch_size,
            run_async,
        )

        xyz = self.XYZ  # (3, num_pts) if batch_size == 1, else (batch_size, 3, num_pts)
        if self.batch_size == 1:
            X, Y, Z = xyz
        else:
            X, Y, Z = xyz[:, 0, :], xyz[:, 1, :], xyz[:, 2, :]

        if self.tdi_chan == "XYZ":
            return X, Y, Z
        elif self.tdi_chan in ("AET", "AE"):
            A, E, T = AET(X, Y, Z)
            return (A, E, T) if self.tdi_chan == "AET" else (A, E)
        else:
            raise ValueError("tdi_chan must be 'XYZ', 'AET' or 'AE'.")


class ResponseWrapper(FastLISAResponseParallelModule):
    """Wrapper to produce LISA TDI from TD waveforms

    This class takes a waveform generator that produces :math:`h_+ \pm ih_x`.
    (:code:`flip_hx` is used if the waveform produces :math:`h_+ - ih_x`).
    It takes the complex waveform in the SSB frame and produces the TDI channels
    according to settings chosen for :class:`pyResponseTDI`.

    The waveform generator must have :code:`kwargs` with :code:`T` for the observation
    time in years and :code:`dt` for the time step in seconds.

    Args:
        waveform_gen (obj): Function or class (with a :code:`__call__` function) that takes parameters and produces
            :math:`h_+ \pm h_x`.
        Tobs (double): Observation time in years.
        dt (double): Time between time samples in seconds. The inverse of the sampling frequency.
        index_lambda (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_lambda`
            tells the class the index of the ecliptic longitude within this list of
            parameters.
        index_beta (int): The user will input parameters. The code will read these in
            with the :code:`*args` formalism producing a list. :code:`index_beta`
            tells the class the index of the ecliptic latitude (or ecliptic polar angle)
            within this list of parameters.
        t0 (double, optional): Initial time at which to start the waveform. (Default: 0.0)
        t_buffer (double, optional): Start of returned waveform in seconds (with respect to the start of the observation) leaving ample time for garbage at
            the beginning of the waveform. It also removed the same amount from the end. (Default: 10000.0)
        flip_hx (bool, optional): If True, :code:`waveform_gen` produces :math:`h_+ - ih_x`.
            :class:`pyResponseTDI` takes :math:`h_+ + ih_x`, so this setting will
            multiply the cross polarization term out of the waveform generator by -1.
            (Default: :code:`False`)
        remove_sky_coords (bool, optional): If True, remove the sky coordinates from
            the :code:`*args` list. This should be set to True if the waveform
            generator does not take in the sky information. (Default: :code:`False`)
        is_ecliptic_latitude (bool, optional): If True, the latitudinal sky
            coordinate is the ecliptic latitude. If False, thes latitudinal sky
            coordinate is the polar angle. In this case, the code will
            convert it with :math:`\beta=\pi / 2 - \Theta`. (Default: :code:`True`)
        force_backend (str, optional): If given, run this class on the requested backend. 
            Options are ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``. (default: ``None``)
        remove_garbage (bool or str, optional): If True, it removes everything before ``t_buffer``
            and after the end time - ``t_buffer``. If ``str``, it must be ``"zero"``. If ``"zero"``,
            it will not remove the points, but set them to zero. This is ideal for PE. (Default: ``True``)
        n_overide (int, optional): If not ``None``, this will override the determination of
            the number of points, ``n``, from ``int(T/dt)`` to the ``n_overide``. This is used
            if there is an issue matching points between the waveform generator and the response
            model.
        orbits (:class:`Orbits`, optional): Orbits class from LISA Analysis Tools. Works with LISA Orbits
            outputs: ``lisa-simulation.pages.in2p3.fr/orbits/``.
            (default: :class:`EqualArmlengthOrbits`)
        **kwargs (dict, optional): Keyword arguments passed to :class:`pyResponseTDI`.

    """

    def __init__(
        self,
        waveform_gen,
        Tobs,
        dt,
        index_lambda,
        index_beta,
        t0=0.0,
        t0_shift_to_data=0.0,
        t_buffer=10000.0,
        flip_hx=False,
        remove_sky_coords=False,
        is_ecliptic_latitude=True,
        force_backend=None,
        remove_garbage=True,
        n_overide=None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        **kwargs,
    ):

        # store all necessary information
        self.waveform_gen = waveform_gen
        self.index_lambda = index_lambda
        self.index_beta = index_beta
        self.dt = dt
        self.t0 = t0
        self.t0_shift_to_data = t0_shift_to_data
        self.t_buffer = t_buffer
        self.sampling_frequency = 1.0 / dt
        super().__init__(force_backend=force_backend)

        if orbits is None:
            orbits = EqualArmlengthOrbits()

        assert isinstance(orbits, Orbits)

        if Tobs * YRSID_SI > orbits.t_base.max():  # Tobs * YRSID_SI > (orbits.ltt_t.max() - orbits.ltt_t.min()):
            warnings.warn(
                f"Tobs is larger than available orbital information time array. Reducing Tobs to {orbits.t_base.max()}"
                # f"Tobs is larger than available orbital information time array. Reducing Tobs to {orbits.ltt_t.max() - orbits.ltt_t.min()}"
            )
            Tobs = orbits.t_base.max() / YRSID_SI
            # Tobs = (orbits.ltt_t.max() - orbits.ltt_t.min()) / YRSID_SI
        if n_overide is not None:
            if not isinstance(n_overide, int):
                raise ValueError("n_overide must be an integer if not None.")
            self.n = n_overide

        else:
            self.n = int(Tobs * YRSID_SI / dt)

        self.Tobs = self.n * dt
        self.is_ecliptic_latitude = is_ecliptic_latitude
        self.remove_sky_coords = remove_sky_coords
        self.flip_hx = flip_hx
        self.remove_garbage = remove_garbage

        # initialize response function class
        self.response_model = pyResponseTDI(
            self.sampling_frequency, self.n, orbits=orbits, force_backend=force_backend, **kwargs
        )

        self.Tobs = (self.n * self.response_model.dt) / YRSID_SI

    @staticmethod
    def get_t0_shift_to_data(t_arr: np.ndarray[float], dt: float, t_start: float) -> float:
        # just in case the time grids do not align
        _fake_data = (np.arange(10000) - int(10000 / 2)) * dt + t_arr[0]
        assert np.abs(t_start - t_arr[0]) / dt < 10000
        diff = np.abs(_fake_data - t_start)
        _fake_data_closest = _fake_data[diff.argmin()]
        t0_shift_to_data = _fake_data_closest - t_start
        return t0_shift_to_data
    
    @property
    def xp(self) -> object:
        return self.backend.xp

    @property
    def citation(self):
        """Get citations for use of this code"""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        # Phase 3L.7l holdout (fixed 2026-06-05): switch from the stale
        # `fastlisaresponse_<flavor>` prefix to the canonical
        # `_BACKEND_PREFIX` ("lisatools" via FastLISAResponseParallelModule).
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(self, *args, convert_to_ra_dec: Optional[bool] = None, **kwargs):
        """Run the waveform and response generation

        Sky coordinates are consumed **in the orbits frame** directly
        (matching the TDI-on-the-fly handling) — no frame conversion is
        applied. Load the orbits in the frame you sample in (see
        ``Orbits.frame``; the sprint convention is ``"ecliptic"``).

        Args:
            *args (list): Arguments to the waveform generator. This must include
                the sky coordinates, expressed in the orbits frame.
            convert_to_ra_dec (bool, optional): **Deprecated.** Legacy
                ecliptic -> ICRS conversion for setups that sample in the
                ecliptic frame against ICRS-frame orbits. Default ``None``
                (no conversion). Passing ``True`` still converts but emits
                a ``DeprecationWarning``.
            **kwargs (dict): kwargs necessary for the waveform generator.

        Return:
            list: TDI Channels.

        """

        args = list(args)

        # get sky coords
        beta = args[self.index_beta]
        lam = args[self.index_lambda]

        # remove them from the list if waveform generator does not take them
        if self.remove_sky_coords:
            args.pop(self.index_beta)
            args.pop(self.index_lambda)

        # transform polar angle
        if not self.is_ecliptic_latitude:
            beta = np.pi / 2.0 - beta

        # add the new Tobs and dt info to the waveform generator kwargs
        kwargs["T"] = self.Tobs
        kwargs["dt"] = self.dt

        # get the waveform
        h = self.waveform_gen(*args, **kwargs)

        if self.flip_hx:
            h = h.real - 1j * h.imag

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam, beta = ecliptic_to_icrs(lam, beta)

        # TODO: make this customizable
        self.response_model.get_projections(h, lam, beta, t0_shift_to_data=self.t0_shift_to_data, t0=self.t0, t_buffer=self.t_buffer)
        tdi_out = self.response_model.get_tdi_delays()  # will take care of t0 automatically to match projections

        out = list(tdi_out)
        if self.remove_garbage is True:  # bool
            for i in range(len(out)):
                out[i] = out[i][
                    self.response_model.tdi_start_ind : -self.response_model.tdi_start_ind
                ]

        elif isinstance(self.remove_garbage, str):  # bool
            if self.remove_garbage != "zero":
                raise ValueError("remove_garbage must be True, False, or 'zero'.")
            for i in range(len(out)):
                out[i][: self.response_model.tdi_start_ind] = 0.0
                out[i][-self.response_model.tdi_start_ind :] = 0.0

        return out
