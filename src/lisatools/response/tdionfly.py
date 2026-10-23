r"""TDI on the fly: the LISA response evaluated directly from a source's amplitude and phase.

Instead of projecting a sampled :math:`h_+, h_\times` time series
(:mod:`lisatools.response.directresponse`), these generators evaluate the
link projections and the TDI combination (:class:`~lisatools.response.tdiconfig.TDIConfig`)
point-wise from the source's slowly varying amplitude and phase, on a
(sparse) grid of evaluation times, in the C++/CUDA ``*TDIWaveformWrap``
kernels. The output is an amplitude / phase representation per channel
(:class:`TDIOutput`), with ``X = Xamp exp(-i (Xphase + phase_ref))``.

Classes:

- :class:`TDIonTheFly`: the base (backend, orbits, TDI configuration).
- :class:`TDTDIonTheFly`: time-domain amplitude / phase splines (BBHx MBH and
  SOBBH TDI-on-the-fly paths; :class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase`).
- :class:`FDTDIonTheFly`: frequency-domain amplitude / frequency splines.
- :class:`GBTDIonTheFly`: the galactic-binary kernel (``gbgpu`` backend); GBGPU's
  lookup and signal-heterodyne engines take its ``wave_gen`` handle.
- :class:`GBFDTDIonTheFly`: heterodyned frequency-domain GB TDI (``gbgpu`` backend).
- :class:`SOBBHTDIonTheFly`: the SOBBH PN kernel (``bbhx`` backend).
- :class:`TDDenseTDIonTheFly`: template-batched multi-harmonic EMRI TDI with exact
  dense-output phases (the ``EMRI_DIRECT_RESPONSE=dense`` production path,
  :class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM`).
- :class:`TDIOutput`, :class:`TDTDIOutput`, :class:`FDTDIOutput`: the outputs.

Sky angles are consumed in the orbits' frame (ICRS in the global fit: ``lam``
= RA, ``beta`` = Dec); ``convert_to_ra_dec=True`` is deprecated. Orbits are
configured once per identity and shared (``LISATOOLS_ORBITS_CACHE``; see
:func:`_get_configured_orbits`). Choose the backend per object with
``force_backend=``.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
import warnings
from typing import Tuple
from copy import deepcopy
from gpubackendtools import wrapper
        
import time
import h5py
from .directresponse import ecliptic_to_icrs, warn_deprecated_frame_conversion

try:
    import cupy as cp

except (ImportError, ModuleNotFoundError) as e:
    import numpy as cp

from scipy.interpolate import CubicSpline as CubicSpline_scipy
from gpubackendtools.interpolate import CubicSplineInterpolant


# ---------------------------------------------------------------------------
# Configured-orbits cache
# ---------------------------------------------------------------------------
# ``TDIonTheFly`` is constructed once per waveform evaluation, and its
# ``orbits`` setter used to ``deepcopy`` the incoming (unconfigured) Orbits and
# then trigger lazy configuration. ``Orbits._configure`` re-runs a scipy
# CubicSpline over every orbit table component, which profiling (job 12930220)
# measured at ~752 s of the 984 s spent in the MBH waveform call -- ~76% of the
# whole run, redone identically on every call.
#
# ``_configure`` writes only plain numpy state (_ltt/_x/_n/_v, t, dt,
# pycppdetector_args) and is backend-independent, so one configured instance is
# reusable everywhere the configuration inputs match. Entries are shared, not
# copied: copying a configured Orbits is itself GBs of memcpy. The response
# path treats orbits as read-only; set ``LISATOOLS_ORBITS_CACHE=0`` to restore
# the old per-instance deepcopy if you need to mutate them.
_ORBITS_CONFIGURED_CACHE: dict = {}


def _orbits_cache_key(orbits) -> Optional[tuple]:
    """Return the identity of everything ``Orbits._configure`` depends on, or None.

    Includes the BACKEND and the current DEVICE: the configured tables are arrays of the
    orbits' own backend, allocated on the device current at configuration. Without them a
    process that builds the same orbits on two backends (TDDenseGPUParityTest: cpu, then
    gpu) handed the numpy tables to ``OrbitsWrapGPU``, and a multi-GPU process would serve
    one device's tables to another.

    Args:
        orbits (Orbits): The (unconfigured) orbits.

    Returns:
        tuple or None: Hashable key (class, backend, device, frame, armlength, ``t0``,
        filename and the ``_configure_kwargs``, arrays hashed by content); ``None`` when
        anything is unhashable (then no caching).
    """
    from ..utils.device import current_device

    try:
        backend = getattr(orbits, "backend", None)
        parts = [
            type(orbits).__module__,
            type(orbits).__name__,
            getattr(backend, "backend_name", None),
            current_device(getattr(backend, "xp", None)),
            getattr(orbits, "frame", None),
            getattr(orbits, "armlength", None),
            getattr(orbits, "t0", None),
            getattr(orbits, "filename", None),
        ]
        for name, val in sorted(getattr(orbits, "_configure_kwargs", {}).items()):
            if isinstance(val, np.ndarray):
                # hash contents: t_arr defines the output grid exactly
                parts.append((name, val.shape, val.dtype.str, hash(val.tobytes())))
            else:
                parts.append((name, val))
        return tuple(parts)
    except Exception:
        # Anything unhashable / unexpected -> no caching, old behaviour.
        return None


def _get_configured_orbits(orbits):
    """Return a configured Orbits for ``orbits``, reusing an identical one.

    Order: ``LISATOOLS_ORBITS_CACHE=0`` -> a configured deepcopy (old
    behaviour); an already-configured ``orbits`` -> itself; no cache key ->
    a configured deepcopy; otherwise the shared cached instance for the key
    (configured once, then reused by every caller, so treat it as read-only).

    Args:
        orbits (Orbits): The orbits passed to a generator.

    Returns:
        Orbits: A configured orbits object.
    """
    import os

    if os.environ.get("LISATOOLS_ORBITS_CACHE", "1") == "0":
        out = deepcopy(orbits)
        out.pycppdetector_args
        return out

    if getattr(orbits, "configured", False):
        # Already configured by the caller: honour it, don't re-spline.
        return orbits

    key = _orbits_cache_key(orbits)
    if key is None:
        out = deepcopy(orbits)
        out.pycppdetector_args
        return out

    cached = _ORBITS_CONFIGURED_CACHE.get(key)
    if cached is None:
        cached = deepcopy(orbits)
        # Force the (expensive) lazy configuration exactly once per key.
        cached.pycppdetector_args
        _ORBITS_CONFIGURED_CACHE[key] = cached
    return cached

from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.utils.utility import AET
from gpubackendtools import wrapper
            
from ..utils.parallelbase import LISAToolsParallelModule
from ..utils.stagetimer import TIMING, TIMERS, TIMER_COUNTS
from ..utils.stagetimer import stage as _stage
from .tdiconfig import TDIConfig

from ..utils.constants import YRSID_SI  # lisaconstants ASTRONOMICAL_YEAR


# TODO check: keep/remove/document -- no callers (a copy of directresponse.get_factorial)
def get_factorial(n):
    """Return ``n!`` by repeated multiplication."""
    fact = 1

    for i in range(1, n + 1):
        fact = fact * i

    return fact


from math import factorial

factorials = np.array([factorial(i) for i in range(30)])

from ..utils.constants import C_SI as _C_SI

C_inv = 1.0 / _C_SI  # 3.3356409519815204e-09, the C side's C_inv


class CubicSpline:
    """Empty placeholder type, used only in annotations.

    Stands in for the retired cython spline class; the splines used here are
    :class:`gpubackendtools.interpolate.CubicSplineInterpolant`.
    """
    pass

class TDIonTheFly(LISAToolsParallelModule):
    """Base class for LISA TDI on the fly (backend, orbits and TDI configuration).

    Holds the backend, the configured orbits (``cpp_orbits``, a backend
    ``OrbitsWrap``) and the TDI configuration (``cpp_tdi_config``, a backend
    ``TDIConfigWrap``) that every subclass's C++ waveform wrap takes. The
    subclasses add the evaluation grid (``t_arr``, ``N``) and the
    ``wave_gen`` kernel handle that :meth:`__call__` runs. Runs on CPU or GPU
    (and ``jax`` for the subclasses that support it).

    Args:
        sampling_frequency (float): Sampling frequency in Hz; ``dt = 1 / sampling_frequency``
            (subclasses overwrite ``dt`` from their time grid).
        num_sub (int): Number of sources (sub-waveforms) per call.
        n_params (int, optional): Parameters per source in the kernel's flat
            parameter array. Default ``4`` (``inc, psi, lam, beta``).
        tdi_config: A :class:`~lisatools.response.tdiconfig.TDIConfig`, a
            generation string (``"1st generation"`` / ``"2nd generation"``,
            built with the default backend), or ``None`` for
            ``TDIConfig("1st generation")``.
        orbits: An :class:`~lisatools.detector.Orbits` instance, an
            ``Orbits`` subclass (instantiated with its defaults), or ``None``
            (:class:`~lisatools.detector.EqualArmlengthOrbits`). Default the
            :class:`~lisatools.detector.EqualArmlengthOrbits` class; the global
            fit passes the unequal-arm mojito orbits.
        tdi_chan (str, optional): Stored as ``tdi_chan`` only; the channels
            returned are always the configuration's X, Y, Z. Default ``"XYZ"``.
        force_backend (str, optional): Backend name (``"cpu"``,
            ``"cuda12x"``, ``"cuda13x"``, ``"jax"``, ...). Default ``None``
            (first available of :meth:`supported_backends`).

    Raises:
        ValueError: ``tdi_config`` is neither ``None``, a string nor a
            :class:`~lisatools.response.tdiconfig.TDIConfig`.

    Attributes:
        cpp_orbits (object): Backend ``OrbitsWrap`` of the configured orbits.
        cpp_tdi_config (object): Backend ``TDIConfigWrap``.
        num_sub (int): Sources per call.
        n_params (int): Parameters per source.
    """

    def __init__(
        self,
        sampling_frequency,
        num_sub,
        n_params=4,
        tdi_config: Optional[TDIConfig] = None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_chan="XYZ",
        force_backend=None,
    ):

        # setup all quantities
        self.sampling_frequency = sampling_frequency
        self.dt = 1 / sampling_frequency
        self.n_params = n_params
        self.num_sub = num_sub

        # setup TDI information
        self.tdi_chan = tdi_chan
        super().__init__(force_backend=force_backend)

        # setup orbits
        with _stage("3a_orbits_setter", self.xp):
            self.orbits = orbits
        with _stage("3b_tdi_config_setter", self.xp):
            self.tdi_config = tdi_config
        # setup TDI info
        
    @property
    def tdi_config(self) -> TDIConfig:
        """The TDI configuration; setting it also rebuilds ``cpp_tdi_config``."""
        return self._tdi_config
    
    @tdi_config.setter
    def tdi_config(self, tdi_config: TDIConfig):
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation")
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config)
        elif not isinstance(tdi_config, TDIConfig):
            raise ValueError("TDI Config needs to be a string or an instnace of TDIConfig.")
        self._tdi_config = tdi_config

        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self._tdi_config.pytdiconfig_args)
       
    @property
    def xp(self) -> object:
        """The backend's array module (numpy, cupy or jax.numpy)."""
        return self.backend.xp
    
    @property
    def orbits(self) -> object:
        """The configured orbits (shared through :func:`_get_configured_orbits`)."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        """Set the orbits (instance, ``Orbits`` subclass or ``None``) and rebuild ``cpp_orbits``."""
        if orbits is None:
            orbits = EqualArmlengthOrbits()
        
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            # assumed default arguments if not initialized as input
            orbits = orbits()

        else:
            assert isinstance(orbits, Orbits)

        # Reuses an identically-configured Orbits instead of re-splining the
        # orbit tables on every waveform call. See _get_configured_orbits.
        with _stage("3a1_get_configured_orbits", self.xp):
            self._orbits = _get_configured_orbits(orbits)

        # pycppdetector_args triggers lazy configuration if needed.
        with _stage("3a2_OrbitsWrap", self.xp):
            self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)
    
    @property
    def citation(self):
        """str: Citations for this code (currently a placeholder string)."""

        return """
        # TODO add
        """
    
    @classmethod
    def supported_backends(cls):
        """Return ``"<prefix>_<tag>"`` for ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``, ``jax``.

        Returns:
            list[str]: Backend names in preference order (``<prefix>`` is the
            class's ``_BACKEND_PREFIX``: ``lisatools``, ``gbgpu`` or ``bbhx``).
        """
        # GPU_RECOMMENDED_WITH_JAX includes 'jax' so that GBTDIonTheFly and
        # SOBBHTDIonTheFly (which inherit this) can dispatch to the pure
        # JAX backend in fastlisaresponse.jax via force_backend='jax'.
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    def __call__(self, inc, psi, lam, beta, return_spline: bool =False) -> TDIOutput:
        """Run the subclass kernel (``wave_gen``) for ``num_sub`` sources.

        Used by the subclasses whose only per-call parameters are the four
        extrinsic angles (:class:`TDTDIonTheFly`, :class:`FDTDIonTheFly`);
        needs the subclass's ``t_arr``, ``N`` and ``wave_gen``.

        Args:
            inc (array-like): Inclination [rad], length ``num_sub``.
            psi (array-like): Polarization angle [rad], length ``num_sub``.
            lam (array-like): Sky longitude [rad] in the orbits' frame (RA for
                ICRS orbits), length ``num_sub``.
            beta (array-like): Sky latitude [rad] in the orbits' frame (Dec for
                ICRS orbits), length ``num_sub``.
            return_spline: Build the output's amplitude / phase splines.

        Returns:
            TDIOutput: The subclass's :meth:`from_tdi_output` of the raw output,
            ``tdi_amp`` / ``tdi_phase`` of shape ``(num_sub, nchannels, N)`` and
            ``phase_ref`` of shape ``(num_sub, N)``.
        """
        
        params = self.xp.asarray([inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 4 * self.num_sub

        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        _wg = self.wave_gen
        with _stage("4b_run_wave_tdi_kernel", self.xp):
            _wg.run_wave_tdi_wrap(
                tdi_channels_arr,
                tdi_amp, tdi_phase,
                phase_ref,
                params, self.t_arr.flatten().copy(),
                self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
            )
        
        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)
        # Propagate THIS generator's backend to the output object (as the four
        # sibling from_tdi_output constructors do): without it the TDIOutput
        # falls back to get_first_backend(supported_backends()), which resolves
        # to CUDA on any cupy-equipped box regardless of the generator's actual
        # backend -- so a CPU/numpy generator's arrays hit a cuda interpolant
        # in build_spline and raise the nanobind device mismatch.
        return self._timed_from_tdi_output(TDIOutput(
            self.t_arr,
            tdi_amp.reshape(reshape_shape),
            tdi_phase.reshape(reshape_shape),
            phase_ref.reshape(self.t_arr.shape),
            force_backend=self.backend.name.split("_")[-1],
        ), fill_splines=return_spline)
    
    def _timed_from_tdi_output(self, tdi_output, fill_splines=False):
        """:meth:`from_tdi_output` inside the ``4c_from_tdi_output_splines`` stage timer."""
        with _stage("4c_from_tdi_output_splines", self.xp):
            return self.from_tdi_output(tdi_output, fill_splines=fill_splines)

    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Convert the raw kernel output; the base returns it unchanged (subclasses override)."""
        return tdi_output


# Knot-spacing codes of the C++ cubic splines (``FDTDIonTheFly.spline_type``).
CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2
CUBIC_SPLINE_GENERAL_SPACING = 3

class TDTDIonTheFly(TDIonTheFly):
    """TDI on the fly from time-domain amplitude and phase splines.

    The source is given as amplitude and phase samples on the input times
    ``t_input``; they are splined with
    :class:`~gpubackendtools.interpolate.CubicSplineInterpolant` on the
    generator's backend, and the backend ``TDSplineTDIWaveformWrap`` kernel
    evaluates the TDI channels at the evaluation times ``t``. Calls take the
    four extrinsic angles (:meth:`TDIonTheFly.__call__`) and return a
    :class:`TDTDIOutput`. Used by the BBHx MBH / SOBBH TDI-on-the-fly
    generators (``mbhtdionfly``, ``sobbhtdionfly``) and
    :class:`~lisatools.sources.waveformbase.TDTDIOnFlyWaveformBase`.

    Args:
        t: Evaluation times [s], ``(N,)`` (shared, repeated over
            ``num_sub``) or ``(num_sub, N)``; numpy or cupy matching ``amp``.
        amp: Amplitude samples, ``(num_sub, n_in)`` (or 1-D for one source),
            numpy or cupy. A scipy ``CubicSpline`` or a ``CubicSplineInterpolant``
            raises ``NotImplementedError``.
        phase: Phase samples [rad], same layout as ``amp``.
        *args: Forwarded to :class:`TDIonTheFly` (``sampling_frequency``,
            ``num_sub``, ...).
        t_input: Times of the ``amp`` / ``phase`` samples, ``(n_in,)`` or
            ``(num_sub, n_in)``; required for array input.
        **kwargs: Forwarded to :class:`TDIonTheFly` (``tdi_config``,
            ``orbits``, ``force_backend``, ...).

    Raises:
        AssertionError: Mixed numpy / cupy inputs, or a missing ``t_input``.
        NotImplementedError: Spline-object inputs.
        ValueError: Any other input type.

    Example:
        .. code-block:: python

            from lisatools.response.tdionfly import TDTDIonTheFly

            gen = TDTDIonTheFly(
                t_eval, amp, phase, sampling_frequency=0.2, num_sub=1,
                t_input=t_samples, tdi_config="2nd generation", orbits=orbits,
                force_backend="cpu",
            )
            out = gen(inc, psi, lam, beta)  # TDTDIOutput; out.X, out.Y, out.Z
    """
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSpline,
        phase: np.ndarray | CubicSpline_scipy | CubicSpline,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        **kwargs
    ): 
        super().__init__(*args, **kwargs)

        self.phase_input = phase
        self.amp_input = amp

        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(phase, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(phase, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(phase)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            phase = self.xp.atleast_2d(self.xp.asarray(phase))

            # TODO: improve when gbt is fixed up
            with _stage("3c_ampphase_spline_build", self.xp):
                amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
                phase = CubicSplineInterpolant(t_input.copy(), phase, force_backend=self.backend.name.split("_")[-1])
            
        elif isinstance(amp, CubicSpline_scipy):
            raise NotImplementedError
            assert isinstance(phase, CubicSpline_scipy)

            self.spline_length = phase.c.shape[-1] + 1

            phase_y = phase.c[3, :].copy()
            phase_c1 = phase.c[2, :].copy()
            phase_c2 = phase.c[1, :].copy()
            phase_c3 = phase.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            # convert to pointers
            targs, twkargs = wrapper(t, phase_y, phase_c1, phase_c2, phase_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            phase = self.backend.pyCubicSplineWrap(_t, _phase_y, _phase_c1, _phase_c2, _phase_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(phase, CubicSplineInterpolant)
            raise NotImplementedError(
                "CubicSplineInterpolant amp/phase input is not yet wired into "
                "TDIonTheFly; pass scipy.CubicSpline objects (handled above) "
                "or pre-built pyCubicSplineWrap instances instead."
            )

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))

        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.phase = phase

        # self.wave_gen = self.backend.pyTDSplineTDIWaveform()
        # self.wave_gen.add_orbit_information(*self.orbits.pycppdetector_args)
        # self.wave_gen.add_tdi_config(*self.tdi_config.pytdiconfig_args)
        # self.wave_gen.add_amp_spline(*self.amp.cpp_class_args)
        # self.wave_gen.add_phase_spline(*self.phase.cpp_class_args)
        
        # import time
        # time.sleep(1.0)
    @property
    def wave_gen(self) -> callable:
        """A new backend ``TDSplineTDIWaveformWrap`` over the amp / phase splines (per access)."""
        with _stage("4a_wave_gen_wrap_build", self.xp):
            self.cpp_amp = self.backend.CubicSplineWrap(*self.amp.cpp_class_args)
            self.cpp_phase = self.backend.CubicSplineWrap(*self.phase.cpp_class_args)
            self._wave_gen = self.backend.TDSplineTDIWaveformWrap(self.cpp_orbits, self.cpp_tdi_config, self.cpp_amp, self.cpp_phase)
        return self._wave_gen
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Wrap the raw output as a :class:`TDTDIOutput` on the evaluation times.

        Args:
            tdi_output (TDIOutput): Raw kernel output (its ``x`` must equal ``t_arr``).
            fill_splines: Build the output's splines.

        Returns:
            TDTDIOutput: The time-domain output on the output's backend.
        """
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref,
            fill_splines=fill_splines, force_backend=tdi_output.backend.name.split("_")[-1]
        )
    

class TDIOutput(LISAToolsParallelModule):
    """Amplitude / phase representation of TDI-on-the-fly channels.

    Channel ``c`` of source ``b`` is
    ``tdi_amp[b, c] * exp(-i (tdi_phase[b, c] + phase_ref[b]))`` on the grid
    ``x`` (times for :class:`TDTDIOutput`, frequencies for
    :class:`FDTDIOutput`); :attr:`X`, :attr:`Y` and :attr:`Z` return it for
    channels 0, 1, 2. With ``fill_splines``, setting ``tdi_amp`` /
    ``tdi_phase`` / ``phase_ref`` also builds a
    :class:`~gpubackendtools.interpolate.CubicSplineInterpolant` over ``x``
    for each, used by :meth:`eval_spline_vals`.

    Args:
        x (xp.ndarray): Grid, ``(num_sub, N)``.
        tdi_amp (xp.ndarray): ``(num_sub, nchannels, N)`` amplitudes.
        tdi_phase (xp.ndarray): ``(num_sub, nchannels, N)`` channel phases [rad].
        phase_ref (xp.ndarray): ``(num_sub, N)`` reference (carrier) phase [rad].
        fill_splines (bool, optional): Build the splines. Default ``True``.
        **kwargs: Forwarded to
            :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`
            (``force_backend``).
    """
    def __init__(self, x, tdi_amp, tdi_phase, phase_ref, fill_splines=True, **kwargs):
        
        self.x = x
        self.fill_splines = fill_splines
        if self.fill_splines:
            self._splines = {}
        
        super().__init__(**kwargs)

        # need to be after for proper setter
        self.tdi_amp, self.tdi_phase = tdi_amp, tdi_phase
        self.phase_ref = phase_ref
       
        
    def _get_spl(self, key: str) -> CubicSpline:
        """Return the spline stored under ``key`` (requires ``fill_splines``)."""
        assert self.fill_splines
        return self._splines[key]
    
    @property
    def phase_ref_spl(self) -> CubicSpline:
        """Spline of ``phase_ref`` over ``x``."""
        return self._get_spl("phase_ref")
    
    def build_spline(self, x, y, **kwargs) -> CubicSplineInterpolant:
        """Build a cubic spline of ``y`` over ``x`` on this object's backend.

        A 2-D ``x`` with a 3-D ``y`` is repeated over ``y``'s channel axis.

        Args:
            x (xp.ndarray): ``(num_sub, N)`` grid.
            y (xp.ndarray): ``(num_sub, N)`` or ``(num_sub, nchannels, N)`` values.
            **kwargs: Forwarded to ``CubicSplineInterpolant``.
        """
        if x.ndim == 2 and y.ndim == 3:
            x_in =  self.xp.repeat(x[:, None, :], y.shape[1], axis=1)
        else:
            x_in = x.copy()

        return CubicSplineInterpolant(x_in, y, force_backend=self.backend.name.split("_")[-1], **kwargs)
    
    @property
    def num_bin(self) -> int:
        """Number of sources (``tdi_amp.shape[0]`` for 3-D, 1 for 2-D ``tdi_amp``)."""
        if self.tdi_amp.ndim == 3:
            return self.tdi_amp.shape[0]
        elif self.tdi_amp.ndim == 2:
            return 1

    @classmethod
    def supported_backends(cls) -> list:
        """Return ``lisatools_<tag>`` for ``cuda13x``/``cuda12x``/``cuda11x``/``cpu``/``jax``."""
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED_WITH_JAX()]

    @property
    def X(self) -> np.ndarray:
        """Complex X channel, ``Xamp exp(-i (Xphase + phase_ref))``, ``(num_sub, N)``."""
        return self.Xamp * self.xp.exp(-1j * (self.Xphase + self.phase_ref))
    @property
    def Y(self) -> np.ndarray:
        """Complex Y channel (as :attr:`X` for channel 1)."""
        return self.Yamp * self.xp.exp(-1j * (self.Yphase + self.phase_ref))
    @property
    def Z(self) -> np.ndarray:
        """Complex Z channel (as :attr:`X` for channel 2)."""
        return self.Zamp * self.xp.exp(-1j * (self.Zphase + self.phase_ref))
    @property
    def Xamp(self) -> np.ndarray:
        """``tdi_amp[:, 0]``."""
        return self.tdi_amp[:, 0]
    @property
    def Yamp(self) -> np.ndarray:
        """``tdi_amp[:, 1]``."""
        return self.tdi_amp[:, 1]
    @property
    def Zamp(self) -> np.ndarray:
        """``tdi_amp[:, 2]``."""
        return self.tdi_amp[:, 2]
    @property
    def Xphase(self) -> np.ndarray:
        """``tdi_phase[:, 0]``."""
        return self.tdi_phase[:, 0]
    @property
    def Yphase(self) -> np.ndarray:
        """``tdi_phase[:, 1]``."""
        return self.tdi_phase[:, 1]
    @property
    def Zphase(self) -> np.ndarray:
        """``tdi_phase[:, 2]``."""
        return self.tdi_phase[:, 2]
    
    @property
    def tdi_amp(self) -> np.ndarray:
        """``(num_sub, nchannels, N)`` channel amplitudes (setting rebuilds the spline)."""
        return self._tdi_amp
    
    @tdi_amp.setter
    def tdi_amp(self, tdi_amp: np.ndarray):
        if self.fill_splines:
            self._splines["tdi_amp"] = self.build_spline(self.x, tdi_amp)

        self._tdi_amp = tdi_amp

    @property
    def tdi_phase(self) -> np.ndarray:
        """``(num_sub, nchannels, N)`` channel phases [rad] (setting rebuilds the spline)."""
        return self._tdi_phase
    
    @tdi_phase.setter
    def tdi_phase(self, tdi_phase: np.ndarray):
        if self.fill_splines:
            self._splines["tdi_phase"] = self.build_spline(self.x, tdi_phase)

        self._tdi_phase = tdi_phase

    @property
    def phase_ref(self) -> np.ndarray:
        """``(num_sub, N)`` reference (carrier) phase [rad] (setting rebuilds the spline)."""
        return self._phase_ref
    
    @phase_ref.setter
    def phase_ref(self, phase_ref: np.ndarray):
        if self.fill_splines:
            self._splines["phase_ref"] = self.build_spline(self.x, phase_ref)

        self._phase_ref = phase_ref
    
    @property
    def tdi_amp_spl(self):
        """Spline of ``tdi_amp`` over ``x``."""
        return self._get_spl("tdi_amp")
    
    @property
    def tdi_phase_spl(self):
        """Spline of ``tdi_phase`` over ``x``."""
        return self._get_spl("tdi_phase")
    
    @property
    def Aamp(self):
        """Not implemented (A/E/T channels are not provided); raises ``NotImplementedError``."""
        raise NotImplementedError
    @property
    def Aphase(self):
        """Not implemented; raises ``NotImplementedError``."""
        raise NotImplementedError
    @property
    def Eamp(self):
        """Not implemented; raises ``NotImplementedError``."""
        raise NotImplementedError
    @property
    def Ephase(self):
        """Not implemented; raises ``NotImplementedError``."""
        raise NotImplementedError
    @property
    def Tamp(self):
        """Not implemented; raises ``NotImplementedError``."""
        raise NotImplementedError
    @property
    def Tphase(self):
        """Not implemented; raises ``NotImplementedError``."""
        raise NotImplementedError
    
    def eval_spline_vals(self, x_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the amplitude, phase and reference-phase splines at ``x_new``.

        Requires ``fill_splines``.

        Args:
            x_new: New grid, ``(M,)`` (shared by every source) or ``(num_sub, M)``.
            **kwargs: Forwarded to the spline calls.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(tdi_amp, tdi_phase,
            phase_ref)`` at ``x_new``: ``(num_sub, 3, M)``, ``(num_sub, 3, M)``
            and ``(num_sub, M)``.
        """

        if x_new.ndim == 1:
            t_amp_phase = self.xp.tile(x_new, (self.num_bin, 3, 1))
            t_phase_ref = self.xp.tile(x_new, (self.num_bin, 1))
        elif x_new.ndim == 2:
            t_amp_phase = self.xp.repeat(x_new[:, None, :], 3, axis=1)
            t_phase_ref = x_new

        tdi_amp_new = self.tdi_amp_spl(t_amp_phase, **kwargs)
        tdi_phase_new = self.tdi_phase_spl(t_amp_phase, **kwargs)
        phase_ref_new = self.phase_ref_spl(t_phase_ref, **kwargs)
        return (tdi_amp_new, tdi_phase_new, phase_ref_new)
    
    def eval_tdi(self, x_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the channels at ``x_new`` (implemented by the TD / FD subclasses)."""
        raise NotImplementedError
    
class TDTDIOutput(TDIOutput):
    """Time-domain :class:`TDIOutput`: ``x`` is the time grid [s].

    The physical channel is the real part of the complex representation;
    :meth:`eval_tdi` returns that real part on a new time grid, while
    :attr:`X` / :attr:`Y` / :attr:`Z` stay complex.
    """

    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> TDTDIOutput:
        """Build a :class:`TDTDIOutput` from any :class:`TDIOutput` (same arrays and backend)."""
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines, force_backend=tdi_output.backend.name.split("_")[-1]
        )

    def eval_tdi(self, t_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the real TDI channels at new times through the splines.

        Args:
            t_new: Times [s], ``(M,)`` or ``(num_sub, M)``.
            **kwargs: Forwarded to the spline calls.

        Returns:
            xp.ndarray: ``Re[amp exp(-i (phase + phase_ref))]``, ``(num_sub, 3, M)``.
        """
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(t_new, **kwargs)
        tdi_output = self.xp.real(tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new[:, None, :])))
        return tdi_output
    
    @property
    def t_arr(self) -> np.ndarray:
        """The time grid (alias of ``x``)."""
        return self.x
        
    
class FDTDIOutput(TDIOutput):
    """Frequency-domain :class:`TDIOutput`: ``x`` is the frequency grid [Hz], channels complex."""

    def eval_tdi(self, f_new: np.ndarray, **kwargs) -> np.ndarray:
        """Evaluate the complex TDI channels at new frequencies through the splines.

        Args:
            f_new: Frequencies [Hz], ``(M,)`` or ``(num_sub, M)``.
            **kwargs: Forwarded to the spline calls.

        Returns:
            xp.ndarray: ``amp exp(-i (phase + phase_ref))``, ``(num_sub, 3, M)``.
        """
        tdi_amp_new, tdi_phase_new, phase_ref_new = self.eval_spline_vals(f_new, **kwargs)
        tdi_output = tdi_amp_new * self.xp.exp(-1j * (tdi_phase_new + phase_ref_new[:, None, :]))
        return tdi_output
    
    @property
    def f_arr(self) -> np.ndarray:
        """The frequency grid (alias of ``x``)."""
        return self.x


# TODO: make it log spaced in frequency?

class FDTDIonTheFly(TDIonTheFly):
    """TDI on the fly for a frequency-domain source given by amplitude and frequency tracks.

    The source's amplitude and instantaneous frequency are given versus time
    (``t_input``) and splined; the backend ``FDSplineTDIWaveformWrap`` kernel
    evaluates the TDI channels at the times ``t``, and :meth:`from_tdi_output`
    maps those times to frequencies through the frequency spline, returning
    an :class:`FDTDIOutput`. Calls take the four extrinsic angles
    (:meth:`TDIonTheFly.__call__`). Not used by the global-fit runs.

    Args:
        t: Evaluation times [s], ``(N,)`` or ``(num_sub, N)``.
        amp: Amplitude samples ``(num_sub, n_in)`` (numpy or cupy, splined
            here) or a prebuilt
            :class:`~gpubackendtools.interpolate.CubicSplineInterpolant`; a
            scipy ``CubicSpline`` raises ``NotImplementedError``.
        freq: Frequency samples [Hz], same layout as ``amp``.
        phase_ref: Stored as ``phase_ref`` only (not passed to the kernel).
        *args: Forwarded to :class:`TDIonTheFly` (``sampling_frequency``,
            ``num_sub``, ...).
        t_input: Times of the samples, required for array input.
        spline_type: Accepted but not applied (``spline_type`` keeps no
            value unless set afterwards). Default
            ``CUBIC_SPLINE_GENERAL_SPACING``.
        force_backend: Backend name; default ``None``.
        **kwargs: Forwarded to :class:`TDIonTheFly`.

    Raises:
        AssertionError: Mixed numpy / cupy inputs or a missing ``t_input``.
        NotImplementedError: scipy ``CubicSpline`` inputs.
        ValueError: Any other input type.
    """
    def __init__(self, 
        t: np.ndarray,
        amp: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        freq: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        phase_ref: np.ndarray | CubicSpline_scipy | CubicSplineInterpolant,
        *args, 
        t_input: Optional[np.ndarray] = None, 
        spline_type: int = CUBIC_SPLINE_GENERAL_SPACING,
        force_backend: str = None,
        **kwargs
    ): 
        super().__init__(*args, force_backend=force_backend, **kwargs)

        self.freq_input = freq
        self.amp_input = amp
        self.phase_ref = phase_ref
        
        if isinstance(amp, np.ndarray) or isinstance(amp, cp.ndarray):
            if isinstance(amp, np.ndarray):
                assert isinstance(freq, np.ndarray) and isinstance(t, np.ndarray)
                assert t_input is not None and isinstance(t_input, np.ndarray)
            else:
                assert isinstance(freq, cp.ndarray) and isinstance(t, cp.ndarray)
                assert t_input is not None and isinstance(t_input, cp.ndarray)
            
            self.spline_length = len(freq)

            t_input = self.xp.atleast_2d(self.xp.asarray(t_input))
            
            if t_input.shape[0] == 1:
                t_input = self.xp.repeat(t_input, amp.shape[0], axis=0)

            amp = self.xp.atleast_2d(self.xp.asarray(amp))
            freq = self.xp.atleast_2d(self.xp.asarray(freq))

            # TODO: improve when gbt is fixed up
            amp = CubicSplineInterpolant(t_input.copy(), amp, force_backend=self.backend.name.split("_")[-1])
            freq = CubicSplineInterpolant(t_input.copy(), freq, force_backend=self.backend.name.split("_")[-1])
            

        elif isinstance(amp, CubicSpline_scipy):
            assert isinstance(freq, CubicSpline_scipy)
            raise NotImplementedError
            self.spline_length = freq.c.shape[-1] + 1

            freq_y = freq.c[3, :].copy()
            freq_c1 = freq.c[2, :].copy()
            freq_c2 = freq.c[1, :].copy()
            freq_c3 = freq.c[0, :].copy()

            amp_y = amp.c[3, :].copy()
            amp_c1 = amp.c[2, :].copy()
            amp_c2 = amp.c[1, :].copy()
            amp_c3 = amp.c[0, :].copy()

            # convert to pointers
            targs, twkargs = wrapper(t, freq_y, freq_c1, freq_c2, freq_c3, amp_y, amp_c1, amp_c2, amp_c3)
            (_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, _amp_y, _amp_c1, _amp_c2, _amp_c3) = targs
            freq = self.backend.pyCubicSplineWrap(_t, _freq_y, _freq_c1, _freq_c2, _freq_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)
            amp = self.backend.pyCubicSplineWrap(_t, _amp_y, _amp_c1, _amp_c2, _amp_c3, self.num_sub, self.n_params, self.spline_length, CUBIC_SPLINE_LINEAR_SPACING)

        elif isinstance(amp, CubicSplineInterpolant):
            assert isinstance(freq, CubicSplineInterpolant)
            # f = freq.y, t = freq.x

        else:
            raise ValueError("# TODO: fix this.")
        
        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
        self.amp = amp
        self.freq = freq

    @property
    def wave_gen(self) -> callable:
        """A new backend ``FDSplineTDIWaveformWrap`` over the amp / freq splines (per access)."""
        self.cpp_amp = self.backend.CubicSplineWrap(*self.amp.cpp_class_args)
        self.cpp_freq = self.backend.CubicSplineWrap(*self.freq.cpp_class_args)
        self._wave_gen = self.backend.FDSplineTDIWaveformWrap(self.cpp_orbits, self.cpp_tdi_config, self.cpp_amp, self.cpp_freq)
        return self._wave_gen
    
    @property
    def spline_type(self) -> int:
        """int: One of the ``CUBIC_SPLINE_*_SPACING`` codes (validated on set)."""
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int):
        assert isinstance(spline_type, int)
        assert spline_type in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]
        self._spline_type = spline_type
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Wrap the raw output as an :class:`FDTDIOutput` on ``freq(t)``, the frequency grid."""
        # TODO: remove the freq spline?
        return FDTDIOutput(
            self.freq(tdi_output.x), tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines, force_backend=tdi_output.backend.name.split("_")[-1]
        )
    

class GBFDTDIonTheFly(LISAToolsParallelModule):
    # Phase 3L.7k: needs gbgpu_<flavor> backend (carries GBTDIonTheFlyWrap).
    _BACKEND_PREFIX = "gbgpu"
    """Heterodyned frequency-domain GB TDI on the fly.

    Generates the heterodyne-shifted GB TDI directly in the frequency domain
    using a power-of-two sparse time grid. The kernel keeps all (nchannels)
    channels of the slow positive-frequency complex signal in shared memory
    at the same time, FFTs them in place, and returns
    ``X_het[bin, channel, m]`` together with ``k_f0[bin]`` and
    ``f0_grid[bin]`` mapping each FFT bin back to the dense rfft grid:
    bin ``m`` (FFT order) lands on dense rfft bin ``k_f0 + m``. Needs a
    ``gbgpu`` backend (``GBTDIonTheFlyWrap``); the orbits and TDI handles
    come from an internal :class:`GBTDIonTheFly`. Not used by the global-fit
    runs.

    Args:
        T: Observation duration in seconds. The dense rfft grid is
            implicitly ``df = 1/T``.
        t_ref: GB phase reference time. Must equal ``t_start`` (checked in
            :meth:`__call__`) so the heterodyne phase factor at the time
            origin is unity.
        N_sparse: Length of the sparse time grid. Must be a power of two.
            The sparse Nyquist is ``N_sparse / (2 * T)``.
        num_sub: Number of sources per call.
        nchannels: 1, 2 or 3 -- usually 3 for XYZ. Default ``3``.
        tdi_config: As in :class:`TDIonTheFly`. Default ``None``.
        orbits: As in :class:`TDIonTheFly`. Default the
            :class:`~lisatools.detector.EqualArmlengthOrbits` class.
        tdi_chan: As in :class:`TDIonTheFly`. Default ``"XYZ"``.
        force_backend: ``gbgpu`` backend flavor (``"cpu"``, ``"cuda12x"``,
            ...). Default ``None``.

    Raises:
        ValueError: ``N_sparse`` is not a power of two.
    """

    def __init__(
        self,
        T: float,
        t_ref: float,
        N_sparse: int,
        num_sub: int,
        nchannels: int = 3,
        tdi_config: Optional[TDIConfig] = None,
        orbits: Optional[Orbits] = EqualArmlengthOrbits,
        tdi_chan: str = "XYZ",
        force_backend: Optional[str] = None,
    ):
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError("N_sparse must be a power of two.")
        self.T = float(T)
        self.t_ref = float(t_ref)
        self.N_sparse = int(N_sparse)
        self.num_sub = int(num_sub)
        self.nchannels = int(nchannels)
        self.n_params = 9  # amp, f0, fdot, fddot, phi0, inc, psi, lam, beta

        self.tdi_chan = tdi_chan
        super().__init__(force_backend=force_backend)

        # Reuse TDIonTheFly's orbits / tdi_config setters via a private helper:
        # build a sibling GBTDIonTheFly and steal its cpp handles.
        self._td = GBTDIonTheFly(
            self.xp.linspace(t_ref, t_ref + T, 2),  # dummy 2 points
            T, t_ref, 1.0, num_sub,
            tdi_config=tdi_config, orbits=orbits,
            tdi_chan=tdi_chan, force_backend=force_backend,
        )
        # gb_wrap exposes run_fd_wave_tdi_wrap
        self.gb_wrap = self.backend.GBTDIonTheFlyWrap(
            self._td.cpp_orbits, self._td.cpp_tdi_config, self.T, self.t_ref
        )

    @property
    def xp(self):
        """The backend's array module."""
        return self.backend.xp

    @classmethod
    def supported_backends(cls):
        """Return ``gbgpu_<tag>`` for the GPU-recommended tags (no ``jax``)."""
        return [cls._BACKEND_PREFIX + "_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]

    def __call__(
        self,
        amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta,
        t_start: float,
        convert_to_ra_dec: Optional[bool] = None,
    ):
        """Generate the heterodyne FD GB TDI for ``num_sub`` sources.

        The nine source parameters are arrays of length ``num_sub``, in the
        kernel's order ``(amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta)``.

        Args:
            amp (array-like): Strain amplitudes.
            f0 (array-like): Initial GW frequencies [Hz].
            fdot0 (array-like): Frequency derivatives [Hz/s].
            fddot0 (array-like): Second frequency derivatives [Hz/s^2].
            phi0 (array-like): Initial phases [rad].
            inc (array-like): Inclinations [rad].
            psi (array-like): Polarization angles [rad].
            lam (array-like): Sky longitudes [rad] in the orbits' frame.
            beta (array-like): Sky latitudes [rad] in the orbits' frame.
            t_start: Start of the sparse observation window in seconds. Must
                equal ``t_ref`` -- the prototype assumes a local time origin at
                the first sparse sample.
            convert_to_ra_dec: **Deprecated.** Sky coordinates are consumed in
                the orbits frame directly (default ``None`` = no conversion).
                ``True`` still applies the legacy ecliptic -> ICRS conversion
                but emits a ``DeprecationWarning``.

        Returns:
            tuple[xp.ndarray, xp.ndarray, xp.ndarray]: ``(X_het, k_f0, f0_grid)``:
            complex ``(num_sub, nchannels, N_sparse)`` values
            ``0.5 * dt_sparse * FFT[s_c]`` in FFT order; int32 ``(num_sub,)``
            dense rfft bin of the carrier; float ``(num_sub,)`` snapped
            carrier ``k_f0 * df`` [Hz].

        Raises:
            ValueError: ``t_start != t_ref``, or the parameter count does not
                match ``num_sub``.
        """
        if abs(float(t_start) - self.t_ref) > 1e-9:
            raise ValueError(
                "GBFDTDIonTheFly assumes t_start == t_ref so the heterodyne "
                "phase factor at the time origin is unity. "
                f"Got t_start={t_start}, t_ref={self.t_ref}."
            )
        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam, beta = ecliptic_to_icrs(lam, beta)

        params = self.xp.asarray(
            [amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]
        ).T.flatten().copy()
        if len(params) != self.n_params * self.num_sub:
            raise ValueError("params length does not match num_sub * n_params.")

        X_het = self.xp.zeros(
            self.num_sub * self.nchannels * self.N_sparse, dtype=complex
        )
        k_f0_out    = self.xp.zeros(self.num_sub, dtype=self.xp.int32)
        f0_grid_out = self.xp.zeros(self.num_sub, dtype=float)

        self.gb_wrap.run_fd_wave_tdi_wrap(
            X_het, k_f0_out, f0_grid_out, params,
            float(t_start), self.T,
            self.N_sparse, self.num_sub, self.n_params, self.nchannels,
        )

        return (
            X_het.reshape(self.num_sub, self.nchannels, self.N_sparse),
            k_f0_out, f0_grid_out,
        )


class GBTDIonTheFly(TDIonTheFly):
    """Galactic-binary TDI on the fly (``gbgpu`` backend kernel ``GBTDIonTheFlyWrap``).

    Evaluates the GB amplitude and phase analytically (9 parameters per
    source, see :meth:`__call__`) and projects them through the LISA TDI on
    the time grid ``t``, returning a :class:`TDTDIOutput`. In the global fit
    GBGPU's lookup and signal-heterodyne engines
    (``gbgpu.gblookupcomputations``, ``gbgpu.gbsignalhetcomputations``)
    construct one on a 16384-point grid and keep only its :attr:`wave_gen`
    handle for their own kernels. Supports the ``jax`` backend (functional
    ``run_wave_tdi``).

    Args:
        t: Evaluation times [s], ``(N,)`` or ``(num_sub, N)``.
        T: Observation duration [s].
        t_ref: GB phase reference time [s].
        *args: Forwarded to :class:`TDIonTheFly` (``sampling_frequency``,
            ``num_sub``); ``n_params`` is fixed to 9.
        **kwargs: Forwarded to :class:`TDIonTheFly` (``tdi_config``,
            ``orbits``, ``tdi_chan``, ``force_backend``: a ``gbgpu`` flavor).

    Example:
        .. code-block:: python

            import numpy as np
            from lisatools.response.tdionfly import GBTDIonTheFly

            t = np.linspace(t0, t0 + Tobs, 16384)
            gen = GBTDIonTheFly(t, Tobs, t_ref, 1.0 / dt, 1, tdi_config=tdi_config,
                                orbits=orbits, tdi_chan="XYZ", force_backend="cpu")
            out = gen(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)  # TDTDIOutput
    """

    # Phase 3L.7k: needs gbgpu_<flavor> backend (carries GBTDIonTheFlyWrap).
    _BACKEND_PREFIX = "gbgpu"
    def __init__(self,
        t: np.ndarray,
        T: float,
        t_ref: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, n_params=9, **kwargs)

        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        self.T = T
        self.t_ref = t_ref
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]
        
    # NOTE: this property / setter pair is shadowed by the second ``wave_gen``
    # property below (the later class-body definition wins), so assigning
    # ``wave_gen`` on a GBTDIonTheFly raises AttributeError.
    @property
    def wave_gen(self) -> callable:
        """Shadowed (see the note above); the effective property is defined below."""
        return self._wave_gen
    
    @wave_gen.setter
    def wave_gen(self, wave_gen):
        self._wave_gen = wave_gen
    
    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Wrap the raw output as a :class:`TDTDIOutput` (its ``x`` must equal ``t_arr``)."""
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines, force_backend=tdi_output.backend.name.split("_")[-1]
        )
    
    @property
    def wave_gen(self) -> callable:
        """A new backend ``GBTDIonTheFlyWrap`` over the orbits, TDI config, ``T`` and ``t_ref``.

        Built on every access. GBGPU's lookup / signal-heterodyne engines keep
        this handle for their kernels.
        """
        self._wave_gen = self.backend.GBTDIonTheFlyWrap(self.cpp_orbits, self.cpp_tdi_config, self.T, self.t_ref)
        return self._wave_gen
    
    def __call__(self, amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta, convert_to_ra_dec: Optional[bool] = None, return_spline: bool = False) -> TDIOutput:
        """Generate the GB TDI for ``num_sub`` sources (arrays of length ``num_sub``).

        On the ``jax`` backend the functional ``run_wave_tdi`` is used (no
        in-place buffers, differentiable); the C++ backends fill buffers in
        place with ``run_wave_tdi_wrap``.

        Args:
            amp (array-like): Strain amplitude.
            f0 (array-like): Initial GW frequency [Hz].
            fdot0 (array-like): Frequency derivative [Hz/s].
            fddot0 (array-like): Second frequency derivative [Hz/s^2].
            phi0 (array-like): Initial phase [rad].
            inc (array-like): Inclination [rad].
            psi (array-like): Polarization angle [rad].
            lam (array-like): Sky longitude [rad] in the orbits' frame (RA for
                ICRS orbits).
            beta (array-like): Sky latitude [rad] in the orbits' frame (Dec for
                ICRS orbits).
            convert_to_ra_dec: **Deprecated**; ``True`` converts ecliptic
                ``(lam, beta)`` to ICRS first, with a ``DeprecationWarning``.
            return_spline: Build the output's splines (unsupported on ``jax``).

        Returns:
            TDTDIOutput: ``tdi_amp`` / ``tdi_phase`` ``(num_sub, nchannels, N)``,
            ``phase_ref`` ``(num_sub, N)``.
        """

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam, beta = ecliptic_to_icrs(lam, beta)
        params = self.xp.asarray([amp, f0, fdot0, fddot0, phi0, inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 9 * self.num_sub
        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)

        # The JAX backend exposes a pure-functional wave_gen.run_wave_tdi
        # that returns (M, tdi_amp, tdi_phase, phase_ref) directly --
        # immutable arrays, no buffer mutation -- so jax.grad / jax.jit
        # work over the full pipeline. The C++ backends keep the
        # in-place buffer signature run_wave_tdi_wrap(buffer, ...).
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            params_2d = params.reshape(self.num_sub, self.n_params)
            t_2d = self.t_arr.reshape(self.num_sub, self.N)
            _, tdi_amp_arr, tdi_phase_arr, phase_ref_arr = self.wave_gen.run_wave_tdi(
                params_2d, t_2d,
            )
            # fill_splines=False on the inner TDIOutput: gpubackendtools
            # doesn't ship a 'jax' cubic-spline backend, so building one
            # on JAX arrays would crash. Splines are a CPU-only post-step
            # for now; users wanting them should pass force_backend='cpu'
            # or pull the JAX arrays out and feed them to a CPU
            # CubicSplineInterpolant manually.
            return self.from_tdi_output(TDIOutput(
                self.t_arr,
                tdi_amp_arr,                  # (num_sub, nchannels, N)
                tdi_phase_arr,                # (num_sub, nchannels, N)
                phase_ref_arr,                # (num_sub, N)
                fill_splines=False,
                force_backend=self.backend,
            ), fill_splines=False if return_spline is False else return_spline)

        # C++ in-place path (unchanged).
        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        self.wave_gen.run_wave_tdi_wrap(
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            params, self.t_arr.flatten().copy(),
            self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
        )

        return self.from_tdi_output(TDIOutput(
            self.t_arr,
            tdi_amp.reshape(reshape_shape),
            tdi_phase.reshape(reshape_shape),
            phase_ref.reshape(self.t_arr.shape),
            force_backend=self.backend
        ), fill_splines=return_spline)


class SOBBHTDIonTheFly(TDIonTheFly):
    """Stellar-origin black-hole binary TDI on the fly (``bbhx`` backend).

    Mirrors :class:`GBTDIonTheFly`. The underlying C++ kernel
    (``SOBBHTDIonTheFlyWrap``) evaluates the SOBBH amplitude and phase
    point-wise from the PN intrinsic-quantity expressions and feeds them
    through the shared LISA TDI projection; see :meth:`__call__` for the 11
    parameters. Supports the ``jax`` backend. Not used by the global-fit
    runs (BBHx's SOBBH TDI-on-the-fly generator ``bbhx.sobbhtdionfly`` uses
    :class:`TDTDIonTheFly`).

    Args:
        t: Evaluation times [s], ``(N,)`` or ``(num_sub, N)``.
        T: Observation duration [s].
        t_ref: Reference time of ``f_low`` / ``phi_c`` [s].
        *args: Forwarded to :class:`TDIonTheFly` (``sampling_frequency``,
            ``num_sub``); ``n_params`` is fixed to 11.
        **kwargs: Forwarded to :class:`TDIonTheFly` (``tdi_config``,
            ``orbits``, ``force_backend``: a ``bbhx`` flavor).
    """

    # Phase 3L.7k: needs bbhx_<flavor> backend (carries SOBBHTDIonTheFlyWrap).
    _BACKEND_PREFIX = "bbhx"
    """str: Backend namespace; the ``bbhx_<flavor>`` backends carry ``SOBBHTDIonTheFlyWrap``."""

    def __init__(self,
        t: np.ndarray,
        T: float,
        t_ref: float,
        *args,
        **kwargs
    ):
        super().__init__(*args, n_params=11, **kwargs)

        self.t_arr = self.xp.atleast_2d(self.xp.asarray(t))
        self.T = T
        self.t_ref = t_ref
        self.N = self.t_arr.shape[1]

        if self.t_arr.shape[0] == 1:
            self.t_arr = self.xp.repeat(self.t_arr, self.num_sub, axis=0)

        self.dt = self.t_arr[:, 1] - self.t_arr[:, 0]

    @property
    def wave_gen(self) -> callable:
        """A new backend ``SOBBHTDIonTheFlyWrap`` (built per access)."""
        self._wave_gen = self.backend.SOBBHTDIonTheFlyWrap(self.cpp_orbits, self.cpp_tdi_config, self.T, self.t_ref)
        return self._wave_gen

    def from_tdi_output(self, tdi_output: TDIOutput, fill_splines: Optional[bool] = False) -> FDTDIOutput:
        """Wrap the raw output as a :class:`TDTDIOutput` (its ``x`` must equal ``t_arr``)."""
        assert self.xp.allclose(tdi_output.x, self.t_arr)
        return TDTDIOutput(
            tdi_output.x, tdi_output.tdi_amp, tdi_output.tdi_phase, tdi_output.phase_ref, fill_splines=fill_splines, force_backend=tdi_output.backend.name.split("_")[-1]
        )

    def __call__(self, m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta,
                 convert_to_ra_dec: Optional[bool] = None, return_spline: bool = False) -> TDIOutput:
        """Generate the SOBBH TDI for ``num_sub`` sources (arrays of length ``num_sub``).

        Args:
            m1 (array-like): Primary mass [solar masses].
            m2 (array-like): Secondary mass [solar masses].
            s1 (array-like): Primary aligned spin [dimensionless].
            s2 (array-like): Secondary aligned spin [dimensionless].
            distance (array-like): Luminosity distance [parsecs].
            f_low (array-like): GW frequency at ``t_ref`` [Hz].
            phi_c (array-like): Reference orbital phase [rad].
            inc (array-like): Inclination [rad].
            psi (array-like): Polarization [rad].
            lam (array-like): Sky longitude [rad] in the orbits' frame (RA for
                ICRS orbits).
            beta (array-like): Sky latitude [rad] in the orbits' frame (Dec for
                ICRS orbits).
            convert_to_ra_dec: **Deprecated**; ``True`` converts ecliptic
                ``(lam, beta)`` to ICRS first, with a ``DeprecationWarning``.
            return_spline: Build the output's splines (not on ``jax``).

        Returns:
            TDTDIOutput: ``tdi_amp`` / ``tdi_phase`` ``(num_sub, nchannels, N)``,
            ``phase_ref`` ``(num_sub, N)``.
        """

        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            lam, beta = ecliptic_to_icrs(lam, beta)
        params = self.xp.asarray([m1, m2, s1, s2, distance, f_low, phi_c, inc, psi, lam, beta]).T.flatten().copy()

        assert len(params) == 11 * self.num_sub
        reshape_shape = (self.num_sub, self.tdi_config.nchannels, self.N)

        # JAX branch: functional wave_gen.run_wave_tdi (see GBTDIonTheFly).
        if self.backend.name == self._BACKEND_PREFIX + "_jax":
            params_2d = params.reshape(self.num_sub, self.n_params)
            t_2d = self.t_arr.reshape(self.num_sub, self.N)
            _, tdi_amp_arr, tdi_phase_arr, phase_ref_arr = self.wave_gen.run_wave_tdi(
                params_2d, t_2d,
            )
            return self.from_tdi_output(TDIOutput(
                self.t_arr,
                tdi_amp_arr,
                tdi_phase_arr,
                phase_ref_arr,
                fill_splines=False,
                force_backend=self.backend,
            ), fill_splines=False if return_spline is False else return_spline)

        # C++ in-place path (unchanged).
        tdi_channels_arr = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=complex)
        tdi_amp = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        tdi_phase = self.xp.zeros((self.N * self.tdi_config.nchannels * self.num_sub), dtype=float)
        phase_ref = self.xp.zeros((self.N * self.num_sub), dtype=float)
        assert int(np.prod(self.t_arr.shape)) == self.N * self.num_sub

        self.wave_gen.run_wave_tdi_wrap(
            tdi_channels_arr,
            tdi_amp, tdi_phase,
            phase_ref,
            params, self.t_arr.flatten().copy(),
            self.N, self.num_sub, self.n_params, self.tdi_config.nchannels
        )

        return self.from_tdi_output(TDIOutput(
            self.t_arr,
            tdi_amp.reshape(reshape_shape),
            tdi_phase.reshape(reshape_shape),
            phase_ref.reshape(self.t_arr.shape),
            force_backend=self.backend
        ), fill_splines=return_spline)

class TDDenseTDIonTheFly(TDIonTheFly):
    """Template-batched TD TDI on the fly for multi-harmonic sources (EMRIs), exact phases.

    For multi-harmonic sources sharing one trajectory (EMRIs), with EXACT
    phases from the integrator's 8th-order dense output.

    Per template ``b``: the evaluation times ``t[b]`` (one common length ``N``), the
    integrator knots ``t_knots[b, :n_knots[b]]`` and DOPR853 phase coefficients
    ``phase_coeffs[b, K-1, 3, 8]`` (conventions -- massratio scaling, sign(xI0), the backwards
    offset -- already applied). Per harmonic ``s`` (templates' harmonics contiguous,
    ``sub_offsets``): integers ``sub_mkn[s] = (m, k, n)`` and a complex amplitude cubic spline
    over its template's knots, ``amp_re/amp_im[s, K-1, 4]`` (monomial coefficients in
    ``t - t_knot``). The strain term is ``amp_factor * c_s(t) * exp(-i Phi_s(t))``.

    The kernel computes the link geometry once per (template, time) and reuses it for every
    harmonic; the output is a :class:`TDTDIOutput` over all harmonics (``phase_ref`` = the pure
    carrier ``Phi_s`` at spacecraft-1 time; the amplitude's phase lands in ``tdi_phase``).

    Production: the ``EMRI_DIRECT_RESPONSE=dense`` path of
    :class:`~lisatools.sources.emri.wdm_direct.EMRIDirectWDM` (both launchers), on a
    sparse time grid. All arrays are held on the instance because the backend
    ``TDDenseTDIWaveformWrap`` keeps raw pointers to them.

    Args:
        t (array-like): ``(n_temp, N)`` evaluation times [s].
        sub_offsets (array-like): ``(n_temp + 1,)`` int harmonic offsets; template
            ``b`` owns harmonics ``sub_offsets[b]:sub_offsets[b + 1]``, ``num_sub =
            sub_offsets[-1]`` (numpy or cupy).
        sub_mkn (array-like): ``(num_sub, 3)`` int ``(m, k, n)`` per harmonic.
        t_knots (array-like): ``(n_temp, K)`` integrator knot times [s].
        n_knots (array-like): ``(n_temp,)`` int valid knots per template.
        phase_coeffs (array-like): ``(n_temp, K - 1, 3, 8)`` DOPR853 dense-output
            coefficients of the three phases.
        amp_re (array-like): ``(num_sub, K - 1, 4)`` monomial cubic coefficients of
            the amplitude's real part.
        amp_im (array-like): Same for the imaginary part.
        amp_factor (float, optional): Overall amplitude factor. Default ``1.0``.
        tdi_config (optional): As in :class:`TDIonTheFly`. Default ``None``.
        orbits (optional): As in :class:`TDIonTheFly`. Default ``None``
            (equal-arm; the global fit passes its unequal-arm orbits).
        force_backend (str, optional): ``lisatools`` backend flavor.

    Raises:
        RuntimeError: The backend module has no ``TDDenseTDIWaveformWrap``.
    """

    def __init__(self, t, sub_offsets, sub_mkn, t_knots, n_knots, phase_coeffs, amp_re, amp_im,
                 amp_factor=1.0, tdi_config=None, orbits=None, force_backend=None):
        sub_offsets = np.asarray(sub_offsets.get() if hasattr(sub_offsets, "get") else sub_offsets, dtype=np.int32)
        num_sub = int(sub_offsets[-1])
        super().__init__(1.0, num_sub, n_params=4, tdi_config=tdi_config, orbits=orbits, force_backend=force_backend)
        if self.backend.TDDenseTDIWaveformWrap is None:
            raise RuntimeError("this lisatools backend module has no TDDenseTDIWaveformWrap; rebuild it")
        xp = self.xp
        self.t = xp.ascontiguousarray(xp.asarray(t, dtype=xp.float64))
        self.n_temp, self.N = self.t.shape
        self.sub_offsets = xp.asarray(sub_offsets, dtype=xp.int32)
        sub_temp = np.repeat(np.arange(self.n_temp, dtype=np.int32), np.diff(sub_offsets))
        self.sub_temp_host = sub_temp
        self.sub_temp = xp.asarray(sub_temp, dtype=xp.int32)
        self.sub_mkn = xp.ascontiguousarray(xp.asarray(sub_mkn, dtype=xp.int32).reshape(num_sub, 3))
        self.K = int(np.asarray(t_knots.shape)[-1])
        self.t_knots = xp.ascontiguousarray(xp.asarray(t_knots, dtype=xp.float64).reshape(self.n_temp, self.K))
        self.n_knots = xp.asarray(n_knots, dtype=xp.int32)
        self.phase_coeffs = xp.ascontiguousarray(xp.asarray(phase_coeffs, dtype=xp.float64).reshape(self.n_temp, self.K - 1, 3, 8))
        self.amp_re = xp.ascontiguousarray(xp.asarray(amp_re, dtype=xp.float64).reshape(num_sub, self.K - 1, 4))
        self.amp_im = xp.ascontiguousarray(xp.asarray(amp_im, dtype=xp.float64).reshape(num_sub, self.K - 1, 4))
        self.amp_factor = float(amp_factor)
        # the wrap keeps raw pointers: these arrays live on self
        self.wave_gen = self.backend.TDDenseTDIWaveformWrap(
            self.cpp_orbits, self.cpp_tdi_config, self.n_temp, self.K, num_sub, self.amp_factor,
            self.sub_temp, self.sub_mkn.ravel(), self.n_knots, self.t_knots.ravel(),
            self.phase_coeffs.ravel(), self.amp_re.ravel(), self.amp_im.ravel())

    def _run(self, params):
        """Run the full kernel; return flat ``(chans, amp, phase, phi_ref)`` buffers.

        ``params`` is reshaped to ``(n_temp, 4)`` = ``(inc, psi, lam, beta)``. The buffers
        are ``num_sub * nchannels * N`` long (``phi_ref``: ``num_sub * N``).
        """
        xp = self.xp
        params = xp.ascontiguousarray(xp.asarray(params, dtype=xp.float64).reshape(self.n_temp, 4))
        nch = self.tdi_config.nchannels
        S, N = self.num_sub, self.N
        chans = xp.zeros(S * nch * N, dtype=complex)
        amp = xp.zeros(S * nch * N, dtype=float)
        phase = xp.zeros(S * nch * N, dtype=float)
        phi_ref = xp.zeros(S * N, dtype=float)
        self.wave_gen.run_wave_tdi_wrap(chans, amp, phase, phi_ref, params.ravel(), self.t.ravel(),
                                        self.sub_offsets, N, 4, nch)
        return chans, amp, phase, phi_ref

    def channels(self, params, sum_subs=False, stop_taper=0.0):
        """Return the raw complex TDI channels at the evaluation times.

        The exact channel is the real part (no amplitude/phase representation, so a signal
        that stops -- an in-window plunge -- keeps the TDI combination's own structure). Runs
        the channel kernel only (no amplitude/phase extraction); a backend module without
        ``run_channels_wrap`` falls back to the full kernel.

        ``stop_taper`` [s] > 0 weights every delay term by a sin^2 window on its EMISSION time that
        falls to 0 at the last knot over that span: the same window for every term, so the TDI
        cancellation holds through the stop, and each term is smooth there. Without it, a sampled
        channel keeps a net area from the terms' abrupt cuts falling between samples, which
        aliases into the lowest frequencies (``tests/test_tdi_dense.py``).

        Args:
            params (array-like): ``(n_temp, 4)`` = ``(inc, psi, lam, beta)`` per template.
            sum_subs (bool, optional): Sum each template's harmonics. Default ``False``.
            stop_taper (float, optional): Taper span [s]; ``0`` (default) is off.

        Returns:
            xp.ndarray: Complex ``(num_sub, nch, N)``, or ``(n_temp, nch, N)`` with
            ``sum_subs``.

        Raises:
            RuntimeError: ``stop_taper > 0`` on a module without ``run_channels_wrap``, or
                a compiled ``run_channels_wrap`` that predates ``stop_taper``.
        """
        xp = self.xp
        nch = self.tdi_config.nchannels
        if not hasattr(self.wave_gen, "run_channels_wrap"):          # an older module
            if stop_taper > 0:
                raise RuntimeError("TDDenseTDIonTheFly.channels(stop_taper>0) needs the compiled "
                                   "run_channels_wrap: rebuild lisatools (pip install -e . --no-build-isolation)")
            chans = self._run(params)[0].reshape(self.num_sub, nch, self.N)
            if not sum_subs:
                return chans
            return xp.stack([chans[int(a):int(b)].sum(axis=0) for a, b in
                             zip(self.sub_offsets[:-1].tolist(), self.sub_offsets[1:].tolist())])
        params = xp.ascontiguousarray(xp.asarray(params, dtype=xp.float64).reshape(self.n_temp, 4))
        rows = self.n_temp if sum_subs else self.num_sub
        chans = xp.zeros(rows * nch * self.N, dtype=complex)
        try:
            self.wave_gen.run_channels_wrap(chans, params.ravel(), self.t.ravel(), self.sub_offsets,
                                            self.N, 4, nch, int(bool(sum_subs)), float(stop_taper))
        except TypeError as exc:                                     # a module without stop_taper
            raise RuntimeError("the compiled run_channels_wrap predates stop_taper: rebuild lisatools "
                               f"(pip install -e . --no-build-isolation) ({exc})") from exc
        return chans.reshape(rows, nch, self.N)

    def __call__(self, params, return_spline: bool = True) -> "TDTDIOutput":
        """Run the kernel and return the amplitude / phase output of every harmonic.

        Args:
            params (array-like): ``(n_temp, 4)`` = ``(inc, psi, lam, beta)`` per template
                (sky in the orbits' frame).
            return_spline: Build the output's splines. Default ``True``.

        Returns:
            TDTDIOutput: On the per-harmonic time grid ``t[sub_temp]`` ``(num_sub, N)``,
            ``tdi_amp`` / ``tdi_phase`` ``(num_sub, nch, N)``, ``phase_ref`` ``(num_sub, N)``.
        """
        nch = self.tdi_config.nchannels
        S, N = self.num_sub, self.N
        chans, amp, phase, phi_ref = self._run(params)
        x = self.t[self.sub_temp]
        return TDTDIOutput(x, amp.reshape(S, nch, N), phase.reshape(S, nch, N), phi_ref.reshape(S, N),
                           fill_splines=return_spline, force_backend=self.backend.name.split("_")[-1])
