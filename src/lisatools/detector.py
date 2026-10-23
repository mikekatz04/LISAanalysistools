"""LISA orbit/detector geometry and constellation noise models.

Orbits (spacecraft positions, velocities, link unit vectors and light travel times):

- :class:`Orbits`: base class; reads a LISA Orbits-style HDF5 file, splines it onto a
  configured grid (lazily, on first use) and wraps the backend ``OrbitsWrap`` that the
  response / sensitivity kernels consume (:attr:`Orbits.pycppdetector`).
- :class:`EqualArmlengthOrbits` / :class:`DefaultOrbits` and :class:`ESAOrbits`: the
  packaged stock orbit files (SSB-ecliptic data, see the Warning in :class:`Orbits`).
- :class:`L1Orbits`: orbits read from a Mojito L1 file (separate position and LTT grids).
  This is the class the global-fit production runs load from the Mojito NOISE file, with
  ``frame="icrs"`` (the stock fits' ``general.orbits_frame``,
  ``EreborGeneralSettings.orbits_frame``).

Noise models: :class:`LISAModel` (OMS + acceleration levels, the stock ``scirdv1``,
``proposal``, ``mrdv1``, ``sangria``) and :class:`ExtendedLISAModel` (per-interferometer
levels, ``sangria_v2``), resolved from strings by :func:`check_lisa_model`. ``LINKS`` /
``SC`` fix the link and spacecraft order used throughout (link ``ij`` = light received at
spacecraft ``i``, emitted by ``j``).

Importing this module enables JAX 64-bit mode (``jax_enable_x64``) and sets
``XLA_PYTHON_CLIENT_PREALLOCATE=false`` when JAX is importable.

Example:
    >>> import numpy as np
    >>> from lisatools.detector import EqualArmlengthOrbits
    >>> # a coarse 1-day grid keeps this cheap (the default 50 s grid spans the whole file)
    >>> orbits = EqualArmlengthOrbits(force_backend="cpu", linear_interp_setup=False, dt=86400.0)
    >>> ltt = orbits.get_light_travel_times(np.array([0.0, 1.0e6]), 12)
    >>> ltt.shape
    (2,)
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import h5py
from logging import getLogger
import numpy as np
import requests
from scipy import interpolate

from .utils.constants import *
from .utils.parallelbase import LISAToolsParallelModule
from .utils.utility import get_array_module

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    jax_here = True
except (ModuleNotFoundError, ImportError):
    jax = None
    jnp = None
    jax_here = False

logger = getLogger(__name__)

# Spacecraft order, and link order (link ij = received at S/C i, emitted by S/C j).
SC = [1, 2, 3]
LINKS = [12, 23, 31, 13, 32, 21]

# Grid step [s] of Orbits' linear_interp_setup=True grid (L1Orbits uses linear_interp_dt).
LINEAR_INTERP_TIMESTEP = 50.00  # 600.00  # sec (0.25 hr)


def icrs_to_ecliptic(positions_icrs: np.ndarray) -> np.ndarray:
    """Convert cartesian positions from ICRS to ecliptic coordinates.

    Uses ``astropy`` (``SkyCoord(..., frame="icrs")`` transformed to
    ``BarycentricMeanEcliptic``), one spacecraft at a time. Being a pure rotation about the
    barycentre, it applies to velocities as well (:class:`L1Orbits` uses it for both).

    Args:
        positions_icrs: Array of shape ``(n_times, 3, 3)`` (time, spacecraft, xyz) of
            cartesian vectors in the ICRS frame, in metres.

    Returns:
        np.ndarray: Array of shape ``(n_times, 3, 3)`` in the ecliptic frame, same units.
    """

    import astropy

    positions_ecliptic = np.zeros_like(positions_icrs)

    for sc in range(3):

        c_icrs = astropy.coordinates.SkyCoord(
            positions_icrs[:, sc, 0],
            positions_icrs[:, sc, 1],
            positions_icrs[:, sc, 2],
            frame="icrs",
            unit="m",
            representation_type="cartesian",
        )
        c_ecliptic = c_icrs.transform_to(astropy.coordinates.BarycentricMeanEcliptic)
        c_ecliptic.representation_type = "cartesian"
        positions_ecliptic[:, sc, :] = np.array(
            [c_ecliptic.x.value, c_ecliptic.y.value, c_ecliptic.z.value]
        ).T

    return positions_ecliptic

class Orbits(LISAToolsParallelModule, ABC):
    """LISA Orbit Base Class.

    Holds the spacecraft positions ``x``, velocities ``v``, link unit vectors ``n`` and light
    travel times ``ltt`` of one orbit file, and builds the backend ``OrbitsWrap`` (C++/CUDA)
    that the response, TDI-on-the-fly and sensitivity kernels read through
    :attr:`pycppdetector` / :attr:`pycppdetector_args`.

    Construction only reads the file attributes (``*_base`` values such as ``size_base``,
    ``dt_base``). The interpolation grid is built LAZILY by :meth:`_configure` on the first
    access to a configured quantity (``t``, ``x``, ``n``, ``v``, ``ltt``, ``size``,
    ``pycppdetector``, ...): each base table is cubic-splined onto the grid selected by
    ``linear_interp_setup`` / ``t_arr`` / ``dt`` (in that order of precedence). Unconfigured
    instances are cheap to hold and deepcopy.

    Args:
        filename: Orbit file in the LISA Orbits HDF5 layout (``tcb/x``, ``tcb/v``, ``tcb/n``,
            ``tcb/ltt`` datasets plus ``size`` / ``dt`` attributes). A path that does not
            exist is looked up in the packaged ``orbit_files/`` directory (and fetched from
            GitHub when missing there).
        armlength: Nominal armlength in metres; must be a ``float``. Default ``2.5e9``.
        force_backend: Backend name (``"cpu"``, ``"cuda12x"``, ``"cuda"``, ``"gpu"``, ...),
            see :class:`~lisatools.utils.parallelbase.LISAToolsParallelModule`. ``None``
            picks the first available of ``cuda13x``, ``cuda12x``, ``cuda11x``, ``cpu``.
        t0: Start time in seconds added to the file's sample times. Default ``0.0``.
        frame: Reference frame of the orbit data, either ``"ecliptic"`` (SSB ecliptic) or
            ``"icrs"`` (equatorial). Default ``"icrs"`` (the global-fit run frame). The base
            class applies no conversion: ``frame`` labels the data, and sky coordinates fed
            to any response built on these orbits must be expressed in the same frame;
            consumers can check ``orbits.frame`` to enforce this. :class:`L1Orbits` does
            convert: its Mojito data are ICRS and ``frame="ecliptic"`` rotates them at load.
        t_arr: Explicit grid to configure onto, within the file's time span. Default
            ``None``.
        dt: Grid step in seconds over the file's span. Default ``None``.
        linear_interp_setup: Build the dense constant-step grid
            (``LINEAR_INTERP_TIMESTEP`` = 50 s) that the backend interpolates linearly.
            Default ``True``; required for :attr:`pycppdetector`.
        **kwargs: Accepted and ignored.

    Raises:
        ValueError: If ``frame`` is not ``"ecliptic"`` / ``"icrs"``, ``armlength`` is not a
            ``float``, or a packaged file cannot be fetched.

    Warning:
        The packaged stock files (``equalarmlength-orbits.h5``, ``esa-trailing-orbits.h5``)
        hold SSB-ecliptic positions (the constellation barycentre stays in the ``z = 0``
        plane), but :class:`EqualArmlengthOrbits` / :class:`ESAOrbits` inherit the
        ``frame="icrs"`` default, so their label does not match the data unless
        ``frame="ecliptic"`` is passed. With the ``"icrs"`` label,
        :meth:`get_constellation_angles` also rotates the (already ecliptic) positions.

    Example:
        >>> from lisatools.detector import EqualArmlengthOrbits
        >>> orbits = EqualArmlengthOrbits(force_backend="cpu", frame="ecliptic")
        >>> orbits.armlength
        2500000000.0
        >>> orbits.LINKS
        [12, 23, 31, 13, 32, 21]
    """

    def __init__(
        self,
        filename: str,
        armlength: Optional[float] = 2.5e9,
        force_backend: Optional[str] = None,
        t0: Optional[float] = 0.0,
        frame: str = "icrs",
        t_arr: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        linear_interp_setup: bool = True,
        **kwargs,
    ) -> None:

        # TODO: should we make it compute armlength.
        self.filename = filename
        self.armlength = armlength
        self.t0 = t0
        self.frame = frame
        self._setup()
        self.configured = False
        # Grid-configuration arguments (formerly passed to the separate
        # ``configure()`` call). Configuration itself is LAZY: it runs on
        # first access to any configured-grid quantity (``t``, ``x``, ``n``,
        # ``v``, ``ltt``, ``pycppdetector``, ...) via
        # :meth:`_ensure_configured`. Construction therefore stays cheap
        # (stock orbit files span years; building the interpolation grid
        # costs seconds and GBs) -- settings trees can hold / deepcopy
        # unconfigured instances for free, and workflows that trim the
        # base tables after construction still work as long as they trim
        # before first use.
        self._configure_kwargs = dict(
            t_arr=t_arr, dt=dt, linear_interp_setup=linear_interp_setup
        )
        LISAToolsParallelModule.__init__(self, force_backend=force_backend)

    @property
    def frame(self) -> str:
        """Reference frame of the orbit data (``"ecliptic"`` or ``"icrs"``)."""
        return self._frame

    @frame.setter
    def frame(self, frame: str) -> None:
        """frame setter."""
        if frame not in ["ecliptic", "icrs"]:
            raise ValueError(
                f"frame must be 'ecliptic' or 'icrs', got {frame!r}."
            )
        self._frame = frame

    @property
    def args(self):
        """Arguments for recreating this class instance."""
        return (self.filename,)

    @property
    def kwargs(self):
        """Keyword arguments for recreating this class instance."""
        return {
            "armlength": self.armlength,
            "force_backend": self.backend.backend_name.split("_")[-1],
            "t0": self.t0,
            "frame": self.frame,
        }

    @property
    def xp(self):
        """numpy or cupy based on backend."""
        return self.backend.xp

    @property
    def armlength(self) -> float:
        """Armlength parameter."""
        return self._armlength

    @armlength.setter
    def armlength(self, armlength: float) -> None:
        """armlength setter."""

        if isinstance(armlength, float):
            # TODO: put error check that it is close
            self._armlength = armlength

        else:
            raise ValueError("armlength must be float.")

    @property
    def LINKS(self) -> List[int]:
        """Link order."""
        return LINKS

    @property
    def SC(self) -> List[int]:
        """Spacecraft order."""
        return SC

    @property
    def link_space_craft_r(self) -> List[int]:
        """Receiver (first) spacecraft"""
        return [int(str(link_i)[0]) for link_i in self.LINKS]

    @property
    def link_space_craft_e(self) -> List[int]:
        """Sender (second) spacecraft"""
        return [int(str(link_i)[1]) for link_i in self.LINKS]

    def _setup(self) -> None:
        """Read in orbital data from file and store."""
        with self.open() as f:
            for key in f.attrs.keys():
                setattr(self, key + "_base", f.attrs[key])

    @property
    def filename(self) -> str:
        """Orbit file name."""
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        """Set file name.

        An existing path is used as is. Otherwise ``filename`` is resolved inside the packaged
        ``orbit_files/`` directory (created if absent), downloading it from the
        ``lisa-analysis-tools`` GitHub repository when not present there. The download URL
        is a GitHub ``/blob/`` page, which serves HTML rather than the raw HDF5 file, so this
        fallback does not produce a readable orbit file.
        """

        assert isinstance(filename, str)

        if os.path.exists(filename):
            self._filename = filename

        else:
            # get path
            path_to_this_file = __file__.split("detector.py")[0]

            # make sure orbit_files directory exists in the right place
            if not os.path.exists(path_to_this_file + "orbit_files/"):
                os.mkdir(path_to_this_file + "orbit_files/")
            path_to_this_file = path_to_this_file + "orbit_files/"

            if not os.path.exists(path_to_this_file + filename):
                # download files from github if they are not there
                github_file = f"https://github.com/lisa-analysis-tools/lisa-analysis-tools/blob/main/src/lisatools/orbit_files/{filename}"
                r = requests.get(github_file)

                # if not success
                if r.status_code != 200:
                    raise ValueError(
                        f"Cannot find {filename} within default files located at github.com/lisa-analysis-tools/lisa-analysis-tools/lisatools/orbit_files/."
                    )
                # write the contents to a local file
                with open(path_to_this_file + filename, "wb") as f:
                    f.write(r.content)

            # store
            self._filename = path_to_this_file + filename

    def open(self) -> h5py.File:
        """Open the orbit file read-only.

        Returns:
            h5py.File: The opened file; use it as a context manager.
        """
        f = h5py.File(self.filename, "r")
        return f

    @property
    def t_base(self) -> np.ndarray:
        """Time array from file."""
        with self.open() as f:
            t_base = self.t0 + np.arange(self.size_base) * self.dt_base
        return t_base

    @property
    def ltt_base(self) -> np.ndarray:
        """Light travel times along links from file."""
        with self.open() as f:
            ltt = f["tcb"]["ltt"][:]
        return ltt

    @property
    def n_base(self) -> np.ndarray:
        """Normal unit vectors towards receiver along links from file."""
        with self.open() as f:
            n = f["tcb"]["n"][:]
        return n

    @property
    def x_base(self) -> np.ndarray:
        """Spacecraft position from file."""
        with self.open() as f:
            x = f["tcb"]["x"][:]
        return x

    @property
    def v_base(self) -> np.ndarray:
        """Spacecraft velocities from file."""
        with self.open() as f:
            v = f["tcb"]["v"][:]
        return v

    @property
    def t(self) -> np.ndarray:
        """Configured time array."""
        self._check_configured()
        return self._t

    @t.setter
    def t(self, t: np.ndarray):
        """Set configured time array."""
        assert isinstance(t, np.ndarray) and t.ndim == 1
        self._t = t

    @property
    def sc_t(self) -> np.ndarray:
        """Spacecraft time grid. In this version of the orbits all the quantities are on the same grid"""
        return self.t
    
    @property
    def ltt_t(self) -> np.ndarray:
        """LTTs time grid. In this version of the orbits all the quantities are on the same grid"""
        return self.t
    
    @property
    def ltt(self) -> np.ndarray:
        """Light travel times [s] on the configured grid, shape ``(size, 6)`` in ``LINKS`` order."""
        self._check_configured()
        return self._ltt

    @ltt.setter
    def ltt(self, ltt: np.ndarray) -> np.ndarray:
        """Check a light-travel-time array against the grid; does NOT store it.

        Only asserts ``ltt.shape[0] == len(self.t)``. :meth:`_configure` writes ``_ltt``
        directly.
        """
        assert ltt.shape[0] == len(self.t)

    @property
    def n(self) -> np.ndarray:
        """Link unit vectors on the configured grid, shape ``(size, 6, 3)`` in ``LINKS`` order."""
        self._check_configured()
        return self._n

    @n.setter
    def n(self, n: np.ndarray) -> np.ndarray:
        """No-op setter: the value is discarded (:meth:`_configure` writes ``_n`` directly)."""
        return self._n

    @property
    def x(self) -> np.ndarray:
        """Spacecraft positions [m] on the configured grid, shape ``(size, 3, 3)`` (S/C, xyz)."""
        self._check_configured()
        return self._x

    @x.setter
    def x(self, x: np.ndarray) -> np.ndarray:
        """No-op setter: the value is discarded (:meth:`_configure` writes ``_x`` directly)."""
        return self._x

    @property
    def v(self) -> np.ndarray:
        """Spacecraft velocities [m/s] on the configured grid, shape ``(size, 3, 3)``."""
        self._check_configured()
        return self._v

    @v.setter
    def v(self, v: np.ndarray) -> np.ndarray:
        """No-op setter: the value is discarded (:meth:`_configure` writes ``_v`` directly)."""
        return self._v

    @property
    def average_ltt(self) -> float:
        """Average light travel time [s] over all links and grid points."""
        return float(np.mean(self.ltt))

    def configure(
        self,
        t_arr: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        linear_interp_setup: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Deprecated: pass grid arguments to ``__init__`` instead.

        Configuration now happens lazily on first use with the arguments
        stored at construction. Explicit calls still work (and are the way
        to RE-configure after mutating the base tables, e.g. trimming the
        LTT arrays), but new code should construct with the desired
        ``t_arr`` / ``dt`` / ``linear_interp_setup`` (/ ``linear_interp_dt``
        for :class:`L1Orbits`) and let first use trigger the build.

        Args:
            t_arr: See :meth:`_configure`.
            dt: See :meth:`_configure`.
            linear_interp_setup: See :meth:`_configure`. Default ``False`` here (the
                constructor default is ``True``).
            **kwargs: Forwarded to :meth:`_configure` (``linear_interp_dt`` for
                :class:`L1Orbits`).

        Warning:
            Emits a ``DeprecationWarning`` on every call.
        """
        warnings.warn(
            "Orbits.configure() is deprecated: pass the grid arguments to "
            "__init__ and let configuration happen lazily on first use. "
            "(Explicit calls remain supported for re-configuring after "
            "mutating the base tables.)",
            DeprecationWarning,
            stacklevel=2,
        )
        self._configure(t_arr=t_arr, dt=dt, linear_interp_setup=linear_interp_setup, **kwargs)

    def _configure(
        self,
        t_arr: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        linear_interp_setup: Optional[bool] = False,
    ) -> None:
        """Configure the orbits to match the signal response generator time basis.

        The base orbits will be scaled up or down as needed using Cubic Spline interpolation.
        The hierarchy of consideration to each keyword argument if multiple are given:
        ``linear_interp_setup``, ``t_arr``, ``dt``.

        If nothing is provided, the base points are used (and no backend arguments are built,
        so :attr:`pycppdetector` is unavailable).

        Sets ``t``, ``_ltt``, ``_x``, ``_n``, ``_v``, ``dt``, ``configured = True`` and
        :attr:`pycppdetector_args`. The backend grid starts at the configured grid's first
        node.

        Args:
            t_arr: New time array; must lie inside the file's time span (asserted). Its step
                is taken from the first two entries.
            dt: New time step. Will take the time duration to be that of the input data.
            linear_interp_setup: If ``True``, it will create a dense grid designed for linear
                interpolation with a constant time step (``LINEAR_INTERP_TIMESTEP``).
        """

        x_orig = self.t_base

        # everything up base on input
        if linear_interp_setup:
            # setup spline
            make_cpp = True
            dt = LINEAR_INTERP_TIMESTEP
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = self.t0 + np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])
        elif t_arr is not None:
            # check array inputs and fill dt
            assert np.all(t_arr >= self.t_base[0]) and np.all(t_arr <= self.t_base[-1])
            make_cpp = True
            dt = abs(t_arr[1] - t_arr[0])

        elif dt is not None:
            # fill array based on dt and base t
            make_cpp = True
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = self.t0 + np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])

        else:
            make_cpp = False
            t_arr = self.t_base

        x_new = t_arr.copy()
        self.t = t_arr.copy()

        # use base quantities, and interpolate to prepare new arrays accordingly
        for which in ["ltt", "x", "n", "v"]:
            arr = getattr(self, which + "_base")
            arr_tmp = arr.reshape(self.size_base, -1)
            arr_out_tmp = np.zeros((len(x_new), arr_tmp.shape[-1]))
            for i in range(arr_tmp.shape[-1]):
                arr_out_tmp[:, i] = interpolate.CubicSpline(x_orig, arr_tmp[:, i])(x_new)
            arr_out = arr_out_tmp.reshape((len(x_new),) + arr.shape[1:])
            setattr(self, "_" + which, arr_out)

        # make sure base spacecraft and link inormation is ready
        lsr = np.asarray(self.link_space_craft_r).copy().astype(np.int32)
        lse = np.asarray(self.link_space_craft_e).copy().astype(np.int32)
        ll = np.asarray(self.LINKS).copy().astype(np.int32)

        # indicate this class instance has been configured
        self.configured = True

        # prepare cpp class args to load when needed
        if make_cpp:
            # The C++ ``Orbits`` locates its window as ``int((t - t0) / dt)``
            # over ``len(self.t)`` nodes, so the start time handed to it MUST
            # be the configured grid's first node. The ``linear_interp_setup``
            # and ``dt`` branches build ``t_arr`` from ``self.t0``, so this is
            # identical to the old ``self.t0`` there; the explicit ``t_arr``
            # branch can start anywhere in the file (a short window around a
            # late ``t_obs_start``), and handing it ``self.t0`` put every
            # evaluation time past the last node -> ``get_window`` = -1 ->
            # zero unit vectors / positions -> EXACTLY ZERO templates
            # (GBGPU tests/psd_mirror_fixture.py, 2026-09-09).
            t_cpp_start = float(t_arr[0])
            self.pycppdetector_args = [ # duplicate ltts and positions informations when using the more general c++ class
                t_cpp_start,
                dt,
                len(self.t),
                t_cpp_start,
                dt,
                len(self.t),
                self.xp.asarray(self.n.flatten().copy()),
                self.xp.asarray(self.ltt.flatten().copy()),
                self.xp.asarray(self.x.flatten().copy()),
                self.xp.asarray(ll),
                self.xp.asarray(lsr),
                self.xp.asarray(lse),
                self.armlength,
            ]
            self.dt = dt
        else:
            self.pycppdetector_args = None
            self.dt = dt

    @property
    def dt(self) -> float:
        """Configured grid step [s]. Raises ``ValueError`` when the grid has no step."""
        if self._dt is None:
            raise ValueError("dt not available for t_arr only.")
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        """Set the configured time step."""
        self._dt = dt

    @property
    def sc_dt(self) -> float:
        """Spacecraft dt. In this version of the orbits all the quantities are on the same grid"""
        return self.dt
    
    @property
    def ltt_dt(self) -> float:
        """LTTs dt. In this version of the orbits all the quantities are on the same grid"""
        return self.dt

    @property
    def pycppdetector(self) -> object:
        """Backend ``OrbitsWrap`` instance (C++ / CUDA) built from :attr:`pycppdetector_args`.

        Configures the orbits first if needed. A NEW wrapper is constructed on every access
        with this object's backend (and stored on ``_pycppdetector``); it points at the
        arrays held in :attr:`pycppdetector_args`.

        Raises:
            ValueError: When the configured grid carries no backend arguments
                (``linear_interp_setup=False`` with no ``t_arr`` / ``dt``).
        """
        self._ensure_configured()
        if self._pycppdetector_args is None:
            raise ValueError(
                "Asking for c++ class. Need linear_interp_setup = True (the "
                "__init__ default) so the configured grid carries C++ args."
            )
        self._pycppdetector = self.backend.OrbitsWrap(*self._pycppdetector_args)
        return self._pycppdetector

    @property
    def pycppdetector_args(self) -> tuple:
        """Constructor arguments of the backend ``OrbitsWrap`` (configures first if needed).

        A list ``[sc_t0, sc_dt, sc_N, ltt_t0, ltt_dt, ltt_N, n, ltt, x, links, sc_r, sc_e,
        armlength]`` with the flattened arrays on this object's backend, or ``None`` when the
        grid was configured without them. Downstream response classes build their own
        wrappers from it.
        """
        self._ensure_configured()
        return self._pycppdetector_args

    @pycppdetector_args.setter
    def pycppdetector_args(self, pycppdetector_args: tuple) -> None:
        """Store args used to build the C++ ``OrbitsWrap`` instance."""
        self._pycppdetector_args = pycppdetector_args

    @property
    def size(self) -> int:
        """Number of points of the configured grid."""
        self._check_configured()
        return len(self.t)

    def _ensure_configured(self) -> None:
        """Lazily build the interpolation grid on first use.

        Runs :meth:`_configure` with the arguments stored at construction
        the first time any configured-grid quantity is requested. Workflows
        that mutate the base tables after construction (e.g. trimming the
        LTT arrays to an analysis window to save memory) must do so BEFORE
        first use; call :meth:`_configure` explicitly to rebuild afterwards.
        """
        if not self.configured:
            self._configure(**self._configure_kwargs)

    # Back-compat alias (semantics changed from raise -> lazy configure).
    _check_configured = _ensure_configured

    def get_light_travel_times(
        self, t: float | np.ndarray, link: int | np.ndarray
    ) -> float | np.ndarray:
        """Compute light travel time as a function of time.

        Computes with the c++ backend (:attr:`pycppdetector`), linear interpolation on the
        configured grid.

        Args:
            t: Time(s) in seconds: a Python ``float`` or an ``xp.ndarray`` on this object's
                backend.
            link: Link id(s), each in ``self.LINKS`` (e.g. ``12``): an ``int`` or an
                ``xp.ndarray`` matching ``t``.

        Returns:
            float or xp.ndarray: Light travel time(s) in seconds; a scalar for
            ``(float, int)`` input, otherwise an array shaped like ``t``.

        Raises:
            ValueError: For any other combination of input types (e.g. a numpy ``t`` on a GPU
                backend, or a ``float`` ``t`` with an array ``link``).
        """
        # test and prepare inputs
        if isinstance(t, float) and isinstance(link, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            link = self.xp.atleast_1d(link).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            link = self.xp.full_like(t, link, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            link = self.xp.asarray(link).astype(np.int32)
        else:
            raise ValueError(
                "(t, link) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray)."
            )

        # buffer array and c computation
        ltt_out = self.xp.zeros_like(t)

        self.pycppdetector.get_light_travel_time_wrap(
            ltt_out, t, link, len(ltt_out)
        )

        # prepare output
        if squeeze:
            return ltt_out[0]
        return ltt_out

    def get_pos(self, t: float | np.ndarray, sc: int | np.ndarray) -> np.ndarray:
        """Compute spacecraft position as a function of time.

        Computes with the c++ backend (:attr:`pycppdetector`), in the orbit data's
        :attr:`frame`.

        Args:
            t: Time(s) in seconds: a Python ``float`` or an ``xp.ndarray`` on this object's
                backend.
            sc: Spacecraft index(es), each in ``self.SC`` (1, 2, 3): an ``int`` or an
                ``xp.ndarray`` matching ``t``.

        Returns:
            xp.ndarray: Position(s) in metres, shape ``(len(t), 3)``; ``(3,)`` for
            ``(float, int)`` input.

        Raises:
            ValueError: For any other combination of input types (including arrays from the
                wrong backend).
        """
        # test and setup inputs accordingly
        if isinstance(t, float) and isinstance(sc, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            sc = self.xp.atleast_1d(sc).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(sc, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            sc = self.xp.full_like(t, sc, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(sc, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            sc = self.xp.asarray(sc).astype(np.int32)

        else:
            raise ValueError(
                "(t, sc) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray). If the inputs follow this, make sure the orbits class GPU setting matches the arrays coming in (GPU or CPU)."
            )

        # buffer arrays for input into c code
        pos_x = self.xp.zeros_like(t)
        pos_y = self.xp.zeros_like(t)
        pos_z = self.xp.zeros_like(t)

        # c code computation
        self.pycppdetector.get_pos_wrap(pos_x, pos_y, pos_z, t, sc, len(pos_x))

        # prepare output
        output = self.xp.array([pos_x, pos_y, pos_z]).T
        if squeeze:
            return output.squeeze()
        return output

    def get_normal_unit_vec(self, t: float | np.ndarray, link: int | np.ndarray) -> np.ndarray:
        """Compute link normal vector as a function of time.

        Computes with the c++ backend (:attr:`pycppdetector`).

        Args:
            t: Time(s) in seconds: a Python ``float`` or an ``xp.ndarray`` on this object's
                backend.
            link: Link id(s), each in ``self.LINKS``: an ``int`` or an ``xp.ndarray``
                matching ``t``.

        Returns:
            xp.ndarray: Unit vector(s), shape ``(len(t), 3)``; ``(3,)`` for ``(float, int)``
            input.

        Raises:
            ValueError: For any other combination of input types.
        """
        # test and prepare inputs
        if isinstance(t, float) and isinstance(link, int):
            squeeze = True
            t = self.xp.atleast_1d(t)
            link = self.xp.atleast_1d(link).astype(np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, int):
            squeeze = False
            t = self.xp.atleast_1d(t)
            link = self.xp.full_like(t, link, dtype=np.int32)

        elif isinstance(t, self.xp.ndarray) and isinstance(link, self.xp.ndarray):
            squeeze = False
            t = self.xp.asarray(t)
            link = self.xp.asarray(link).astype(np.int32)
        else:
            raise ValueError(
                "(t, link) can be (float, int), (np.ndarray, int), (np.ndarray, np.ndarray)."
            )

        # c code with buffers
        normal_unit_vec_x = self.xp.zeros_like(t)
        normal_unit_vec_y = self.xp.zeros_like(t)
        normal_unit_vec_z = self.xp.zeros_like(t)

        # c code
        self.pycppdetector.get_normal_unit_vec_wrap(
            normal_unit_vec_x,
            normal_unit_vec_y,
            normal_unit_vec_z,
            t,
            link,
            len(normal_unit_vec_x),
        )

        # prep outputs
        output = self.xp.array([normal_unit_vec_x, normal_unit_vec_y, normal_unit_vec_z]).T
        if squeeze:
            return output.squeeze()
        return output
    
    def get_constellation_angles(self, t: float | np.ndarray) -> tuple[float | np.ndarray, float | np.ndarray]:
        r"""Compute LISA orbital phase (:math:`\alpha`) and constellation rotation (:math:`\beta`).

        Derived from spacecraft positions at the given time(s), computed with the c++ backend
        and rotated to the ecliptic first when :attr:`frame` is ``"icrs"``. :math:`\alpha` is
        the ecliptic longitude of the constellation barycentre; :math:`\beta` is the angle of
        arm ``SC3 - SC2`` in the ecliptic plane minus :math:`\alpha`, wrapped into
        :math:`[-\pi, \pi]` with ``arctan2``. Used by the galactic-grid sensitivity setup.
        Adapted from @Federico Pozzoli.

        Args:
            t: Time(s) in seconds: a Python ``float`` (clamped, with a warning, to the start of
                the spacecraft grid when earlier) or an array (all entries must be on the
                grid).

        Returns:
            tuple: ``(alpha, beta)`` in radians, floats for a ``float`` input, otherwise
            ``np.ndarray`` of shape ``(len(t),)``.

        Raises:
            ValueError: If an array ``t`` has entries before the start of the spacecraft grid.
        """
        squeeze = isinstance(t, float)
        if isinstance(t, float) and t < self.sc_t[0]:
            msg = (
                f"Time {t} is before the start of the spacecraft time grid {self.sc_t[0]}. "
                "Setting `t` to the start of the grid for this computation."
                )
            logger.warning(msg)
                           
            t = self.sc_t[0]
        
        elif isinstance(t, (np.ndarray, self.xp.ndarray)) and np.any(t < self.sc_t[0]):
            msg = (
                    f"Some times in `t` are before the start of the spacecraft time grid `t_0 = {self.sc_t[0]} s`. "
                    "If providing an array, make sure all times are within the spacecraft time grid: "
                    f"{self.sc_t[0]} s to {self.sc_t[-1]} s."
                )
            raise ValueError(msg)

        sc1_pos = self.get_pos(t, sc=1)
        sc2_pos = self.get_pos(t, sc=2)
        sc3_pos = self.get_pos(t, sc=3)

        # move to host if on gpu
        if hasattr(sc1_pos, "get"):
            sc1_pos = sc1_pos.get()
            sc2_pos = sc2_pos.get()
            sc3_pos = sc3_pos.get()

        # make sure everything is 2D here
        sc1_pos = np.atleast_2d(sc1_pos)
        sc2_pos = np.atleast_2d(sc2_pos)
        sc3_pos = np.atleast_2d(sc3_pos)

        if hasattr(self, "frame") and self.frame == "icrs":
            # convert to ecliptic coordinates if needed
            all_positions = np.stack([sc1_pos, sc2_pos, sc3_pos], axis=1)  # shape (n_times, 3, 3)
            all_positions_ecliptic = icrs_to_ecliptic(all_positions)
            sc1_pos = all_positions_ecliptic[:, 0, :]
            sc2_pos = all_positions_ecliptic[:, 1, :]
            sc3_pos = all_positions_ecliptic[:, 2, :]

        # Compute LISA barycenter in ecliptic frame
        r_bc = (sc1_pos + sc2_pos + sc3_pos) / 3.0

        # α: Phase of barycenter in ecliptic plane
        alpha = np.arctan2(r_bc[:, 1], r_bc[:, 0])
        
        # For beta, measure rotation of constellation
        # Use arm L̂2 (SC3 - SC2) which should point in +x̂ direction when beta = 0
        arm2_vec = sc3_pos - sc2_pos
        # Project onto ecliptic plane (xy components)
        arm2_x = arm2_vec[:, 0]
        arm2_y = arm2_vec[:, 1]

        # Angle of L̂2 in ecliptic SSB coordinates
        psi2 = np.arctan2(arm2_y, arm2_x)

        # In the constellation frame at beta=0, L̂2 points in +x̂ direction
        # The constellation frame rotates with the barycenter, so:
        # beta = (angle of L̂2 in ecliptic SSB) - alpha
        beta = psi2 - alpha
        
        # Normalize to [-π, π)
        beta = np.arctan2(np.sin(beta), np.cos(beta))

        if squeeze:
            return float(alpha[0]), float(beta[0])
        return alpha, beta


    @property
    def ptr(self) -> int:
        """Pointer to the C++ ``Orbits`` object of a freshly built :attr:`pycppdetector`."""
        return self.pycppdetector.ptr

    @classmethod
    def supported_backends(cls):
        """Return the backends this orbit class supports, in order of preference.

        Returns:
            list[str]: ``lisatools_cuda13x``, ``lisatools_cuda12x``, ``lisatools_cuda11x``,
            ``lisatools_cpu``.
        """
        return ["lisatools_" + _tmp for _tmp in cls.GPU_RECOMMENDED()]


class EqualArmlengthOrbits(Orbits):
    """Equal Armlength Orbits.

    Orbit file: ``equalarmlength-orbits.h5`` (packaged under ``lisatools/orbit_files``).
    Used for quick-start work and as the
    :class:`DefaultOrbits` of the stock :class:`LISAModel` objects; production global fits
    load :class:`L1Orbits` from the Mojito data instead. The file's positions are
    SSB-ecliptic while the inherited default label is ``frame="icrs"`` (see the Warning in
    :class:`Orbits`).

    Args:
        *args: Positional arguments for :class:`Orbits` after ``filename`` (``armlength``,
            ``force_backend``, ...).
        **kwargs: Keyword arguments for :class:`Orbits` (``force_backend``, ``t0``,
            ``frame``, ``t_arr``, ``dt``, ``linear_interp_setup``).

    Example:
        >>> from lisatools.detector import EqualArmlengthOrbits
        >>> orbits = EqualArmlengthOrbits(
        ...     force_backend="cpu", frame="ecliptic", linear_interp_setup=False, dt=86400.0
        ... )
        >>> orbits.ltt.shape[1]
        6
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__("equalarmlength-orbits.h5", *args, **kwargs)

    @property
    def args(self):
        """Arguments for recreating this class instance."""
        return ()

class ESAOrbits(Orbits):
    """ESA Orbits.

    Orbit file: ``esa-trailing-orbits.h5`` (packaged under ``lisatools/orbit_files``). The
    file's positions are SSB-ecliptic while the inherited default label is ``frame="icrs"``
    (see the Warning in :class:`Orbits`).

    Args:
        *args: Positional arguments for :class:`Orbits` after ``filename``.
        **kwargs: Keyword arguments for :class:`Orbits` (``force_backend``, ``t0``,
            ``frame``, ``t_arr``, ``dt``, ``linear_interp_setup``).

    Example:
        >>> from lisatools.detector import ESAOrbits
        >>> orbits = ESAOrbits(force_backend="cpu")
    """

    def __init__(self, *args, **kwargs):
        super().__init__("esa-trailing-orbits.h5", *args, **kwargs)

    @property
    def args(self):
        """Arguments for recreating this class instance."""
        return ()



class L1Orbits(Orbits):
    """Base class for LISA Orbits from Mojito L1 File structure.

    This class handles orbit data from MojitoL1File where:

    - Light travel times and positions have different time arrays
    - Both time arrays may start at t0 != 0

    The global-fit production runs build it from the Mojito ``INSTRUMENT/L1`` NOISE file
    (:mod:`lisatools.globalfit.preprocessing`) with ``frame="icrs"`` and configure it
    eagerly at data load. Positions / velocities are cubic-splined onto a grid of step
    ``min(file step, linear_interp_dt)``, link unit vectors are rebuilt from the splined
    positions at emission and reception times, and the dense light travel times keep their
    native grid; the backend ``OrbitsWrap`` receives both grids separately.

    Args:
        filename: Path to the Mojito L1 HDF5 file (read with ``mojito.MojitoL1File``).
        armlength: Nominal armlength of detector in metres (default 2.5e9 m).
        force_backend: Backend name, see :class:`Orbits`. Default ``None``.
        frame: ``"icrs"`` (default) keeps the file's ICRS positions / velocities;
            ``"ecliptic"`` rotates both to the ecliptic at load (:func:`icrs_to_ecliptic`).
        linear_interp_dt: Cap in seconds on the position / unit-vector grid step used with
            ``linear_interp_setup=True``; ``None`` keeps the file's native cadence. Default
            ``500.0``. See :meth:`_configure`.
        **kwargs: Forwarded to :class:`Orbits` (``t0``, ``t_arr``, ``dt``,
            ``linear_interp_setup``; the latter defaults to ``True``).

    Raises:
        ImportError: When the ``mojito`` package is not installed.

    Example:
        .. code-block:: python

            from lisatools.detector import L1Orbits

            # noise_l1_path: a Mojito INSTRUMENT/L1 NOISE file
            orbits = L1Orbits(noise_l1_path, force_backend="cuda13x", frame="icrs")
            ltt_12 = orbits.get_light_travel_times(orbits.xp.asarray([1.0e7]), 12)
    """

    def __init__(
        self,
        filename: str,
        armlength: float = 2.5e9,
        force_backend: Optional[str] = None,
        frame: str = "icrs",
        linear_interp_dt: Optional[float] = 500.0,
        **kwargs,
    ):
        # frame is validated and stored by the Orbits base class (before
        # ``_setup`` runs, which reads ``self.frame`` to decide whether to
        # rotate the mojito ICRS positions to ecliptic).
        super().__init__(filename, armlength, force_backend, frame=frame, **kwargs)
        # Cap on the lazily-built position/unit-vector grid step -- see
        # :meth:`_configure`. Stored with the other grid arguments so the
        # lazy configuration picks it up.
        self._configure_kwargs["linear_interp_dt"] = linear_interp_dt

    @property
    def kwargs(self):
        """Keyword arguments for recreating this class instance."""
        return {
            "armlength": self.armlength,
            "force_backend": self.backend.backend_name.split("_")[-1],
            "frame": self.frame,
            "linear_interp_dt": self._configure_kwargs.get("linear_interp_dt", 500.0),
        }
        
    def open(self):
        """Open the Mojito L1 file.

        Returns:
            mojito.MojitoL1File: The opened file; use it as a context manager.

        Raises:
            ImportError: When ``mojito`` is not installed.
        """
        try:
            from mojito import MojitoL1File
        except ImportError:
            raise ImportError(
                "mojito package required for L1Orbits. Follow instructions at: https://mojito-e66317.io.esa.int/content/installation.html"
            )
        f = MojitoL1File(self.filename)
        return f

    def _setup(self):
        """Load orbit and LTT data from Mojito file.

        Reads the LTTs ``(N_ltt, 6)`` and their times, the positions and velocities
        ``(N_pos, 3, 3)`` (rotated to the ecliptic when ``frame == "ecliptic"``) and their
        times, and the per-grid steps and start times (``ltt_dt``, ``sc_dt``, ``dt_base``,
        ``ltt_t0``, ``sc_t0``).
        """

        with self.open() as f:
            # Load light travel times and their time array
            self.ltt = f.ltts.ltts[:]  # Shape: (N_ltt_times, 6) 
            self.ltt_t = f.ltts.time_sampling.t()  # Shape: (N_ltt_times,)
            
            # Load spacecraft positions and their time array
            pos_icrs = f.orbits.positions[:]  # Shape: (N_pos_times, 3, 3)
            vel_icrs = f.orbits.velocities[:]  # Shape: (N_pos_times, 3, 3)
            if self.frame == "ecliptic":
                # ICRS -> ecliptic is a pure barycentric rotation, so
                # velocities rotate with the same transformation as
                # positions (they must live in the same frame).
                self.x_base = icrs_to_ecliptic(pos_icrs)
                self.v_base = icrs_to_ecliptic(vel_icrs)
            else:
                self.x_base = pos_icrs
                self.v_base = vel_icrs
            self.sc_t_base = f.orbits.time_sampling.t()  # Shape: (N_pos_times,)
            self.size_base = self.sc_t_base.shape[0]
            self.dt_base = float(f.orbits.time_sampling.dt)

            # Store dt from each dataset (they may differ)
            self.ltt_dt = f.ltts.time_sampling.dt
            self.sc_dt = f.orbits.time_sampling.dt

            # Store t0 values
            self.ltt_t0 = float(self.ltt_t[0])
            self.sc_t0 = float(self.sc_t_base[0])

    @property
    def ltt_t(self):
        """LTT file time."""
        return self._ltt_t
    
    @ltt_t.setter
    def ltt_t(self, x):
        self._ltt_t = x
    
    @property
    def ltt(self):
        """Light travel times from Mojito file."""
        return self._ltt

    @ltt.setter
    def ltt(self, x):
        self._ltt = x
    
    @property
    def x_base(self):
        """Spacecraft positions from Mojito file."""
        return self._x_base
    
    @x_base.setter
    def x_base(self, x):
        self._x_base = x

    @property
    def v_base(self):
        """Velocities from Mojito file."""
        return self._v_base
    
    @v_base.setter
    def v_base(self, x):
        self._v_base = x

    @property
    def sc_t_base(self):
        """Spacecraft position file time."""
        return self._sc_t_base
    
    @sc_t_base.setter
    def sc_t_base(self, x):
        self._sc_t_base = x
    
    @property
    def ltt_dt(self):
        """Time step of LTT data."""
        return self._ltt_dt
    
    @ltt_dt.setter
    def ltt_dt(self, x):
        self._ltt_dt = x
    
    @property
    def sc_dt_base(self):    
        """Time step of spacecraft position data.

        Never assigned (:meth:`_setup` stores ``dt_base`` / ``sc_dt``), so reading it raises
        ``AttributeError``.
        """
        return self._sc_dt_base
    
    @sc_dt_base.setter
    def sc_dt_base(self, x):
        self._sc_dt_base = x
    
    @property
    def ltt_t0(self):
        """Start time of LTT data."""
        return self._ltt_t0
    
    @ltt_t0.setter
    def ltt_t0(self, x):
        self._ltt_t0 = x    

    @property
    def sc_t0(self):
        """Start time of spacecraft position data."""
        return self._sc_t0
    
    @sc_t0.setter
    def sc_t0(self, x):
        self._sc_t0 = x

    @property
    def sc_dt(self):    
        """Time step of spacecraft position data."""
        return self._sc_dt
    
    @sc_dt.setter
    def sc_dt(self, x):
        self._sc_dt = x

    @property
    def sc_t(self):
        """Configured spacecraft time array."""
        return self._sc_t
    
    @sc_t.setter
    def sc_t(self, x):
        self._sc_t = x
    
    @property
    def t(self):
        """Configured time array (spacecraft positions)."""
        return self._sc_t

    @property
    def x(self):
        """Configured spacecraft positions."""
        self._check_configured()
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = x
    
    @property
    def v(self):
        """Configured spacecraft velocities."""
        self._check_configured()
        return self._v
    
    @v.setter
    def v(self, x):
        self._v = x
    
    @property
    def n(self):
        """Configured normal unit vectors."""
        self._check_configured()
        return self._n
    
    @n.setter
    def n(self, x):
        self._n = x

    # ``pycppdetector`` override removed 2026-07-07: it was identical to the
    # base-class property except for a load-bearing typo
    # (``self._pycppdetect_args``) that would have raised AttributeError on
    # first use -- consumers build from ``pycppdetector_args`` directly, so
    # it never fired. The base property (with lazy configuration) is used.

    def _configure(self, t_arr=None, dt=None, linear_interp_setup=False,
                   linear_interp_dt=500.0):
        """Configure orbits with interpolation to a target time grid.

        Handles different time arrays for LTTs and positions in Mojito files.

        Positions and velocities are cubic-splined onto the target grid; each link unit
        vector is ``(x_rec(t) - x_emit(t - ltt(t))) / |...|`` with the LTT linearly
        interpolated from its dense native grid. Stores ``sc_t``, ``x``, ``v``, ``n`` (as
        backend arrays) and ``sc_dt``, marks the object configured and, unless the file's own
        position grid is used, fills :attr:`pycppdetector_args` with the position grid AND
        the native LTT grid.

        Args:
            t_arr (np.ndarray, optional): Target time array (if None, will be constructed).
                The backend arguments still start the position grid at the file's
                ``sc_t0`` with step ``sc_dt = dt`` (``None`` unless ``dt`` is also given), so
                an explicit ``t_arr`` must start at ``sc_t0`` with a constant step ``dt``.
            dt (float, optional): Target time step. Only read when ``linear_interp_setup`` is
                ``False`` and ``t_arr`` is ``None``; that branch reads ``self.t0_pos`` /
                ``self._pos_times``, attributes this class never sets, so it raises
                ``AttributeError``.
            linear_interp_setup (bool, optional): If True, create dense grid for fast linear
                interpolation. Default ``False`` here; the constructor stores ``True``.
            linear_interp_dt (float, optional): Cap (sec) on the position/unit-vector grid step
                used when ``linear_interp_setup`` is True. The C++ kernel
                interpolates spacecraft positions LINEARLY between grid nodes,
                so the grid must be fine enough that the linear-interp sag of
                the annual orbit, ``(dt^2 / 8) |r..| ~ (dt^2 / 8) * AU * (2pi/yr)^2``,
                is negligible as a light-travel time. Mojito L1 files carry
                positions at ``dt_base ~ 5.8 days`` -> sag ~ 0.13 light-sec ->
                a periodic Doppler phase error 2*pi*f0*0.13*cos(beta_ecl) rad
                that floored every template-vs-data mismatch at 1e-6..2e-4
                (measured on the VGB stream, 2026-07-07). At the 500 s default
                the sag is ~6e-7 light-sec (phase error < 1e-7 rad at 20 mHz;
                mm far below double precision) and the full-mission grid is
                still tiny (~1.3e5 points, built once). Pass ``None`` to use
                the file's native cadence unmodified (old behavior). Default ``500.0``.
        """

        # Determine target time array
        if linear_interp_setup:
            make_cpp = True
            # Positions / velocities / link unit vectors are cubic-splined
            # from the file's native grid onto this grid ONCE here; the C++
            # kernel then interpolates them linearly at evaluation time, so
            # the grid step sets the response fidelity (see linear_interp_dt
            # docstring). The dense light-travel-times keep their own native
            # grid (``ltt_dt``, ~2.5 s) and are handed to C++ separately below
            # — they are NOT splined here (the ``np.interp`` in the link loop
            # only samples them onto ``t_arr`` to build the unit vectors).
            # A full-mission grid at 3600 s is ~2e4 points — negligible; do
            # NOT use the raw TDI cadence (2.5 s, tens of millions of points).
            dt = float(self.dt_base)
            if linear_interp_dt is not None:
                dt = min(dt, float(linear_interp_dt))
            # interpolate only the orbit quantities, the ltts are already dense enough
            t0 = self.sc_t0
            t_end = float(self._sc_t_base[-1])

            t_arr = np.arange(t0, t_end + dt, dt)
            t_arr = t_arr[t_arr <= t_end]
            
        elif t_arr is None:
            if dt is not None:
                # Use position time range as reference
                t0 = self.t0_pos
                t_end = float(self._pos_times[-1])
                t_arr = np.arange(t0, t_end + dt, dt)
                t_arr = t_arr[t_arr <= t_end]
                make_cpp = True
            else:
                # Use position time array as default
                t_arr = self._sc_t_base.copy()
                make_cpp = False
        else:
            make_cpp = True
                
        # Interpolate positions from their native time grid to target grid
        # _pos_data is (N_pos, 3_sc, 3_xyz)

        pos_interp_shape = (len(t_arr), 3, 3)
        pos_interpolated = np.zeros(pos_interp_shape)
        vel_interpolated = np.zeros(pos_interp_shape)
        n_interpolated = np.zeros((len(t_arr), 6, 3))
        
        # Store position splines for later use in unit vector calculation
        pos_splines = [[None for _ in range(3)] for _ in range(3)]

        for isc in range(3):  # 3 spacecraft
            for icoord in range(3):  # x, y, z
                # Cubic spline interpolation
                cs = interpolate.CubicSpline(
                    self.sc_t_base, 
                    self.x_base[:, isc, icoord]
                )
                pos_splines[isc][icoord] = cs
                pos_interpolated[:, isc, icoord] = cs(t_arr)

                # interpolate velocities as well
                v_cs = interpolate.CubicSpline(self.sc_t_base, self.v_base[:, isc, icoord])
                vel_interpolated[:, isc, icoord] = v_cs(t_arr)

                del v_cs

        # Calculate unit vectors
        # Link order: 12, 23, 31, 13, 32, 21
        # indices: 0, 1, 2, 3, 4, 5
        
        # Retrieve rec and emit spacecraft indices (0-based)
        # self.link_space_craft_r/e are 1-based lists
        rec_indices = [x - 1 for x in self.link_space_craft_r]
        emit_indices = [x - 1 for x in self.link_space_craft_e]


        for i in range(6):
            rec_idx = rec_indices[i]
            emit_idx = emit_indices[i]

            # LTTs are stored at TDI cadence (~2.5 s), much denser than the
            # 600 s target grid — linear interpolation here matches the input
            # discretization and avoids a CubicSpline tridiagonal solve over
            # tens of millions of points (the dominant cost in this method).
            # cs_ltt = interpolate.CubicSpline(self.ltt_t, self.ltt[:, i])
            # ltt_i = cs_ltt(t_arr)
            ltt_i = np.interp(t_arr, self.ltt_t, self.ltt[:, i])
            # Emission time
            t_emit = t_arr - ltt_i
            
            # Emitter position at emission time
            pos_emit = np.zeros((len(t_arr), 3))
            for icoord in range(3):
                pos_emit[:, icoord] = pos_splines[emit_idx][icoord](t_emit)
            
            # Receiver position at reception time (already computed)
            pos_rec = pos_interpolated[:, rec_idx, :]
            
            # Vector from emitter to receiver
            vec = pos_rec - pos_emit
            
            # Normalize
            norm = np.linalg.norm(vec, axis=1)[:, None]
            n_interpolated[:, i, :] = vec / norm

        # Store interpolated data
        self.sc_dt = dt
        self.sc_t = self.xp.asarray(t_arr)
        self.x = self.xp.asarray(pos_interpolated)
        self.v = self.xp.asarray(vel_interpolated)
        self.n = self.xp.asarray(n_interpolated)

        # make sure base spacecraft and link information is ready
        lsr = np.asarray(self.link_space_craft_r).copy().astype(np.int32)
        lse = np.asarray(self.link_space_craft_e).copy().astype(np.int32)
        ll = np.asarray(self.LINKS).copy().astype(np.int32)
                
        # Mark as configured
        self.configured = True

        if make_cpp:
            self.pycppdetector_args = [
                self.sc_t0, # spacecraft time start
                self.sc_dt,   # spacecraft time step
                len(self.sc_t), # number of spacecraft time points
                self.ltt_t0,    # ltt time start
                self.ltt_dt,    # ltt time step
                len(self.ltt_t), # number of ltt time points
                self.xp.asarray(self.n.flatten().copy()),
                self.xp.asarray(self.ltt.flatten().copy()),
                self.xp.asarray(self.x.flatten().copy()),
                self.xp.asarray(ll),
                self.xp.asarray(lsr),
                self.xp.asarray(lse),
                self.armlength,
            ]
        else:
            self.pycppdetector_args = None

        # call garbage collection to free memory from large arrays no longer needed
        import gc
        gc.collect()

    # def _setup(self):
    #     """Override base class _setup - we load data in `_load_mojito_data` instead."""
    #     pass
    
if jax_here:
    # TODO check: keep/remove/document -- only caller is JAXL1Orbits.get_pos, and JAXL1Orbits has no callers
    @jax.jit
    def interpolate_pos(query_t, sc_idx, t_grid, pos_grid):
        """
        Interpolate spacecraft position using JAX.
        
        Args:
            query_t: shape (N,)
            sc_idx: shape (N,) or (1,) - 0-based index (0, 1, 2)
            t_grid: shape (T_dense,)
            pos_grid: shape (T_dense, 3_sc, 3_coords)
        """

        def _single_point(t, sc):
            # t: scalar float
            # sc: scalar int (0, 1, 2)
            
            # pos_grid is (Times, SC, 3)
            # We want to interpolate along axis 0.
            
            # Helper for 1D interp
            def interp_1d(vals):
                return jnp.interp(t, t_grid, vals)
                
            # Select the trajectory for this spacecraft
            # pos_grid[:, sc, 0] gives x(t) for spacecraft sc
            # We use simple indexing. Since 'sc' is a tracer in JIT, simple indexing might need dynamic_slice 
            # or just work if shape is static. JAX numpy indexing usually works fine.
            
            val_x = interp_1d(pos_grid[:, sc, 0])
            val_y = interp_1d(pos_grid[:, sc, 1])
            val_z = interp_1d(pos_grid[:, sc, 2])
            return jnp.array([val_x, val_y, val_z])

        # If inputs are arrays, we vmap.
        # query_t is (N,), sc_idx is (3,) or (1,)
        return jax.vmap(jax.vmap(_single_point, in_axes=(None, 0)), in_axes=(0, None))(
            query_t, sc_idx
        )

    # TODO check: keep/remove/document -- only caller is JAXL1Orbits.get_light_travel_times, and JAXL1Orbits has no callers
    @jax.jit
    def interpolate_ltt(query_t, link_idx, t_grid, ltt_grid):
        """
        Interpolate Light Travel Time using JAX.
        
        Args:
            query_t: shape (N,)
            link_idx: shape (N_links,) or (1,) - 0-based index
            t_grid: shape (T_dense,)
            ltt_grid: shape (T_dense, N_links)
        """

        def _single_point(t, l):
            vals = ltt_grid[:, l]
            return jnp.interp(t, t_grid, vals)

        return jax.vmap(jax.vmap(_single_point, in_axes=(None, 0)), in_axes=(0, None))(
            query_t, link_idx
        )

    # TODO check: keep/remove/document -- only caller is JAXL1Orbits.get_normal_unit_vec, and JAXL1Orbits has no callers
    @jax.jit
    def interpolate_n(query_t, link_idx, t_grid, n_grid):
        """
        Interpolate Normal Unit Vectors using JAX.

        Args:
            query_t: shape (N,)
            link_idx: shape (N_links,) or (1,) - 0-based index
            t_grid: shape (T_dense,)
            n_grid: shape (T_dense, N_links, 3_coords)
        """

        def _single_point(t, l):
            # Helper for 1D interp
            def interp_1d(vals):
                return jnp.interp(t, t_grid, vals)
            
            val_x = interp_1d(n_grid[:, l, 0])
            val_y = interp_1d(n_grid[:, l, 1])
            val_z = interp_1d(n_grid[:, l, 2])
            return jnp.array([val_x, val_y, val_z])

        return jax.vmap(jax.vmap(_single_point, in_axes=(None, 0)), in_axes=(0, None))(
            query_t, link_idx
        )

    # TODO check: keep/remove/document -- no callers in lisatools/tests/LATW (only a copy in mojito_detector.py); not reached by the 9mo/12mo runs
    @jax.tree_util.register_pytree_node_class
    class JAXL1Orbits(L1Orbits):
        """LISA Orbits from Mojito L1 File structure with JAX support.
        
        This class handles orbit data from MojitoL1File where:
        - Light travel times and positions have different time arrays
        - Both time arrays may start at t0 != 0
        
        Uses JAX for JIT-compiled interpolation on CPU or GPU.
        
        Args:
            filename: Path to Mojito L1 HDF5 file
            armlength: Armlength of detector (default 2.5e9 m)
            force_backend: If 'gpu' or 'cuda', use GPU; if 'cpu', use CPU
        """
        
        def _configure(self, t_arr=None, dt=None, linear_interp_setup=False,
                       linear_interp_dt=500.0):
            super()._configure(t_arr, dt, linear_interp_setup,
                               linear_interp_dt=linear_interp_dt)
            # No C++ backend for this implementation, ovverride attribute
            self._pycppdetector_args = None

            # move arrays to JAX
            self.sc_t = jnp.asarray(self.sc_t)
            self.x = jnp.asarray(self.x)
            self.v = jnp.asarray(self.v)
            self.n = jnp.asarray(self.n)

            self.ltt_t = jnp.asarray(self.ltt_t)
            self.ltt = jnp.asarray(self.ltt)
        
        @property
        def dt(self):
            """Time step of configured grid."""
            if not self.configured:
                return self._dt
            return self._dt

        @dt.setter
        def dt(self, dt):
            self._dt = dt

        def _map_link_to_index(self, link_arr):
            """Map link IDs to array indices.

            Args:
                link_arr: Array of link IDs (12, 23, 31, 13, 32, 21)

            Returns:
                Array of indices (0-5) corresponding to link positions
            """
            link_map_keys = jnp.array(self.LINKS)
            link_map_vals = jnp.arange(len(self.LINKS))

            def map_single_link(l):
                return jnp.sum(link_map_vals * (link_map_keys == l))

            return jax.vmap(map_single_link)(link_arr)

        def get_pos(self, t, sc):
            """Compute spacecraft position using JAX interpolation.

            Args:
                t: Time (scalar or array)
                sc: Spacecraft index (1, 2, or 3) or array of indices

            Returns:
                Spacecraft position(s) with shape (..., 3) for coordinates
            """
            if not self.configured:
                raise RuntimeError("Must call configure() before get_pos()")

            squeeze_t = jnp.isscalar(t)
            squeeze_sc = jnp.isscalar(sc)

            t_arr = jnp.atleast_1d(t)
            sc_arr = (jnp.atleast_1d(sc) - 1).astype(int)

            output = interpolate_pos(t_arr, sc_arr, self.sc_t, self.x)
            if squeeze_sc:
                output = output.squeeze(axis=1)
            if squeeze_t:
                output = output.squeeze(axis=0)

            return output

        def get_light_travel_times(self, t, link):
            """Compute light travel times using JAX interpolation.

            Args:
                t: Time (scalar or array)
                link: Link index (12, 23, 31, 13, 32, 21) or array of indices

            Returns:
                Light travel time(s)
            """
            if not self.configured:
                raise RuntimeError("Must call configure() before get_light_travel_times()")
            squeeze_t = jnp.isscalar(t)
            squeeze_link = jnp.isscalar(link)

            t_arr = jnp.atleast_1d(t)
            link_arr = jnp.atleast_1d(link)
            link_idx = self._map_link_to_index(link_arr)
            output = interpolate_ltt(t_arr, link_idx, self.ltt_t, self.ltt)

            if squeeze_link:
                output = output.squeeze(axis=1)
            if squeeze_t:
                output = output.squeeze(axis=0)

            return output

        def get_normal_unit_vec(self, t, link):
            """Compute normal unit vectors using JAX interpolation.

            Args:
                t: Time (scalar or array)
                link: Link index (12, 23, 31, 13, 32, 21) or array of indices

            Returns:
                Normal unit vector(s) with shape (..., 3) for coordinates
            """
            if not self.configured:
                raise RuntimeError("Must call configure() before get_normal_unit_vec()")
            squeeze_t = jnp.isscalar(t)
            squeeze_link = jnp.isscalar(link)

            t_arr = jnp.atleast_1d(t)
            link_arr = jnp.atleast_1d(link)
            link_idx = self._map_link_to_index(link_arr)
            output = interpolate_n(t_arr, link_idx, self.sc_t, self.n)

            if squeeze_link:
                output = output.squeeze(axis=1)
            if squeeze_t:
                output = output.squeeze(axis=0)

            return output

        def tree_flatten(self):
            # Collect children (JAX arrays)
            children = (
                self.ltt,
                self.ltt_t,
                self.x_base,
                self.v_base,
                self.sc_t_base,
            )

            # If configured, add interpolated arrays
            if self.configured:
                children += (self.sc_t, self.x, self.v, self.n)

            # Collect aux_data (static configuration)
            aux_data = {
                "filename": self.filename,
                "armlength": self._armlength,
                "configured": self.configured,
                "frame": getattr(self, "_frame", "icrs"),
                "ltt_dt": self.ltt_dt,
                "sc_dt": self.sc_dt,
                "ltt_t0": self.ltt_t0,
                "sc_t0": self.sc_t0,
                # Capture other potential attributes
                "_dt": getattr(self, "_dt", None),
                "_t0": getattr(self, "_t0", None),
                # Preserve backend info if needed (though usually static)
                "use_gpu": getattr(self, "use_gpu", False),
            }

            return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            # Create empty instance without calling __init__
            obj = object.__new__(cls)

            # Restore aux_data
            obj.filename = aux_data["filename"]
            obj._armlength = aux_data["armlength"]
            obj.configured = aux_data["configured"]
            obj._frame = aux_data.get("frame", "icrs")
            obj.ltt_dt = aux_data["ltt_dt"]
            obj.sc_dt = aux_data["sc_dt"]
            obj.ltt_t0 = aux_data["ltt_t0"]
            obj.sc_t0 = aux_data["sc_t0"]
            obj.use_gpu = aux_data["use_gpu"]
            obj._filename = aux_data["filename"]

            if aux_data.get("_dt") is not None:
                obj._dt = aux_data["_dt"]
            if aux_data.get("_t0") is not None:
                obj._t0 = aux_data["_t0"]

            # Restore children
            # Base arrays always present (first 5)
            obj.ltt = children[0]
            obj.ltt_t = children[1]
            obj.x_base = children[2]
            obj.v_base = children[3]
            obj.sc_t_base = children[4]
            if obj.configured:
                # Configured arrays (next 4)
                obj.sc_t = children[5]
                obj.x = children[6]
                obj.v = children[7]
                obj.n = children[8]

            # Initialize other attributes needed for method calls
            obj._pycppdetector_args = None

            return obj

else:

    # TODO check: keep/remove/document -- placeholder raising ImportError when JAX is absent; JAXL1Orbits has no callers
    class JAXL1Orbits(L1Orbits):
        """Placeholder for the JAX orbits when JAX is not installed: construction raises."""

        def __init__(self, *args, **kwargs):
            raise ImportError("JAX is not available. Install JAX to use JAXL1Orbits.")

class DefaultOrbits(EqualArmlengthOrbits):
    """Set default orbit class to Equal Armlength orbits for now.

    Same arguments as :class:`EqualArmlengthOrbits`. The stock :class:`LISAModel` objects
    (``scirdv1``, ``sangria``, ...) carry a ``DefaultOrbits()`` built at import.
    """

    pass


@dataclass
class CurrentNoises:
    """Noise values at a given frequency.

    Returned by :meth:`LISAModel.lisanoises` / :meth:`ExtendedLISAModel.lisanoises` and
    consumed by the TDI sensitivity classes of :mod:`lisatools.sensitivity`. Each field is a
    PSD (scalar or array matching the input frequencies) in ``units``.

    Args:
        isi_oms_noise: Interspacecraft OMS noise value.
        rfi_oms_noise: Reference interferometer OMS noise value.
        tmi_oms_noise: Test-mass interferometer OMS noise value.
        tm_noise: Test-mass acceleration noise value.
        rfi_backlink_noise: Reference interferometer backlink noise value.
        tmi_backlink_noise: Test-mass interferometer backlink noise value.
        units: Either ``"relative_frequency"`` (AKA fractional frequency deviation [ffd]) or
            ``"displacement"``.
    """

    isi_oms_noise: float
    rfi_oms_noise: float
    tmi_oms_noise: float
    tm_noise: float
    rfi_backlink_noise: float
    tmi_backlink_noise: float
    units: str


@dataclass
class LISAModelSettings:
    """Required LISA model settings.

    Args:
        Soms_d: OMS displacement noise level, SQUARED (m^2/Hz), e.g. ``(15e-12) ** 2``.
        Sa_a: Acceleration noise level, SQUARED (m^2 s^-4 / Hz), e.g. ``(3e-15) ** 2``.
        orbits: Orbital information.
        name: Name of model.
    """

    Soms_d: float
    Sa_a: float
    orbits: Orbits
    name: str


class LISAModel(LISAModelSettings, ABC):
    """Model for the LISA Constellation.

    This includes sensitivity information computed in
    :py:mod:`lisatools.sensitivity` and orbital information
    contained in an :class:`Orbits` class object.

    This class is used to house high-level methods useful
    to various needed computations. The two levels feed the OMS and test-mass terms of the
    TDI noise PSDs (:meth:`lisanoises`) and of the unequal-arm instrument noise, whose two
    levels the global fit's PSD branch samples. Stock instances: ``scirdv1``,
    ``proposal``, ``mrdv1``, ``sangria`` (pass the name as a string wherever a ``model`` is
    accepted, see :func:`check_lisa_model`).

    Args:
        Soms_d (float): OMS displacement noise level, squared (m^2/Hz).
        Sa_a (float): Acceleration noise level, squared (m^2 s^-4 / Hz).
        orbits (Orbits): Orbital information.
        name (str): Name of model.

    Example:
        >>> from lisatools.detector import LISAModel, DefaultOrbits
        >>> model = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "mine")
        >>> model.name
        'mine'
    """

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out

    def lisanoises(
        self,
        f: float | np.ndarray,
        unit: Optional[str] = "relative_frequency",
    ) -> CurrentNoises:
        r"""Calculate both LISA noise terms based on input model.

        .. math::

            S_{a} = S_{a,a} \left[1 + (0.4\,\mathrm{mHz}/f)^2\right]
            \left[1 + (f/8\,\mathrm{mHz})^4\right] (2 \pi f)^{-4},
            \qquad
            S_{oms} = S_{oms,d} \left[1 + (2\,\mathrm{mHz}/f)^4\right]

        (both in displacement), each multiplied by :math:`(2 \pi f / c)^2` for relative
        frequency.

        Args:
            f: Frequency array (Hz).
            unit: Either ``"relative_frequency"`` (default) or ``"displacement"``.

        Returns:
            CurrentNoises: ``isi_oms_noise`` = OMS term, ``tm_noise`` = acceleration term;
            the reference / test-mass interferometer and backlink fields are ``0.0``.

        Raises:
            UnboundLocalError: For any other ``unit`` (no explicit check).
        """

        # TODO: fix this up
        Soms_d_in = self.Soms_d
        Sa_a_in = self.Sa_a

        frq = f
        ### Acceleration noise
        ## In acceleration
        Sa_a = Sa_a_in * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
        ## In displacement
        Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
        ## In relative frequency unit
        Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
        Spm = Sa_nu

        ### Optical Metrology System
        ## In displacement
        Soms_d = Soms_d_in * (1.0 + (2.0e-3 / f) ** 4)
        ## In relative frequency unit
        Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
        Sop = Soms_nu

        # for mapping to more detailed noise setup
        if unit == "displacement":
            isi_oms_noise = Soms_d
            tm_noise = Sa_d

        elif unit == "relative_frequency":
            isi_oms_noise = Sop
            tm_noise = Spm

        # for mapping to more detailed noise setup
        rfi_oms_noise = 0.0
        tmi_oms_noise = 0.0
        rfi_backlink_noise = 0.0
        tmi_backlink_noise = 0.0

        return CurrentNoises(
            isi_oms_noise,
            rfi_oms_noise,
            tmi_oms_noise,
            tm_noise,
            rfi_backlink_noise,
            tmi_backlink_noise,
            unit,
        )


# defaults: stock LISAModel instances (squared OMS / acceleration levels).
# scirdv1 is assigned twice (identical values).
scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "scirdv1")
proposal = LISAModel((10.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "proposal")
mrdv1 = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "mrdv1")
sangria = LISAModel((7.9e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "sangria")
scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "scirdv1")


@dataclass
class ExtendedLISAModelSettings:
    """Required Extended LISA model settings.

    Levels are amplitude (ASD-like) values; :meth:`ExtendedLISAModel.lisanoises` squares
    them.

    Args:
        isi_oms_level: Interspacecraft OMS noise level (m/sqrt(Hz)).
        rfi_oms_level: Reference interferometer OMS noise level.
        tmi_oms_level: Test-mass interferometer OMS noise level.
        tm_noise_level: Test-mass acceleration noise level (formerly acceleration noise).
        rfi_backlink_noise_level: Reference interferometer backlink noise level.
        tmi_backlink_noise_level: Test-mass interferometer backlink noise level.
        orbits: Orbital information.
        name: Name of model.
    """

    isi_oms_level: float
    rfi_oms_level: float
    tmi_oms_level: float
    tm_noise_level: float  # formerly acceleration noise
    rfi_backlink_noise_level: float
    tmi_backlink_noise_level: float
    orbits: Orbits
    name: str


# TODO: verify this
# conversion factors into ffd units used in LDC
lamb = 1064.5e-9
nu0 = C_SI / lamb


class ExtendedLISAModel(ExtendedLISAModelSettings, ABC):
    """Model for the LISA Constellation with per-interferometer noise levels.

    This includes sensitivity information computed in
    :py:mod:`lisatools.sensitivity` and orbital information
    contained in an :class:`Orbits` class object.

    This class is used to house high-level methods useful
    to various needed computations. Only part of :mod:`lisatools.sensitivity` supports it
    (several TDI classes raise for an ``ExtendedLISAModel``). Stock instance:
    ``sangria_v2``.

    Args:
        isi_oms_level (float): Interspacecraft OMS noise level.
        rfi_oms_level (float): Reference interferometer OMS noise level.
        tmi_oms_level (float): Test-mass interferometer OMS noise level.
        tm_noise_level (float): Test-mass acceleration noise level.
        rfi_backlink_noise_level (float): Reference interferometer backlink noise level.
        tmi_backlink_noise_level (float): Test-mass interferometer backlink noise level.
        orbits (Orbits): Orbital information.
        name (str): Name of model.
    """

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out

    def disp_2_ffd(self, f: float | np.ndarray) -> float | np.ndarray:
        """Return the displacement-to-relative-frequency PSD factor ``(2 pi f / (lamb nu0))^2``.

        ``lamb`` = 1064.5 nm laser wavelength, ``nu0 = c / lamb``, so the factor equals
        ``(2 pi f / c)^2``.
        """
        return (2 * np.pi * f / lamb / nu0) ** 2

    def acc_2_ffd(self, f: float | np.ndarray) -> float | np.ndarray:
        """Return the acceleration-to-relative-frequency PSD factor ``(1 / (2 pi f c))^2``."""
        return (1 / (lamb * 2 * np.pi * f) / nu0) ** 2

    def lisanoises(
        self,
        f: float | np.ndarray,
        unit: Optional[str] = "relative_frequency",
        method: Optional[str] = "modern",
    ) -> CurrentNoises:
        """Calculate the per-interferometer LISA noise terms based on input model.

        Each level is squared; the test-mass term carries ``1 + (0.4 mHz / f)^2`` and the two
        backlink terms ``1 + (2 mHz / f)^4``. The ``"modern"`` and ``"old"`` methods are
        currently identical.

        Args:
            f: Frequency array (Hz).
            unit: Either ``"relative_frequency"`` (default; displacement terms times
                :meth:`disp_2_ffd`, test-mass term times :meth:`acc_2_ffd`) or
                ``"displacement"``.
            method: ``"modern"`` (default) or ``"old"``.

        Returns:
            CurrentNoises: All six noise terms at ``f``.

        Raises:
            ValueError: For an unknown ``unit``.
            UnboundLocalError: For an unknown ``method`` (no explicit check).
        """

        # BASED on code from Olaf Hartwig
        if method == "modern":
            isi_oms_noise = self.isi_oms_level**2 * f**0
            rfi_oms_noise = self.rfi_oms_level**2 * f**0
            tmi_oms_noise = self.tmi_oms_level**2 * f**0

            tm_noise = (self.tm_noise_level**2) * (1 + (0.4e-3 / f) ** 2)
            rfi_backlink_noise = self.rfi_backlink_noise_level**2 * (1.0 + (2.0e-3 / f) ** 4)
            tmi_backlink_noise = self.tmi_backlink_noise_level**2 * (1.0 + (2.0e-3 / f) ** 4)

        elif method == "old":
            isi_oms_noise = self.isi_oms_level**2 * f**0
            rfi_oms_noise = self.rfi_oms_level**2 * f**0
            tmi_oms_noise = self.tmi_oms_level**2 * f**0

            tm_noise = (self.tm_noise_level**2) * (1 + (0.4e-3 / f) ** 2)
            rfi_backlink_noise = self.rfi_backlink_noise_level**2 * (1.0 + (2.0e-3 / f) ** 4)
            tmi_backlink_noise = self.tmi_backlink_noise_level**2 * (1.0 + (2.0e-3 / f) ** 4)

        if unit == "displacement":
            return CurrentNoises(
                isi_oms_noise,
                rfi_oms_noise,
                tmi_oms_noise,
                tm_noise,
                rfi_backlink_noise,
                tmi_backlink_noise,
                unit,
            )
        elif unit == "relative_frequency":
            return CurrentNoises(
                isi_oms_noise * self.disp_2_ffd(f),
                rfi_oms_noise * self.disp_2_ffd(f),
                tmi_oms_noise * self.disp_2_ffd(f),
                tm_noise * self.acc_2_ffd(f),
                rfi_backlink_noise * self.disp_2_ffd(f),
                tmi_backlink_noise * self.disp_2_ffd(f),
                unit,
            )
        else:
            raise ValueError("unit kwarg must be 'displacement' or 'relative_frequency'.")


# defaults

# HERE we simulate the old LDC way of generating the sensitivity by pretending
# the rfi_backlink_noise, which has the same functionality of the OMS noise in the
# LDC code, is the oms noise.
sangria_v2 = ExtendedLISAModel(
    6.35e-12,
    3.32e-12,
    1.42e-12,
    2.4e-15,
    3.0e-12,
    3.0e-12,
    DefaultOrbits(),
    "sangria_v2",
)


__stock_list_models__ = [scirdv1, proposal, mrdv1, sangria, sangria_v2]
__stock_list_models_name__ = [tmp.name for tmp in __stock_list_models__]


def get_available_default_lisa_models() -> List[LISAModel]:
    """Get list of default LISA models.

    Returns:
        list: The stock models ``[scirdv1, proposal, mrdv1, sangria, sangria_v2]``
        (:class:`LISAModel`, plus the :class:`ExtendedLISAModel` ``sangria_v2``).

    Example:
        >>> from lisatools.detector import get_available_default_lisa_models
        >>> [m.name for m in get_available_default_lisa_models()]
        ['scirdv1', 'proposal', 'mrdv1', 'sangria', 'sangria_v2']
    """
    return __stock_list_models__


def get_default_lisa_model_from_str(model: str) -> LISAModel:
    """Return a LISA model from a ``str`` input.

    Args:
        model: Model indicated with a ``str``: one of ``"scirdv1"``, ``"proposal"``,
            ``"mrdv1"``, ``"sangria"``, ``"sangria_v2"``.

    Returns:
        LISAModel: LISA model associated to that ``str`` (an :class:`ExtendedLISAModel` for
        ``"sangria_v2"``).

    Raises:
        ValueError: If ``model`` is not a stock model name.

    Example:
        >>> from lisatools.detector import get_default_lisa_model_from_str
        >>> get_default_lisa_model_from_str("sangria").name
        'sangria'
    """
    if model not in __stock_list_models_name__:
        raise ValueError(
            "Requested string model is not available. See lisatools.detector documentation."
        )
    return globals()[model]


def check_lisa_model(model: Any) -> LISAModel:
    """Check input LISA model.

    Args:
        model: LISA model to check: a :class:`LISAModel`, an :class:`ExtendedLISAModel`, or
            a stock model name (see :func:`get_default_lisa_model_from_str`).

    Returns:
        LISAModel: LISA Model checked. Adjusted from ``str`` if ``str`` input.

    Raises:
        ValueError: If ``model`` is an unknown name or not a model object.

    Example:
        >>> from lisatools.detector import check_lisa_model
        >>> check_lisa_model("scirdv1").Soms_d == (15.0e-12) ** 2
        True
    """
    if isinstance(model, str):
        model = get_default_lisa_model_from_str(model)

    if not isinstance(model, LISAModel) and not isinstance(model, ExtendedLISAModel):
        raise ValueError("Model argument not given correctly.")

    return model