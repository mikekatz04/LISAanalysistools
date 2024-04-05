import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional
from dataclasses import dataclass
import requests

import numpy as np
import h5py
from scipy import interpolate

from lisatools.cutils.detector import pycppDetector

SC = [1, 2, 3]
LINKS = [12, 23, 31, 13, 32, 21]

LINEAR_INTERP_TIMESTEP = 600.00  # sec (0.25 hr)


class Orbits(ABC):
    """LISA Orbit Base Class

    Args:
        filename: File name. File should be in the style of LISAOrbits

    """

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._setup()
        self.configured = False

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
        """Set file name."""

        assert isinstance(filename, str)

        # get path
        path_to_this_file = __file__.split("detector.py")[0]

        # make sure orbit_files directory exists in the right place
        if not os.path.exists(path_to_this_file + "orbit_files/"):
            os.mkdir(path_to_this_file + "orbit_files/")
        path_to_this_file = path_to_this_file + "orbit_files/"

        if not os.path.exists(path_to_this_file + filename):
            # download files from github if they are not there
            github_file = f"https://github.com/mikekatz04/LISAanalysistools/raw/main/lisatools/orbit_files/{filename}"
            r = requests.get(github_file)

            # if not success
            if r.status_code != 200:
                raise ValueError(
                    f"Cannot find {filename} within default files located at github.com/mikekatz04/LISAanalysistools/lisatools/orbit_files/."
                )

            # write the contents to a local file
            with open(path_to_this_file + filename, "wb") as f:
                f.write(r.content)

        # store
        self._filename = path_to_this_file + filename

    def open(self) -> h5py.File:
        """Opens the h5 file in the proper mode.

        Returns:
            H5 file object: Opened file.

        Raises:
            RuntimeError: If backend is opened for writing when it is read-only.

        """
        f = h5py.File(self.filename, "r")
        return f

    @property
    def t_base(self) -> np.ndarray:
        """Time array from file."""
        with self.open() as f:
            t_base = np.arange(self.size_base) * self.dt_base
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
    def ltt(self) -> np.ndarray:
        """Light travel time."""
        self._check_configured()
        return self._ltt

    @ltt.setter
    def ltt(self, ltt: np.ndarray) -> np.ndarray:
        """Set light travel time."""
        assert ltt.shape[0] == len(self.t)

    @property
    def n(self) -> np.ndarray:
        """Normal vectors along links."""
        self._check_configured()
        return self._n

    @n.setter
    def n(self, n: np.ndarray) -> np.ndarray:
        """Set Normal vectors along links."""
        return self._n

    @property
    def x(self) -> np.ndarray:
        """Spacecraft positions."""
        self._check_configured()
        return self._x

    @x.setter
    def x(self, x: np.ndarray) -> np.ndarray:
        """Set Spacecraft positions."""
        return self._x

    @property
    def v(self) -> np.ndarray:
        """Spacecraft velocities."""
        self._check_configured()
        return self._v

    @v.setter
    def v(self, v: np.ndarray) -> np.ndarray:
        """Set Spacecraft velocities."""
        return self._v

    def configure(
        self,
        t_arr: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        linear_interp_setup: Optional[bool] = False,
    ) -> None:
        """Configure the orbits to match the signal response generator time basis.

        The base orbits will be scaled up or down as needed using Cubic Spline interpolation.
        The higherarchy of consideration to each keyword argument if multiple are given:
        ``linear_interp_setup``, ``t_arr``, ``dt``.

        If nothing is provided, the base points are used.

        Args:
            t_arr: New time array.
            dt: New time step. Will take the time duration to be that of the input data.
            linear_interp_setup: If ``True``, it will create a dense grid designed for linear interpolation with a constant time step.

        """

        x_orig = self.t_base

        if linear_interp_setup:
            make_cpp = True
            dt = LINEAR_INTERP_TIMESTEP
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])
        elif t_arr is not None:
            assert np.all(t_arr >= self.t_base[0]) and np.all(t_arr <= self.t_base[-1])

        elif dt is not None:
            make_cpp = True
            Tobs = self.t_base[-1]
            Nobs = int(Tobs / dt)
            t_arr = np.arange(Nobs) * dt
            if t_arr[-1] < self.t_base[-1]:
                t_arr = np.concatenate([t_arr, self.t_base[-1:]])

        else:
            make_cpp = False
            t_arr = self.t_base

        x_new = t_arr.copy()
        self.t = t_arr.copy()

        for which in ["ltt", "x", "n", "v"]:
            arr = getattr(self, which + "_base")
            arr_tmp = arr.reshape(self.size_base, -1)
            arr_out_tmp = np.zeros((len(x_new), arr_tmp.shape[-1]))
            for i in range(arr_tmp.shape[-1]):
                arr_out_tmp[:, i] = interpolate.CubicSpline(x_orig, arr_tmp[:, i])(
                    x_new
                )
            arr_out = arr_out_tmp.reshape((len(x_new),) + arr.shape[1:])
            setattr(self, "_" + which, arr_out)

        self.configured = True

        lsr = np.asarray(self.link_space_craft_r).copy().astype(np.int32)
        lse = np.asarray(self.link_space_craft_e).copy().astype(np.int32)
        ll = np.asarray(self.LINKS).copy().astype(np.int32)

        if make_cpp:
            self.pycppdetector_args = (
                dt,
                len(self.t),
                self.n.flatten().copy(),
                self.ltt.flatten().copy(),
                self.x.flatten().copy(),
                ll,
                lsr,
                lse,
            )
            self.dt = dt
        else:
            self.pycppdetector_args = None
            self.dt = dt

    @property
    def dt(self) -> float:
        """new time step if it exists"""
        if self._dt is None:
            raise ValueError("dt not available for t_arr only.")
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self._dt = dt

    @property
    def pycppdetector(self) -> pycppDetector:
        """C++ class"""
        if self._pycppdetector_args is None:
            raise ValueError(
                "Asking for c++ class. Need to set linear_interp_setup = True when configuring."
            )
        self._pycppdetector = pycppDetector(*self._pycppdetector_args)
        return self._pycppdetector

    @property
    def pycppdetector_args(self) -> tuple:
        """args for the c++ class."""
        return self._pycppdetector_args

    @pycppdetector_args.setter
    def pycppdetector_args(self, pycppdetector_args: tuple) -> None:
        self._pycppdetector_args = pycppdetector_args

    @property
    def size(self) -> int:
        """Number of time points."""
        self._check_configured()
        return len(self.t)

    def _check_configured(self) -> None:
        if not self.configured:
            raise ValueError(
                "Cannot request property. Need to use configure() method first."
            )

    def get_light_travel_times(
        self, t: float | np.ndarray, link: int
    ) -> float | np.ndarray:
        """Compute light travel time as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            link: which link. Must be ``in self.LINKS``.

        Returns:
            Light travel times.

        """
        return self.pycppdetector.get_light_travel_time(t, link)

    def get_normal_unit_vec(self, t: float | np.ndarray, link: int) -> np.ndarray:
        """Compute link normal vector as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            link: which link. Must be ``in self.LINKS``.

        Returns:
            Link normal vectors.

        """
        return self.pycppdetector.get_normal_unit_vec(t, link)

    def get_pos(self, t: float | np.ndarray, sc: int) -> np.ndarray:
        """Compute spacecraft position as a function of time.

        Computes with the c++ backend.

        Args:
            t: Time array in seconds.
            sc: which spacecraft. Must be ``in self.SC``.

        Returns:
            Spacecraft positions.

        """
        return self.pycppdetector.get_pos(t, sc)

    @property
    def ptr(self) -> int:
        """pointer to c++ class"""
        return self.pycppdetector.ptr


class EqualArmlengthOrbits(Orbits):
    """Equal Armlength Orbits

    Orbit file: equalarmlength-orbits.h5

    """

    def __init__(self):
        super().__init__("equalarmlength-orbits.h5")


class ESAOrbits(Orbits):
    """ESA Orbits

    Orbit file: esa-trailing-orbits.h5

    """

    def __init__(self):
        # TODO: fix this up
        super().__init__("esa-trailing-orbits.h5")


class DefaultOrbits(EqualArmlengthOrbits):
    """Set default orbit class to Equal Armlength orbits for now."""

    pass


@dataclass
class LISAModelSettings:
    """Required LISA model settings:

    Args:
        Soms_d: OMS displacement noise.
        Sa_a: Acceleration noise.
        orbits: Orbital information.
        name: Name of model.

    """

    Soms_d: float
    Sa_a: float
    orbits: Orbits
    name: str


class LISAModel(LISAModelSettings, ABC):
    """Model for the LISA Constellation

    This includes sensitivity information computed in
    :module:`lisatools.sensitivity` and orbital information
    contained in an :class:`Orbits` class object.

    This class is used to house high-level methods useful
    to various needed computations.

    """

    def __str__(self) -> str:
        out = "LISA Constellation Configurations Settings:\n"
        for key, item in self.__dict__.items():
            out += f"{key}: {item}\n"
        return out


# defaults
scirdv1 = LISAModel((15.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "scirdv1")
proposal = LISAModel((10.0e-12) ** 2, (3.0e-15) ** 2, DefaultOrbits(), "proposal")
mrdv1 = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "mrdv1")
sangria = LISAModel((10.0e-12) ** 2, (2.4e-15) ** 2, DefaultOrbits(), "sangria")

__stock_list_models__ = [scirdv1, proposal, mrdv1, sangria]
__stock_list_models_name__ = [tmp.name for tmp in __stock_list_models__]


def get_available_default_lisa_models() -> List[LISAModel]:
    """Get list of default LISA models

    Returns:
        List of LISA models.

    """
    return __stock_list_models__


def get_default_lisa_model_from_str(model: str) -> LISAModel:
    """Return a LISA model from a ``str`` input.

    Args:
        model: Model indicated with a ``str``.

    Returns:
        LISA model associated to that ``str``.

    """
    if model not in __stock_list_models_name__:
        raise ValueError(
            "Requested string model is not available. See lisatools.detector documentation."
        )
    return globals()[model]


def check_lisa_model(model: Any) -> LISAModel:
    """Check input LISA model.

    Args:
        model: LISA model to check.

    Returns:
        LISA Model checked. Adjusted from ``str`` if ``str`` input.

    """
    if isinstance(model, str):
        model = get_default_lisa_model_from_str(model)

    if not isinstance(model, LISAModel):
        raise ValueError("model argument not given correctly.")

    return model
