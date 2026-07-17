"""
Preprocessing module for loading and handling Mojito L1 files.

Credits for most of the signal processing functions to Ollie Burke and Martina Muratore for the repository: https://github.com/OllieBurke/mojito-noise-sprint/blob/main/Mojito_Sprint/
"""

import dataclasses
import logging
import os
from typing import Callable, Optional, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mojito import MojitoL1File
from MojitoProcessor import SignalProcessor as MPSignalProcessor
from tqdm import tqdm

from ..domains import DomainBase
from ..detector import L1Orbits, Orbits
from ..domains import (
    DomainBase,
    DomainSettingsBase,
    FDSettings,
    TDSettings,
    TDSignal,
    place_td_signal_on_grid,
)
from ..utils.utility import get_array_module

logger = logging.getLogger(__name__)

level = logging.INFO
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # * Prevent duplicate messages*
    logger.propagate = False

ALLOWED_SOURCES = ["NOISE", "GB", "VGB", "MBHB", "EMRI", "SOBHB"]


def find_file(folder: str, source_type: str, source_id: int) -> str:
    """
    Find the Mojito L1 file for a given source type and ID in the specified folder.
    It avoids passing through all the symbols in the name (for example, NOISE_731d_2.5s_L1_source0_0_20251206T220508924302Z.h5)

    Args:
        folder (str): The folder where Mojito L1 files are stored.
        source_type (str): The type of the source (e.g., 'GB' 'MBHB', etc.).
        source_id (int): The identifier of the source.

    Returns:
        str: The path to the found Mojito L1 file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    for filename in os.listdir(folder):
        if filename.startswith(f"{source_type}_") and f"source{source_id}_" in filename:
            return os.path.join(folder, filename)

    raise FileNotFoundError(
        f"No Mojito L1 file found for source type '{source_type}' and source ID '{source_id}' in folder '{folder}'."
    )


def normalize_source_ids(d: dict) -> dict:
    """Coerce a per-class source-id mapping to lists.

    Each value is normalized so ``None -> []``, a scalar ``v -> [v]``, and a
    list/tuple is returned as a ``list``. Used by settings recipes to build a
    uniform ``{source_type: [ids...]}`` mapping from user-edited literals /
    environment overrides.
    """
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = []
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = [v]
    return out


class L1DataLoader:
    """
    Class to load L1 files.

    Parameters
    ----------
    L1_folder : str
        Path to the folder containing Mojito L1 files. Downloading the files from the brickmarket, the folder should contain two subfolders: 'data' and 'catalogues'.
    source_types : list
        List of source types to load (e.g., ['NOISE', 'GB']).
    source_ids : dict
        Dictionary mapping source types to lists of source IDs to load.
    orbits_class : Orbits, optional
        Class to use for orbits. Default is L1Orbits.
    orbits_kwargs : dict, optional
        Additional keyword arguments for the orbits class.
    verbose : bool, optional
        Verbosity flag. Default is True.
    store_individual_timeseries : bool, optional
        Whether to store individual timeseries for each source type and ID. Default is False.
    """

    def __init__(
        self,
        L1_folder: str,
        source_types: list,
        source_ids: dict = None,
        orbits_class: Orbits = L1Orbits,
        orbits_kwargs: dict = None,
        verbose: bool = True,
        store_individual_timeseries: bool = False,  # whether to store individual timeseries for each source type and ID
    ):
        self.data_folder = os.path.join(L1_folder, "data")
        self.catalogues_folder = os.path.join(L1_folder, "catalogues")

        self.catalogue = {}

        source_types = [st.upper() for st in source_types]

        for source_type in source_types:
            if source_type not in ALLOWED_SOURCES:
                raise ValueError(
                    f"Source type '{source_type}' is not allowed. Allowed types are: {ALLOWED_SOURCES}"
                )

        self.source_types = source_types

        # uppercase the keys of source_ids
        if source_ids is not None:
            source_ids = {k.upper(): v for k, v in source_ids.items()}
        self.source_ids = source_ids

        self.orbits_class = orbits_class
        self.orbits_kwargs = orbits_kwargs
        self.verbose = verbose
        self.store_individual_timeseries = store_individual_timeseries
        if self.verbose:
            logger.info(f"L1DataLoader initialized with data folder: {self.data_folder}")
            logger.info(f"Source types to load: {self.source_types}")
            logger.info(f"Source IDs: {self.source_ids}")

            logger.info(f"Orbits class: {self.orbits_class.__name__}")
            if self.orbits_kwargs:
                logger.info(f"Orbits kwargs: {self.orbits_kwargs}")

    @property
    def catalogues_map(self) -> dict:
        """
        Property to get the mapping of source types to their catalogue file names.

        Returns:
            dict: Mapping of source types to catalogue file names.
        """
        return {
            "GB": "wdwd_cat_mojito_lite_processed.hdf5",
            "VGB": "vgb_cat_mojito_lite_processed.hdf5",
            "MBHB": "mbhb_cat_mojito_lite_processed_MT_rounding_fixed.hdf5",
            "EMRI": "emri_cat_mojito_lite_processed_MT.hdf5",
            "SOBHB": "sobhb_cat_mojito_lite_processed_MT.hdf5",
        }

    def load_single_binary(self, group: h5py.Group, binary_id: int, source_type: str) -> dict:
        """
        Load a single binary source from the given HDF5 group.

        Args:
            group (h5py.Group): The HDF5 group containing binary sources.
            binary_id (int): The identifier of the binary source to load.

        Returns:
            dict: A dictionary containing the parameters of the binary source.
        """
        params = {}
        for key in group.keys():
            try:
                if source_type in ["VGB", "GB"]:
                    ds = group[key][:]
                else:
                    ds = group[key][binary_id]

                if ds.dtype.kind == "O":  # object type (e.g. variable-length strings)
                    val = ds
                    if isinstance(val, bytes):
                        val = val.decode()
                else:
                    val = ds[()]
                params[key] = val
            except Exception as e:
                raise e
        return params

    def _open(self, file_path: str) -> MojitoL1File:
        raise NotImplementedError("_open method should be implemented in subclasses.")

    @property
    def individual_timeseries(self) -> dict | None:
        """
        Property to access individual timeseries for each source type and ID.

        Returns:
            dict: A dictionary containing individual timeseries for each source type and ID.
        """
        if not hasattr(self, "_individual_timeseries"):
            return None
        return self._individual_timeseries

    def load_data(self) -> tuple:
        """
        Load data from L1 files based on the specified source types and IDs.

        Returns:
            tuple: A tuple containing:
                - tdi_times (np.ndarray): Array of time samples.
                - tdi_fs (float): Sampling frequency in Hz.
                - xyz (np.ndarray): Array of channel data with shape (n_channels, n_times).
                - orbits (Orbits): An instance of the Orbits class initialized with the loaded data.
        """
        xyz = None
        orbits: Orbits = None

        if self.store_individual_timeseries:
            _individual_timeseries = (
                {}
            )  # to store individual timeseries for each source type and ID, if needed for debugging or further analysis.

        import time as _t
        if "NOISE" in self.source_types:
            subfolder = os.path.join(self.data_folder, "INSTRUMENT", "L1")
            file_path = find_file(subfolder, "NOISE", 00)

            if orbits is None:
                _t0 = _t.perf_counter()
                orbits = self.orbits_class(file_path, **(self.orbits_kwargs or {}))
                logger.debug(f"[load_data] L1Orbits(...) (incl. _setup) took {_t.perf_counter()-_t0:.2f}s")
                _t0 = _t.perf_counter()
                # Eagerly build the interpolation grid here (data-load time is
                # the right place to pay the one-time cost); equivalent to the
                # lazy first-use configuration.
                orbits._ensure_configured()
                logger.debug(f"[load_data] orbits._ensure_configured() took {_t.perf_counter()-_t0:.2f}s")
                logger.info(f"Initialized orbits from NOISE file.")

            with self._open(file_path) as f:
                _t0 = _t.perf_counter()
                xyz = f.tdis.xyz_doppler[:]
                logger.debug(f"[load_data] NOISE f.tdis.xyz_doppler[:] took {_t.perf_counter()-_t0:.2f}s shape={xyz.shape}")

                tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                tdi_times = f.tdis.time_sampling.t()

                if self.store_individual_timeseries:
                    noise_covariance = f.noise_estimates.xyz[:] / (f.laser_frequency**2)
                    noise_frequencies = f.noise_estimates.freq_sampling.f()
                    noise_times = f.noise_estimates.time_sampling.t()

                    _individual_timeseries["NOISE"] = (
                        xyz.T.copy()
                    )  # store the noise timeseries separately if needed
                    _individual_timeseries["PSD_MATRIX"] = noise_covariance
                    _individual_timeseries["PSD_FREQUENCIES"] = noise_frequencies
                    _individual_timeseries["PSD_TIMES"] = noise_times

            self.source_types.remove("NOISE")

            if self.verbose:
                logger.info(f"Loaded NOISE data from {file_path}")
                logger.info(f"data, times and orbits initialized from NOISE file.")
                logger.info(f"TDI time step: {tdi_dt} seconds")
                logger.info(f"TDI sampling frequency: {tdi_fs} Hz")

        for source_type in self.source_types:

            subfolder = os.path.join(self.data_folder, source_type, "L1")

            if source_type in ["GB", "VGB"]:
                ids = [0]  # only one file for GB/VGB
            else:
                ids = self.source_ids.get(source_type, [])

            if not isinstance(ids, list):
                ids = [ids]
            if len(ids) == 0:
                raise ValueError(f"No source IDs provided for source type '{source_type}'.")

            catalogue_path = os.path.join(
                self.catalogues_folder, self.catalogues_map.get(source_type)
            )

            self.catalogue[source_type] = {}

            if self.verbose:
                logger.info(f"Loading data for source type '{source_type}'")

            with h5py.File(catalogue_path, "r") as catalogue_file:
                binary_params = catalogue_file["Binaries"]
                for source_id in tqdm(
                    ids, desc=f"Loading {source_type} sources", disable=not self.verbose
                ):
                    file_path = find_file(subfolder, source_type, source_id)

                    self.catalogue[source_type][source_id] = self.load_single_binary(
                        binary_params, source_id, source_type
                    )
                    if self.verbose:
                        logger.info(
                            f"Loaded catalogue parameters for {source_type} source ID {source_id} from catalogue."
                        )

                    with self._open(file_path) as f:
                        _t0 = _t.perf_counter()
                        _xyz = f.tdis.xyz_doppler[:]
                        logger.debug(f"[load_data] {source_type} source {source_id} f.tdis.xyz_doppler[:] took {_t.perf_counter()-_t0:.2f}s shape={_xyz.shape}")
                        _tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                        _tdi_times = f.tdis.time_sampling.t()
                        _tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                        if xyz is None:
                            xyz = _xyz
                            tdi_dt = _tdi_dt
                            tdi_times = _tdi_times
                            tdi_fs = _tdi_fs

                            if orbits is None:
                                orbits: Orbits = self.orbits_class(
                                    file_path, **(self.orbits_kwargs or {})
                                )  # configures lazily on first use
                                logger.info(f"Initialized orbits from {source_type} file.")

                        else:
                            xyz += _xyz
                            assert tdi_dt == _tdi_dt, "Time steps do not match between files."
                            assert tdi_fs == _tdi_fs, "Sampling frequencies do not match between files."
                            assert (
                                tdi_times == _tdi_times
                            ).all(), "Time arrays do not match between files."

                        if self.store_individual_timeseries:
                            _individual_timeseries[f"{source_type}_{source_id}"] = (
                                _xyz.T.copy()
                            )  # store individual timeseries for this source

        xyz = xyz.T  # Transpose to have shape (n_channels, n_times)
        assert (
            xyz.shape[1] == tdi_times.shape[0]
        ), "Data time dimension does not match time array length."

        assert orbits is not None, "Orbits were not initialized from any file."

        if self.store_individual_timeseries:
            self._individual_timeseries = (
                _individual_timeseries  # store the individual timeseries for potential further use
            )

        return tdi_times, tdi_fs, xyz, orbits

    def dump_individual_timeseries(self, file_path: str, delete_after_dump: bool = False):
        """
        Dump the individual timeseries for each source type and ID to a .h5 file for debugging or further analysis.

        Args:
            file_path (str): The path to the output .h5 file.
            delete_after_dump (bool, optional): Whether to delete the individual timeseries after dumping them. Defaults to False.
        """
        with h5py.File(file_path, "w") as f:
            for key, ts_data in self.individual_timeseries.items():
                f.create_dataset(key, data=ts_data)
        if self.verbose:
            logger.info(f"Dumped individual timeseries to {file_path}")

        if delete_after_dump:
            self.individual_timeseries.clear()

    def dump_catalogue(self, file_path: str):
        """
        Dump the loaded catalogue parameters to a .h5 file for debugging or further analysis.

        Args:
            file_path (str): The path to the output .h5 file.
        """
        with h5py.File(file_path, "w") as f:
            for source_type, sources in self.catalogue.items():
                grp = f.create_group(source_type)
                for source_id, params in sources.items():
                    subgrp = grp.create_group(f"source_{source_id}")
                    for param_key, param_value in params.items():
                        if isinstance(param_value, str):
                            param_value = np.string_(param_value)  # convert to bytes for h5py
                        subgrp.create_dataset(param_key, data=param_value)
        if self.verbose:
            logger.info(f"Dumped catalogue parameters to {file_path}")


class SangriaDataLoader:
    """
    Class to load data injections from Sangria HDF5 file
    """

    def __init__(
        self,
        data_input_path: str,
        remove_from_data: Optional[list] = None,
    ):
        self.data_input_path = data_input_path
        self.remove_from_data = remove_from_data

    def _open(self, file_path: str) -> h5py.File:
        return h5py.File(file_path, "r")

    def load_data(
        self,
    ):
        if self.remove_from_data is None:
            self.remove_from_data = []

        assert isinstance(self.remove_from_data, list)

        # TODO: Generalize input.
        with self._open(self.data_input_path) as f:
            if "noise" not in self.remove_from_data:
                tXYZ = f["obs"]["tdi"][:]

                # remove sources
                for source in self.remove_from_data:  # , "dgb", "igb"]:  # "vgb" ,
                    if source == "noise":
                        continue
                    logger.info(f"Removing {source} from data injection.")
                    change_arr = f["sky"][source]["tdi"][:]
                    for change in ["X", "Y", "Z"]:
                        tXYZ[change] -= change_arr[change]

            else:
                keys = list(f["sky"])
                logger.info("Initial keys in data injection: %s", keys)
                tmp_keys = keys.copy()
                for key in tmp_keys:
                    logger.debug(key)
                    if key in self.remove_from_data:
                        keys.remove(key)
                        logger.info(f"Removing {key} from data injection.")
                tXYZ = f["sky"][keys[0]]["tdi"][:]
                for key in keys[1:]:
                    tXYZ["X"] += f["sky"][key]["tdi"][:]["X"]
                    tXYZ["Y"] += f["sky"][key]["tdi"][:]["Y"]
                    tXYZ["Z"] += f["sky"][key]["tdi"][:]["Z"]

        times, X, Y, Z = (
            tXYZ["t"].squeeze(),
            tXYZ["X"].squeeze(),
            tXYZ["Y"].squeeze(),
            tXYZ["Z"].squeeze(),
        )
        data_xyz = np.vstack([X, Y, Z])  # shape (n_channels, n_times)
        dt = times[1] - times[0]
        fs = 1.0 / dt

        return times, fs, data_xyz, None  # no orbits available


# todo: add catalogue loading functionality for signal parameters.


class SignalProcessor:
    """
    Class to preprocess time-domain signals. Credits: Ollie Burke.

    Parameters
    ----------
    times : np.ndarray
        Array of time samples. Shape: (n_times,)
    data : np.ndarray
        Array of channel data. Shape: (n_channels, n_times)
    fs : float
        Sampling frequency in Hz
    verbose : bool, optional
        Verbosity flag. Default is True.
    do_plots : bool, optional
        Whether to generate plots during processing steps. Default is False.
    """

    _CHANNEL_NAMES = ["X", "Y", "Z"]

    def __init__(
        self,
        times: np.ndarray,
        data: np.ndarray,
        fs: float,
        verbose: bool = True,
        do_plots: bool = False,
    ):

        self.times = times
        self.dt = times[1] - times[0]
        self.N = times.shape[0]
        self.T = self.N * self.dt # total duration. Takes into account that times[0] may not be zero.
        self.data = data
        self.fs = fs
        self.verbose = verbose
        self.do_plots = do_plots

        self._original_t0 = float(times[0])  # store original t0 for reference in case of trimming

    @property
    def original_t0(self) -> float:
        """Original start time of the data before any trimming."""
        return self._original_t0

    def _to_mp_dict(self):
        """ndarray (n_ch, N) -> {'X': arr, 'Y': arr, 'Z': arr}"""
        return {ch: self.data[i] for i, ch in enumerate(self._CHANNEL_NAMES[: self.data.shape[0]])}

    def _from_mp_dict(self, mp_data):
        """{'X': arr, 'Y': arr, 'Z': arr} -> ndarray (n_ch, N)"""
        return np.vstack([mp_data[ch] for ch in self._CHANNEL_NAMES if ch in mp_data])

    def _filter_via_mp(
        self, *, low=None, high=None, order=2, filter_type="butterworth", zero_phase=True
    ):
        mp = MPSignalProcessor(self._to_mp_dict(), fs=self.fs)
        mp.filter(low=low, high=high, order=order, filter_type=filter_type, zero_phase=zero_phase)
        return self._from_mp_dict(mp.data)

    def _plot_filter(self, filtered, btype, order, filter_type, **kwargs):
        """Plot original vs filtered X channel."""
        decimate_factor = kwargs.get("decimate_factor", 1000)
        plt.figure(figsize=(10, 4))
        plt.plot(
            self.data[0, ::decimate_factor],
            label="Original X channel",
            c="gray",
            alpha=0.5,
        )
        plt.plot(filtered[0, ::decimate_factor], label="Filtered X channel", c="blue")
        plt.xlabel("Sample Index")
        plt.ylabel("Strain")
        plt.title(f"Signal {btype.capitalize()} Filtering ({filter_type}, order={order})")
        plt.legend()
        plt.savefig(os.path.join(kwargs.get("plot_folder", "."), f"{btype}_filter_signal.png"))
        plt.close()

    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply bandpass filter to all channels.

        Parameters
        ----------
        low : float
            Lower cutoff frequency in Hz
        high : float
            Upper cutoff frequency in Hz
        order : int, optional
            Filter order (default: 2)
        filter_type : str, optional
            Filter type: 'butterworth', 'chebyshev1', 'chebyshev2', 'bessel'
            (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering (filtfilt) if True, else single-pass (default: True)
        **kwargs: Additional keyword arguments for the eventual plot.

        Returns
        -------
        filtered_data : np.ndarray
            Array of filtered channel data
        """
        filtered = self._filter_via_mp(
            low=low, high=high, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "bandpass", order, filter_type, **kwargs)
        return filtered

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply lowpass filter to all channels.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz
        order : int, optional
            Filter order (default: 2)
        filter_type : str, optional
            Filter type (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering if True (default: True)
        **kwargs: Additional keyword arguments for the eventual plot.

        Returns
        -------
        filtered_data : np.ndarray
            Array of filtered channel data
        """
        filtered = self._filter_via_mp(
            high=cutoff, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "lowpass", order, filter_type, **kwargs)
        return filtered

    def highpass_filter(
        self,
        cutoff: float,
        order: int = 2,
        filter_type: str = "butterworth",
        zero_phase: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply highpass filter to all channels.

        Parameters
        ----------
        cutoff : float
            Cutoff frequency in Hz
        order : int, optional
            Filter order (default: 2)
        filter_type : str, optional
            Filter type (default: 'butterworth')
        zero_phase : bool, optional
            Use zero-phase filtering if True (default: True)
        **kwargs: Additional keyword arguments for the eventual plot.

        Returns
        -------
        filtered_data : np.ndarray
            Array of filtered channel data
        """
        filtered = self._filter_via_mp(
            low=cutoff, order=order, filter_type=filter_type, zero_phase=zero_phase
        )
        if self.do_plots:
            self._plot_filter(filtered, "highpass", order, filter_type, **kwargs)
        return filtered

    def _update_params(self):
        """Update internal parameters after data modification."""
        self.N = self.data.shape[1]
        self.T = self.times[-1] - self.times[0]

        if self.verbose:
            logger.info(f"Updated parameters: N={self.N}, T={self.T}")

    def downsample(self, target_fs, window=("kaiser", 31.0), padtype="line"):
        """Downsample all channels to target_fs via polyphase filtering.

        Delegates to MojitoProcessor.SignalProcessor.downsample().
        Updates self.data, self.fs, self.times and derived params (N, T, dt).

        Parameters
        ----------
        target_fs : float
            Target sampling frequency in Hz.
        window : tuple, optional
            Window for the anti-aliasing filter (default: ('kaiser', 31.0)).
        padtype : str, optional
            Padding type for the resampling (default: 'line').

        Returns
        -------
        np.ndarray
            Downsampled data array.
        """
        if target_fs == self.fs:
            if self.verbose:
                logger.info(
                    f"Target sampling frequency is the same as current fs ({self.fs} Hz). No downsampling applied."
                )
            return self.data
        elif target_fs > self.fs:
            raise ValueError(
                f"Target sampling frequency ({target_fs} Hz) must be less than current fs ({self.fs} Hz) for downsampling."
            )

        mp = MPSignalProcessor(self._to_mp_dict(), fs=self.fs)
        mp.downsample(target_fs=target_fs, window=window, padtype=padtype)
        self.data = self._from_mp_dict(mp.data)
        self.fs = mp.fs
        self.dt = 1.0 / self.fs
        self.times = self.times[0] + np.arange(self.data.shape[1]) * self.dt
        self._update_params()
        return self.data

    def trim(
        self,
        duration: float,
        is_percent: bool = False,
        trimming_type: str = "from_each_end",
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Trim data by removing specified duration.

        Parameters
        ----------
        duration : float
            Duration to remove in seconds or percent of total duration.
        is_percent : bool, optional
            If True, `duration` is treated as a percentage of total duration (0-1).
            If False, `duration` is in seconds (default: False)
        trimming_type : str, optional
            Type of trimming: 'from_each_end', 'keep_from_start', or 'discard_from_start' (default: 'from_each_end')
        **kwargs: Additional keyword arguments for the eventual plot.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Trimmed time and data arrays
        """

        assert trimming_type in [
            "from_each_end",
            "keep_from_start",
            "discard_from_start",
        ]  # todo find proper name and implement

        if is_percent:
            assert 0 <= duration <= 1, "Percentage duration must be between 0 and 1."
            duration = duration * self.T

        trim_samples = int(duration / self.dt)

        if trimming_type == "from_each_end":
            if 2 * trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {2*duration}s from {self.T}s data. "
                    f"Total trim exceeds data length."
                )
            if self.verbose:
                logger.info(
                    f"Trimming {duration}s ({trim_samples} samples) from each end of the data."
                )

            trimmed_data = self.data[:, trim_samples:-trim_samples]
            trimmed_times = self.times[trim_samples:-trim_samples]

        elif trimming_type == "discard_from_start":
            if trim_samples >= self.N:
                raise ValueError(
                    f"Cannot trim {duration}s from {self.T}s data. " f"Trim exceeds data length."
                )
            if self.verbose:
                logger.info(
                    f"Trimming {duration}s ({trim_samples} samples) from the start of the data."
                )

            trimmed_data = self.data[:, trim_samples:]
            trimmed_times = self.times[trim_samples:]

        else:  # keep_from_start
            if trim_samples >= self.N:
                raise ValueError(
                    f"Cannot keep {duration}s from {self.T}s data. "
                    f"Keep duration exceeds data length."
                )
            if self.verbose:
                logger.info(f"Keeping first {duration}s ({trim_samples} samples) of the data.")

            trimmed_data = self.data[:, :trim_samples]
            trimmed_times = self.times[:trim_samples]

        if self.do_plots:
            decimate_factor = kwargs.get(
                "decimate_factor", 1000
            )  # Decimate for plotting if too many points
            plt.figure(figsize=(10, 4))
            plt.plot(
                self.times[::decimate_factor],
                self.data[0, ::decimate_factor],
                label="Original X channel",
                c="gray",
                alpha=0.5,
            )
            plt.plot(
                trimmed_times[::decimate_factor],
                trimmed_data[0, ::decimate_factor],
                label="Trimmed X channel",
                c="blue",
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Strain")
            plt.title("Signal Trimming")
            plt.legend()
            plt.savefig(
                os.path.join(
                    kwargs.get("plot_folder", "."),
                    kwargs.get("filename", "trim_signal.png"),
                )
            )
            plt.close()

        # Update internal state
        self.data = trimmed_data
        self.times = trimmed_times
        self._update_params()

        return trimmed_times, trimmed_data


class BaseProcessingStep(SignalProcessor):
    """:class:`SignalProcessor` extended with :meth:`process` and :meth:`pour` helpers.

    Subclasses inherit a high-level :meth:`process` that runs the configured
    sequence of filters / downsamples / trims and a :meth:`pour` that turns
    the conditioned time-domain data into a :class:`DataResidualArray`.
    """

    def __init__(self, *args, **kwargs):
        SignalProcessor.__init__(self, *args, **kwargs)

    def __repr__(self):
        return (
            f"ProcessingStep with {self.N} samples, "
            f"dt={self.dt:.3e}s, T={self.T:.3e}s, "
            f"fs={self.fs:.3e}Hz, "
            f"and channels: {self.data.shape[0]}"
        )

    def process(
        self,
        highpass_kwargs: dict = None,
        lowpass_kwargs: dict = None,
        bandpass_kwargs: dict = None,
        downsample_kwargs: dict = None,
        trim_kwargs: dict = None,
        Tobs: float = None,
        **kwargs,
    ) -> tuple:
        """
        Apply preprocessing steps to the loaded data. for each step, if the corresponding kwargs is None, the step is skipped.

        Args:
            highpass_kwargs (dict, optional): Keyword arguments for highpass_filter method.
            lowpass_kwargs (dict, optional): Keyword arguments for lowpass_filter method.
            bandpass_kwargs (dict, optional): Keyword arguments for bandpass_filter method.
            downsample_kwargs (dict, optional): Keyword arguments for downsample method.
                Applied after filtering and before trimming. Example: dict(target_fs=1.0).
            trim_kwargs (dict, optional): Keyword arguments for trim method.
            Tobs (float, optional): Observation time in seconds. If provided, applies trimming to keep only the first `Tobs` duration of data.
            **kwargs: Additional keyword arguments for the eventual plots.

        Returns:
            tuple: Processed times and data arrays.
        """
        filtered = False

        if highpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying highpass filter...")
            self.data = self.highpass_filter(**highpass_kwargs, **kwargs)
            filtered = True

        if lowpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying lowpass filter...")
            self.data = self.lowpass_filter(**lowpass_kwargs, **kwargs)
            filtered = True

        if bandpass_kwargs is not None:
            if self.verbose:
                logger.info("Applying bandpass filter...")
            self.data = self.bandpass_filter(**bandpass_kwargs, **kwargs)
            filtered = True

        if downsample_kwargs is not None:
            if self.verbose:
                logger.info("Downsampling data...")
            self.downsample(**downsample_kwargs)

        if trim_kwargs is not None:
            if self.verbose:
                logger.info("Trimming data...")
            _, _ = self.trim(**trim_kwargs, **kwargs)

        elif trim_kwargs is None and filtered:
            logger.warning(
                "Data was filtered but no trim_kwargs provided. Applying default trimming of 100 hours from each end to avoid edge effects."
            )
            trim_duration = 100 * 3600  # 100 hours in seconds
            _, _ = self.trim(
                duration=trim_duration,
                is_percent=False,
                trimming_type="from_each_end",
                **kwargs,
            )

        if Tobs is not None:
            if Tobs > self.T:
                logger.warning(
                    f"Requested observation time Tobs={Tobs}s exceeds total data duration T={self.T}s. No trimming applied."
                )
                Tobs = self.T

            _, _ = self.trim(
                duration=Tobs,
                is_percent=False,
                trimming_type="keep_from_start",
                filename="science_data.png",
                **kwargs,
            )

        xp = get_array_module(self.data)
        
        settings = TDSettings(
            t0=self.times[0],
            dt=self.dt,
            N=self.N,
            force_backend="cpu" if xp is np else "gpu",
            # xp = get_array_module(self.data)
        )
        self.td_signal = TDSignal(arr=self.data, settings=settings)

        return self.times, self.data

    def pour(
        self,
        settings: DomainSettingsBase,
        window: Optional[np.ndarray | str] = None,
        return_orbits: bool = False,
    ) -> "DomainBase | tuple[DomainBase, Orbits]":
        """
        Pour the loaded and preprocessed data into a domain instance.

        Args:
            settings (DomainSettingsBase): Settings for the domain.
            window (Optional[np.ndarray | str], optional): Window to apply to the data. Defaults to None.
            return_orbits (bool, optional): Whether to return the orbits instance along with the domain. Defaults to False.

        Returns:
            A :class:`DomainBase` child (FDSignal / WDMSignal / TDSignal / STFTSignal)
            in the requested domain. Optionally also returns the orbits instance.
        """

        # The TD signal is already a DomainBase; ``.transform(settings, window)``
        # produces the matching DomainBase child in the target domain.
        #
        # TODO (2026-07-07, backend auto-conversion follow-up): transform()
        # now auto-transfers a CPU-loaded signal onto the target settings'
        # backend (DomainBase._coerce_transform_backend), which fixes the
        # GPU-settings-vs-CPU-data mismatch on this pour() path. Two related
        # boundaries still need an audit pass:
        #   1. the equality short-circuit below can hand back the
        #      CPU-resident td_signal WITHOUT any transfer when the target
        #      settings compare equal but live on another backend — if a
        #      NumPy/CuPy mix fires just downstream of pour(), apply
        #      ``data_signal = data_signal.with_backend(settings.backend)``;
        #   2. other CPU-side ingest points (noise-knot arrays into
        #      XYZSensitivityBackend.set_sensitivity_matrix, hand-built
        #      window arrays in settings files) are not auto-converted —
        #      DomainBase.with_backend / xp.asarray at the boundary is the
        #      pattern to use, one transfer per array, when they fire.
        if self.td_signal.settings == settings:
            data_signal = self.td_signal
        else:
            # TODO: fix this to be chopped based on Tobs originally
            # Target TD length for the chop: FDSettings.N counts rfft BINS,
            # not TD samples -- derive the TD length from df there. Other
            # domains carry the TD sample count on ``settings.N``.
            if isinstance(settings, FDSettings):
                target_td_n = round(1 / (settings.df * self.td_signal.dt))
            else:
                target_td_n = settings.N
            if self.td_signal.N != target_td_n:
                assert target_td_n < self.td_signal.N
                _td_sig = TDSignal(
                    self.td_signal[:, :target_td_n],
                    TDSettings(target_td_n, self.td_signal.dt),
                )
            else:
                _td_sig = self.td_signal
            data_signal = _td_sig.transform(settings, window=window)

        if return_orbits:
            return data_signal, self.orbits
        return data_signal


class L1ProcessingStep(L1DataLoader, BaseProcessingStep):
    """
    Class to load and preprocess Mojito L1 files.
    Inherits from L1DataLoader and BaseProcessingStep.
    """

    def __init__(
        self,
        L1_folder: str,
        source_types: list,
        source_ids: dict = None,
        orbits_class: Orbits = L1Orbits,
        orbits_kwargs: dict = None,
        verbose: bool = True,
        store_individual_timeseries: bool = False,
        do_plots: bool = False,
        Tobs: float = None,
        window_start_offset: float = 0.0,
    ):
        L1DataLoader.__init__(
            self,
            L1_folder=L1_folder,
            source_types=source_types,
            source_ids=source_ids,
            orbits_class=orbits_class,
            orbits_kwargs=orbits_kwargs,
            verbose=verbose,
            store_individual_timeseries=store_individual_timeseries,
        )

        times, fs, data_xyz, orbits = self.load_data()
        # TODO: add this to other processors?
        # ``window_start_offset`` (seconds from the file start) lets a caller
        # chop a sub-window that does NOT begin at the file start -- e.g. a
        # snippet centered on an MBH merger mid-mission. Default 0.0 keeps the
        # legacy behavior (first ``Tobs`` from the file start) byte-for-byte.
        if Tobs is not None:
            rel = times[:] - times[0]
            offset = float(window_start_offset)
            keep = (rel >= offset) & (rel < offset + Tobs)
            times = times[keep]
            data_xyz = data_xyz[:, keep]

        BaseProcessingStep.__init__(self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots)

        self.orbits = orbits

    def _open(self, file_path: str) -> MojitoL1File:
        return MojitoL1File(file_path)

    def process(self, *args, **kwargs):
        """
        Apply identical processing to the main data and all individual timeseries.
        """
        if hasattr(self, "individual_timeseries") and self.individual_timeseries is not None:
            for key, ts_data in self.individual_timeseries.items():
                if "PSD" in key:  # skip processing for PSD data
                    continue
                temp_processor = BaseProcessingStep(
                    times=self.times.copy(),
                    data=ts_data.copy(),
                    fs=self.fs,
                    verbose=False,  # Suppress logs for the individual processing
                    do_plots=False,
                )
                _, ts_data = temp_processor.process(*args, **kwargs)
                self.individual_timeseries[key] = ts_data

        processed_times, processed_data = super().process(*args, **kwargs)

        if hasattr(self, "individual_timeseries") and self.individual_timeseries is not None:
            self.individual_timeseries["TIMES"] = (
                processed_times  # store the processed times as well
            )
            self.individual_timeseries["COMBINED"] = processed_data

        return processed_times, processed_data


class SangriaProcessingStep(SangriaDataLoader, BaseProcessingStep):
    """
    Class to load and preprocess Sangria HDF5 data injections.
    Inherits from SangriaDataLoader and BaseProcessingStep.
    """

    def __init__(
        self,
        data_input_path: str,
        remove_from_data: Optional[list] = None,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        SangriaDataLoader.__init__(
            self, data_input_path=data_input_path, remove_from_data=remove_from_data
        )

        times, fs, data_xyz, _ = self.load_data()

        BaseProcessingStep.__init__(self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots)

        self.orbits = None  # no orbital information available in Sangria files
        self.individual_timeseries = (
            {}
        )  # to store individual timeseries for each source type and ID, if needed for debugging or further analysis


# ============================================================
# Synthetic-data / injection processing
# ============================================================
@dataclasses.dataclass
class SyntheticSourceSpec:
    """One source class to inject onto a synthetic data grid.

    Args:
        label: Identifier for the returned per-source stream (e.g. ``"EMRI"``).
        wave_gen: Callable producing a per-source signal. When
            ``times_from_gen`` is ``False`` it returns a ``(nch, n)`` (or ``(n,)``)
            array on the data grid; when ``True`` it returns ``(times, channels)``.
        injections: ``(num_sources, ndim)`` array of parameter rows (a single 1-D
            row is broadcast to ``(1, ndim)``).
        param_transform: Optional per-row transform applied before calling
            ``wave_gen`` (e.g. an MBH sampling->waveform basis transform).
        times_from_gen: If ``True``, ``wave_gen`` returns ``(times, channels)`` and
            the channels are placed on the grid at their native ``times``.
        call_kwargs: Optional extra keyword arguments forwarded to ``wave_gen``
            (e.g. ``{"convert_to_ra_dec": False}`` for the EMRI wrapper).
    """

    label: str
    wave_gen: Callable
    injections: np.ndarray
    param_transform: Optional[Callable] = None
    times_from_gen: bool = False
    call_kwargs: Optional[dict] = None


def build_synthetic_source_streams(
    grid: TDSettings,
    *,
    sources: Sequence[SyntheticSourceSpec],
    nchannels: int,
    progress: bool = False,
) -> "dict[str, np.ndarray]":
    """Sum per-source TD signal streams onto ``grid``.

    Each :class:`SyntheticSourceSpec` is evaluated over its injection rows and
    the resulting signals are placed onto the shared ``grid`` via
    :func:`~lisatools.domains.place_td_signal_on_grid`. Returns a mapping
    ``{label: (nchannels, grid.N)}`` (one summed stream per source class).
    """
    N = grid.N
    out: "dict[str, np.ndarray]" = {}
    for spec in sources:
        inj = np.atleast_2d(spec.injections)
        stream = np.zeros((nchannels, N), dtype=np.float64)
        call_kwargs = spec.call_kwargs or {}
        if inj.shape[0] > 0 and inj.size > 0:
            for ii, row in enumerate(inj):
                if progress:
                    logger.info(
                        "Injecting %s signal %d of %d", spec.label, ii + 1, inj.shape[0]
                    )
                if spec.param_transform is not None:
                    params = spec.param_transform(np.asarray(row, dtype=float))
                else:
                    params = row
                if spec.times_from_gen:
                    times, channels = spec.wave_gen(*params, **call_kwargs)
                    placed = place_td_signal_on_grid(
                        np.atleast_2d(channels), grid, times=times
                    ).arr
                else:
                    sig = np.asarray(spec.wave_gen(*params, **call_kwargs))
                    placed = place_td_signal_on_grid(np.atleast_2d(sig), grid).arr
                stream += np.asarray(placed)[:nchannels]
        out[spec.label] = stream
    return out


class SyntheticSourceProcessingStep(BaseProcessingStep):
    """Generate a single source class's injection in-process.

    Builds the injected TD stream from a source-specific response wrapper and
    hands ``(times, data, fs)`` to :class:`BaseProcessingStep` — the same
    interface as :class:`SangriaProcessingStep`, with no external file needed.

    Supply the wrapper either directly (``wave_gen``) for runtime use, or via a
    ``wave_gen_factory`` for the global-fit settings path: the response wrappers
    hold unpicklable C++ objects, so when this step is registered as
    ``GeneralSettings.data_processor_class`` (whose ``processor_init_kwargs`` are
    deep-copied), pass a **module-level** ``wave_gen_factory`` function and let it
    build the wrapper lazily inside ``__init__``. The factory is called as
    ``wave_gen_factory(Tobs=..., dt=..., t_start=..., tdi_chan=...)``.
    """

    def __init__(
        self,
        *,
        Tobs: float,
        dt: float,
        t_start: float,
        wave_gen: Optional[Callable] = None,
        wave_gen_factory: Optional[Callable] = None,
        injections: np.ndarray,
        param_transform: Optional[Callable] = None,
        times_from_gen: bool = False,
        call_kwargs: Optional[dict] = None,
        tdi_chan: str = "XYZ",
        nchannels: int = 3,
        verbose: bool = True,
        do_plots: bool = False,
    ):
        if wave_gen is None:
            if wave_gen_factory is None:
                raise ValueError(
                    "SyntheticSourceProcessingStep needs either wave_gen or "
                    "wave_gen_factory."
                )
            wave_gen = wave_gen_factory(
                Tobs=Tobs, dt=dt, t_start=t_start, tdi_chan=tdi_chan
            )
        target_N = int(round(Tobs / dt))
        grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")
        spec = SyntheticSourceSpec(
            label=tdi_chan,
            wave_gen=wave_gen,
            injections=injections,
            param_transform=param_transform,
            times_from_gen=times_from_gen,
            call_kwargs=call_kwargs,
        )
        streams = build_synthetic_source_streams(
            grid, sources=[spec], nchannels=nchannels, progress=verbose
        )
        combined = next(iter(streams.values()))

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        self.injections = np.atleast_2d(injections)


class SyntheticDataProcessor(BaseProcessingStep):
    """All-synthetic processor: injected sources + optional noise + foreground.

    Sums the per-class source streams (:func:`build_synthetic_source_streams`)
    onto a data grid, optionally adding correlated instrument noise
    (:func:`~lisatools.sensitivity.generate_correlated_instrument_noise_td`) and
    a galactic-foreground TD realization
    (:func:`~lisatools.sensitivity.generate_foreground_td`), all placed on the
    same grid. An optional ``base_data`` (e.g. a Sangria sky-only slice or a
    mojito L1 stream) is added underneath. Exposes ``catalogue`` (if supplied)
    so downstream factories can read leaf counts uniformly.

    Args:
        Tobs / dt / t_start: Data-grid definition (``N = round(Tobs/dt)``).
        source_specs: :class:`SyntheticSourceSpec` list to inject.
        nchannels: Number of TDI channels.
        add_instrument_noise: ``None`` or kwargs for
            :func:`~lisatools.sensitivity.generate_correlated_instrument_noise_td`
            (``Soms_d``, ``Sa_a``, ``tdi_generation``, ``seed``, ...).
        add_foreground: ``None`` or kwargs for
            :func:`~lisatools.sensitivity.generate_foreground_td`
            (``foreground_params``, ``tdi_generation``, ``seed``, ``envelope`` ...;
            ``Tobs`` defaults to this processor's ``Tobs``).
        base_data: Optional ``(nch, n)`` TD stream added underneath the injections.
        catalogue: Optional catalogue dict stored on ``self.catalogue``.
        tdi_chan: TDI channel label recorded on ``self.tdi_chan``.
    """

    def __init__(
        self,
        *,
        Tobs: float,
        dt: float,
        t_start: float,
        source_specs: Sequence[SyntheticSourceSpec],
        nchannels: int = 3,
        add_instrument_noise: Optional[dict] = None,
        add_foreground: Optional[dict] = None,
        base_data: Optional[np.ndarray] = None,
        catalogue: Optional[dict] = None,
        tdi_chan: str = "XYZ",
        verbose: bool = True,
        do_plots: bool = False,
    ):
        # Local import: sensitivity is a heavier module and only needed when
        # synthetic noise / foreground are requested (no import cycle —
        # sensitivity does not import globalfit).
        from ..sensitivity import (
            generate_correlated_instrument_noise_td,
            generate_foreground_td,
        )

        target_N = int(round(Tobs / dt))
        grid = TDSettings(N=target_N, dt=dt, t0=t_start, force_backend="cpu")

        combined = np.zeros((nchannels, target_N), dtype=np.float64)
        if base_data is not None:
            combined = combined + np.asarray(
                place_td_signal_on_grid(np.atleast_2d(base_data), grid).arr
            )[:nchannels]

        streams = build_synthetic_source_streams(
            grid, sources=source_specs, nchannels=nchannels, progress=verbose
        )
        for stream in streams.values():
            combined = combined + stream

        if add_instrument_noise is not None:
            noise_td = generate_correlated_instrument_noise_td(
                target_N, dt, **add_instrument_noise
            )[:nchannels]
            combined = combined + noise_td

        if add_foreground is not None:
            fg_kwargs = dict(add_foreground)
            fg_kwargs.setdefault("Tobs", Tobs)
            fg_td = generate_foreground_td(target_N, dt, **fg_kwargs)[:nchannels]
            combined = combined + fg_td

        times = np.arange(target_N) * dt + t_start
        fs = 1.0 / dt
        BaseProcessingStep.__init__(
            self, times, combined, fs, verbose=verbose, do_plots=do_plots
        )
        self.orbits = None
        self.tdi_chan = tdi_chan
        if catalogue is not None:
            self.catalogue = catalogue
