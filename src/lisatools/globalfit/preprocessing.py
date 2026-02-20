"""
Preprocessing module for loading and handling Mojito L1 files.

Credits for most of the signal processing functions to Ollie Burke and Martina Muratore for the repository: https://github.com/OllieBurke/mojito-noise-sprint/blob/main/Mojito_Sprint/
"""

import logging
import os
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mojito import MojitoL1File
from scipy import signal
from tqdm import tqdm

from ..datacontainer import DataResidualArray
from ..detector import L1Orbits, Orbits
from ..domains import DomainBase, DomainSettingsBase, TDSettings, TDSignal
from ..utils.utility import detrend, get_array_module

logger = logging.getLogger(__name__)

level = logging.INFO
logger.setLevel(level)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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


def _apply_filter(
    data_in: np.ndarray,
    fs: float,
    low_or_cutoff: float,
    high: Optional[float],
    btype: str,
    order: int,
    filter_type: str,
    zero_phase: bool,
    axis: int = -1,
    **kwargs,
) -> np.ndarray:
    """
    Internal method to apply filter to all channels. Credits: Ollie Burke.

    Args:
        data_in (np.ndarray): Input data array with shape (n_channels, n_times).
        fs (float): Sampling frequency in Hz.
        low_or_cutoff (float): Low cutoff frequency for highpass/bandpass or cutoff frequency for lowpass.
        high (Optional[float]): High cutoff frequency for bandpass. None for lowpass/highpass.
        btype (str): Type of filter: 'lowpass', 'highpass', 'bandpass'.
        order (int): Filter order.
        filter_type (str): Filter type: 'butterworth', 'chebyshev1', 'chebyshev2', 'bessel'.
        zero_phase (bool): If True, use zero-phase filtering (filtfilt), else single-pass.
        axis (int): Axis    along which to apply the filter. Defaults to -1 (last axis).
        **kwargs: Additional keyword arguments for the eventual plot.

    Returns:
        np.ndarray: Filtered data array.
    """
    # Determine critical frequencies
    if btype == "bandpass":
        Wn = [low_or_cutoff, high]
    else:
        Wn = low_or_cutoff

    # Design filter
    filter_funcs = {
        "butterworth": signal.butter,
        "chebyshev1": signal.cheby1,
        "chebyshev2": signal.cheby2,
        "bessel": signal.bessel,
    }

    if filter_type not in filter_funcs:
        raise ValueError(
            f"Unknown filter type: {filter_type}. "
            f"Choose from {list(filter_funcs.keys())}"
        )

    sos = filter_funcs[filter_type](order, Wn, btype=btype, fs=fs, output="sos")

    # Apply filter to all channels
    if zero_phase:
        filtered_data = signal.sosfiltfilt(sos, data_in, axis=axis)
    else:
        filtered_data = signal.sosfilt(sos, data_in, axis=axis)

    if kwargs.get("do_plots", False):
        decimate_factor = kwargs.get(
            "decimate_factor", 1000
        )  # Decimate for plotting if too many points

        plt.figure(figsize=(10, 4))
        plt.plot(
            data_in[0, ::decimate_factor],
            label="Original X channel",
            c="gray",
            alpha=0.5,
        )
        plt.plot(
            filtered_data[0, ::decimate_factor], label="Filtered X channel", c="blue"
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Strain")
        plt.title(
            f"Signal {btype.capitalize()} Filtering ({filter_type}, order={order})"
        )
        plt.legend()
        plt.savefig(
            os.path.join(kwargs.get("plot_folder", "."), f"{btype}_filter_signal.png")
        )
        plt.close()

    return filtered_data


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
    """

    def __init__(
        self,
        L1_folder: str,
        source_types: list,
        source_ids: dict = None,
        orbits_class: Orbits = L1Orbits,
        orbits_kwargs: dict = None,
        verbose: bool = True,
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

        if self.verbose:
            logger.info(
                f"L1DataLoader initialized with data folder: {self.data_folder}"
            )
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
            "MBHB": "mbhb_cat_mojito_lite_processed_MT.hdf5",
            "EMRI": "emri_cat_mojito_lite_processed_MT.hdf5",
            "SOBHB": "sobhb_cat_mojito_lite_processed_MT.hdf5",
        }

    def load_single_binary(self, group: h5py.Group, binary_id: int) -> dict:
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
            params[key] = group[key][binary_id][()]

        return params

    def _open(self, file_path: str) -> MojitoL1File:
        raise NotImplementedError("_open method should be implemented in subclasses.")

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

        if "NOISE" in self.source_types:
            subfolder = os.path.join(self.data_folder, "INSTRUMENT", "L1")
            file_path = find_file(subfolder, "NOISE", 00)

            with self._open(file_path) as f:
                xyz = f.tdis.xyz_doppler[:]

                tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                tdi_times = f.tdis.time_sampling.t()

            orbits = self.orbits_class(file_path, **(self.orbits_kwargs or {}))
            orbits.configure(linear_interp_setup=True)
            self.source_types.remove("NOISE")

            if self.verbose:
                logger.info(f"Loaded NOISE data from {file_path}")
                logger.info(f"data, times and orbits initialized from NOISE file.")
                logger.info(f"TDI time step: {tdi_dt} seconds")
                logger.info(f"TDI sampling frequency: {tdi_fs} Hz")

        for source_type in self.source_types:

            subfolder = os.path.join(self.data_folder, source_type, "L1")
            ids = self.source_ids.get(source_type, [])

            if source_type in ["GB", "VGB"]:
                ids = [0]  # only one file for GB/VGB

            if not isinstance(ids, list):
                ids = [ids]
            if len(ids) == 0:
                raise ValueError(
                    f"No source IDs provided for source type '{source_type}'."
                )

            binary_params = h5py.File(
                os.path.join(
                    self.catalogues_folder, self.catalogues_map.get(source_type)
                ),
                "r",
            )["Binaries"]

            self.catalogue[source_type] = {}

            if self.verbose:
                logger.info(f"Loading data for source type '{source_type}'")

            for source_id in tqdm(
                ids, desc=f"Loading {source_type} sources", disable=not self.verbose
            ):
                file_path = find_file(subfolder, source_type, source_id)

                self.catalogue[source_type][source_id] = self.load_single_binary(
                    binary_params, source_id
                )
                if self.verbose:
                    logger.info(
                        f"Loaded catalogue parameters for {source_type} source ID {source_id} from catalogue."
                    )

                with self._open(file_path) as f:
                    _xyz = f.tdis.xyz_doppler[:]
                    _tdi_dt = f.tdis.time_sampling.dt  # time step in seconds
                    _tdi_times = f.tdis.time_sampling.t()
                    _tdi_fs = f.tdis.time_sampling.fs  # sampling frequency in Hz
                    if xyz is None:
                        xyz = _xyz
                        tdi_dt = _tdi_dt
                        tdi_times = _tdi_times
                        tdi_fs = _tdi_fs

                        orbits = self.orbits_class(
                            file_path, **(self.orbits_kwargs or {})
                        )
                        orbits.configure(linear_interp_setup=True)

                    else:
                        xyz += _xyz
                        assert (
                            tdi_dt == _tdi_dt
                        ), "Time steps do not match between files."
                        assert (
                            tdi_fs == _tdi_fs
                        ), "Sampling frequencies do not match between files."
                        assert (
                            tdi_times == _tdi_times
                        ).all(), "Time arrays do not match between files."

        xyz = xyz.T  # Transpose to have shape (n_channels, n_times)
        assert (
            xyz.shape[1] == tdi_times.shape[0]
        ), "Data time dimension does not match time array length."

        return tdi_times, tdi_fs, xyz, orbits


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
                    print(f"Removing {source} from data injection.")
                    change_arr = f["sky"][source]["tdi"][:]
                    for change in ["X", "Y", "Z"]:
                        tXYZ[change] -= change_arr[change]

            else:
                keys = list(f["sky"])
                print("Initial keys in data injection: ", keys)
                tmp_keys = keys.copy()
                for key in tmp_keys:
                    print(key)
                    if key in self.remove_from_data:
                        keys.remove(key)
                        print(f"Removing {key} from data injection.")
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
        self.T = (
            times[-1] - times[0]
        )  # total duration. Takes into account that times[0] may not be zero.
        self.data = data
        self.fs = fs
        self.verbose = verbose
        self.do_plots = do_plots

    def bandpass_filter(
        self,
        low: float,
        high: float,
        order: int = 6,
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
            Filter order (default: 6)
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
        return _apply_filter(
            self.data,
            self.fs,
            low,
            high,
            "bandpass",
            order,
            filter_type,
            zero_phase,
            axis=-1,
            do_plots=self.do_plots,
            **kwargs,
        )

    def lowpass_filter(
        self,
        cutoff: float,
        order: int = 6,
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
            Filter order (default: 6)
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
        return _apply_filter(
            self.data,
            self.fs,
            cutoff,
            None,
            "lowpass",
            order,
            filter_type,
            zero_phase,
            axis=-1,
            do_plots=self.do_plots,
            **kwargs,
        )

    def highpass_filter(
        self,
        cutoff: float,
        order: int = 6,
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
            Filter order (default: 6)
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
        return _apply_filter(
            self.data,
            self.fs,
            cutoff,
            None,
            "highpass",
            order,
            filter_type,
            zero_phase,
            axis=-1,
            do_plots=self.do_plots,
            **kwargs,
        )

    def _update_params(self):
        """Update internal parameters after data modification."""
        self.N = self.data.shape[1]
        self.T = self.times[-1] - self.times[0]

        if self.verbose:
            logger.info(f"Updated parameters: N={self.N}, T={self.T}")

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
                    f"Cannot trim {duration}s from {self.T}s data. "
                    f"Trim exceeds data length."
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
                logger.info(
                    f"Keeping first {duration}s ({trim_samples} samples) of the data."
                )

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
        do_detrend: bool = True,
        highpass_kwargs: dict = None,
        lowpass_kwargs: dict = None,
        bandpass_kwargs: dict = None,
        trim_kwargs: dict = None,
        Tobs: float = None,
        **kwargs,
    ) -> tuple:
        """
        Apply preprocessing steps to the loaded data. for each step, if the corresponding kwargs is None, the step is skipped.

        Args:
            do_detrend (bool, optional): Whether to detrend the data. Defaults to True.
            highpass_kwargs (dict, optional): Keyword arguments for highpass_filter method.
            lowpass_kwargs (dict, optional): Keyword arguments for lowpass_filter method.
            bandpass_kwargs (dict, optional): Keyword arguments for bandpass_filter method.
            trim_kwargs (dict, optional): Keyword arguments for trim method.
            Tobs (float, optional): Observation time in seconds. If provided, applies trimming to keep only the first `Tobs` duration of data.
            **kwargs: Additional keyword arguments for the eventual plots.

        Returns:
            tuple: Processed times and data arrays.
        """
        filtered = False

        if do_detrend:
            if self.verbose:
                logger.info("Detrending data...")
            _X = detrend(self.times, self.data[0])
            _Y = detrend(self.times, self.data[1])
            _Z = detrend(self.times, self.data[2])
            self.data = np.vstack([_X, _Y, _Z])
            assert (
                self.data.shape[1] == self.times.shape[0]
            ), "Data and times length mismatch after detrending."

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

        settings = TDSettings(
            t0=self.times[0],
            dt=self.dt,
            N=self.N,
            # xp = get_array_module(self.data)
        )
        self.td_signal = TDSignal(arr=self.data, settings=settings)

        return self.times, self.data

    def pour(
        self,
        settings: DomainSettingsBase,
        window: Optional[np.ndarray | str] = None,
        return_orbits: bool = False,
    ) -> DataResidualArray | tuple[DataResidualArray, Orbits]:
        """
        Pour the loaded and preprocessed data into a domain instance.

        Args:
            settings (DomainSettingsBase): Settings for the domain.
            window (Optional[np.ndarray | str], optional): Window to apply to the data. Defaults to None.
            return_orbits (bool, optional): Whether to return the orbits instance along with the domain. Defaults to False.

        Returns:
            DataResidualArray | tuple[DataResidualArray, Orbits]: The data residual array in the specified domain, and optionally the orbits instance.
        """

        data_residual_array = DataResidualArray(
            data_res_in=self.td_signal,
            signal_domain=settings,
            input_signal_domain=self.td_signal.settings,
            window=window,
        )

        # result = self.td_signal.transform(new_domain=settings, window=window)

        if return_orbits:
            return data_residual_array, self.orbits
        return data_residual_array


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
        do_plots: bool = False,
    ):
        L1DataLoader.__init__(
            self,
            L1_folder=L1_folder,
            source_types=source_types,
            source_ids=source_ids,
            orbits_class=orbits_class,
            orbits_kwargs=orbits_kwargs,
            verbose=verbose,
        )

        times, fs, data_xyz, orbits = self.load_data()

        BaseProcessingStep.__init__(
            self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots
        )

        self.orbits = orbits

    def _open(self, file_path: str) -> MojitoL1File:
        return MojitoL1File(file_path)


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

        BaseProcessingStep.__init__(
            self, times, data_xyz, fs, verbose=verbose, do_plots=do_plots
        )

        self.orbits = None  # no orbital information available in Sangria files
