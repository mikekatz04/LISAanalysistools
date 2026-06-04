"""
postprocessing.py
=================

Post-processing routines for the output of a global fit run.

Produces L3C-compliant HDF5 submission files and GitHub Pages-compatible
JSON manifests for web dashboard display.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
from eryn.backends import HDFBackend
from eryn.utils import get_integrated_act
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from ..domains import FDSettings, STFTSettings, TDSettings
from ..utils.utility import windowfun
from .gathergalaxy import gather_gb_samples

if TYPE_CHECKING:
    from eryn.utils.transform import TransformContainer
    from ..analysiscontainer import AnalysisContainerArray, AnalysisContainer, DataResidualArray
    from ..detector import Orbits
    from ..domains import DomainSettingsBase, TDSettings
    from ..sensitivity import XYZSensitivityBackend
    from .run import CurrentInfoGlobalFit

logger = getLogger(__name__)
# ─── Parameter metadata ───────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ParameterInfo:
    """Stores the L3C plain name, LaTeX display label, and unit for one parameter."""

    l3c_name: str
    latex_name: str
    unit: str


_GB_PARAM_INFO: Dict[str, ParameterInfo] = {
    "A": ParameterInfo("amplitude", r"$A$", "dimensionless"),
    "f0": ParameterInfo("frequency", r"$f_0\,[\mathrm{Hz}]$", "Hz"),
    "fdot": ParameterInfo("frequency_dot", r"$\dot{f}\,[\mathrm{Hz\,s^{-1}}]$", "Hz/s"),
    "fddot": ParameterInfo("frequency_double_dot", r"$\ddot{f}\,[\mathrm{Hz\,s^{-2}}]$", "Hz/s^2"),
    "phi0": ParameterInfo("initial_phase", r"$\phi_0\,[\mathrm{rad}]$", "rad"),
    "iota": ParameterInfo("inclination", r"$\iota\,[\mathrm{rad}]$", "rad"),
    "psi": ParameterInfo("polarization", r"$\psi\,[\mathrm{rad}]$", "rad"),
    "ra": ParameterInfo("right_ascension", r"$\alpha\,[\mathrm{rad}]$", "rad"),
    "dec": ParameterInfo("declination", r"$\delta\,[\mathrm{rad}]$", "rad"),
}

_MBH_PARAM_INFO: Dict[str, ParameterInfo] = {
    "m1": ParameterInfo("primary_mass_det_frame", r"$m_1\,[M_\odot]$", "solMass"),
    "m2": ParameterInfo("secondary_mass_det_frame", r"$m_2\,[M_\odot]$", "solMass"),
    "s1z": ParameterInfo("spin1", r"$s_{1z}$", "dimensionless"),
    "s2z": ParameterInfo("spin2", r"$s_{2z}$", "dimensionless"),
    "distance": ParameterInfo("luminosity_distance", r"$d_L\,[\mathrm{Gpc}]$", "Gpc"),
    "phi_ref": ParameterInfo("phase_at_reference_time", r"$\phi_{\mathrm{ref}}\,[\mathrm{rad}]$", "rad"),
    "iota": ParameterInfo("inclination", r"$\iota\,[\mathrm{rad}]$", "rad"),
    "psi": ParameterInfo("polarization", r"$\psi\,[\mathrm{rad}]$", "rad"),
    "ra": ParameterInfo("right_ascension", r"$\alpha\,[\mathrm{rad}]$", "rad"),
    "dec": ParameterInfo("declination", r"$\delta\,[\mathrm{rad}]$", "rad"),
    "t_c": ParameterInfo("coalescence_time", r"$t_c\,[\mathrm{s}]$", "s"),
}

_NOISE_PARAM_INFO: Dict[str, ParameterInfo] = {
    "S_oms": ParameterInfo(
        "S_oms", r"$S_{\mathrm{oms}}\,[\mathrm{m}\mathrm{Hz}^{-1/2}]$", "m Hz^{-1/2}"
    ),
    "S_tm": ParameterInfo(
        "S_tm",
        r"$S_{\mathrm{tm}}\,[\mathrm{m}\mathrm{s}^{-2}\mathrm{Hz}^{-1/2}]$",
        "m s^{-2} Hz^{-1/2}",
    ),
}

_GALFOR_PARAM_INFO: Dict[str, ParameterInfo] = {
    "A_gal": ParameterInfo("galactic_amplitude", r"$A_{\mathrm{gal}}$", "dimensionless"),
    "alpha_gal": ParameterInfo("galactic_spectral_index", r"$\alpha_{\mathrm{gal}}$", "dimensionless"),
    "f_1": ParameterInfo("galactic_freq_1", r"$f_1\,[\mathrm{Hz}]$", "Hz"),
    "f_knee": ParameterInfo("galactic_knee_frequency", r"$f_{\mathrm{knee}}\,[\mathrm{Hz}]$", "Hz"),
    "f_2": ParameterInfo("galactic_freq_2", r"$f_2\,[\mathrm{Hz}]$", "Hz"),
}

# todo add galactic foreground

PARAMETER_INFO_REGISTRY: Dict[str, Dict[str, ParameterInfo]] = {
    "gb": _GB_PARAM_INFO,
    "mbh": _MBH_PARAM_INFO,
    "psd": _NOISE_PARAM_INFO,
    "galfor": _GALFOR_PARAM_INFO,
    # add entries for other source types as needed
}

_STOCHASTIC_BRANCHES = ["psd", "galfor"]

# Corrections applied after TransformContainer to fix waveform-gen vs L3C unit mismatches.
# The MBH transform applies gpc_to_mpc (Gpc→Mpc) for the waveform generator;
# L3C requires distance in Gpc, so we invert here.
_OUTPUT_CORRECTIONS_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "gb": {},
    "mbh": {"luminosity_distance": lambda x: x * 1e-3},
    "psd": {},
    "galfor": {},
}

_REMOVED_PARAMS_REGISTRY: Dict[str, List[str]] = {
    "gb": ["frequency_double_dot"],
    "mbh": [],
    "psd": [],
    "galfor": [],
}

source_types_names = dict(
    gb="GB",  # todo: separate the verification binaries into their own source type
    mbh="MBHB",
    emri="EMRI",
    sobh="SOBHB",
    psd="NOISE",
    galfor="NOISE",  # todo: should we merge noise and stochastic together under a common "stochastic" source type?
)

# order of the mojito light mbh sources, sorted by merger time (earliest to latest), for the purpose of assigning `known_injection` labels in the metadata
_MOJITO_LIGHT_MBH_ORDER = ["source_18", "source_5", "source_16", "source_7", "source_2", "source_12", "source_9", "source_4", "source_0", "source_15", "source_3", "source_10", "source_19", "source_13", "source_6", "source_17", "source_8", "source_1", "source_11", "source_14"] 

MAX_SOURCES_PER_BATCH = 500

def _apply_output_corrections(branch: str, samples_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply branch-specific output corrections to a posterior mapping."""

    corrected = dict(samples_dict)
    transform = _OUTPUT_CORRECTIONS_REGISTRY[branch]

    for key, func in transform.items():
        if key in corrected:
            corrected[key] = func(corrected[key])
            logger.info(
                f"Applied output correction for parameter '{key}' in branch '{branch}'"
            )

    return corrected


def _filter_removed_params(branch: str, samples_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Remove parameters that should not be written for the given branch."""

    removed_params = set(_REMOVED_PARAMS_REGISTRY[branch])
    filtered = {}

    for name, values in samples_dict.items():
        if name in removed_params:
            logger.info(
                f"Parameter '{name}' is marked for removal in branch '{branch}'. Skipping this parameter in the output."
            )
            continue
        filtered[name] = values

    return filtered


def _samples_dict_to_structured_array(samples_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a posterior mapping into a structured NumPy array."""

    if not samples_dict:
        raise ValueError("Cannot build a structured array from an empty posterior mapping.")

    param_names = list(samples_dict.keys())
    param_data = [samples_dict[name] for name in param_names]
    dtype = [(name, "f8") for name in param_names]
    structured_array = np.zeros(len(param_data[0]), dtype=dtype)

    for name, data in zip(param_names, param_data):
        structured_array[name] = data

    return structured_array


def _save_metadata_attributes(h5obj: h5py.File | h5py.Group, metadata: MetadataBase) -> None:
    """Save dataclass metadata fields as HDF5 attributes."""

    for field in dataclasses.fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, (str, int, float)):
            h5obj.attrs[field.name] = value
        elif isinstance(value, list):
            h5obj.attrs[field.name] = json.dumps(value)
        elif isinstance(value, dict):
            h5obj.attrs[field.name] = json.dumps(value)
        else:
            logger.warning(
                f"Unsupported metadata field type for '{field.name}': {type(value)}. Skipping this field."
            )


def _save_posterior_dataset(
    h5obj: h5py.File | h5py.Group,
    source_label: str,
    samples_dict: dict[str, np.ndarray],
) -> None:
    """Write one posterior dataset into an open HDF5 object."""

    h5obj.create_dataset(source_label, data=_samples_dict_to_structured_array(samples_dict))


def _build_leaf_samples_dict(
    samples_here: np.ndarray,
    parameter_names: list[str],
    leaf: int,
    branch: str,
    log_prior: np.ndarray,
    log_likelihood: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build one leaf's posterior mapping and apply corrections/filtering."""

    samples_dict = {name: samples_here[..., i] for i, name in enumerate(parameter_names)}
    samples_dict["logprior"] = log_prior
    samples_dict["loglikelihood"] = log_likelihood
    samples_dict = _apply_output_corrections(branch, samples_dict)
    samples_dict = _filter_removed_params(branch, samples_dict)

    return {
        name: values[:, leaf] if len(values.shape) == 2 else values
        for name, values in samples_dict.items()
    }


def _build_stochastic_samples_dict(
    samples: np.ndarray,
    parameter_names: list[str],
    branches_here: list[str],
    log_prior: np.ndarray,
    log_likelihood: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the combined posterior mapping for stochastic branches."""

    samples_dict = {name: samples[:, 0, i] for i, name in enumerate(parameter_names)}
    samples_dict["logprior"] = log_prior
    samples_dict["loglikelihood"] = log_likelihood

    combined: dict[str, np.ndarray] = {}
    for branch in branches_here:
        branch_samples = _apply_output_corrections(branch, samples_dict)
        branch_samples = _filter_removed_params(branch, branch_samples)
        combined.update(branch_samples)

    return combined

# ─── Helpers ──────────────────────────────────────────────────────────────────


def get_source_types(curr: CurrentInfoGlobalFit) -> List[str]:
    """Get the list of source types searched for in this run, inferred from curr."""
    sources = list(curr.source_info.keys())

    out = []
    for s in sources:
        if s in source_types_names.keys():
            out.append(source_types_names[s])
        else:
            logger.debug(f"Excluding source type '{s}' from metadata")

    return out


def _seconds_to_l3c_datetime(t: float) -> str:
    """Convert a UTC timestamp in seconds with respect to LISA_EPOCH_TCB to the L3C format yyyy.mm.dd.hh.mm.ss."""
    from lisaconstants import LISA_EPOCH_TCB

    format_str = "%Y-%m-%dT%H:%M:%S.%f"
    lisa_epoch_seconds = datetime.strptime(LISA_EPOCH_TCB, format_str).timestamp()

    dt = datetime.fromtimestamp(t + lisa_epoch_seconds)
    return dt.strftime("%Y.%m.%d.%H.%M.%S")


def _seconds_to_duration_str(t: float) -> str:
    """Express a duration in seconds as L3C effective_observation_duration string."""
    days = int(t // 86400)
    rem = t % 86400
    hours = int(rem // 3600)
    rem %= 3600
    minutes = int(rem // 60)
    seconds = int(rem % 60)
    return f"0000.00.{days:02d}.{hours:02d}.{minutes:02d}.{seconds:02d}"


def _infer_tdi_channels(curr: CurrentInfoGlobalFit) -> List[str]:
    """Infer TDI channel list from GB setup or sensitivity backend class name."""
    if "gb" in curr.source_info and hasattr(curr.source_info["gb"], "tdi_setup"):
        setup_str = curr.source_info["gb"].tdi_setup or "XYZ"
        return list(setup_str)
    backend_name = type(curr.general_info.sensitivity_backend).__name__
    if "AET" in backend_name:
        return ["A", "E", "T"]
    if "AE" in backend_name:
        return ["A", "E"]
    return ["X", "Y", "Z"]


# ─── BackendConsumer ──────────────────────────────────────────────────────────


class BackendConsumer:
    """
    Wraps an Eryn HDFBackend and provides cold-chain extraction utilities.

    Can be constructed from a live `curr` object or a standalone HDFBackend,
    making it usable both in-session and in offline post-processing.
    """

    def __init__(
        self,
        curr: CurrentInfoGlobalFit = None,
        backend: HDFBackend | str = None,
    ):
        if backend is not None:
            if isinstance(backend, str):
                self.backend = HDFBackend(filename=backend, read_only=True)
            else:
                self.backend = backend
        elif curr is not None:
            self.backend = HDFBackend(filename=curr.main_file_path, read_only=True)
        else:
            raise ValueError("Must provide either curr or backend.")

        if curr is not None:
            self._curr = curr

            self.from_curr()

    def from_curr(self):
        """
        Extract any additional info needed from curr to configure the consumer for post-processing.
        """

    @property
    def curr(self) -> CurrentInfoGlobalFit:
        if not hasattr(self, "_curr"):
            raise AttributeError("BackendConsumer was not initialized with a curr object.")
        return self._curr

    @property
    def branches(self) -> List[str]:
        return self.backend.branch_names

    @property
    def ndims(self) -> dict[str, int]:
        return self.backend.ndims

    @property
    def nleaves_max(self) -> dict[str, int]:
        return self.backend.nleaves_max

    @property
    def transform_containers(self) -> dict[str, TransformContainer | None]:
        """Return the TransformContainer for each branch, if present."""

        if not hasattr(self, '_transforms'):
            self._transforms = {}
            for name in self.branches:
                if hasattr(self.curr.source_info[name], "transform") and self.curr.source_info[name].transform is not None:
                        self._transforms[name] = deepcopy(self.curr.source_info[name].transform)
                        #self._transforms[name].fill_dict = None
                else:
                    logger.warning(f"No TransformContainer found for branch '{name}'.")
                    self._transforms[name] = None

        return self._transforms

    def store_cold_chains(self):
        """Extract cold chains and inds for all branches and cache them."""
        self._cold_chains = {}
        self._cold_inds = {}
        all_chains = self.backend.get_chain()
        all_inds = self.backend.get_inds()
        for branch in self.branches:
            self._cold_chains[branch] = all_chains[branch][:, :1]
            self._cold_inds[branch] = all_inds[branch][:, :1]

        self._cold_log_prior = self.backend.get_log_prior()[:, :1]
        self._cold_log_likelihood = self.backend.get_log_like()[:, :1]

    @property
    def configured(self) -> bool:
        return hasattr(self, "_cold_chains")

    @property
    def cold_chains(self) -> Dict[str, np.ndarray]:
        if not self.configured:
            raise AttributeError("Cold chains have not been extracted yet.")
        return self._cold_chains

    @property
    def cold_inds(self) -> Dict[str, np.ndarray]:
        if not self.configured:
            raise AttributeError("Cold chains inds have not been extracted yet.")
        return self._cold_inds

    def get_cold_chain(self, branch: str = None) -> Dict[str, np.ndarray] | np.ndarray:
        if not self.configured:
            self.store_cold_chains()
        if branch is not None:
            if branch in self.branches:
                return self._cold_chains[branch]
            raise ValueError(f"Branch '{branch}' not found in cold chains.")
        return self.cold_chains

    def get_cold_inds(self, branch: str) -> np.ndarray:
        if not self.configured:
            self.store_cold_chains()
        if branch not in self.branches:
            raise ValueError(f"Branch '{branch}' not found in cold chains.")
        return self._cold_inds[branch]

    def compute_auto_correlation_time(
        self, branch: str = None, **act_kwargs
    ) -> Dict[str, np.ndarray] | np.ndarray:
        """Compute the integrated ACT for the specified branch or all branches."""
        if not self.configured:
            self.store_cold_chains()
        if branch is not None:
            if branch in self.branches:
                return get_integrated_act(self._cold_chains[branch], **act_kwargs)
            raise ValueError(f"Branch '{branch}' not found in cold chains.")
        act: dict = get_integrated_act(self.cold_chains, **act_kwargs)

        # now remove nans if present
        for branch, value in act.items():
            if not np.isfinite(value).all():
                logger.warning(
                    f"NaN values found in ACT for branch '{branch}'. These will be replaced with 1."
                )
                act[branch] = np.where(np.isfinite(value), value, 1.0)

        return act

    def store_independent_samples(self, discard: int | float = 0.0, ess: int = 10000, **act_kwargs):
        """
        Thin the cold chains by the ACT and keep the last `ess`.

        Args:
            discard: int or float (optional). If int, the number of initial samples to discard. Else, fraction of initial steps to discard
            ess: int (optional). Effective sample size
        """
        all_act = self.compute_auto_correlation_time(**act_kwargs)
        max_act = int(np.round(max(np.max(act) for act in all_act.values())))
        max_act = max(1, max_act)  # ensure at least 1
        logger.debug(f"Thinning factor: {max_act}")

        self._thinned_chains = {}
        self._thinned_inds = {}

        nsteps = None
        for branch in self.branches:
            print(f"Processing branch '{branch}'")
            if nsteps == None:
                nsteps = self.cold_chains[branch].shape[0]
                if discard < 1.0:
                    discard = int(discard * nsteps)
            tmp = self.cold_chains[branch][discard::max_act].reshape(
                -1, self.nleaves_max[branch], self.ndims[branch]
            )
            tmp_inds = self.cold_inds[branch][discard::max_act].reshape(
                -1, self.nleaves_max[branch]
            )

            if len(tmp.shape) == 5:  # it still has the temperature dimension
                tmp = tmp[:, 0]  # take only the coldest temperature

            if tmp.shape[0] < ess:
                logger.warning(f"Branch '{branch}' has fewer than {ess} thinned samples.")
            self._thinned_chains[branch] = tmp[-ess:]
            self._thinned_inds[branch] = tmp_inds[-ess:]

        self._thinned_log_prior = self._cold_log_prior[discard::max_act].flatten()[-ess:]
        self._thinned_log_likelihood = self._cold_log_likelihood[discard::max_act].flatten()[-ess:]

        self.ess = ess

    @property
    def thinned_chains(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_thinned_chains"):
            raise ValueError("Thinned samples have not been computed yet")

        return self._thinned_chains

    @property
    def thinned_inds(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_thinned_inds"):
            raise ValueError("Thinned samples have not been computed yet")

        return self._thinned_inds

    @property
    def thinned_log_prior(self) -> np.ndarray:
        if not hasattr(self, "_thinned_log_prior"):
            raise ValueError("Thinned log prior has not been computed yet")
        return self._thinned_log_prior

    @property
    def thinned_log_likelihood(self) -> np.ndarray:
        if not hasattr(self, "_thinned_log_likelihood"):
            raise ValueError("Thinned log likelihood has not been computed yet")
        return self._thinned_log_likelihood

    def get_independent_samples(
        self,
        discard: int | float = 0.0,
        ess: int = 10000,
        branch: str | None = None,
        return_inds: bool = False,
    ) -> dict | np.ndarray | Tuple[dict | np.ndarray, dict | np.ndarray]:
        """
        Get the thinned samples for the specified branch or all branches.
        """

        if not hasattr(self, "_thinned_chains") or (hasattr(self, "ess") and self.ess != ess):
            self.store_independent_samples(discard=discard, ess=ess)

        if branch is not None:
            if branch not in self.branches:
                raise ValueError(f"Branch '{branch}' not found in thinned chains.")
            if return_inds:
                out = (self.thinned_chains[branch], self.thinned_inds[branch])
            else:
                out = (self.thinned_chains[branch],)
        else:
            if return_inds:
                out = (self.thinned_chains, self.thinned_inds)
            else:
                out = (self.thinned_chains,)
        return out + (self.thinned_log_prior, self.thinned_log_likelihood)

    def transform(self, samples: dict | np.ndarray, branch: str | None = None) -> dict | np.ndarray:
        """
        Apply the TransformContainer for the specified samples to the given samples.

        Args:
            samples: dict of branch to (n_samples, nleaves_max, ndim) array, or a single (n_samples, nleaves_max, ndim) array if branch is specified.
            branch: str (optional). If provided, apply the transform for this branch to the samples. Else, apply the appropriate transform to each branch in the dict.

        Returns:
            dict of branch to transformed samples, or a single array if branch is specified.
        """

        if branch is not None:
            if branch not in self.branches:
                raise ValueError(f"Branch '{branch}' not found in backend.")
            if branch not in self.transform_containers or self.transform_containers[branch] is None:
                logger.warning(
                    f"No TransformContainer found for branch '{branch}'. Returning input samples."
                )
                return samples
            return self.transform_containers[branch].both_transforms(samples)

        transformed = {}
        for b, s in samples.items():
            if b in self.transform_containers and self.transform_containers[b] is not None:
                transformed[b] = self.transform_containers[b].both_transforms(s)
            else:
                logger.warning(
                    f"No TransformContainer found for branch '{b}'. Returning input samples for this branch."
                )
                transformed[b] = s

        return transformed

    def process_samples(
        self, discard: int | float = 0.0, ess: int = 10000, return_inds: bool = False
    ) -> Tuple[dict, Optional[dict], np.ndarray, np.ndarray]:
        """
        Convenience method to run the end-to-end processing pipeline, starting from the raw samples.

        Args:
            discard: int or float (optional). If int, the number of initial samples to discard. Else, fraction of initial steps to discard
            ess: int (optional). Effective sample size
            return_inds: bool (optional). Whether to return the corresponding inds arrays.

        Returns:
            tuple: (transformed_samples, inds, log_prior, log_likelihood)
        """
        if not self.configured:
            self.store_cold_chains()

        samples, inds, log_prior, log_likelihood = self.get_independent_samples(
            discard=discard, ess=ess, return_inds=True
        )

        transformed_samples = self.transform(samples)

        if return_inds:
            return transformed_samples, inds, log_prior, log_likelihood

        return transformed_samples, log_prior, log_likelihood


# ——— Plotter ──────────────────────────────────────────────────────────────————


def to_periodogram(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a timeseries to a periodogram (frequency, power) pair.

    Uses a Tukey window and proper normalization to produce a one-sided PSD estimate.
    """
    from scipy.signal import windows

    window = windows.tukey(len(x), alpha=0.1)
    x_windowed = x * window
    freqs = np.fft.rfftfreq(len(x), dt)
    norm = 2 / (dt * np.sum(window**2))  # "two-sided" normalization
    periodogram = np.abs(np.fft.rfft(x_windowed) * dt) ** 2 * norm

    return freqs, periodogram


def to_characteristic_strain(freqs: np.ndarray, periodogram: np.ndarray) -> np.ndarray:
    """
    Convert a periodogram to characteristic strain.

    Uses the relation h_c(f) = sqrt(f * S_n(f)), where S_n is the one-sided PSD estimate.
    """
    return np.sqrt(freqs * periodogram)


def log_decimate(freqs, power, f_min=None, f_max=None, decimation_factor=10):
    """
    Logarithmically decimate the periodogram by a given factor.

    Args:
        freqs: Array of frequencies (must be sorted)
        power: Array of power values
        f_min, f_max: Frequency range of interest
        decimation_factor: Factor by which to reduce the number of points

    Returns:
        tuple: (decimated_freqs, decimated_power)
    """
    # Mask frequencies in range of interest
    f_min = f_min or freqs.min()
    f_max = f_max or freqs.max()

    mask = (freqs >= f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    power = power[mask, ...]

    # Create decimated frequency array
    n_points = len(freqs)
    n_decimated = max(1, n_points // decimation_factor)

    decimated_indices = np.unique(
        np.round(np.logspace(0, np.log10(n_points - 1), n_decimated)).astype(int)
    )

    decimated_freqs = freqs[decimated_indices]
    decimated_power = power[decimated_indices]

    return decimated_freqs, decimated_power


class GlobalFitPlotter:
    """
    Produce summary plots at the end of a global fit run, including posterior predictive plots and corner plots.
    """

    def __init__(self, curr: CurrentInfoGlobalFit):

        self.curr = curr

        # corner plots: separated per leaf, joint over all leaves color coded by snr

    def convert_input_data_for_plotting(
        self,
    ):
        """
        Convert the raw timeseries data into characteristic strain. We keep only the X channel for plotting.
        """
        if not hasattr(self.curr.general_info.data_processor, "_individual_timeseries"):

            logger.warning(
                "No individual timeseries found in data processor. Skipping input data conversion for plotting."
            )
            return {}

        out = {}

        data_components = self.curr.general_info.data_processor.individual_timeseries

        times, combined = data_components.pop("TIMES"), data_components.pop("COMBINED")[0]

        dt = times[1] - times[0]
        freqs, periodogram = to_periodogram(combined, dt)
        freqs_h, power_avg = log_decimate(
            freqs,
            periodogram,
            f_min=self.curr.general_info.start_freq,
            f_max=self.curr.general_info.end_freq,
            decimation_factor=100,
        )

        combined_char_strain = to_characteristic_strain(freqs_h, power_avg)
        out["freqs"] = freqs_h
        out["combined"] = combined_char_strain

        del (
            freqs,
            periodogram,
            power_avg,
        )  # free memory as we don't need the raw periodogram anymore

        logger.info("Converted input data to characteristic strain for plotting.")

        covariance_matrix = data_components.pop("PSD_MATRIX")[0, :, 0, 0].real
        covariance_frequencies = data_components.pop("PSD_FREQUENCIES")
        _ = data_components.pop("PSD_TIMES")

        noise_amplitude = to_characteristic_strain(covariance_frequencies, covariance_matrix)
        out["noise_amplitude_estimate"] = CubicSpline(covariance_frequencies, noise_amplitude)(
            freqs_h
        )

        logger.info("Converted noise covariance to characteristic strain for plotting.")

        for k, v in tqdm(
            data_components.items(), desc="Converting components to characteristic strain"
        ):

            freqs, periodogram = to_periodogram(v[0], dt)
            char_strain = to_characteristic_strain(freqs, periodogram)

            out[k.lower()] = CubicSpline(freqs, char_strain)(freqs_h)

            del (
                freqs,
                periodogram,
                char_strain,
            )  # free memory as we don't need the raw periodogram anymore

        del data_components  # free memory as we don't need the raw timeseries anymore
        import gc

        gc.collect()

        return out

    def save_input_data(self, converted_data: dict = None):
        """
        Dump the converted input data to a h5 file for later plotting in the web dashboard.
        """
        if converted_data is None:
            converted_data = self.convert_input_data_for_plotting()

        if len(converted_data) == 0:
            logger.warning("No converted data to save for plotting.")
            return

        parts = self.curr.general_info.global_fit_version.split("_", 1)
        if len(parts) == 2:
            run_type, run_id = parts
            filepath = os.path.join(
                self.curr.general_info.submission_parent_folder, run_type, run_id, "input_data.h5"
            )

        filepath = os.path.join(
            self.curr.general_info.submission_parent_folder,
            self.curr.general_info.global_fit_version,
            "input_data.h5",
        )

        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            logger.info(f"Created directory {os.path.dirname(filepath)} for saving converted input data.")

        if os.path.exists(filepath):
            logger.warning(f"File {filepath} already exists. It will be overwritten.")

        with h5py.File(filepath, "w") as f:
            for k, v in converted_data.items():
                f.create_dataset(k, data=v)

                logger.info(f"Saved converted input data component '{k}' to {filepath}")


# ─── Metadata extractors ─────────────────────────────────────────────────────


def _extract_sensitivity_metadata(gi) -> tuple[dict, dict]:
    """Extract noise model configuration from the initialised sensitivity backend."""
    backend: XYZSensitivityBackend = gi.sensitivity_backend

    kwargs = backend.kwargs.copy()

    domain_class = kwargs["settings"].__class__.__name__
    domain_args = deepcopy(kwargs["settings"].args)
    domain_kwargs = kwargs["settings"].kwargs.copy()
    domain_kwargs["force_backend"] = "cpu"

    kwargs.pop("orbits")
    kwargs.pop("settings")
    kwargs.pop("window_values")
    kwargs["force_backend"] = "cpu"

    domain_metadata = {
        "class": domain_class,
        "args": domain_args,
        "kwargs": domain_kwargs,
    }

    sensitivity_metadata = {
        "class": type(backend).__name__,
        "kwargs": kwargs,
    }

    return domain_metadata, sensitivity_metadata


def _extract_orbit_metadata(gi) -> dict:
    """Extract orbit configuration from the initialised orbits object."""
    orbits = gi.orbits
    kwargs = orbits.kwargs.copy()
    kwargs["force_backend"] = "cpu"
    out = {
        "class": type(orbits).__name__,
        "kwargs": kwargs,
    }

    return out


# ─── RunMetadata ──────────────────────────────────────────────────────────────
@dataclasses.dataclass
class MetadataBase:
    """
    Base class for metadata objects, providing common utilities.
    """

    def to_dict(self) -> dict:
        """Convert the dataclass to a dict, excluding private fields."""
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__.keys() if not k.startswith("_")
        }

    def to_json(self, filepath: str):
        """Save the metadata to a JSON file at the specified path."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Saved metadata to {filepath}")


@dataclasses.dataclass
class RunMetadata(MetadataBase):
    """
    Holds all metadata for a global fit run.

    User-supplied fields must be provided at construction.
    Auto-populated fields are filled by `from_curr`.
    """

    # user-supplied — required
    global_fit_version: str = ""
    global_fit_contact: str = ""
    global_fit_code_link: str = ""
    input_data_link: str = ""
    input_reference: str = ""
    noise_model: str = ""
    noise_model_code_link: str = ""
    domain_metadata: dict = dataclasses.field(default_factory=dict)
    orbits_metadata: dict = dataclasses.field(default_factory=dict)
    sensitivity_metadata: dict = dataclasses.field(default_factory=dict)
    preprocessing_metadata: dict = dataclasses.field(default_factory=dict)
    submission_parent_folder: str = ""

    # auto-populated from curr or defaults
    submission_timestamp: str = ""
    global_fit_codename: str = "Erebor"
    comment: str = ""
    found_source_types_list: List[str] = dataclasses.field(default_factory=list)
    observation_period_begin: str = ""
    observation_period_end: str = ""
    time_step: float = 0.0
    number_of_time_samples: int = 0
    window_type: str = "tukey"
    window_alpha: float = 0.1
    effective_observation_duration: str = ""
    tdi_channels: List[str] = dataclasses.field(default_factory=list)
    searched_source_types_list: List[str] = dataclasses.field(default_factory=list)

    # extra info for web display (not part of L3C spec)
    _web_extras: Dict[str, Any] = dataclasses.field(default_factory=dict, repr=False)

    @classmethod
    def from_curr(cls, curr: CurrentInfoGlobalFit, **user_fields) -> "RunMetadata":
        """
        Construct RunMetadata, auto-populating timing, channel, and metadata
        fields from `curr`.

        Descriptive fields (version, contact, etc.) are read from
        ``curr.general_info`` when set there; ``**user_fields`` takes precedence.

        Args:
            curr: Live CurrentInfoGlobalFit object.
            **user_fields: Optional overrides for any RunMetadata field.
        """
        gi = curr.general_info

        # Map L3C-named properties directly from `gi`
        auto = {}
        for attr in cls.__dataclass_fields__.keys():
            if hasattr(gi, attr) and getattr(gi, attr, None) is not None:
                auto[attr] = getattr(gi, attr)

        merged = {**auto, **user_fields}

        # Derive noise_model from the backend class name when not explicitly set.
        merged.setdefault("noise_model", type(gi.sensitivity_backend).__name__)

        instance = cls(**merged)
        instance.submission_timestamp = datetime.now().isoformat(
            timespec="seconds"
        )  # stop at seconds for cleaner display
        instance.observation_period_begin = _seconds_to_l3c_datetime(gi.data_t0)
        instance.observation_period_end = _seconds_to_l3c_datetime(gi.data_t0 + gi.Tobs)
        instance.time_step = float(gi.dt)
        instance.number_of_time_samples = int(gi.data_td_settings.N)
        instance.effective_observation_duration = _seconds_to_duration_str(gi.Tobs)
        instance.tdi_channels = _infer_tdi_channels(curr)
        instance.searched_source_types_list = get_source_types(curr)

        preprocess_kwargs = getattr(gi, "preprocess_kwargs", {})
        _ = preprocess_kwargs.pop("plot_folder", None)  # not relevant for the metadata

        domain_metadata, sensitivity_metadata = _extract_sensitivity_metadata(gi)
        orbits_metadata = _extract_orbit_metadata(gi)
        instance.domain_metadata = domain_metadata
        instance.sensitivity_metadata = sensitivity_metadata
        instance.orbits_metadata = orbits_metadata
        instance.preprocessing_metadata = preprocess_kwargs
        instance.input_data_link = os.path.join(instance.submission_folder, "input_data.h5")

        instance.found_source_types_list = get_source_types(curr)  # todo: for now we assume all searched sources are found; use detection results to populate this

        instance._web_extras = {
            "Tobs_s": float(gi.Tobs),
            "dt_s": float(gi.dt),
            "basis_domain": gi.basis_domain,
            "start_freq_hz": float(gi.start_freq) if gi.start_freq is not None else None,
            "end_freq_hz": float(gi.end_freq) if gi.end_freq is not None else None,
            "nwalkers": gi.nwalkers,
            "ntemps": gi.ntemps,
        }

        return instance

    def _to_l3c_dict(self) -> dict:
        """Return a dict matching the l2_output_metadata template keys exactly."""

        d = self.to_dict()

        d["global_fit_release_date"] = self.submission_timestamp
        if "noise_model_config_file_link" not in d:
            d["noise_model_config_file_link"] = ""
        d["list_of_detected_sources"] = ", ".join(self.found_source_types_list)

        return d

    @classmethod
    def from_submission(cls, submission: str | dict) -> "RunMetadata":
        """
        Construct RunMetadata from a submission matching the l2_output_metadata template keys.

        Args:
            submission: Either a dict containing the metadata fields, or a path to a JSON file containing the metadata.

        Returns:
            RunMetadata instance with fields populated from the submission.
        """

        if isinstance(submission, str):
            with open(submission, "r") as f:
                d = json.load(f)
        else:
            d = submission

        user_fields = {
            k: v for k, v in d.items() if k in cls.__dataclass_fields__ and not k.startswith("_")
        }
        # supply missing required fields if not present in the output JSON
        if "submission_parent_folder" not in user_fields:
            user_fields["submission_parent_folder"] = ""

        instance = cls(**user_fields)

        return instance

    def get_orbits(self, filename: str) -> Orbits:
        """
        Reconstruct the Orbits object from the stored metadata.

        Args:
            filename: str. Path to the the file containing the orbits information. For the Mojito datasets, any file would work.

        Returns:
            Orbits object reconstructed from the stored metadata.
        """

        if not self.orbits_metadata:
            raise ValueError("No orbits metadata found.")

        orbits_class_name = self.orbits_metadata["class"]
        orbits_kwargs = self.orbits_metadata["kwargs"]
        orbits_class = getattr(importlib.import_module("lisatools.detector"), orbits_class_name)

        logger.info(
            f"Reconstructing Orbits object of class '{orbits_class_name}' with kwargs: {orbits_kwargs}"
        )

        orbits = orbits_class(filename=filename, **orbits_kwargs)
        orbits.configure(linear_interp_setup=True)

        return orbits

    def get_domain_settings(self) -> DomainSettingsBase:
        """Reconstruct the DomainSettings object from the stored metadata."""

        if not self.domain_metadata:
            raise ValueError("No domain settings metadata found.")

        domain_class_name = self.domain_metadata["class"]
        domain_args = self.domain_metadata["args"]
        domain_kwargs = self.domain_metadata["kwargs"]
        domain_class = getattr(importlib.import_module("lisatools.domains"), domain_class_name)

        logger.info(
            f"Reconstructing DomainSettings object of class '{domain_class_name}' with args: {domain_args} and kwargs: {domain_kwargs}"
        )

        return domain_class(*domain_args, **domain_kwargs)

    def get_data_td_settings(self) -> TDSettings:
        """Reconstruct the data TDSettings object from the stored metadata."""

        logger.info(
            f"Reconstructing TDSettings object with t0: {self.observation_period_begin}, N: {self.number_of_time_samples}, dt: {self.time_step}"
        )
        return TDSettings(
            t0=self.observation_period_begin,
            N=self.number_of_time_samples,
            dt=self.time_step,
            force_backend="cpu",
        )

    def get_sensitivity_matrix(
        self,
        orbits: Optional[Orbits] = None,
        filename: str = None,
        domain_settings: Optional[DomainSettingsBase] = None,
    ) -> XYZSensitivityBackend:
        """Reconstruct the sensitivity matrix from the stored metadata.

        Args:
            orbits: Optional Orbits object. If not provided, it will be reconstructed from the stored metadata using `get_orbits()`.
            filename: Optional str. Path to the file containing the orbits information, needed if `orbits` is not provided.
            domain_settings: Optional DomainSettings object. If not provided, it will be reconstructed from the stored metadata using `get_domain_settings()`.

        Returns:
            XYZSensitivityBackend object reconstructed from the stored metadata.
        """
        if orbits is None and filename is None:
            raise ValueError(
                "Must provide either an Orbits object or a filename to reconstruct it from."
            )

        orbits = orbits or self.get_orbits(filename=filename)
        domain_settings = domain_settings or self.get_domain_settings()

        if not self.sensitivity_metadata:
            raise ValueError("No sensitivity metadata found.")

        sensitivity_class_name = self.sensitivity_metadata["class"]
        sensitivity_kwargs = self.sensitivity_metadata["kwargs"]
        sensitivity_class = getattr(
            importlib.import_module("lisatools.sensitivity"), sensitivity_class_name
        )

        logger.info(
            f"Reconstructing sensitivity backend of class '{sensitivity_class_name}' with kwargs: {sensitivity_kwargs} and domain settings: {domain_settings}"
        )

        if isinstance(domain_settings, FDSettings):
            N = self.number_of_time_samples

        elif isinstance(domain_settings, STFTSettings):
            N = domain_settings.get_nperseg(self.time_step)

        else:
            raise NotImplementedError(
                f"Domain settings of type '{type(domain_settings).__name__}' not supported"
            )

        window_values, _ = windowfun(self.window_type, N, alpha=self.window_alpha)

        return sensitivity_class(
            orbits=orbits,
            settings=domain_settings,
            window_values=window_values,
            **sensitivity_kwargs,
        )

    def to_web_dict(self) -> dict:
        """Return a richer dict for web display, extending the L3C dict with run config."""
        d = self._to_l3c_dict()
        d.update(self._web_extras)
        return d

    @property
    def run_type(self) -> str:
        """Extract the run type from the global_fit_version string."""
        parts = self.global_fit_version.split("_", 1)
        if len(parts) == 2:
            return parts[0]
        logger.warning(
            f"Version string '{self.global_fit_version}' does not follow expected format 'type_id'. Using entire version string as run_type."
        )
        return self.global_fit_version

    @property
    def run_id(self) -> str:
        """Extract the run id from the global_fit_version string."""
        parts = self.global_fit_version.split("_", 1)
        if len(parts) == 2:
            return parts[1]
        logger.warning(
            f"Version string '{self.global_fit_version}' does not follow expected format 'type_id'. Using 'v0' as default run_id."
        )
        return "v0"

    @property
    def submission_folder(self) -> str:
        """Return the full path to the submission folder for this run."""

        folder_name = (
            f"{self.run_type}_{self.global_fit_codename}_{self.run_id}_{self.submission_timestamp}"
        )

        if not os.path.exists(os.path.join(self.submission_parent_folder, folder_name)):
            logger.warning(
                f"Submission folder '{folder_name}' does not exist in parent folder '{self.submission_parent_folder}'. Creating it."
            )
            os.makedirs(os.path.join(self.submission_parent_folder, folder_name))

        return os.path.join(self.submission_parent_folder, folder_name)


@dataclasses.dataclass
class SourceMetadata(MetadataBase):
    """Holds metadata for a detected source, to be included in the L3C submission manifest."""

    source_type: str
    frequency_ranges: list[tuple[float, float]]  # = dataclasses.field(default_factory=list)
    waveform_model: str
    waveform_model_code_link: str
    waveform_model_config: dict  # = dataclasses.field(default_factory=dict)

    detection_statistic: list[float] = dataclasses.field(default_factory=list)
    quality_flags: list[int] = dataclasses.field(default_factory=list)

    prior_model: str = ""
    prior_model_code_link: str = ""
    prior_model_config: dict = dataclasses.field(default_factory=dict)

    posterior_files: list[str] = dataclasses.field(
        default_factory=list
    )  # links to the posterior samples for this source, to be included after the run
    comment: str = ""


@dataclasses.dataclass
class StochasticMetadata(MetadataBase):
    """Holds metadata for the overall stochastic component of the data"""

    model_config: dict
    frequency_ranges: list[tuple[float, float]]

    prior_model: str = ""
    prior_model_code_link: str = ""
    prior_model_config: dict = dataclasses.field(default_factory=dict)

    posterior_files: list[str] = dataclasses.field(
        default_factory=list
    )  # links to the posterior samples for this source, to be included after the run
    comment: str = ""


class SubmissionWriter(BackendConsumer):
    """
    BackendConsumer subclass that produces L3C-compliant HDF5 submission files and JSON manifests.

    This class is responsible for applying the detection criteria to identify genuine sources,
    and for writing the final outputs in the required formats.
    """

    def __init__(
        self,
        curr: CurrentInfoGlobalFit = None,
        backend: HDFBackend | str = None,
        ess: int = 10000,
        detection_criteria: DetectionCriteria = None,
    ):

        super().__init__(curr=curr, backend=backend)

        self.samples, self.inds, self.log_prior, self.log_likelihood = self.process_samples(
            ess=ess, return_inds=True
        )

        self.detection_criteria = detection_criteria or OccupancyDetectionCriteria()
        self.run_metadata = RunMetadata.from_curr(self.curr)

    def prepare_samples_for_submission(self, acs: AnalysisContainerArray):
        """
        Prepare the samples for submission by clustering the GBs, and sorting the mbhbs according to the coalescence time.
        Each source type needs a callable to handle the specific processing, and this method act as a dispatcher.
        """

        for branch in self.branches:
            prepare_fn = self.prepare_samples_registry.get(branch, None)
            if prepare_fn:
                logger.info(f"Preparing samples for branch '{branch}' using '{prepare_fn.__name__}'")
                self.samples[branch], self.inds[branch] = prepare_fn(
                    acs, self.samples[branch], self.inds[branch]
                )

    def _prepare_mbhb_samples(self, acs: AnalysisContainerArray, samples: np.ndarray, inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort the MBHB samples according to the coalescence time, and cluster them if there are multiple sources in the same block."""

        coalescence_time_idx = list(PARAMETER_INFO_REGISTRY["mbh"].keys()).index("t_c")
        mean_coalescence_times = samples[:, :, coalescence_time_idx].mean(axis=0) # shape (nleaves_max,)
        sorted_indices = np.argsort(mean_coalescence_times)
        samples = samples[:, sorted_indices, :]
        inds = inds[:, sorted_indices]

        return samples, inds
    
    def _prepare_gb_samples(self, acs: AnalysisContainerArray, samples: np.ndarray, inds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare GB samples for submission running the clustering algorithm.
        """
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("cupy is required for GB clustering. Please install cupy to use this feature.")
        
        from gbgpu.gbgpu import GBGPU
        from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend
        from lisatools.globalfit.state import GBState

        gb_info = self.curr.source_info["gb"]
        gb_wave_gen = GBGPU(**gb_info.initialize_kwargs)
        gb_wave_gen.gpus = self.curr.general_info.gpus[:1] # use only one GPU for the clustering

        cluster_kwargs = dict(
            num_compare_samples=200,
            samples_keep=5,
            thin_by=1,
            snr_lim_first_cut=7.0,
            snr_lim_second_cut=5.0,
            overlap_lim=0.7,
            snr_diff_lim=20.0,
        )
        
        reader = GFHDFBackend(
            self.backend.filename, sub_state_bases={"gb": GBState}, sub_backend={"gb": GBHDFBackend}
        )
        
        gb_wave_gen.d_d = 0.0
        max_logl_walker = np.argmax(acs.likelihood()).item()
        sens_mat = acs[max_logl_walker].sens_mat
        
        if len(acs.gpus) > 1:
            # we probably need everything on the same GPU
            cp.cuda.runtime.setDevice(gb_wave_gen.gpus[0])
            sens_mat._sens_mat = cp.asarray(sens_mat._sens_mat)
            
        logger.info('starting to gather GB samples for clustering')
        
        groups = gather_gb_samples(
            acs.f_arr,
            gb_info.transform,
            gb_wave_gen,
            gb_info.waveform_kwargs.copy(),
            cp.asarray(gb_info.band_edges),
            gb_info.band_N_vals,
            reader,
            sens_mat,
            gb_wave_gen.gpus[0],
            gb_samples=samples,
            gb_inds=inds,
            **cluster_kwargs,
        )

        logger.info(f"Completed clustering. Number of groups found: {len(groups)}")

        num_in_groups = np.asarray([len(tmp) for tmp in groups])
        keep = num_in_groups > reader.nwalkers * cluster_kwargs['samples_keep'] / 2

        logger.info(
            f"Groups passing sample count filter: {keep.sum()} / {len(keep)}. "
            f"num_in_groups: {num_in_groups}"
        )
        max_num_source = max([tmp.shape[0] for tmp in groups])
        samples = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
        for i, group in enumerate(groups):
            samples[i, : len(group)] = group

        samples_fin = samples[keep]
        num_in_groups_fin = num_in_groups[keep]

        breakpoint()


    # todo resume from here.

    @property
    def prepare_samples_registry(self) -> dict[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        """Registry mapping each source type to its corresponding sample preparation function."""
        return {
            "mbh": self._prepare_mbhb_samples,
            "gb": self._prepare_gb_samples,
        }

    def _get_mbhb_label(self, leaf_samples: np.ndarray, leaf_index: int) -> str:
        """Get a label for an MBHB source based on its coalescence time."""
        
        coalescence_time_idx = list(PARAMETER_INFO_REGISTRY["mbh"].keys()).index("t_c")
        mean_coalescence_time = int(leaf_samples[:, coalescence_time_idx].mean())

        xxxxx = f"{leaf_index:05d}"
        yyyyy = f"{mean_coalescence_time}"
        
        return f"{xxxxx}_{yyyyy}"

    def _get_gb_label(self, leaf_samples: np.ndarray, leaf_index: int) -> str:
        """Get a label for a GB source based on its frequency."""
        
        frequency_idx = list(PARAMETER_INFO_REGISTRY["gb"].keys()).index("f0")
        mean_frequency = leaf_samples[:, frequency_idx].mean()
        mean_frequency_microhz = int(mean_frequency * 1e6)

        xxxxx = f"{leaf_index:05d}"
        yyyyy = f"{mean_frequency_microhz}"

        return f"{xxxxx}_{yyyyy}"
    
    @property
    def get_source_label_registry(self) -> dict[str, Callable]:
        """Define how sources should be labelled in the output files.
        For Mojito light, we use the XXXXX_YYYYY format, with XXXXX a 5 digit ID based on the leaf count (eg, 00000, 00001, etc) and YYYYY a descriptor of the source (eg, coalescence time for MBHBs, frequency for GBs, etc).
        """

        return {
            "mbh": self._get_mbhb_label,
            "gb": self._get_gb_label,
        }

    @property
    def known_injections(self) -> dict[str, list[str]]:
        """Return a dict mapping each source type to a list of known injections for that source type, to be included in the metadata."""
        # for Mojito light, we don't have any known injections, so we return an empty list for each source type
        return {
            'mbh': _MOJITO_LIGHT_MBH_ORDER, 
            'gb': []
            }


    @property
    def submission_folder(self) -> str:
        """Return the full path to the submission folder for this run."""
        return self.run_metadata.submission_folder

    @property
    def posterior_folders(self) -> dict[str, str]:
        """Return the full paths to the posterior folders for each branch."""
        if not hasattr(self, "_relative_folders"):
            raise ValueError("Folders have not been created yet. Call create_folders() first.")
        return self._relative_folders

    def create_folders(self):
        """Create the submission folder and any necessary subfolders."""

        os.makedirs(self.submission_folder, exist_ok=True)
        logger.info(f"Created submission folder at {self.submission_folder}")

        self._relative_folders = {}

        for branch in self.branches:
            source_name = source_types_names[branch]

            subfolder_name = f"{self.run_metadata.run_type}_{self.run_metadata.global_fit_codename}_{self.run_metadata.run_id}_{source_name}_posteriordir"

            subfolder_path = os.path.join(self.submission_folder, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            logger.info(f"Created subfolder for branch '{branch}' at {subfolder_path}")

            self._relative_folders[branch] = subfolder_name # we store the relative path to the posterior folder, to be included in the metadata and used in the output files

    def save_posteriors(self):
        """Save the posterior samples for each detected source in the required format."""
        from tqdm import tqdm

        self._save_stochastic_posterior()  
        branches_resolvable = self.branches.copy()

        for key in _STOCHASTIC_BRANCHES:
            if key in branches_resolvable:
                branches_resolvable.remove(key)

        # todo need a way to have ids
        for branch in tqdm(branches_resolvable, desc="Saving posteriors for resolvable sources"):
            logger.info(f"Saving posteriors for branch {branch}")

            self._save_source_posterior(branch)

    def _save_source_posterior(self, branch: str):
        """Save the posterior samples for a single detected source type, identified by `branch`."""

        source_name = source_types_names[branch]
        samples = self.samples[branch]
        inds = self.inds[branch]

        parameter_info: list[ParameterInfo] = list(PARAMETER_INFO_REGISTRY[branch].values())

        parameter_names = [p.l3c_name for p in parameter_info]
        latex_names = [p.latex_name for p in parameter_info]
        units = [p.unit for p in parameter_info]

        metadata: SourceMetadata = self.curr.source_metadata[branch]

        metadata.parameter_info = parameter_names
        metadata.parameter_units = units

        detection_stats = self.detection_criteria.detection_statistics(samples, inds).tolist()

        posterior_files_map: dict[str, str] = {}
        label_to_statistic: dict[str, float] = {}

        num_leaves = samples.shape[1]
        for leaf_count in range(0, num_leaves, MAX_SOURCES_PER_BATCH):
            leaves = slice(leaf_count, min(leaf_count + MAX_SOURCES_PER_BATCH, num_leaves))

            samples_here = samples[:, leaves, :]
            num_sources_here = samples_here.shape[1]
            logger.debug(f"Number of sources in this batch: {num_sources_here}")

            # save to h5 file
            filename = f"{self.run_metadata.run_type}_{self.run_metadata.global_fit_codename}_{self.run_metadata.run_id}_{source_name}_posteriors_{num_sources_here}_{leaf_count}_{self.run_metadata.submission_timestamp}.h5"
            filepath = os.path.join(self.posterior_folders[branch], filename)
            with h5py.File(os.path.join(self.run_metadata.submission_folder, filepath), "w") as f:
                for leaf in range(num_sources_here):

                    source_label = self.get_source_label_registry[branch](samples_here[:, leaf, :], leaf_count + leaf)
                    label_to_statistic[source_label] = detection_stats[leaf_count + leaf]

                    leaf_samples = _build_leaf_samples_dict(
                        samples_here=samples_here,
                        parameter_names=parameter_names,
                        leaf=leaf,
                        branch=branch,
                        log_prior=self.log_prior,
                        log_likelihood=self.log_likelihood,
                    )
                    _save_posterior_dataset(f, source_label, leaf_samples)

                    posterior_files_map[source_label] = filepath

            logger.info(
                f"Saved posterior samples for branch '{branch}', leaves {leaves} to {filepath}"
            )

        detections = []
        known_injections_here = self.known_injections.get(branch, [])

        for j, source_idx in enumerate(posterior_files_map.keys()):
            detections.append({
                "source_id": str(source_idx),
                "posterior_id": f"posterior_{source_idx}",
                "comment": "",
                "quality_flag": int(metadata.quality_flags[j]) if j < len(metadata.quality_flags) else 0,
                "known_injection": known_injections_here[j] if j < len(known_injections_here) else "",
                "detection_statistic": float(label_to_statistic[source_idx]),
            })

        metadata.detection_statistic = list(label_to_statistic.values())
        
        metadata.detections = detections

        metadata.posterior_files = list(posterior_files_map.values())

        # now save the metadata for this source
        metadata_base_filename = f"{self.run_metadata.run_type}_{self.run_metadata.global_fit_codename}_{self.run_metadata.run_id}_{source_name}_{self.run_metadata.submission_timestamp}"

        metadata_h5_filepath = os.path.join(
            self.run_metadata.submission_folder, f"{metadata_base_filename}.h5"
        )
        metadata_json_filepath = os.path.join(
            self.run_metadata.submission_folder, f"{metadata_base_filename}.json"
        )

        with h5py.File(metadata_h5_filepath, "w") as f:
            source_group = f.create_group(name="sources")
            posterior_group = source_group.create_group(name="posterior_files")
            detection_group = source_group.create_group(name="detection")

            for j, (source_idx, posterior_file_path) in enumerate(posterior_files_map.items()):
                posterior_group.create_dataset(source_idx, data=str(posterior_file_path))

            detection_group.create_dataset("source_id", data=[d["source_id"] for d in detections])
            detection_group.create_dataset("posterior_id", data=[d["posterior_id"] for d in detections])
            detection_group.create_dataset("comment", data=[d["comment"] for d in detections])
            detection_group.create_dataset("quality_flag", data=[d["quality_flag"] for d in detections])
            detection_group.create_dataset("known_injection", data=[d["known_injection"] for d in detections])
            detection_group.create_dataset("detection_statistic", data=[d["detection_statistic"] for d in detections])

            _save_metadata_attributes(f, metadata)

        logger.info(f"Saved metadata for branch '{branch}' to {metadata_h5_filepath}")

        metadata.to_json(metadata_json_filepath)
        logger.info(f"Saved metadata for branch '{branch}' to {metadata_json_filepath}")


    def _save_stochastic_posterior(self):
        """Save the posterior samples for the stochastic component, if applicable."""

        if "psd" not in self.branches and "galfor" not in self.branches:
            logger.info("No stochastic component detected in the branches. Skipping stochastic posterior saving.")
            return

        # check if both psd and galfor are present and store everything together
        elif "psd" in self.branches and "galfor" in self.branches:
            branches_here = ["psd", "galfor"]

        # if only one of them is present, we save it as the stochastic posterior
        else:
            branch = "psd" if "psd" in self.branches else "galfor"
            branches_here = [branch]
        
        samples, inds = [], []
        parameter_info = []
        metadata_list = []

        for branch in branches_here:
            _samples = self.samples[branch]
            _inds = self.inds[branch]
            _parameter_info = list(PARAMETER_INFO_REGISTRY[branch].values())
            _metadata = self.curr.source_metadata[branch]

            samples.append(_samples)
            inds.append(_inds)
            parameter_info.extend(_parameter_info)
            metadata_list.append(_metadata)
            
        samples = np.concatenate(samples, axis=-1)
        inds = np.concatenate(inds, axis=-1)

        parameter_names = [p.l3c_name for p in parameter_info]
        latex_names = [p.latex_name for p in parameter_info]
        units = [p.unit for p in parameter_info]
        metadata: StochasticMetadata = StochasticMetadata(
            model_config={branch: metadata_list[i].model_config for i, branch in enumerate(branches_here)},
            frequency_ranges=metadata_list[0].frequency_ranges,  # we assume the same frequency ranges for all stochastic branches; this can be relaxed if needed
            prior_model=" ,".join(set(m.prior_model for m in metadata_list)),
            prior_model_code_link=" ,".join(set(m.prior_model_code_link for m in metadata_list)),
            prior_model_config={branch: metadata_list[i].prior_model_config for i, branch in enumerate(branches_here)},
            comment=" ,".join(m.comment for m in metadata_list if m.comment)
        )

        metadata.parameter_info = parameter_names
        metadata.parameter_units = units

        if samples.shape[1] > 1:
            raise NotImplementedError("multiple leaves detected, not implemented yet")

        samples_dict = _build_stochastic_samples_dict(
            samples=samples,
            parameter_names=parameter_names,
            branches_here=branches_here,
            log_prior=self.log_prior,
            log_likelihood=self.log_likelihood,
        )
        param_names = list(samples_dict.keys())
        physical_param_names = [p for p in param_names if p not in ["logprior", "loglikelihood"]]
        structured_array = _samples_dict_to_structured_array(samples_dict)

        effective_branch_name = "noise" #todo or stochastic?
        filename = f"{self.run_metadata.run_type}_{self.run_metadata.global_fit_codename}_{self.run_metadata.run_id}_{effective_branch_name}_posteriors_{self.run_metadata.submission_timestamp}.h5"
        filepath = os.path.join(self.posterior_folders[branches_here[0]], filename)

        with h5py.File(os.path.join(self.run_metadata.submission_folder, filepath), "w") as f:
            group = f.create_group(name=effective_branch_name)
            group.attrs["labels"] = ", ".join(physical_param_names)
            group.attrs["npars"] = len(physical_param_names)
            group.attrs["nsamples"] = len(structured_array)
            group.create_dataset("posterior", data=structured_array)

            # add the noise/p, noise/logprior, noise/loglikelihood as separate datasets for convenience to match the requested setup. I would prefer to keep them as part of the structured array, but this can be easily changed if needed.
            posterior_samples = np.stack([samples_dict[param] for param in physical_param_names], axis=-1) # we extract the posterior samples as a 2D array of shape (nsamples, npars) for convenience. These do not include the logprior and loglikelihood, which are stored separately.
            group.create_dataset("p", data=posterior_samples)
            group.create_dataset("logprior", data=samples_dict["logprior"])
            group.create_dataset("loglikelihood", data=samples_dict["loglikelihood"]) 

        metadata.posterior_file = filepath # we store the relative path to the posterior file in the metadata        
        metadata_base_filename = f"{self.run_metadata.run_type}_{self.run_metadata.global_fit_codename}_{self.run_metadata.run_id}_{effective_branch_name}_{self.run_metadata.submission_timestamp}"
        metadata_h5_filepath = os.path.join(
            self.run_metadata.submission_folder, f"{metadata_base_filename}.h5"
        )
        metadata_json_filepath = os.path.join(
            self.run_metadata.submission_folder, f"{metadata_base_filename}.json"
        )
        with h5py.File(metadata_h5_filepath, "w") as f:
            noise_group = f.create_group(name=effective_branch_name)
            posterior_group = noise_group.create_group(name="posterior_file")
            posterior_group.create_dataset("posterior", data=str(filepath))
            _save_metadata_attributes(noise_group, metadata)

        logger.info(f"Saved metadata for stochastic branch(es) to {metadata_h5_filepath}")

        metadata.to_json(metadata_json_filepath)
        logger.info(f"Saved metadata for  stochastic branch(es) to {metadata_json_filepath}")

    def save_data_and_residuals(self, acs: AnalysisContainerArray):
        """Save the input data and residuals to h5 files in the submission folder."""
        data_filepath = os.path.join(self.submission_folder, "input_data.h5")
        save_residuals(
            self.curr.general_info.input_data_residual_array,
            data_filepath,
        )

        logger.info(f"Saved input data to {data_filepath}")

        residuals_filepath = os.path.join(self.submission_folder, "residuals.h5")
        save_residuals(acs, residuals_filepath)
        logger.info(f"Saved residuals to {residuals_filepath}")

    def write_submission(self, acs: AnalysisContainerArray):
        """Run the full submission writing pipeline."""
        self.create_folders()
        self.prepare_samples_for_submission(acs)
        self.save_posteriors()

        # finally save the overall run metadata
        run_metadata_filepath = os.path.join(self.submission_folder, "global_metadata.json")
        self.run_metadata.to_json(run_metadata_filepath)
        logger.info(f"Saved overall run metadata to {run_metadata_filepath}")

        self.save_data_and_residuals(acs)



# === Save residuals ====

def save_residuals(data: AnalysisContainerArray | DataResidualArray, filepath: str):
    """
    Save residuals from the analysis container array to a file.
    
    Args:
        data: AnalysisContainerArray or DataResidualArray containing the residuals to save.
        filepath: Path to the file where the residuals should be saved.
    """
    with h5py.File(filepath, "w") as f:
        _save_residuals_to_handle(data, f)

def _save_residuals_to_handle(data, h5obj, label=None):
    if isinstance(data, DataResidualArray):
        residual_array = data.data_res_arr.arr
        if hasattr(residual_array, "get"):
            residual_array = residual_array.get()
        if label is None:
            label = "data"
        h5obj.create_dataset(label, data=residual_array)

    elif isinstance(data, AnalysisContainerArray):
        if label is None:
            label = "residual"
        for i, ac in enumerate(data.acs):
            _save_residuals_to_handle(ac.data_res_arr, h5obj, label=f"{label}_{i}")


# ─── DetectionCriteria ────────────────────────────────────────────────────────


class DetectionCriteria(ABC):
    """
    Abstract base class for source detection strategies.

    Subclasses decide which leaves in the trans-dimensional chain represent
    genuine detections, returning a boolean mask over nleaves_max.
    """

    @abstractmethod
    def detect(self, samples: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """
        Identify detected leaves.

        Args:
            samples: (n_independent, nwalkers, nleaves_max, ndim)
            inds:    (n_independent, nwalkers, nleaves_max) bool

        Returns:
            Boolean mask of shape (nleaves_max,).
        """


class OccupancyDetectionCriteria(DetectionCriteria):
    """
    Detects a leaf as a genuine source if it is active (inds=True) in at least
    `min_occupancy` fraction of the independent samples.

    This is the default strategy and requires no waveform calls.
    """

    def __init__(self, min_occupancy: float = 0.5):
        self.min_occupancy = min_occupancy

    def detection_statistics(self, samples: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """
        Compute the occupancy of each leaf across the independent samples.

        Args:
            samples: (n_independent, nleaves_max, ndim). Not used in this strategy but included for interface consistency.
            inds: (n_independent, nleaves_max) bool

        Returns:
            Occupancy array of shape (nleaves_max,), with values in [0, 1].
        """

        return inds.mean(axis=0)

    def detect(self, samples: np.ndarray, inds: np.ndarray) -> np.ndarray:

        if not (0 <= self.min_occupancy <= 1):
            raise ValueError(f"min_occupancy must be in [0, 1], got {self.min_occupancy}")
        occupancy = self.detection_statistics(samples, inds)

        return occupancy >= self.min_occupancy


class SNRDetectionCriteria(DetectionCriteria):
    """
    Detects a leaf as a genuine source if its MAP-sample SNR exceeds `snr_threshold`.

    `snr_fn` must be a callable that takes a (ndim,) parameter array (in sampling
    space) and returns a scalar SNR. The MAP sample is approximated as the last
    independent draw.
    """

    def __init__(self, snr_fn: Callable, snr_threshold: float = 7.0):
        self.snr_fn = snr_fn
        self.snr_threshold = snr_threshold

    def detect(self, samples: np.ndarray, inds: np.ndarray) -> np.ndarray:
        n_steps, n_walkers, n_leaves, ndim = samples.shape
        # Use the last independent step as a proxy for MAP
        map_samples = samples[-1]  # (nwalkers, nleaves_max, ndim)
        map_inds = inds[-1]  # (nwalkers, nleaves_max)

        # Average over walkers: take the walker with maximum occupancy per leaf
        leaf_occupancy = map_inds.mean(axis=0)  # (nleaves_max,)
        best_walker = map_inds.sum(axis=1).argmax()

        detected = np.zeros(n_leaves, dtype=bool)
        for i in range(n_leaves):
            if not map_inds[best_walker, i]:
                continue
            snr = self.snr_fn(map_samples[best_walker, i])
            detected[i] = snr >= self.snr_threshold

        return detected
