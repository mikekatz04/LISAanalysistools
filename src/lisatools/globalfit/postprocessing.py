"""
postprocessing.py
=================

Post-processing routines for the output of a global fit run.

Produces L3C-compliant HDF5 submission files and GitHub Pages-compatible
JSON manifests for web dashboard display.
"""

from __future__ import annotations

import dataclasses
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

import h5py
import numpy as np
from scipy.interpolate import CubicSpline
from eryn.backends import HDFBackend
from eryn.utils import get_integrated_act

if TYPE_CHECKING:
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
    "phi0": ParameterInfo("initial_phase", r"$\phi_0\,[\mathrm{rad}]$", "rad"),
    "cos_iota": ParameterInfo("inclination", r"$\iota\,[\mathrm{rad}]$", "rad"),
    "psi": ParameterInfo("polarization", r"$\psi\,[\mathrm{rad}]$", "rad"),
    "lam": ParameterInfo("ecliptic_longitude", r"$\lambda\,[\mathrm{rad}]$", "rad"),
    "sin_beta": ParameterInfo("ecliptic_latitude", r"$\beta\,[\mathrm{rad}]$", "rad"),
}

_MBH_PARAM_INFO: Dict[str, ParameterInfo] = {
    "logM": ParameterInfo("mass1", r"$M_1\,[M_\odot]$", "solMass"),
    "q": ParameterInfo("mass2", r"$M_2\,[M_\odot]$", "solMass"),
    "s1z": ParameterInfo("spin1", r"$a_1$", "dimensionless"),
    "s2z": ParameterInfo("spin2", r"$a_2$", "dimensionless"),
    "dist": ParameterInfo("distance", r"$d_L\,[\mathrm{Gpc}]$", "Gpc"),
    "phi_ref": ParameterInfo("phase_at_coalescence", r"$\phi_c\,[\mathrm{rad}]$", "rad"),
    "cos_iota": ParameterInfo("inclination", r"$\iota\,[\mathrm{rad}]$", "rad"),
    "psi": ParameterInfo("polarization", r"$\psi\,[\mathrm{rad}]$", "rad"),
    "lam": ParameterInfo("ecliptic_longitude", r"$\lambda\,[\mathrm{rad}]$", "rad"),
    "sin_beta": ParameterInfo("ecliptic_latitude", r"$\beta\,[\mathrm{rad}]$", "rad"),
    "t_plunge": ParameterInfo("coalescence_time", r"$t_c\,[\mathrm{s}]$", "s"),
}

_PARAM_INFO_REGISTRY: Dict[str, Dict[str, ParameterInfo]] = {
    "gb": _GB_PARAM_INFO,
    "mbh": _MBH_PARAM_INFO,
}

# Parameters omitted from L3C output (filled constants unused by analysts)
_EXCLUDE_REGISTRY: Dict[str, set] = {
    "gb": {"fddot"},
    "mbh": set(),
}

# Corrections applied after TransformContainer to fix waveform-gen vs L3C unit mismatches.
# The MBH transform applies gpc_to_mpc (Gpc→Mpc) for the waveform generator;
# L3C requires distance in Gpc, so we invert here.
_OUTPUT_CORRECTIONS_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "gb": {},
    "mbh": {"dist": lambda x: x * 1e-3},
}

source_types_names = dict(
    gb="GB",
    mbh="MBHB",
    emri="EMRI",
    sobh="SOBHB",
)

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

    dt = datetime.fromtimestamp(t + lisa_epoch_seconds, tz=timezone.utc)
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
    def ndims(self) -> dict:
        return self.backend.ndims

    @property
    def nleaves_max(self) -> dict:
        return self.backend.nleaves_max
    
    @property
    def transform_containers(self) -> dict:
        """Return the TransformContainer for each branch, if present."""

        _transforms = {name: self.curr.source_info[name].transform for name in self.branches if hasattr(self.curr.source_info[name], "transform")}
        return _transforms

    def store_cold_chains(self):
        """Extract cold chains and inds for all branches and cache them."""
        self._cold_chains = {}
        self._cold_inds = {}
        all_chains = self.backend.get_chain()
        all_inds = self.backend.get_inds()
        for branch in self.branches:
            self._cold_chains[branch] = all_chains[branch][:, :1]
            self._cold_inds[branch] = all_inds[branch][:, :1]

    @property
    def configured(self) -> bool:
        return hasattr(self, "_cold_chains")

    @property
    def branches(self) -> List[str]:
        return self.backend.branch_names

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
        max_act = int(np.ceil(max(np.max(act) for act in all_act.values())))
        logger.debug(f"Thinning factor: {max_act}")

        breakpoint()

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

        self.ess = ess

    @property
    def thinned_chains(self) -> Dict[str, np.ndarray]:
        if not hasattr(self, "_thinned_chains"):
            raise ValueError("Thinned samples have not been computed yet")

        return self._thinned_chains

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
                return self._thinned_chains[branch], self._thinned_inds[branch]
            return self._thinned_chains[branch]
        if return_inds:
            return self._thinned_chains, self._thinned_inds
        return self._thinned_chains
    
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
                logger.warning(f"No TransformContainer found for branch '{branch}'. Returning input samples.")
                return samples
            return self.transform_containers[branch].transform_base_parameters(samples)
        
        transformed = {}
        for b, s in samples.items():
            if b in self.transform_containers and self.transform_containers[b] is not None:
                transformed[b] = self.transform_containers[b].transform_base_parameters(s)
            else:
                logger.warning(f"No TransformContainer found for branch '{b}'. Returning input samples for this branch.")
                transformed[b] = s

        return transformed
    
    def process_samples(self, discard: int | float = 0.0, ess: int = 10000, return_inds: bool = False) -> dict | Tuple[dict, dict]:
        """
        Convenience method to run the end-to-end processing pipeline, starting from the raw samples.

        Args:
            discard: int or float (optional). If int, the number of initial samples to discard. Else, fraction of initial steps to discard
            ess: int (optional). Effective sample size
            return_inds: bool (optional). Whether to return the corresponding inds arrays.

        Returns:
            dict of branch to transformed samples, or a tuple of (samples_dict, inds_dict) if return_inds is True.
        """
        if not self.configured:
            self.store_cold_chains()

        samples, inds = self.get_independent_samples(discard=discard, ess=ess, return_inds=True)

        transformed_samples = self.transform(samples)

        if return_inds:
            return transformed_samples, inds

        return transformed_samples


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
        periodogram = np.abs(np.fft.rfft(x_windowed) * dt)**2 * norm

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
    
    decimated_indices = np.unique(np.round(np.logspace(0, np.log10(n_points - 1), n_decimated)).astype(int))
    
    decimated_freqs = freqs[decimated_indices]
    decimated_power = power[decimated_indices]

    return decimated_freqs, decimated_power


class GlobalFitPlotter:
    """
    Produce summary plots at the end of a global fit run, including posterior predictive plots and corner plots.
    """
    def __init__(self,
                 curr: CurrentInfoGlobalFit):

        self.curr = curr

        # corner plots: separated per leaf, joint over all leaves color coded by snr

    def convert_input_data_for_plotting(self,):
        """
        Convert the raw timeseries data into characteristic strain. We keep only the X channel for plotting.
        """
        if not hasattr(self.curr.general_info.data_processor, "individual_timeseries"):
            raise ValueError("Data processor does not have individual_timeseries attribute.")
        
        out = {}
        
        data_components = self.curr.general_info.data_processor.individual_timeseries

        times, combined = data_components.pop("TIMES"), data_components.pop("COMBINED")[0]

        dt = times[1] - times[0]
        freqs, periodogram = to_periodogram(combined, dt)
        freqs_h, power_avg = log_decimate(freqs, periodogram, f_min=self.curr.general_info.start_freq, f_max=self.curr.general_info.end_freq, decimation_factor=100)

        combined_char_strain = to_characteristic_strain(freqs_h, power_avg)
        out["freqs"] = freqs_h
        out["combined"] = combined_char_strain

        logger.info("Converted input data to characteristic strain for plotting.")

        covariance_matrix = data_components.pop("PSD_MATRIX")[0, :, 0, 0].real
        covariance_frequencies = data_components.pop("PSD_FREQUENCIES")
        _ = data_components.pop("PSD_TIMES")

        noise_amplitude = to_characteristic_strain(covariance_frequencies, covariance_matrix)
        out["noise_amplitude_estimate"] = CubicSpline(covariance_frequencies, noise_amplitude)(freqs_h)

        logger.info("Converted noise covariance to characteristic strain for plotting.")

        for k, v in tqdm(data_components.items(), desc="Converting components to characteristic strain"):
            
            freqs, periodogram = to_periodogram(v[0], dt)
            char_strain = to_characteristic_strain(freqs, periodogram)
            
            out[k.lower()] = CubicSpline(freqs, char_strain)(freqs_h)

        

# ─── Metadata extractors ─────────────────────────────────────────────────────

def _extract_sensitivity_metadata(gi) -> dict:
    """Extract noise model configuration from the initialised sensitivity backend."""
    backend = gi.sensitivity_backend
    return {
        "class": type(backend).__name__,
        "tdi_generation": backend.tdi_generation,
        "use_splines": backend.use_splines,
        "spline_order": backend.spline_order,
        "mask_percentage": backend.mask_percentage,
    }


def _extract_orbit_metadata(gi) -> dict:
    """Extract orbit configuration from the initialised orbits object."""
    orbits = gi.orbits
    out = {
        "class": type(orbits).__name__,
        "filename": orbits.filename,
        "armlength_m": float(orbits.armlength),
    }
    if hasattr(orbits, "frame"):
        out["frame"] = orbits.frame
    return out


# Maps GeneralSettings attribute names → RunMetadata __init__ parameter names.
_SETTINGS_TO_METADATA: Dict[str, str] = {
    "run_codename":                 "codename",
    "run_version":                  "version",
    "run_contact":                  "contact",
    "run_code_link":                "code_link",
    "run_input_data_link":          "input_data_link",
    "run_input_reference":          "input_reference",
    "run_noise_model":              "noise_model",
    "run_noise_model_code_link":    "noise_model_code_link",
    "run_waveform_model":           "waveform_model",
    "run_waveform_model_code_link": "waveform_model_code_link",
    "run_quality":                  "quality",
    "run_comment":                  "comment",
    "submission_folder":            "submission_parent_folder",
}

# ─── RunMetadata ──────────────────────────────────────────────────────────────


@dataclasses.dataclass
class RunMetadata:
    """
    Holds all metadata for a global fit run.

    User-supplied fields must be provided at construction.
    Auto-populated fields are filled by `from_curr`.
    """

    # user-supplied — required
    version: str
    contact: str
    code_link: str
    input_data_link: str
    input_reference: str
    noise_model: str
    noise_model_code_link: str
    waveform_model: str
    waveform_model_code_link: str
    submission_parent_folder: str

    # user-supplied — optional
    codename: str = "Erebor"
    quality: str = "nominal"
    comment: str = ""

    # set after detection is complete
    found_source_types: List[str] = dataclasses.field(default_factory=list)

    # auto-populated from curr — not set at init
    obs_begin: str = dataclasses.field(default="", init=False)
    obs_end: str = dataclasses.field(default="", init=False)
    effective_duration: str = dataclasses.field(default="", init=False)
    tdi_channels: List[str] = dataclasses.field(default_factory=list, init=False)
    searched_source_types: List[str] = dataclasses.field(default_factory=list, init=False)

    # extra info for web display (not part of L3C spec)
    _web_extras: Dict[str, Any] = dataclasses.field(default_factory=dict, init=False, repr=False)

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

        # Pull non-None metadata values from GeneralSettings; user_fields wins.
        auto = {
            meta_name: getattr(gi, settings_name)
            for settings_name, meta_name in _SETTINGS_TO_METADATA.items()
            if getattr(gi, settings_name, None) is not None
        }
        merged = {**auto, **user_fields}

        # Derive noise_model from the backend class name when not explicitly set.
        merged.setdefault("noise_model", type(gi.sensitivity_backend).__name__)

        instance = cls(**merged)
        breakpoint()
        instance.obs_begin = _seconds_to_l3c_datetime(gi.data_t0)
        instance.obs_end = _seconds_to_l3c_datetime(gi.data_t0 + gi.Tobs)
        instance.effective_duration = _seconds_to_duration_str(gi.Tobs)
        instance.tdi_channels = _infer_tdi_channels(curr)
        instance.searched_source_types = get_source_types(curr)

        instance._web_extras = {
            "Tobs_s": float(gi.Tobs),
            "dt_s": float(gi.dt),
            "basis_domain": gi.basis_domain,
            "start_freq_hz": float(gi.start_freq) if gi.start_freq is not None else None,
            "end_freq_hz": float(gi.end_freq) if gi.end_freq is not None else None,
            "nwalkers": gi.nwalkers,
            "ntemps": gi.ntemps,
            "sensitivity_backend": _extract_sensitivity_metadata(gi),
            "orbits": _extract_orbit_metadata(gi),
        }
        return instance

    def to_l3c_dict(self) -> dict:
        """Return a dict matching the l2_output_metadata template keys exactly."""
        return {
            "global_fit_codename": self.codename,
            "global_fit_version": self.version,
            "global_fit_release_date": datetime.now(tz=timezone.utc).strftime("%Y.%m.%d"),
            "global_fit_contact": self.contact,
            "input_data_link": self.input_data_link,
            "input_reference": self.input_reference,
            "global_fit_code_link": self.code_link,
            "observation_period_begin": self.obs_begin,
            "observation_period_end": self.obs_end,
            "effective_observation_duration": self.effective_duration,
            "quality": self.quality,
            "searched_source_types_list": self.searched_source_types,
            "found_source_types_list": self.found_source_types,
            "noise_model": self.noise_model,
            "noise_model_code_link": self.noise_model_code_link,
            "noise_model_config_file_link": "",
            "waveform_model": self.waveform_model,
            "waveform_model_code_link": self.waveform_model_code_link,
            "waveform_model_config_file_link": "",
            "tdi_channels": self.tdi_channels,
            "list_of_detected_sources": ", ".join(self.found_source_types),
            "comment": self.comment,
        }

    def to_web_dict(self) -> dict:
        """Return a richer dict for web display, extending the L3C dict with run config."""
        d = self.to_l3c_dict()
        d.update(self._web_extras)
        return d
    
    @property
    def submission_folder(self) -> str:
        """Return the full path to the submission folder for this run."""
        
        run_type, run_id = self.version.split("_")
        return os.path.join(self.submission_parent_folder, run_type, run_id)
        
        #return os.path.join(self.submission_parent_folder, f"{self.codename}_v{self.version}")
    
class SubmissionWriter(BackendConsumer):
    """
    BackendConsumer subclass that produces L3C-compliant HDF5 submission files and JSON manifests.

    This class is responsible for applying the detection criteria to identify genuine sources,
    and for writing the final outputs in the required formats.
    """
    def __init__(self, 
                 curr: CurrentInfoGlobalFit = None,
                 backend: HDFBackend | str = None,
                 ess: int = 10000,
                 detection_criteria: DetectionCriteria = None):

        super().__init__(curr=curr, backend=backend)

        self.samples, self.inds = self.process_samples(ess=ess, return_inds=True)

        self.detection_criteria = detection_criteria or OccupancyDetectionCriteria()

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

    def detect(self, samples: np.ndarray, inds: np.ndarray) -> np.ndarray:
        n_steps, n_walkers, n_leaves = inds.shape[:3]
        flat_inds = inds.reshape(n_steps * n_walkers, n_leaves)
        occupancy = flat_inds.mean(axis=0)
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
