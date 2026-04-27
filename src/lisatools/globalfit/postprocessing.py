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
from turtle import st
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
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


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _seconds_to_l3c_datetime(t: float) -> str:
    """Convert a UTC timestamp in seconds to the L3C format yyyy.mm.dd.hh.mm.ss."""
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
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
        max_act = int(max(np.max(act) for act in all_act.values()))
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
            if branch not in self.transform_containers:
                logger.warning(f"No TransformContainer found for branch '{branch}'. Returning input samples.")
                return samples
            return self.transform_containers[branch].transform_base_parameters(samples)
        
        transformed = {}
        for b, s in samples.items():
            if b in self.transform_containers:
                transformed[b] = self.transform_containers[b].transform_base_parameters(s)
            else:
                logger.warning(f"No TransformContainer found for branch '{b}'. Returning input samples for this branch.")
                transformed[b] = s

        return transformed


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
        Construct RunMetadata, auto-populating timing and channel fields from `curr`.

        Args:
            curr: Live CurrentInfoGlobalFit object.
            **user_fields: All user-supplied fields (version, contact, etc.).
        """
        instance = cls(**user_fields)
        gi = curr.general_info

        instance.obs_begin = _seconds_to_l3c_datetime(gi.data_t0)
        instance.obs_end = _seconds_to_l3c_datetime(gi.data_t0 + gi.Tobs)
        instance.effective_duration = _seconds_to_duration_str(gi.Tobs)
        instance.tdi_channels = _infer_tdi_channels(curr)
        instance.searched_source_types = list(curr.source_info.keys())

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


# ─── ParameterMapper ──────────────────────────────────────────────────────────


class ParameterMapper:
    """
    Transforms posterior samples from sampling space to physical L3C space.

    Uses the TransformContainer from the source Setup for the mathematical
    transforms (same functions used by the waveform generator), then renames
    output_basis parameters to L3C names and applies any unit corrections
    needed for output (e.g., Mpc → Gpc for MBH distance).
    """

    def __init__(
        self,
        transform,
        output_basis: List[str],
        param_info: Dict[str, ParameterInfo],
        exclude: set = None,
        output_corrections: Dict[str, Callable] = None,
    ):
        self._transform = transform
        self._output_basis = output_basis
        self._param_info = param_info
        self._exclude = exclude or set()
        self._output_corrections = output_corrections or {}

    @classmethod
    def from_curr(cls, curr: CurrentInfoGlobalFit, source_type: str) -> "ParameterMapper":
        """
        Build a ParameterMapper from the TransformContainer in curr.source_info[source_type].

        Raises ValueError if source_type has no registered parameter info.
        """
        if source_type not in _PARAM_INFO_REGISTRY:
            raise ValueError(
                f"No ParameterMapper defined for source type '{source_type}'. "
                f"Registered types: {list(_PARAM_INFO_REGISTRY.keys())}"
            )
        setup = curr.source_info[source_type]
        return cls(
            transform=setup.transform,
            output_basis=list(setup.transform.output_basis),
            param_info=_PARAM_INFO_REGISTRY[source_type],
            exclude=_EXCLUDE_REGISTRY.get(source_type, set()),
            output_corrections=_OUTPUT_CORRECTIONS_REGISTRY.get(source_type, {}),
        )

    def _active_output_params(self) -> List[str]:
        return [p for p in self._output_basis if p not in self._exclude and p in self._param_info]

    @property
    def l3c_names(self) -> List[str]:
        return [self._param_info[p].l3c_name for p in self._active_output_params()]

    @property
    def latex_names(self) -> List[str]:
        return [self._param_info[p].latex_name for p in self._active_output_params()]

    @property
    def units(self) -> List[str]:
        return [self._param_info[p].unit for p in self._active_output_params()]

    def map(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply the full transform chain and rename to L3C parameter names.

        Args:
            samples: (n_samples, ndim_sampling) array in sampling parameter space.

        Returns:
            Dict keyed by L3C parameter name, each value is a (n_samples,) array.
        """
        # TransformContainer accepts (n_samples, ndim_input) and returns (n_samples, ndim_output)
        physical = self._transform(samples)

        result = {}
        for i, param_name in enumerate(self._output_basis):
            if param_name in self._exclude or param_name not in self._param_info:
                continue
            values = physical[:, i].copy()
            if param_name in self._output_corrections:
                values = self._output_corrections[param_name](values)
            result[self._param_info[param_name].l3c_name] = values

        return result


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


# ─── L3CSubmissionWriter ──────────────────────────────────────────────────────


class L3CSubmissionWriter:
    """
    Orchestrates the full L3C submission output for one or more source types.

    For each source type, produces:
    - A main HDF5 file (EREBOR_{SOURCE}.h5) with L3C metadata and MAP estimates.
    - A directory (posteriors_erebor_{source}/) with one HDF5 file per detection.
    """

    TEAM_NAME = "erebor"

    def __init__(
        self,
        backend_consumer: BackendConsumer,
        run_metadata: RunMetadata,
        curr: CurrentInfoGlobalFit,
        output_dir: str,
        detection_criteria: Optional[Dict[str, DetectionCriteria]] = None,
        discard_fraction: float = 0.0,
    ):
        self.consumer = backend_consumer
        self.metadata = run_metadata
        self.curr = curr
        self.output_dir = output_dir
        self.discard_fraction = discard_fraction
        self.detection_criteria = detection_criteria or {}
        os.makedirs(output_dir, exist_ok=True)

    def _default_criteria(self, source_type: str) -> DetectionCriteria:
        return OccupancyDetectionCriteria(min_occupancy=0.5)

    def write(self, source_types: List[str]):
        """Run the full submission pipeline for the given source types."""
        found = []
        for source_type in source_types:
            n_det = self._write_source_type(source_type)
            if n_det > 0:
                found.append(source_type)
        self.metadata.found_source_types = found

    def _write_source_type(self, source_type: str) -> int:
        samples, inds = self.consumer.get_independent_samples(
            source_type, discard_fraction=self.discard_fraction
        )
        criteria = self.detection_criteria.get(source_type) or self._default_criteria(source_type)
        detected_mask = criteria.detect(samples, inds)
        detected_indices = np.where(detected_mask)[0]

        mapper = ParameterMapper.from_curr(self.curr, source_type)

        posteriors_dir = os.path.join(self.output_dir, f"posteriors_{self.TEAM_NAME}_{source_type}")
        os.makedirs(posteriors_dir, exist_ok=True)

        posterior_files = []
        map_estimates = []

        for det_i, leaf_idx in enumerate(detected_indices):
            # Flatten steps × walkers for this leaf, keep only active samples
            leaf_samples = samples[:, :, leaf_idx, :].reshape(-1, samples.shape[-1])
            leaf_active = inds[:, :, leaf_idx].reshape(-1)
            leaf_samples = leaf_samples[leaf_active]

            if len(leaf_samples) == 0:
                continue

            physical = mapper.map(leaf_samples)
            map_est = {k: float(np.median(v)) for k, v in physical.items()}
            map_estimates.append(map_est)

            fname = os.path.join(posteriors_dir, f"source_{det_i}.h5")
            self._write_posterior_file(fname, physical, source_type)
            posterior_files.append(fname)

        self._write_main_hdf5(source_type, map_estimates, posterior_files, mapper)
        return len(posterior_files)

    def _write_posterior_file(
        self,
        filepath: str,
        physical: Dict[str, np.ndarray],
        source_type: str,
    ):
        with h5py.File(filepath, "w") as f:
            f.attrs["source_type"] = source_type.upper()
            f.attrs["prior_model"] = "uniform"
            f.attrs["prior_model_code_link"] = self.metadata.code_link
            f.attrs["prior_model_config_file_link"] = ""
            f.attrs["vb_references"] = ""
            f.attrs["estimation_method"] = "MCMC (Eryn)"

            for l3c_name, values in physical.items():
                f.create_dataset(l3c_name, data=values.astype(np.float64))

    def _write_main_hdf5(
        self,
        source_type: str,
        map_estimates: List[Dict[str, float]],
        posterior_files: List[str],
        mapper: ParameterMapper,
    ):
        team = self.TEAM_NAME.upper()
        src = source_type.upper()
        filepath = os.path.join(self.output_dir, f"{team}_{src}.h5")

        with h5py.File(filepath, "w") as f:
            # / — l2_output_metadata attributes
            for k, v in self.metadata.to_l3c_dict().items():
                f.attrs[k] = (
                    json.dumps(v) if isinstance(v, list) else (str(v) if v is not None else "")
                )

            # /sources — dataset_metadata attributes + N×P MAP detections matrix
            sources_grp = f.create_group("sources")
            sources_grp.attrs["source_type"] = source_type.upper()
            sources_grp.attrs["prior_model"] = "uniform"
            sources_grp.attrs["prior_model_code_link"] = self.metadata.code_link
            sources_grp.attrs["prior_model_config_file_link"] = ""
            sources_grp.attrs["vb_references"] = ""

            if map_estimates:
                det_matrix = np.array(
                    [[est[k] for k in mapper.l3c_names] for est in map_estimates],
                    dtype=np.float64,
                )  # (N_detections, N_params)
                ds = sources_grp.create_dataset("detections", data=det_matrix)
                ds.attrs["columns"] = json.dumps(mapper.l3c_names)
                ds.attrs["latex_names"] = json.dumps(mapper.latex_names)
                ds.attrs["units"] = json.dumps(mapper.units)

            # /posteriors_files — path strings to per-source posterior HDF5 files
            pf_grp = f.create_group("posteriors_files")
            for i, fpath in enumerate(posterior_files):
                pf_grp.attrs[f"source_{i}"] = fpath


# ─── WebManifestWriter ────────────────────────────────────────────────────────


class WebManifestWriter:
    """
    Writes GitHub Pages-compatible JSON output for web dashboard display.

    Produces:
    - manifest.json  — run metadata + per-source summary statistics (median + 90% CI)
    - {source_type}/source_N_samples.json — full posterior samples per detection

    For large catalogs, `max_sources_with_full_samples` limits how many
    per-source JSON files are written (summary stats are always written for all).
    """

    def __init__(
        self,
        l3c_writer: L3CSubmissionWriter,
        web_output_dir: str,
        max_sources_with_full_samples: Optional[int] = None,
    ):
        self.l3c_writer = l3c_writer
        self.web_output_dir = web_output_dir
        self.max_full = max_sources_with_full_samples
        os.makedirs(web_output_dir, exist_ok=True)

    def write(self, source_types: List[str]):
        """Write manifest.json and per-source sample files for all source types."""
        manifest: Dict[str, Any] = {
            "run": self.l3c_writer.metadata.to_web_dict(),
            "sources": {},
        }
        for source_type in source_types:
            manifest["sources"][source_type] = self._write_source_type_web(source_type)

        manifest_path = os.path.join(self.web_output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_source_type_web(self, source_type: str) -> dict:
        samples, inds = self.l3c_writer.consumer.get_independent_samples(
            source_type,
            discard_fraction=self.l3c_writer.discard_fraction,
        )
        criteria = self.l3c_writer.detection_criteria.get(
            source_type
        ) or self.l3c_writer._default_criteria(source_type)
        detected_mask = criteria.detect(samples, inds)
        detected_indices = np.where(detected_mask)[0]

        mapper = ParameterMapper.from_curr(self.l3c_writer.curr, source_type)
        src_dir = os.path.join(self.web_output_dir, source_type)
        os.makedirs(src_dir, exist_ok=True)

        detections_meta = []
        for det_i, leaf_idx in enumerate(detected_indices):
            leaf_samples = samples[:, :, leaf_idx, :].reshape(-1, samples.shape[-1])
            leaf_active = inds[:, :, leaf_idx].reshape(-1)
            leaf_samples = leaf_samples[leaf_active]
            if len(leaf_samples) == 0:
                continue

            physical = mapper.map(leaf_samples)
            entry = {
                "id": det_i,
                "summary": self._summary_stats(physical),
            }

            if self.max_full is None or det_i < self.max_full:
                samples_file = f"{source_type}/source_{det_i}_samples.json"
                self._write_samples_json(
                    os.path.join(self.web_output_dir, samples_file),
                    physical,
                    mapper,
                )
                entry["samples_file"] = samples_file

            detections_meta.append(entry)

        return {
            "n_detections": len(detected_indices),
            "parameters": mapper.l3c_names,
            "latex_names": mapper.latex_names,
            "units": mapper.units,
            "detections": detections_meta,
        }

    @staticmethod
    def _summary_stats(physical: Dict[str, np.ndarray]) -> Dict[str, dict]:
        return {
            name: {
                "median": float(np.median(vals)),
                "ci_90": [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))],
            }
            for name, vals in physical.items()
        }

    @staticmethod
    def _write_samples_json(
        filepath: str,
        physical: Dict[str, np.ndarray],
        mapper: ParameterMapper,
    ):
        payload = {
            "parameters": mapper.l3c_names,
            "latex_names": mapper.latex_names,
            "units": mapper.units,
            "samples": {name: vals.tolist() for name, vals in physical.items()},
        }
        with open(filepath, "w") as f:
            json.dump(payload, f)
