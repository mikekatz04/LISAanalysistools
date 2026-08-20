from __future__ import annotations

import logging
import dataclasses
from typing import Dict, Any
import numpy as np
from copy import deepcopy

try:
    import cupy as cp
except ImportError:
    raise ImportError("cupy is required for GB clustering. Please install cupy to use this feature.")

from gbgpu.gbgpu import GBGPU
from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend
from lisatools.globalfit.state import GFState, GBState
from lisatools.domains import FDSettings
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.globalfit.gbclustering import gather_gb_samples
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.globalfit.stock.erebor import GBSetup, PSDSetup, GalForSetup
from mojito_input.galaxy_foreground_psd_global_fit_settings import get_global_fit_settings

logger = logging.getLogger(__name__)


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

PARAMETER_INFO_REGISTRY: Dict[str, Dict[str, ParameterInfo]] = {
    "gb": _GB_PARAM_INFO,
} # for the other sources, look in lisatools.globalfit.postprocessing

#*=====================================================================================
# set your data file here
backend_file_name = "/sps/lisaf/crondeel/erebor/mojito_light/full_galaxy_psd_run_1/CDL1run1_v3_fd_parameter_estimation_main.h5"


curr = get_global_fit_settings()
gb_info: GBSetup = curr.source_info["gb"] # type: ignore

assert gb_info.initialize_kwargs is not None, "GBSetup must have initialize_kwargs defined."
gb_wave_gen = GBGPU(**gb_info.initialize_kwargs) 
gb_wave_gen.gpus = [0] # curr.general_info.gpus[:1] # use only one GPU for the clustering
setattr(gb_wave_gen, "d_d", 0.0)

cluster_kwargs: Dict[str, Any] = dict(
    num_compare_samples=200,
    samples_keep=5,
    thin_by=1,
    snr_lim_first_cut=7.0,
    snr_lim_second_cut=5.0,
    overlap_lim=0.7,
    snr_diff_lim=20.0,
)

logger.info("loading backend...")

reader = GFHDFBackend(
    backend_file_name, 
    sub_state_bases=curr.engine_info.branch_states, 
    sub_backend=curr.engine_info.branch_backends
)

chain = reader.get_chain(
    branch_names=curr.engine_info.branch_names, 
    temp_index=0, 
)
inds = reader.get_inds(
    branch_names=curr.engine_info.branch_names, 
    temp_index=0, 
)

breakpoint()

coords_final = ...
inds_final = ... 

state = GFState(
    coords_final,
    inds=inds_final,
    random_state=np.random.get_state(),
    sub_state_bases=curr.engine_info.branch_states,
)

# taken from GlobalFit.setup_acs
acs_tmp = []
assert curr.general_info.nwalkers
for w in range(curr.general_info.nwalkers):
    data_res_arr = deepcopy(curr.general_info.input_data_residual_array)

    if "psd" in state.branches_coords.keys():
        psd_params = state.branches_coords["psd"][0, w, 0]
        psd_info: PSDSetup = curr.source_info["psd"] # type: ignore
        psd_params = (
            psd_info.transform.both_transforms(psd_params)
            if psd_info.transform is not None
            else psd_params
        )
        # need to generalize for other stochastic functions
        if "galfor" in state.branches_coords.keys():
            galfor_params = state.branches_coords["galfor"][0, w, 0]
            galfor_info: GalForSetup = curr.source_info["galfor"] # type: ignore
            galfor_params = (
                galfor_info.transform.both_transforms(galfor_params)
                if galfor_info.transform is not None
                else galfor_params
            )
        else:
            galfor_params = None
        sens_here = curr.general_info.sensitivity_backend(
            f"walker_{w}", psd_params, galfor_params=galfor_params # type: ignore
        )
    else:
        fixed_psd_kwargs: Dict[str, Any] = curr.general_info.fixed_psd_kwargs # type: ignore
        sens_here = curr.general_info.sensitivity_backend(
            f"walker_{w}", **fixed_psd_kwargs
        )

    acs_tmp.append(AnalysisContainer(deepcopy(data_res_arr), deepcopy(sens_here)))

gpus = curr.general_info.gpus
acs = AnalysisContainerArray(acs_tmp, gpus=gpus)
#* We do not need to remove the GBs since we do not need data|source inner products for clustering.
max_logl_walker = np.argmax(acs.likelihood()).item()
max_logl_acs: AnalysisContainer = acs[max_logl_walker] # type: ignore
sens_mat = max_logl_acs.sens_mat

assert acs.gpus is not None, "AnalysisContainerArray must have a GPU attribute for clustering."
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
    **cluster_kwargs,
)

logger.info(f"Completed clustering. Number of groups found: {len(groups)}")

num_in_groups = np.asarray([len(tmp) for tmp in groups])

assert isinstance(reader.nwalkers, int), "reader.nwalkers must be an integer"
keep = num_in_groups > reader.nwalkers * cluster_kwargs['samples_keep'] / 2

logger.info(
    f"Groups passing sample count filter: {keep.sum()} / {len(keep)}. "
    f"num_in_groups: {num_in_groups}"
)
max_num_source = max([tmp.shape[0] for tmp in groups])

samples_fin = np.full((len(groups), max_num_source, groups[0].shape[-1]), np.nan)
for i, group in enumerate(groups):
    samples_fin[i, : len(group)] = group

samples_fin = samples_fin[keep] # shape (nclusters, nsteps, ndim)
num_in_groups_fin = num_in_groups[keep]

samples_fin = samples_fin.transpose(1, 0, 2)
inds_fin = np.isfinite(samples_fin[..., 0])

# now transform again from the sampling space to the physical space
if gb_info.transform is not None:
    samples_fin = gb_info.transform.both_transforms(samples_fin)
else:
    pass # samples are already in physical basis

# now sort by frequency
frequency_idx = list(PARAMETER_INFO_REGISTRY["gb"].keys()).index("f0")
mean_frequencies = samples_fin[:, :, frequency_idx].mean(axis=0) # shape (nleaves_max,)
sorted_indices = np.argsort(mean_frequencies)
samples_fin = samples_fin[:, sorted_indices, :]
inds_fin = inds_fin[:, sorted_indices]