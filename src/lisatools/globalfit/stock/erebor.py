from __future__ import annotations

import dataclasses
import typing
from typing import Any, Optional

import h5py
import numpy as np

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp

import logging

from eryn.backends import HDFBackend as eryn_Backend
from eryn.moves.tempering import make_ladder
from eryn.state import State as ErynState
from eryn.state import Branch as ErynBranch
from eryn.utils import TransformContainer
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.sources.utils import ecliptic_to_icrs
from lisatools.utils.constants import YRSID_SI, PC_SI

from ..engine import Settings, Setup, GeneralSetup
from ..loginfo import init_logger

from ..priors import (
    GBConfig, 
    MBHConfig, 
    PSDAnalyticalConfig, 
    GalForConfig,
    EMRIConfig,
    HyperConfig
)


#* ==============================================================================
#* GALACTIC BINARIES (GB)
#* ==============================================================================

@dataclasses.dataclass
class GBSettings(Settings):
    prior_file: Optional[str] = None
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    oversample: int = 4
    extra_buffer: int = 5
    start_resample_iter: Optional[typing.Tuple[int]] = (-1,)  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    num_repeat_proposals: int = 100
    search_kwargs: Optional[dict] = None
    start_freq_ind: Optional[int] = 0  # goes into GPU for start of data stream
    waveform_kwargs: dict = dataclasses.field(default_factory=dict)


from ..hdfbackend import GBHDFBackend
from ..state import GBState

class GBSetup(Setup, GBSettings):
    def __init__(self, gb_settings: GBSettings, source_config = None):
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        self.logger = init_logger(
            filename="gb_setup.log", 
            level=logging.DEBUG, 
            name="GBSetup", 
            log_dir=getattr(self, 'log_dir', None)
        )
        if source_config is None:
            self.source_config = GBConfig(
                prior_file=self.prior_file, 
                use_cupy=True, # TODO grab dynamically
                return_gpu=False
            )
        else:
            self.source_config = source_config
        
        self.init_setup()

    def init_sampling_info(self):   
        
        if self.new_f0_lims is not None:
            # NOTE also update fdot when there is an explicit fdot range dictated
            # For the example below we have a joint prior on f0 and fdot, 
            # so only f0 needs to be reset.
            self.source_config.update_prior_kwargs(
                ("f0", "fdot"), 
                f0_min=self.new_f0_lims[0].item() * 1e3, 
                f0_max=self.new_f0_lims[1].item() * 1e3 # sampling in mHz
            )
        
        if self.priors is None:
            self.priors = self.source_config.priors
        
        if self.transform is None:
            self.transform = self.source_config.transform
        
        if self.periodic is None:
            self.periodic = self.source_config.periodic.periodic

        if self.betas is None:
            # snrs_ladder = np.array(
            #     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
            #      15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2]
            # )
            ntemps_pe = 24 # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        # self.waveform_kwargs = dict(
        #     dt=self.dt,
        #     T=self.Tobs,
        #     use_c_implementation=True,
        #     oversample=self.oversample,
        #     start_freq_ind=self.start_freq_ind,
        #     tdi_channel_setup=self.tdi_setup,
        #     tdi2=self.use_tdi2
        # )

        # self.group_proposal_kwargs = dict(n_iter_update=1, live_dangerously=True, a=1.75, num_repeat_proposals=70)
        
        if self.search_kwargs is None:
            self.search_kwargs: typing.Dict[str, Any] = dict(
                nwalkers = 32,
                ntemps = 24,
                shutoff_band_iteration = 5,
                shutoff_frequency_threshold = None, # 4e-3 
                burn_1 = 200,
                nsteps_1 = 200,
                snr_threshold = 8.0,
                burn_2 = 500,
                nsteps_2 = 500,
                refit_start_iteration = 5 
            )

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        self.init_band_structure()
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        if self.branch_state is None:
            self.branch_state = GBState

        if self.branch_backend is None:
            self.branch_backend = GBHDFBackend

    def init_band_structure(self):
        # band separation setup
        if self.oversample is None and self.Tobs < YRSID_SI / 2.0:
            self.oversample = 2
        elif self.oversample is None:
            self.oversample = 4

        assert self.oversample >= 1

        # TODO: assign to binned f or leave general? probably better to be general
        band_edges_in_reverse_order = [self.end_freq]
        current_N = get_N(1e-30, self.end_freq, self.Tobs, oversample=self.oversample).item()
        min_N = get_N(1e-30, self.start_freq, self.Tobs, oversample=self.oversample).item()
        band_N_vals_reverse_order = [current_N]

        current_freq = self.end_freq - self.df / 2
        last_freq = self.end_freq
        while current_freq > self.start_freq + min_N * self.df:
            current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
            band_edges_in_reverse_order.append(current_freq)
            current_N = get_N(1e-30, current_freq, self.Tobs, oversample=self.oversample).item()
            band_N_vals_reverse_order.append(current_N)
            last_freq = current_freq
        
        band_edges_in_reverse_order.append(
            last_freq - (current_N * 2 + self.extra_buffer) * self.df
        )

        band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
        band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]
        
        # trim edges to avoid out of bound indexing
        self.band_edges = band_edges[2:-1]
        self.band_N_vals = band_N_vals[2:-1]

        self.new_f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]

        self.num_sub_bands = len(self.band_edges) - 1
        
        self.logger.info(
            f"GB f0 prior range is set from {round(self.new_f0_lims[0],7)} to {round(self.new_f0_lims[1],7)}"
        )
        self.logger.info(f"The number of subbands is {self.num_sub_bands}")
        self.logger.info(f"Min freq of subbands is {self.band_edges.min()}")
        self.logger.info(f"Max freq of subbands is {self.band_edges.max()}")

        
     
#* ==============================================================================
#* MASSIVE BLACK HOLE BINARIES (MBHB)
#* ==============================================================================
        
def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc


def gpc_to_mpc(x):
    """
    Transform from Gpc to Mpc, for distance prior.
    """
    return x * 1e3

def mT_Q(M, Q):
    """
    Transform from total mass and mass ratio m1/m2 to m1 and m2.
    """
    m2 = M / (1 + Q)
    m1 = Q * m2
    assert np.all(m1 >= m2), "m1 should be the larger mass"
    return m1, m2

#! SKIPPING FOR NOW, MBHB is quite difficult due to the complex transforms, maybe we should define priors for them

from bbhx.utils.transform import mT_q, LISA_to_SSB
from eryn.moves import Move

from ..hdfbackend import MBHHDFBackend
from ..state import MBHState

@dataclasses.dataclass
class MBHSettings(Settings):
    waveform_kwargs: Optional[dict] = None
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 200
    mbh_search_file_key: Optional[str] = "_mbh_search_tmp_file"
    injection: Optional[np.ndarray] = None


class MBHSetup(Setup):
    def __init__(self, mbh_settings: MBHSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(mbh_settings)

        level = logging.DEBUG
        name = "MBHSetup"
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):

        input_basis = [
            "logM",
            "Q",
            "s1z",
            "s2z",
            "dist",
            "phi_ref",
            "cos_iota",
            "psi",
            "lam",
            "sin_beta",
            "t_plunge",
        ]

        if self.transform is None:

            output_basis = [
                "logM",
                "Q",
                "s1z",
                "s2z",
                "dist",
                "phi_ref",
                # "f_ref",
                "cos_iota",
                "psi",
                "lam",
                "sin_beta",
                # "psi",
                "t_plunge",
            ]

            mbh_transform_fn_in = {
                "logM": np.exp,
                "dist": gpc_to_mpc,
                "cos_iota": np.arccos,
                "sin_beta": np.arcsin,
                ("logM", "Q"): mT_Q,
                ("t_plunge", "lam", "sin_beta", "psi"): LISA_to_SSB,
                ("lam", "sin_beta", "psi"): ecliptic_to_icrs,
            }

            # for transforms
            # mbh_fill_dict = {"f_ref": 0.0}
            mbh_fill_dict = {}

            self.transform = TransformContainer(
                input_basis=input_basis,
                output_basis=output_basis,
                parameter_transforms=mbh_transform_fn_in,
                fill_dict=mbh_fill_dict,
            )

        if self.periodic is None:
            self.periodic = {"mbh": {"phi_ref": 2 * np.pi, "lam": 2 * np.pi, "psi": np.pi}}

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            priors_mbh = {
                "logM": uniform_dist(np.log(1e5), np.log(1e8)),
                "Q": log_uniform(1., 10.),
                "s1z": uniform_dist(-0.99999999, +0.99999999),
                "s2z": uniform_dist(-0.99999999, +0.99999999),
                "dist": uniform_dist(1, 150.0), # uniform_dist(0.01, 1000.0),
                "phi_ref": uniform_dist(0.0, 2 * np.pi),
                "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "psi": uniform_dist(0.0, np.pi), #is this right?
                "lam": uniform_dist(0.0, 2 * np.pi),
                "sin_beta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "t_plunge": uniform_dist(0.0, self.Tobs + 3600.0),
            }

            self.priors = {"mbh": ProbDistContainer(priors_mbh)}

        if self.betas is None:
            snrs_ladder = np.array(
                [
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    7.5,
                    10.0,
                    15.0,
                    20.0,
                    35.0,
                    50.0,
                    75.0,
                    125.0,
                    250.0,
                    5e2,
                ]
            )
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.waveform_kwargs is None:
            self.logger.warning(
                "No waveform kwargs provided for MBHSetup, using defaults. These are the legacy BBHx settings"
            )
            self.waveform_kwargs = dict(
                modes=[(2, 2)],
                length=1024,
            )

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            from lisatools.sampling.moves.skymodehop import SkyMove

            angles_map = dict(cosinc=6, psi=7, lam=8, sinbeta=9)

            self.inner_moves = [
                (SkyMove(ind_map=angles_map, which="both"), 0.02),
                (SkyMove(ind_map=angles_map, which="long"), 0.05),
                (SkyMove(ind_map=angles_map, which="lat"), 0.05),
                (StretchMove(), 0.88),
            ]

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        if self.branch_state is None:
            self.branch_state = MBHState

        if self.branch_backend is None:
            self.branch_backend = MBHHDFBackend


#* ==============================================================================
#* EXTREME MASS RATIO INSPIRALS (EMRI)
#* ==============================================================================

from eryn.moves import Move
from ..hdfbackend import EMRIHDFBackend
from ..state import EMRIState

@dataclasses.dataclass
class EMRISettings(Settings):
    prior_file: Optional[str] = None
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    emri_search_file_key: Optional[str] = "_emri_search_tmp_file"

        
class EMRISetup(Setup, EMRISettings):
    def __init__(self, emri_settings: EMRISettings):
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, emri_settings)

        self.logger = init_logger(
            filename="emri_setup.log", 
            level=logging.DEBUG, 
            name="EMRISetup", 
            log_dir=getattr(self, 'log_dir', None)
        )

        self.source_config = EMRIConfig(
            prior_file=self.prior_file, 
            use_cupy=True, # TODO grab dynamically
            return_gpu=False
        )
        self.init_setup()

    def init_sampling_info(self):

        if self.priors is None:
            self.priors = self.source_config.priors
        
        if self.transform is None:
            self.transform = self.source_config.transform
            
        if self.periodic is None:
            self.periodic = self.source_config.periodic.periodic
        
        if self.betas is None:
            # snrs_ladder = np.array(
            #     [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0,
            #      15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2]
            # )
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            # betas[-1] = 0.0001
            self.betas = betas

        self.logger.info(f"Using betas: {self.betas} in EMRI branch")

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from eryn.moves import StretchMove

            self.inner_moves = [(StretchMove(), 1.0)]

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        if self.branch_state is None:
            self.branch_state = EMRIState

        if self.branch_backend is None:
            self.branch_backend = EMRIHDFBackend


#* ==============================================================================
#* Instrumental Noise (PSD)
#* ==============================================================================

@dataclasses.dataclass
class PSDSettings(Settings):
    prior_file: Optional[str] = None
    psd_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 4
    transform: Optional[TransformContainer] = None
    injection: Optional[np.ndarray] = None 
    nknots: Optional[int] = None
    num_prop_repeats: int = 50


class PSDSetup(Setup, PSDSettings):
    def __init__(self, psd_settings: PSDSettings):
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, psd_settings)

        self.logger = init_logger(
            filename="psd_setup.log", 
            level=logging.DEBUG, 
            name="PSDSetup", 
            log_dir=getattr(self, 'log_dir', None)
        )

        self.source_config = PSDAnalyticalConfig(
            prior_file=self.prior_file, 
            use_cupy=True, # TODO grab dynamically
            return_gpu=False
        )
        
        self.init_setup()

    def init_sampling_info(self):

        if self.psd_kwargs is None:
            self.psd_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            self.priors = self.source_config.priors
        else:
            self.logger.info("Using custom priors for PSD branch")
            
        if self.transform is None:
            self.transform = self.source_config.transform
            self.psd_kwargs["transform_fn"] = self.transform
            
        if self.periodic is None:
            self.periodic = self.source_config.periodic.periodic

        if self.betas is None:
            # TODO: fix this to be generic
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  #

            betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=ntemps_pe)
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        self.init_sampling_info()


#* ==============================================================================
#* Galactic Foreground (GalFor)
#* ==============================================================================

@dataclasses.dataclass
class GalForSettings(Settings):
    prior_file: Optional[str] = None
    galfor_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 5


class GalForSetup(Setup, GalForSettings):
    def __init__(self, galfor_settings: GalForSettings, source_config = None):
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, galfor_settings)

        self.logger = init_logger(
            filename="galfor_setup.log", 
            level=logging.DEBUG, 
            name="GalForSetup", 
            log_dir=getattr(self, 'log_dir', None)
        )
        if source_config is None:
            self.source_config = GalForConfig(
                prior_file=self.prior_file,
                use_cupy=True, # TODO grab dynamically
                return_gpu=False
            )
        else:
            self.source_config = source_config
            
        self.init_setup()

    def init_sampling_info(self):

        if self.galfor_kwargs is None:
            self.galfor_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            self.priors = self.source_config.priors

        if self.transform is None:
            self.transform = self.source_config.transform
            
        if self.periodic is None:
            self.periodic = self.source_config.periodic.periodic
            
        # if self.betas is None:
        #     # TODO: fix this to be generic
        #     ntemps_pe = 24  # len(snrs_ladder)
        #     # betas =  1 / snrs_ladder ** 2  #

        #     betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=ntemps_pe)
        #     self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        self.init_sampling_info()


@dataclasses.dataclass
class HyperSettings(Settings):
    prior_file: Optional[str] = None
    hyper_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    branch_name_map: Optional[typing.Dict] = None
    catalogues: typing.List = dataclasses.field(default_factory=list)
    resolvability_threshold: float = 7.0
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    Nmodels: int = 2
    betas: Optional[np.ndarray] = None
    
    
class HyperSetup(Setup, HyperSettings):
    def __init__(self, hyper_settings: HyperSettings):
        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, hyper_settings)

        self.logger = init_logger(
            filename="hyper_setup.log", 
            level=logging.DEBUG, 
            name="HyperSetup", 
            log_dir=getattr(self, 'log_dir', None)
        )

        self.source_config = HyperConfig(
            prior_file=self.prior_file, 
            use_cupy=True, # TODO grab dynamically
            return_gpu=False
        )
        
        self.init_setup()

    def init_sampling_info(self):

        if self.hyper_kwargs is None:
            self.hyper_kwargs = dict()

        if self.branch_name_map is None:
            self.branch_name_map = dict(
                resolved = "gb",
                stochastic = "galfor"
            )
    
        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            self.priors = self.source_config.priors
        else:
            self.logger.info("Using custom priors for hyper branch")
            
        if self.transform is None:
            self.transform = self.source_config.transform
            
        if self.periodic is None:
            self.periodic = self.source_config.periodic.periodic
        
        if self.betas is None:
            assert self.ntemps is not None, "ntemps must be specified if betas is not provided"
            betas = 1 / 1.2 ** np.arange(self.ntemps)
            betas[-1] = 0.0001
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

    def init_setup(self):
        self.init_sampling_info()



if __name__ == "__main__":
    # gb_setup = GBSetup(GBSettings())
    # gb_set = get_gb_erebor_settings(general_set)
    # mbh_set = get_mbh_erebor_settings(general_set)
    # psd_set = get_psd_erebor_settings(general_set)
    # galfor_set = get_galfor_erebor_settings(general_set)
    breakpoint()

    # # mcmc info for main run
    # gb_main_run_mcmc_info = dict(
    #     branch_names=["gb"],
    #     nleaves_max=15000,
    #     ndim=8,
    #     ntemps=len(betas),
    #     betas=betas,
    #     nwalkers=nwalkers,

    #     pe_waveform_kwargs=pe_gb_waveform_kwargs,
    #     ,

    #     use_prior_removal=False,
    #     rj_refit_fraction=0.2,
    #     rj_search_fraction=0.2,
    #     rj_prior_fraction=0.6,
    #     nsteps=10000,
    #     update_iterations=1,
    #     thin_by=3,
    #     progress=True,
    #     rho_star=rho_star,
    #     stop_kwargs=stopping_kwargs,
    #     stop_search_kwargs=dict(convergence_iter=5, verbose=True),  # really 5 * thin_by
    #     stopping_iterations=1,
    #     in_model_phase_maximize=False,
    #     rj_phase_maximize=False,
    # )

    # # mcmc info for search runs
    # gb_search_run_mcmc_info = dict(
    #     ndim=8,
    #     ntemps=10,
    #     nwalkers=100,
    #     pe_waveform_kwargs=pe_gb_waveform_kwargs,
    #     m_chirp_lims=[0.001, 1.2],
    #     snr_lim=5.0,
    #     # stop_kwargs=dict(newly_added_limit=1, verbose=True),
    #     stopping_iterations=1,
    # )

    # # template generator
    # get_gb_templates = GetGBTemplates(
    #     gb_initialize_kwargs,
    #     gb_waveform_kwargs
    # )

    # all_gb_info = dict(
    #     band_edges=band_edges,
    #     band_N_vals=band_N_vals,
    #     periodic=gb_periodic,
    #     priors=priors_gb_fin,
    #     transform=gb_transform_fn,
    #     waveform_kwargs=gb_waveform_kwargs,
    #     initialize_kwargs=gb_initialize_kwargs,
    #     pe_info=gb_main_run_mcmc_info,
    #     search_info=gb_search_run_mcmc_info,
    #     get_templates=get_gb_templates,
    # )
