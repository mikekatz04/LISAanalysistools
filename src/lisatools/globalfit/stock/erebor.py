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
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State as eryn_State
from eryn.utils import TransformContainer
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.utils.utility import AET, detrend, tukey

from ..engine import Settings, Setup
from ..loginfo import init_logger


# TODO: better way than None?
@dataclasses.dataclass
class GBSettings(Settings):
    A_lims: typing.List[float, float] = None
    f0_lims: typing.List[float, float] = None
    m_chirp_lims: typing.List[float, float] = None
    fdot_lims: typing.List[float, float] = None
    phi0_lims: typing.List[float, float] = None
    iota_lims: typing.List[float, float] = None
    psi_lims: typing.List[float, float] = None
    lam_lims: typing.List[float, float] = None
    beta_lims: typing.List[float, float] = None
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    oversample: int = 4
    extra_buffer: int = 5
    start_resample_iter: Optional[int] = (
        -1,
    )  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    group_proposal_kwargs: Optional[dict] = None
    start_freq_ind: Optional[int] = 0  # goes into GPU for start of data stream


# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3


from ..hdfbackend import GBHDFBackend
from ..state import GBState


class GBSetup(Setup, GBSettings):
    def __init__(self, gb_settings: GBSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name)

        self.init_setup()

    def init_sampling_info(self):

        input_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

        if self.transform is None:
            gb_transform_fn_in = {
                "A": np.exp,
                "f0": f_ms_to_s,
                "cos_iota": np.arccos,
                "sin_beta": np.arcsin,
            }

            output_basis = [
                "A",
                "f0",
                "fdot",
                "fddot",
                "phi0",
                "cos_iota",
                "psi",
                "lam",
                "sin_beta",
            ]
            gb_fill_dict = {"fddot": 0.0}

            # gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

            self.transform = TransformContainer(
                input_basis=input_basis,
                output_basis=output_basis,
                parameter_transforms=gb_transform_fn_in,
                fill_dict=gb_fill_dict,
            )

        if self.periodic is None:
            self.periodic = {"gb": {"phi0": 2 * np.pi, "psi": np.pi, "lam": 2 * np.pi}}

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_gb = {
                input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                input_basis[1]: uniform_dist(
                    *(np.asarray(self.f0_lims) * 1e3)
                ),  # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                input_basis[2]: uniform_dist(*self.fdot_lims),
                input_basis[3]: uniform_dist(*self.phi0_lims),
                input_basis[4]: uniform_dist(*np.cos(self.iota_lims)),
                input_basis[5]: uniform_dist(*self.psi_lims),
                input_basis[6]: uniform_dist(*self.lam_lims),
                input_basis[7]: uniform_dist(*np.sin(self.beta_lims)),
            }

            # TODO: orbits check against sangria/sangria_hm

            # priors_gb_fin = GBPriorWrap(8, ProbDistContainer(priors_gb))
            self.priors = {"gb": ProbDistContainer(priors_gb)}

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

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(adaptation_time=2, permute=True)

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        self.waveform_kwargs = dict(
            dt=self.dt,
            T=self.Tobs,
            use_c_implementation=True,
            oversample=self.oversample,
            start_freq_ind=self.start_freq_ind,
        )

        if self.group_proposal_kwargs is None:
            self.group_proposal_kwargs = dict(
                n_iter_update=1, live_dangerously=True, a=1.75, num_repeat_proposals=200
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

        if self.oversample is None and Tobs < YEAR / 2.0:
            self.oversample = 2
        elif self.oversample is None:
            self.oversample = 4

        assert self.oversample >= 1

        # TODO: assign to binned f or leave general? probably better to be general
        band_edges_in_reverse_order = [self.end_freq]
        band_N_vals_reverse_order = []
        # determines N from high_Frequency edge of sub-band
        current_N = get_N(
            1e-30, self.end_freq, self.Tobs, oversample=self.oversample
        ).item()
        band_N_vals_reverse_order.append(current_N)

        current_freq = self.end_freq
        last_freq = self.end_freq
        while current_freq > self.start_freq:
            current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
            band_edges_in_reverse_order.append(current_freq)
            current_N = get_N(
                1e-30, current_freq, self.Tobs, oversample=self.oversample
            ).item()
            band_N_vals_reverse_order.append(current_N)
            last_freq = current_freq
        band_edges_in_reverse_order.append(
            last_freq - (current_N * 2 + self.extra_buffer) * self.df
        )

        self.band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
        self.band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]

        self.logger.debug("NEED TO THINK ABOUT mCHIRP prior")
        self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]
        fdot_max_val = get_fdot(self.f0_lims[-1], Mc=self.m_chirp_lims[-1])

        self.fdot_lims = [-fdot_max_val, fdot_max_val]

        self.num_sub_bands = len(self.band_edges)


def mbh_dist_trans(x):
    return x * PC_SI * 1e9  # Gpc

def gpc_to_mpc(x):
    """
    Transform from Gpc to Mpc, for distance prior.
    """
    return x * 1e3


from bbhx.utils.transform import *
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


class MBHSetup(Setup):
    def __init__(self, mbh_settings: MBHSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(mbh_settings)

        level = logging.DEBUG
        name = "MBHSetup"
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name)

        self.init_setup()

    def init_sampling_info(self):

        # input_basis = [
        #     "logM",
        #     "q",
        #     "s1z",
        #     "s2z",
        #     "dist",
        #     "phi_ref",
        #     "cos_iota",
        #     "lam",
        #     "sin_beta",
        #     "psi",
        #     "t_ref",
        # ]

        input_basis = [
            "logM",
            "q",
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
                "q",
                "s1z",
                "s2z",
                "dist",
                "phi_ref",
                #"f_ref",
                "cos_iota",
                "psi",
                "lam",
                "sin_beta",
                #"psi",
                "t_plunge",
            ]

            mbh_transform_fn_in = {
                "logM": np.exp,
                "dist": gpc_to_mpc,
                "cos_iota": np.arccos,
                "sin_beta": np.arcsin,
                ("logM", "q"): mT_q,
                ("t_plunge", "lam", "sin_beta", "psi"): LISA_to_SSB,
            }

            # for transforms
            #mbh_fill_dict = {"f_ref": 0.0}
            mbh_fill_dict = {}

            self.transform = TransformContainer(
                input_basis=input_basis,
                output_basis=output_basis,
                parameter_transforms=mbh_transform_fn_in,
                fill_dict=mbh_fill_dict,
            )

        if self.periodic is None:
            self.periodic = {
                "mbh": {"phi_ref": 2 * np.pi, "lam": 2 * np.pi, "psi": np.pi}
            }

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            priors_mbh = {
                "logM": uniform_dist(np.log(1e4), np.log(1e8)),
                "q": uniform_dist(0.01, 0.999999999),
                "s1z": uniform_dist(-0.99999999, +0.99999999),
                "s2z": uniform_dist(-0.99999999, +0.99999999),
                "dist": uniform_dist(0.01, 1000.0),
                "phi_ref": uniform_dist(0.0, 2 * np.pi),
                "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "psi": uniform_dist(0.0, 2 * np.pi),
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


from ..hdfbackend import EMRIHDFBackend
from ..state import EMRIState


@dataclasses.dataclass
class EMRISettings(Settings):
    logm1_lims: typing.List[float, float] = None
    m2_lims: typing.List[float, float] = None
    a_lims: typing.List[float, float] = None
    p0_lims: typing.List[float, float] = None
    e0_lims: typing.List[float, float] = None
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    fill_values: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 0.0])
    )
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    emri_search_file_key: Optional[str] = "_emri_search_tmp_file"


class EMRISetup(Setup):
    def __init__(self, emri_settings: EMRISettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(emri_settings)

        level = logging.DEBUG
        name = "EMRISetup"
        self.logger = init_logger(filename="emri_setup.log", level=level, name=name)

        self.init_setup()

    def init_sampling_info(self):

        input_basis = [
            "logm1",
            "m2",
            "a",
            "p0",
            "e0",
            "dist",
            "qS",
            "phiS",
            "qK",
            "phiK",
            "Phi_phi0",
            "Phi_r0",
        ]

        if self.transform is None:

            output_basis = [
                "logm1",
                "m2",
                "a",
                "p0",
                "e0",
                "xI0",
                "dist",
                "qS",
                "phiS",
                "qK",
                "phiK",
                "Phi_phi0",
                "Phi_theta0",
                "Phi_r0",
            ]

            # for transforms

            emri_fill_dict = {
                "xI0": self.fill_values[0],  # inclination
                "Phi_theta0": self.fill_values[1],  # Phi_theta
            }

            emri_transform_fn_in = {
                output_basis[0]: np.exp,  # M
                output_basis[7]: np.arccos,  # qS
                output_basis[9]: np.arccos,  # qK
            }

            self.transform = TransformContainer(
                input_basis=input_basis,
                output_basis=output_basis,
                parameter_transforms=emri_transform_fn_in,
                fill_dict=emri_fill_dict,
            )

        if self.periodic is None:
            self.periodic = {
                "emri": {
                    "phiS": 2 * np.pi,
                    "phiK": 2 * np.pi,
                    "Phi_phi0": 2 * np.pi,
                    "Phi_r0": 2 * np.pi,
                }
            }

        self.setup_priors(input_basis)

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

    def setup_priors(self, input_basis):
        """
        Get the prior distributions for the EMRI parameters.
        override the default priors with custom boundaries for the intrinsic parameters.

        Args:

        Returns:
            ProbDistContainer: Container with prior distributions for each parameter.
        """

        priors_emri = {
            input_basis[0]: uniform_dist(np.log(5e5), np.log(5e6)),  # log m1
            input_basis[1]: uniform_dist(1, 100),  # m2
            input_basis[2]: uniform_dist(0.01, 0.999),  # a
            input_basis[3]: uniform_dist(5.0, 100.0),  # p0
            input_basis[4]: uniform_dist(0.001, 0.8),  # e0
            input_basis[5]: uniform_dist(0.01, 100.0),  # dist in Gpc
            input_basis[6]: uniform_dist(-0.99999, 0.99999),  # qS
            input_basis[7]: uniform_dist(0.0, 2 * np.pi),  # phiS
            input_basis[8]: uniform_dist(-0.99999, 0.99999),  # qK
            input_basis[9]: uniform_dist(0.0, 2 * np.pi),  # phiK
            input_basis[10]: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
            input_basis[11]: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
        }

        limits = ["logm1_lims", "m2_lims", "a_lims", "p0_lims", "e0_lims"]
        for i, lims in enumerate(limits):
            if getattr(self, lims) is not None:
                self.logger.info(
                    f"Setting prior for parameter {i} using limits {getattr(self, lims)}"
                )
                priors_emri[input_basis[i]] = uniform_dist(*getattr(self, lims))

        self.priors = {"emri": ProbDistContainer(priors_emri)}

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        if self.branch_state is None:
            self.branch_state = EMRIState

        if self.branch_backend is None:
            self.branch_backend = EMRIHDFBackend


from lisatools.detector import EqualArmlengthOrbits


@dataclasses.dataclass
class PSDSettings(Settings):
    psd_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 4
    transform_fn: TransformContainer = None


class PSDSetup(Setup):
    def __init__(self, psd_settings: PSDSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(psd_settings)

        level = logging.DEBUG
        name = "PSDSetup"
        self.logger = init_logger(filename="psd_setup.log", level=level, name=name)

        self.init_setup()

    def init_sampling_info(self):

        if self.psd_kwargs is None:
            self.psd_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_psd = {
                r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
                r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
                # 2: uniform_dist(6.0e-12, 20.0e-12),  # Soms_d
                # 3: uniform_dist(1.0e-15, 20.0e-15),  # Sa_a
            }

            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"psd": ProbDistContainer(priors_psd)}

        else:
            self.logger.info("Using custom priors for PSD branch")

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

        self.transform_fn = self.psd_kwargs.get("transform_fn", None)

    def init_setup(self):
        self.init_sampling_info()


@dataclasses.dataclass
class GalForSettings(Settings):
    galfor_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 5


class GalForSetup(Setup):
    def __init__(self, galfor_settings: GalForSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(galfor_settings)

        level = logging.DEBUG
        name = "GalForSetup"
        self.logger = init_logger(filename="galfor_setup.log", level=level, name=name)

        self.init_setup()

    def init_sampling_info(self):

        if self.galfor_kwargs is None:
            self.galfor_kwargs = dict(sens_fn="A1TDISens")

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_galfor = {
                0: uniform_dist(1e-45, 2e-43),  # amp
                1: uniform_dist(1e-4, 5e-2),  # knee
                2: uniform_dist(0.01, 3.0),  # alpha
                3: uniform_dist(1e0, 1e7),  # Slope1
                4: uniform_dist(5e1, 8e3),  # Slope2
            }

            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"galfor": ProbDistContainer(priors_galfor)}

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


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:

    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.utils.constants import YRSID_SI

    Tobs = YRSID_SI
    dt = 10.0

    galfor_settings = GalForSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
    )

    return GalForSetup(galfor_settings)


if __name__ == "__main__":
    general_set = get_general_erebor_settings()
    gb_set = get_gb_erebor_settings(general_set)
    mbh_set = get_mbh_erebor_settings(general_set)
    psd_set = get_psd_erebor_settings(general_set)
    galfor_set = get_galfor_erebor_settings(general_set)
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
