from __future__ import annotations
import dataclasses
import typing
from typing import Optional
import numpy as np

import logging

from gbgpu.utils.utility import get_fdot
from gbgpu.utils.utility import get_N
from lisatools.globalfit.run import init_logger

from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer


class Settings:
    pass

@dataclasses.dataclass
class GBSettings(Settings):
    A_lims: typing.List[float, float]
    f0_lims: typing.List[float, float]
    m_chirp_lims: typing.List[float, float]
    fdot_lims: typing.List[float, float]
    phi0_lims: typing.List[float, float]
    iota_lims: typing.List[float, float]
    psi_lims: typing.List[float, float]
    lam_lims: typing.List[float, float]
    beta_lims: typing.List[float, float]
    Tobs: float
    dt: float
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    oversample: int = 4
    extra_buffer: int = 5
    gb_initialize_kwargs: dict = None
    transform: Optional[TransformContainer] = None
    priors: Optional[ProbDistContainer] = None
    periodic: Optional[dict] = None
    betas: Optional[np.ndarray] = None
    start_resample_iter: Optional[int] = -1,  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    other_tempering_kwargs: Optional[dict] = None
    group_proposal_kwargs: Optional[dict] = None
    nleaves_max: Optional[int] = 15000
    ndim: Optional[int] = 8


# basic transform functions for pickling
def f_ms_to_s(x):
    return x * 1e-3




class GBSetup(GBSettings):


    def __init__(self, gb_settings: GBSettings):
        
        # had a better way to do this but it stopped allowing for pickle
        self._settings_names = [field.name for field in dataclasses.fields(GBSettings)]
        self._settings_args_names = [field.name for field in dataclasses.fields(GBSettings) if field.default == dataclasses.MISSING]  # args
        self._settings_kwargs_names = [field.name for field in dataclasses.fields(GBSettings) if field.default != dataclasses.MISSING]  # args
        _args = tuple([getattr(gb_settings, key) for key in self._settings_args_names])
        _kwargs = {key: getattr(gb_settings, key) for key in self._settings_kwargs_names}
        super().__init__(*_args, **_kwargs)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name)
        
        self.init_setup()

    def init_sampling_info(self):

        if self.transform is None:
            gb_transform_fn_in = {
                0: np.exp,
                1: f_ms_to_s,
                5: np.arccos,
                8: np.arcsin,
            }

            gb_fill_dict = {"fill_inds": np.array([3]), "ndim_full": 9, "fill_values": np.array([0.0])}

            self.transform = TransformContainer(
                parameter_transforms=gb_transform_fn_in, fill_dict=gb_fill_dict
            )

        if self.periodic is None:
            self.periodic = {"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}}

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            # TODO: change to scaled linear in amplitude!?!
            priors_gb = {
                0: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                1: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)), # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                2: uniform_dist(*self.fdot_lims),
                3: uniform_dist(*self.phi0_lims),
                4: uniform_dist(*np.cos(self.iota_lims)),
                5: uniform_dist(*self.psi_lims),
                6: uniform_dist(*self.lam_lims),
                7: uniform_dist(*np.sin(self.beta_lims)),
            }

            # TODO: orbits check against sangria/sangria_hm

            # priors_gb_fin = GBPriorWrap(8, ProbDistContainer(priors_gb))
            self.priors = {"gb": ProbDistContainer(priors_gb)}

        if self.betas is None:
            snrs_ladder = np.array([1., 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 35.0, 50.0, 75.0, 125.0, 250.0, 5e2])
            ntemps_pe = 24  # len(snrs_ladder)
            # betas =  1 / snrs_ladder ** 2  # make_ladder(ndim * 10, Tmax=5e6, ntemps=ntemps_pe)
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            betas[-1] = 0.0001
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(
                adaptation_time=2,
                permute=True
            )

        if self.gb_initialize_kwargs is None:
            self.gb_initialize_kwargs = {}

        self.waveform_kwargs = dict(
            dt=self.dt, T=self.Tobs, use_c_implementation=True, oversample=self.oversample
        )

        if self.group_proposal_kwargs is None:
            self.group_proposal_kwargs = dict(
                n_iter_update=1,
                live_dangerously=True,
                a=1.75,
                num_repeat_proposals=200
            )
            
    @property
    def gb_settings(self) -> GBSettings:
        return self._gb_settings

    @gb_settings.setter
    def gb_settings(self, gb_settings: GBSettings):
        assert isinstance(gb_settings, GBSettings)
        self._gb_settings = gb_settings

    # def __getattr__(self, attr: str) -> typing.Any:
    #     if hasattr(self.gb_settings, attr):
    #         return getattr(self.gb_settings, attr)

    def init_setup(self):
        self.init_df()
        self.init_band_structure()
        self.init_sampling_info()

    def init_df(self):
        self.Tobs = int(self.Tobs / self.dt) * self.dt
        self.df = 1. / self.Tobs

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
        current_N = get_N(1e-30, self.end_freq, self.Tobs, oversample=self.oversample).item()
        band_N_vals_reverse_order.append(current_N)

        current_freq = self.end_freq
        last_freq = self.end_freq
        while current_freq > self.start_freq:
            current_freq = last_freq - (current_N * 2 + self.extra_buffer) * self.df
            band_edges_in_reverse_order.append(current_freq)
            current_N = get_N(1e-30, current_freq, self.Tobs, oversample=self.oversample).item()
            band_N_vals_reverse_order.append(current_N)
            last_freq = current_freq
        band_edges_in_reverse_order.append(last_freq - (current_N * 2 + self.extra_buffer) * self.df)
        
        self.band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
        self.band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]
        
        self.logger.debug("NEED TO THINK ABOUT mCHIRP prior")
        self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]
        fdot_max_val = get_fdot(self.f0_lims[-1], Mc=self.m_chirp_lims[-1])

        self.fdot_lims = [-fdot_max_val, fdot_max_val]
        
        self.num_sub_bands = len(self.band_edges)


def get_gb_erebor_settings() -> GBSetup:
       # limits on parameters
    delta_safe = 1e-5
    # now with negative fdots
    
    from lisatools.utils.constants import YRSID_SI
    Tobs = YRSID_SI
    dt = 10.0
    A_lims = [7e-26, 1e-19]
    f0_lims = [0.05e-3, 2.5e-2]  # TODO: this upper limit leads to an issue at 23 mHz where there is no source?
    
    m_chirp_lims = [0.001, 1.0]
    fdot_max_val = get_fdot(f0_lims[-1], Mc=m_chirp_lims[-1])
    
    fdot_lims = [-fdot_max_val, fdot_max_val]
    phi0_lims = [0.0, 2 * np.pi]
    iota_lims = [0.0 + delta_safe, np.pi - delta_safe]
    psi_lims = [0.0, np.pi]
    lam_lims = [0.0, 2 * np.pi]
    beta_lims = [-np.pi / 2.0 + delta_safe, np.pi / 2.0 - delta_safe]

    end_freq = 0.025
    start_freq = 0.0001
    oversample = 4
    extra_buffer = 5
    gb_initialize_kwargs = dict(force_backend="cuda12x")

    gb_settings = GBSettings(
        A_lims,
        f0_lims,
        m_chirp_lims,
        fdot_lims,
        phi0_lims,
        iota_lims,
        psi_lims,
        lam_lims,
        beta_lims,
        Tobs,
        dt,
        gb_initialize_kwargs=gb_initialize_kwargs,
    )

    gb_setup = GBSetup(gb_settings)
    return gb_setup


if __name__ == "__main__":
    check = get_gb_erebor_settings()

 
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


# class Erebor:

#     def mbh_settings
