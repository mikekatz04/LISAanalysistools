"""Stock Erebor recipe: per-source :class:`Setup` / :class:`Settings` definitions."""

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
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform
from eryn.state import State as ErynState
from eryn.state import Branch as ErynBranch
from eryn.utils import TransformContainer
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.sources.utils import ecliptic_to_icrs
from lisatools.utils.utility import AET, detrend, tukey
from lisatools.utils.constants import YRSID_SI, PC_SI

from ...domains import DomainSettingsBase, FDSettings, WDMSettings
from ..engine import Settings, Setup, GeneralSetup
from ..loginfo import init_logger
from ..priors.gbpriors import get_fdot_mojito


@dataclasses.dataclass
class GBSettings(Settings):
    """Settings dataclass describing the GB branch in an Erebor-style recipe.

    Mostly holds parameter limits, frequency-band oversampling, and the
    starting frequency / TDI configuration consumed by :class:`GBSetup`.
    """

    A_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    m_chirp_lims: typing.List[float] = dataclasses.field(default_factory=list)
    fdot_lims: typing.List[float] = dataclasses.field(default_factory=list)
    phi0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    iota_lims: typing.List[float] = dataclasses.field(default_factory=list)
    psi_lims: typing.List[float] = dataclasses.field(default_factory=list)
    alpha_lims: typing.List[float] = dataclasses.field(default_factory=list)
    delta_lims: typing.List[float] = dataclasses.field(default_factory=list)
    start_freq: float = 0.0001  # this might get adjusted ?
    end_freq: float = 0.025
    oversample: int = 4. # FD
    extra_buffer: int = 5
    start_resample_iter: Optional[typing.Tuple[int]] = (-1,)  # -1 so that it starts right at the start of PE
    iter_count_per_resample: Optional[int] = 10
    num_repeat_proposals: int = 100
    search_kwargs: Optional[dict] = None
    group_proposal_kwargs: Optional[dict] = None
    start_freq_ind: Optional[int] = 0  # goes into GPU for start of data stream
    t0: Optional[float] = 0.0
    tdi_setup: Optional[str] = "XYZ" # other options are AET and AE.
    use_tdi2: Optional[bool] = True
    waveform_kwargs: dict = dataclasses.field(default_factory=dict)
    # Domain selection for this branch. The user passes the same
    # ``DomainSettingsBase`` that lives on the parent ``GeneralSetup``
    # (FDSettings, STFTSettings, WDMSettings, ...). The band structure /
    # waveform-engine selection branches on ``isinstance(domain_settings, ...)``
    # — no string mode flag.
    domain_settings: Optional[DomainSettingsBase] = None
    # Optional WDM-domain likelihood object (a
    # ``gbgpu.gbcomps.GBWDMComputations`` instance). Required when
    # ``domain_settings`` is a :class:`WDMSettings`; ignored otherwise. The
    # user builds this once their WDM grid + lookup table are known
    # (see global-fit input scripts).
    gb_wdm_comp: typing.Any = None

# basic transform functions for pickling
def f_ms_to_s(x):
    """Convert frequencies from mHz to Hz (named for pickling support)."""
    return x * 1e-3


def f_s_to_ms(x):
    """Convert frequencies from Hz to mHz (inverse of :func:`f_ms_to_s`)."""
    return x * 1e3


def make_gb_transform_container():
    """Build the stock GB :class:`TransformContainer` (forward + inverse).

    Sampling basis: ``A (ln), f0 (mHz), fdot, phi0, cos_iota, psi, lam,
    sin_beta``; output basis adds the filled ``fddot``. The inverse
    transforms enable ``both_inverse_transforms`` (full basis -> sampling
    basis).
    """
    input_basis = ["A", "f0", "fdot", "phi0", "cos_iota", "psi", "lam", "sin_beta"]

    gb_transform_fn_in = {
        "A": np.exp,
        "f0": f_ms_to_s,
        "cos_iota": np.arccos,
        "sin_beta": np.arcsin,
    }

    gb_inverse_transform_fn_in = {
        "A": np.log,
        "f0": f_s_to_ms,
        "cos_iota": np.cos,
        "sin_beta": np.sin,
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

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=gb_transform_fn_in,
        fill_dict=gb_fill_dict,
        inverse_parameter_transforms=gb_inverse_transform_fn_in,
    )


from ..hdfbackend import GBHDFBackend
from ..state import GBState


class GBSetup(Setup, GBSettings):
    """:class:`Setup` for galactic binaries in the Erebor recipe.

    Args:
        gb_settings: Settings dataclass holding GB parameter ranges and
            domain configuration.
    """

    def __init__(self, gb_settings: GBSettings):

        # had a better way to do this but it stopped allowing for pickle
        Setup.__init__(self, gb_settings)

        level = logging.DEBUG
        name = "GBSetup"
        self.logger = init_logger(filename="gb_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the GB :class:`TransformContainer`, prior, periodicity, and waveform kwargs.

        stft_tof basis (kept at the 2026-06 merge): named LaTeX parameters
        with ICRS sky angles (alpha / sin(delta)) -- the run frame is ICRS --
        and the named transforms that the GB info-matrix proposal looks up
        (``parameter_transforms.original_parameter_transforms[r"$\\log A$"]``
        etc. in :mod:`..moves.gbspecialstretch`).
        """
        input_basis = [r"$\log A$", r"$f_0$", r"$\dot{f}$", r"$\phi_0$",
                       r"$\cos\iota$", r"$\psi$", r"$\alpha$", r"$\sin\delta$"]

        if self.transform is None:
            gb_transform_fn_in = {
                r"$\log A$": np.exp,
                r"$f_0$": f_ms_to_s,
                r"$\phi_0$": lambda x: -1 * x,  # flip sign of phi0 to match JaxGB convention.
                r"$\cos\iota$": np.arccos,
                r"$\sin\delta$": np.arcsin,

            }

            output_basis = [
                r"$\log A$", r"$f_0$", r"$\dot{f}$",
                r"$\ddot{f}$", r"$\phi_0$", r"$\cos\iota$",
                r"$\psi$", r"$\alpha$", r"$\sin\delta$"
            ]

            gb_fill_dict = {r"$\ddot{f}$": 0.0}

            self.transform = TransformContainer(
                input_basis=input_basis,
                output_basis=output_basis,
                parameter_transforms=gb_transform_fn_in,
                fill_dict=gb_fill_dict,
            )

        if self.periodic is None:
            self.periodic = {"gb": {r"$\phi_0$": 2*np.pi, r"$\psi$": np.pi, r"$\alpha$": 2 * np.pi}}

        if self.priors is None:
            priors_gb = {
                input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),  # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                input_basis[2]: uniform_dist(self.fdot_lims[0], self.fdot_lims[1]),
                input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                input_basis[4]: uniform_dist(*np.cos(self.iota_lims)),
                input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                input_basis[7]: uniform_dist(*np.sin(self.delta_lims)),
            }

            self.priors = {"gb": ProbDistContainer(priors_gb)}

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

        # GBGPU FD waveform kwargs. The WDM-domain engine in
        # ``gb_likelihood.WDMBandLikelihoodEngine`` ignores
        # ``start_freq_ind`` / ``oversample`` (it dispatches to the WDM C
        # kernel via ``GBWDMComputations``), but we keep the same dict so
        # the down-stream Buffer's ``tdi_channel_setup`` read still works
        # on either path. The settings file may pre-populate
        # ``waveform_kwargs`` (stft_tof field); this default is built from
        # the other fields only when it is left empty.
        if not self.waveform_kwargs:
            self.waveform_kwargs = dict(
                dt=self.dt,
                T=self.Tobs,
                use_c_implementation=True,
                oversample=self.oversample,
                start_freq_ind=self.start_freq_ind,
                tdi_channel_setup=self.tdi_setup,
                tdi2=self.use_tdi2
            )

        if self.group_proposal_kwargs is None:
            self.group_proposal_kwargs: typing.Dict[str, Any] = dict(
                n_iter_update=1, live_dangerously=True, a=1.75, num_repeat_proposals=200
            )
        
        if self.search_kwargs is None:
            # Configuration for the stft_tof per-band serial GB search
            # (consumed by ``GBSpecialRJSerialSearchMCMC.setup`` and
            # ``GBSpecialRJRefitMove.setup``):
            #   nwalkers / ntemps          — per-band ParaEnsembleSampler size
            #   burn_1 / nsteps_1          — stage 1: F-stat (phase-maximized)
            #                                MCMC, Gibbs on the 4 intrinsic-
            #                                like params
            #   snr_threshold              — band "found a source" gate on the
            #                                stage-1 chain's min optimal SNR
            #   burn_2 / nsteps_2          — stage 2: full-likelihood refine
            #                                from stage-1's last sample; its
            #                                samples feed the GMM fit that
            #                                becomes the RJ proposal
            #   shutoff_band_iteration     — shut a band off after this many
            #                                consecutive source-less search
            #                                iterations (all-off reverts the
            #                                RJ proposal to the global prior)
            #   shutoff_frequency_threshold— only bands above this frequency
            #                                are shutoff-eligible (None = all)
            #   refit_start_iteration      — samples kept per leaf for the
            #                                GMM refit in GBSpecialRJRefitMove
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
        """Run the band-structure, sampling-info, and state-backend init helpers."""
        self.init_band_structure()
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default :attr:`branch_state` to :class:`GBState` and :attr:`branch_backend` to :class:`GBHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = GBState

        if self.branch_backend is None:
            self.branch_backend = GBHDFBackend

    def init_band_structure(self):
        """Compute :attr:`band_edges` and :attr:`band_N_vals` from the GB frequency range.

        Both FD and WDM domains use a frequency-banded layout. The edges
        are derived from ``df = 1/Tobs`` (always meaningful for either
        basis) and the FD-oversampled per-band N. For WDM runs the
        per-band ``band_N_vals`` is unused by the WDM likelihood engine
        (which sizes its buffers from ``WDMSettings.Nf_active`` /
        ``Nt_active``) but is preserved for shape parity.
        """
        # band separation setup
        if self.oversample is None and self.Tobs < YRSID_SI / 2.0:
            self.oversample = 2
        elif self.oversample is None:
            self.oversample = 4

        assert self.oversample >= 1

        # Clamp the GB band to the WDM active band when running on a WDM
        # grid. The WDM kernel only carries layers in [ind_min_f, ind_max_f];
        # putting band edges outside that range silently produces zero-fill
        # behaviour, so we trim to stay inside.
        if isinstance(self.domain_settings, WDMSettings):
            wdm = self.domain_settings
            wdm_min = float(wdm.ind_min_f * wdm.layer_df)
            wdm_max = float(wdm.ind_max_f * wdm.layer_df)
            self.start_freq = max(self.start_freq, wdm_min)
            self.end_freq = min(self.end_freq, wdm_max)
            if self.start_freq >= self.end_freq:
                raise ValueError(
                    "WDM active band [{:.4e}, {:.4e}] does not overlap GB "
                    "frequency range; widen WDMSettings.min_freq/max_freq "
                    "or the GB start/end_freq.".format(wdm_min, wdm_max)
                )

        if isinstance(self.domain_settings, WDMSettings):
            # WDM path: one band per WDM frequency layer. Edges land on
            # ``k * layer_df`` boundaries so the band grid aligns with the
            # WDM grid; ``band_N_vals`` is unused by the WDM likelihood
            # engine but is preserved with one entry per band for shape
            # parity with the FD layout.
            wdm = self.domain_settings
            layer_df = float(wdm.layer_df)
            k_lo = int(np.ceil(self.start_freq / layer_df))
            k_hi = int(np.floor(self.end_freq / layer_df))
            if k_hi <= k_lo:
                raise ValueError(
                    "GB frequency range [{:.4e}, {:.4e}] spans fewer than "
                    "one WDM layer (layer_df={:.4e}).".format(
                        self.start_freq, self.end_freq, layer_df
                    )
                )
            self.band_edges = np.asarray(
                [k * layer_df for k in range(k_lo, k_hi + 1)]
            )
            self.band_N_vals = np.asarray(
                [
                    get_N(1e-30, edge, self.Tobs, oversample=self.oversample).item()
                    for edge in self.band_edges[:-1]
                ]
            )
        else:
            # FD path: bands sized in multiples of ``df = 1/Tobs`` using the
            # FD-oversampled per-band N. Walks down from ``end_freq``
            # (stft_tof refinements: half-bin start, min_N stop guard, and
            # edge trim against out-of-bound indexing).
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

            band_edges = np.asarray(band_edges_in_reverse_order)[::-1]
            band_N_vals = np.asarray(band_N_vals_reverse_order)[::-1]

            # trim edges to avoid out of bound indexing
            self.band_edges = band_edges[2:-1]
            self.band_N_vals = band_N_vals[2:-1]

        self.f0_lims = [self.band_edges[1].min(), self.band_edges[-2].max()]

        self.fdot_lims = [
            get_fdot_mojito(self.f0_lims[1], sign="-"), 
            get_fdot_mojito(self.f0_lims[1], sign="+"), 
        ]

        self.num_sub_bands = len(self.band_edges) - 1
        
        self.logger.info(
            f"GB f0 prior range is set from {round(self.f0_lims[0],7)} to {round(self.f0_lims[1],7)}"
        )
        self.logger.info(f"The number of subbands is {self.num_sub_bands}")
        self.logger.info(f"Min freq of subbands is {self.band_edges.min()}")
        self.logger.info(f"Max freq of subbands is {self.band_edges.max()}")

        
def mbh_dist_trans(x):
    """Transform an MBH ``ln total mass`` parameter (named for pickling support)."""
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

def mpc_to_gpc(x):
    """
    Transform from Mpc to Gpc (inverse of :func:`gpc_to_mpc`).
    """
    return x * 1e-3


def m1_m2_to_mT_q(m1, m2):
    """Convert m1, m2 to total mass and mass ratio (inverse of ``mT_q``)."""
    return (m1 + m2, m2 / m1)


def m1_m2_to_mT_Q(m1, m2):
    """Convert m1, m2 to total mass and ``Q = m1/m2 >= 1`` (inverse of :func:`mT_Q`)."""
    return (m1 + m2, m1 / m2)


from bbhx.utils.transform import LISA_to_SSB, SSB_to_LISA, mT_q  # frame helpers kept for settings files; the stock MBH basis is direct ICRS
from eryn.moves import Move


def make_mbh_transform_container():
    """Build the stock MBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (ICRS — the 2026-06 run frame, matching
    :func:`lisatools.globalfit.recipe_steps.mbh_catalogue_to_sampling_basis`):
    ``logM, Q (= m1/m2 >= 1), s1z, s2z, dist (Gpc), phi_ref, cos_iota,
    psi (ICRS), alpha (RA), sin_delta, t_plunge (SSB)``. The output basis
    holds ``m1, m2 (via mT_Q), dist (Mpc), iota, delta`` with sky /
    polarization passed through unchanged in ICRS — no frame
    conversion; the orbits must be loaded with ``frame='icrs'`` so the
    response consumes ``(alpha, delta)`` directly. The inverse
    transforms enable ``both_inverse_transforms`` (full basis ->
    sampling basis).
    """
    input_basis = [
        "logM",
        "Q",
        "s1z",
        "s2z",
        "dist",
        "phi_ref",
        "cos_iota",
        "psi",
        "alpha",
        "sin_delta",
        "t_plunge",
    ]

    output_basis = list(input_basis)

    mbh_transform_fn_in = {
        "logM": np.exp,
        "dist": gpc_to_mpc,
        "cos_iota": np.arccos,
        "sin_delta": np.arcsin,
        ("logM", "Q"): mT_Q,
    }

    # keyed identically to mbh_transform_fn_in; same insertion order so
    # the multi-parameter inverses unwind in reverse of the forward order
    mbh_inverse_transform_fn_in = {
        "logM": np.log,
        "dist": mpc_to_gpc,
        "cos_iota": np.cos,
        "sin_delta": np.sin,
        ("logM", "Q"): m1_m2_to_mT_Q,
    }

    # for transforms
    # mbh_fill_dict = {"f_ref": 0.0}
    mbh_fill_dict = {}

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=mbh_transform_fn_in,
        fill_dict=mbh_fill_dict,
        inverse_parameter_transforms=mbh_inverse_transform_fn_in,
    )

from ..hdfbackend import MBHHDFBackend
from ..state import MBHState




@dataclasses.dataclass
class MBHSettings(Settings):
    """Settings dataclass describing the MBH branch in an Erebor-style recipe."""

    waveform_kwargs: Optional[dict] = None
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 200
    mbh_search_file_key: Optional[str] = "_mbh_search_tmp_file"
    injection: Optional[np.ndarray] = None


class MBHSetup(Setup):
    """:class:`Setup` for massive black-hole binaries in the Erebor recipe.

    Args:
        mbh_settings: Settings dataclass with MBH waveform / prior config.
    """

    def __init__(self, mbh_settings: MBHSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(mbh_settings)

        level = logging.DEBUG
        name = "MBHSetup"
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the MBH :class:`TransformContainer`, prior, periodicity, and waveform kwargs."""

        # ICRS sampling basis (2026-06 run-frame directive): sky and
        # polarization are sampled directly in ICRS (alpha = RA,
        # sin_delta, psi = ICRS polarization), matching
        # ``mbh_catalogue_to_sampling_basis``. No LISA->SSB->ICRS chain;
        # the orbits must be loaded with ``frame='icrs'`` so the response
        # consumes ``(alpha, delta)`` directly.
        input_basis = [
            "logM",
            "Q",
            "s1z",
            "s2z",
            "dist",
            "phi_ref",
            "cos_iota",
            "psi",
            "alpha",
            "sin_delta",
            "t_plunge",
        ]

        if self.transform is None:
            # Stock forward + inverse container (direct ICRS; same basis as
            # ``input_basis`` above). The inverse transforms enable
            # ``both_inverse_transforms`` (full basis -> sampling basis)
            # for injection/diagnostic round trips.
            self.transform = make_mbh_transform_container()

        if self.periodic is None:
            self.periodic = {"mbh": {"phi_ref": 2 * np.pi, "alpha": 2 * np.pi, "psi": np.pi}}

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
                "alpha": uniform_dist(0.0, 2 * np.pi),
                "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
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

            # TODO(post-merge): re-enable SkyMove hops once the move
            # supports the ICRS sampling basis — the stock MBH basis is
            # now direct ICRS (alpha, sin_delta, psi ICRS) while the
            # existing SkyMove geometry assumes SSB ecliptic. The index
            # map is unchanged (cos_iota=6, psi=7, alpha=8, sin_delta=9):
            #
            #   from lisatools.sampling.moves.skymodehop import SkyMove
            #   angles_map = dict(cosinc=6, psi=7, lam=8, sinbeta=9)
            #   ... (SkyMove(ind_map=angles_map, which=...), w) ...
            self.inner_moves = [
                (StretchMove(), 1.0),
            ]

    def init_setup(self):
        """Run sampling-info and state-backend initialization."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the MBH state and backend to :class:`MBHState` / :class:`MBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = MBHState

        if self.branch_backend is None:
            self.branch_backend = MBHHDFBackend


def make_emri_transform_container(fill_values):
    """Build the stock EMRI :class:`TransformContainer` (forward + inverse).

    Sampling basis (12 params): ``logm1, m2, a, p0, e0, dist, qS (cos),
    phiS, qK (cos), phiK, Phi_phi0, Phi_r0``; the output basis adds the
    filled ``xI0`` and ``Phi_theta0``. The inverse transforms enable
    ``both_inverse_transforms`` (full basis -> sampling basis).

    Args:
        fill_values: Length-2 sequence with the fill values for ``xI0``
            (inclination) and ``Phi_theta0``.
    """
    input_basis = [
        "logm1",
        "m2",
        "a",
        "p0",
        "e0",
        "dist",
        "cosqS",
        "phiS",
        "cosqK",
        "phiK",
        "Phi_phi0",
        "Phi_r0",
    ]

    output_basis = [
        "m1",
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
        "xI0": fill_values[0],  # inclination
        "Phi_theta0": fill_values[1],  # Phi_theta
    }

    emri_transform_fn_in = {
        output_basis[0]: np.exp,  # M
        output_basis[7]: np.arccos,  # qS
        output_basis[9]: np.arccos,  # qK
    }

    emri_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # M
        output_basis[7]: np.cos,  # qS
        output_basis[9]: np.cos,  # qK
    }
    key_map = {
        "logm1": "m1",
        "cosqK": "qK",
        "cosqS": "qS",
    }

    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=emri_transform_fn_in,
        fill_dict=emri_fill_dict,
        inverse_parameter_transforms=emri_inverse_transform_fn_in,
        key_map=key_map
    )


from ..hdfbackend import EMRIHDFBackend
from ..state import EMRIState


@dataclasses.dataclass
class EMRISettings(Settings):
    """Settings dataclass describing the EMRI branch in an Erebor-style recipe.

    Holds parameter ranges for the intrinsic EMRI parameters (primary mass,
    secondary mass, spin, semi-latus rectum, eccentricity), the waveform
    configuration, and search/sampling helpers consumed by :class:`EMRISetup`.

    # TODO/DOCS: confirm the role of ``info_matrix_gen`` and ``fill_values``
    # in coordinating EMRI starts with the Fisher / search pipeline.
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    m2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    a_lims: typing.List[float] = dataclasses.field(default_factory=list)
    p0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    e0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([1.0, 0.0]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    emri_search_file_key: Optional[str] = "_emri_search_tmp_file"


class EMRISetup(Setup):
    """:class:`Setup` for extreme mass-ratio inspirals in the Erebor recipe.

    Args:
        emri_settings: Settings dataclass with EMRI parameter ranges,
            waveform configuration, and search helpers.
    """

    def __init__(self, emri_settings: EMRISettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(emri_settings)

        level = logging.DEBUG
        name = "EMRISetup"
        self.logger = init_logger(filename="emri_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the EMRI :class:`TransformContainer`, prior, periodicity, and tempering ladder."""
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
            self.transform = make_emri_transform_container(self.fill_values)

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
        """Build the EMRI prior dictionary, overriding intrinsic ranges from settings.

        Starts from a default uniform prior on each input-basis parameter and
        overrides the intrinsic parameter ranges (``logm1``, ``m2``, ``a``,
        ``p0``, ``e0``) with the corresponding ``*_lims`` attributes when
        those have been set.

        Args:
            input_basis: Ordered list of EMRI parameter names matching the
                sampler's input basis.
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
        """Run sampling-info and state-backend initialization."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the EMRI state and backend to :class:`EMRIState` / :class:`EMRIHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = EMRIState

        if self.branch_backend is None:
            self.branch_backend = EMRIHDFBackend


def cosqS_to_beta(x):
    """Transform ``cosqS`` to ecliptic latitude ``beta = pi/2 - arccos(cosqS)``."""
    return np.pi / 2.0 - np.arccos(x)


def beta_to_cosqS(x):
    """Transform ecliptic latitude ``beta`` to ``cosqS = sin(beta)`` (inverse of :func:`cosqS_to_beta`)."""
    return np.sin(x)


def make_sobbh_transform_container():
    """Build the stock SOBBH :class:`TransformContainer` (forward + inverse).

    Sampling basis (11 params): ``logm1, logm2, s1, s2, dist, cosinc,
    f_low, phiS, cosqS, psi, phi0``; output basis: ``m1, m2, s1, s2,
    dist, inc, f_low, lam, beta, psi, phi0``. The inverse transforms
    enable ``both_inverse_transforms`` (full basis -> sampling basis).
    """
    input_basis = [
        "logm1",
        "logm2",
        "s1",
        "s2",
        "dist",
        "cosinc",
        "f_low",
        "phiS",
        "cosqS",
        "psi",
        "phi0",
    ]

    output_basis = [
        "m1",
        "m2",
        "s1",
        "s2",
        "dist",
        "inc",
        "f_low",
        "lam",
        "beta",
        "psi",
        "phi0",
    ]

    sobbh_transform_fn_in = {
        output_basis[0]: np.exp,  # m1
        output_basis[1]: np.exp,  # m2
        output_basis[5]: np.arccos,  # inc <- cosinc
        output_basis[8]: cosqS_to_beta,  # beta <- cosqS
    }

    sobbh_inverse_transform_fn_in = {
        output_basis[0]: np.log,  # logm1 <- m1
        output_basis[1]: np.log,  # logm2 <- m2
        output_basis[5]: np.cos,  # cosinc <- inc
        output_basis[8]: beta_to_cosqS,  # cosqS <- beta
    }

    # Every input-basis name not already present in output_basis
    # needs a key_map entry pointing to its renamed counterpart.
    return TransformContainer(
        input_basis=input_basis,
        output_basis=output_basis,
        parameter_transforms=sobbh_transform_fn_in,
        key_map={
            "logm1": "m1",
            "logm2": "m2",
            "cosinc": "inc",
            "cosqS": "beta",
            "phiS": "lam",
        },
        inverse_parameter_transforms=sobbh_inverse_transform_fn_in,
    )


from ..hdfbackend import SOBBHHDFBackend
from ..state import SOBBHState


@dataclasses.dataclass
class SOBBHSettings(Settings):
    """Settings dataclass describing the SOBBH branch in an Erebor-style recipe.

    Mirrors :class:`EMRISettings` for stellar-origin BBHs: parameter
    ranges for the intrinsic SOBBH parameters (component masses, aligned
    spins, distance, starting GW frequency) plus the waveform / sampling
    knobs consumed by :class:`SOBBHSetup`.

    The sampling basis is 11 params (no ``fill_values`` by default — the
    SOBBH waveform consumes everything sampled).
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    logm2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    s1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    s2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    f_low_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None
    info_matrix_gen: Optional[Any] = None
    # No fill_values used by default; keep the dataclass field for parity
    # with EMRISettings so settings files can opt-in without breakage.
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    num_prop_repeats: Optional[int] = 10
    sobbh_search_file_key: Optional[str] = "_sobbh_search_tmp_file"


class SOBBHSetup(Setup):
    """:class:`Setup` for stellar-origin BBHs in the Erebor recipe.

    Args:
        sobbh_settings: Settings dataclass with SOBBH parameter ranges,
            waveform configuration, and search helpers.
    """

    def __init__(self, sobbh_settings: SOBBHSettings):
        super().__init__(sobbh_settings)

        level = logging.DEBUG
        name = "SOBBHSetup"
        self.logger = init_logger(
            filename="sobbh_setup.log",
            level=level,
            name=name,
            log_dir=getattr(self, "log_dir", None),
        )

        self.init_setup()

    def init_sampling_info(self):
        """Build the SOBBH :class:`TransformContainer`, prior, periodicity, and tempering ladder.

        Sampling basis (11 params): ``logm1, logm2, s1, s2, dist, cosinc,
        f_low, phiS, cosqS, psi, phi0``. ``M`` (==logm1.exp) and ``m2``
        are exponentiated; ``cosinc, cosqS`` are inverted to ``inc, qS``;
        ``phiS`` is mapped to the ``lambda`` argument that ResponseWrapper
        uses, ``qS -> beta = pi/2 - qS``.
        """
        input_basis = [
            "logm1",
            "logm2",
            "s1",
            "s2",
            "dist",
            "cosinc",
            "f_low",
            "phiS",
            "cosqS",
            "psi",
            "phi0",
        ]

        if self.transform is None:
            self.transform = make_sobbh_transform_container()

        if self.periodic is None:
            self.periodic = {
                "sobbh": {
                    "phiS": 2 * np.pi,
                    "psi": np.pi,
                    "phi0": 2 * np.pi,
                }
            }

        self.setup_priors(input_basis)

        if self.betas is None:
            ntemps_pe = 24
            betas = 1 / 1.2 ** np.arange(ntemps_pe)
            self.betas = betas

        self.logger.info(f"Using betas: {self.betas} in SOBBH branch")

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
        """Build the SOBBH prior dictionary, overriding intrinsic ranges from settings."""
        priors_sobbh = {
            input_basis[0]: uniform_dist(np.log(2.0), np.log(100.0)),     # logm1 (M_sun)
            input_basis[1]: uniform_dist(np.log(2.0), np.log(100.0)),     # logm2
            input_basis[2]: uniform_dist(-0.99, 0.99),                    # s1z
            input_basis[3]: uniform_dist(-0.99, 0.99),                    # s2z
            input_basis[4]: uniform_dist(0.01, 10.0),                     # dist (Gpc)
            input_basis[5]: uniform_dist(-1.0, 1.0),                      # cosinc
            input_basis[6]: uniform_dist(1.0e-3, 1.0e-1),                 # f_low (Hz)
            input_basis[7]: uniform_dist(0.0, 2 * np.pi),                 # phiS
            input_basis[8]: uniform_dist(-1.0, 1.0),                      # cosqS
            input_basis[9]: uniform_dist(0.0, np.pi),                     # psi
            input_basis[10]: uniform_dist(0.0, 2 * np.pi),                # phi0
        }

        limits = ["logm1_lims", "logm2_lims", "s1_lims", "s2_lims", "f_low_lims"]
        target_idx = {"logm1_lims": 0, "logm2_lims": 1, "s1_lims": 2, "s2_lims": 3, "f_low_lims": 6}
        for lims in limits:
            cfg = getattr(self, lims)
            if cfg is not None and len(cfg) == 2:
                i = target_idx[lims]
                self.logger.info(
                    f"Setting prior for parameter {i} ({input_basis[i]}) using limits {cfg}"
                )
                priors_sobbh[input_basis[i]] = uniform_dist(*cfg)

        self.priors = {"sobbh": ProbDistContainer(priors_sobbh)}

    def init_setup(self):
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the SOBBH state and backend to :class:`SOBBHState` / :class:`SOBBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = SOBBHState

        if self.branch_backend is None:
            self.branch_backend = SOBBHHDFBackend


from lisatools.detector import EqualArmlengthOrbits


@dataclasses.dataclass
class PSDSettings(Settings):
    """Settings dataclass describing the PSD branch in an Erebor-style recipe.

    Configures the PSD model parameters that are sampled jointly with the
    other source branches (e.g. instrumental noise levels) and feeds into
    :class:`PSDSetup`.

    # TODO/DOCS: confirm coordination between ``nknots`` and ``ndim`` for
    # spline-style PSD parameterizations vs. the default 4-parameter setup.
    """

    psd_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 4
    transform: Optional[TransformContainer] = None
    injection: Optional[np.ndarray] = None 
    nknots: Optional[int] = None
    num_prop_repeats: int = 50

class PSDSetup(Setup):
    """:class:`Setup` for the instrumental PSD branch in the Erebor recipe.

    Args:
        psd_settings: Settings dataclass with PSD kwargs, prior, and
            tempering configuration.
    """

    def __init__(self, psd_settings: PSDSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(psd_settings)

        level = logging.DEBUG
        name = "PSDSetup"
        self.logger = init_logger(filename="psd_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the PSD prior, tempering ladder, and default ``psd_kwargs``."""
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

    def init_setup(self):
        """Run sampling-info initialization for the PSD branch."""
        self.init_sampling_info()


@dataclasses.dataclass
class GalForSettings(Settings):
    """Settings dataclass describing the galactic-foreground branch in an Erebor-style recipe.

    Holds the foreground-model kwargs and the leaf / dimension counts used
    by :class:`GalForSetup` to build the prior on the stochastic galactic
    foreground (amplitude, knee, slopes, etc.).
    """

    galfor_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 5


class GalForSetup(Setup):
    """:class:`Setup` for the galactic-foreground branch in the Erebor recipe.

    Args:
        galfor_settings: Settings dataclass with foreground kwargs, prior,
            and tempering configuration.
    """

    def __init__(self, galfor_settings: GalForSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(galfor_settings)

        level = logging.DEBUG
        name = "GalForSetup"
        self.logger = init_logger(filename="galfor_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None))

        self.init_setup()

    def init_sampling_info(self):
        """Build the galactic-foreground prior, tempering kwargs, and default ``galfor_kwargs``."""
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
        """Run sampling-info initialization for the galactic-foreground branch."""
        self.init_sampling_info()


def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    """Construct the default :class:`GalForSetup` for an Erebor run.

    Builds a :class:`GalForSettings` from the run-wide ``Tobs`` / ``dt``
    carried on ``general_set`` and wraps it in a :class:`GalForSetup`.

    # TODO/DOCS: the local ``Tobs = YRSID_SI`` and ``dt = 10.0`` are dead
    # code (overridden by ``general_set``); confirm whether they should be
    # used as fallbacks.

    Args:
        general_set: Run-wide :class:`GeneralSetup` providing ``Tobs`` and
            ``dt``.

    Returns:
        Configured :class:`GalForSetup` ready for use in the Erebor
        pipeline.
    """
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
    # general_set = get_general_erebor_settings()
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
