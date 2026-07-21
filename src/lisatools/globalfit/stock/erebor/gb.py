"""GB (galactic-binary) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from gbgpu.utils.utility import get_fdot, get_N

from lisatools.domains import DomainSettingsBase, FDSettings, WDMSettings
from lisatools.utils.constants import YRSID_SI

from ..base import env_default
from ...engine import Settings, Setup
from ...hdfbackend import GBHDFBackend
from ...loginfo import init_logger
from ...priors.gbpriors import get_fdot_mojito
from ...state import GBState
from .transforms import make_gb_transform_container

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
    # Peak GPU-memory cap for the GB special move. ``n_subbands`` is the
    # maximum number of (temp, walker, band) sub-band buffer cells resident
    # at once (the ``BandScheduler`` slot count / the move's
    # ``num_band_preload``). The move preloads at most this many cells and
    # swaps finished cells out for pending ones, so PEAK sub-band-buffer
    # memory scales with ``n_subbands``, not the total cell count
    # (ntemps*nwalkers*n_bands). Lower it to bound peak GPU memory on large
    # runs; the historical move default is 20000 (effectively "load every
    # cell"). Env: ``GB_N_SUBBANDS``. Flows to the move via
    # ``group_proposal_kwargs["num_band_preload"]`` (see ``__post_init__``).
    # Peak sub-band-buffer memory is ``n_subbands * per_slot`` once the cap
    # binds (total cells > n_subbands); ntemps/nwalkers/n_bands only decide
    # whether it binds and how many scheduler rounds run, NOT the peak. Default
    # 1024 (was an effectively-uncapped 20000): bounds peak on large multi-band
    # runs while leaving small runs -- whose cell count is already < 1024 --
    # untouched. Env: ``GB_N_SUBBANDS``.
    n_subbands: int = dataclasses.field(
        default_factory=env_default("GB_N_SUBBANDS", 1024, int)
    )
    # Sampling-basis knob: sample **chirp mass ``Mc``** at slot 2 (Msol)
    # instead of ``fdot``. The physical basis is unchanged (fdot); the
    # sampling->physical :class:`TransformContainer` gets a multi-parameter
    # ``(f0, Mc) -> (f0, fdot)`` transform via
    # :func:`gbgpu.utils.utility.get_fdot`, and ``key_map={"Mc": "fdot"}``
    # places the sampled Mc at the fdot slot before the transform runs.
    # **Default: True** -- Mc is the physically meaningful axis (fdot varies
    # by ~10 orders of magnitude across the GB band, while Mc is ~O(1) Msol
    # everywhere), and the F-stat proposal / astrophysical joint prior both
    # live natively in the Mc axis. Set ``GB_USE_CHIRP_MASS=0`` to fall back
    # to the legacy fdot sampling basis (requires ``fdot_lims`` populated).
    use_chirp_mass: bool = dataclasses.field(
        default_factory=env_default("GB_USE_CHIRP_MASS", True, bool)
    )
    # When ``use_chirp_mass`` is on, swap the separate uniform priors on
    # ``f0`` and ``Mc`` for the 6-component astrophysical GMM fit from
    # ``heatmap_GMMs.ipynb`` (combined heatmap of 102 population-synthesis
    # models; :class:`lisatools.sampling.f0_mchirp_prior.F0McGMMSampling`,
    # truncated to the run's (f0, Mc) box and renormalized). **Default:
    # True** -- every GB-carrying stock fit now samples the astrophysical
    # joint (f0, Mc) prior; set ``GB_USE_ASTROPHYSICAL_F0_MC_PRIOR=0`` for
    # the legacy separate uniforms. Ignored (with the legacy fdot basis)
    # when ``use_chirp_mass=False``.
    use_astrophysical_f0_mc_prior: bool = dataclasses.field(
        default_factory=env_default("GB_USE_ASTROPHYSICAL_F0_MC_PRIOR", True, bool)
    )
    # Custom RJ-birth distribution for the prior RJ moves. An eryn
    # duck-typed distribution over the FULL 8-column GB sampling basis
    # (``rvs(size) -> (size, 8)`` with f0 in mHz at column 1 and Mc at
    # column 2; ``logpdf((n, 8)) -> (n,)``). Build one from a 4-D intrinsic
    # proposal (e.g. :class:`lisatools.sampling.fstat_proposal
    # .FStatProposal4D`) with :func:`lisatools.sampling.fstat_proposal
    # .make_gb_rj_birth_container`; wrap narrow proposals in a
    # ``UniformFloorMixture`` so death factors stay finite wherever
    # refined leaves can drift. Must pickle/deepcopy (sprint rule) --
    # FStatProposal4D-backed containers do. None (default) keeps the
    # stock global-prior births.
    rj_birth_distribution: typing.Optional[typing.Any] = None
    # Task-b: narrow per-band WDM slabs. Each per-band sub-band-buffer slab
    # spans a few WDM layers centered on the band instead of the full analysis
    # band ``Nf_active``, cutting the dominant buffer memory term by
    # ~Nf_active/slab_Nf. WDM path only (FD is already per-band narrow).
    # Requires the chunked-het backend built with the task-b per-slab origin
    # (the default CUDA/CPU build). Env: GB_WDM_BAND_SLAB_LAYERS.
    #   None -> OFF (full active band; bit-identical to pre-task-b). [default]
    #   0    -> AUTO: band layer span + 2*(leakage + wdm_slab_guard_layers),
    #           leakage=2 (recommended-Tukey estimate) -> ~band_span + 6.
    #   N>0  -> EXPLICIT N layers.
    wdm_band_slab_layers: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GB_WDM_BAND_SLAB_LAYERS", None, int)
    )
    # Adjustable guard (extra WDM layers each side) used by the AUTO slab size
    # (wdm_band_slab_layers=0). Env: GB_WDM_SLAB_GUARD_LAYERS.
    wdm_slab_guard_layers: int = dataclasses.field(
        default_factory=env_default("GB_WDM_SLAB_GUARD_LAYERS", 1, int)
    )
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

        Plain parameter names (same convention as
        :func:`make_gb_transform_container`) with ICRS sky angles
        (``alpha`` / ``sin_delta``) -- the run frame is ICRS. ``A`` is
        sampled in ``ln A`` (the ``np.exp`` transform maps to the physical
        amplitude), ``f0`` in mHz.

        When :attr:`GBSettings.use_chirp_mass` is on, slot 2 of the sampling
        basis carries ``Mc`` (chirp mass, Msol) instead of ``fdot``. A
        multi-parameter ``(f0, Mc) -> (f0, fdot)`` transform via
        :func:`gbgpu.utils.utility.get_fdot` recovers the physical fdot after
        ``fill_values`` moves the sampled ``Mc`` to the fdot slot (via
        ``key_map={"Mc": "fdot"}``). When
        :attr:`GBSettings.use_astrophysical_f0_mc_prior` is also on, the
        separate ``f0`` and ``Mc`` uniforms are replaced by the 6-component
        heatmap GMM from :func:`~lisatools.sampling.f0_mchirp_prior
        .F0McGMMSampling.from_heatmap` under a tuple key ``("f0", "Mc")``.
        """
        third_name = "Mc" if self.use_chirp_mass else "fdot"
        input_basis = ["A", "f0", third_name, "phi0",
                       "cos_iota", "psi", "alpha", "sin_delta"]

        if self.transform is None:
            # THE single GB transform factory (phi0 sign convention lives
            # there, nowhere else).
            self.transform = make_gb_transform_container(
                use_chirp_mass=self.use_chirp_mass
            )

        if self.periodic is None:
            self.periodic = {"gb": {"phi0": 2*np.pi, "psi": np.pi, "alpha": 2 * np.pi}}

        if self.priors is None:
            if self.use_chirp_mass and self.use_astrophysical_f0_mc_prior:
                # Joint tuple-key prior on (f0[mHz], Mc[Msol]) from the
                # heatmap 6-component GMM fit.
                from lisatools.sampling.f0_mchirp_prior import F0McGMMSampling

                # Truncate the heatmap GMM to the run's sampled box and
                # renormalize (RJ birth/death compares prior masses across
                # leaf counts, so the truncated prior must integrate to 1
                # over exactly the space the sampler covers). Gaussian
                # tails keep the density finite anywhere in the box, even
                # outside the original heatmap support.
                _f0_mc_prior = F0McGMMSampling.from_heatmap(
                    f0_lims_mHz=tuple(np.asarray(self.f0_lims) * 1e3),
                    mc_lims=tuple(self.m_chirp_lims) if self.m_chirp_lims
                    else None,
                )
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    ("f0", "Mc"): _f0_mc_prior,
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }
            elif self.use_chirp_mass:
                if not self.m_chirp_lims:
                    raise ValueError(
                        "use_chirp_mass=True requires GBSettings.m_chirp_lims "
                        "to be set (a two-element [Mc_min, Mc_max] in Msol)."
                    )
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),
                    input_basis[2]: uniform_dist(self.m_chirp_lims[0], self.m_chirp_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: sort defensively.
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
                }
            else:
                priors_gb = {
                    input_basis[0]: uniform_dist(*(np.log(np.asarray(self.A_lims)))),
                    input_basis[1]: uniform_dist(*(np.asarray(self.f0_lims) * 1e3)),  # AmplitudeFrequencySNRPrior(rho_star, frequency_prior, L, Tobs, fd=fd),  # use sangria as a default
                    input_basis[2]: uniform_dist(self.fdot_lims[0], self.fdot_lims[1]),
                    input_basis[3]: uniform_dist(self.phi0_lims[0], self.phi0_lims[1]),
                    # cos is DECREASING on [0, pi]: cos(iota_lims) comes out
                    # (max, min), so sort into increasing order before handing
                    # it to uniform_dist -- never rely on the dist silently
                    # swapping reversed bounds. (Same defensive sort on the
                    # sin(delta) line even though sin is increasing there.)
                    input_basis[4]: uniform_dist(*np.sort(np.cos(self.iota_lims))),
                    input_basis[5]: uniform_dist(self.psi_lims[0], self.psi_lims[1]),
                    input_basis[6]: uniform_dist(self.alpha_lims[0], self.alpha_lims[1]),
                    input_basis[7]: uniform_dist(*np.sort(np.sin(self.delta_lims))),
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
        # Peak sub-band-buffer memory cap. Injected here (not only in the
        # None-default dict above) so an explicit ``group_proposal_kwargs``
        # still gets the ``n_subbands`` knob unless it names one itself.
        self.group_proposal_kwargs.setdefault("num_band_preload", int(self.n_subbands))
        # Task-b narrow per-band WDM slabs (None -> full active band).
        self.group_proposal_kwargs.setdefault(
            "wdm_band_slab_layers", self.wdm_band_slab_layers
        )
        self.group_proposal_kwargs.setdefault(
            "wdm_slab_guard_layers", self.wdm_slab_guard_layers
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

