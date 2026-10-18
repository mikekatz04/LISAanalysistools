"""MBH (massive-black-hole binary) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

from ....sampling.prior import LogUniformLinear
from ....sampling.snrladder import top_snr_value
from ...engine import Settings, Setup
from ...hdfbackend import MBHHDFBackend
from ...loginfo import init_logger
from ..base import env_default
from ...state import MBHState
from .transforms import make_mbh_transform_container

@dataclasses.dataclass
class MBHSettings(Settings):
    """Settings dataclass describing the MBH branch in an Erebor-style recipe."""

    waveform_kwargs: Optional[dict] = None
    # per-leaf ladder size for the add/remove move (engine is cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("MBH_NTEMPS", 24, int))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    # default inner-move stack when ``inner_moves`` is unset: "eigen"
    # (information-matrix eigen-axis jump) or "stretch" (legacy escape)
    inner_move_kind: str = dataclasses.field(
        default_factory=env_default("MBH_INNER_MOVE_KIND", "eigen", str)
    )
    # LISA-frame sky-mode hops mixed into the inner-move stack (the
    # ``SkyMove`` with ``coord_frame="icrs"`` -- the stock basis samples
    # alpha / sin_delta / psi in ICRS, so each row is converted to the
    # constellation frame at its own orbital phase, hopped, and converted
    # back; see ``source_runtime.make_mbh_inner_moves``). Fractions of the
    # per-leaf inner-move draws: the full 8-mode hop ("both"), the
    # reflection through the constellation plane ("lat") and the 90-degree
    # in-plane rotation ("long"); the remainder is the ``inner_move_kind``
    # stack. 0.02 / 0.04 / 0.02 (user 2026-10-12); 0 disables a kind.
    sky_move_both_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_BOTH_WEIGHT", 0.02, float)
    )
    sky_move_lat_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_LAT_WEIGHT", 0.04, float)
    )
    sky_move_long_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_LONG_WEIGHT", 0.02, float)
    )
    # leaf-visit cadence for the eigen inner-move table refresh — SPARSE:
    # MBH likelihood rows are per-row dense (~1.4 s), so a table build is
    # minutes; it happens on the first visit and then every N-th
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_REFRESH", 100, int)
    )
    # table scope: per-row-dense likelihood -> ONE table, built at the
    # max-lnL cold-chain walker
    eigen_table_scope: str = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_SCOPE", "walker_max", str)
    )
    num_prop_repeats: Optional[int] = 200
    mbh_search_file_key: Optional[str] = "_mbh_search_tmp_file"
    injection: Optional[np.ndarray] = None
    # ---- per-leaf ladder + eigen-step knobs (user rulings 2026-10-13) -------
    # Same meaning as the EMRISettings fields of the same names
    # (MBH_LADDER_RESET_ON_RESUME / MBH_LADDER_REANCHOR_NATS /
    # MBH_EIGEN_JUMP_MIX / MBH_EIGEN_JUMP_FACTOR). The jump mixture applies to
    # the eigen part of the stack only, never to the sky-mode hops.
    ladder_reset_on_resume: bool = dataclasses.field(
        default_factory=env_default("MBH_LADDER_RESET_ON_RESUME", False, bool)
    )
    ladder_reanchor_nats: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_REANCHOR_NATS", 200.0, float)
    )
    # Cap on the default ladder's top temperature (None = eryn's uncapped
    # ~25 % spacing, T ~ 150 at 8 rungs). See common.default_source_ladder.
    ladder_tmax: Optional[float] = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TMAX", None, float)
    )
    # SNR-keyed per-leaf ladders (ruling 2026-10-18; default OFF): same meaning
    # as the EMRISettings fields of the same names (MBH_LADDER_TOP_SNR /
    # _TMAX_MIN / _TOP_SNR_REFRESH / _TOP_SNR_REFRESH_PE).
    ladder_top_snr: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR", 0.0, top_snr_value)
    )
    ladder_tmax_min: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TMAX_MIN", 4.0, float)
    )
    ladder_top_snr_refresh: int = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR_REFRESH", 0, int)
    )
    ladder_top_snr_refresh_pe: bool = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR_REFRESH_PE", False, bool)
    )
    eigen_jump_mix: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_JUMP_MIX", 0.0, float)
    )
    eigen_jump_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_JUMP_FACTOR", 3.0, float)
    )
    # Same meaning as EMRISettings.eigen_axis_adapt (MBH_EIGEN_AXIS_ADAPT;
    # MBH_EIGEN_ADAPT_VISITS / MBH_EIGEN_ADAPT_MIN_VISITS on the move). The
    # sky-mode hops are not eigen moves and are unaffected.
    eigen_axis_adapt: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ADAPT", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_pool / eigen_temper_sigmas
    # (MBH_EIGEN_ADAPT_POOL / MBH_EIGEN_TEMPER_SIGMAS; ruling 2026-10-15).
    eigen_adapt_pool: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_ADAPT_POOL", False, bool)
    )
    eigen_temper_sigmas: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_TEMPER_SIGMAS", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_max_factor
    # (MBH_EIGEN_ADAPT_MAX_FACTOR, default 100; <= 0 or inf = no cap).
    eigen_adapt_max_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_ADAPT_MAX_FACTOR", 100.0, float)
    )
    # Same meaning as the EMRISettings fields of the 2026-10-18 audit
    # (MBH_EIGEN_AXIS_ACC_LOG / _TUNE / _MIN / _MAX / _FACTOR, MBH_DE_JUMP_MIX /
    # _MODE_FRAC); the DE jump rides the eigen part of the stack, never the
    # sky hops.
    eigen_axis_acc_log: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_LOG", False, bool)
    )
    eigen_axis_acc_tune: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_TUNE", False, bool)
    )
    eigen_axis_acc_min: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_MIN", 0.1, float)
    )
    eigen_axis_acc_max: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_MAX", 0.5, float)
    )
    eigen_axis_acc_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_FACTOR", 2.0, float)
    )
    de_jump_mix: float = dataclasses.field(
        default_factory=env_default("MBH_DE_JUMP_MIX", 0.0, float)
    )
    de_jump_mode_frac: float = dataclasses.field(
        default_factory=env_default("MBH_DE_JUMP_MODE_FRAC", 0.5, float)
    )



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
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

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
                # LINEAR-column log-uniform: the transform ``mT_Q`` reads Q as
                # m1/m2 >= 1. eryn's ``log_uniform`` samples/scores in ln-space
                # (support [0, 2.30] here) -- see LogUniformLinear's docstring.
                "Q": LogUniformLinear(1.0, 10.0),
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
            # make_ladder(ndim, ntemps=MBH_NTEMPS), not 1/1.2^k (user ruling
            # 2026-10-13: "the default ladders for non-RJ sources should be
            # from make_ladder"); see common.default_source_ladder. Still no
            # pinned near-prior last rung (the 2026-08-27 ruling: at
            # MBH_NTEMPS=2 ``betas[-1] = 1e-4`` left the branch untempered,
            # job 353). A run that wants another ladder sets ``mbh.betas``
            # explicitly (the field is an override and wins).
            from .common import default_source_ladder

            self.betas = default_source_ladder(self.ndim, input_basis, self.ntemps,
                                               tmax=self.ladder_tmax)

        self.logger.info(f"Using betas: {self.betas} in MBH branch")

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
            from .common import resolve_inner_moves

            # Bare ``inner_move_kind`` stack. The stock build path
            # (``source_runtime.prepare_mbh_branch``) fills ``inner_moves``
            # BEFORE this Setup is constructed, with the LISA-frame sky-mode
            # hops mixed in at ``sky_move_*_weight`` (they need the run's
            # orbits for the constellation phase, which only the build has).
            resolve_inner_moves(self)

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
