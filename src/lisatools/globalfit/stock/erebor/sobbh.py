"""SOBBH (stellar-origin BBH) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

from ...engine import Settings, Setup
from ...hdfbackend import SOBBHHDFBackend
from ...loginfo import init_logger
from ..base import env_default
from ...state import SOBBHState
from .transforms import make_sobbh_transform_container

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
    # per-leaf ladder size for the add/remove move (engine is cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("SOBBH_NTEMPS", 4, int))
    f_low_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None
    # external eigen-table builder for the addremove eigen inner move:
    # ``builder(move, leaf, widths) -> (axes, sigmas)``. None (default)
    # uses likelihood second differences on the branch's own scorer
    # (eigen_refresh.eigen_table_from_ll); eigen_table_from_waveform is
    # the Gram-form alternative a builder can wrap.
    info_matrix_gen: Optional[Any] = None
    # No fill_values used by default; keep the dataclass field for parity
    # with EMRISettings so settings files can opt-in without breakage.
    fill_values: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    # default inner-move stack when ``inner_moves`` is unset: "eigen"
    # (information-matrix eigen-axis jump) or "stretch" (legacy escape)
    inner_move_kind: str = dataclasses.field(
        default_factory=env_default("SOBBH_INNER_MOVE_KIND", "eigen", str)
    )
    # leaf-visit cadence for the eigen inner-move table refresh (the SOBBH
    # information matrix costs ~0.7 s on the batched chunked scorer)
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_REFRESH", 10, int)
    )
    # table scope: SOBBH's batched chunked scorer makes per-(temperature,
    # walker) information matrices affordable — every point its own table
    eigen_table_scope: str = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_SCOPE", "per_walker", str)
    )
    num_prop_repeats: Optional[int] = 10
    sobbh_search_file_key: Optional[str] = "_sobbh_search_tmp_file"
    # ---- per-leaf ladder + eigen-step knobs (user rulings 2026-10-13) -------
    # Same meaning as the EMRISettings fields of the same names
    # (SOBBH_LADDER_RESET_ON_RESUME / SOBBH_LADDER_REANCHOR_NATS /
    # SOBBH_EIGEN_JUMP_MIX / SOBBH_EIGEN_JUMP_FACTOR).
    ladder_reset_on_resume: bool = dataclasses.field(
        default_factory=env_default("SOBBH_LADDER_RESET_ON_RESUME", False, bool)
    )
    ladder_reanchor_nats: float = dataclasses.field(
        default_factory=env_default("SOBBH_LADDER_REANCHOR_NATS", 200.0, float)
    )
    # Cap on the default ladder's top temperature (None = eryn's uncapped
    # ~25 % spacing, T ~ 150 at 8 rungs). See common.default_source_ladder.
    ladder_tmax: Optional[float] = dataclasses.field(
        default_factory=env_default("SOBBH_LADDER_TMAX", None, float)
    )
    eigen_jump_mix: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_JUMP_MIX", 0.0, float)
    )
    eigen_jump_factor: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_JUMP_FACTOR", 3.0, float)
    )
    # Same meaning as EMRISettings.eigen_axis_adapt (SOBBH_EIGEN_AXIS_ADAPT;
    # SOBBH_EIGEN_ADAPT_VISITS / SOBBH_EIGEN_ADAPT_MIN_VISITS on the move).
    eigen_axis_adapt: bool = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ADAPT", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_pool / eigen_temper_sigmas
    # (SOBBH_EIGEN_ADAPT_POOL / SOBBH_EIGEN_TEMPER_SIGMAS; ruling 2026-10-15).
    eigen_adapt_pool: bool = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_ADAPT_POOL", False, bool)
    )
    eigen_temper_sigmas: bool = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_TEMPER_SIGMAS", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_max_factor
    # (SOBBH_EIGEN_ADAPT_MAX_FACTOR, default 100; <= 0 or inf = no cap).
    eigen_adapt_max_factor: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_ADAPT_MAX_FACTOR", 100.0, float)
    )
    # Same meaning as the EMRISettings fields of the 2026-10-18 audit
    # (SOBBH_EIGEN_AXIS_ACC_LOG / _TUNE / _MIN / _MAX / _FACTOR,
    # SOBBH_DE_JUMP_MIX / _MODE_FRAC).
    eigen_axis_acc_log: bool = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ACC_LOG", False, bool)
    )
    eigen_axis_acc_tune: bool = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ACC_TUNE", False, bool)
    )
    eigen_axis_acc_min: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ACC_MIN", 0.1, float)
    )
    eigen_axis_acc_max: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ACC_MAX", 0.5, float)
    )
    eigen_axis_acc_factor: float = dataclasses.field(
        default_factory=env_default("SOBBH_EIGEN_AXIS_ACC_FACTOR", 2.0, float)
    )
    de_jump_mix: float = dataclasses.field(
        default_factory=env_default("SOBBH_DE_JUMP_MIX", 0.0, float)
    )
    de_jump_mode_frac: float = dataclasses.field(
        default_factory=env_default("SOBBH_DE_JUMP_MODE_FRAC", 0.5, float)
    )


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
            shared_console=False,
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
            # make_ladder(ndim, ntemps=SOBBH_NTEMPS), not 1/1.2^k (user
            # ruling 2026-10-13); sized by the branch's own ntemps knob (a
            # hard-coded 24 here used to shadow it and the lite presets).
            # An explicit ``betas`` wins. See common.default_source_ladder.
            from .common import default_source_ladder

            self.betas = default_source_ladder(self.ndim, input_basis, self.ntemps,
                                               tmax=self.ladder_tmax)

        self.logger.info(f"Using betas: {self.betas} in SOBBH branch")

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from .common import resolve_inner_moves

            resolve_inner_moves(self)

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
