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
            # sized by the branch's own ntemps knob (SOBBH_NTEMPS); a
            # hard-coded 24 here used to shadow it (and the lite presets)
            betas = 1 / 1.2 ** np.arange(self.ntemps)
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
