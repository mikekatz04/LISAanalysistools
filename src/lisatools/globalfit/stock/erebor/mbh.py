"""MBH (massive-black-hole binary) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, log_uniform, uniform_dist

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
    ntemps: int = dataclasses.field(default_factory=env_default("MBH_NTEMPS", 4, int))
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
