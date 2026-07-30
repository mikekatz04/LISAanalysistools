"""Stochastic-signal branch blocks (SGWB) for the Erebor recipe.

The stochastic gravitational-wave background is a *signal* branch (as opposed
to the instrument PSD + galactic-confusion foreground, which are treated as
noise and live in :mod:`.noise`). It is sampled jointly with the noise branches
by the single :class:`~lisatools.globalfit.moves.psdmove.PSDMove`, and folded
into the covariance through the :class:`~lisatools.sensitivity.SGWB` component
of :class:`~lisatools.sensitivity.CompositeSensitivityBackend`.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Optional

import numpy as np
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer

from ...engine import GeneralSetup, Settings, Setup
from ...hdfbackend import ModuleSubBackend
from ...state import ModuleSubState
from ...loginfo import init_logger
from ..base import env_default


@dataclasses.dataclass
class SGWBSettings(Settings):
    """Settings dataclass describing the SGWB branch in an Erebor-style recipe.

    Holds the spectral-template kwargs and the leaf / dimension counts used by
    :class:`SGWBSetup` to build the prior on the stochastic gravitational-wave
    background. The default ``ndim=2`` matches
    :class:`~lisatools.stochastic.PowerLawSGWB` ``(log10_A, alpha)``; other
    templates (LogNormal, PhaseTransition) need matching ``ndim`` + priors and a
    matching ``sgwb_stochastic_fn`` on the sensitivity backend.
    """

    # the sgwb move's OWN ladder size (noise-move split 2026-07)
    ntemps: int = dataclasses.field(default_factory=env_default("SGWB_NTEMPS", 12, int))
    sgwb_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 2
    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("SGWB_NUM_PROP_REPEATS", 50, int)
    )
    injection: Optional[np.ndarray] = None
    # SGWB spectral-template choice for this branch — swap it the way a source
    # branch swaps its waveform (``fit.sgwb.stochastic_fn = "LogNormalSGWB"``,
    # with a matching ``ndim`` + prior). The variant's ``finalize_general``
    # threads this onto the CompositeSensitivityBackend.
    stochastic_fn: str = "PowerLawSGWB"


class SGWBSetup(Setup):
    """:class:`Setup` for the SGWB branch in the Erebor recipe.

    Args:
        sgwb_settings: Settings dataclass with SGWB kwargs, prior, and
            tempering configuration.
    """

    def __init__(self, sgwb_settings: SGWBSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(sgwb_settings)

        level = logging.DEBUG
        name = "SGWBSetup"
        self.logger = init_logger(
            filename="sgwb_setup.log",
            level=level,
            name=name,
            log_dir=getattr(self, "log_dir", None),
        )

        self.init_setup()

    def init_sampling_info(self):
        """Build the SGWB prior and tempering kwargs."""
        if self.sgwb_kwargs is None:
            self.sgwb_kwargs = {}

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.priors is None:
            # PowerLawSGWB: (log10_A at SGWB_FREF = 25 Hz, spectral index alpha).
            # NB: this GLASS-style default range does NOT contain a strong
            # (detectable-in-short-datastream) injection like log10_A = -9.5;
            # variants that inject such a background must override the prior.
            priors_sgwb = {
                0: uniform_dist(-22.0, -18.0),  # log10_A
                1: uniform_dist(-1.0, 2.0),  # alpha
            }

            self.priors = {"sgwb": ProbDistContainer(priors_sgwb)}

        else:
            self.logger.info("Using custom priors for SGWB branch")

        if self.betas is None:
            # the sgwb move's own ladder, sized by the ntemps knob
            # (SGWB_NTEMPS; an explicit betas array wins)
            betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=self.ntemps)
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        """Run sampling-info and state-backend initialization for the SGWB branch."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the SGWB state/backend to the generic module sub-state/backend."""
        if self.branch_state is None:
            self.branch_state = ModuleSubState
        if self.branch_backend is None:
            self.branch_backend = ModuleSubBackend


def get_sgwb_erebor_settings(general_set: GeneralSetup) -> SGWBSetup:
    """Construct the default :class:`SGWBSetup` for an Erebor run."""
    sgwb_settings = SGWBSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs={},
        log_dir=getattr(general_set, "file_store_dir", None),
    )
    return SGWBSetup(sgwb_settings)
