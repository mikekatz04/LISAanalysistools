"""EMRI (extreme-mass-ratio inspiral) branch blocks for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

from ....sampling.prior import EMRIKerrDomainPrior

from ...engine import Settings, Setup
from ...hdfbackend import EMRIHDFBackend
from ...loginfo import init_logger
from ..base import env_default
from ...state import EMRIState
from .transforms import make_emri_transform_container

@dataclasses.dataclass
class EMRISettings(Settings):
    """Settings dataclass describing the EMRI branch in an Erebor-style recipe.

    Holds parameter ranges for the intrinsic EMRI parameters (primary mass,
    secondary mass, spin, semi-latus rectum, eccentricity), the waveform
    configuration, and search/sampling helpers consumed by :class:`EMRISetup`.

    # TODO/DOCS: confirm the role of ``info_matrix_gen`` and ``fill_values``
    # in coordinating EMRI starts with the information matrix / search pipeline.
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    m2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    # per-leaf ladder size for the add/remove move (engine is cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("EMRI_NTEMPS", 24, int))
    a_lims: typing.List[float] = dataclasses.field(default_factory=list)
    p0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    e0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    # ``[xI0, Phi_theta0]`` transform fills. None -> resolved per leaf from
    # the injection catalogue in ``prepare_emri_branch`` (an ``(nleaves, 2)``
    # table — xI0 is the intrinsic prograde/retrograde flag and can differ
    # per leaf), with a ``[1.0, 0.0]`` prograde fallback in
    # ``init_sampling_info`` when nothing resolves it.
    fill_values: Optional[np.ndarray] = None
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    # default inner-move stack when ``inner_moves`` is unset: "eigen"
    # (information-matrix eigen-axis jump) or "stretch" (legacy escape)
    inner_move_kind: str = dataclasses.field(
        default_factory=env_default("EMRI_INNER_MOVE_KIND", "eigen", str)
    )
    # leaf-visit cadence for the eigen inner-move table refresh — SPARSE:
    # EMRI likelihood rows are per-row dense (~1 s), so a table build is
    # minutes; it happens on the first visit and then every N-th
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_REFRESH", 100, int)
    )
    # table scope: per-row-dense likelihood -> ONE table, built at the
    # max-lnL cold-chain walker
    eigen_table_scope: str = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_SCOPE", "walker_max", str)
    )
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

        if self.fill_values is None:
            # prograde-equatorial fallback (synthetic/no-catalogue runs)
            self.fill_values = np.array([1.0, 0.0])
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
            # self.ntemps, NOT a hardcoded 24 (fixed 2026-08-26): the
            # EMRI_NTEMPS knob (and the _lite presets' ntemps=2!) was
            # silently ignored -- the observed ladders were always 24
            # rungs. Field default is now 24 so an UNSET knob keeps the
            # validated full-year ladder bit-identical.
            ntemps_pe = int(self.ntemps)
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
            from .common import resolve_inner_moves

            resolve_inner_moves(self)

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

        # The logm1 / m2 / p0 boxes must contain every mojito_light catalogue
        # source (a start point outside the prior makes lnpdiff NaN and
        # freezes the sampler): the catalogue spans M in [3.92e5, 1e7],
        # m2 up to 127.7, p0 down to 2.12.
        priors_emri = {
            input_basis[0]: uniform_dist(np.log(3e5), np.log(1.5e7)),  # log m1
            input_basis[1]: uniform_dist(1, 200),  # m2
            input_basis[2]: uniform_dist(0.01, 0.999),  # a
            input_basis[3]: uniform_dist(2.0, 100.0),  # p0
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

        # Joint (a, p0, e0) FEW domain-of-validity cut on top of the boxes:
        # combinations the Kerr ecc-eq grid cannot generate get logpdf -inf
        # (and rvs rejection-resamples), matching the -1e300 likelihood
        # sentinel applied when a waveform call fails on a domain error.
        # Shared prior-level FEW domain gate. With PER-LEAF fills the prior
        # has no leaf identity: a uniform prograde/retrograde population
        # keeps its exact gate; mixed populations fall back to the prograde
        # grid here and rely on the likelihood's FEW domain-error sentinel
        # (-1e300) for per-leaf validity.
        # TODO: per-leaf domain gate in EMRIKerrDomainPrior.
        _fv = np.asarray(self.fill_values, dtype=float)
        _xi_vals = np.unique(_fv[:, 0]) if _fv.ndim == 2 else _fv[:1]
        if len(_xi_vals) == 1:
            _xi_fill = float(_xi_vals[0])
        else:
            self.logger.warning(
                "EMRI leaves mix prograde and retrograde xI0 "
                f"({_xi_vals.tolist()}); the shared prior domain gate uses "
                "the prograde grid (per-leaf validity via the likelihood's "
                "FEW domain-error sentinel)."
            )
            _xi_fill = 1.0
        self.priors = {
            "emri": EMRIKerrDomainPrior(
                priors_emri,
                a_index=input_basis.index("a"),
                p0_index=input_basis.index("p0"),
                e0_index=input_basis.index("e0"),
                xI_fill=_xi_fill,
            )
        }

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
