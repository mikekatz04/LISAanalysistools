"""Noise branch blocks (instrumental PSD + galactic foreground) for the Erebor recipe."""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer

from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI

from ...engine import GeneralSetup, Settings, Setup
from ...hdfbackend import ModuleSubBackend
from ...state import ModuleSubState
from ...loginfo import init_logger
from ..base import env_default

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class PSDSettings(Settings):
    """Settings dataclass describing the PSD branch in an Erebor-style recipe.

    Configures the PSD model parameters that are sampled jointly with the
    other source branches (e.g. instrumental noise levels) and feeds into
    :class:`PSDSetup`.

    # TODO/DOCS: confirm coordination between ``nknots`` and ``ndim`` for
    # spline-style PSD parameterizations vs. the default 4-parameter setup.
    """

    # the psd move's OWN ladder size (noise-move split 2026-07: psd, galfor,
    # and sgwb each carry an independent move + ladder + ntemps knob)
    ntemps: int = dataclasses.field(default_factory=env_default("PSD_NTEMPS", 12, int))
    psd_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 4
    transform: Optional[TransformContainer] = None
    injection: Optional[np.ndarray] = None
    nknots: Optional[int] = None
    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("PSD_NUM_PROP_REPEATS", 50, int)
    )
    # Instrument-noise model choice for this branch — swap it the way a source
    # branch swaps its waveform. ``None`` -> the backend default
    # (``InstrumentNoise`` + ``LISAModel``, name ``model_name``). The variant's
    # ``finalize_general`` threads these onto the CompositeSensitivityBackend.
    instrument_component_cls: Any = None
    instrument_model_cls: Any = None
    model_name: Optional[str] = None

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
            # the psd move's own ladder, sized by the ntemps knob (an
            # explicit betas array wins and defines its own ntemps)
            betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=self.ntemps)
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        """Run sampling-info and state-backend initialization for the PSD branch."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the PSD state/backend to the generic module sub-state/backend."""
        if self.branch_state is None:
            self.branch_state = ModuleSubState
        if self.branch_backend is None:
            self.branch_backend = ModuleSubBackend


@dataclasses.dataclass
class GalForSettings(Settings):
    """Settings dataclass describing the galactic-foreground branch in an Erebor-style recipe.

    Holds the foreground-model kwargs and the leaf / dimension counts used
    by :class:`GalForSetup` to build the prior on the stochastic galactic
    foreground (amplitude, knee, slopes, etc.).
    """

    # the galfor move's OWN ladder size (noise-move split 2026-07)
    ntemps: int = dataclasses.field(default_factory=env_default("GALFOR_NTEMPS", 12, int))
    galfor_kwargs: typing.Dict = dataclasses.field(default_factory=dict)
    transform: Optional[TransformContainer] = None
    nleaves_max: int = 1
    nleaves_min: int = 1
    ndim: int = 5
    num_prop_repeats: int = dataclasses.field(
        default_factory=env_default("GALFOR_NUM_PROP_REPEATS", 50, int)
    )
    # Spectral / modulation model choice for this branch — swap it the way a
    # source branch swaps its waveform (``fit.galfor.stochastic_fn = ...``).
    # ``None`` -> the sensitivity backend's default
    # (``HyperbolicTangentGalacticForeground`` / stationary). The variant's
    # ``finalize_general`` threads these onto the CompositeSensitivityBackend.
    stochastic_fn: Any = None
    modulation: Any = None


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
            # Slots 3/4 are FREQUENCIES in Hz, not slopes.
            # ``HyperbolicTangentGalacticForeground.specific_Sh_function``
            # evaluates ``exp(-(f/f_1)**alpha)`` and ``tanh(-(f - fk)/f_2)``,
            # so f_1/f_2 are scale frequencies. The historical
            # Robson-&-Cornish-style priors here were the RECIPROCAL slope
            # convention (gamma in 1/Hz, ~1e3): fed in as frequencies those
            # make both factors inert (exp -> 1, tanh -> 0), collapsing the
            # whole model to a bare ``amp * f**(-7/3)`` power law with no
            # knee -- 4 of the 5 parameters unidentifiable. Ranges cover the
            # LISA band and bracket the reference scales (~3.3e-4 Hz) by
            # ~1.5 decades each side. (Box WIDTH is not why ~43% of stretch
            # proposals land out of prior: stretch is x' = y + z(x - y) with
            # x, y from the same box, so the in-box rate is scale-free --
            # measured 56.6% = 0.892**5 both before and after this change.)
            priors_galfor = {
                0: uniform_dist(1e-45, 2e-43),  # amp
                1: uniform_dist(1e-4, 5e-2),  # knee frequency f_k [Hz]
                2: uniform_dist(0.01, 3.0),  # alpha
                3: uniform_dist(1e-5, 1e-2),  # f_1, exponential scale freq [Hz]
                4: uniform_dist(1e-5, 1e-2),  # f_2, tanh transition width [Hz]
            }

            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"galfor": ProbDistContainer(priors_galfor)}

        if self.betas is None:
            # the galfor move's own ladder, sized by the ntemps knob
            # (GALFOR_NTEMPS; an explicit betas array wins)
            betas = make_ladder(self.ndim * 10, Tmax=np.inf, ntemps=self.ntemps)
            self.betas = betas

        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

    def init_setup(self):
        """Run sampling-info and state-backend initialization for the foreground branch."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the foreground state/backend to the generic module sub-state/backend."""
        if self.branch_state is None:
            self.branch_state = ModuleSubState
        if self.branch_backend is None:
            self.branch_backend = ModuleSubBackend


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


# ============================================================
# Shared noise (PSD + galactic-foreground) branch setup
# ------------------------------------------------------------
# These module-level helpers are the single source of truth for the noise
# branch setup, used by both the ``noise_only``/``noise_sgwb`` variants and by
# ``all_sources`` (whose psd/galfor setup "matches the noise setup").
# ============================================================


def resolve_galfor_modulation(path):
    """``None`` (stationary) or a :class:`GalForTimeModulation` from ``path``."""
    from lisatools.sensitivity import GalForTimeModulation

    return GalForTimeModulation(path) if path else None


def resolve_noise_file(
    mojito_data_path: str, explicit: typing.Optional[str] = None
) -> typing.Optional[str]:
    """Resolve the mojito NOISE brick path for a run.

    An ``explicit`` path (``general.noise_file`` / env ``NOISE_FILE``) wins and
    must exist; otherwise look for the standard
    ``<mojito_data_path>/data/INSTRUMENT/L1/NOISE_*`` brick. Returns ``None``
    when nothing is found (callers fall back to the analytic stock levels).
    """
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(
                f"noise_file={explicit!r} does not exist; unset it or point it "
                "at a mojito NOISE L1 .h5 file."
            )
        return explicit
    folder = os.path.join(mojito_data_path, "data", "INSTRUMENT", "L1")
    if not os.path.isdir(folder):
        return None
    from ...preprocessing import find_file

    try:
        return find_file(folder, "NOISE", 0)
    except FileNotFoundError:
        return None


def noise_params_from_file(
    noise_file: str,
    band: typing.Optional[typing.Tuple[float, float]] = None,
    tdi_generation: int = 2,
) -> typing.Optional[typing.List[float]]:
    """``[Soms_d, Sa_a]`` fit to the NOISE brick's tabulated estimates.

    Wraps :func:`lisatools.sensitivity.estimate_noise_params_from_file`; the
    fit band is clipped to the tabulated grid. Returns ``None`` (with a
    warning) if the fit fails, so callers can fall back to the stock levels.
    """
    from lisatools.sensitivity import estimate_noise_params_from_file

    kwargs = {"tdi_generation": tdi_generation}
    if band is not None:
        kwargs["band"] = (float(band[0]), float(band[1]))
    try:
        soms_d, sa_a = estimate_noise_params_from_file(noise_file, **kwargs)
    except Exception as exc:  # fit problems must not kill a build
        logger.warning(
            "noise-parameter fit against %s failed (%s); falling back to the "
            "stock analytic levels.", noise_file, exc,
        )
        return None
    logger.info(
        "noise parameters read from %s: Soms_d=%.6e, Sa_a=%.6e",
        noise_file, soms_d, sa_a,
    )
    return [soms_d, sa_a]


def prepare_psd_branch(psd, psd_injection=None):
    """Fill the 2-param instrument PSD prior + (optional) injection.

    The injection ``[Soms_d, Sa_a]`` sits inside the sampled prior so the fit
    recovers it. Shared by the noise variants and all_sources.
    """
    if psd.initialize_kwargs is None:
        psd.initialize_kwargs = dict()
    if psd.priors is None:
        psd.priors = {
            "psd": ProbDistContainer(
                {
                    r"$S_{\rm oms}$": uniform_dist(6.0e-12, 20.0e-11),  # Soms_d
                    r"$S_{\rm tm}$": uniform_dist(1.0e-15, 20.0e-14),  # Sa_a
                }
            )
        }
    if psd.injection is None and psd_injection is not None:
        psd.injection = np.asarray(psd_injection, dtype=float)
    return psd


def prepare_galfor_branch(galfor):
    """Galactic-foreground branch prep (prior comes from ``GalForSetup``)."""
    if galfor.initialize_kwargs is None:
        galfor.initialize_kwargs = {}
    return galfor


def noise_sensitivity_init_kwargs(
    base, *, tdi_generation, galfor=None, psd=None, galfor_modulation_path=None, extra=None
):
    """Thread the per-branch noise-MODEL choice onto ``sensitivity_init_kwargs``.

    A user swaps the noise model via the branch Settings the same way a source
    branch swaps its waveform: ``fit.galfor.stochastic_fn`` / ``.modulation`` /
    ``fit.psd.instrument_model_cls`` — read off ``galfor``/``psd`` here and
    forwarded to :class:`CompositeSensitivityBackend`. ``extra`` (e.g. the SGWB
    template) is merged last. Shared by the noise variants and all_sources.
    """
    out = dict(base or {})
    out.setdefault("tdi_generation", tdi_generation)
    if galfor is not None and getattr(galfor, "stochastic_fn", None) is not None:
        out["galfor_stochastic_fn"] = galfor.stochastic_fn
    branch_mod = getattr(galfor, "modulation", None) if galfor is not None else None
    out["galfor_modulation"] = (
        branch_mod if branch_mod is not None else resolve_galfor_modulation(galfor_modulation_path)
    )
    if psd is not None:
        for attr in ("instrument_component_cls", "instrument_model_cls", "model_name"):
            val = getattr(psd, attr, None)
            if val is not None:
                out[attr] = val
    if extra:
        out.update(extra)
    return out
