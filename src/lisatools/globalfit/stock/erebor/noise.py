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
from ..base import env_default, ten_to_the_x

logger = logging.getLogger(__name__)

# Physical (linear) support of the 2-parameter instrument-noise prior,
# ``(Soms_d, Sa_a)`` in sqrt units. The log-sampled branch covers exactly the
# same range -- uniform in ln instead of uniform in S.
PSD_PRIOR_RANGE = ((6.0e-12, 20.0e-11), (1.0e-15, 20.0e-14))


def psd_prior_dict(log_sampling: bool = False) -> dict:
    """``{label: uniform_dist}`` for the 2-param instrument PSD branch.

    ``log_sampling`` switches the sampling basis to ``ln(Soms_d), ln(Sa_a)``
    over the same physical support (:data:`PSD_PRIOR_RANGE`); pair it with
    :func:`make_psd_log_transform_container` so the likelihood still receives
    linear levels.
    """
    (oms_lo, oms_hi), (tm_lo, tm_hi) = PSD_PRIOR_RANGE
    if not log_sampling:
        return {
            r"$S_{\rm oms}$": uniform_dist(oms_lo, oms_hi),  # Soms_d
            r"$S_{\rm tm}$": uniform_dist(tm_lo, tm_hi),  # Sa_a
        }
    return {
        r"$\ln S_{\rm oms}$": uniform_dist(np.log(oms_lo), np.log(oms_hi)),
        r"$\ln S_{\rm tm}$": uniform_dist(np.log(tm_lo), np.log(tm_hi)),
    }


def check_psd_log_sampling(psd) -> bool:
    """``psd.log_sampling``, refused for anything but the 2-param branch.

    The log basis is defined for ``(Soms_d, Sa_a)`` only; a spline-style psd
    branch would silently exponentiate its knot positions/amplitudes.
    """
    log_sampling = bool(getattr(psd, "log_sampling", False))
    if log_sampling and getattr(psd, "ndim", None) not in (None, 2):
        raise ValueError(
            "psd.log_sampling is only defined for the 2-parameter "
            f"(Soms_d, Sa_a) branch, but ndim={psd.ndim}. Unset it "
            "(PSD_LOG_SAMPLING=0) or supply your own prior + transform."
        )
    return log_sampling


def make_psd_log_transform_container() -> TransformContainer:
    """``(ln Soms_d, ln Sa_a) -> (Soms_d, Sa_a)`` for the log-sampled psd branch.

    Same ``np.exp`` / ``np.log`` pair (and the same picklable-by-reference
    ufuncs) the EMRI/SOBBH ``logm1`` bases use. Lives here rather than in
    ``transforms.py`` so the noise path keeps its import surface -- the
    transform module pulls in ``bbhx`` at import.
    """
    return TransformContainer(
        input_basis=["logSoms_d", "logSa_a"],
        output_basis=["Soms_d", "Sa_a"],
        parameter_transforms={"Soms_d": np.exp, "Sa_a": np.exp},
        inverse_parameter_transforms={"Soms_d": np.log, "Sa_a": np.log},
        key_map={"logSoms_d": "Soms_d", "logSa_a": "Sa_a"},
    )


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
    # Extra constructor arguments for ``instrument_component_cls``. Needed by
    # models carrying more than the two levels — notably
    # ``UnequalArmInstrumentNoise``, which takes ``ltts=`` (a plain (6,) or
    # (Nt, 6) numpy array of per-link light travel times). Keep it plain data:
    # this dataclass is deepcopied and pickled with the rest of the settings
    # tree, so an orbits object must not be stored here.
    instrument_component_kwargs: Optional[dict] = None
    model_name: Optional[str] = None
    # Sample ``ln(Soms_d), ln(Sa_a)`` instead of the linear levels (2-param
    # branch only). The branch then carries a uniform-in-ln prior over the
    # SAME physical support plus an ``exp`` transform, so every consumer
    # (PSDMove, the sensitivity backend) still sees linear levels -- the two
    # levels span decades, so a linear-uniform prior/proposal wastes most of
    # its steps at the top of the range. Knob: ``PSD_LOG_SAMPLING``.
    log_sampling: bool = dataclasses.field(
        default_factory=env_default("PSD_LOG_SAMPLING", False, bool)
    )

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

        log_sampling = check_psd_log_sampling(self)

        if self.priors is None:
            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"psd": ProbDistContainer(psd_prior_dict(log_sampling))}

        else:
            self.logger.info("Using custom priors for PSD branch")

        if log_sampling and self.transform is None:
            self.transform = make_psd_log_transform_container()

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


# Physical (linear) support of the 5-parameter galactic-foreground prior, in
# the model's own order ``(amp, fk, alpha, f_1, f_2)`` -- the argument order of
# ``HyperbolicTangentGalacticForeground.specific_Sh_function``.
#
# ``f_1``/``f_2`` are the exponential roll-off and tanh-transition frequency
# scales, in **Hz** -- NOT the "Slope1"/"Slope2" numbers tabulated on
# ``FittedHyperbolicTangentGalacticForeground``. Those are converted before
# use (``stochastic.py``: ``F1 = slope1 ** (-1/alpha)``, ``F2 = 1/slope2``),
# which for the 4-yr entry lands at f_1 ~ 1.15e-3 Hz and f_2 ~ 3.38e-4 Hz --
# both mHz-band, like every other frequency in this model. The pre-2026-08
# ranges (f_1 in 1..1e7, f_2 in 50..8000) were the slope-unit numbers and
# EXCLUDED the physical values, so a foreground fit could not place the knee
# in band: f_1 railed at its floor, and alpha -> 0 plus a shrinking amp
# compensated by flattening ``(f/f_1)**alpha`` into a constant. The floors
# below reach 1e-5 Hz for both.
#
# The CEILINGS are 1e-2 (fk, f_1) / 1e-2 (f_2), an order above the top of the
# analysis band, and they are load-bearing: above ~1e-2 Hz both shape factors
# go degenerate across the band -- ``exp(-(f/f_1)**alpha) -> 1`` and
# ``0.5(1 + tanh(-(f - fk)/f_2)) -> 0.5`` -- so every decade past that is the
# SAME model, the likelihood is flat over it, and alpha stops being identified
# at all. They used to be the old slope-unit numbers (f_1 to 1e7, f_2 to 1e4),
# kept for configurations still carrying slope-unit values, and that margin
# cost a two-year run its posterior: under the uniform-in-log prior the
# plateau was ~9 of f_1's 12 decades, i.e. where 3 of every 4 walkers start
# (``run.py`` initializes from ``priors[key].rvs``). Measured on the two-year
# foreground fit (noise-galfor-pe2 try3, 2026-08-11): the chain settled at
# ``f_1 ~ 7e2 Hz`` (5-95%: 2e-2 .. 3e6) with the roll-off switched off and
# ``amp``/``f_2`` at ~0.45x compensating inside a ~1%-wide local optimum, for
# a whitened ``<w^2/C>`` of 0.983 (z = -11 to -17 per channel) and a 15%-high
# ``Sa_a`` soaking up the residual; the same data at an in-band
# ``f_1 = 2.5e-3 Hz`` whitens to 1.0002 (z = +0.1). Escaping the plateau needs
# a simultaneous move in (amp, f_1, f_2), which a stretch proposal essentially
# never makes -- so the range, not the sampler, is what has to give.
#
# f_2's ceiling is 1e-2 rather than 1e-3: the Fourier-domain least-squares fit
# to this brick (cd1l-validation foreground_estimation.ipynb) lands at
# f_2 = 9.89e-4, which a 1e-3 ceiling would truncate at 99% of its range.
# amp centred on the GALFOR brick's measured level (2026-08). A
# least-squares fit of this model to the brick's own X-channel PSD
# (scripts/noise/fit_galfor.py) gives amp ~ 1.2e-44, i.e. log10 -43.9, within
# half a decade of the stock injection -- once ``X2TDISens.stochastic_transform``
# carries the TDI-2 factor sin^2(2x) (fixed in the same change; see
# sensitivity.py). Before that fix the foreground was ~1.6 decades too loud
# per unit amp, so the sampler drove amp down to ~1e-45 and railed at the
# floor. Three decades either side of the measured value.
GALFOR_PRIOR_RANGE = (
    (1e-47, 1e-41),  # amp
    (1e-5, 1e-2),  # fk (knee)
    (1e-3, 5.0),  # alpha
    (1e-5, 1e-2),  # f_1
    (1e-5, 1e-2),  # f_2
)
GALFOR_BASIS = ("amp", "fk", "alpha", "f_1", "f_2")
# Everything except the power-law index spans decades and is strictly
# positive, so those four are the ones a log basis helps.
GALFOR_LOG_PARAMS = ("amp", "fk", "f_1", "f_2")


def galfor_prior_dict(log_sampling: bool = False, *, alpha_max=None) -> dict:
    """``{index: uniform_dist}`` for the 5-param galactic-foreground branch.

    ``log_sampling`` switches ``amp, fk, f_1, f_2`` (:data:`GALFOR_LOG_PARAMS`)
    to ``log10`` over the same physical support (:data:`GALFOR_PRIOR_RANGE`);
    ``alpha`` is an O(1) power-law index and stays linear. Pair it with
    :func:`make_galfor_log_transform_container`.

    ``alpha_max`` (env ``GALFOR_ALPHA_MAX``; default the module range's 5.0)
    raises ONLY the alpha upper cap. Diagnostic (2026-09-04): in the 3-month
    v8 run alpha rails against 5.0 while the instrument PSD is biased ~1.4x --
    the foreground appears to want a steeper shape than the cap allows, so the
    mismatch may be leaking into the instrument PSD. Widening the cap lets the
    slope explore; nothing else in the prior changes, and the default leaves
    every other run bit-identical. Resuming a chain under the wider cap is
    safe -- railed alpha values stay inside the new support.

    **Base-10, not natural log** (2026-08): these four span 4-12 decades and
    every one of them is quoted in decades in the literature and in run logs,
    so a chain value of ``-43.5`` reads directly as ``10**-43.5``. The psd
    branch stays in ``ln`` (:func:`psd_prior_dict`) -- it carries two O(1)-
    spread levels where the basis is a wash, and flipping it would invalidate
    stored chains. Uniform-in-log10 and uniform-in-ln are the SAME measure up
    to the constant ``ln 10``, so the posterior is unchanged; only the stored
    numbers and the step scale differ.
    """
    if alpha_max is None:
        _env = os.environ.get("GALFOR_ALPHA_MAX")
        alpha_max = float(_env) if _env else None
    ranges = list(GALFOR_PRIOR_RANGE)
    if alpha_max is not None:
        ia = GALFOR_BASIS.index("alpha")
        lo_a, _hi_a = ranges[ia]
        ranges[ia] = (lo_a, float(alpha_max))
    priors = {}
    for i, (name, (lo, hi)) in enumerate(zip(GALFOR_BASIS, ranges)):
        if log_sampling and name in GALFOR_LOG_PARAMS:
            lo, hi = np.log10(lo), np.log10(hi)
        priors[i] = uniform_dist(lo, hi)
    return priors


def check_galfor_log_sampling(galfor) -> bool:
    """``galfor.log_sampling``, refused for anything but the 5-param branch."""
    log_sampling = bool(getattr(galfor, "log_sampling", False))
    if log_sampling and getattr(galfor, "ndim", None) not in (None, 5):
        raise ValueError(
            "galfor.log_sampling is only defined for the 5-parameter "
            f"(amp, fk, alpha, f_1, f_2) branch, but ndim={galfor.ndim}. "
            "Unset it (GALFOR_LOG_SAMPLING=0) or supply your own prior + "
            "transform."
        )
    return log_sampling


def make_galfor_log_transform_container() -> TransformContainer:
    """``(log10 amp, log10 fk, alpha, log10 f_1, log10 f_2) -> the linear five``.

    ``alpha`` passes through untouched (it is in both bases under its own
    name), so the output order stays exactly what the foreground model's
    ``specific_Sh_function`` takes.

    ``ten_to_the_x`` is imported from ``stock.base`` rather than the sibling
    ``transforms`` module: both define it, but ``transforms`` imports ``bbhx``
    at module scope and the noise path keeps a narrower import surface. Both
    it and ``np.log10`` are module-level names, so the container pickles.
    """
    return TransformContainer(
        input_basis=[
            f"log10_{name}" if name in GALFOR_LOG_PARAMS else name for name in GALFOR_BASIS
        ],
        output_basis=list(GALFOR_BASIS),
        parameter_transforms={name: ten_to_the_x for name in GALFOR_LOG_PARAMS},
        inverse_parameter_transforms={name: np.log10 for name in GALFOR_LOG_PARAMS},
        key_map={f"log10_{name}": name for name in GALFOR_LOG_PARAMS},
    )


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
    # Sample log10(amp), log10(fk), log10(f_1), log10(f_2) -- alpha stays
    # linear (see GALFOR_LOG_PARAMS). Same deal as PSDSettings.log_sampling,
    # in base 10 rather than e: a uniform-in-log10 prior over the SAME
    # physical support plus a ``10**x`` transform, so the foreground model
    # still receives linear parameters. Each of the four spans 4-12 decades,
    # where linear proposals crawl. Knob: ``GALFOR_LOG_SAMPLING``.
    log_sampling: bool = dataclasses.field(
        default_factory=env_default("GALFOR_LOG_SAMPLING", False, bool)
    )


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

        log_sampling = check_galfor_log_sampling(self)

        if self.priors is None:
            # TODO: orbits check against sangria/sangria_hm
            self.priors = {"galfor": ProbDistContainer(galfor_prior_dict(log_sampling))}

        if log_sampling and self.transform is None:
            self.transform = make_galfor_log_transform_container()

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


def validate_coarse_settings(gs, *, all_source: bool) -> None:
    """Shared validation for the coarse real-WDM knobs (plan-2 §7).

    Noise-only rules are the historical ones (CPU backend replacement); an
    all-source run must opt into a sidecar mode explicitly, because coarse
    scoring there changes the PSD/GALFOR transition kernel, not just speed.
    """
    import numbers

    if (
        isinstance(gs.coarse_Q, bool)
        or not isinstance(gs.coarse_Q, numbers.Integral)
        or gs.coarse_Q < 1
    ):
        raise ValueError(f"coarse_Q must be an integer >= 1; got {gs.coarse_Q!r}.")
    gs.coarse_Q = int(gs.coarse_Q)
    if gs.coarse_fiducial not in ("injection", "initial"):
        raise ValueError(
            "coarse_fiducial must be 'injection' or 'initial'; got "
            f"{gs.coarse_fiducial!r}."
        )
    mode = str(getattr(gs, "coarse_gpu_mode", "off") or "off")
    if mode not in ("off", "auto", "search_approx", "delayed_acceptance"):
        raise ValueError(
            "coarse_gpu_mode must be 'off', 'auto', 'search_approx', or "
            f"'delayed_acceptance'; got {mode!r}."
        )
    gs.coarse_gpu_mode = mode
    if not all_source:
        if mode != "off":
            raise ValueError(
                "coarse_gpu_mode applies to all-source runs only; the "
                "noise-only variants use the CPU backend-replacement coarse "
                "path (COARSE_Q alone)."
            )
        if gs.coarse_Q > 1 and gs.gpus is not None:
            raise ValueError(
                "coarse_Q > 1 is CPU-only for now; unset gpus/use_gpu. "
                "Single-GPU support is a planned follow-up."
            )
        return
    if gs.coarse_Q > 1 and mode == "off":
        raise ValueError(
            "coarse_Q > 1 in an all-source run requires an explicit "
            "COARSE_GPU_MODE ('search_approx' for optimization stages, "
            "'delayed_acceptance' for production PE): the coarse statistic "
            "changes the PSD/GALFOR transition kernel, so it is never an "
            "implicit speed knob here."
        )
    if mode != "off" and gs.coarse_Q <= 1:
        raise ValueError(
            f"coarse_gpu_mode={mode!r} needs COARSE_Q > 1 (got "
            f"{gs.coarse_Q}); with Q=1 there is nothing coarse to score."
        )


def resolve_galfor_modulation(path, t0: float = 0.0):
    """``None`` (stationary) or a :class:`GalForTimeModulation` from ``path``.

    ``t0`` is the absolute epoch the table's time column is written against
    (default 0.0 keeps a table already relative to the data start unchanged).
    Construction is lazy, so a deferred anchor may also overwrite ``.t0``
    before first use — see ``GeneralSetup._resolve_deferred_noise_model``.
    """
    from lisatools.sensitivity import GalForTimeModulation

    return GalForTimeModulation(path, t0=t0) if path else None


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


def wire_unequal_arm_psd(gs, psd) -> None:
    """UNEQUAL_ARM=1: unequal-arm instrument noise fed by the brick's ``/ltts``.

    Swaps the psd branch's equal-arm ``InstrumentNoise`` for
    :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`, whose six link
    light-travel times are read from the mojito NOISE brick's ``/ltts`` group
    and averaged per WDM time column (``LinkDelayTable``). Only plain
    scalars/paths go on the settings tree here; the
    :class:`~lisatools.sensitivity.LinkDelayTable` itself is built by
    ``GeneralSetup._resolve_deferred_noise_model`` once the data epoch
    (``data_t0``) is authoritative. Mojito data only: synthetic/sangria carry
    no delay table and are refused loudly.

    Shared by ``all_sources`` and the noise-only variants so the two paths stay
    in lockstep; reads ``gs.unequal_arm_stride`` / ``gs.wdm_psd_method`` /
    ``gs.mojito_data_path`` / ``gs.noise_file`` and mutates ``psd`` in place.
    """
    import h5py

    from lisatools.sensitivity import UnequalArmInstrumentNoise

    if psd is None:
        raise ValueError(
            "unequal_arm=1 requires the psd branch (it swaps the psd "
            "branch's instrument component)."
        )
    if gs.data_mode != "mojito":
        raise ValueError(
            f"unequal_arm=1 requires data_mode='mojito' (got "
            f"{gs.data_mode!r}): the per-link delay table is read from "
            "the mojito NOISE brick's /ltts group, and synthetic/sangria "
            "data carries none."
        )
    noise_file = resolve_noise_file(gs.mojito_data_path, gs.noise_file)
    if noise_file is None:
        raise FileNotFoundError(
            "unequal_arm=1 but no mojito NOISE brick was found: set "
            "general.noise_file / NOISE_FILE or add "
            f"data/INSTRUMENT/L1/NOISE_* under {gs.mojito_data_path!r}."
        )
    with h5py.File(noise_file, "r") as fh:
        if "ltts" not in fh:
            raise ValueError(
                f"unequal_arm=1 but {noise_file!r} has no /ltts group; "
                "use a NOISE brick that carries the per-link delays."
            )
    if psd.instrument_component_cls is None:
        psd.instrument_component_cls = UnequalArmInstrumentNoise
    kwargs = dict(psd.instrument_component_kwargs or {})
    kwargs.setdefault("ltts_l1_file", noise_file)
    kwargs.setdefault("ltts_stride", int(gs.unequal_arm_stride))
    if gs.wdm_psd_method:
        kwargs.setdefault("wdm_psd_method", gs.wdm_psd_method)
    psd.instrument_component_kwargs = kwargs


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
    """Fill the 2-param instrument PSD prior, transform, and (optional) injection.

    The injection ``[Soms_d, Sa_a]`` is given in LINEAR units and sits inside
    the sampled prior so the fit recovers it. With ``psd.log_sampling``
    (``PSD_LOG_SAMPLING``) the branch samples ``ln`` of the two levels: same
    physical support, uniform in ln, an ``exp``
    :class:`~eryn.utils.TransformContainer` back to linear for every consumer,
    and the injection carried into the sampling basis so injection-truth
    overlays stay aligned with the chain. Shared by the noise variants and
    all_sources.
    """
    log_sampling = check_psd_log_sampling(psd)
    if psd.initialize_kwargs is None:
        psd.initialize_kwargs = dict()
    if psd.priors is None:
        psd.priors = {"psd": ProbDistContainer(psd_prior_dict(log_sampling))}
    if log_sampling and psd.transform is None:
        psd.transform = make_psd_log_transform_container()
    if psd.injection is None and psd_injection is not None:
        injection = np.asarray(psd_injection, dtype=float)
        psd.injection = np.log(injection) if log_sampling else injection
    return psd


def prepare_galfor_branch(galfor):
    """Galactic-foreground branch prep: prior + transform for the sampling basis.

    With ``galfor.log_sampling`` (``GALFOR_LOG_SAMPLING``) the branch samples
    ``log10`` of ``amp, fk, f_1, f_2`` (alpha stays linear) over the same
    physical support, with a ``10**x``
    :class:`~eryn.utils.TransformContainer` back to the foreground model's own
    basis. Note the base: the psd branch is ``ln``, this one is ``log10``. Filling them here rather than leaving it to
    :class:`GalForSetup` keeps the knob effective for callers that pass their
    own Setup. Shared by the noise variants and all_sources.
    """
    log_sampling = check_galfor_log_sampling(galfor)
    if galfor.initialize_kwargs is None:
        galfor.initialize_kwargs = {}
    if galfor.priors is None:
        galfor.priors = {"galfor": ProbDistContainer(galfor_prior_dict(log_sampling))}
    if log_sampling and galfor.transform is None:
        galfor.transform = make_galfor_log_transform_container()
    return galfor


def noise_sensitivity_init_kwargs(
    base,
    *,
    tdi_generation,
    galfor=None,
    psd=None,
    galfor_modulation_path=None,
    galfor_modulation_t0=0.0,
    extra=None,
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
        branch_mod
        if branch_mod is not None
        else resolve_galfor_modulation(galfor_modulation_path, t0=galfor_modulation_t0)
    )
    if psd is not None:
        for attr in (
            "instrument_component_cls",
            "instrument_model_cls",
            "instrument_component_kwargs",
            "model_name",
        ):
            val = getattr(psd, attr, None)
            if val is not None:
                out[attr] = val
        # ``wdm_psd_method`` used to live only inside the unequal-arm
        # instrument constructor kwargs. Promote that legacy spelling to the
        # backend-wide policy so foreground and SGWB cannot silently remain on
        # the exact fold while the instrument uses a layer approximation.
        component_kwargs = getattr(psd, "instrument_component_kwargs", None)
        if component_kwargs and "wdm_psd_method" in component_kwargs:
            out.setdefault(
                "wdm_psd_method", component_kwargs["wdm_psd_method"]
            )
    if extra:
        out.update(extra)
    return out
