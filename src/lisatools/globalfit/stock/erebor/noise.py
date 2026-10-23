"""Noise branch blocks (instrumental PSD + galactic foreground) for the Erebor recipe.

The two noise branches of every noise-fitting Erebor variant:

* ``psd`` — :class:`PSDSettings` / :class:`PSDSetup`: the 2-parameter
  instrument model ``(Soms_d, Sa_a)`` (sqrt units, prior
  :data:`PSD_PRIOR_RANGE`, linear or ``ln`` basis via ``PSD_LOG_SAMPLING``).
* ``galfor`` — :class:`GalForSettings` / :class:`GalForSetup`: the
  5-parameter galactic-confusion foreground ``(amp, fk, alpha, f_1, f_2)``
  (:data:`GALFOR_BASIS`, prior :func:`galfor_prior_ranges`, linear or
  ``log10`` basis via ``GALFOR_LOG_SAMPLING``).

Each branch runs its own :class:`~lisatools.globalfit.moves.psdmove.PSDMove`
(stock names ``psd_pe`` / ``galfor_pe`` and ``*_search``) on its own ladder.
The shared helpers here prepare both branches for the noise variants and for
``all_sources`` (:func:`prepare_psd_branch`, :func:`prepare_galfor_branch`,
:func:`noise_sensitivity_init_kwargs`, :func:`wire_unequal_arm_psd`,
:func:`validate_coarse_settings`), and the store / monitor readers use
:func:`read_noise_model_identity`, :func:`galfor_params_to_physical` and
:func:`psd_truth_levels`.

Production 9mo / 12mo (``all_sources``): ``UNEQUAL_ARM=1`` (unequal-arm
instrument noise fed by the NOISE brick's link delays), ``PSD_LOG_SAMPLING``
unset (linear psd basis), ``GALFOR_LOG_SAMPLING=1``,
``GALFOR_ALPHA_MAX=20``, ``GALFOR_FK_PRIOR`` / ``GALFOR_F1_PRIOR`` /
``GALFOR_F2_PRIOR`` set, ``GALFOR_MODULATION_PATH`` (a tabulated foreground
time modulation, ``modulation_unequal.dat``) with ``GALFOR_MODULATION_T0=data``,
``PSD_NUM_PROP_REPEATS`` / ``GALFOR_NUM_PROP_REPEATS`` = 10. The noise is
PINNED at launch (``PSD_START_PARAMS`` from ``warmstart.noise_pin`` and a
hand-set ``GALFOR_START_PARAMS``), so the noise stages are skipped; ``psd_pe``
/ ``galfor_pe`` sample in ``full_pe`` and inside the ``gb_search_2``
cycle-end noise search.
"""

from __future__ import annotations

import dataclasses
import json
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

    Args:
        log_sampling: Uniform in ``ln`` instead of linear. Default ``False``.

    Returns:
        dict: Two :func:`~eryn.prior.uniform_dist` entries keyed by LaTeX
        labels (``Soms_d`` first, ``Sa_a`` second), ready for
        :class:`~eryn.prior.ProbDistContainer`.
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

    Args:
        psd (Settings): The psd block (``log_sampling``, ``ndim`` read).

    Returns:
        bool: The effective ``log_sampling`` flag.

    Raises:
        ValueError: ``log_sampling`` with ``ndim`` other than ``None`` / 2.
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

    Returns:
        TransformContainer: Input basis ``logSoms_d, logSa_a``, output basis
        ``Soms_d, Sa_a``.
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

    The stock prior (:func:`psd_prior_dict`) is the 2-parameter
    ``(Soms_d, Sa_a)`` model, so the variants use the ``ndim=2`` subclass
    ``NoisePSDSettings`` (``variants/noise.py``; ``all_sources`` aliases it as
    ``AllSourcesPSDSettings``). Reached as ``fit.psd``; env knobs ``PSD_<FIELD>``.
    ``PSD_EIGEN_INFO`` (launchers ``fisher``) is read by the move, not here.

    Attributes:
        ntemps: The psd move's own ladder size. Env ``PSD_NTEMPS``. Default 12.
        psd_kwargs: Reserved; the stock path does not read it (the ``None``
            default branch of :meth:`PSDSetup.init_sampling_info` never fires
            because the field defaults to ``{}``).
        nleaves_max: Always 1.
        nleaves_min: Always 1.
        ndim: Sampled dimension. Default 4 here, which does NOT match the
            2-parameter stock prior; ``NoisePSDSettings`` sets 2.
        transform: Sampling -> linear transform; filled with
            :func:`make_psd_log_transform_container` under ``log_sampling``.
        injection: Start/truth row in the SAMPLING basis (``prepare_psd_branch``
            converts a linear ``[Soms_d, Sa_a]`` injection).
        nknots: Not read anywhere (spline-PSD placeholder).
        num_prop_repeats: Inner proposals per PSDMove call. Env
            ``PSD_NUM_PROP_REPEATS``. Default 50; launchers 10.
        instrument_component_cls: Instrument-noise component class for the
            :class:`~lisatools.sensitivity.CompositeSensitivityBackend`;
            ``None`` -> the backend default (``InstrumentNoise``). Production
            (``UNEQUAL_ARM=1``) swaps in
            :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`.
        instrument_model_cls: Instrument model class; ``None`` -> the backend
            default (``LISAModel``).
        instrument_component_kwargs: Extra constructor kwargs of the component
            (plain data only: ``ltts_l1_file``, ``ltts_stride``,
            ``wdm_psd_method`` for the unequal-arm model).
        model_name: Instrument model name; ``None`` -> the backend default.
        log_sampling: Sample ``ln(Soms_d), ln(Sa_a)`` over the same support.
            Env ``PSD_LOG_SAMPLING``. Default ``False`` (launchers unset).
        inner_move_kind: PSDMove inner proposal, ``"eigen"`` or ``"stretch"``;
            ``None`` resolves from the walker count. Env
            ``PSD_INNER_MOVE_KIND``.
        eigen_refresh_every: Proposes between eigen-table refreshes. Env
            ``PSD_EIGEN_REFRESH``. Default 10.
        eigen_eps_rel: Information-matrix finite-difference step (fraction
            of each prior-box width). Env ``PSD_EIGEN_EPS_REL``. Default 1e-4.
        ensemble_search: Tiled per-walker inner ensemble in SEARCH stages
            only. Env ``PSD_ENSEMBLE_SEARCH``. Default ``False``.
        ensemble_repeats: Tiled copies per walker. Env
            ``PSD_ENSEMBLE_REPEATS``. Default 10.
        ensemble_spread_lo: Target lower lnL spread of the tiled ensemble.
            Env ``PSD_ENSEMBLE_SPREAD_LO``. Default 1.0.
        ensemble_spread_hi: Target upper lnL spread. Env
            ``PSD_ENSEMBLE_SPREAD_HI``. Default 10.0.
        ensemble_scale0: Opening perturbation scale (fraction of the prior
            box). Env ``PSD_ENSEMBLE_SCALE0``. Default 1e-3.
        ensemble_scale_tries: Rescale attempts. Env
            ``PSD_ENSEMBLE_SCALE_TRIES``. Default 4.
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
    # TODO check: keep/remove/document -- field has no reader anywhere (spline-PSD placeholder)
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
    # Inner proposal of the PSDMove sampling this block: `eigen` (per-rung
    # info-matrix axes; the one-walker default) or `stretch` (needs >= 2
    # walkers). See PSDMove._resolve_inner_kind.
    inner_move_kind: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("PSD_INNER_MOVE_KIND", None, str)
    )
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("PSD_EIGEN_REFRESH", 10, int)
    )
    eigen_eps_rel: float = dataclasses.field(
        default_factory=env_default("PSD_EIGEN_EPS_REL", 1e-4, float)
    )
    # --- tiled per-walker ensemble search (SEARCH stages only) ---------------
    # See the matching block on GalForSettings for the rationale. 10 repeats
    # clears eryn's red/blue floor of 2*ndim for the psd branch's 4 columns
    # with room to spare.
    ensemble_search: bool = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_SEARCH", False, bool)
    )
    ensemble_repeats: int = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_REPEATS", 10, int)
    )
    ensemble_spread_lo: float = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_SPREAD_LO", 1.0, float)
    )
    ensemble_spread_hi: float = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_SPREAD_HI", 10.0, float)
    )
    ensemble_scale0: float = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_SCALE0", 1e-3, float)
    )
    ensemble_scale_tries: int = dataclasses.field(
        default_factory=env_default("PSD_ENSEMBLE_SCALE_TRIES", 4, int)
    )


class PSDSetup(Setup):
    """:class:`Setup` for the instrumental PSD branch in the Erebor recipe.

    Construction copies the settings, opens the ``psd_setup.log`` logger and
    runs :meth:`init_setup` (prior / transform / ladder, then the generic
    module sub-state / sub-backend).

    Args:
        psd_settings: Settings dataclass with PSD kwargs, prior, and
            tempering configuration.
    """

    def __init__(self, psd_settings: PSDSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(psd_settings)

        level = logging.DEBUG
        name = "PSDSetup"
        self.logger = init_logger(filename="psd_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

        self.init_setup()

    def init_sampling_info(self):
        """Build the PSD prior, tempering ladder, and default ``psd_kwargs``.

        Fills (only when unset): ``initialize_kwargs``, ``priors``
        (:func:`psd_prior_dict` in the basis :func:`check_psd_log_sampling`
        allows), ``transform`` (log basis only), ``betas``
        (``make_ladder(10 * ndim, Tmax=inf, ntemps=ntemps)``) and
        ``other_tempering_kwargs`` (``permute=False``).

        Raises:
            ValueError: From :func:`check_psd_log_sampling`.
            AssertionError: When ``other_tempering_kwargs["permute"]`` is true.
        """
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
#
# THE KNEE FLOOR IS 0.8 mHz (user ruling 2026-09-24: "set the knee frequency
# galfor prior bounds from 0.8 mHz to 10 mHz"). ``fk`` alone among the three
# frequency parameters keeps a 1e-5 floor no longer -- 0.01 mHz is three
# decades below anything the confusion foreground can physically knee at,
# and the whole lower half of that range is unphysical support the sampler
# still had to start in and walk out of. The ceiling is unchanged: 10 mHz
# IS the 1e-2 the paragraph above argues for, so this is a floor change
# only.
#
# ⚠ WHY A FLOOR MATTERS HERE and not just as tidiness. ``run.py`` starts
# galfor from ``priors["galfor"].rvs``, and the basis is uniform-in-log10
# under GALFOR_LOG_SAMPLING=1 -- so with a 1e-5 floor, 1.9 of the 3 decades
# were below 0.8 mHz: 63% of walkers started with the knee under the whole
# analysis band and ``0.5(1 + tanh(-(f - fk)/f_2)) -> 0`` across it. That is
# the same class of failure the f_1 ceiling caused on the two-year run,
# approached from the other end: a large fraction of the prior mass sits
# where the model has no shape left, and a stretch proposal does not escape
# it. The new floor is 1.1 decades wide against the ceiling, all of it in
# band.
#
# ⚠ RESUME HAZARD, and the one thing to check before reusing a store: a
# chain already sitting below 0.8 mHz is OUTSIDE the new support, so on the
# next launch its walkers price at ``log_prior = -inf``. Nothing in the
# noise-model identity covers prior RANGES, so this will not be refused for
# you -- grep a candidate store's galfor fk column before resuming it, or
# start that branch fresh. The same applies to a noise PIN taken from such a
# store, which ``warmstart.noise_pin`` DOES refuse, by name and with the
# reason.
GALFOR_PRIOR_RANGE = (
    (1e-47, 1e-41),  # amp
    (0.8e-3, 1e-2),  # fk (knee): 0.8 mHz .. 10 mHz, user ruling 2026-09-24
    (1e-3, 5.0),  # alpha
    (1e-5, 1e-2),  # f_1
    (1e-5, 1e-2),  # f_2
)
GALFOR_BASIS = ("amp", "fk", "alpha", "f_1", "f_2")
# Everything except the power-law index spans decades and is strictly
# positive, so those four are the ones a log basis helps.
GALFOR_LOG_PARAMS = ("amp", "fk", "f_1", "f_2")
#: The three FREQUENCY columns (Hz) that ``GALFOR_FREQ_PRIOR`` moves together.
GALFOR_FREQ_PARAMS = ("fk", "f_1", "f_2")


#: Per-column overrides of the frequency box: ``GALFOR_FK_PRIOR``,
#: ``GALFOR_F1_PRIOR``, ``GALFOR_F2_PRIOR`` (each ``"lo,hi"`` in Hz); they win
#: over the shared ``GALFOR_FREQ_PRIOR`` for their column.
GALFOR_FREQ_PRIOR_ENV = {"fk": "GALFOR_FK_PRIOR", "f_1": "GALFOR_F1_PRIOR",
                         "f_2": "GALFOR_F2_PRIOR"}


def _range_from_env(name: str):
    """``<name>="lo,hi"`` (Hz) -> ``(lo, hi)``, or ``None`` when unset."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"{name}={raw!r}: expected 'lo,hi' in Hz (two comma-separated numbers).")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r}: not two numbers ({exc}).") from None


def _freq_range_from_env():
    """``GALFOR_FREQ_PRIOR="lo,hi"`` (Hz) -> ``(lo, hi)``, or ``None`` when unset."""
    return _range_from_env("GALFOR_FREQ_PRIOR")


def _checked_range(label: str, rng) -> tuple:
    """``(lo, hi)`` as floats, refusing anything but ``0 < lo < hi`` (ValueError)."""
    lo_f, hi_f = (float(rng[0]), float(rng[1]))
    if not (0.0 < lo_f < hi_f):
        raise ValueError(
            f"galfor {label} prior ({lo_f:g}, {hi_f:g}) Hz must satisfy 0 < lo < hi.")
    return (lo_f, hi_f)


def galfor_prior_ranges(*, alpha_max=None, freq_range=None, fk_range=None,
                        f1_range=None, f2_range=None) -> tuple:
    """The PHYSICAL (linear) support of the 5 galfor columns, knobs applied.

    THE ONE RESOLVER. :func:`galfor_prior_dict` builds the prior from it, and
    the two places that need the support without the prior object -- the
    ``{PSD,GALFOR}_START_PARAMS`` window in ``run.py`` and the
    ``warmstart.noise_pin`` refusal -- read it from here, so a knob can never
    move the prior without moving the checks (a pin inside the new prior would
    otherwise be refused by the old box).

    ``alpha_max`` (env ``GALFOR_ALPHA_MAX``) raises ONLY the alpha upper cap.
    ``freq_range`` (env ``GALFOR_FREQ_PRIOR="lo,hi"`` in Hz) moves the THREE
    frequency columns :data:`GALFOR_FREQ_PARAMS` (fk, f_1, f_2) to one shared
    box -- user ruling 2026-10-08 for the 9mo / 1yr runs: "(fk, f1, f2) ...
    from 1e-4 to 1e-2" (the stock box is fk 0.8-10 mHz, f_1 / f_2 10 uHz-10
    mHz). ``fk_range`` / ``f1_range`` / ``f2_range`` (env ``GALFOR_FK_PRIOR`` /
    ``GALFOR_F1_PRIOR`` / ``GALFOR_F2_PRIOR``) then override ONE column each
    -- user ruling 2026-10-09: "adjust the foreground prior for f_k and f_2 to
    go from 0.8 mHz to 10 mHz. Keep f_1 as is." Unset, every knob leaves the
    bound bit-identical. A kwarg beats its env.

    ⚠ RESUME HAZARD, as for the stock box: a chain whose fk / f_1 / f_2 sits
    outside the new support prices at ``log_prior = -inf`` on the next launch,
    and nothing in the noise-model identity covers prior ranges. The 9mo / 1yr
    stores start fresh; the 6mo relaunch keeps the stock box.

    Production launchers: ``GALFOR_ALPHA_MAX=20``,
    ``GALFOR_FK_PRIOR=0.8e-3,1e-2``, ``GALFOR_F1_PRIOR=1e-4,1e-2``,
    ``GALFOR_F2_PRIOR=0.8e-3,1e-2`` (``GALFOR_FREQ_PRIOR`` unset).

    Args:
        alpha_max (float, optional): Upper cap of ``alpha``; ``None`` -> env
            ``GALFOR_ALPHA_MAX`` -> the stock 5.0.
        freq_range (tuple, optional): Shared ``(lo, hi)`` (Hz) box for fk,
            f_1 and f_2; ``None`` -> env ``GALFOR_FREQ_PRIOR``.
        fk_range (tuple, optional): ``(lo, hi)`` (Hz) for fk alone; ``None``
            -> env ``GALFOR_FK_PRIOR``.
        f1_range (tuple, optional): Same for f_1 (``GALFOR_F1_PRIOR``).
        f2_range (tuple, optional): Same for f_2 (``GALFOR_F2_PRIOR``).

    Returns:
        tuple: Five ``(lo, hi)`` float pairs in :data:`GALFOR_BASIS` order
        (linear, physical units: ``amp`` dimensionless, frequencies Hz).

    Raises:
        ValueError: A malformed ``"lo,hi"`` env value or a box violating
            ``0 < lo < hi``.
    """
    if alpha_max is None:
        _env = os.environ.get("GALFOR_ALPHA_MAX")
        alpha_max = float(_env) if _env else None
    if freq_range is None:
        freq_range = _freq_range_from_env()
    ranges = [tuple(map(float, r)) for r in GALFOR_PRIOR_RANGE]
    if alpha_max is not None:
        ia = GALFOR_BASIS.index("alpha")
        lo_a, _hi_a = ranges[ia]
        ranges[ia] = (lo_a, float(alpha_max))
    if freq_range is not None:
        box = _checked_range("frequency", freq_range)
        for name in GALFOR_FREQ_PARAMS:
            ranges[GALFOR_BASIS.index(name)] = box
    per_column = {"fk": fk_range, "f_1": f1_range, "f_2": f2_range}
    for name, rng in per_column.items():
        if rng is None:
            rng = _range_from_env(GALFOR_FREQ_PRIOR_ENV[name])
        if rng is not None:
            ranges[GALFOR_BASIS.index(name)] = _checked_range(name, rng)
    return tuple(ranges)


def galfor_prior_dict(log_sampling: bool = False, *, alpha_max=None,
                      freq_range=None, fk_range=None, f1_range=None,
                      f2_range=None) -> dict:
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

    Args:
        log_sampling: ``log10`` basis for :data:`GALFOR_LOG_PARAMS`. Default
            ``False``; production ``GALFOR_LOG_SAMPLING=1``.
        alpha_max (float, optional): See :func:`galfor_prior_ranges`.
        freq_range (tuple, optional): See :func:`galfor_prior_ranges`.
        fk_range (tuple, optional): See :func:`galfor_prior_ranges`.
        f1_range (tuple, optional): See :func:`galfor_prior_ranges`.
        f2_range (tuple, optional): See :func:`galfor_prior_ranges`.

    Returns:
        dict: ``{0..4: uniform_dist}`` in :data:`GALFOR_BASIS` order, in the
        sampling basis.
    """
    # ``freq_range`` (env ``GALFOR_FREQ_PRIOR``): the fk / f_1 / f_2 box, see
    # :func:`galfor_prior_ranges` (the one resolver, shared with the pin checks).
    ranges = galfor_prior_ranges(alpha_max=alpha_max, freq_range=freq_range,
                                 fk_range=fk_range, f1_range=f1_range,
                                 f2_range=f2_range)
    priors = {}
    for i, (name, (lo, hi)) in enumerate(zip(GALFOR_BASIS, ranges)):
        if log_sampling and name in GALFOR_LOG_PARAMS:
            lo, hi = np.log10(lo), np.log10(hi)
        priors[i] = uniform_dist(lo, hi)
    return priors


def read_noise_model_identity(h5_path) -> dict:
    """``noise_model_identity`` out of a global-fit store, as a plain dict.

    The record is written as a GROUP, ``global_fit/noise_model_identity``,
    carrying one ATTRIBUTE per key -- not as a JSON string attribute. Readers
    that assume the latter find nothing, fall back to their defaults and
    silently reinterpret the store (a linear-basis read of a log-sampled
    galfor chain, say). A JSON-attribute fallback is kept for any store that
    ever wrote it that way; an unreadable or absent record returns ``{}``,
    which callers must treat as "the defaults", never as "all False".

    Read by the HDF backend / run resume checks, the monitor rank and the
    LATW store-reading tutorial (keys include ``galfor_log_sampling``,
    ``unequal_arm``, ``wdm_psd_method``, ``data_t0``, ``coarse_Q``).

    Args:
        h5_path (str): Path of the global-fit HDF store.

    Returns:
        dict: ``{key: value}`` (numpy scalars converted with ``.item()``), or
        ``{}``.
    """
    import h5py

    out = {}
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get("global_fit/noise_model_identity")
            if grp is not None and hasattr(grp, "attrs"):
                for k, v in grp.attrs.items():
                    out[k] = v.item() if hasattr(v, "item") else v
            if not out:
                raw = f.attrs.get("noise_model_identity")
                if raw is None and "global_fit" in f:
                    raw = f["global_fit"].attrs.get("noise_model_identity")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                if isinstance(raw, str):
                    out = dict(json.loads(raw))
                elif isinstance(raw, dict):
                    out = dict(raw)
    except (OSError, ValueError, TypeError, KeyError):
        return {}
    return out


def galfor_params_to_physical(params, log_sampling: bool):
    """Stored galfor coordinates -> the LINEAR five the foreground model wants.

    Under ``log_sampling`` the four :data:`GALFOR_LOG_PARAMS` columns are
    stored as ``log10`` while ``alpha`` stays linear, so a consumer that
    hands the stored vector straight to
    :class:`~lisatools.stochastic.HyperbolicTangentGalacticForeground`
    passes a NEGATIVE ``amp`` and ``f_1`` and gets NaN out of
    ``(f / f_1) ** alpha`` -- with no exception at the call site.

    That is not hypothetical: it silently produced an all-zero SNR / all-False
    truth set in ``build_truth.py`` and a fatal ``KeyError: 'nbins'`` in
    ``gf_monitor_gen.py`` on the first stores written with
    ``GALFOR_LOG_SAMPLING=1`` (2026-09-18). Every reader of a stored galfor
    row must route through here, keyed off the store's own
    ``noise_model_identity["galfor_log_sampling"]`` -- never off an env var,
    which reflects the CURRENT process, not the basis the store was written in.

    Accepts any shape whose LAST axis is the 5-column basis; returns a float
    array of the same shape. ``log_sampling=False`` is an identity copy.

    Args:
        params (array_like): Stored galfor coordinates, shape ``(..., 5)``.
        log_sampling: The store's ``noise_model_identity["galfor_log_sampling"]``.

    Returns:
        np.ndarray: ``(..., 5)`` float64 copy in the linear
        ``(amp, fk, alpha, f_1, f_2)`` basis.

    Raises:
        ValueError: When the last axis is not 5 wide.

    Example:
        >>> import numpy as np
        >>> from lisatools.globalfit.stock.erebor.noise import galfor_params_to_physical
        >>> galfor_params_to_physical(np.array([-44.0, -3.0, 2.0, -3.0, -3.0]), True)[:2]
        array([1.e-44, 1.e-03])
    """
    out = np.array(params, dtype=float, copy=True)
    if out.shape[-1] != len(GALFOR_BASIS):
        raise ValueError(
            f"galfor parameters must have {len(GALFOR_BASIS)} columns "
            f"{GALFOR_BASIS} on the last axis; got shape {out.shape}"
        )
    if not log_sampling:
        return out
    idx = [GALFOR_BASIS.index(n) for n in GALFOR_LOG_PARAMS]
    out[..., idx] = 10.0 ** out[..., idx]
    return out


def check_galfor_log_sampling(galfor) -> bool:
    """``galfor.log_sampling``, refused for anything but the 5-param branch.

    Args:
        galfor (Settings): The galfor block (``log_sampling``, ``ndim`` read).

    Returns:
        bool: The effective ``log_sampling`` flag.

    Raises:
        ValueError: ``log_sampling`` with ``ndim`` other than ``None`` / 5.
    """
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

    Returns:
        TransformContainer: Input basis ``log10_amp, log10_fk, alpha,
        log10_f_1, log10_f_2``; output basis :data:`GALFOR_BASIS`.

    Example:
        .. code-block:: python

            tc = make_galfor_log_transform_container()
            linear = tc.both_transforms(stored_rows)   # (n, 5) log10 -> linear
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

    Reached as ``fit.galfor``; env knobs ``GALFOR_<FIELD>``. The prior box
    knobs (``GALFOR_ALPHA_MAX``, ``GALFOR_FREQ_PRIOR``, ``GALFOR_FK_PRIOR``,
    ``GALFOR_F1_PRIOR``, ``GALFOR_F2_PRIOR``) are read by
    :func:`galfor_prior_ranges`, the modulation knobs
    (``GALFOR_MODULATION_PATH`` / ``_T0``) by the variant's general block,
    and ``GALFOR_EIGEN_INFO`` (launchers ``fisher``) by the move.

    Attributes:
        ntemps: The galfor move's own ladder size. Env ``GALFOR_NTEMPS``.
            Default 12.
        galfor_kwargs: Reserved; the stock path does not read it (the field
            defaults to ``{}``, so the ``None`` branch in
            :meth:`GalForSetup.init_sampling_info` never fires).
        transform: Sampling -> linear transform; filled with
            :func:`make_galfor_log_transform_container` under ``log_sampling``.
        nleaves_max: Always 1.
        nleaves_min: Always 1.
        ndim: 5 (:data:`GALFOR_BASIS`).
        num_prop_repeats: Inner proposals per PSDMove call. Env
            ``GALFOR_NUM_PROP_REPEATS``. Default 50; launchers 10.
        stochastic_fn: Foreground spectral model for the sensitivity backend;
            ``None`` -> ``HyperbolicTangentGalacticForeground``.
        modulation: Foreground time modulation object; ``None`` -> the
            general block's ``galfor_modulation_path`` table, or stationary.
        log_sampling: Sample ``log10`` of ``amp, fk, f_1, f_2`` (alpha
            linear). Env ``GALFOR_LOG_SAMPLING``. Default ``False``;
            launchers 1.
        inner_move_kind: PSDMove inner proposal, ``"eigen"`` or ``"stretch"``;
            ``None`` resolves from the walker count. Env
            ``GALFOR_INNER_MOVE_KIND``.
        eigen_refresh_every: Proposes between eigen-table refreshes. Env
            ``GALFOR_EIGEN_REFRESH``. Default 10.
        eigen_eps_rel: Information-matrix finite-difference step (fraction of
            each prior-box width). Env ``GALFOR_EIGEN_EPS_REL``. Default 1e-4.
        ensemble_search: Tiled per-walker inner ensemble in SEARCH stages
            only. Env ``GALFOR_ENSEMBLE_SEARCH``. Default ``False``.
        ensemble_repeats: Env ``GALFOR_ENSEMBLE_REPEATS``. Default 10.
        ensemble_spread_lo: Env ``GALFOR_ENSEMBLE_SPREAD_LO``. Default 1.0.
        ensemble_spread_hi: Env ``GALFOR_ENSEMBLE_SPREAD_HI``. Default 10.0.
        ensemble_scale0: Env ``GALFOR_ENSEMBLE_SCALE0``. Default 1e-3.
        ensemble_scale_tries: Env ``GALFOR_ENSEMBLE_SCALE_TRIES``. Default 4.

    Example:
        .. code-block:: python

            fit = erebor.all_sources(nwalkers=4)
            fit.galfor.log_sampling = True     # or env GALFOR_LOG_SAMPLING=1
            fit.galfor.num_prop_repeats = 10
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
    # Inner proposal of the PSDMove sampling this block: `eigen` (per-rung
    # info-matrix axes; the one-walker default) or `stretch` (needs >= 2
    # walkers). See PSDMove._resolve_inner_kind.
    inner_move_kind: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_INNER_MOVE_KIND", None, str)
    )
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("GALFOR_EIGEN_REFRESH", 10, int)
    )
    eigen_eps_rel: float = dataclasses.field(
        default_factory=env_default("GALFOR_EIGEN_EPS_REL", 1e-4, float)
    )
    # --- tiled per-walker ensemble search (SEARCH stages only) ---------------
    # A one-walker block has no stretch complement, so the inner proposal
    # falls back to `eigen` -- and that path returned a non-positive
    # information matrix on 34 of 55 galfor builds on the live 3-month run.
    # With this on, each walker gets its own inner ensemble of
    # `ensemble_repeats` tiled copies, which manufactures a complement and
    # takes the information matrix off the SEARCH critical path entirely.
    # Only fires inside a search stage (PSDMove._ensemble_search_active);
    # PE is untouched. 10 repeats is eryn's red/blue floor of 2*ndim for
    # galfor's 5 columns.
    ensemble_search: bool = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_SEARCH", False, bool)
    )
    ensemble_repeats: int = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_REPEATS", 10, int)
    )
    # Closed loop on the MEASURED logL spread: perturb, score, rescale until
    # each walker's inner ensemble spans [lo, hi] logL. `scale0` is the
    # opening guess as a fraction of the prior box width -- safe at O(1) now
    # that GALFOR_LOG_SAMPLING makes the columns O(1) wide.
    ensemble_spread_lo: float = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_SPREAD_LO", 1.0, float)
    )
    ensemble_spread_hi: float = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_SPREAD_HI", 10.0, float)
    )
    ensemble_scale0: float = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_SCALE0", 1e-3, float)
    )
    ensemble_scale_tries: int = dataclasses.field(
        default_factory=env_default("GALFOR_ENSEMBLE_SCALE_TRIES", 4, int)
    )


class GalForSetup(Setup):
    """:class:`Setup` for the galactic-foreground branch in the Erebor recipe.

    Construction copies the settings, opens the ``galfor_setup.log`` logger
    and runs :meth:`init_setup`.

    Args:
        galfor_settings: Settings dataclass with foreground kwargs, prior,
            and tempering configuration.
    """

    def __init__(self, galfor_settings: GalForSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(galfor_settings)

        level = logging.DEBUG
        name = "GalForSetup"
        self.logger = init_logger(filename="galfor_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

        self.init_setup()

    def init_sampling_info(self):
        """Build the galactic-foreground prior, tempering kwargs, and default ``galfor_kwargs``.

        Fills (only when unset): ``initialize_kwargs``, ``priors``
        (:func:`galfor_prior_dict` in the allowed basis), ``transform`` (log
        basis only), ``betas`` (``make_ladder(10 * ndim, Tmax=inf,
        ntemps=ntemps)``) and ``other_tempering_kwargs`` (``permute=False``).

        Raises:
            ValueError: From :func:`check_galfor_log_sampling` or the prior
                knobs.
            AssertionError: When ``other_tempering_kwargs["permute"]`` is true.
        """
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


# TODO check: keep/remove/document -- legacy helper; no caller anywhere (erebor re-export only)
def get_galfor_erebor_settings(general_set: GeneralSetup) -> GalForSetup:
    """Construct the default :class:`GalForSetup` for an Erebor run.

    Builds a :class:`GalForSettings` from the run-wide ``Tobs`` / ``dt``
    carried on ``general_set`` and wraps it in a :class:`GalForSetup`.

    Note:
        The local ``Tobs = YRSID_SI`` and ``dt = 10.0`` are never used (the
        settings take ``general_set.Tobs`` / ``.dt``).

    Args:
        general_set: Run-wide :class:`GeneralSetup` providing ``Tobs`` and
            ``dt``.

    Returns:
        GalForSetup: Configured setup ready for use in the Erebor pipeline.
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
    Production (``COARSE_Q=1``, ``COARSE_GPU_MODE=off``) passes trivially.

    Args:
        gs (EreborGeneralSettings): The general block (``coarse_Q``,
            ``coarse_fiducial``, ``coarse_gpu_mode``, ``gpus``); ``coarse_Q`` and
            ``coarse_gpu_mode`` are normalized in place.
        all_source: ``True`` for ``all_sources`` (sidecar rules), ``False``
            for the noise-only variants.

    Raises:
        ValueError: ``coarse_Q`` not an integer >= 1, an unknown
            ``coarse_fiducial`` / ``coarse_gpu_mode``, a sidecar mode on a
            noise-only run, ``coarse_Q > 1`` with GPUs on a noise-only run,
            ``coarse_Q > 1`` with mode ``"off"`` on an all-source run, or a
            sidecar mode with ``coarse_Q <= 1``.
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

    Args:
        path (str | None): Modulation table (``GALFOR_MODULATION_PATH``);
            falsy -> stationary.
        t0: Absolute epoch (s) of the table's time origin.

    Returns:
        GalForTimeModulation | None: The modulation, or ``None``.
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

    Args:
        mojito_data_path: Brick-cache root (``MOJITO_DATA_PATH``).
        explicit: Explicit brick path (``NOISE_FILE``).

    Returns:
        str | None: The brick path, or ``None``.

    Raises:
        FileNotFoundError: When ``explicit`` is given but does not exist.
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
    (``all_sources`` actually calls its own identical copy,
    ``AllSourcesGlobalFit._wire_unequal_arm``; production ``UNEQUAL_ARM=1``,
    ``UNEQUAL_ARM_STRIDE=200``, ``WDM_PSD_METHOD=layer_calibrated``.)

    Args:
        gs (EreborGeneralSettings): The general block (``data_mode``,
            ``mojito_data_path``, ``noise_file``, ``unequal_arm_stride``,
            ``wdm_psd_method``).
        psd (PSDSettings): The psd branch block; its ``instrument_component_cls`` (if
            unset) and ``instrument_component_kwargs`` (``ltts_l1_file``,
            ``ltts_stride``, ``wdm_psd_method`` via ``setdefault``) are set.

    Raises:
        ValueError: No psd branch, non-mojito data, or a brick without
            ``/ltts``.
        FileNotFoundError: No NOISE brick found.
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
    honor_resume_pin: bool = True,
) -> typing.Optional[typing.List[float]]:
    """``[Soms_d, Sa_a]`` fit to the NOISE brick's tabulated estimates.

    Wraps :func:`lisatools.sensitivity.estimate_noise_params_from_file`; the
    fit band is clipped to the tabulated grid. Returns ``None`` (with a
    warning) if the fit fails, so callers can fall back to the stock levels.
    ``honor_resume_pin=False`` ignores ``MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM``
    (diagnostics only; a run's ``general.psd_injection`` keeps honouring it).
    The production launchers export ``MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=1``.

    Args:
        noise_file: Mojito NOISE L1 brick path.
        band: Optional fit band ``(f_lo, f_hi)`` (Hz).
        tdi_generation: TDI generation of the model (1 or 2). Default 2.
        honor_resume_pin: Honor the equal-arm resume pin env knob. Default
            ``True``.

    Returns:
        list[float] | None: ``[Soms_d, Sa_a]`` (sqrt units), or ``None`` when
        the fit raised (logged as a warning).
    """
    from lisatools.sensitivity import estimate_noise_params_from_file

    kwargs = {"tdi_generation": tdi_generation,
              "honor_resume_pin": bool(honor_resume_pin)}
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


#: The instrument levels the mojito-light v1.0.0 bricks are built with. Used
#: only when no brick is reachable -- a measured fit against the brick's own
#: tabulated estimates is preferred, and recovers these to 3e-6 / 4e-5.
MOJITO_LIGHT_PSD_INJECTION = (1.5e-11, 3.0e-15)


def psd_truth_levels(
    noise_file: typing.Optional[str] = None,
    mojito_data_path: typing.Optional[str] = None,
    default: typing.Tuple[float, float] = MOJITO_LIGHT_PSD_INJECTION,
) -> typing.Tuple[float, float]:
    """``(Soms_d, Sa_a)`` to draw as the truth line on a noise plot.

    The one place the diagnostic pages should get this pair. Resolution order:

    1. ``PSD_TRUTH="<soms_d>,<sa_a>"`` in the environment — an explicit
       override for a run against a non-stock brick;
    2. the NOISE brick's own tabulated estimates via
       :func:`noise_params_from_file`, i.e. the unequal-arm fit at the brick's
       ``/ltts`` -- ALWAYS unequal-arm: ``MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=0``
       pins a run's own ``general.psd_injection`` to the equal-arm pair so an
       old store can resume, and the monitor rank inherits that environment;
       it used to draw the equal-arm pair as "injected" and report a correct
       fit as +0.2% high (2026-10-09). The pin is ignored here;
    3. ``default``.

    WHY THIS EXISTS. The pair used to be copy-pasted as a literal into every
    generator (``1.496182e-11, 2.982412e-15``). That literal came from the
    EQUAL-arm fit, which lands 0.26% / 0.59% low (see
    :meth:`~lisatools.sensitivity.MojitoNoiseEstimates.fit_scalar_params`), so
    every page drew its truth line in the wrong place and a 6-month noise-only
    fit that is accurate to -0.02% / -0.22% read as biased by +0.23% / +0.37%.
    Five copies of a number nobody re-derived is how that survived; resolve it
    here instead.

    Args:
        noise_file: Explicit path to a mojito NOISE L1 brick.
        mojito_data_path: Brick-market root to auto-discover the NOISE file
            under, when ``noise_file`` is not given.
        default: Returned when neither the environment nor a brick provides
            the levels — snapshot pages routinely run where the 5.9 GB brick
            is not present.

    Returns:
        tuple[float, float]: ``(Soms_d, Sa_a)`` in linear (square-root) units.

    Raises:
        ValueError: When ``PSD_TRUTH`` is set but is not two numbers.
    """
    env = os.environ.get("PSD_TRUTH")
    if env:
        parts = [p for p in env.replace(",", " ").split() if p]
        if len(parts) != 2:
            raise ValueError(
                f'PSD_TRUTH must be "<Soms_d>,<Sa_a>"; got {env!r}.'
            )
        return (float(parts[0]), float(parts[1]))

    path = noise_file
    if path is None and mojito_data_path is not None:
        path = resolve_noise_file(mojito_data_path, None)
    if path is not None:
        params = noise_params_from_file(path, honor_resume_pin=False)
        if params is not None:
            return (float(params[0]), float(params[1]))
    return (float(default[0]), float(default[1]))


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

    Args:
        psd (PSDSettings): The psd block (mutated in place; fields already
            set win).
        psd_injection (sequence, optional): Linear ``[Soms_d, Sa_a]``;
            stored as ``psd.injection`` (``ln`` of it under ``log_sampling``)
            when ``psd.injection`` is unset.

    Returns:
        PSDSettings: ``psd``.

    Raises:
        ValueError: From :func:`check_psd_log_sampling`.
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
    basis. Note the base: the psd branch is ``ln``, this one is ``log10``.
    Filling them here rather than leaving it to :class:`GalForSetup` keeps the
    knob effective for callers that pass their own Setup. Shared by the noise
    variants and all_sources.

    Args:
        galfor (GalForSettings): The galfor block (mutated in place; fields
            already set win).

    Returns:
        GalForSettings: ``galfor``.

    Raises:
        ValueError: From :func:`check_galfor_log_sampling` or the prior knobs.
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

    Args:
        base (dict, optional): Existing ``sensitivity_init_kwargs`` (copied).
        tdi_generation (int): TDI generation (``setdefault``).
        galfor (GalForSettings, optional): Supplies ``stochastic_fn`` /
            ``modulation``.
        psd (PSDSettings, optional): Supplies the ``instrument_*`` choices and
            ``model_name``; a ``wdm_psd_method`` in
            ``instrument_component_kwargs`` is promoted to the backend-wide
            ``wdm_psd_method``.
        galfor_modulation_path (str, optional): Modulation table used when the
            branch carries no ``modulation`` object.
        galfor_modulation_t0 (float): Epoch of that table. Default 0.0.
        extra (dict, optional): Merged last (overrides).

    Returns:
        dict: The kwargs for the run's sensitivity backend
        (``galfor_modulation`` is always present, ``None`` = stationary).
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
