"""Stock variants ``noise_only`` / ``noise_sgwb``: joint WDM noise fit(s).

Both variants fit the LISA noise — there are no GW *source* branches:

* ``noise_only`` — instrument PSD (``psd``) + galactic-confusion foreground
  (``galfor``). **This is the tested variant.**
* ``noise_sgwb`` — the same, plus a power-law stochastic GW background
  (``sgwb``). Built end-to-end; its SGWB posterior is not validated here.

Data (one knob, ``data_mode`` / env ``DATA_MODE``):

* ``"mojito"`` (default) — the **real mojito NOISE brick**
  (``<mojito_data_path>/data/INSTRUMENT/L1/NOISE_*``), downsampled onto the
  run grid; the psd branch's start/reference levels are read off the brick's
  tabulated ``noise_estimates`` (see
  :func:`lisatools.sensitivity.estimate_noise_params_from_file`). Falls back
  to ``"synthetic"`` with a warning when the brick is missing. Note the
  brick contains **instrument noise only** — the galfor branch then rails at
  its amplitude floor; drop it with ``fit.remove_branch("galfor")`` for a
  pure-instrument fit.
* ``"synthetic"`` — a self-consistent WDM-domain realization Cholesky-drawn
  from the *injected* composite covariance (instrument + foreground
  [+ SGWB]), so the fit recovers the injected truths exactly.

All branches are sampled jointly by the single
:class:`~lisatools.globalfit.moves.psdmove.PSDMove`. The foreground time
modulation flows through the general
:class:`~lisatools.sensitivity.GalForTimeModulation` framework (default:
stationary; set ``galfor_modulation_path`` / ``GALFOR_MODULATION_PATH`` to load
a tabulated modulation).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.noise_only(nwalkers=4)
    fit.build(); fit.run()          # or: run_global.py --stock noise_only
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist

from lisatools.domains import WDMSettings

from ....engine import GeneralSetup, Settings
from ....moves import Move, MoveBuildContext
from ....recipe import Recipe, Stage, build_noise_moves
from ...base import env_default
from ..fit import EreborFit, EreborGeneralSettings
from ..noise import (
    GalForSettings,
    GalForSetup,
    PSDSettings,
    PSDSetup,
    noise_sensitivity_init_kwargs,
    prepare_galfor_branch,
    prepare_psd_branch,
    resolve_galfor_modulation,
)
from ..stochastic import SGWBSettings, SGWBSetup

logger = logging.getLogger(__name__)

# Injection truths. These parameterize the covariance the synthetic data are
# drawn from AND sit inside the sampled priors, so the fit should recover them.
PSD_INJECTION = [15e-12, 3e-15]  # (Soms_d, Sa_a), sqrt units
# f_1 / f_2 are in the MODEL's basis (Hz), i.e. the arguments
# ``HyperbolicTangentGalacticForeground.specific_Sh_function`` actually takes
# -- not the "Slope1"/"Slope2" numbers tabulated on
# ``FittedHyperbolicTangentGalacticForeground``. Those need the same
# conversion that class applies before calling the model (stochastic.py):
#     f_1 = slope1 ** (-1/alpha) = 3.01430978e03 ** (-1/1.18300266) = 1.1456e-3
#     f_2 = 1 / slope2          = 1 / 2.95774596e03                 = 3.3810e-4
# Fixed 2026-08: the raw slope values were injected here, which put the
# roll-off and the tanh knee at kHz and left the injected foreground a pure
# f^(-7/3) power law with no bend anywhere in the LISA band (a factor 2.4e3
# error in S_gal at 3 mHz, 1.6e19 at 8 mHz).
GALFOR_INJECTION = [
    3.26651613e-44,  # amp
    2.09278117e-03,  # fk (knee) [Hz]
    1.18300266e00,   # alpha
    1.14556409e-03,  # f_1 (Hz), = slope1 ** (-1/alpha)
    3.38095297e-04,  # f_2 (Hz), = 1 / slope2
]
# PowerLawSGWB amplitude is Omega_gw at 25 Hz; chosen detectable in a short
# datastream (the stock SGWB prior (-22, -18) would NOT contain this).
SGWB_INJECTION = [-9.5, 2.0 / 3.0]  # (log10_A @ 25 Hz, alpha)


@dataclasses.dataclass
class NoisePSDSettings(PSDSettings):
    """PSD branch: 2 params ``(Soms_d, Sa_a)`` (not the 4-param spline default)."""

    ndim: int = 2
    injection: typing.Optional[np.ndarray] = None


@dataclasses.dataclass
class NoiseSGWBSettings(SGWBSettings):
    """SGWB branch: 2-param power law ``(log10_A, alpha)``."""

    ndim: int = 2
    injection: typing.Optional[np.ndarray] = None


@dataclasses.dataclass
class NoiseGeneralSettings(EreborGeneralSettings):
    """General block for the noise fits (fixed smoke-friendly WDM grid).

    ``Tobs == Nf*Nt*dt`` exactly (768*1024*5 s ~ 45.5 d), 0.3-8 mHz band.
    """

    dt: float = 5.0
    nf: typing.Optional[int] = 768
    nt: typing.Optional[int] = 1024
    min_freq: float = 3e-4
    max_freq: float = 8e-3
    window_tukey_alpha: float = 0.0  # rectangular; the WDM draw ignores the window
    edge_crop_wavelets: typing.Optional[int] = 0  # no time crop -> draw matches grid
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 4, int))
    # engine ntemps inherited from EreborGeneralSettings: retired to 1
    # (branches temper internally; see the per-branch <BRANCH>_NTEMPS knobs)
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_noise/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "noise_smoke_test")
    )
    source_types: typing.Tuple[str, ...] = ()  # no external source loading
    # The NOISE is the signal here, so keep the ``-logdet_factor * sum log det C``
    # term in the likelihood (source-only variants set this True to drop it).
    likelihood_source_only: bool = False
    # Sample psd + galfor in ONE move on ONE ladder instead of the default
    # per-branch split (2026-08). Knob: ``JOINT_NOISE_MOVE``.
    #
    # Default (False): one PSDMove per branch, each on its own ladder, run
    # in sequence with the partner frozen at its cold row -- Metropolis-
    # within-Gibbs. Correct (the loop is closed: each move sees the previous
    # move's update), and cheaper per iteration, but it proposes along the
    # coordinate axes, so it climbs the psd<->galfor correlation ridge in a
    # staircase. That ridge is real: both branches sum into one covariance
    # C = instrument + foreground and overlap in the 1-3 mHz window, which is
    # why ln S_tm and ln amp come out correlated.
    #
    # True: a single PSDMove over ["psd", "galfor"], so the stretch proposal
    # moves all 7 parameters together and can travel ALONG the ridge, and the
    # tempered ladder is genuinely joint (in the split scheme each branch's
    # hot rungs are scored against the partner's COLD row, so tempering
    # cannot carry the pair across a barrier together). Costs a wider
    # proposal to tune and requires PSD_NTEMPS == GALFOR_NTEMPS -- both
    # default to 12; ``build_noise_moves`` raises if they disagree.
    joint_noise_move: bool = dataclasses.field(
        default_factory=env_default("JOINT_NOISE_MOVE", False, bool)
    )

    # CPU threads for the per-walker sensitivity builds inside the noise move
    # (PSDMove.build_threads; knob ``PSD_BUILD_THREADS``). 1 = the historical
    # serial loop. Only bites on the ACA route -- i.e. the WDM basis these
    # fits use, where the C++ XYZ kernel fast path is unavailable. Scaling
    # saturates at ~4 (3.11 ms/walker serial -> 1.73 at 4 -> 1.78 at 6 on
    # the stock 59x1024 active grid): the arrays are small, so Python
    # orchestration sets the floor. Ignored by the GPU path, which spreads
    # over devices instead.
    psd_build_threads: int = dataclasses.field(
        default_factory=env_default("PSD_BUILD_THREADS", 1, int)
    )

    # "mojito" (default): the real NOISE brick; "synthetic": the Cholesky
    # draw from the injected covariance. Auto-falls back to synthetic when
    # the brick / mojito folder is missing.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_MODE", "mojito", str)
    )
    noise_seed: int = dataclasses.field(default_factory=env_default("NOISE_SEED", 0, int))
    t_start: float = 0.0

    # Injections consumed by the synthetic processor + sensitivity backend.
    # psd_injection None -> auto: the NOISE brick's fitted [Soms_d, Sa_a] in
    # mojito mode, else the stock PSD_INJECTION.
    psd_injection: typing.Optional[typing.Sequence[float]] = None
    galfor_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: list(GALFOR_INJECTION)
    )
    sgwb_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: list(SGWB_INJECTION)
    )
    sgwb_stochastic_fn: str = "PowerLawSGWB"

    # Galactic-foreground time modulation. None (default) => stationary, fully
    # self-consistent (injection AND model both stationary). Set a path (or env
    # GALFOR_MODULATION_PATH) to opt into a tabulated GalForTimeModulation.
    galfor_modulation_path: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_MODULATION_PATH", None, str)
    )


class _NoiseFitBase(EreborFit):
    """Shared build behaviour for the noise fits (not registered directly)."""

    general_settings_class = NoiseGeneralSettings

    # -- foreground modulation (None => stationary) --
    def _resolve_modulation(self, gs: NoiseGeneralSettings):
        return resolve_galfor_modulation(gs.galfor_modulation_path)

    # -- domain: bare WDM factory, no time crop so the (nch, Nf, Nt) draw
    #    matches the full active grid --
    def make_domain_settings(self, gs, Nf, Nt, wavelet_duration, edge_crop):
        return WDMSettings.make_factory(
            Nf=Nf, Nt=Nt, min_freq=gs.min_freq, max_freq=gs.max_freq
        )

    # -- data source: the real NOISE brick (mojito) or the synthetic draw --
    def resolve_data_source(self) -> None:
        super().resolve_data_source()
        # Mutates the FIT-LEVEL block (like the base method): _prepare_psd
        # reads self.general.psd_injection at branch prep.
        gs = self.general
        if gs.data_mode == "mojito":
            from ..noise import resolve_noise_file

            if resolve_noise_file(gs.mojito_data_path, gs.noise_file) is None:
                logger.warning(
                    "noise fit data_mode='mojito' but no NOISE brick was found "
                    "under %s — falling back to data_mode='synthetic' (set "
                    "NOISE_FILE / general.noise_file to use the real noise).",
                    gs.mojito_data_path,
                )
                gs.data_mode = "synthetic"
        if gs.psd_injection is None:
            # NOISE-brick fit in mojito mode (auto), stock levels otherwise.
            params = self.resolve_noise_file_psd_params(gs)
            gs.psd_injection = list(params) if params is not None else list(PSD_INJECTION)

    def default_preprocess_kwargs(self) -> dict:
        if self.general.data_mode == "mojito":
            # Real NOISE brick (2.5 s native): downsample onto the run grid
            # and keep the engine's default highpass / edge-trim / Tobs trim
            # so exactly Nf*Nt samples survive to the WDM pour.
            return dict(
                downsample_kwargs=dict(target_fs=1.0 / self.general.dt),
                normalize=False,
            )
        # The synthetic WDM draw needs no highpass/trim/normalize.
        return dict(highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False)

    def set_default_processor(self, gs: NoiseGeneralSettings) -> None:
        if gs.data_mode == "mojito":
            from lisatools.detector import L1Orbits

            from ....preprocessing import L1ProcessingStep

            force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
            gs.data_processor_class = L1ProcessingStep
            gs.processor_init_kwargs = dict(
                L1_folder=gs.mojito_data_path,
                source_types=["NOISE"],
                source_ids=None,
                orbits_class=L1Orbits,
                orbits_kwargs=dict(
                    force_backend=force_backend, frame=gs.orbits_frame
                ),
                verbose=gs.verbose,
            )
            return
        if gs.data_mode != "synthetic":
            raise ValueError(
                f"noise fit data_mode={gs.data_mode!r}; use 'mojito' (the real "
                "NOISE brick) or 'synthetic' (or swap data_processor_class "
                "wholesale)."
            )
        from ..injections import SyntheticNoiseProcessingStep

        gs.data_processor_class = SyntheticNoiseProcessingStep
        gs.processor_init_kwargs = dict(
            Tobs=gs.Tobs,
            dt=gs.dt,
            t_start=gs.t_start,
            psd_injection=list(gs.psd_injection),
            galfor_injection=list(gs.galfor_injection),
            sgwb_injection=self._sgwb_injection(gs),  # None for noise_only
            galfor_modulation=self._resolve_modulation(gs),
            sgwb_stochastic_fn=gs.sgwb_stochastic_fn,
            tdi_generation=gs.tdi_gen,
            seed=gs.noise_seed,
            verbose=gs.verbose,
        )

    # -- sensitivity backend: thread the per-branch NOISE-MODEL choice onto the
    #    CompositeSensitivityBackend, the same way a source branch's waveform
    #    model is read off its Settings. A user swaps the noise model via
    #    ``fit.galfor.stochastic_fn`` / ``fit.galfor.modulation`` /
    #    ``fit.sgwb.stochastic_fn`` / ``fit.psd.instrument_model_cls`` — no
    #    editing of general-level ``sensitivity_init_kwargs`` needed. --
    def finalize_general(self, gs: NoiseGeneralSettings) -> None:
        gs.sensitivity_init_kwargs = noise_sensitivity_init_kwargs(
            gs.sensitivity_init_kwargs,
            tdi_generation=gs.tdi_gen,
            galfor=getattr(self, "galfor", None),
            psd=getattr(self, "psd", None),
            galfor_modulation_path=gs.galfor_modulation_path,
            extra=self._sgwb_sens_kwargs(gs),
        )

    # -- per-branch priors / injections --
    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _prepare_psd(self, psd: NoisePSDSettings, general_setup: GeneralSetup):
        return prepare_psd_branch(psd, self.general.psd_injection)

    def _prepare_galfor(self, galfor: GalForSettings, general_setup: GeneralSetup):
        return prepare_galfor_branch(galfor)

    # -- sgwb presence hooks (overridden by NoiseSGWBGlobalFit) --
    def _sgwb_injection(self, gs: NoiseGeneralSettings):
        return None

    def _sgwb_sens_kwargs(self, gs: NoiseGeneralSettings) -> dict:
        return {}


class NoiseOnlyGlobalFit(_NoiseFitBase):
    """Instrument PSD + galactic-foreground fit on synthetic WDM noise."""

    option_name = "noise_only"
    description = (
        "Joint WDM noise fit: instrument PSD + hyperbolic-tangent galactic "
        "foreground, on a self-consistent synthetic composite-noise realization."
    )
    general_settings_class = NoiseGeneralSettings
    setup_classes = {"psd": PSDSetup, "galfor": GalForSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {"psd": NoisePSDSettings(), "galfor": GalForSettings()}

    def _apply_knobs(self, knobs: dict):
        """Rebuild the default recipe when ``joint_noise_move`` arrives as a kwarg.

        Base ``__init__`` builds the recipe (base.py:312) BEFORE applying
        knobs (base.py:319), so :meth:`default_recipe` sees only the
        env-resolved value -- already on the field by then, which is why
        ``JOINT_NOISE_MOVE=1`` works unaided but
        ``noise_only(joint_noise_move=True)`` would not. Catch the change
        here, the documented interception point for variant knobs
        (``apply_overrides``), so both spellings agree.

        Rebuilds ONLY when the current recipe is still exactly what
        ``default_recipe`` produced under the old value -- a caller-supplied
        or already-edited recipe is never silently replaced.
        """
        before = getattr(self.general, "joint_noise_move", False)
        current = [m.name for st in self.recipe.stages for m in st.moves]
        super()._apply_knobs(knobs)
        after = getattr(self.general, "joint_noise_move", False)
        if after == before:
            return
        self.general.joint_noise_move = before
        try:
            pristine = [m.name for st in self.default_recipe().stages for m in st.moves]
        finally:
            self.general.joint_noise_move = after
        if current == pristine:
            self.recipe = self.default_recipe()

    def default_recipe(self) -> Recipe:
        # Noise-move split (2026-07): each noise branch gets its OWN move +
        # ladder inside the one PE stage (share_temperature_control=False is
        # what keeps the per-move ladders independent under GFCombineMove).
        # ``general.joint_noise_move`` collapses the two into ONE move over
        # both branches instead -- see the knob's comment for the trade-off.
        # Safe to read self.general here: base.py sets it (298) before it
        # calls default_recipe (312).
        if getattr(self.general, "joint_noise_move", False):
            moves = [Move("noise_pe", branch="psd")]
        else:
            moves = [
                Move("psd_pe", branch="psd"),
                Move("galfor_pe", branch="galfor"),
            ]
        return Recipe(
            [
                Stage(
                    name="noise_pe",
                    kind="pe",
                    moves=moves,
                    # kept in the joint case too: NoiseSGWBGlobalFit appends
                    # its own sgwb move to this stage, and that one still
                    # needs an independent ladder.
                    combine_kwargs=dict(share_temperature_control=False),
                )
            ]
        )


class NoiseSGWBGlobalFit(NoiseOnlyGlobalFit):
    """PSD + galactic foreground + power-law SGWB fit on synthetic WDM noise."""

    option_name = "noise_sgwb"
    description = (
        "Joint WDM noise fit: instrument PSD + galactic foreground + power-law "
        "SGWB, on a self-consistent synthetic composite-noise realization."
    )
    setup_classes = {"psd": PSDSetup, "galfor": GalForSetup, "sgwb": SGWBSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        branches = super().default_branches()
        branches["sgwb"] = NoiseSGWBSettings()
        return branches

    def default_recipe(self) -> Recipe:
        recipe = super().default_recipe()
        recipe.stages[0].moves.append(Move("sgwb_pe", branch="sgwb"))
        return recipe

    def _sgwb_injection(self, gs: NoiseGeneralSettings):
        return list(gs.sgwb_injection)

    def _sgwb_sens_kwargs(self, gs: NoiseGeneralSettings) -> dict:
        # Read the SGWB template off the branch Settings (symmetry); fall back to
        # the general-level default.
        sgwb = getattr(self, "sgwb", None)
        fn = getattr(sgwb, "stochastic_fn", None) if sgwb is not None else None
        return dict(sgwb_stochastic_fn=fn or gs.sgwb_stochastic_fn)

    def _prepare_sgwb(self, sgwb: NoiseSGWBSettings, general_setup: GeneralSetup):
        if sgwb.initialize_kwargs is None:
            sgwb.initialize_kwargs = {}
        if sgwb.priors is None:
            # Override the stock (-22, -18) default so the prior CONTAINS the
            # injected log10_A = -9.5 (GLASS-style default would exclude it).
            sgwb.priors = {
                "sgwb": ProbDistContainer(
                    {
                        0: uniform_dist(-16.0, -9.0),  # log10_A
                        1: uniform_dist(-1.0, 2.0),  # alpha
                    }
                )
            }
        if sgwb.injection is None:
            sgwb.injection = np.asarray(self.general.sgwb_injection, dtype=float)
        return sgwb


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup: build one independent move pair per noise branch, then materialize."""
    general_info = curr.general_info
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    requested = recipe.stock_names()
    stock_moves = {}
    # noise-move split: each present noise branch gets its own move pair +
    # ladder; the stock names are <branch>_pe / <branch>_search.
    # ``joint_noise_move`` instead groups psd+galfor under ONE move pair
    # named noise_pe / noise_search, sampled together on a single ladder.
    # sgwb keeps its own move either way -- it is a separate physical
    # component and is not what the ridge couples.
    if getattr(general_info, "joint_noise_move", False):
        noise_move_spec = {"noise": ["psd", "galfor"], "sgwb": ["sgwb"]}
    else:
        noise_move_spec = {"psd": ["psd"], "galfor": ["galfor"], "sgwb": ["sgwb"]}
    for name, sampled in noise_move_spec.items():
        # Drop branches this run does not have rather than skipping the whole
        # group: it lets the joint move degrade cleanly to the branches that
        # ARE present (instrument mode removes galfor, so noise_pe becomes a
        # psd-only move) instead of vanishing and leaving the recipe asking
        # for a stock name nobody built.
        sampled = [b for b in sampled if b in curr.source_info]
        if not sampled:
            continue
        if not any(n.startswith(name) for n in requested):
            continue
        # CPU runs use few repeats (laptop smokes); GPU runs use the branch
        # settings' num_prop_repeats
        num_repeats = 5 if gpus is None else None
        search_move, pe_move = build_noise_moves(
            engine_info, curr, acs, priors,
            sampled_branches=sampled, num_repeats=num_repeats,
        )
        stock_moves[f"{name}_pe"] = pe_move
        stock_moves[f"{name}_search"] = search_move

    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves=stock_moves,
        ntemps=ntemps, nwalkers=nwalkers,
    )
    recipe.setup(ctx)


NoiseOnlyGlobalFit.default_setup_function = staticmethod(setup_recipe)
NoiseSGWBGlobalFit.default_setup_function = staticmethod(setup_recipe)
