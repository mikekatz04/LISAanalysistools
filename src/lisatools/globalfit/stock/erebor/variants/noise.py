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

By default each branch has its own
:class:`~lisatools.globalfit.moves.psdmove.PSDMove` on its own ladder inside
the one ``noise_pe`` stage; ``joint_noise_move`` (``JOINT_NOISE_MOVE=1``)
samples psd + galfor together with a single ``noise_pe`` move. The foreground time
modulation flows through the general
:class:`~lisatools.sensitivity.GalForTimeModulation` framework (default:
stationary; set ``galfor_modulation_path`` / ``GALFOR_MODULATION_PATH`` to load
a tabulated modulation).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.noise_only(nwalkers=4)
    fit.build(); fit.run()          # or: run_global.py --stock noise_only

Role in production: the variants are standalone noise studies, but three
pieces of this module are part of the production ``all_sources`` branch set:
:class:`NoisePSDSettings` (its ``psd`` block), :data:`PSD_INJECTION` (the
fallback psd levels) and :data:`GALFOR_INJECTION` (the foreground reference).
The unequal-arm switch (``UNEQUAL_ARM``) is wired here by
:func:`~lisatools.globalfit.stock.erebor.noise.wire_unequal_arm_psd`;
``all_sources`` carries a duplicate of that wiring
(``AllSourcesGlobalFit._wire_unequal_arm``).
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
    validate_coarse_settings,
    GalForSettings,
    GalForSetup,
    PSDSettings,
    PSDSetup,
    noise_sensitivity_init_kwargs,
    prepare_galfor_branch,
    prepare_psd_branch,
    resolve_galfor_modulation,
    wire_unequal_arm_psd,
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
    """PSD branch: 2 params ``(Soms_d, Sa_a)`` (not the 4-param spline default).

    The ``psd`` block of the noise variants and of the production
    ``all_sources`` fit (``fit.psd``). Other fields are those of
    :class:`~lisatools.globalfit.stock.erebor.noise.PSDSettings`.

    Attributes:
        ndim: Sampled parameters. Default ``2``.
        injection: Reference / start ``[Soms_d, Sa_a]``; ``None`` (default)
            is filled by the branch prep from ``general.psd_injection``.
    """

    ndim: int = 2
    injection: typing.Optional[np.ndarray] = None


@dataclasses.dataclass
class NoiseSGWBSettings(SGWBSettings):
    """SGWB branch: 2-param power law ``(log10_A, alpha)``.

    Other fields are those of
    :class:`~lisatools.globalfit.stock.erebor.stochastic.SGWBSettings`.

    Attributes:
        ndim: Sampled parameters. Default ``2``.
        injection: Reference ``[log10_A, alpha]``; ``None`` (default) is
            filled from ``general.sgwb_injection``.
    """

    ndim: int = 2
    injection: typing.Optional[np.ndarray] = None


@dataclasses.dataclass
class NoiseGeneralSettings(EreborGeneralSettings):
    """General block for the noise fits (fixed smoke-friendly WDM grid).

    ``Tobs == Nf*Nt*dt`` exactly (768*1024*5 s ~ 45.5 d), 0.3-8 mHz band.

    Subclass of :class:`~lisatools.globalfit.stock.erebor.fit.EreborGeneralSettings`
    (inherited fields, including the coarse-likelihood knobs, are
    documented there).

    Attributes:
        dt: Sample spacing [s]. Default ``5.0``.
        nf: WDM frequency layers. Default ``768``.
        nt: WDM time layers. Default ``1024``.
        min_freq: Data-band floor [Hz]. Default ``3e-4``.
        max_freq: Data-band ceiling [Hz]. Default ``8e-3``.
        window_tukey_alpha: Default ``0.0`` (rectangular).
        edge_crop_wavelets: Default ``0`` (no crop; the synthetic draw fills
            the full grid).
        nwalkers: Env ``NWALKERS``. Default ``4``.
        file_store_dir: Env ``FILE_STORE_DIR``. Default
            ``"./gf_output_noise/"``.
        base_file_name: Env ``BASE_FILE_NAME``. Default ``"noise_smoke_test"``.
        source_types: Default ``()`` (the mojito processor loads ``NOISE``).
        likelihood_source_only: Default ``False`` (keep the noise
            normalization: the noise is the signal here).
        joint_noise_move: One ``noise_pe`` move over psd + galfor on one
            ladder instead of a move per branch (needs
            ``PSD_NTEMPS == GALFOR_NTEMPS``). Env ``JOINT_NOISE_MOVE``.
            Default ``False``.
        psd_build_threads: CPU threads for the per-walker sensitivity builds
            in the noise move. Env ``PSD_BUILD_THREADS``. Default ``1``.
        unequal_arm: Unequal-arm instrument noise from the NOISE brick's
            ``/ltts`` (mojito only). Env ``UNEQUAL_ARM``. Default ``False``.
        unequal_arm_stride: Delay-table decimation. Env
            ``UNEQUAL_ARM_STRIDE``. Default ``200``.
        wdm_psd_method: Backend-wide WDM PSD method (``None`` = exact fold).
            Env ``WDM_PSD_METHOD``. Default ``None``.
        data_mode: ``"mojito"`` (the NOISE brick; falls back to
            ``"synthetic"`` with a warning when missing) or ``"synthetic"``.
            Env ``DATA_MODE``. Default ``"mojito"``.
        noise_seed: Synthetic draw seed. Env ``NOISE_SEED``. Default ``0``.
        t_start: Synthetic start [s]. Default ``0.0``.
        psd_injection: ``[Soms_d, Sa_a]``; ``None`` (default) -> the
            NOISE-brick fit in mojito mode, else :data:`PSD_INJECTION`.
        galfor_injection: Foreground injection. Default
            :data:`GALFOR_INJECTION`.
        sgwb_injection: SGWB injection (``noise_sgwb``). Default
            :data:`SGWB_INJECTION`.
        sgwb_stochastic_fn: Default ``"PowerLawSGWB"``.
        galfor_modulation_path: Foreground modulation table. Env
            ``GALFOR_MODULATION_PATH``. Default ``None`` (stationary).
        galfor_modulation_t0: Table epoch (``None``, ``"data"`` or seconds).
            Env ``GALFOR_MODULATION_T0``. Default ``None``.
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

    # --- unequal-arm instrument noise (env UNEQUAL_ARM=1) ---
    # Mirror all_sources so this diagnostic exercises v8's EXACT noise model:
    # swap the equal-arm InstrumentNoise for UnequalArmInstrumentNoise, the six
    # link light travel times read from the mojito NOISE brick's /ltts group and
    # averaged per WDM time column (LinkDelayTable). Mojito data only;
    # synthetic/sangria carry no delay table and are refused loudly. Default
    # False -> standard InstrumentNoise, bit-identical to before.
    unequal_arm: bool = dataclasses.field(
        default_factory=env_default("UNEQUAL_ARM", False, bool)
    )
    # Decimation of the tabulated per-link delays (brick tabulates one sample
    # per 2.5 s; the delays vary on month timescales).
    unequal_arm_stride: int = dataclasses.field(
        default_factory=env_default("UNEQUAL_ARM_STRIDE", 200, int)
    )
    # WDM PSD evaluation method for the whole sensitivity backend: None ->
    # the backend default ("fold", exact); "layer_constant";
    # "layer_calibrated" (recommended with unequal_arm: one exact fold
    # calibrates the ~200x cheaper layer-center evaluation; see
    # sensitivity.py for its validity self-check).
    wdm_psd_method: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("WDM_PSD_METHOD", None, str)
    )

    # coarse_Q / coarse_use_ws / coarse_fiducial (+ the coarse_gpu_* knobs)
    # are inherited from EreborGeneralSettings (lifted 2026-08-28 so
    # all_sources shares them). Noise-only semantics are unchanged: Q=1 is
    # the ordinary fine likelihood; Q>1 replaces the backend with the CPU
    # coarse path, WS weighting frozen at injection/initial fiducial.

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

    # Modulation table epoch (mirrors all_sources): None/"" => 0.0 (table
    # already in the data frame); "data" => anchor the lazy table at the
    # authoritative data_t0 in the engine; a number => that explicit epoch.
    # Required when the tabulated modulation (e.g. modulation_unequal.dat) is
    # on the ABSOLUTE mission clock: a noise-only run with
    # GALFOR_MODULATION_T0=data must anchor it, or the table's absolute span
    # mismatches the data's [0, Tobs] frame (a build-time ValueError).
    galfor_modulation_t0: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_MODULATION_T0", None, str)
    )


class _NoiseFitBase(EreborFit):
    """Shared build behaviour for the noise fits (not registered directly).

    Data: the mojito NOISE brick (``L1ProcessingStep`` on ``["NOISE"]``,
    downsampled onto the run grid) or the synthetic Cholesky draw
    (``SyntheticNoiseProcessingStep``); a bare WDM domain with no time crop;
    the sensitivity kwargs threaded from the psd / galfor / sgwb blocks; the
    psd / galfor branch preps. Subclassed by :class:`NoiseOnlyGlobalFit` and
    the :mod:`.noise_mojito` variants.
    """

    general_settings_class = NoiseGeneralSettings

    # -- foreground modulation (None => stationary) --
    def _resolve_modulation(self, gs: NoiseGeneralSettings):
        """The tabulated foreground modulation, or ``None`` (stationary)."""
        return resolve_galfor_modulation(gs.galfor_modulation_path)

    # -- domain: bare WDM factory, no time crop so the (nch, Nf, Nt) draw
    #    matches the full active grid --
    def make_domain_settings(self, gs, Nf, Nt, wavelet_duration, edge_crop):
        """A bare WDM factory on the data band (no time crop)."""
        return WDMSettings.make_factory(
            Nf=Nf, Nt=Nt, min_freq=gs.min_freq, max_freq=gs.max_freq
        )

    # -- data source: the real NOISE brick (mojito) or the synthetic draw --
    def resolve_data_source(self) -> None:
        """Fall back to synthetic data without a NOISE brick; resolve ``psd_injection``."""
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
        """Mojito: downsample to ``1/dt`` with the default trims; synthetic: nothing."""
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
        """Mojito: ``L1ProcessingStep`` on ``NOISE``; synthetic: ``SyntheticNoiseProcessingStep``.

        Args:
            gs: The build-time general block (mutated in place).

        Raises:
            ValueError: On another ``data_mode``.
        """
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
        """Coarse checks, unequal-arm wiring, and the noise sensitivity kwargs.

        Args:
            gs: The build-time general block (mutated in place).
        """
        validate_coarse_settings(gs, all_source=False)
        # v8 EXACT noise model (env UNEQUAL_ARM=1): swap the equal-arm
        # InstrumentNoise for UnequalArmInstrumentNoise + (optionally) the
        # layer_calibrated WDM PSD method, exactly as all_sources does, via the
        # shared wire_unequal_arm_psd helper so the two paths stay in lockstep.
        # The psd branch block (self.psd) is the fit-level settings object
        # created by add_branch at construction; mutating it here is picked up
        # both by noise_sensitivity_init_kwargs below (the data/likelihood noise
        # model) and by prepare_branch_settings' later deepcopy (the sampled
        # model) -- same object, same as resolve_data_source mutates it. Inert
        # unless UNEQUAL_ARM is set (default False -> standard InstrumentNoise,
        # bit-identical to before).
        psd = getattr(self, "psd", None) if "psd" in self._branch_names else None
        if gs.unequal_arm:
            wire_unequal_arm_psd(gs, psd)
        # Modulation epoch resolution (same logic as all_sources._wire...):
        # "data" defers to the engine, which anchors the lazy table at the
        # authoritative data_t0 after processing; a number is an explicit
        # absolute epoch; None/"" leaves the table in the data frame (0.0).
        extra = dict(self._sgwb_sens_kwargs(gs) or {})
        if gs.wdm_psd_method:
            # Backend-wide policy (instrument, foreground, SGWB together).
            extra["wdm_psd_method"] = gs.wdm_psd_method
        modulation_t0 = 0.0
        raw_t0 = gs.galfor_modulation_t0
        if raw_t0 not in (None, ""):
            if str(raw_t0).strip().lower() == "data":
                extra["galfor_modulation_anchor"] = "data_t0"
            else:
                modulation_t0 = float(raw_t0)
        gs.sensitivity_init_kwargs = noise_sensitivity_init_kwargs(
            gs.sensitivity_init_kwargs,
            tdi_generation=gs.tdi_gen,
            galfor=getattr(self, "galfor", None),
            psd=psd,
            galfor_modulation_path=gs.galfor_modulation_path,
            galfor_modulation_t0=modulation_t0,
            extra=extra,
        )

    # -- per-branch priors / injections --
    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        """Erebor completion, then ``_prepare_<name>`` when defined."""
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _prepare_psd(self, psd: NoisePSDSettings, general_setup: GeneralSetup):
        """PSD prep around ``general.psd_injection``."""
        return prepare_psd_branch(psd, self.general.psd_injection)

    def _prepare_galfor(self, galfor: GalForSettings, general_setup: GeneralSetup):
        """Galfor prep (``prepare_galfor_branch``)."""
        return prepare_galfor_branch(galfor)

    # -- sgwb presence hooks (overridden by NoiseSGWBGlobalFit) --
    def _sgwb_injection(self, gs: NoiseGeneralSettings):
        """SGWB injection for the synthetic draw (``None`` here)."""
        return None

    def _sgwb_sens_kwargs(self, gs: NoiseGeneralSettings) -> dict:
        """SGWB sensitivity kwargs (``{}`` here)."""
        return {}


class NoiseOnlyGlobalFit(_NoiseFitBase):
    """Instrument PSD + galactic-foreground fit (mojito NOISE brick or synthetic WDM noise).

    Branches ``psd`` (:class:`NoisePSDSettings`) and ``galfor``
    (:class:`~lisatools.globalfit.stock.erebor.noise.GalForSettings`).
    Default recipe: stage ``noise_pe`` with ``psd_pe`` and ``galfor_pe``
    (each on its own ladder), or a single ``noise_pe`` move with
    ``joint_noise_move``. Data default ``"mojito"`` (the real NOISE brick,
    instrument noise only, so galfor rails at its floor; drop it with
    ``fit.remove_branch("galfor")``), falling back to ``"synthetic"`` when
    no brick is found. The registry description still says "synthetic"; the
    code default is mojito. Laptop twin: ``noise_only_lite``.

    Attributes:
        option_name: ``"noise_only"``.
        setup_classes: ``{"psd": PSDSetup, "galfor": GalForSetup}``.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            fit = erebor.noise_only(nwalkers=8, data_mode="synthetic",
                                    joint_noise_move=True)
            print(fit.list_moves())          # [pe] noise_pe: noise_pe
            fit.build()
            fit.run()
    """

    option_name = "noise_only"
    description = (
        "Joint WDM noise fit: instrument PSD + hyperbolic-tangent galactic "
        "foreground, on a self-consistent synthetic composite-noise realization."
    )
    general_settings_class = NoiseGeneralSettings
    setup_classes = {"psd": PSDSetup, "galfor": GalForSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        """``psd`` (2 parameters) and ``galfor`` blocks."""
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
        """Stage ``noise_pe``: ``psd_pe`` + ``galfor_pe``, or one ``noise_pe`` (joint)."""
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
    """PSD + galactic foreground + power-law SGWB fit on synthetic WDM noise.

    :class:`NoiseOnlyGlobalFit` plus an ``sgwb`` branch
    (:class:`NoiseSGWBSettings`) and an ``sgwb_pe`` move appended to the
    ``noise_pe`` stage. The SGWB is injected only by the synthetic draw
    (``SGWB_INJECTION``), so use ``data_mode="synthetic"``; the SGWB
    posterior is not validated. Laptop twin: ``noise_sgwb_lite``.

    Attributes:
        option_name: ``"noise_sgwb"``.
        setup_classes: ``psd`` PSDSetup, ``galfor`` GalForSetup, ``sgwb``
            SGWBSetup.

    Example:
        .. code-block:: python

            fit = erebor.noise_sgwb(nwalkers=8, data_mode="synthetic")
            fit.build()
            fit.run()
    """

    option_name = "noise_sgwb"
    description = (
        "Joint WDM noise fit: instrument PSD + galactic foreground + power-law "
        "SGWB, on a self-consistent synthetic composite-noise realization."
    )
    setup_classes = {"psd": PSDSetup, "galfor": GalForSetup, "sgwb": SGWBSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        """The ``noise_only`` blocks plus ``sgwb``."""
        branches = super().default_branches()
        branches["sgwb"] = NoiseSGWBSettings()
        return branches

    def default_recipe(self) -> Recipe:
        """The ``noise_only`` recipe with ``sgwb_pe`` appended to ``noise_pe``."""
        recipe = super().default_recipe()
        recipe.stages[0].moves.append(Move("sgwb_pe", branch="sgwb"))
        return recipe

    def _sgwb_injection(self, gs: NoiseGeneralSettings):
        """``general.sgwb_injection`` for the synthetic draw."""
        return list(gs.sgwb_injection)

    def _sgwb_sens_kwargs(self, gs: NoiseGeneralSettings) -> dict:
        """``{"sgwb_stochastic_fn": ...}`` from the sgwb block (else the general default)."""
        # Read the SGWB template off the branch Settings (symmetry); fall back to
        # the general-level default.
        sgwb = getattr(self, "sgwb", None)
        fn = getattr(sgwb, "stochastic_fn", None) if sgwb is not None else None
        return dict(sgwb_stochastic_fn=fn or gs.sgwb_stochastic_fn)

    def _prepare_sgwb(self, sgwb: NoiseSGWBSettings, general_setup: GeneralSetup):
        """SGWB prep: uniform priors ``log10_A`` in [-16, -9], ``alpha`` in [-1, 2]; injection."""
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
    """Recipe setup: build one independent move pair per noise branch, then materialize.

    The ``setup_function`` of ``noise_only``, ``noise_sgwb``,
    ``noise_mojito`` and ``noise_galfor_mojito``. Builds ``<name>_pe`` /
    ``<name>_search`` with :func:`~lisatools.globalfit.recipe.build_noise_moves`
    for ``psd``, ``galfor`` and ``sgwb`` (or ``noise`` over psd + galfor with
    ``joint_noise_move``), only for branches present and names the recipe
    requests; 5 repeats on CPU, the branch's ``num_prop_repeats`` on GPU.

    Args:
        recipe (Recipe): The fit's recipe.
        engine_info: The engine settings.
        curr: The built current info.
        acs: The rank's ``AnalysisContainerArray``.
        priors (dict): Per-branch priors.
        state (GFState): The starting state.
    """
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
