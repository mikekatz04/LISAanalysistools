"""Stock variant ``all_sources``: every branch, mojito or synthetic data.

THE PRODUCTION VARIANT. The 9mo and 12mo global fits run it: the launchers
``scripts/fstat_proposal/submit_gf_9mo_v9.sh`` and
``scripts/flagship/submit_gf_12mo.sh`` start
``scripts/fstat_proposal/run_combined_staged.py`` under ``mpiexec``, which
clones ``erebor.all_sources(nwalkers=4)`` (``NWALKERS``), sets the source ids
and ``source_types`` from the environment, clears the fixed ``nf`` / ``nt``
grid when ``TOBS_TARGET`` is set (23328000 s / 31104000 s), replaces
``fit.recipe`` with its own staged recipe (``gb_search_seed``,
``source_search``, ``gb_search_1``, ``gb_search_2``, ``full_pe``), then calls
``fit.build()`` and ``fit.run(comm)``. The variant's :func:`setup_recipe`
materializes that recipe at run start. (``full_year_combined`` is a separate,
non-production variant.)

Seven branches: ``gb``, ``vgb``, ``psd``, ``galfor``, ``mbh``, ``emri``,
``sobbh``, plus ``sgwb`` with ``FIT_SGWB=1``. ``general.data_mode`` (env
``DATA_MODE``) selects the data pipeline: ``"mojito"`` (default: the mojito
L1 streams, optionally the pre-summed ``COMBINED`` brick, plus instrument
noise: the real NOISE brick when found, else a synthetic draw),
``"synthetic"`` (all streams built in-process, no
external data), or ``"sangria"`` (the legacy LDC file). The class-default
grid is ``dt = 5`` s, ``Nf = 720``, ``Nt = 2160`` (90 d); in mojito mode an
untouched default grid becomes ``dt = 2.5`` s, ``Nf = 1440`` (still 90 d).

This variant is a **composition of the canonical per-branch setups** — each
branch is the same shared code the specialized variant runs:

* **GB** = the ``gb_no_fg`` setup (``prepare_gb_branch`` + ``setup_gb_moves``:
  chunked-het WDM likelihood with the sig-het in-model engine, catalogue
  true-point seeding in ``GB_MODE=pe``, dynamic ``nleaves_max``), with the GB
  band widened to ``[1e-4, 2.5e-2]`` Hz by default (``GB_MIN_FREQ`` /
  ``GB_MAX_FREQ``; the launchers use ``[5.5e-4, 2.2e-2]``). The driver sets
  ``GB_MODE=search`` (zero-leaf start, RJ births).
* **VGB** = the ``vgb`` setup (``prepare_vgb_branch`` + ``setup_vgb_moves``):
  fixed-dimensional verification binaries, ``vgb_pe`` (+ ``vgb_ridge_gibbs``).
* **PSD / galfor** = the ``noise`` setup (``prepare_psd_branch`` /
  ``prepare_galfor_branch`` + ``noise_sensitivity_init_kwargs`` threading the
  per-branch noise-model choice onto the CompositeSensitivityBackend); each
  branch gets its own ``PSDMove`` (``psd_pe``, ``galfor_pe``). With
  ``UNEQUAL_ARM=1`` (the launchers' setting) the instrument component is
  :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise` fed by the NOISE
  brick's per-link delays; the class default (``0``) keeps the equal-arm
  component.
* **MBH / EMRI / SOBBH** = the shared
  :mod:`~lisatools.globalfit.stock.erebor.source_runtime` setup (identical to
  ``full_year_combined``): engine-side ``SourceSignalGen`` residual
  subtraction, and on a WDM / XYZ run the fast likelihood paths by default
  (``MBH_LIKELIHOOD=auto`` batched scoring, ``EMRI_LIKELIHOOD=auto``
  direct-to-WDM, ``SOBBH_LIKELIHOOD=lookup``); waveform defaults SOBBH
  TDI-on-the-fly, MBH legacy phentax (``fit.mbh.use_tdionfly``,
  ``USE_TDIONFLY``).

Example:
    .. code-block:: python

        from lisatools.globalfit.stock import erebor

        fit = erebor.all_sources(nwalkers=4)
        fit.general.mojito_source_ids = {
            "GB": [0], "VGB": [0], "MBHB": [2, 5], "EMRI": [0], "SOBHB": [0],
        }
        fit.remove_branch("galfor")            # sub whole branches in/out
        fit.pop_move("sobbh_pe")               # sub moves in/out (stage full_pe)
        fit.mbh.use_tdionfly = True            # swap the MBH waveform path
        fit.build()
        fit.run()                              # under mpiexec: fit.run(comm)

        # laptop smoke on in-process data
        fit = erebor.all_sources_lite(data_mode="synthetic")
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from ....engine import GeneralSetup, Settings
from ....moves import Move, MoveBuildContext
from .....utils.device import pin_main_device
from ....recipe import MOJITO_REFERENCE_TIME, Recipe, Stage, build_noise_moves
from ...base import bool_or_str, env_default
from ..common import tdi_generation_info
from ..emri import EMRISetup
from ..fit import EreborFit, EreborGeneralSettings
from ..gb import GBSetup
from ..mbh import MBHSetup
from ..noise import (
    GalForSettings,
    validate_coarse_settings,
    GalForSetup,
    PSDSetup,
    noise_sensitivity_init_kwargs,
    prepare_galfor_branch,
    prepare_psd_branch,
)
from ..sobbh import SOBBHSetup
from ..source_runtime import (
    SourceEMRISettings,
    SourceMBHSettings,
    SourceSOBBHSettings,
    SourceSignalGen,
    build_source_moves,
    find_source_cfg,
    mbh_injection_duration,
    make_emri_injections,
    make_mbh_injections,
    make_sobbh_injections,
    prepare_emri_branch,
    prepare_mbh_branch,
    prepare_sobbh_branch,
    run_domain_spec,
    source_signal_cfg,
)
from ..stochastic import SGWBSetup
from .gb_no_fg import GB_MOJITO_T_REF, GBNoFgGBSettings, prepare_gb_branch, setup_gb_moves
from .vgb import setup_vgb_moves
from ..vgb import VGBSettings, VGBSetup, prepare_vgb_branch
from .noise import (
    GALFOR_INJECTION,
    PSD_INJECTION,
    NoisePSDSettings,
    NoiseSGWBSettings,
)

logger = logging.getLogger(__name__)

# Kept as the historical public name (the mojito catalogue epoch; also the
# legacy constant used for the sangria stream's GB reference).
GB_T_REF = GB_MOJITO_T_REF

# Default synthetic-data start (matches EreborGeneralSettings.synthetic_t_start):
# 10,000 s keeps the TDI2 warm-up look-back inside the orbit span.
_SYNTH_T0_DEFAULT = 10_000.0

_SOURCE_BRANCH_TO_CLASS = {"mbh": "MBHB", "emri": "EMRI", "sobbh": "SOBHB"}


@dataclasses.dataclass
class AllSourcesGBSettings(GBNoFgGBSettings):
    """GB branch block: the ``gb_no_fg`` setup on the FULL data band.

    Identical to :class:`GBNoFgGBSettings` except the GB band spans the whole
    data band, extending the f0 prior floor down to ~0.1 mHz (snapped inward
    to WDM layer boundaries like every gb_no_fg band). Reached as ``fit.gb``.
    Every other field (and its env knob) is documented on
    :class:`~lisatools.globalfit.stock.erebor.variants.gb_no_fg.GBNoFgGBSettings`
    and :class:`~lisatools.globalfit.stock.erebor.gb.GBSettings`.

    Attributes:
        min_freq: GB band floor [Hz] before the inward layer snap. Env
            ``GB_MIN_FREQ``. Default ``1e-4``; launchers ``5.5e-4``.
        max_freq: GB band ceiling [Hz]. Env ``GB_MAX_FREQ``. Default
            ``2.5e-2``; launchers ``2.2e-2``.
    """

    min_freq: float = dataclasses.field(
        default_factory=env_default("GB_MIN_FREQ", 1e-4, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("GB_MAX_FREQ", 2.5e-2, float)
    )


# Per-branch source blocks are the shared source_runtime bases (identical to
# full_year_combined); aliased so the variant's knob classes keep their names.
AllSourcesMBHSettings = SourceMBHSettings
AllSourcesEMRISettings = SourceEMRISettings
AllSourcesSOBBHSettings = SourceSOBBHSettings
# PSD / SGWB blocks are the noise-variant blocks (identical to noise_only /
# noise_sgwb: 2-param instrument PSD, 2-param power-law SGWB).
AllSourcesPSDSettings = NoisePSDSettings
AllSourcesSGWBSettings = NoiseSGWBSettings


@dataclasses.dataclass
class AllSourcesGeneralSettings(EreborGeneralSettings):
    """General block for ``all_sources`` (defaults mirror the legacy file).

    Reached as ``fit.general``. Subclass of
    :class:`~lisatools.globalfit.stock.erebor.fit.EreborGeneralSettings`
    (every inherited field and env knob is documented there); the fields
    below are the ones this variant adds or overrides. Launcher values are
    those of ``submit_gf_9mo_v9.sh`` / ``submit_gf_12mo.sh``.

    Attributes:
        dt: Sample spacing [s]. Default ``5.0``; mojito mode turns an
            untouched ``5.0`` into ``2.5`` (:meth:`AllSourcesGlobalFit.adjust_general`).
        nf: Fixed WDM frequency-layer count. Default ``720`` (``1440`` after
            the mojito flip); ``None`` derives the grid from ``tobs_target``.
            The production driver clears it when ``TOBS_TARGET`` is set.
        nt: Fixed WDM time-layer count. Default ``2160``; cleared likewise.
        min_freq: Data-band floor [Hz]. Env ``MIN_FREQ``. Default ``1e-4``;
            launchers ``2.5e-4``.
        max_freq: Data-band ceiling [Hz]. Env ``MAX_FREQ``. Default ``2.5e-2``;
            launchers ``2.5e-2``.
        window_tukey_alpha: Tukey window parameter. Default ``0.0``
            (rectangular).
        edge_crop_wavelets: WDM time layers cropped at each end. Env
            ``EDGE_CROP_WAVELETS``. Default ``20``; launchers ``60``.
        file_store_dir: Output directory. Env ``FILE_STORE_DIR``. Default
            ``"./gf_output/"``; launchers the run's ``STORE_DIR``.
        base_file_name: Store file stem. Env ``BASE_FILE_NAME``. Default
            ``"global_fit_smoke_test"``; launchers ``gf_prod_9mo`` /
            ``gf_prod_12mo``.
        likelihood_source_only: Drop the noise-normalization term. Env
            ``LIKELIHOOD_SOURCE_ONLY``. Default ``False`` (the psd branch is
            fit; ``run.py`` force-disables it whenever a psd branch exists).
        data_mode: ``"mojito"``, ``"synthetic"`` or ``"sangria"``. Env
            ``DATA_MODE``. Default ``"mojito"`` (launchers unset).
        mojito_source_ids: ``{class: [ids]}`` for ``GB``, ``VGB``, ``MBHB``,
            ``EMRI``, ``SOBHB``; gates which sources load (GB / VGB load the
            whole catalogue file, the list only gates inclusion), filtered by
            the branches present; in synthetic mode the list lengths are the
            injection counts. Default ``{"GB": [0], "VGB": [0], "MBHB": [0],
            "EMRI": [1], "SOBHB": [0]}``; the driver sets ``MBHB`` / ``EMRI``
            / ``SOBHB`` from ``MBHB_IDS`` / ``EMRI_IDS`` / ``SOBHB_IDS``.
        synthesize_missing_bricks: Mojito mode: build a missing per-source
            MBHB / EMRI / SOBHB stream from its catalogue parameters instead
            of failing. Env ``SYNTHESIZE_MISSING_BRICKS``. Default ``False``.
        add_instrument_noise: ``True`` (default) resolves to ``"mojito"``
            (the NOISE brick, mojito data and brick found) or
            ``"synthetic"`` (an FD-correlated draw); ``"mojito"`` /
            ``"synthetic"`` force one; ``False`` adds none. Env
            ``ADD_INSTRUMENT_NOISE``.
        noise_soms_d: Instrument ``Soms_d`` level (sqrt units). ``None``
            (default) -> fit to the NOISE brick's tabulated estimates, else
            :data:`~lisatools.globalfit.stock.erebor.variants.noise.PSD_INJECTION`.
        noise_sa_a: Instrument ``Sa_a`` level (sqrt units), same rule.
        noise_seed: Seed of the synthetic noise draw. Default ``12345``.
        t_start: Synthetic-data start [s]; kept in sync with
            ``synthetic_t_start``. Default ``10000.0``.
        gb_injection_params: Synthetic mode: GB rows ``(N, 9)`` in the GBGPU
            basis ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``.
            ``None`` -> the stock two-source table.
        psd_injection: ``[Soms_d, Sa_a]`` reference / start of the psd branch.
            ``None`` -> ``[noise_soms_d, noise_sa_a]`` after resolution.
        galfor_injection: Foreground reference parameters (informational in
            mojito mode, where the foreground is the GB galaxy itself).
            Default :data:`~lisatools.globalfit.stock.erebor.variants.noise.GALFOR_INJECTION`.
        galfor_modulation_path: Tabulated foreground time-modulation file.
            Env ``GALFOR_MODULATION_PATH``. Default ``"auto"``: on mojito data
            with a ``galfor`` branch, the packaged
            :data:`~lisatools.sensitivity.GALFOR_MODULATION_TABLE` with the
            epoch defaulting to ``"data"``; stationary otherwise. ``None`` /
            ``"none"`` -> stationary. Launchers pass the packaged table
            ``src/lisatools/galfor_modulation/modulation_unequal.dat``.
        galfor_modulation_t0: Epoch of the modulation table: ``None`` (table
            already relative to the data start; ``"data"`` for the ``"auto"``
            table), ``"data"`` (anchor at the data's first sample), or an
            absolute epoch [s]. Env ``GALFOR_MODULATION_T0``. Default
            ``None``; launchers ``data``.
        coarse_Q: Coarse WDM noise-likelihood factor (``1`` = off). Env
            ``COARSE_Q``. Default ``8``; launchers ``1``.
        coarse_gpu_mode: Coarse-likelihood mode. Env ``COARSE_GPU_MODE``.
            Default ``"delayed_acceptance"`` (exact two-stage); launchers
            ``off``.
        unequal_arm: Use :class:`~lisatools.sensitivity.UnequalArmInstrumentNoise`
            with the NOISE brick's six link delays (mojito only). Env
            ``UNEQUAL_ARM``. Default ``False``; launchers ``1``.
        unequal_arm_stride: Decimation of the 2.5 s per-link delay table.
            Env ``UNEQUAL_ARM_STRIDE``. Default ``200``; launchers ``200``.
        wdm_psd_method: Backend-wide WDM PSD evaluation: ``None`` (the
            backend's exact ``"fold"``), ``"layer_constant"`` or
            ``"layer_calibrated"``. Env ``WDM_PSD_METHOD``. Default ``None``;
            launchers ``layer_calibrated``.
        fit_sgwb: Add a power-law ``sgwb`` branch fit with psd / galfor. Env
            ``FIT_SGWB``. Default ``False`` (the stock data processors inject
            no SGWB).
        sgwb_injection: SGWB reference ``[log10_A, alpha]``. Default
            ``[-9.5, 2/3]``.
        sgwb_stochastic_fn: SGWB spectral model name. Default
            ``"PowerLawSGWB"``.
    """

    dt: float = 5.0
    nf: typing.Optional[int] = 720
    nt: typing.Optional[int] = 2160
    # env-wired (2026-09-06, THIRD shadow of the same bug -- see the
    # EDGE_CROP_WAVELETS note just below, and nf/nt before it). A plain
    # ``= 1e-4`` here SHADOWED the base EreborGeneralSettings env factory, so
    # the MIN_FREQ=4e-4 that every production submit script exports has NEVER
    # taken effect on this path. Consequence measured on the 3mo v8 store
    # (kappa probe, job 459): min_freq=1e-4 -> ind_min_f=1, putting WDM layer 1
    # (support 0.069-0.208 mHz, sharing an edge with DC) inside the analysed
    # band. Its covariance is singular there -- det(C)=0, because the analytic
    # model diverges as f->0 and instrument_fill_nans=0.0 turns the NaN into a
    # hard zero -- and it carries q = w^T C^-1 w / 3 = 150.3, i.e. **43% of the
    # entire chi^2 budget of the fit**. The ML then returns
    # alpha = sqrt(mean(q/3)) = 1.389, which IS the observed S_oms bias
    # (1.3874 measured, 0.21% agreement). MIN_FREQ=4e-4 -> layer 3 predicts
    # alpha -> 1.049.
    min_freq: float = dataclasses.field(
        default_factory=env_default("MIN_FREQ", 1e-4, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("MAX_FREQ", 2.5e-2, float)
    )
    window_tukey_alpha: float = 0.0  # rectangular window
    # env-wired (2026-08-19, second shadow of the same bug fixed in
    # full_year_combined 7d537d32): a plain ``= 20`` here shadowed the base
    # EDGE_CROP_WAVELETS env factory, so the submit-script pins were ignored
    # on THIS path too -- and run_combined_staged builds through
    # AllSourcesGeneralSettings, so the 1-yr launch hit the sig-het
    # edge-exclusion guard twice.
    edge_crop_wavelets: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("EDGE_CROP_WAVELETS", 20, int)
    )
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "global_fit_smoke_test")
    )
    # The psd branch is FIT here, so keep the noise-normalization
    # ``-logdet_factor * sum log det C`` term in the likelihood.
    # Env-backed (rule 0) for the compositions that REMOVE the psd branch:
    # the truth-injection null test (REMOVE_BRANCHES=...,psd) runs on a FIXED
    # sensitivity, where the noise term is an additive constant and the user
    # wants the readout to be the residual inner product alone -- "our
    # injection log-likelihood WITHOUT the noise term" (ruling 2026-09-14).
    # run.py force-disables it whenever a psd branch IS present, so a stray
    # LIKELIHOOD_SOURCE_ONLY=1 on a psd-sampling run cannot corrupt the
    # acceptance ratios; it only warns.
    likelihood_source_only: bool = dataclasses.field(
        default_factory=env_default("LIKELIHOOD_SOURCE_ONLY", False, bool)
    )
    # Data source: "mojito" (default) loads the GB galaxy + selected
    # MBHB/EMRI/SOBHB sources from a mojito L1 folder (+ synthetic
    # instrument noise for the psd branch); "synthetic" builds every
    # present branch's stream in-process (no external data needed);
    # "sangria" (legacy) loads the LDC training file.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_MODE", "mojito", str)
    )
    # Mojito-mode source selection per class (GB loads the whole galaxy
    # file; the id list just gates inclusion). Filtered by the branches
    # actually present on the fit. In synthetic mode the per-class COUNTS
    # set how many stock injections are built.
    mojito_source_ids: dict = dataclasses.field(
        default_factory=lambda: {
            "GB": [0], "VGB": [0], "MBHB": [0], "EMRI": [1], "SOBHB": [0]
        }
    )
    # HYBRID mojito (2026-09-14): when a per-source MBHB/EMRI/SOBHB L1 brick
    # is missing from the mojito folder (transfer incomplete), synthesize
    # that source's stream in-process from its CATALOGUE parameters (same
    # converters as the branch preps, mojito epochs, the run's real orbits)
    # instead of failing. Default off: a missing brick stays a loud error.
    synthesize_missing_bricks: bool = dataclasses.field(
        default_factory=env_default("SYNTHESIZE_MISSING_BRICKS", False, bool)
    )
    # Mojito L1 signals are noiseless: add instrument noise so the psd branch
    # has something to fit (the loaded GB galaxy itself plays the role of the
    # foreground for the galfor branch). True (default) auto-resolves to
    # "mojito" — the real NOISE brick summed in by the L1 loader — when the
    # run uses mojito data and the brick is found, else "synthetic" (the
    # FD-correlated draw). Explicit "mojito"/"synthetic"/False force it.
    # Env-backed (rule 0): ADD_INSTRUMENT_NOISE=0 is what the truth-injection
    # null test needs -- "NO INJECTED NOISE" (ruling 2026-09-14). With no psd
    # branch there is nothing to fit a noise realization with, so injecting
    # one would leave it unmodelled in the residual and the null would not
    # reach zero. A boolean spelling gives the bool; "mojito"/"synthetic"
    # pass through as the named mode.
    add_instrument_noise: typing.Union[bool, str] = dataclasses.field(
        default_factory=env_default(
            "ADD_INSTRUMENT_NOISE", True, bool_or_str
        )
    )
    # None -> auto at build: [Soms_d, Sa_a] fit to the NOISE brick's
    # tabulated estimates when available, else the stock 15e-12 / 3e-15.
    # Feed the synthetic draw (and the psd-branch reference) so data and
    # model stay consistent whichever noise source resolves.
    noise_soms_d: typing.Optional[float] = None
    noise_sa_a: typing.Optional[float] = None
    noise_seed: int = 12345
    # Synthetic-data start time. Default 10,000 s so the TDI2 warm-up
    # look-back stays inside the orbit span (which starts at t=0); the
    # legacy Sangria stream itself starts at 0 (see ``mbh_waveform_t0``).
    t_start: float = _SYNTH_T0_DEFAULT
    # data_mode="synthetic": GB rows in the GBGPU basis
    # ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``; None -> the
    # stock two-source table.
    gb_injection_params: typing.Optional[typing.Any] = None

    # --- noise-branch knobs (identical to the noise variants) ---
    # PSD injection ``[Soms_d, Sa_a]`` consumed by prepare_psd_branch.
    # None -> auto: the resolved (noise_soms_d, noise_sa_a) — i.e. the NOISE
    # brick fit when available, else the stock PSD_INJECTION levels.
    psd_injection: typing.Optional[typing.Sequence[float]] = None
    # Galactic-foreground truth reference (informational; the foreground in
    # the DATA is the GB galaxy itself in mojito mode).
    galfor_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: list(GALFOR_INJECTION)
    )
    # Foreground time modulation: "auto" (default) => the packaged
    # modulation_unequal.dat anchored at the data start when the data are
    # mojito and a galfor branch is fit, stationary otherwise; None / "none"
    # => stationary; a path (or GALFOR_MODULATION_PATH) => that table.
    galfor_modulation_path: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_MODULATION_PATH", "auto", str)
    )
    # Epoch of the modulation table's time column: None (default) keeps a
    # table already written relative to the data start unchanged (the "auto"
    # table resolves it to "data"); the string "data" anchors the table at the
    # data's first sample time (mission-clock tables such as
    # modulation_unequal.dat), resolved once the data has been processed; a
    # float is an explicit absolute epoch [s].
    galfor_modulation_t0: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("GALFOR_MODULATION_T0", None, str)
    )

    # --- coarse real-WDM noise likelihood: ON BY DEFAULT here (user ruling
    #     2026-08-28, "make the coarse the default ... faster PSD fitting is
    #     always good"). Overrides the EreborGeneralSettings defaults, which
    #     stay off/1 -- the noise-only variants REFUSE a coarse mode (they use
    #     the CPU backend-replacement path), so this must not move on the
    #     shared base.
    #
    #     Mode is `delayed_acceptance` in EVERY stage (user ruling
    #     2026-08-28, accuracy over speed: "make sure accuracy is right since
    #     it is really a small part of the time"). DA is EXACT -- stage 1
    #     screens on the coarse surrogate, stage 2 corrects with the exact
    #     fine/coarse ratio -- so the sampled target is the fine likelihood
    #     in every stage, whatever the surrogate's quality. `search_approx`
    #     (available via COARSE_GPU_MODE, and stage-dependent via `auto`) is
    #     the ONLY approximate setting here: it optimises the surrogate and
    #     is faster, but it is not a posterior sampler and, in the job-376
    #     probe, the search it drove railed galfor against two prior edges.
    #     Since the noise block is a small share of the wall clock, the
    #     approximation buys little and is not worth the ambiguity.
    #     Accuracy is OBSERVABLE in production: DA computes both likelihoods
    #     for every stage-1 survivor, so each propose logs a [COARSE_AUDIT]
    #     line whose |dlogl| spread and stage-2 acceptance are exactly the
    #     coarse-vs-fine Delta-logL error scripts/noise/coarse_q_scan.py
    #     measures offline. 0 / 100% would mean an exact surrogate.
    #     Every noise move republishes the exact FINE covariance before it
    #     returns, in BOTH modes, so source moves never see coarse state.
    #
    #     COARSE_Q=8 matches the probe leg so the configuration that gets
    #     measured is the configuration that ships; scripts/noise/coarse_q_scan.py
    #     tunes it (bigger Q = cheaper surrogate but lower stage-1 survival,
    #     which costs efficiency, never correctness).
    #     Off switch: COARSE_GPU_MODE=off (or COARSE_Q=1).
    coarse_Q: int = dataclasses.field(default_factory=env_default("COARSE_Q", 8, int))
    coarse_gpu_mode: str = dataclasses.field(
        default_factory=env_default("COARSE_GPU_MODE", "delayed_acceptance", str)
    )

    # --- unequal-arm instrument noise (env UNEQUAL_ARM=1) ---
    # Swap the equal-arm InstrumentNoise for UnequalArmInstrumentNoise, the
    # six link light travel times read from the mojito NOISE brick's /ltts
    # group and averaged per WDM time column (LinkDelayTable). Mojito data
    # only: synthetic/sangria carry no delay table and are refused loudly.
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

    # --- opt-in SGWB branch (env FIT_SGWB=1) ---
    # Default off keeps the six-branch behaviour unchanged; when on, a
    # power-law SGWB is fit jointly with psd+galfor through the same PSDMove.
    # Note: the stock data processors do not yet inject an SGWB, so enable
    # this only with data that contains one (or extend the processor).
    fit_sgwb: bool = dataclasses.field(
        default_factory=env_default("FIT_SGWB", False, bool)
    )
    sgwb_injection: typing.Sequence[float] = dataclasses.field(
        default_factory=lambda: [-9.5, 2.0 / 3.0]
    )
    sgwb_stochastic_fn: str = "PowerLawSGWB"

    # --- properties read by the shared source_runtime glue (duck-typed) ---
    # TODO check: keep/remove/document -- no reader (source_runtime does not read it)
    @property
    def active_source(self) -> typing.Optional[str]:
        """The single non-GB class with ids, or ``None`` when zero or several have."""
        active = [
            k for k, v in self.mojito_source_ids.items() if v and k != "GB"
        ]
        return active[0] if len(active) == 1 else None

    @property
    def n_injections(self) -> typing.Dict[str, int]:
        """``{class: len(ids)}`` of :attr:`mojito_source_ids` (read by the source branch preps)."""
        return {k: len(v) for k, v in self.mojito_source_ids.items()}

    @property
    def mbh_waveform_t0(self) -> float:
        """Epoch [s] merger times (t_plunge) are referenced to.

        :data:`~lisatools.globalfit.recipe.MOJITO_REFERENCE_TIME` in mojito
        mode, ``0.0`` for sangria, ``t_start`` for synthetic data.
        """
        if self.data_mode == "mojito":
            return MOJITO_REFERENCE_TIME
        if self.data_mode == "sangria":
            return 0.0  # the legacy Sangria stream starts at 0
        return float(self.t_start)

    @property
    def sobbh_reference_time(self) -> typing.Optional[float]:
        """Epoch [s] the SOBBH ``f_low`` is defined at.

        :data:`~lisatools.globalfit.recipe.MOJITO_REFERENCE_TIME` in mojito
        mode; ``None`` (the window start) otherwise.
        """
        return MOJITO_REFERENCE_TIME if self.data_mode == "mojito" else None


class AllSourcesGlobalFit(EreborFit):
    """All-source global fit (gb, vgb, psd, galfor, mbh, emri, sobbh), mojito default.

    The production variant of the 9mo / 12mo runs (see the module
    docstring for how the driver ``run_combined_staged.py`` uses it). Users
    get it as ``erebor.all_sources(...)`` (a clone of the module default) or
    ``erebor.get_stock("all_sources", ...)``; the laptop twin is
    ``all_sources_lite`` (``lite=True``). Constructor knobs are those of
    :class:`~lisatools.globalfit.stock.base.StockGlobalFit` (headline knobs
    plus any field of :class:`AllSourcesGeneralSettings`).

    Branch blocks (``fit.<name>``): ``gb``
    :class:`AllSourcesGBSettings`; ``vgb``
    :class:`~lisatools.globalfit.stock.erebor.vgb.VGBSettings`; ``psd``
    :class:`~lisatools.globalfit.stock.erebor.variants.noise.NoisePSDSettings`
    (2 parameters ``Soms_d``, ``Sa_a``); ``galfor``
    :class:`~lisatools.globalfit.stock.erebor.noise.GalForSettings`; ``mbh`` /
    ``emri`` / ``sobbh`` the
    :class:`~lisatools.globalfit.stock.erebor.source_runtime.SourceMBHSettings`
    family; ``sgwb``
    :class:`~lisatools.globalfit.stock.erebor.variants.noise.NoiseSGWBSettings`
    only with ``fit_sgwb``.

    Default recipe (the driver replaces it in production): stage ``full_pe``
    (kind ``pe``) with ``psd_pe``, ``galfor_pe`` (``sgwb_pe``), ``sobbh_pe``,
    ``mbh_pe``, ``emri_pe``, ``rj_fstat_pe``; stage ``vgb_pe`` with
    ``vgb_pe``. Both stages use ``share_temperature_control=False``.

    Build hooks: :meth:`resolve_data_source` (noise source and levels),
    :meth:`adjust_general` (mojito grid flip), :meth:`set_default_processor`
    (mojito / sangria / synthetic processors), :meth:`finalize_general`
    (coarse checks, unequal-arm wiring, sensitivity kwargs),
    :meth:`prepare_branch_settings` (per-branch ``_prepare_<name>``),
    :meth:`attach_runtime_objects` (source ``signal_gen``).

    Attributes:
        option_name: ``"all_sources"``.
        setup_classes: ``gb`` GBSetup, ``vgb`` VGBSetup, ``psd`` PSDSetup,
            ``galfor`` GalForSetup, ``mbh`` MBHSetup, ``emri`` EMRISetup,
            ``sobbh`` SOBBHSetup, ``sgwb`` SGWBSetup.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            fit = erebor.get_stock("all_sources", nwalkers=4, num_iterations=50)
            print(fit.list_moves())          # "[pe] full_pe:" then psd_pe, galfor_pe, ...
            fit.add_move("rj_prior_pe", stage="full_pe", branch="gb")
            fit.build()
            fit.run()

        .. code-block:: bash

            # what the production launchers run (-n $SLURM_NTASKS: 6 ranks at
            # NGPUS=4 = head, 3 compute, saver, monitor)
            mpiexec -n 6 python scripts/fstat_proposal/run_combined_staged.py
    """

    option_name = "all_sources"
    description = (
        "All six branches (gb, psd, galfor, mbh, emri, sobbh) composed from "
        "the canonical per-branch setups (gb_no_fg GB down to 0.1 mHz, noise "
        "PSD/galfor, full_year phentax MBH + EMRI + SOBBH) on mojito L1 "
        "(default), synthetic, or legacy Sangria data."
    )
    general_settings_class = AllSourcesGeneralSettings
    setup_classes = {
        "gb": GBSetup,
        "vgb": VGBSetup,
        "psd": PSDSetup,
        "galfor": GalForSetup,
        "mbh": MBHSetup,
        "emri": EMRISetup,
        "sobbh": SOBBHSetup,
        "sgwb": SGWBSetup,
    }

    def resolve_data_source(self) -> None:
        """Resolve the data source, then the instrument-noise source and levels.

        After the base resolution, on the FIT-LEVEL ``self.general``:
        ``add_instrument_noise`` becomes the concrete source
        (:meth:`~lisatools.globalfit.stock.erebor.fit.EreborFit.resolve_noise_source`),
        unset ``noise_soms_d`` / ``noise_sa_a`` are filled from the NOISE
        brick fit (else ``PSD_INJECTION``), and an unset ``psd_injection``
        becomes ``[noise_soms_d, noise_sa_a]``.
        """
        super().resolve_data_source()
        # Resolve the noise source + levels on the FIT-LEVEL block (like the
        # base data-source resolution): prepare_psd_branch reads the fit-level
        # ``psd_injection``, so the resolved values must live there too.
        gs = self.general
        gs.add_instrument_noise = self.resolve_noise_source(gs)
        if gs.noise_soms_d is None or gs.noise_sa_a is None:
            file_params = self.resolve_noise_file_psd_params(gs)
            soms_d, sa_a = (
                file_params if file_params is not None else list(PSD_INJECTION)
            )
            if gs.noise_soms_d is None:
                gs.noise_soms_d = soms_d
            if gs.noise_sa_a is None:
                gs.noise_sa_a = sa_a
        if gs.psd_injection is None:
            gs.psd_injection = [gs.noise_soms_d, gs.noise_sa_a]

    def adjust_general(self, gs: AllSourcesGeneralSettings) -> None:
        """Variant hook on the build-time copy: mojito grid flip and ``t_start`` sync.

        In mojito mode ``dt = 5`` becomes ``2.5`` and, if the grid is still
        the default ``720 x 2160``, ``nf`` becomes ``1440`` (same 90 d span).
        ``t_start`` and ``synthetic_t_start`` are synced (a non-default
        value wins). A set ``gb.data_band_layers`` is ignored with a warning.

        Args:
            gs: The build-time general block (mutated in place).
        """
        # Mojito L1 data is sampled at dt = 2.5 s (the class defaults are the
        # Sangria-legacy dt = 5 grid). When the user left the grid at its
        # defaults, keep the 90-d span and ~1-h wavelets by doubling Nf.
        if gs.data_mode == "mojito" and gs.dt == 5.0:
            gs.dt = 2.5
            if gs.nf == 720 and gs.nt == 2160:
                gs.nf = 1440
        # Keep the legacy ``t_start`` knob and the inherited
        # ``synthetic_t_start`` in sync (either may be set; a value that
        # differs from the shared default is the user's).
        if gs.t_start != _SYNTH_T0_DEFAULT:
            gs.synthetic_t_start = gs.t_start
        elif gs.synthetic_t_start != _SYNTH_T0_DEFAULT:
            gs.t_start = gs.synthetic_t_start
        # gb_no_fg's ACA data-band clipping is a single-band memory knob; in
        # this multi-source variant the data band must cover every branch.
        gb = getattr(self, "gb", None) if "gb" in self._branch_names else None
        if gb is not None and getattr(gb, "data_band_layers", None) is not None:
            logger.warning(
                "all_sources ignores gb.data_band_layers (the data band must "
                "cover the MBH/EMRI/SOBBH branches); narrowing the GB band "
                "itself is gb.min_freq/max_freq."
            )

    def default_branches(self) -> typing.Dict[str, Settings]:
        """Fresh ``gb``, ``vgb``, ``psd``, ``galfor``, ``mbh``, ``emri``, ``sobbh`` blocks.

        Plus ``sgwb`` when ``general.fit_sgwb`` is set.
        """
        branches = {
            "gb": AllSourcesGBSettings(),
            "vgb": VGBSettings(),
            "psd": AllSourcesPSDSettings(),
            "galfor": GalForSettings(),
            "mbh": AllSourcesMBHSettings(),
            "emri": AllSourcesEMRISettings(),
            "sobbh": AllSourcesSOBBHSettings(),
        }
        if self.general.fit_sgwb:
            branches["sgwb"] = AllSourcesSGWBSettings()
        return branches

    def default_recipe(self) -> Recipe:
        """Stages ``full_pe`` (noise, sobbh, mbh, emri, ``rj_fstat_pe``) and ``vgb_pe``.

        Not what production runs: the driver replaces ``fit.recipe`` with its
        staged search + PE recipe before the build.
        """
        # One combined PE stage: noise moves first (split: psd / galfor /
        # sgwb each under their own ladder), then sobbh BEFORE mbh/emri
        # (user ruling 2026-08-27: the cheap branch banks its progress
        # before the minutes-per-leaf MBH/EMRI proposals put the iteration
        # at risk on a preemption-prone partition; move order is not a
        # sampling statement), then gb.
        noise_moves = [
            Move("psd_pe", branch="psd"),
            Move("galfor_pe", branch="galfor"),
        ]
        if self.general.fit_sgwb:
            noise_moves.append(Move("sgwb_pe", branch="sgwb"))
        return Recipe(
            [
                Stage(
                    name="full_pe",
                    kind="pe",
                    moves=noise_moves
                    + [
                        Move("sobbh_pe", branch="sobbh"),
                        Move("mbh_pe", branch="mbh"),
                        Move("emri_pe", branch="emri"),
                        Move("rj_fstat_pe", branch="gb"),
                    ],
                    combine_kwargs=dict(share_temperature_control=False),
                ),
                # Verification binaries run in their own dedicated stage
                # (fixed-dimensional 5D branch, same-leaf stretch, no RJ).
                Stage(
                    name="vgb_pe",
                    kind="pe",
                    moves=[Move("vgb_pe", branch="vgb")],
                    combine_kwargs=dict(share_temperature_control=False),
                ),
            ]
        )

    # -- data processors ------------------------------------------------------

    def set_default_processor(self, gs: AllSourcesGeneralSettings) -> None:
        """Pick the data processor for ``gs.data_mode`` (skipped when one is set).

        - ``"mojito"`` (production):
          :class:`~lisatools.globalfit.stock.erebor.injections.L1ProcessingStepWithSyntheticNoise`
          on ``mojito_data_path`` with the present branches' source ids, the
          instrument-noise options, ``Tobs`` and the MBH phenom kwargs (for a
          synthesized missing brick). When ``source_types`` lists
          ``COMBINED`` the list is passed through verbatim (the pre-summed
          stream).
        - ``"sangria"``: :class:`~lisatools.globalfit.preprocessing.SangriaProcessingStep`
          on ``ldc_source_file``.
        - ``"synthetic"``:
          :class:`~lisatools.globalfit.stock.erebor.injections.SyntheticCombinedProcessingStep`
          summing a GB stream and a source + noise stream built with the same
          generators as the branch templates.

        Args:
            gs: The build-time general block (mutated in place).

        Raises:
            ValueError: On an unknown ``data_mode``, or when no present
                branch can be loaded / injected.
        """
        if gs.data_mode == "mojito":
            from ..injections import L1ProcessingStepWithSyntheticNoise

            force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
            branch_to_class = {
                "gb": "GB", "vgb": "VGB", "mbh": "MBHB", "emri": "EMRI",
                "sobbh": "SOBHB",
            }
            source_ids = {
                cls: list(gs.mojito_source_ids.get(cls, []))
                for branch, cls in branch_to_class.items()
                if branch in self._branch_names and gs.mojito_source_ids.get(cls)
            }
            if not source_ids:
                raise ValueError(
                    "all_sources data_mode='mojito' has no loadable classes "
                    "(check mojito_source_ids vs the present branches)."
                )
            # HYBRID missing-brick fill needs the MBH phenom kwargs so a
            # synthesized MBH stream matches the branch template exactly.
            _mbh = (
                self.mbh
                if "mbh" in self._branch_names
                else AllSourcesMBHSettings()
            )
            # COMBINED stream (user ruling 2026-09-14): when the run's
            # source_types lists COMBINED, pass the list through VERBATIM
            # -- the processor otherwise derives NOISE+classes from
            # source_ids and the pre-summed stream never reaches the
            # loader (the 6mo first-launch EMRI brick FileNotFoundError).
            _types = [str(s).upper() for s in (gs.source_types or ())]
            _combined_kw = (
                {"source_types": _types} if "COMBINED" in _types else {}
            )
            gs.data_processor_class = L1ProcessingStepWithSyntheticNoise
            gs.processor_init_kwargs = dict(
                L1_folder=gs.mojito_data_path,
                source_ids=source_ids,
                **_combined_kw,
                orbits_kwargs=dict(
                    force_backend=force_backend, frame=gs.orbits_frame
                ),
                verbose=gs.verbose,
                do_plots=False,
                Tobs=gs.Tobs,
                add_instrument_noise=gs.add_instrument_noise,
                noise_soms_d=gs.noise_soms_d,
                noise_sa_a=gs.noise_sa_a,
                noise_seed=gs.noise_seed,
                tdi_generation=tdi_generation_info(gs.tdi_chan)[0],
                synthesize_missing=gs.synthesize_missing_bricks,
                tdi_chan=gs.tdi_chan,
                tdi_gen_str=gs.tdi_gen_str,
                synth_force_backend=force_backend,
                mbh_phenom_kwargs=dict(
                    waveform_duration=mbh_injection_duration(
                        _mbh, domain_settings=gs.domain_settings
                    ),
                    higher_modes=_mbh.higher_modes,
                    phenom_tol=_mbh.phenom_tol,
                    start_freq=_mbh.start_freq,
                    response_order=_mbh.response_order,
                    buffer_time=_mbh.buffer_time,
                    min_freq=gs.min_freq,
                    max_freq=gs.max_freq,
                ),
            )
            return
        if gs.data_mode == "sangria":
            from lisatools.globalfit.preprocessing import SangriaProcessingStep

            gs.data_processor_class = SangriaProcessingStep
            gs.processor_init_kwargs = dict(
                data_input_path=gs.ldc_source_file, verbose=gs.verbose
            )
            return
        if gs.data_mode == "synthetic":
            # In-process streams for the PRESENT branches, summed by
            # SyntheticCombinedProcessingStep: GB via the same generator as
            # gb_no_fg; MBH/EMRI/SOBBH (+ instrument noise for the psd
            # branch) via full_year's SyntheticDataProcessor, which uses the
            # SAME stock injections/generators as the branch templates so
            # the residual nulls at the injected start.
            from ..injections import (
                GB_INJECTION_PARAMS,
                SyntheticCombinedProcessingStep,
                SyntheticDataProcessor,
                SyntheticGBProcessingStep,
            )

            use_tdi2 = tdi_generation_info(gs.tdi_chan)[0] == 2
            # Data production follows the run backend (same rule as the
            # mojito path above). Production transients are reclaimed once
            # at the build->sampling transition (see run.py::setup_acs).
            force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
            specs = []
            if "gb" in self._branch_names:
                params = (
                    gs.gb_injection_params
                    if gs.gb_injection_params is not None
                    else GB_INJECTION_PARAMS
                )
                specs.append(
                    (
                        SyntheticGBProcessingStep,
                        dict(
                            injection_params=np.atleast_2d(
                                np.asarray(params, dtype=float)
                            ),
                            tdi_chan=gs.tdi_chan,
                            use_tdi2=use_tdi2,
                            force_backend=force_backend,
                        ),
                    )
                )
            source_branches = [
                b for b in _SOURCE_BRANCH_TO_CLASS if b in self._branch_names
            ]
            if source_branches:
                # Injection counts gated by branch presence (a removed branch
                # injects nothing).
                n_inj = {
                    cls: (
                        gs.n_injections.get(cls, 0)
                        if branch in self._branch_names
                        else 0
                    )
                    for branch, cls in _SOURCE_BRANCH_TO_CLASS.items()
                }
                source_ids = {
                    cls: (
                        list(gs.mojito_source_ids.get(cls, []))
                        if branch in self._branch_names
                        else []
                    )
                    for branch, cls in _SOURCE_BRANCH_TO_CLASS.items()
                }
                mbh = (
                    self.mbh
                    if "mbh" in self._branch_names
                    else AllSourcesMBHSettings()
                )
                # Same (mode, seed) as the branch prep -> identical tables.
                inj_mode = gs.synthetic_injections or "stock"
                inj_seed = gs.synthetic_injection_seed
                specs.append(
                    (
                        SyntheticDataProcessor,
                        dict(
                            emri_injection_params_full_basis=make_emri_injections(
                                n_inj["EMRI"], mode=inj_mode, seed=inj_seed
                            ),
                            sobbh_injection_params_full_basis=make_sobbh_injections(
                                n_inj["SOBHB"], mode=inj_mode, seed=inj_seed
                            ),
                            mbh_injection_params_sampling_basis=make_mbh_injections(
                                n_inj["MBHB"], gs.Tobs, mode=inj_mode, seed=inj_seed
                            ),
                            source_ids=source_ids,
                            force_backend=force_backend,
                            tdi_chan=gs.tdi_chan,
                            tdi_gen_str=gs.tdi_gen_str,
                            sobbh_reference_time=gs.sobbh_reference_time,
                            mbh_phenom_kwargs=dict(
                                waveform_duration=mbh_injection_duration(
                                    mbh, domain_settings=gs.domain_settings
                                ),
                                higher_modes=mbh.higher_modes,
                                phenom_tol=mbh.phenom_tol,
                                start_freq=mbh.start_freq,
                                response_order=mbh.response_order,
                                buffer_time=mbh.buffer_time,
                                min_freq=gs.min_freq,
                                max_freq=gs.max_freq,
                            ),
                            add_instrument_noise=gs.add_instrument_noise,
                            noise_soms_d=gs.noise_soms_d,
                            noise_sa_a=gs.noise_sa_a,
                            noise_seed=gs.noise_seed,
                            tdi_generation=gs.tdi_gen,
                        ),
                    )
                )
            if not specs:
                raise ValueError(
                    "all_sources data_mode='synthetic' has no injectable "
                    "branches (gb/mbh/emri/sobbh all removed)."
                )
            gs.data_processor_class = SyntheticCombinedProcessingStep
            gs.processor_init_kwargs = dict(
                Tobs=gs.Tobs,
                dt=gs.dt,
                t_start=gs.t_start,
                processor_specs=specs,
                nchannels=gs.nchannels,
                verbose=gs.verbose,
            )
            return
        raise ValueError(
            f"all_sources data_mode={gs.data_mode!r} not recognised; use "
            "'mojito', 'synthetic', or 'sangria' (or swap "
            "data_processor_class wholesale)."
        )

    def default_preprocess_kwargs(self) -> dict:
        """No highpass / trim for mojito and synthetic data; the engine defaults for sangria."""
        if self.general.data_mode in ("mojito", "synthetic"):
            # Both emit exactly Tobs = Nf*Nt*dt samples (the mojito loader
            # trims at load via its Tobs kwarg); skip the engine's default
            # highpass + edge-trim + Tobs trim.
            return dict(
                highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False
            )
        # Sangria is a year-long stream: keep the engine's default
        # highpass + edge-trim + Tobs trim so the final data is exactly
        # Nf*Nt samples (the window is built on the processed grid).
        return dict(normalize=False)

    # -- sensitivity backend: thread the per-branch noise-MODEL choice, same
    #    as the noise variants (fit.galfor.stochastic_fn / .modulation /
    #    fit.psd.instrument_model_cls / fit.sgwb.stochastic_fn) --

    def finalize_general(self, gs: AllSourcesGeneralSettings) -> None:
        """Last variant hook: coarse checks, unequal-arm wiring, sensitivity kwargs.

        Validates the coarse-likelihood settings, wires the unequal-arm
        instrument noise when ``unequal_arm`` (:meth:`_wire_unequal_arm`),
        and builds ``gs.sensitivity_init_kwargs`` with
        :func:`~lisatools.globalfit.stock.erebor.noise.noise_sensitivity_init_kwargs`
        from the psd / galfor blocks, the foreground modulation (path and
        epoch; ``"data"`` defers the anchor to the engine), ``wdm_psd_method``
        and the SGWB model.

        Args:
            gs: The build-time general block (mutated in place).
        """
        validate_coarse_settings(gs, all_source=True)
        psd = getattr(self, "psd", None) if "psd" in self._branch_names else None
        if gs.unequal_arm:
            self._wire_unequal_arm(gs, psd)

        extra = dict(self._sgwb_sens_kwargs(gs))
        if gs.wdm_psd_method:
            # Backend-wide policy (instrument, foreground, SGWB together).
            extra["wdm_psd_method"] = gs.wdm_psd_method

        # Modulation table: "auto" -> the packaged mission-clock table,
        # anchored at the data start, when the data are mojito (after the
        # missing-folder fallback) and the foreground is fit; else stationary.
        # The resolved values are written back onto this build-time copy.
        mod_path = gs.galfor_modulation_path
        if isinstance(mod_path, str) and mod_path.strip().lower() in ("", "none"):
            mod_path = None
        elif isinstance(mod_path, str) and mod_path.strip().lower() == "auto":
            mod_path = None
            if gs.data_mode == "mojito" and "galfor" in self._branch_names:
                from lisatools.sensitivity import packaged_galfor_modulation_path

                mod_path = packaged_galfor_modulation_path()
                if gs.galfor_modulation_t0 in (None, ""):
                    gs.galfor_modulation_t0 = "data"
        gs.galfor_modulation_path = mod_path

        # Modulation epoch: 0.0 (table already in the data frame), an explicit
        # absolute epoch, or "data" -> deferred to the engine, which anchors
        # the lazy table at the authoritative data_t0 after processing.
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
            galfor=getattr(self, "galfor", None) if "galfor" in self._branch_names else None,
            psd=psd,
            galfor_modulation_path=gs.galfor_modulation_path,
            galfor_modulation_t0=modulation_t0,
            extra=extra,
        )

    def _wire_unequal_arm(self, gs: AllSourcesGeneralSettings, psd) -> None:
        """UNEQUAL_ARM=1: unequal-arm instrument noise fed by the brick's /ltts.

        Only plain scalars/paths go on the settings tree here; the
        :class:`~lisatools.sensitivity.LinkDelayTable` itself is built by
        ``GeneralSetup._resolve_deferred_noise_model`` once the data epoch
        (``data_t0``) is authoritative.

        Sets ``psd.instrument_component_cls`` (when unset) and adds
        ``ltts_l1_file``, ``ltts_stride`` and ``wdm_psd_method`` to
        ``psd.instrument_component_kwargs`` (existing keys win). Duplicates
        :func:`~lisatools.globalfit.stock.erebor.noise.wire_unequal_arm_psd`
        (the noise variants' copy).

        Args:
            gs: The build-time general block.
            psd: The psd branch block, or ``None``.

        Raises:
            ValueError: Without a psd branch, outside mojito mode, or when the
                NOISE brick has no ``/ltts`` group.
            FileNotFoundError: When no NOISE brick is found.
        """
        import h5py

        from lisatools.sensitivity import UnequalArmInstrumentNoise

        from ..noise import resolve_noise_file

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

    def _sgwb_sens_kwargs(self, gs: AllSourcesGeneralSettings) -> dict:
        """``{"sgwb_stochastic_fn": ...}`` with an sgwb branch, else ``{}``."""
        if "sgwb" not in self._branch_names:
            return {}
        sgwb = getattr(self, "sgwb", None)
        fn = getattr(sgwb, "stochastic_fn", None) if sgwb is not None else None
        return dict(sgwb_stochastic_fn=fn or gs.sgwb_stochastic_fn)

    # -- branch resolution --------------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        """Erebor completion (deepcopy, Tobs / dt, build seed, domain), then ``_prepare_<name>``.

        Args:
            name: Branch name.
            general_setup: The built general setup.

        Returns:
            Settings: The prepared branch block.
        """
        settings = super().prepare_branch_settings(name, general_setup)
        prep = getattr(self, f"_prepare_{name}", None)
        return prep(settings, general_setup) if prep is not None else settings

    def _prepare_gb(self, gb: AllSourcesGBSettings, general_setup: GeneralSetup):
        """GB prep: :func:`~lisatools.globalfit.stock.erebor.variants.gb_no_fg.prepare_gb_branch`.

        Sangria data with no ``t0`` uses :data:`GB_T_REF`.
        """
        if gb.t0 is None and self.general.data_mode == "sangria":
            # Legacy constant (the Sangria stream carries no catalogue epoch).
            gb.t0 = GB_T_REF
        return prepare_gb_branch(
            gb,
            general_setup,
            data_mode=self.general.data_mode,
            synthetic_t_start=float(self.general.t_start),
        )

    def _prepare_vgb(self, vgb: VGBSettings, general_setup: GeneralSetup):
        """VGB prep: :func:`~lisatools.globalfit.stock.erebor.vgb.prepare_vgb_branch`."""
        return prepare_vgb_branch(
            vgb, general_setup, data_mode=self.general.data_mode
        )

    def _prepare_psd(self, psd: AllSourcesPSDSettings, general_setup: GeneralSetup):
        """PSD prep: priors / start around ``general.psd_injection`` (``prepare_psd_branch``)."""
        return prepare_psd_branch(psd, self.general.psd_injection)

    def _prepare_galfor(self, galfor: GalForSettings, general_setup: GeneralSetup):
        """Galfor prep: ``prepare_galfor_branch``."""
        return prepare_galfor_branch(galfor)

    def _prepare_sgwb(self, sgwb: AllSourcesSGWBSettings, general_setup: GeneralSetup):
        """SGWB prep: wide uniform priors (``log10_A`` in [-16, -9], ``alpha`` in [-1, 2])."""
        from eryn.prior import ProbDistContainer, uniform_dist

        if sgwb.initialize_kwargs is None:
            sgwb.initialize_kwargs = {}
        if sgwb.priors is None:
            # Wide enough to contain a detectable log10_A injection (the stock
            # (-22, -18) default would exclude it).
            sgwb.priors = {
                "sgwb": ProbDistContainer(
                    {0: uniform_dist(-16.0, -9.0), 1: uniform_dist(-1.0, 2.0)}
                )
            }
        if sgwb.injection is None:
            sgwb.injection = np.asarray(self.general.sgwb_injection, dtype=float)
        return sgwb

    def _prepare_mbh(self, mbh: AllSourcesMBHSettings, general_setup: GeneralSetup):
        """MBH prep: ``source_runtime.prepare_mbh_branch``."""
        return prepare_mbh_branch(mbh, general_setup, self.general)

    def _prepare_emri(self, emri: AllSourcesEMRISettings, general_setup: GeneralSetup):
        """EMRI prep: ``source_runtime.prepare_emri_branch``."""
        return prepare_emri_branch(emri, general_setup, self.general)

    def _prepare_sobbh(self, sobbh: AllSourcesSOBBHSettings, general_setup: GeneralSetup):
        """SOBBH prep: ``source_runtime.prepare_sobbh_branch``."""
        return prepare_sobbh_branch(sobbh, general_setup, self.general)

    # -- runtime objects (post-deepcopy) -------------------------------------------

    def attach_runtime_objects(self) -> None:
        """Register engine-side ``signal_gen`` adapters on the source Setups.

        Only mbh/emri/sobbh get one (the engine builds/subtracts their
        templates in ``setup_acs``); gb/psd/galfor keep the recipe-side path.
        """
        source_branches = [
            b for b in _SOURCE_BRANCH_TO_CLASS if b in self.branch_names
        ]
        if not source_branches:
            return
        gs: AllSourcesGeneralSettings = self.general
        # PREPARED settings, not the fit-level blocks (see the full_year
        # twin): preparation-computed values (the SOBBH chirp bound) live
        # only on source_info[...].settings.
        def _prep(name, fallback):
            if name in self.branches and name in self.source_info:
                return self.source_info[name].settings
            return getattr(self, name) if name in self.branches else fallback

        mbh = _prep("mbh", AllSourcesMBHSettings())
        sobbh = _prep("sobbh", AllSourcesSOBBHSettings())
        emri = _prep("emri", AllSourcesEMRISettings())
        # The run-domain SPEC the injection sites resolved MBH_LIKELIHOOD=auto
        # from (set_default_processor sees the resolved general settings).
        cfg = source_signal_cfg(
            gs, mbh, sobbh, emri, domain_settings=run_domain_spec(self.general_info)
        )
        for branch in source_branches:
            info = self.source_info[branch]
            info.signal_gen = SourceSignalGen(
                branch, info.transform, self.general_info, cfg
            )


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``all_sources``: the canonical per-branch move stacks.

    GB rides ``setup_gb_moves`` (the gb_no_fg stack), PSD/galfor/sgwb the
    single PSDMove, and MBH/EMRI/SOBBH the shared source_runtime builders
    (their residual subtraction already ran engine-side via ``signal_gen``).

    The run's ``setup_function`` (``AllSourcesGlobalFit.default_setup_function``),
    called on every compute rank at run start with whatever recipe
    ``fit.recipe`` holds; in production that is the driver's staged recipe. A stack is
    built only when the recipe asks for it: GB when any requested stock
    name starts with ``rj_`` or ``gb_``; VGB for any ``vgb*`` name; one
    ``<name>_pe`` / ``<name>_search`` pair per noise branch
    (:func:`~lisatools.globalfit.recipe.build_noise_moves`, 5 repeats on
    CPU, the branch's ``num_prop_repeats`` on GPU); MBH / EMRI / SOBBH when
    ``<branch>_pe`` is requested. Then :meth:`Recipe.setup
    <lisatools.globalfit.recipe.Recipe.setup>` materializes every stage.

    Args:
        recipe (Recipe): The fit's recipe.
        engine_info: The engine settings handed to every move builder.
        curr: The built current info (``general_info``, ``source_info``).
        acs: The rank's ``AnalysisContainerArray``.
        priors (dict): Per-branch priors.
        state (GFState): The starting state.

    Raises:
        RuntimeError: When source moves are requested but no
            ``SourceSignalGen`` was attached (fit not built through
            :meth:`AllSourcesGlobalFit.build`).
    """
    general_info = curr.general_info
    nwalkers, ntemps = general_info.nwalkers, general_info.ntemps
    gpus = general_info.gpus
    pin_main_device(acs.xp, gpus)

    requested = recipe.stock_names()
    stock_moves = {}

    # --- GB: the shared gb_no_fg stack (chunked-het likelihood, seeding, moves) ---
    if "gb" in curr.source_info and any(
        n.startswith("rj_") or n.startswith("gb_") for n in requested
    ):
        stock_moves.update(setup_gb_moves(engine_info, curr, acs, priors, state))

    # --- VGB: the fixed-dimensional same-leaf stretch stack ---
    # Any vgb-namespaced move arms the stack (not just "vgb_pe"): since
    # 2026-09-16 the stack can also carry "vgb_ridge_gibbs", and a recipe
    # requesting only that one must still get its moves built.
    if "vgb" in curr.source_info and any(
        n.startswith("vgb") for n in requested
    ):
        stock_moves.update(setup_vgb_moves(engine_info, curr, acs, priors, state))

    # --- Noise moves (split: psd / galfor / sgwb each with own ladder) ---
    noise_move_spec = {"psd": ["psd"], "galfor": ["galfor"], "sgwb": ["sgwb"]}
    for name, sampled in noise_move_spec.items():
        if not all(b in curr.source_info for b in sampled):
            continue
        if not any(n.startswith(name) for n in requested):
            continue
        # CPU runs use few repeats (laptop smokes); GPU runs default to the
        # branch settings' num_prop_repeats
        num_repeats = 5 if gpus is None else None
        search_move, pe_move = build_noise_moves(
            engine_info, curr, acs, priors,
            sampled_branches=sampled, num_repeats=num_repeats,
        )
        stock_moves[f"{name}_pe"] = pe_move
        stock_moves[f"{name}_search"] = search_move

    # --- MBH / EMRI / SOBBH: the shared full_year runtime builders ---
    source_present = [b for b in _SOURCE_BRANCH_TO_CLASS if b in curr.source_info]
    if source_present and any(f"{b}_pe" in requested for b in source_present):
        cfg = find_source_cfg(curr)
        if cfg is None:
            raise RuntimeError(
                "all_sources setup_recipe found no SourceSignalGen on any "
                "source branch — build through AllSourcesGlobalFit.build()."
            )
        stock_moves.update(build_source_moves(curr, acs, priors, state, cfg))

    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves=stock_moves,
        ntemps=ntemps, nwalkers=nwalkers,
    )
    recipe.setup(ctx)


AllSourcesGlobalFit.default_setup_function = staticmethod(setup_recipe)
