"""Erebor-family base fit: shared general knobs + build resolution.

:class:`EreborGeneralSettings` extends the engine's
:class:`~lisatools.globalfit.engine.GeneralSettings` dataclass with the
Erebor-family knobs every variant shares (WDM-grid targets, analysis band,
TDI channel, mojito data path, fixed-PSD values, compute-backend choice).
Env-var-backed defaults resolve *explicit kwarg > env var > hard default*
at construction (see :func:`lisatools.globalfit.stock.base.env_default`).

:class:`EreborFit` extends :class:`StockGlobalFit` with the shared
build-time resolution: the deferred general block is turned into a final
:class:`GeneralSettings` (grid derived, window taper realized, domain
factory built, data processor chosen) only inside
:meth:`EreborFit.make_general_settings` — never at construction.

Pipeline position: every Erebor variant (``all_sources`` in production)
subclasses :class:`EreborFit`; :meth:`StockGlobalFit.build
<lisatools.globalfit.stock.base.StockGlobalFit.build>` calls
:meth:`EreborFit.make_general_settings` once, before the heavy
:class:`~lisatools.globalfit.engine.GeneralSetup` (data load + WDM pour).
The production launchers set the grid / band / output knobs of this block
through the environment (``TOBS_TARGET``, ``MIN_FREQ``, ``MAX_FREQ``,
``EDGE_CROP_WAVELETS``, ``FILE_STORE_DIR``, ``BASE_FILE_NAME``,
``NUM_ITERATIONS``, ``NWALKERS``, ``USE_GPU``, ``GPU_BACKEND``, ...).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import typing
from copy import deepcopy

from lisatools.domains import WDMSettings

from ...engine import GeneralSettings, GeneralSetup, Settings
from ..base import StockGlobalFit, engine_ntemps_default, env_default
from .common import (
    default_edge_crop_wavelets,
    derive_wdm_grid,
    resolve_compute,
    tdi_generation_info,
)

__all__ = ["EreborGeneralSettings", "EreborFit"]

logger = logging.getLogger(__name__)


def _parse_gpu_list(raw: str) -> typing.List[int]:
    """GPUS env parser: comma-separated device indices (``"2,3"`` -> ``[2, 3]``)."""
    return [int(x) for x in raw.split(",") if x.strip() != ""]


@dataclasses.dataclass
class EreborGeneralSettings(GeneralSettings):
    """Erebor-family general block: engine fields + shared Erebor knobs.

    Fields left ``None`` here are *derived at build time* by
    :meth:`EreborFit.make_general_settings` (e.g. ``Tobs`` from the WDM
    grid, ``domain_settings`` from the band knobs, ``window_taper_duration``
    from ``window_tukey_alpha``); setting them explicitly overrides the
    derivation — swap whole objects, don't fight the resolver.

    Reached as ``fit.general`` (headline knobs also as ``fit.<knob>``).
    Every env knob is the capitalized field name; defaults below are this
    class's — variants subclass it and override some (the production
    ``AllSourcesGeneralSettings`` uses ``dt=5.0`` (re-set to 2.5 s for mojito
    data), a fixed ``nf=720`` / ``nt=2160`` grid (cleared by the production
    driver, so the grid derives from ``tobs_target``),
    ``window_tukey_alpha=0.0``, ``edge_crop_wavelets`` default 20, and the
    coarse sidecar defaults ``COARSE_Q=8`` / ``COARSE_GPU_MODE=delayed_acceptance``).
    Inherited engine fields (``Tobs``, ``domain_settings``,
    ``data_processor_class``, ``processor_init_kwargs``,
    ``sensitivity_init_kwargs``, ``fixed_psd_kwargs``, ``likelihood_source_only``,
    ``hdf_compression``, the run metadata, ...) are documented on
    :class:`~lisatools.globalfit.engine.GeneralSettings`.

    Attributes:
        num_iterations: Sampler iterations (stored rows). Env ``NUM_ITERATIONS``
            (legacy ``GF_NUM_ITER``). Default 500; launchers 2000.
        nwalkers: Cold-chain walkers. Env ``NWALKERS``. Default 4; launchers 4
            (one walker block per compute rank).
        gpus_per_rank: Devices per compute rank. Env ``GPUS_PER_RANK``. Default
            ``None`` = auto (a lone compute rank on a node drives all of
            ``gpus``; several ranks get one device each). At most one of
            ``gpus_per_rank`` / ``ranks_per_gpu`` may exceed 1. Launchers leave it
            unset.
        ranks_per_gpu: Compute ranks sharing one device. Env ``RANKS_PER_GPU``.
            Default 1 (the launchers leave it unset).
        ranks_per_block: Compute ranks sharing one walker block. Env
            ``RANKS_PER_BLOCK``. Default ``None`` = auto,
            ``n_blocks = gcd(nwalkers, n_compute)``; an int must divide the
            compute-rank count (``1`` forces the plain walker-block layout).
        ntemps: RETIRED engine ladder size, pinned to 1 (the engine runs
            cold-chain only; each branch tempers on its own ``<BRANCH>_NTEMPS``
            ladder). A set ``NTEMPS`` env var raises at construction.
        random_seed: Run seed (no env). Default 103209. Per-rank build seeds
            for prior / proposal objects derive from it
            (:func:`~lisatools.globalfit.communication.ranks.rank_build_seed`).
        backup_iter: Save-step cadence of the running backup copy of the store
            (1 = every save). Env ``BACKUP_ITER``. Default 1; launchers 10.
        midit_checkpoint: Mid-iteration checkpointing (resume from inside an
            iteration). Env ``MIDIT_CHECKPOINT``. Default ``True``; launchers 0.
        midit_checkpoint_min_interval: Minimum seconds between mid-iteration
            checkpoints. Env ``MIDIT_CHECKPOINT_MIN_INTERVAL``. Default 600.0.
        main_file_key: Store-name suffix (no env): the main HDF store is
            ``<file_store_dir>/<base_file_name>_<main_file_key>.h5``. Default
            ``"testing"``.
        make_diagnostic_plots: Eryn diagnostic plots during the run. Env
            ``MAKE_DIAGNOSTIC_PLOTS`` (legacy ``MAKE_PLOTS``). Default ``True``.
        verbose: Console verbosity (headline knob): ``False`` keeps logs in the
            run's log files only. Env ``VERBOSE``. Default ``False``; launchers 1.
        progress: Progress bar only; ``None`` follows ``verbose``. Env
            ``PROGRESS``. Default ``None``; launchers 0.
        plot_iterations: Iterations between diagnostic-plot refreshes. Env
            ``PLOT_ITERATIONS``. Default 100.
        file_store_dir: Output directory of the store, logs and artifacts. Env
            ``FILE_STORE_DIR``. Default ``"./gf_output/"``; launchers the run's
            ``STORE_DIR``.
        base_file_name: Store base name. Env ``BASE_FILE_NAME``. Default
            ``"erebor_run"``; launchers ``gf_prod_9mo`` / ``gf_prod_12mo``.
        coarse_Q: Coarse real-WDM noise likelihood factor (time columns
            averaged per coarse cell; :mod:`lisatools.coarsewdm`); 1 = off. Env
            ``COARSE_Q``. Default 1; launchers 1 (coarse path OFF in production).
        coarse_use_ws: Coarse weights from Welch-Satterthwaite dof (``True``) or
            Bartlett cell sizes. Env ``COARSE_USE_WS``. Default ``True``.
        coarse_fiducial: Covariance the frozen coarse dof are evaluated at,
            ``"injection"`` or ``"initial"``. Env ``COARSE_FIDUCIAL``. Default
            ``"injection"``.
        coarse_gpu_mode: All-source coarse sidecar: ``"off"``, ``"auto"``,
            ``"search_approx"`` (optimization stages only) or
            ``"delayed_acceptance"`` (the only coarse mode valid for PE). Env
            ``COARSE_GPU_MODE``. Default ``"off"``; launchers ``off``.
        coarse_gpu_batch_bytes: Batch memory budget of the coarse sidecar
            (bytes). Env ``COARSE_GPU_BATCH_BYTES``. Default 256 MiB.
        coarse_gpu_debug: Coarse sidecar debug checks. Env ``COARSE_GPU_DEBUG``.
            Default ``False``.
        dt: Time-domain sample spacing of the analysed data (s; no env).
            Default 2.5.
        tobs_target: Target observation span (s) the WDM grid is derived from
            (:func:`~lisatools.globalfit.stock.erebor.common.derive_wdm_grid`;
            the exact ``Tobs = Nf * Nt * dt`` lands near it). Env
            ``TOBS_TARGET``. Default 90 d; launchers 23328000 (9mo) /
            31104000 (12mo).
        nf: Fixed-grid override: WDM frequency layers. Used only when ``nf`` and
            ``nt`` are both set (then ``Tobs = nf * nt * dt``). Default ``None``.
        nt: Fixed-grid override: WDM time columns (see ``nf``).
        wavelet_duration_min: Lower bound (s) on the WDM wavelet duration
            ``Nf * dt`` for the grid derivation. Env ``WAVELET_DURATION_MIN``
            (legacy ``WAVELET_DUR_MIN``). Default 3600.0.
        wavelet_duration_max: Upper bound (s), env ``WAVELET_DURATION_MAX``
            (legacy ``WAVELET_DUR_MAX``). Default 4400.0.
        min_freq: Lower edge of the analysed band (Hz; headline knob). Keep it
            ``>= 2 * layer_df`` (a build-time warning fires otherwise: the
            DC-adjacent layer is singular). Env ``MIN_FREQ``. Default 1e-4;
            launchers 2.5e-4.
        max_freq: Upper edge of the analysed band (Hz; headline knob). Env
            ``MAX_FREQ``. Default 2.5e-2; launchers 2.5e-2.
        window_type: Data window type (no env). Default ``"tukey"``.
        window_tukey_alpha: Tukey alpha of the data window (fraction of
            ``Tobs``; headline knob). Overridden at build by
            ``window_taper_wavelets > 0``. Env ``WINDOW_TUKEY_ALPHA``. Default 0.05.
        edge_crop_wavelets: WDM time columns excluded at each end of the
            analysed domain (``min_time`` / ``max_time`` of every
            likelihood-facing WDM grid). ``None`` = auto
            (:func:`~lisatools.globalfit.stock.erebor.common.default_edge_crop_wavelets`).
            Env ``EDGE_CROP_WAVELETS``. Default ``None``; launchers 60 (the
            GB/VGB sig-het build refuses a crop below
            ``ceil(0.5 * SIGHET_TUKEY_ALPHA * Nt) + 8``, see
            :func:`~lisatools.globalfit.stock.erebor.gb.check_sighet_build_config`).
        window_taper_wavelets: Data-window taper width in wavelets (each
            ``Nf * dt`` s); ``> 0`` replaces ``window_tukey_alpha`` by
            ``2 * K / Nt`` (a fixed absolute taper), 0 = use the alpha. Env
            ``WINDOW_TAPER_WAVELETS``. Default 2.
        tdi_chan: TDI channel set (no env). Default ``"XYZ"`` (TDI 2nd
            generation; :func:`~lisatools.globalfit.stock.erebor.common.tdi_generation_info`).
        nchannels: Number of TDI channels. Default 3.
        data_mode: Stock data pipeline: ``"mojito"`` (L1 bricks under
            ``mojito_data_path``), ``"synthetic"`` (all streams in-process) and,
            on ``all_sources``, legacy ``"sangria"``. An explicit
            ``data_processor_class`` always wins. A missing mojito folder falls
            back to ``"synthetic"`` (:meth:`EreborFit.resolve_data_source`). Env
            ``DATA_MODE`` (legacy ``DATA_PROCESSOR``). Default ``"mojito"``
            (``blank``: ``"synthetic"``); launchers leave it unset.
        synthetic_t_start: Start time (s) of synthetic streams. Default 10000.0
            (keeps the TDI-2 look-back inside the orbit span).
        mojito_data_path: Root of the mojito L1 brick cache (headline knob). Env
            ``MOJITO_DATA_PATH``. Default
            ``~/.mojito_cache/brickmarket/mojito_light_v1_0_0/``; launchers
            ``/shared/data/mojito_cache``.
        noise_file: Explicit mojito NOISE L1 brick; ``None`` auto-discovers
            ``<mojito_data_path>/data/INSTRUMENT/L1/NOISE_*``. Env ``NOISE_FILE``.
        psd_from_noise_file: Fixed-PSD variants only (``gb_no_fg``,
            ``full_year_combined``): read ``[Soms_d, Sa_a]`` off the NOISE brick
            (:meth:`EreborFit.resolve_noise_file_psd_params`). ``None`` = auto
            (mojito data + brick found), ``True`` = required, ``False`` = stock
            levels. Env ``PSD_FROM_NOISE_FILE``.
        ldc_source_file: Sangria (LDC2) data file for ``data_mode="sangria"``.
            Env ``LDC_SOURCE_FILE``. The default is a developer-machine path.
        source_types: Mojito stream classes the L1 loader reads (no env;
            ``("GB",)``). The production driver sets it from its own
            ``SOURCE_TYPES`` export (``COMBINED,GB,VGB,MBHB,EMRI,SOBHB``).
        orbits_frame: Frame the L1 orbits are loaded in. Default ``"icrs"`` (the
            run frame: sky / polarization parameters are sampled in ICRS).
        synthetic_injections: Injection coordinates for synthetic data:
            ``"stock"`` (fixed tables), ``"prior"`` (seeded prior draws) or
            ``None`` = auto (``"prior"`` after a mojito->synthetic fallback,
            else ``"stock"``). Env ``SYNTHETIC_INJECTIONS``.
        synthetic_injection_seed: Seed of the ``"prior"`` draws. Env
            ``SYNTHETIC_INJECTION_SEED``. Default 1234.
        fixed_psd_params: ``[Soms_d, Sa_a]`` (sqrt units) of the fixed
            instrument PSD for runs WITHOUT a psd branch; becomes
            ``fixed_psd_kwargs`` at build. Default ``None``.
        use_gpu: Run on GPU (headline knob). ``None`` = auto (cupy importable).
            Env ``USE_GPU``. Launchers 1.
        gpu_backend: CUDA backend wheel (``"cuda11x"`` / ``"cuda12x"`` /
            ``"cuda13x"``; headline knob). ``"auto"`` resolves to the wheel
            matching the cupy runtime
            (:func:`~lisatools.globalfit.stock.erebor.common.resolve_gpu_backend`);
            explicit values are honored verbatim. Env ``GPU_BACKEND``. Default
            ``"auto"``; launchers ``cuda13x``.
        gpus: Device indices (headline knob); ``None`` -> ``[0]`` on the GPU
            path. Env ``GPUS`` (``"0"`` or ``"2,3"``).

    Example:
        .. code-block:: python

            fit = erebor.all_sources(nwalkers=4)
            fit.general.tobs_target = 30 * 86400.0   # or env TOBS_TARGET
            fit.general.edge_crop_wavelets = 60
            fit.min_freq = 2.5e-4                     # headline knob -> fit.general
    """

    # --- run shape (env-backed) ---
    num_iterations: int = dataclasses.field(
        default_factory=env_default("NUM_ITERATIONS", 500, int)
    )
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 4, int))
    # multi-rank layout (design spec 2026-09-15, Decision 2). ``gpus_per_rank``
    # None = AUTO: a lone compute rank on a node drives the node's whole
    # ``gpus`` pool (today's in-process multi-GPU run at ``-n 1``); several
    # compute ranks on a node get one device each. An int pins it. At most one
    # of the two may exceed 1. Plain attributes, never headline knobs.
    # (Plain ``#`` comments: the class docstring's Attributes section documents
    # these fields, so they are not rendered twice.)
    gpus_per_rank: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GPUS_PER_RANK", None, int)
    )
    ranks_per_gpu: int = dataclasses.field(default_factory=env_default("RANKS_PER_GPU", 1, int))
    # R -- compute ranks sharing ONE walker block, the axis that makes
    # ``n_gpus > nwalkers`` legal (2026-09-23). ``None`` = AUTO:
    # ``n_blocks = gcd(nwalkers, n_compute)``, i.e. maximize walker blocks
    # and only then replicate, because a block is the cheap axis (GB is
    # sublinear in block width) while a replica costs a whole ACA. An int
    # pins it and must divide the compute-rank count. ``1`` forces today's
    # walker-block layout and then ``nwalkers`` must divide ``n_compute``.
    ranks_per_block: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("RANKS_PER_BLOCK", None, int)
    )
    # RETIRED as a tempering knob: the engine runs cold-chain only (each
    # branch tempers internally; see the per-branch <BRANCH>_NTEMPS knobs).
    # A set NTEMPS env var raises. erebor.blank overrides this field to keep
    # a real engine ladder for simple-API branches.
    ntemps: int = dataclasses.field(default_factory=engine_ntemps_default())
    random_seed: int = 103209
    # Save-step cadence of the running backup copy (env BACKUP_ITER). 1 =
    # every save. The copy is a full-file copy + fsync on the saver rank, and
    # the sampler's next save is a BLOCKING pickled send to that rank, so
    # once the store is large the sampler waits for the copy: 3mo v8 job 479
    # measured [SAVE] save_step 60-80 s per iteration on a 14 GB store (24%
    # of a 5.4 min PE iteration; 0.5 s at 1.5 GB). With the mid-iteration
    # checkpoint (MIDIT_CHECKPOINT, 600 s) covering the tail, a cadence of
    # ~10 bounds a torn-store loss to the checkpoint interval, not 10
    # iterations. Production scripts export BACKUP_ITER=10 (2026-09-14).
    backup_iter: int = dataclasses.field(default_factory=env_default("BACKUP_ITER", 1, int))
    # Mid-iteration checkpointing (env MIDIT_CHECKPOINT /
    # MIDIT_CHECKPOINT_MIN_INTERVAL; see GeneralSettings for semantics).
    midit_checkpoint: bool = dataclasses.field(
        default_factory=env_default("MIDIT_CHECKPOINT", True, bool)
    )
    midit_checkpoint_min_interval: float = dataclasses.field(
        default_factory=env_default("MIDIT_CHECKPOINT_MIN_INTERVAL", 600.0, float)
    )
    main_file_key: str = "testing"
    # Diagnostic plotting during the run. MAKE_DIAGNOSTIC_PLOTS=0 disables the eryn
    # diagnostic plots entirely (fastest, and dodges plot-only crashes);
    # PLOT_ITERATIONS sets how many sampler iterations between plot refreshes.
    make_diagnostic_plots: bool = dataclasses.field(
        default_factory=env_default("MAKE_DIAGNOSTIC_PLOTS", True, bool)
    )
    # Console verbosity (headline knob; env VERBOSE). Default: quiet — logs
    # go to the run's log files only, no progress bars.
    verbose: bool = dataclasses.field(default_factory=env_default("VERBOSE", False, bool))
    # None -> follow verbose (see GeneralSettings.progress). Env: PROGRESS.
    progress: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("PROGRESS", None, bool)
    )
    plot_iterations: int = dataclasses.field(
        default_factory=env_default("PLOT_ITERATIONS", 100, int)
    )
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "erebor_run")
    )

    # --- coarse real-WDM likelihood (shared by the noise variants and
    #     all_sources; lifted from NoiseGeneralSettings, plan-2 T3) ---
    # Q=1 (default) leaves the ordinary fine likelihood untouched everywhere.
    # Noise-only runs at Q>1 keep the CPU backend-replacement path; an
    # all-source run additionally needs coarse_gpu_mode != "off" and builds a
    # CoarseWDMRuntime sidecar (per-walker statistics) instead — the fine
    # backend stays canonical and source moves never see coarse state.
    coarse_Q: int = dataclasses.field(default_factory=env_default("COARSE_Q", 1, int))
    coarse_use_ws: bool = dataclasses.field(
        default_factory=env_default("COARSE_USE_WS", True, bool)
    )
    coarse_fiducial: str = dataclasses.field(
        default_factory=env_default("COARSE_FIDUCIAL", "injection", str)
    )
    # "off" | "search_approx" (optimization stages only; not a posterior
    # sampler) | "delayed_acceptance" (the only coarse mode valid for
    # production PE: stage-1 coarse screen, stage-2 fine correction).
    coarse_gpu_mode: str = dataclasses.field(
        default_factory=env_default("COARSE_GPU_MODE", "off", str)
    )
    coarse_gpu_batch_bytes: int = dataclasses.field(
        default_factory=env_default("COARSE_GPU_BATCH_BYTES", 256 * 1024 * 1024, int)
    )
    coarse_gpu_debug: bool = dataclasses.field(
        default_factory=env_default("COARSE_GPU_DEBUG", False, bool)
    )

    # --- grid / domain targets (env-backed) ---
    # ``Tobs`` (inherited) stays None; the build derives it as Nf*Nt*dt.
    dt: float = 2.5
    tobs_target: float = dataclasses.field(
        default_factory=env_default("TOBS_TARGET", 90 * 86400.0, float)
    )
    # Fixed-grid override: when BOTH are set, the WDM grid is (nf, nt)
    # directly (Tobs = nf*nt*dt) instead of adjust_to_even_bins on
    # tobs_target / the wavelet-duration bounds.
    nf: typing.Optional[int] = None
    nt: typing.Optional[int] = None
    wavelet_duration_min: float = dataclasses.field(
        default_factory=env_default("WAVELET_DURATION_MIN", 3600.0, float)
    )
    wavelet_duration_max: float = dataclasses.field(
        default_factory=env_default("WAVELET_DURATION_MAX", 4400.0, float)
    )
    # Env-backed per the stock convention (knob = capitalized field name):
    # these were plain defaults, so MIN_FREQ/MAX_FREQ in the environment
    # were SILENTLY ignored — the exact failure mode rule 0 warns about.
    min_freq: float = dataclasses.field(
        default_factory=env_default("MIN_FREQ", 1e-4, float)
    )
    max_freq: float = dataclasses.field(
        default_factory=env_default("MAX_FREQ", 2.5e-2, float)
    )
    window_type: str = "tukey"
    window_tukey_alpha: float = dataclasses.field(
        default_factory=env_default("WINDOW_TUKEY_ALPHA", 0.05, float)
    )
    # None -> auto from the data-taper alpha. EDGE_CROP_WAVELETS is THE
    # uniform edge-exclusion knob (user ruling 2026-08-19): it sets the ONE
    # domain [min_time, max_time] every likelihood-facing WDMSettings
    # inherits (chunked comp, sig-het engine, buffers, invC) -- bringing the
    # edges in here removes edge-created error EVERYWHERE at once, with
    # min/max_freq still free to vary per source. Candidate for the
    # edge-localized sig-het h_h inflation: 60 (covers the old 54-layer
    # taper region; costs 2x60/Nt of the data).
    edge_crop_wavelets: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("EDGE_CROP_WAVELETS", None, int)
    )
    # Deterministic taper width in WDM wavelets (each Nf*dt seconds). When
    # > 0 this OVERRIDES window_tukey_alpha with 2*K/Nt: the taper is a
    # fixed ABSOLUTE duration set by the wavelet grid, not a fraction of
    # Tobs (a fixed alpha burns alpha/2 * Tobs of data per side -- ~18 d
    # at 2 yr). DERIVATION of the default (2 wavelets): a Tukey ramp of
    # duration tau leaks over bandwidth ~1/tau; confining edge leakage to
    # ~one WDM layer requires tau >= 1/layer_df = 2*Nf*dt = 2 wavelet
    # widths -- shorter smears across layers, longer only discards data.
    # The DIRTY region (taper + the wavelet's own ~2-pixel support +
    # boundary-wavelet contamination) is removed from the ANALYZED region
    # by the edge crop (min_time/max_time, auto = max(20, taper+4), floor
    # dominating), so the analysis always sees window == 1 on clean
    # wavelets. 0 = legacy fraction-of-Tobs via window_tukey_alpha.
    window_taper_wavelets: int = dataclasses.field(
        default_factory=env_default("WINDOW_TAPER_WAVELETS", 2, int)
    )

    # --- TDI / channels ---
    tdi_chan: str = "XYZ"
    nchannels: int = 3

    # --- data source ---
    # Which stock data processor the variant builds when the user has not
    # swapped one in wholesale. Allowed values are variant-defined (every
    # variant supports "synthetic"; every variant but blank ("synthetic")
    # defaults "mojito"; all_sources also keeps legacy "sangria"). One assignment (or
    # DATA_MODE=<mode>) swaps the whole data pipeline; an explicit
    # ``data_processor_class`` always wins over this knob.
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_MODE", "mojito", str)
    )
    # Start time of the synthetic data stream (mojito/sangria modes pull the
    # start from the data files). Default 10,000 s — NOT 0 — so the TDI2
    # warm-up look-back (output at t needs orbit data back to t - ~8
    # arm-delays ~ 85 s) stays inside the orbit span, which starts at t=0;
    # a t=0 data start makes the first ~85 s of every response unevaluable
    # (they would be NaN-scrubbed to zero).
    synthetic_t_start: float = 10_000.0
    mojito_data_path: str = dataclasses.field(
        default_factory=env_default(
            "MOJITO_DATA_PATH",
            # Home-relative so the same default resolves on any machine
            # (laptop /Users/<u>, cluster /home/<u>); override with
            # MOJITO_DATA_PATH when the cache lives elsewhere.
            os.path.expanduser(
                "~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"
            ),
        )
    )
    # --- mojito NOISE brick (proper noise) ---
    # Explicit path to a mojito NOISE L1 file. None -> auto-discover
    # ``<mojito_data_path>/data/INSTRUMENT/L1/NOISE_*``.
    noise_file: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("NOISE_FILE", None, str)
    )
    # Fixed-PSD variants (gb_no_fg, full_year_combined): read the
    # instrument-noise parameters off the NOISE brick's tabulated estimates
    # (linear least-squares of the analytic model, see
    # ``lisatools.sensitivity.estimate_noise_params_from_file``) instead of
    # the hardcoded stock levels. None -> auto: True when the run uses mojito
    # data and the brick is found; explicit True errors if the brick is
    # missing; False keeps the stock analytic levels.
    psd_from_noise_file: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("PSD_FROM_NOISE_FILE", None, bool)
    )
    ldc_source_file: str = dataclasses.field(
        default_factory=env_default(
            "LDC_SOURCE_FILE",
            "/Users/mkatz/Research/LISAanalysistools/LDC2_sangria_training_v2.h5",
        )
    )
    source_types: typing.Tuple[str, ...] = ("GB",)
    orbits_frame: str = "icrs"

    # --- synthetic-injection coordinates ---
    # How the synthetic data processors pick injection coordinates when the
    # user has not supplied them (``gb_injection_params`` / per-branch
    # ``.injection``): "stock" -> the fixed stock tables; "prior" -> seeded
    # draws from (an interior region of) each source class's sampling
    # priors. ``None`` -> auto: "stock" normally, "prior" when a missing
    # mojito folder forced the synthetic fallback (see
    # :meth:`EreborFit.resolve_data_source`). Explicit injection tables
    # always win over either mode.
    synthetic_injections: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("SYNTHETIC_INJECTIONS", None, str)
    )
    synthetic_injection_seed: int = dataclasses.field(
        default_factory=env_default("SYNTHETIC_INJECTION_SEED", 1234, int)
    )

    # --- fixed sensitivity (used when no psd branch is present) ---
    fixed_psd_params: typing.Optional[typing.List[float]] = None

    # --- compute backend (construction-time choice; sprint rule) ---
    # USE_GPU=0 forces CPU even when cupy is present; unset -> auto-detect.
    use_gpu: typing.Optional[bool] = dataclasses.field(
        default_factory=env_default("USE_GPU", None, bool)
    )
    # GPU_BACKEND selects the lisatools/gbgpu backend wheel flavor
    # (cuda11x / cuda12x / cuda13x); only consulted when the GPU is active.
    # Default "auto" -> resolve_compute detects the wheel matching this
    # machine's cupy runtime at build time, so a non-propagated GPU_BACKEND
    # env var can't silently pick the wrong flavor. An explicit value
    # (env or kwarg) is always honored verbatim.
    gpu_backend: str = dataclasses.field(
        default_factory=env_default("GPU_BACKEND", "auto", str)
    )
    # GPUS: comma-separated device indices (e.g. "0" or "2,3"); unset ->
    # device 0 when the GPU is active.
    gpus: typing.Optional[typing.List[int]] = dataclasses.field(
        default_factory=env_default("GPUS", None, _parse_gpu_list)
    )

    @property
    def tdi_gen(self) -> int:
        """TDI generation (1 or 2) implied by ``tdi_chan``."""
        return tdi_generation_info(self.tdi_chan)[0]

    @property
    def tdi_gen_str(self) -> str:
        """``"1st generation"`` / ``"2nd generation"`` for ``tdi_chan``."""
        return tdi_generation_info(self.tdi_chan)[1]

    @property
    def wavelet_duration_bounds(self) -> typing.Tuple[float, float]:
        """``(wavelet_duration_min, wavelet_duration_max)`` in seconds."""
        return (self.wavelet_duration_min, self.wavelet_duration_max)


class EreborFit(StockGlobalFit):
    """Deferred-build base for Erebor stock variants.

    Headline knobs (top-level attribute/constructor access, delegated to
    ``fit.general``): everything in :attr:`_HEADLINE_KNOBS`. All other
    general fields: ``fit.general.<field>``; per-branch fields:
    ``fit.<branch>.<field>``; whole-object swaps everywhere.

    The headline knobs are the base set (``nwalkers``, ``ntemps``, ``dt``,
    ``num_iterations``, ``file_store_dir``, ``base_file_name``, ``gpus``,
    ``gpu_backend``, ``verbose``) plus ``tobs_target``, ``min_freq``,
    ``max_freq``, ``tdi_chan``, ``window_tukey_alpha``, ``mojito_data_path``
    and ``use_gpu``. Constructor arguments are those of
    :class:`~lisatools.globalfit.stock.base.StockGlobalFit`.

    Build-time resolution (:meth:`make_general_settings`, called once by
    :meth:`~lisatools.globalfit.stock.base.StockGlobalFit.build`):
    :meth:`resolve_data_source` -> deepcopy of ``self.general`` ->
    :meth:`adjust_general` (variant hook) -> compute backend
    (:func:`~lisatools.globalfit.stock.erebor.common.resolve_compute`) -> WDM
    grid and exact ``Tobs`` -> window taper -> edge crop -> DC-layer warning
    -> :meth:`make_domain_settings` -> data processor
    (:meth:`set_default_processor` unless ``data_processor_class`` is set),
    preprocess / sensitivity kwargs -> :meth:`finalize_general` (variant hook).
    Each branch block is then completed by :meth:`prepare_branch_settings`.

    The class attribute ``general_settings_class`` is the general-block
    dataclass :meth:`default_general` instantiates (variants point it at their
    subclass, e.g. ``AllSourcesGeneralSettings``).
    """

    general_settings_class: typing.ClassVar[type] = EreborGeneralSettings

    _HEADLINE_KNOBS = StockGlobalFit._HEADLINE_KNOBS + (
        "tobs_target",
        "min_freq",
        "max_freq",
        "tdi_chan",
        "window_tukey_alpha",
        "mojito_data_path",
        "use_gpu",
    )

    # -- defaults ------------------------------------------------------------

    def default_general(self) -> EreborGeneralSettings:
        """A fresh ``general_settings_class()`` (env-backed defaults read now)."""
        return type(self).general_settings_class()

    # -- derived views (recompute from current knobs; mutation-safe) -----------

    @property
    def wdm_grid(self) -> typing.Tuple[int, int, float, float]:
        """``(Nf, Nt, wavelet_duration [s], Tobs [s])`` from the current grid knobs.

        The fixed ``nf`` / ``nt`` override wins; otherwise
        :func:`~lisatools.globalfit.stock.erebor.common.derive_wdm_grid` on
        ``tobs_target`` / ``dt`` / the wavelet-duration bounds. Recomputed on
        every access (safe after knob edits; no build needed).
        """
        gs = self.general
        if gs.nf is not None and gs.nt is not None:
            return gs.nf, gs.nt, gs.nf * gs.dt, gs.nf * gs.nt * gs.dt
        return derive_wdm_grid(gs.tobs_target, gs.dt, gs.wavelet_duration_bounds)

    @property
    def layer_df(self) -> float:
        """WDM layer width ``1 / (2 * Nf * dt)`` (Hz) of the current :attr:`wdm_grid`."""
        Nf, _, _, _ = self.wdm_grid
        return 1.0 / (2 * Nf * self.general.dt)

    # -- build-time resolution --------------------------------------------------

    def adjust_general(self, gs: EreborGeneralSettings) -> None:
        """Variant hook: mutate the copied general block before resolution.

        Args:
            gs: The build-time copy of ``self.general`` (mutate in place).
        """

    def resolve_data_source(self) -> None:
        """Resolve the data source on the FIT-LEVEL general block (idempotent).

        When ``data_mode == "mojito"`` but the mojito L1 folder does not
        exist on this machine, fall back to ``data_mode = "synthetic"`` so
        the bare quickstart (``fit = erebor.all_sources(); fit.build()``)
        works with no external data. Fallback injection coordinates are
        drawn from the source priors (``synthetic_injections = "prior"``)
        unless the user chose a mode or supplied explicit tables
        (``gb_injection_params`` / per-branch ``.injection``), which always
        win.

        Deliberately mutates ``self.general`` (not just the build-time
        copy): branch preparation reads the fit-level block, so the data
        processor and the branch injections must resolve identically.

        In synthetic + ``"prior"`` mode with a ``gb`` branch and
        ``gb_injection_params is None``, the GB rows are drawn here with
        :func:`~lisatools.globalfit.stock.erebor.injections.make_gb_injections`
        over the GB branch band.

        Raises:
            ValueError: When ``synthetic_injections`` is not ``"stock"``,
                ``"prior"`` or ``None``.
        """
        gs = self.general
        if gs.data_mode == "mojito" and not os.path.isdir(gs.mojito_data_path):
            logger.warning(
                "data_mode='mojito' but the mojito data folder %r does not "
                "exist — falling back to data_mode='synthetic' (in-process "
                "injections, coordinates drawn from the source priors). Set "
                "MOJITO_DATA_PATH / general.mojito_data_path to use mojito "
                "data.",
                gs.mojito_data_path,
            )
            gs.data_mode = "synthetic"
            if gs.synthetic_injections is None:
                gs.synthetic_injections = "prior"
        if gs.synthetic_injections is None:
            gs.synthetic_injections = "stock"
        elif gs.synthetic_injections not in ("stock", "prior"):
            raise ValueError(
                f"synthetic_injections={gs.synthetic_injections!r} not "
                "recognised; use 'stock', 'prior', or None (auto)."
            )
        # Prior-mode GB rows ride the existing ``gb_injection_params`` knob
        # (the synthetic GB processor and the GB branch prep both read it),
        # so variants need no extra wiring. An explicit user table wins.
        if (
            gs.data_mode == "synthetic"
            and gs.synthetic_injections == "prior"
            and "gb" in self._branch_names
            and getattr(gs, "gb_injection_params", "absent") is None
        ):
            from .injections import make_gb_injections

            gs.gb_injection_params = make_gb_injections(
                self._gb_injection_count(),
                mode="prior",
                seed=gs.synthetic_injection_seed,
                band=self._gb_injection_band(),
            )

    def resolve_noise_file_psd_params(
        self, gs: EreborGeneralSettings
    ) -> typing.Optional[typing.List[float]]:
        """``[Soms_d, Sa_a]`` read off the mojito NOISE brick, or ``None``.

        Honors ``gs.psd_from_noise_file``: ``False`` -> ``None``; ``None``
        (auto) -> only for mojito-data runs with a resolvable brick;
        ``True`` -> required (raises when the brick is missing or the fit
        fails). The scalar levels come from a linear least-squares fit of the
        analytic instrument model to the brick's tabulated
        ``noise_estimates`` (:func:`lisatools.sensitivity.estimate_noise_params_from_file`).
        Results are cached per file path on the fit instance.

        Args:
            gs: The build-time general block (``psd_from_noise_file``,
                ``data_mode``, ``mojito_data_path``, ``noise_file``,
                ``tdi_gen`` are read).

        Returns:
            list[float] | None: ``[Soms_d, Sa_a]`` (sqrt units), or ``None``
            when not wanted / not available.

        Raises:
            FileNotFoundError: ``psd_from_noise_file=True`` and no brick.
            ValueError: ``psd_from_noise_file=True`` and the fit failed.
        """
        from .noise import noise_params_from_file, resolve_noise_file

        want = gs.psd_from_noise_file
        if want is False:
            return None
        if want is None and gs.data_mode != "mojito":
            return None
        noise_file = resolve_noise_file(gs.mojito_data_path, gs.noise_file)
        if noise_file is None:
            if want:
                raise FileNotFoundError(
                    "psd_from_noise_file=True but no mojito NOISE brick was "
                    "found: set general.noise_file / NOISE_FILE or add "
                    "data/INSTRUMENT/L1/NOISE_* under "
                    f"{gs.mojito_data_path!r}."
                )
            return None
        cache = getattr(self, "_noise_file_params_cache", None)
        if cache is None:
            cache = self._noise_file_params_cache = {}
        if noise_file not in cache:
            cache[noise_file] = noise_params_from_file(
                noise_file, tdi_generation=gs.tdi_gen
            )
        params = cache[noise_file]
        if params is None and want:
            raise ValueError(
                f"psd_from_noise_file=True but the noise-parameter fit failed "
                f"for {noise_file!r} (see the warning above)."
            )
        return list(params) if params is not None else None

    def resolve_noise_source(
        self, gs: EreborGeneralSettings
    ) -> typing.Union[bool, str]:
        """Concrete data-noise source: ``False``, ``"synthetic"``, or ``"mojito"``.

        Resolves a variant's ``add_instrument_noise`` knob: ``True`` (auto)
        becomes ``"mojito"`` — the real NOISE brick summed into the data by
        the L1 loader — when the run uses mojito data and the brick is found,
        else ``"synthetic"`` (the FD-correlated draw). Explicit strings are
        validated and passed through; ``"mojito"`` requires mojito data.
        In production (``ADD_INSTRUMENT_NOISE`` unset -> ``True``, mojito data
        with the NOISE brick present) this resolves to ``"mojito"``.

        Args:
            gs: General block carrying ``add_instrument_noise`` (a variant
                field, e.g. on ``AllSourcesGeneralSettings``).

        Returns:
            bool | str: ``False``, ``"synthetic"`` or ``"mojito"``.

        Raises:
            ValueError: For an unrecognized value, or ``"mojito"`` without
                ``data_mode == "mojito"``.
        """
        from .noise import resolve_noise_file

        value = gs.add_instrument_noise
        if value is True:
            has_brick = (
                gs.data_mode == "mojito"
                and resolve_noise_file(gs.mojito_data_path, gs.noise_file) is not None
            )
            value = "mojito" if has_brick else "synthetic"
            logger.info("add_instrument_noise=True resolved to %r.", value)
        elif value is not False and value not in ("synthetic", "mojito"):
            raise ValueError(
                f"add_instrument_noise={gs.add_instrument_noise!r} not "
                "recognised; use False, True (auto), 'synthetic', or 'mojito'."
            )
        if value == "mojito" and gs.data_mode != "mojito":
            raise ValueError(
                "add_instrument_noise='mojito' (the real NOISE brick) requires "
                f"data_mode='mojito'; got data_mode={gs.data_mode!r}."
            )
        return value

    def _gb_injection_count(self) -> int:
        """Number of prior-drawn synthetic GB rows (mirrors the class counts)."""
        ids = getattr(self.general, "mojito_source_ids", None) or {}
        return len(ids.get("GB", [])) or 2  # 2 matches the stock table

    def _gb_injection_band(self) -> typing.Tuple[float, float]:
        """Frequency band for prior-drawn GB f0s: the GB branch band."""
        gb = self.gb if "gb" in self._branch_names else None
        lo = hi = None
        if gb is not None:
            center = getattr(gb, "center_freq", None)
            if center is not None:
                n_layers = int(getattr(gb, "n_layers", None) or 3)
                half = 0.5 * n_layers * self.layer_df
                lo, hi = center - half, center + half
            else:
                lo = getattr(gb, "min_freq", None)
                hi = getattr(gb, "max_freq", None)
        if lo is None:
            lo = self.general.min_freq
        if hi is None:
            hi = self.general.max_freq
        lo, hi = max(float(lo), 3.0e-4), min(float(hi), 2.3e-2)
        return (lo, hi) if lo < hi else (1.0e-3, 1.0e-2)

    def make_general_settings(self) -> EreborGeneralSettings:
        """Resolve the fit-level general block into the final build-time block.

        See the class docstring for the step order. Always resolved on the
        copy: ``gpus`` / ``gpu_backend``, ``edge_crop_wavelets`` (an unset
        value becomes the auto crop) and, when ``window_taper_wavelets > 0``,
        ``window_tukey_alpha`` (``2 * K / Nt``). Filled only when unset on the
        copy: ``Tobs`` (``Nf * Nt * dt``), ``window_taper_duration``
        (``alpha * Tobs``), ``domain_settings``, ``data_processor_class`` /
        ``processor_init_kwargs``, ``preprocess_kwargs``,
        ``sensitivity_init_kwargs`` and ``fixed_psd_kwargs`` (from
        ``fixed_psd_params``). Logs a warning when ``min_freq`` resolves to
        WDM layer 0 or 1.

        Returns:
            EreborGeneralSettings: The resolved copy (``self.general`` itself
            only receives the :meth:`resolve_data_source` edits).
        """
        self.resolve_data_source()
        gs = deepcopy(self.general)
        self.adjust_general(gs)

        # Compute backend (gpus=None -> engine resolves force_backend="cpu").
        gs.gpus, gs.gpu_backend = resolve_compute(gs.use_gpu, gs.gpu_backend, gs.gpus)

        # WDM grid + exact Tobs (fixed nf/nt override wins over derivation).
        if gs.nf is not None and gs.nt is not None:
            Nf, Nt = gs.nf, gs.nt
            wavelet_duration, tobs = Nf * gs.dt, Nf * Nt * gs.dt
        else:
            Nf, Nt, wavelet_duration, tobs = derive_wdm_grid(
                gs.tobs_target, gs.dt, gs.wavelet_duration_bounds
            )
        if gs.Tobs is None:
            gs.Tobs = tobs

        # Window taper realized from the alpha knob on the resolved grid.
        # window_taper_wavelets > 0 pins the taper to a fixed wavelet count
        # (deterministic per WDM settings) instead of a Tobs fraction.
        if int(getattr(gs, "window_taper_wavelets", 0) or 0) > 0:
            gs.window_tukey_alpha = min(
                1.0, 2.0 * int(gs.window_taper_wavelets) / float(Nt))
        if gs.window_taper_duration is None:
            gs.window_taper_duration = gs.window_tukey_alpha * gs.Tobs
        import logging as _logging
        _logging.getLogger(__name__).info(
            "window taper: %.1f s = %.2f wavelets per side "
            "(alpha=%.5f of Tobs=%.3e s)",
            0.5 * gs.window_taper_duration,
            0.5 * gs.window_taper_duration / (gs.Tobs / Nt),
            gs.window_tukey_alpha, gs.Tobs)

        # WDM time-edge crop: cover boundary wavelets AND the Tukey taper.
        edge_crop = (
            gs.edge_crop_wavelets
            if gs.edge_crop_wavelets is not None
            else default_edge_crop_wavelets(gs.window_tukey_alpha, Nt)
        )
        gs.edge_crop_wavelets = edge_crop

        # ---- DC-ADJACENT LOW-FREQUENCY LAYER GUARD --------------------------
        # WDMSettings sets ind_min_f = ceil(min_freq / layer_df), so a min_freq
        # below ~1.5*layer_df admits layer 1, whose support shares an edge with
        # the DC layer. The analytic instrument model diverges as f -> 0 (Sa_d
        # goes as (2 pi f)^-4), the fold picks that up, and
        # instrument_fill_nans=0.0 turns the resulting NaN into a HARD ZERO --
        # a singular covariance INSIDE the analysed band.
        #
        # This is not hypothetical. Measured on the 3mo v8 store (kappa probe,
        # job 459) with min_freq=1e-4: layer 1 had det(C)=0 and carried
        # q = w^T C^-1 w / 3 = 150.3, i.e. 43% of the whole fit's chi^2 budget,
        # dragging the ML to alpha = sqrt(mean(q/3)) = 1.389 -- the entire
        # observed instrument-PSD bias. It was silent for the whole campaign.
        # The WDM audit's rule is min_freq >= 2 * layer_df.
        _layer_df = 1.0 / (2.0 * Nf * gs.dt)
        _ind_min_f = int(-(-float(gs.min_freq) // _layer_df))  # ceil
        if _ind_min_f < 2:
            logger.warning(
                "min_freq=%.6g Hz resolves to WDM layer %d (layer_df=%.6g Hz). "
                "Layer 1 borders DC, where the instrument model diverges and "
                "fill_nans zeroes the covariance -- a SINGULAR layer inside the "
                "analysed band, measured to carry ~43%% of the chi^2 budget and "
                "to bias the fitted instrument PSD by ~1.39x in amplitude. Set "
                "MIN_FREQ >= %.6g Hz (2 * layer_df) to start at layer 2 or "
                "above; production uses 4e-4.",
                float(gs.min_freq), _ind_min_f, _layer_df, 2.0 * _layer_df,
            )

        # Domain: WDM factory from the grid knobs unless the user swapped in
        # their own DomainSettings instance/factory.
        if gs.domain_settings is None:
            gs.domain_settings = self.make_domain_settings(
                gs, Nf, Nt, wavelet_duration, edge_crop
            )

        # Data processor (whole-object swap wins over the variant default).
        if gs.data_processor_class is None:
            self.set_default_processor(gs)
        if gs.preprocess_kwargs is None:
            gs.preprocess_kwargs = self.default_preprocess_kwargs()
        if gs.sensitivity_init_kwargs is None:
            gs.sensitivity_init_kwargs = dict(tdi_generation=gs.tdi_gen)
        if gs.fixed_psd_kwargs is None and gs.fixed_psd_params is not None:
            gs.fixed_psd_kwargs = dict(
                psd_params=list(gs.fixed_psd_params), galfor_params=None
            )
        self.finalize_general(gs)
        return gs

    def finalize_general(self, gs: EreborGeneralSettings) -> None:
        """Variant hook: last-touch mutation after Tobs/domain are resolved.

        Args:
            gs: The resolved build-time block (mutate in place).
        """

    def make_domain_settings(
        self,
        gs: EreborGeneralSettings,
        Nf: int,
        Nt: int,
        wavelet_duration: float,
        edge_crop: int,
    ):
        """Default domain block: WDM factory on the derived grid.

        Args:
            gs: The build-time general block (``min_freq`` / ``max_freq``).
            Nf: WDM frequency layers.
            Nt: WDM time columns.
            wavelet_duration: ``Nf * dt`` (s).
            edge_crop: Columns excluded at each end
                (``min_time = edge_crop * wavelet_duration``,
                ``max_time = (Nt - edge_crop) * wavelet_duration``).

        Returns:
            callable: :meth:`WDMSettings.make_factory
            <lisatools.domains.WDMSettings.make_factory>` output, a
            ``(times, dt, force_backend) -> WDMSettings`` factory.
        """
        return WDMSettings.make_factory(
            Nf=Nf,
            Nt=Nt,
            min_freq=gs.min_freq,
            max_freq=gs.max_freq,
            min_time=edge_crop * wavelet_duration,
            max_time=(Nt - edge_crop) * wavelet_duration,
        )

    def default_preprocess_kwargs(self) -> dict:
        """Variant hook: kwargs merged over the engine's preprocess defaults.

        Returns:
            dict: ``{"normalize": False}`` in the base.
        """
        return dict(normalize=False)

    def set_default_processor(self, gs: EreborGeneralSettings) -> None:
        """Default data source: mojito L1 loader on ``gs.source_types``.

        Sets ``gs.data_processor_class`` to
        :class:`~lisatools.globalfit.preprocessing.L1ProcessingStep` with
        :class:`~lisatools.detector.L1Orbits` in ``gs.orbits_frame``. Variants
        override it (``all_sources`` uses
        :class:`~lisatools.globalfit.stock.erebor.injections.L1ProcessingStepWithSyntheticNoise`
        for mojito data).

        Args:
            gs: The build-time general block (mutated in place).
        """
        from lisatools.detector import L1Orbits
        from lisatools.globalfit.preprocessing import L1ProcessingStep

        force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
        gs.data_processor_class = L1ProcessingStep
        gs.processor_init_kwargs = dict(
            L1_folder=gs.mojito_data_path,
            source_types=list(gs.source_types),
            source_ids=None,
            orbits_class=L1Orbits,
            orbits_kwargs=dict(force_backend=force_backend, frame=gs.orbits_frame),
            verbose=gs.verbose,
            # NOTE: do NOT pass Tobs here. The engine's preprocess trims the
            # ends and then keeps the first ``Tobs`` seconds, so the final
            # data must be exactly Nf*Nt samples; pre-trimming at load would
            # leave less than Tobs and mismatch the fixed WDM grid.
        )

    # -- branch preparation ---------------------------------------------------

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        """Base resolution plus the shared Erebor bindings for one branch.

        On top of :meth:`StockGlobalFit.prepare_branch_settings
        <lisatools.globalfit.stock.base.StockGlobalFit.prepare_branch_settings>`
        (``Tobs`` / ``dt`` / ``build_seed``), a branch that declares
        ``domain_settings`` and leaves it ``None`` gets the run's resolved
        domain, and an unset ``log_dir`` becomes ``general_setup.file_store_dir``.

        Args:
            name: Branch name.
            general_setup: The built general setup.

        Returns:
            Settings: The completed deepcopy of ``fit.<name>``.
        """
        settings = super().prepare_branch_settings(name, general_setup)
        # Common Erebor bindings: every branch runs on the resolved domain
        # and logs beside the run output.
        if getattr(settings, "domain_settings", None) is None and hasattr(
            settings, "domain_settings"
        ):
            settings.domain_settings = general_setup.domain_settings
        if settings.log_dir is None:
            settings.log_dir = general_setup.file_store_dir
        return settings
