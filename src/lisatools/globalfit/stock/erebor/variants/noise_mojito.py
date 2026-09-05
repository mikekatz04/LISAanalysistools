"""Stock variant ``noise_mojito``: instrument-PSD fit on mojito's real noise.

The sibling :mod:`~lisatools.globalfit.stock.erebor.variants.noise` variants fit
a *synthetic* WDM noise realization drawn from the injected covariance — the
data are self-consistent with the model by construction. This variant instead
fits the **instrument-noise realization mojito actually simulated**, read from
the L1 folder's ``data/INSTRUMENT/L1`` stream (the ``"NOISE"`` source type of
:class:`~lisatools.globalfit.preprocessing.L1DataLoader`). It is the honest
end-to-end check of the noise model: nothing about the data is drawn from the
sampled PSD family, so a biased or mis-shaped model shows up as a biased
posterior.

Only the instrument PSD (``psd``, 2 params ``Soms_d``/``Sa_a``) is sampled.
There is **no galfor branch**: mojito's INSTRUMENT stream carries no galaxy
(the confusion foreground lives in the GB source file), so a foreground branch
would have nothing to fit and would rail at its prior edge. ``general
.psd_injection`` is kept as the *reference truth* for the plot overlays — it is
never injected into the data.

The loaded file also carries mojito's own noise estimate
(``f.noise_estimates.xyz``): set ``keep_noise_estimates=True`` to have the
loader stash it (plus the raw noise timeseries) on
``data_processor.individual_timeseries`` under ``PSD_MATRIX`` /
``PSD_FREQUENCIES`` / ``PSD_TIMES``, an independent cross-check for the
recovered PSD.

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.noise_mojito(nwalkers=16)
    fit.build(); fit.run()      # or: run_global.py --stock noise_mojito

Requires a mojito L1 folder (``MOJITO_DATA_PATH`` / ``general
.mojito_data_path``) containing ``data/INSTRUMENT/L1``.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

from ....engine import Settings
from ...base import env_default
from ....moves import Move
from ....recipe import Recipe, Stage
from ..fit import EreborFit
from ..noise import GalForSettings, GalForSetup, PSDSetup
from .noise import NoiseGeneralSettings, NoisePSDSettings, _NoiseFitBase, setup_recipe

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MojitoNoiseGeneralSettings(NoiseGeneralSettings):
    """General block for the mojito noise fit (mojito's native L1 grid).

    The grid is the one the mojito source variants already run on: ``dt = 2.5``
    s with ``Nf = 1440`` / ``Nt = 2160`` (90 d, ~1 h wavelets), over the full
    1e-4 -- 2.5e-2 Hz analysis band so both noise parameters are constrained
    (``Sa_a`` from the low-frequency end, ``Soms_d`` from the high).
    """

    # mojito L1 is sampled at dt = 2.5 s (the noise-variant default is 5.0).
    dt: float = 2.5
    nf: typing.Optional[int] = 1440
    nt: typing.Optional[int] = 2160

    # Full analysis band, not the smoke-friendly 0.3-8 mHz of the synthetic fits.
    # Env-backed (2026-09-04) so a run can pin the SAME band a production GB run
    # uses -- MIN_FREQ/MAX_FREQ, mirroring gb_no_fg. Default 1e-4 unchanged.
    min_freq: float = dataclasses.field(
        default_factory=env_default("MIN_FREQ", 1e-4, float))
    max_freq: float = dataclasses.field(
        default_factory=env_default("MAX_FREQ", 2.5e-2, float))

    # Real data: rectangular window + a wavelet-edge crop, matching the tested
    # mojito settings in ``all_sources`` (the synthetic noise fits crop nothing
    # because their draw fills the full active grid exactly).
    window_tukey_alpha: float = 0.0
    # env-wired (2026-08-19): same EDGE_CROP_WAVELETS shadowing bug as
    # full_year_combined/all_sources -- a noise-only run must be able to
    # match a GB run's domain crop.
    edge_crop_wavelets: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("EDGE_CROP_WAVELETS", 20, int)
    )

    # Load ONLY the instrument-noise stream (data/INSTRUMENT/L1). Adding "GB"
    # here would fold in the whole WDWD galaxy — a different experiment, and one
    # that needs a galfor branch back on the fit to absorb it.
    source_types: typing.Tuple[str, ...] = ("NOISE",)

    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_PROCESSOR", "mojito", str)
    )

    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_noise_mojito/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "noise_mojito")
    )

    # Stash mojito's own noise estimate (f.noise_estimates.xyz) + the raw noise
    # timeseries on the processor for cross-checking the posterior. Costs a copy
    # of the data, so it is off by default.
    keep_noise_estimates: bool = dataclasses.field(
        default_factory=env_default("KEEP_NOISE_ESTIMATES", False, bool)
    )


class MojitoNoiseGlobalFit(_NoiseFitBase):
    """Instrument-PSD fit on mojito's simulated instrument-noise stream."""

    option_name = "noise_mojito"
    description = (
        "Instrument-PSD fit (Soms_d, Sa_a) on mojito's real L1 instrument-noise "
        "realization (data/INSTRUMENT/L1) — no galfor branch, no GW sources, "
        "nothing drawn from the sampled model."
    )
    general_settings_class = MojitoNoiseGeneralSettings
    setup_classes = {"psd": PSDSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        # psd only: the INSTRUMENT stream has no galaxy for a galfor branch.
        return {"psd": NoisePSDSettings()}

    def default_recipe(self) -> Recipe:
        # psd-ONLY PE stage (no galfor branch on this variant). New
        # Move/Recipe/Stage API (2026-09-04 migration; the old
        # RecipeSpec/StageSpec/MoveSpec version was pre-refactor dead code).
        # The shared setup_recipe (default_setup_function, module end)
        # materializes it and likewise degrades to psd-only.
        return Recipe(
            [
                Stage(
                    name="noise_pe",
                    kind="pe",
                    moves=[Move("psd_pe", branch="psd")],
                    combine_kwargs=dict(share_temperature_control=False),
                )
            ]
        )

    def make_domain_settings(self, gs, Nf, Nt, wavelet_duration, edge_crop):
        # Back to the Erebor default (honors the wavelet edge crop); the
        # synthetic noise fits deliberately drop the crop so their (nch, Nf, Nt)
        # draw covers the full grid, which does not apply to real data.
        return EreborFit.make_domain_settings(
            self, gs, Nf, Nt, wavelet_duration, edge_crop
        )

    def default_preprocess_kwargs(self) -> dict:
        # mojito L1 is already conditioned, and the Tobs chop happens at load
        # (below), so the engine's highpass / edge-trim / Tobs-trim would only
        # eat samples the fixed Nf*Nt grid needs. Same choice as all_sources.
        return dict(highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False)

    def set_default_processor(self, gs: MojitoNoiseGeneralSettings) -> None:
        if gs.data_mode != "mojito":
            raise ValueError(
                f"noise_mojito data_mode={gs.data_mode!r}; only 'mojito' is "
                "supported (for a synthetic noise realization use the "
                "'noise_only' variant, or swap data_processor_class)."
            )
        if "NOISE" not in [st.upper() for st in gs.source_types]:
            raise ValueError(
                f"noise_mojito source_types={gs.source_types!r} does not "
                "include 'NOISE' — there would be no noise stream to fit."
            )

        from lisatools.detector import L1Orbits
        from lisatools.globalfit.preprocessing import L1ProcessingStep

        force_backend = gs.gpu_backend if gs.gpus is not None else "cpu"
        gs.data_processor_class = L1ProcessingStep
        gs.processor_init_kwargs = dict(
            L1_folder=gs.mojito_data_path,
            # A fresh list per construction: L1DataLoader.load_data pops "NOISE"
            # off the list it is handed.
            source_types=list(gs.source_types),
            source_ids=None,
            orbits_class=L1Orbits,
            orbits_kwargs=dict(force_backend=force_backend, frame=gs.orbits_frame),
            store_individual_timeseries=gs.keep_noise_estimates,
            verbose=True,
            do_plots=False,
            # Chop to exactly Nf*Nt samples at load, since preprocess does no
            # trimming (see default_preprocess_kwargs).
            Tobs=gs.Tobs,
        )


MojitoNoiseGlobalFit.default_setup_function = staticmethod(setup_recipe)


@dataclasses.dataclass
class MojitoNoiseGalForGeneralSettings(MojitoNoiseGeneralSettings):
    """General block for the joint instrument-PSD + foreground fit on real data.

    Identical to :class:`MojitoNoiseGeneralSettings` except that the data is
    the instrument-noise stream PLUS mojito's galactic-foreground stream, and
    the store defaults are separate (the branch set differs, and a resume
    across a branch-set change fails with a bare ``KeyError``).
    """

    # NOISE + GALFOR: data/INSTRUMENT/L1 summed with data/GALFOR/L1. Both are
    # whole-stream types handled next to each other in L1DataLoader.load_data
    # (GALFOR has no catalogue and no source ids -- see ALLOWED_SOURCES).
    source_types: typing.Tuple[str, ...] = ("NOISE", "GALFOR")

    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_noise_galfor_mojito/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "noise_galfor_mojito")
    )


class MojitoNoiseGalForGlobalFit(MojitoNoiseGlobalFit):
    """Joint instrument-PSD + galactic-foreground fit on mojito's real streams.

    The two-branch sibling of :class:`MojitoNoiseGlobalFit`: same v8 noise
    model, same grid, same exact-fine scoring, but the data is the instrument
    stream **plus** ``data/GALFOR/L1`` and the fit carries a ``galfor`` branch
    (5 params ``amp, fk, alpha, f_1, f_2``) alongside ``psd`` (2 params
    ``Soms_d, Sa_a``), each on its own tempering ladder.

    **What this tests, and what it does NOT.** The mojito-light GALFOR brick
    is *derived_from* the GB brick with "GalacticStochastic resolvable
    binaries subtracted" — it is the REAL unresolved-GB confusion residual,
    not a draw from the analytic ``HyperbolicTangentGalacticForeground`` this
    branch fits. So:

    * ``general.psd_injection`` (from the NOISE brick) IS truth, and whether
      the instrument PSD stays unbiased while galfor absorbs the confusion is
      the headline result — the clean 2-branch version of the ~1.4x
      instrument-PSD bias measured in the full GB production run.
    * ``general.galfor_injection`` is a REFERENCE CURVE, not truth. There is
      no "true" ``(amp, fk, alpha, f_1, f_2)`` for a real GB residual, so a
      galfor parameter offset here is a statement about model adequacy, not
      about recovery. Watch whether ``alpha`` rails at its cap
      (``GALFOR_ALPHA_MAX``): a railed slope means the tanh form cannot take
      the confusion's shape.

    Usage::

        fit = erebor.noise_galfor_mojito(nwalkers=24)
        fit.build(); fit.run()   # or: run_global.py --stock noise_galfor_mojito

    ``nwalkers`` must be at least ``2 * ndim = 10`` — the galfor move is a
    plain stretch without ``live_dangerously``, so eryn refuses fewer.
    """

    option_name = "noise_galfor_mojito"
    description = (
        "Joint instrument-PSD (Soms_d, Sa_a) + galactic-foreground fit on "
        "mojito's real L1 instrument-noise stream summed with the GALFOR "
        "confusion-residual stream — no GW source branches."
    )
    general_settings_class = MojitoNoiseGalForGeneralSettings
    setup_classes = {"psd": PSDSetup, "galfor": GalForSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {"psd": NoisePSDSettings(), "galfor": GalForSettings()}

    def default_recipe(self) -> Recipe:
        # Mirrors NoiseOnlyGlobalFit.default_recipe: the noise-move split
        # gives each branch its own move + ladder inside one PE stage
        # (share_temperature_control=False is what keeps the ladders
        # independent under GFCombineMove). ``general.joint_noise_move``
        # collapses them into ONE move over both branches; honored here so
        # JOINT_NOISE_MOVE=1 cannot leave the recipe asking for a stock name
        # setup_recipe never built.
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
                    combine_kwargs=dict(share_temperature_control=False),
                )
            ]
        )

    def set_default_processor(self, gs: MojitoNoiseGalForGeneralSettings) -> None:
        # The parent validates NOISE and builds the L1 processor kwargs from
        # gs.source_types, which already carries GALFOR. Only the extra
        # requirement is checked here: without the GALFOR stream this is just
        # noise_mojito with an unfittable foreground branch bolted on, which
        # would rail at its prior edge exactly as that variant's docstring
        # warns.
        if "GALFOR" not in [st.upper() for st in gs.source_types]:
            raise ValueError(
                f"noise_galfor_mojito source_types={gs.source_types!r} does "
                "not include 'GALFOR' — there would be no foreground in the "
                "data for the galfor branch to fit (it would rail at its "
                "prior edge). Use the 'noise_mojito' variant for an "
                "instrument-only fit."
            )
        super().set_default_processor(gs)


MojitoNoiseGalForGlobalFit.default_setup_function = staticmethod(setup_recipe)
