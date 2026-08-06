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
from ...base import MoveSpec, RecipeSpec, StageSpec, env_default
from ..fit import EreborFit
from ..noise import PSDSetup
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
    min_freq: float = 1e-4
    max_freq: float = 2.5e-2

    # Real data: rectangular window + a wavelet-edge crop, matching the tested
    # mojito settings in ``all_sources`` (the synthetic noise fits crop nothing
    # because their draw fills the full active grid exactly).
    window_tukey_alpha: float = 0.0
    edge_crop_wavelets: typing.Optional[int] = 20

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

    def default_recipe(self) -> RecipeSpec:
        return RecipeSpec(
            [
                StageSpec(
                    name="noise_pe",
                    kind="pe",
                    moves=[MoveSpec("psd_pe", branch="psd")],
                    combine_kwargs=dict(verbose=True, share_temperature_control=False),
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
