"""Stock variant ``blank``: the empty canvas for your own module.

Stock data/general settings, ZERO branches, and a single empty ``"main"`` PE
stage. The expectation: you add a branch (plain branch info is enough) and at
least one move (a plain ``fn(model, state)`` function is enough), then run —
or drive the fit as a generator with ``fit.sample()``.

Usage::

    from eryn.prior import uniform_dist
    from lisatools.globalfit.stock import erebor

    fit = erebor.blank(nwalkers=4, ntemps=2)

    def my_move(model, state):
        aca = model.analysis_container_arr   # read residual, adjust, write back
        return state, None

    fit.add_branch(
        "line", ndim=2, priors={0: uniform_dist(1e-3, 8e-3), 1: uniform_dist(0, 1)},
        moves=[my_move],
    )
    fit.run()                                # or, the generator:
    for model, state in fit.sample(iterations=100):
        ...
    # storage is a choice: the added branch is recorded to the HDF backend
    # by the default storage machinery every step, or pass store=False and
    # record what you want yourself inside the loop.

Data: **all zeros by default** on a small fixed WDM grid — the residual starts
at exactly zero, so the null log-like is exactly 0 and everything you see in
the residual is yours. Adjustable: ``include_noise=True`` (env
``INCLUDE_NOISE``) swaps in a synthetic instrument-noise realization drawn
from ``fixed_psd_params`` (reproducible via ``NOISE_SEED``); bring your own
data by swapping the processor — an explicit ``data_processor_class`` always
wins over the variant default::

    from lisatools.globalfit.stock.erebor.injections import UserDataProcessor

    fit = erebor.blank(
        data_processor_class=UserDataProcessor,
        processor_init_kwargs=dict(data=my_xyz, dt=5.0, orbits=my_orbits),
    )

The PSD is FIXED (there is no psd branch): the sensitivity is built from
``fixed_psd_params`` and the likelihood is source-only.
"""

from __future__ import annotations

import dataclasses
import logging
import typing

from ....engine import Settings
from ....moves import MoveBuildContext
from ....recipe import Recipe, Stage
from ...base import env_default
from ..fit import EreborFit, EreborGeneralSettings

logger = logging.getLogger(__name__)

#: Fixed instrument-noise levels ``(Soms_d, Sa_a)`` (sqrt units) — both the
#: synthetic-data injection and the fixed sensitivity the likelihood uses.
PSD_INJECTION = [15e-12, 3e-15]


@dataclasses.dataclass
class BlankGeneralSettings(EreborGeneralSettings):
    """General block for the blank fit (small smoke-friendly WDM grid).

    ``Tobs == nf*nt*dt`` exactly (256*256*5 s ~ 3.8 d), 0.3-8 mHz band.
    """

    dt: float = 5.0
    nf: typing.Optional[int] = 256
    nt: typing.Optional[int] = 256
    min_freq: float = 3e-4
    max_freq: float = 8e-3
    window_tukey_alpha: float = 0.0  # rectangular; the WDM draw ignores the window
    edge_crop_wavelets: typing.Optional[int] = 0  # no time crop -> draw matches grid
    nwalkers: int = dataclasses.field(default_factory=env_default("NWALKERS", 4, int))
    ntemps: int = dataclasses.field(default_factory=env_default("NTEMPS", 2, int))
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_blank/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "blank_fit")
    )
    source_types: typing.Tuple[str, ...] = ()  # no external source loading
    data_mode: str = dataclasses.field(
        default_factory=env_default("DATA_MODE", "synthetic", str)
    )
    # Zero-noise data by default (the residual starts at exactly 0);
    # include_noise=True draws a synthetic realization from fixed_psd_params.
    include_noise: bool = dataclasses.field(
        default_factory=env_default("INCLUDE_NOISE", False, bool)
    )
    noise_seed: int = dataclasses.field(default_factory=env_default("NOISE_SEED", 0, int))
    t_start: float = 0.0
    # Fixed PSD (no psd branch): the sensitivity is built from these levels
    # and the likelihood is the plain residual inner product. Used for the
    # noise draw too when include_noise=True.
    fixed_psd_params: typing.Optional[typing.Sequence[float]] = dataclasses.field(
        default_factory=lambda: list(PSD_INJECTION)
    )
    likelihood_source_only: bool = True


class BlankGlobalFit(EreborFit):
    """Zero-branch stock fit: add your branch + move(s) and run/sample."""

    option_name = "blank"
    description = (
        "Minimal single-stage fit on zero data (noise optional via "
        "include_noise): stock general settings, zero branches — add a branch "
        "(plain branch info) and a move (plain function) and run, or drive it "
        "with fit.sample()."
    )
    general_settings_class = BlankGeneralSettings
    setup_classes: typing.ClassVar[typing.Dict[str, type]] = {}

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {}

    def default_recipe(self) -> Recipe:
        # One empty PE stage: add_move / add_branch(moves=...) land here.
        return Recipe([Stage("main", kind="pe")])

    def default_preprocess_kwargs(self) -> dict:
        # The synthetic WDM draw needs no highpass/trim/normalize.
        return dict(highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False)

    def set_default_processor(self, gs: BlankGeneralSettings) -> None:
        if gs.data_mode != "synthetic":
            raise ValueError(
                f"blank data_mode={gs.data_mode!r}; the blank fit is synthetic-only. "
                "Bring your own data by swapping data_processor_class wholesale "
                "(e.g. lisatools.globalfit.stock.erebor.injections.UserDataProcessor)."
            )
        if not gs.include_noise:
            # Default: all-zero data — the residual starts at exactly 0.
            from ..injections import ZeroDataProcessingStep

            gs.data_processor_class = ZeroDataProcessingStep
            gs.processor_init_kwargs = dict(Tobs=gs.Tobs, dt=gs.dt, t_start=gs.t_start)
            return
        from ..injections import SyntheticNoiseProcessingStep

        gs.data_processor_class = SyntheticNoiseProcessingStep
        gs.processor_init_kwargs = dict(
            Tobs=gs.Tobs,
            dt=gs.dt,
            t_start=gs.t_start,
            psd_injection=list(gs.fixed_psd_params),
            galfor_injection=None,
            sgwb_injection=None,
            tdi_generation=gs.tdi_gen,
            seed=gs.noise_seed,
        )


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``blank``: no stock moves — exactly what the user added."""
    general_info = curr.general_info
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves={},
        ntemps=general_info.ntemps, nwalkers=general_info.nwalkers,
    )
    recipe.setup(ctx)


BlankGlobalFit.default_setup_function = staticmethod(setup_recipe)
