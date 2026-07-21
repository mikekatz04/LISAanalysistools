"""Stock variant ``vgb``: verification-galactic-binary-only global fit.

Fits the mojito L1 VGB brick with the fixed-dimensional 5D VGB branch
(``[lnA, fdot, phi0, cos_iota, psi]`` sampled; f0/sky fixed per leaf from
the catalogue — see :mod:`..vgb`):

* **All catalogue VGBs** (55 in mojito lite) as fixed leaves — leaf i is
  the same physical source at every walker/temperature; NO RJ.
* **Fixed PSD** — no ``psd`` branch (same convention as ``gb_no_fg``).
* One PE stage: the ``vgb_pe`` same-leaf stretch move
  (:class:`~lisatools.globalfit.moves.VGBSpecialStretchMove`).
* Data band sized to the VGB f0 span (0.31–6.22 mHz in mojito lite).

Usage::

    from lisatools.globalfit.stock import erebor

    fit = erebor.vgb(nwalkers=8)
    fit.build()
    fit.run()

``VGB_START_FACTOR=0`` starts every leaf exactly at the catalogue truth
(the default small scatter is ``1e-5``).
"""

from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np

from lisatools.domains import FDSettings, WDMSettings

from ....engine import GeneralSetup, Settings
from ....moves import Move, MoveBuildContext
from ....recipe import Recipe, Stage, build_vgb_moves
from ...base import env_default
from ..fit import EreborFit, EreborGeneralSettings
from ..vgb import VGBSettings, VGBSetup, prepare_vgb_branch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VGBGeneralSettings(EreborGeneralSettings):
    """General block for the ``vgb`` variant."""

    # Data band covering the mojito VGB f0 span (0.31–6.22 mHz) + margin.
    min_freq: float = 2e-4
    max_freq: float = 7e-3
    file_store_dir: str = dataclasses.field(
        default_factory=env_default("FILE_STORE_DIR", "./gf_output_vgb/")
    )
    base_file_name: str = dataclasses.field(
        default_factory=env_default("BASE_FILE_NAME", "vgb_test_1")
    )
    source_types: typing.Tuple[str, ...] = ("VGB",)
    # Fixed PSD (no psd branch): None -> [Soms_d, Sa_a] fit to the mojito
    # NOISE brick when available, else the stock analytic levels.
    fixed_psd_params: typing.Optional[typing.List[float]] = None
    likelihood_source_only: bool = True


class VGBGlobalFit(EreborFit):
    """VGB-only global fit: 55 known binaries, 5D fixed-dimensional PE."""

    option_name = "vgb"
    description = (
        "Verification-GB fit on the mojito L1 VGB brick: fixed-dimensional "
        "5D branch (f0/sky fixed per leaf), same-leaf stretch move, fixed "
        "PSD."
    )
    general_settings_class = VGBGeneralSettings
    setup_classes = {"vgb": VGBSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        return {"vgb": VGBSettings()}

    def default_recipe(self) -> Recipe:
        return Recipe(
            [
                Stage(
                    name="vgb_pe",
                    kind="pe",
                    moves=[Move("vgb_pe", branch="vgb")],
                    combine_kwargs=dict(share_temperature_control=False),
                )
            ]
        )

    def adjust_general(self, gs: VGBGeneralSettings) -> None:
        # Fixed PSD levels: explicit list > mojito NOISE brick fit > stock.
        if gs.fixed_psd_params is None:
            file_params = self.resolve_noise_file_psd_params(gs)
            gs.fixed_psd_params = (
                file_params if file_params is not None else [15e-12, 3e-15]
            )

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        settings = super().prepare_branch_settings(name, general_setup)
        if name != "vgb":
            return settings
        return prepare_vgb_branch(
            settings, general_setup, data_mode=self.general.data_mode
        )


def setup_vgb_moves(engine_info, curr, acs, priors, state) -> dict:
    """Build the ``vgb`` move stack (shared helper, all_sources reuses it).

    WDM path: build the chunked-het ``GBWDMComputations`` for the VGB band
    (post-deepcopy — the orbits wrap is not picklable). FD path: wire the
    orbits/TDI handles for the move's ``GBFDComputations`` prototype. Then
    :func:`~lisatools.globalfit.recipe.build_vgb_moves`.
    """
    general_info = curr.general_info
    gpus = general_info.gpus
    if gpus is not None:
        import cupy as cp

        cp.cuda.runtime.setDevice(gpus[0])

    vgb_info = curr.source_info["vgb"]
    tdi_gen = 2 if getattr(vgb_info, "use_tdi2", True) else 1
    tdi_gen_str = f"{tdi_gen}{'nd' if tdi_gen == 2 else 'st'} generation"

    if (
        isinstance(general_info.domain_settings, WDMSettings)
        and vgb_info.gb_wdm_comp is None
    ):
        from gbgpu.gbcomps import GBWDMComputations

        _wdm = general_info.domain_settings
        _wdm.t0 = float(getattr(general_info, "data_t0", 0.0))
        vgb_info.gb_wdm_comp = GBWDMComputations(
            _wdm,
            t_ref=vgb_info.t0,
            Nt_sub=int(vgb_info.nt_sub),
            n_pad=int(vgb_info.n_pad),
            N_sparse=int(vgb_info.n_sparse),
            N_cp_sig=int(vgb_info.n_cp_sig),
            N_cp_orbit=int(vgb_info.n_cp_orbit),
            orbits=general_info.gpu_orbits,
            tdi_config=tdi_gen_str,
            force_backend=general_info.force_backend,
            tdi_type="XYZ",
        )
        logger.info(
            "Chunked-het VGB likelihood: Nf=%d Nt=%d Nt_sub=%d N_sparse=%d",
            _wdm.Nf, _wdm.Nt,
            vgb_info.gb_wdm_comp.Nt_sub, vgb_info.gb_wdm_comp.N_sparse,
        )

    if isinstance(general_info.domain_settings, FDSettings):
        if getattr(vgb_info, "orbits", None) is None:
            vgb_info.orbits = general_info.gpu_orbits
        if getattr(vgb_info, "tdi_config", None) is None:
            vgb_info.tdi_config = tdi_gen_str

    pe_moves = build_vgb_moves(engine_info, curr, acs, priors, state)
    return {m.name: m for m in pe_moves}


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``vgb`` (the run's ``setup_function``)."""
    general_info = curr.general_info
    stock_moves = setup_vgb_moves(engine_info, curr, acs, priors, state)
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves=stock_moves,
        ntemps=general_info.ntemps, nwalkers=general_info.nwalkers,
    )
    recipe.setup(ctx)


VGBGlobalFit.default_setup_function = staticmethod(setup_recipe)
