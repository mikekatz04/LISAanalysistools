"""Stock variant ``vgb``: verification-galactic-binary-only global fit.

Fits the mojito L1 VGB brick with the fixed-dimensional VGB branch
(default distance basis ``[dist, phi0, cos_iota, psi, fdot_astro_ratio]``
with f0/sky/Mc fixed per leaf from the catalogue; ``VGB_CHIRP_MASS_BASIS=1``
opts into the 6-dim chirp-mass basis with Mc sampled — see :mod:`..vgb`):

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

``data_mode="synthetic"`` (env ``DATA_MODE=synthetic``) builds the data
in-process by injecting ALL catalogue VGBs at the synthetic epoch -- it
needs only the (small) ``catalogues/vgb_cat_mojito_lite_processed.hdf5``
file, not the L1 VGB data brick, so it runs before any data transfer.
Truth-null conventions (``VGB_START_FACTOR=0`` -> residual ~ 0) hold
identically.

Band separations DEFAULT to the same per-WDM-layer edges as the GB branch;
``VGB_BAND_LAYERS=L`` coarsens them to L layers per band. Only bands that
contain VGB leaves do any work (the band sorter holds real leaves only).

Role in production: the 9mo / 12mo runs use :mod:`.all_sources`, whose VGB
branch is the same block (:class:`~lisatools.globalfit.stock.erebor.vgb.VGBSettings`)
built by this module's :func:`setup_vgb_moves`, with the launcher knobs
``VGB_CHIRP_MASS_BASIS=1``, ``VGB_NTEMPS=8``, ``VGB_BAND_LAYERS=8``,
``VGB_START_FACTOR=0`` and ``VGB_SIGHET_INMODEL=1``. The ``vgb`` variant
class itself (and ``vgb_lite``) is the standalone VGB-only fit.
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
from ..gb import build_sighet_engine
from ..vgb import VGBSettings, VGBSetup, prepare_vgb_branch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VGBGeneralSettings(EreborGeneralSettings):
    """General block for the ``vgb`` variant.

    Subclass of :class:`~lisatools.globalfit.stock.erebor.fit.EreborGeneralSettings`
    (inherited fields documented there).

    Attributes:
        min_freq: Data-band floor [Hz] (no env here). Default ``2e-4``.
        max_freq: Data-band ceiling [Hz]. Default ``7e-3``.
        file_store_dir: Output directory. Env ``FILE_STORE_DIR``. Default
            ``"./gf_output_vgb/"``.
        base_file_name: Store file stem. Env ``BASE_FILE_NAME``. Default
            ``"vgb_test_1"``.
        source_types: Mojito streams loaded. Default ``("VGB",)``.
        fixed_psd_params: Fixed ``[Soms_d, Sa_a]``; ``None`` (default) ->
            the NOISE-brick fit, else ``[15e-12, 3e-15]``.
        likelihood_source_only: Default ``True`` (fixed PSD).
    """

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
    """VGB-only global fit: 55 known binaries, fixed-dimensional PE.

    One branch, ``vgb`` (:class:`~lisatools.globalfit.stock.erebor.vgb.VGBSettings`;
    its ``VGB_*`` knobs are documented there), fixed PSD, data from the
    mojito VGB brick (``DATA_MODE=mojito``) or the catalogue injected
    in-process (``"synthetic"``). Default recipe: stage ``vgb_pe`` (kind
    ``pe``) with ``vgb_pe``. Registered as ``vgb`` (laptop twin ``vgb_lite``).
    Note that ``erebor.vgb`` is this variant's default INSTANCE (the branch
    module stays importable as ``lisatools.globalfit.stock.erebor.vgb``).

    Attributes:
        option_name: ``"vgb"``.
        setup_classes: ``{"vgb": VGBSetup}``.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            fit = erebor.vgb(nwalkers=8, data_mode="synthetic")
            fit.vgb.ntemps = 4
            fit.build()
            fit.run()
    """

    option_name = "vgb"
    description = (
        "Verification-GB fit on the mojito L1 VGB brick: fixed-dimensional "
        "branch (f0/sky fixed per leaf; 5-dim distance basis, or the 6-dim "
        "chirp-mass basis with VGB_CHIRP_MASS_BASIS=1), same-leaf stretch "
        "move, fixed PSD."
    )
    general_settings_class = VGBGeneralSettings
    setup_classes = {"vgb": VGBSetup}

    def default_branches(self) -> typing.Dict[str, Settings]:
        """``{"vgb": VGBSettings()}``."""
        return {"vgb": VGBSettings()}

    def default_recipe(self) -> Recipe:
        """One PE stage ``vgb_pe`` with the ``vgb_pe`` move."""
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
        """Variant hook: unset ``fixed_psd_params`` -> NOISE-brick fit, else ``[15e-12, 3e-15]``."""
        # Fixed PSD levels: explicit list > mojito NOISE brick fit > stock.
        if gs.fixed_psd_params is None:
            file_params = self.resolve_noise_file_psd_params(gs)
            gs.fixed_psd_params = (
                file_params if file_params is not None else [15e-12, 3e-15]
            )

    def set_default_processor(self, gs: VGBGeneralSettings) -> None:
        """Mojito: the base L1 loader; synthetic: inject every catalogue VGB in-process.

        The synthetic path reads ``catalogues/vgb_cat_mojito_lite_processed.hdf5``
        under ``mojito_data_path``, converts the rows to physical GBGPU
        parameters and injects them with ``SyntheticGBProcessingStep`` (CPU)
        at ``synthetic_t_start``.

        Args:
            gs: The build-time general block (mutated in place).

        Raises:
            ValueError: On another ``data_mode``.
        """
        if gs.data_mode == "mojito":
            super().set_default_processor(gs)
            return
        if gs.data_mode == "synthetic":
            # Catalogue-faithful in-process injection: build the physical
            # 9-parameter rows for ALL catalogue VGBs from the (small)
            # catalogue file -- no L1 data brick needed -- and inject them
            # with the synthetic GB processor at the synthetic epoch (the
            # branch's t0 resolves to the same epoch in prepare_vgb_branch,
            # so a VGB_START_FACTOR=0 run is truth-null exactly like the
            # mojito-mode checks).
            from ..common import tdi_generation_info
            from ..injections import SyntheticGBProcessingStep
            from ..transforms import make_gb_transform_container
            from ..vgb import load_vgb_catalogue_file
            from ....recipe import gb_catalogue_to_sampling_basis

            cat = load_vgb_catalogue_file(gs.mojito_data_path)
            rows = np.array(
                [gb_catalogue_to_sampling_basis(cat[k]) for k in sorted(cat)]
            )
            if rows.ndim == 3:
                rows = rows.reshape(-1, rows.shape[-1])
            tc = make_gb_transform_container(use_chirp_mass=False)
            params_phys = np.asarray(tc.both_transforms(rows), dtype=float)

            gs.data_processor_class = SyntheticGBProcessingStep
            gs.processor_init_kwargs = dict(
                Tobs=self.wdm_grid[3],
                dt=gs.dt,
                t_start=gs.synthetic_t_start,
                injection_params=np.atleast_2d(params_phys),
                tdi_chan=gs.tdi_chan,
                nchannels=gs.nchannels,
                verbose=gs.verbose,
                use_tdi2=tdi_generation_info(gs.tdi_chan)[0] == 2,
                force_backend="cpu",
            )
            logger.info(
                "VGB synthetic mode: injecting %d catalogue VGBs in-process "
                "(catalogue file only; no L1 data).", params_phys.shape[0],
            )
            return
        raise ValueError(
            f"vgb data_mode={gs.data_mode!r} not recognised; use 'mojito' or "
            "'synthetic' (or swap data_processor_class wholesale)."
        )

    def default_preprocess_kwargs(self) -> dict:
        """Synthetic data: no highpass / trim; mojito: the Erebor default."""
        if self.general.data_mode == "synthetic":
            # The synthetic stream covers exactly Tobs = Nf*Nt*dt; skip the
            # engine's default highpass + edge-trim + Tobs trim so the WDM
            # shape stays exact (same convention as gb_no_fg synthetic).
            return dict(
                highpass_kwargs=None, trim_kwargs=None, Tobs=None, normalize=False
            )
        return super().default_preprocess_kwargs()

    def prepare_branch_settings(self, name: str, general_setup: GeneralSetup) -> Settings:
        """Erebor completion, then ``prepare_vgb_branch`` for ``vgb``."""
        settings = super().prepare_branch_settings(name, general_setup)
        if name != "vgb":
            return settings
        return prepare_vgb_branch(
            settings, general_setup, data_mode=self.general.data_mode,
            synthetic_t_start=self.general.synthetic_t_start,
        )

    def attach_runtime_objects(self) -> None:
        """Register the engine-side ``signal_gen`` on the vgb Setup.

        One GB-family generator (:class:`~lisatools.globalfit.recipe.
        GBFillGlobalSignalGen`) prepares transform/index/factor inputs for
        the GBGPU ``fill_global`` kernels. With it registered, run.py's
        ``setup_acs(rebuild_residuals=True)`` subtracts the seeded VGB
        templates itself (initial log likelihood reads ~0 at
        ``VGB_START_FACTOR=0``) and the recipe's manual subtraction block
        skips (it gates on ``signal_gen is None``); warm-start reloads
        rebuild identically.
        """
        from ....recipe import GBFillGlobalSignalGen

        info = self.source_info["vgb"]
        info.signal_gen = GBFillGlobalSignalGen(
            "vgb", info.transform, self.general_info, info
        )


def setup_vgb_moves(engine_info, curr, acs, priors, state) -> dict:
    """Build the ``vgb`` move stack (shared helper, all_sources reuses it).

    WDM path: build the chunked-het ``GBWDMComputations`` for the VGB band
    (post-deepcopy — the orbits wrap is not picklable). FD path: wire the
    orbits/TDI handles for the move's ``GBFDComputations`` prototype. Then
    :func:`~lisatools.globalfit.recipe.build_vgb_moves`.

    With ``sighet_inmodel`` (``VGB_SIGHET_INMODEL``, falling back to
    ``GB_SIGHET_INMODEL``; launchers ``VGB_SIGHET_INMODEL=1``) the chunked
    comp is wrapped once in the sig-het engine
    (:func:`~lisatools.globalfit.stock.erebor.gb.build_sighet_engine`, the
    same builder as the GB branch). Called by ``all_sources.setup_recipe``
    in production.

    Args:
        engine_info: The engine settings.
        curr: The built current info (``source_info["vgb"]``).
        acs: The rank's ``AnalysisContainerArray``.
        priors (dict): Per-branch priors.
        state (GFState): The starting state.

    Returns:
        dict: ``{move name: move}`` (``vgb_pe``, plus ``vgb_ridge_gibbs``
        when the branch is ridge-Gibbs eligible, i.e. the chirp-mass basis).
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

    # Sig-het in-model scoring: wrap the chunked comp so ONLY the in-model
    # repeat proposals route through the sig-het kernel; fills / swaps /
    # information matrices delegate to the chunked comp byte-identically
    # (pure type dispatch). Mirrors setup_gb_moves' block in gb_no_fg.
    #
    # VGB is FIXED-DIMENSIONAL -- no RJ leg, a catalogue-fixed source count,
    # and f0/sky fixed per leaf -- so this is the clean apples-to-apples A/B
    # against pure chunked-het: same sources, same moves, same iterations,
    # only the in-model likelihood differs. The GB branch's births/deaths
    # change the work per iteration between arms and confound the timing.
    #
    # NOTE the knob is VGB_SIGHET_INMODEL when set, else the shared
    # GB_SIGHET_INMODEL (VGBSettings' default factory), which in a run
    # carrying BOTH branches -- e.g. all_sources -- flips gb and vgb together.
    if isinstance(general_info.domain_settings, WDMSettings) and getattr(
        vgb_info, "sighet_inmodel", False
    ):
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

        # Idempotent: setup runs again on resume / all_sources reuse, and the
        # chunked build above is skipped once gb_wdm_comp is non-None, so a
        # second pass must not wrap the wrapper.
        if not isinstance(vgb_info.gb_wdm_comp, GBSignalHetComputations):
            # SAME builder as the GB branch (erebor.gb.build_sighet_engine):
            # the v5 gating + edge-exclusion checks and every for_band_engine
            # knob, the Tukey alpha included. Until 2026-09-17 this call
            # passed no tukey_alpha and the engine inherited the chunked
            # delegate's 0.05 as a whole-observation taper (~110 layers a
            # side at 6mo against a 60-layer crop, engine WARNING only) --
            # the likely source of the [GB_CELL_LL] growth that had
            # VGB_SIGHET_INMODEL pinned to 0 in the 6mo campaign.
            vgb_info.gb_wdm_comp = build_sighet_engine(
                vgb_info, vgb_info.gb_wdm_comp, branch="vgb"
            )
            logger.info(
                "VGB in-model likelihood: SIGNAL-HET (tukey_alpha=%s, "
                "nt_layer=%d, n_sparse_fd=%d, max_r=%s, n_cp=%d, "
                "v3_nodes=%d, v4_knots=%d, v4_band=%d, v5=%d; chunked-het "
                "delegate for fills / swaps).",
                float(vgb_info.sighet_tukey_alpha),
                int(vgb_info.sighet_nt_layer),
                int(vgb_info.sighet_n_sparse_fd),
                float(getattr(vgb_info, "sighet_max_r", 0.0)),
                int(vgb_info.gb_wdm_comp._g["n_cp_build"]),
                int(vgb_info.gb_wdm_comp._g.get("v3_n_nodes", 0)),
                int(vgb_info.gb_wdm_comp._g.get("v4_knots", 0)),
                int(vgb_info.gb_wdm_comp._g.get("v4_band", 0)),
                int(vgb_info.sighet_v5),
            )

    if isinstance(general_info.domain_settings, FDSettings):
        if getattr(vgb_info, "orbits", None) is None:
            vgb_info.orbits = general_info.gpu_orbits
        if getattr(vgb_info, "tdi_config", None) is None:
            vgb_info.tdi_config = tdi_gen_str

    pe_moves = build_vgb_moves(engine_info, curr, acs, priors, state)
    return {m.name: m for m in pe_moves}


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Recipe setup for ``vgb`` (the run's ``setup_function``).

    Builds :func:`setup_vgb_moves` and materializes the recipe. Arguments as
    for :func:`setup_vgb_moves`.
    """
    general_info = curr.general_info
    stock_moves = setup_vgb_moves(engine_info, curr, acs, priors, state)
    ctx = MoveBuildContext(
        recipe=recipe, engine_info=engine_info, curr=curr, acs=acs,
        priors=priors, state=state, stock_moves=stock_moves,
        ntemps=general_info.ntemps, nwalkers=general_info.nwalkers,
    )
    recipe.setup(ctx)


VGBGlobalFit.default_setup_function = staticmethod(setup_recipe)
