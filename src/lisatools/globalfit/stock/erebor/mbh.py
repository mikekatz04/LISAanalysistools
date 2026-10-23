"""MBH (massive-black-hole binary) branch blocks for the Erebor recipe.

Holds the MBH knob block :class:`MBHSettings` (ladder, inner-move stack, eigen
adaptation, sky-mode hops, proposal frame) and its :class:`MBHSetup`, which
fills the sampling info (transform, prior, periodicity, ladder) and the state /
backend classes (:class:`~lisatools.globalfit.state.MBHState`,
:class:`~lisatools.globalfit.hdfbackend.MBHHDFBackend`, re-exported here).

Where it sits in the pipeline: the production variant ``erebor.all_sources``
uses the subclass
:class:`~lisatools.globalfit.stock.erebor.source_runtime.SourceMBHSettings`
as ``fit.mbh``; at build time
:func:`~lisatools.globalfit.stock.erebor.source_runtime.prepare_mbh_branch`
resolves the injection, transform, prior, inner moves and proposal-frame map
on the block BEFORE :class:`MBHSetup` is constructed, so the Setup's own
transform / prior / periodicity / waveform-kwargs / inner-move defaults only
apply to a block that bypasses the stock preparation; its default ladder
(``betas``) applies on the stock path too.
The PE move (``mbh_pe``) is built by
:func:`~lisatools.globalfit.stock.erebor.source_runtime.build_mbh_move_runtime`.
Knobs: ``MBH_*`` (see :class:`MBHSettings`).
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any, Optional

import numpy as np
from eryn.moves import Move
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

from ....sampling.prior import LogUniformLinear
from ....sampling.snrladder import top_snr_value
from ...engine import Settings, Setup
from ...hdfbackend import MBHHDFBackend
from ...loginfo import init_logger
from ..base import env_default
from ...state import MBHState
from .transforms import make_mbh_transform_container

@dataclasses.dataclass
class MBHSettings(Settings):
    """Settings dataclass describing the MBH branch in an Erebor-style recipe.

    Fixed-dimensional branch: one leaf per catalogue MBHB that passes the
    merger-window filter (``nleaves_min == nleaves_max``, set by
    :func:`~lisatools.globalfit.stock.erebor.source_runtime.prepare_mbh_branch`),
    each leaf with its own tempering ladder (``ntemps`` rungs; the engine runs
    cold-chain only). Sampling basis (11, direct ICRS): ``logM, Q, s1z, s2z,
    dist [Gpc], phi_ref, cos_iota, psi, alpha, sin_delta, t_plunge``, with
    ``Q = m1 / m2 >= 1`` the mass ratio (see
    :func:`~lisatools.globalfit.stock.erebor.transforms.make_mbh_transform_container`).
    The production block is the subclass
    :class:`~lisatools.globalfit.stock.erebor.source_runtime.SourceMBHSettings`
    (scoring path, waveform window, warm start). Env knobs are
    ``MBH_<FIELD>`` except where named below; the launcher values are those of
    ``submit_gf_9mo_v9.sh`` / ``submit_gf_12mo.sh`` (the two agree on every
    ``MBH_*`` knob except ``MBH_SOURCE_WARM_START_IDS``). Other launcher MBH knobs
    (``MBH_PERMUTE_EVERY``, ``MBH_COLD_TO_HOT_EVERY``, ``MBH_EIGEN_INFO``,
    ``MBH_START_FACTOR``, ``MBH_CHECK_LL``, ``MBHB_IDS``) are read by the move,
    the recipe or the driver. The fields of the :class:`~lisatools.globalfit.engine.Settings`
    base (``transform``, ``priors``, ``periodic``, ``initialize_kwargs``, ...)
    are filled by ``prepare_mbh_branch`` when unset.

    Attributes:
        waveform_kwargs: Per-call waveform kwargs of the MBH move; ``None`` ->
            ``{}`` at branch prep (a bare :class:`MBHSetup` falls back to the
            legacy BBHx ``modes=[(2, 2)], length=1024``).
        ntemps: Per-leaf ladder rungs. Env ``MBH_NTEMPS``. Default 24;
            launchers 8.
        betas: Explicit per-leaf ladder (wins over ``ntemps`` /
            ``ladder_tmax``); ``None`` ->
            :func:`~lisatools.globalfit.stock.erebor.common.default_source_ladder`.
        inner_moves: Explicit eryn inner-move stack ``[(move, weight), ...]``;
            ``None`` -> built by
            :func:`~lisatools.globalfit.stock.erebor.source_runtime.make_mbh_inner_moves`
            (the ``inner_move_kind`` stack plus the sky-mode hops).
        inner_move_kind: ``"eigen"`` (information-matrix eigen-axis jump) or
            ``"stretch"``. Env ``MBH_INNER_MOVE_KIND``. Default ``"eigen"``;
            launchers ``eigen``.
        sky_move_both_weight: Draw fraction of the full 8-mode sky hop
            (:class:`~lisatools.sampling.moves.skymodehop.SkyMove`
            ``which="both"``, ``coord_frame="icrs"``). Env
            ``MBH_SKY_MOVE_BOTH_WEIGHT``. Default 0.02; launchers 0.02.
        sky_move_lat_weight: Draw fraction of the reflection through the
            constellation plane (``which="lat"``). Env
            ``MBH_SKY_MOVE_LAT_WEIGHT``. Default 0.04; launchers 0.04.
        sky_move_long_weight: Draw fraction of the in-plane rotation by a
            random multiple of 90 degrees (``which="long"``). Env
            ``MBH_SKY_MOVE_LONG_WEIGHT``. Default 0.02; launchers 0.02. Each
            weight must lie in ``[0, 1]`` and their sum below 1 (the
            ``inner_move_kind`` stack takes the remainder); 0 disables a kind.
        eigen_refresh_every: Leaf visits between eigen-table rebuilds (first
            visit, then every N-th). Env ``MBH_EIGEN_REFRESH``. Default 100;
            launchers 10.
        eigen_table_scope: Where tables are built: ``"walker_max"`` (one table
            at the max-lnL cold walker) or ``"per_walker"``. Env
            ``MBH_EIGEN_SCOPE``. Default ``"walker_max"``; launchers
            ``walker_max``.
        num_prop_repeats: Inner-move repeats per leaf visit. Plain default 200
            here; ``SourceMBHSettings`` makes it env-backed
            (``MBH_NUM_PROP_REPEATS``, default 2; launchers 25).
        mbh_search_file_key: Legacy search temp-file key. Read only by the
            legacy ``global_fit_input/emri_mbh_psd_global_fit_setting.py`` and
            ``global_fit_input/mbh_only_global_fit_settings.py``, which import
            the deleted ``lisatools.globalfit.galaxyglobal``; nothing on the
            stock path reads it.
        injection: ``(nleaves, 11)`` sampling-basis start / truth rows; filled
            from the catalogue (or the synthetic injections) by
            ``prepare_mbh_branch``, after the merger-window filter.
        ladder_reset_on_resume: On resume, replace every stored per-leaf ladder
            with the configured one. Env ``MBH_LADDER_RESET_ON_RESUME``.
            Default ``False``.
        ladder_reanchor_nats: Lost-rung guard threshold in nats (a rung that
            far below its leaf's cold rung, beyond the expected tempering
            drop, is re-anchored to the cold state); 0 disables. Env
            ``MBH_LADDER_REANCHOR_NATS``. Default 200.0; launchers 0 (off).
        ladder_tmax: Top temperature of the default ladder (``None`` =
            uncapped ~25 % spacing). Env ``MBH_LADDER_TMAX``; launchers 10.
        ladder_top_snr: SNR-keyed per-leaf ladders (``> 0``: leaf ``j`` gets
            ``Tmax = clip((SNR_j / ladder_top_snr)^2, ladder_tmax_min,
            ladder_tmax)``); 0 = off. Env ``MBH_LADDER_TOP_SNR``. Default 0.0
            (launchers unset).
        ladder_tmax_min: Floor of the SNR-keyed top temperature. Env
            ``MBH_LADDER_TMAX_MIN``. Default 4.0.
        ladder_top_snr_refresh: Proposes between SNR-keyed ladder rebuilds; 0
            = never. Env ``MBH_LADDER_TOP_SNR_REFRESH``. Default 0.
        ladder_top_snr_refresh_pe: Allow those rebuilds in PE stages too. Env
            ``MBH_LADDER_TOP_SNR_REFRESH_PE``. Default ``False``.
        eigen_jump_mix: Fraction of eigen draws taken at ``eigen_jump_factor``
            x the table step (the eigen part of the stack only, never the sky
            hops). Env ``MBH_EIGEN_JUMP_MIX``. Default 0.0; launchers 0.1.
        eigen_jump_factor: Step multiplier of those draws. Env
            ``MBH_EIGEN_JUMP_FACTOR``. Default 3.0; launchers 3.0.
        eigen_axis_adapt: Adapt each table axis's step scale to the chain's
            robust width along it. Env ``MBH_EIGEN_AXIS_ADAPT``. Default
            ``False``; launchers 1.
        eigen_adapt_pool: Adapt from the union of all walkers' rung ensembles
            (walker-block fan-out). Env ``MBH_EIGEN_ADAPT_POOL``. Default
            ``False``; launchers 1.
        eigen_temper_sigmas: Rung ``k`` steps with ``sigma / sqrt(beta_k)``
            (capped). Env ``MBH_EIGEN_TEMPER_SIGMAS``. Default ``False``;
            launchers 1.
        eigen_adapt_max_factor: Cap of an adapted sigma as a multiple of the
            information-matrix sigma (``<= 0`` or ``inf`` = no cap). Env
            ``MBH_EIGEN_ADAPT_MAX_FACTOR``. Default 100.0; launchers 100.
        eigen_axis_acc_log: Per-axis cold / hot acceptance log, once per table
            refresh. Env ``MBH_EIGEN_AXIS_ACC_LOG``. Default ``False``;
            launchers 1.
        eigen_axis_acc_tune: Acceptance-driven per-axis step controller. Env
            ``MBH_EIGEN_AXIS_ACC_TUNE``. Default ``False``; launchers 1.
        eigen_axis_acc_min: Controller lower acceptance target. Env
            ``MBH_EIGEN_AXIS_ACC_MIN``. Default 0.1 (launchers unset).
        eigen_axis_acc_max: Controller upper acceptance target. Env
            ``MBH_EIGEN_AXIS_ACC_MAX``. Default 0.5 (launchers unset).
        eigen_axis_acc_factor: Controller step factor. Env
            ``MBH_EIGEN_AXIS_ACC_FACTOR``. Default 2.0 (launchers unset).
        de_jump_mix: Stack weight of the cross-walker difference jump
            (:class:`~lisatools.sampling.moves.walkerdiffjump.WalkerDifferenceJumpMove`;
            the eigen part of the stack, never the sky hops); 0 = not in the
            stack. Env ``MBH_DE_JUMP_MIX``. Default 0.0; launchers 0.1.
        de_jump_mode_frac: Share of ``gamma = 1`` mode jumps in that move. Env
            ``MBH_DE_JUMP_MODE_FRAC``. Default 0.5 (launchers unset).
        proposal_frame: The coordinates the inner moves PROPOSE in (ruling
            2026-10-19): ``"sampling"`` (the ICRS sampling basis,
            bit-identical to the pre-knob path) or ``"lisa"`` (every inner
            move except the sky hops proposes in ``(lam_L, sinbeta_L, psi_L,
            t_L)``, mapped back to ICRS with the frame Jacobian in the
            Metropolis factor; the eigen tables, the adaptation, the
            controller and the jump partners live in that frame). Read by
            ``ResidualAddOneRemoveOneMove._proposal_frame``, which raises on any
            other value. Env ``MBH_PROPOSAL_FRAME``. Default ``"sampling"``;
            launchers ``lisa``.
        proposal_frame_map: The frame map of ``"lisa"``: a
            :class:`~lisatools.sampling.moves.skymodehop.LISAFrameMap` on
            ``alpha / sin_delta / psi / t_plunge`` with the run's
            constellation phase. COMPUTED by ``prepare_mbh_branch``
            (:func:`~lisatools.globalfit.stock.erebor.source_runtime.make_mbh_proposal_frame_map`),
            never an env knob; ``None`` when the basis is not the stock one.

    Example:
        .. code-block:: python

            from lisatools.globalfit.stock import erebor

            fit = erebor.all_sources(nwalkers=4)
            fit.mbh.ntemps = 8                 # or env MBH_NTEMPS=8
            fit.mbh.ladder_tmax = 10.0
            fit.mbh.proposal_frame = "lisa"    # or env MBH_PROPOSAL_FRAME=lisa
    """

    waveform_kwargs: Optional[dict] = None
    # per-leaf ladder size for the add/remove move (engine is cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("MBH_NTEMPS", 24, int))
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    # default inner-move stack when ``inner_moves`` is unset: "eigen"
    # (information-matrix eigen-axis jump) or "stretch" (legacy escape)
    inner_move_kind: str = dataclasses.field(
        default_factory=env_default("MBH_INNER_MOVE_KIND", "eigen", str)
    )
    # LISA-frame sky-mode hops mixed into the inner-move stack (the
    # ``SkyMove`` with ``coord_frame="icrs"`` -- the stock basis samples
    # alpha / sin_delta / psi in ICRS, so each row is converted to the
    # constellation frame at its own orbital phase, hopped, and converted
    # back; see ``source_runtime.make_mbh_inner_moves``). Fractions of the
    # per-leaf inner-move draws: the full 8-mode hop ("both"), the
    # reflection through the constellation plane ("lat") and the 90-degree
    # in-plane rotation ("long"); the remainder is the ``inner_move_kind``
    # stack. 0.02 / 0.04 / 0.02 (user 2026-10-12); 0 disables a kind.
    sky_move_both_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_BOTH_WEIGHT", 0.02, float)
    )
    sky_move_lat_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_LAT_WEIGHT", 0.04, float)
    )
    sky_move_long_weight: float = dataclasses.field(
        default_factory=env_default("MBH_SKY_MOVE_LONG_WEIGHT", 0.02, float)
    )
    # leaf-visit cadence for the eigen inner-move table refresh — SPARSE:
    # MBH likelihood rows are per-row dense (~1.4 s), so a table build is
    # minutes; it happens on the first visit and then every N-th
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_REFRESH", 100, int)
    )
    # table scope: per-row-dense likelihood -> ONE table, built at the
    # max-lnL cold-chain walker
    eigen_table_scope: str = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_SCOPE", "walker_max", str)
    )
    num_prop_repeats: Optional[int] = 200
    # TODO check: keep/remove/document -- only readers: 2 unimportable global_fit_input files
    mbh_search_file_key: Optional[str] = "_mbh_search_tmp_file"
    injection: Optional[np.ndarray] = None
    # ---- per-leaf ladder + eigen-step knobs (user rulings 2026-10-13) -------
    # Same meaning as the EMRISettings fields of the same names
    # (MBH_LADDER_RESET_ON_RESUME / MBH_LADDER_REANCHOR_NATS /
    # MBH_EIGEN_JUMP_MIX / MBH_EIGEN_JUMP_FACTOR). The jump mixture applies to
    # the eigen part of the stack only, never to the sky-mode hops.
    ladder_reset_on_resume: bool = dataclasses.field(
        default_factory=env_default("MBH_LADDER_RESET_ON_RESUME", False, bool)
    )
    ladder_reanchor_nats: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_REANCHOR_NATS", 200.0, float)
    )
    # Cap on the default ladder's top temperature (None = eryn's uncapped
    # ~25 % spacing, T ~ 150 at 8 rungs). See common.default_source_ladder.
    ladder_tmax: Optional[float] = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TMAX", None, float)
    )
    # SNR-keyed per-leaf ladders (ruling 2026-10-18; default OFF): same meaning
    # as the EMRISettings fields of the same names (MBH_LADDER_TOP_SNR /
    # _TMAX_MIN / _TOP_SNR_REFRESH / _TOP_SNR_REFRESH_PE).
    ladder_top_snr: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR", 0.0, top_snr_value)
    )
    ladder_tmax_min: float = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TMAX_MIN", 4.0, float)
    )
    ladder_top_snr_refresh: int = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR_REFRESH", 0, int)
    )
    ladder_top_snr_refresh_pe: bool = dataclasses.field(
        default_factory=env_default("MBH_LADDER_TOP_SNR_REFRESH_PE", False, bool)
    )
    eigen_jump_mix: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_JUMP_MIX", 0.0, float)
    )
    eigen_jump_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_JUMP_FACTOR", 3.0, float)
    )
    # Same meaning as EMRISettings.eigen_axis_adapt (MBH_EIGEN_AXIS_ADAPT;
    # MBH_EIGEN_ADAPT_VISITS / MBH_EIGEN_ADAPT_MIN_VISITS on the move). The
    # sky-mode hops are not eigen moves and are unaffected.
    eigen_axis_adapt: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ADAPT", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_pool / eigen_temper_sigmas
    # (MBH_EIGEN_ADAPT_POOL / MBH_EIGEN_TEMPER_SIGMAS; ruling 2026-10-15).
    eigen_adapt_pool: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_ADAPT_POOL", False, bool)
    )
    eigen_temper_sigmas: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_TEMPER_SIGMAS", False, bool)
    )
    # Same meaning as EMRISettings.eigen_adapt_max_factor
    # (MBH_EIGEN_ADAPT_MAX_FACTOR, default 100; <= 0 or inf = no cap).
    eigen_adapt_max_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_ADAPT_MAX_FACTOR", 100.0, float)
    )
    # Same meaning as the EMRISettings fields of the 2026-10-18 audit
    # (MBH_EIGEN_AXIS_ACC_LOG / _TUNE / _MIN / _MAX / _FACTOR, MBH_DE_JUMP_MIX /
    # _MODE_FRAC); the DE jump rides the eigen part of the stack, never the
    # sky hops.
    eigen_axis_acc_log: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_LOG", False, bool)
    )
    eigen_axis_acc_tune: bool = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_TUNE", False, bool)
    )
    eigen_axis_acc_min: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_MIN", 0.1, float)
    )
    eigen_axis_acc_max: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_MAX", 0.5, float)
    )
    eigen_axis_acc_factor: float = dataclasses.field(
        default_factory=env_default("MBH_EIGEN_AXIS_ACC_FACTOR", 2.0, float)
    )
    de_jump_mix: float = dataclasses.field(
        default_factory=env_default("MBH_DE_JUMP_MIX", 0.0, float)
    )
    de_jump_mode_frac: float = dataclasses.field(
        default_factory=env_default("MBH_DE_JUMP_MODE_FRAC", 0.5, float)
    )
    # The coordinates the inner moves PROPOSE in (user ruling 2026-10-19:
    # "keep the same sampling/waveform parameters on the MBHB, but when we do
    # the sampling moves, move in the LISA reference frame ... Only the move
    # that produces the new coordinates should be moved in the LISA frame"):
    # "sampling" (default: the ICRS sampling basis, bit-identical) or "lisa"
    # (every inner move -- eigen, jump mixture, cross-walker jump, stretch --
    # proposes in (lam_L, sinbeta_L, psi_L, t_L), mapped back to ICRS with
    # the frame Jacobian in the Metropolis factor; the tables, the
    # adaptation, the controller and the jump partners work in that frame;
    # the sky-mode hops already do). Stamped onto the move with the frame
    # map below (MBH_PROPOSAL_FRAME; ResidualAddOneRemoveOneMove._frame_proposal).
    proposal_frame: str = dataclasses.field(
        default_factory=env_default("MBH_PROPOSAL_FRAME", "sampling", str)
    )
    # The frame map of "lisa" (a LISAFrameMap on alpha / sin_delta / psi /
    # t_plunge with the run's constellation phase); COMPUTED by
    # source_runtime.prepare_mbh_branch (never an env knob).
    proposal_frame_map: Optional[Any] = None



class MBHSetup(Setup):
    """:class:`Setup` for massive black-hole binaries in the Erebor recipe.

    Copies the settings fields onto itself (:class:`~lisatools.globalfit.engine.Setup`),
    opens the ``mbh_setup.log`` logger and runs :meth:`init_setup`. Built by
    the variant's setup classes (``all_sources`` / ``full_year_combined``:
    ``"mbh": MBHSetup``) after
    :func:`~lisatools.globalfit.stock.erebor.source_runtime.prepare_mbh_branch`.

    Args:
        mbh_settings: Settings dataclass with MBH waveform / prior config.
    """

    def __init__(self, mbh_settings: MBHSettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(mbh_settings)

        level = logging.DEBUG
        name = "MBHSetup"
        self.logger = init_logger(filename="mbh_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

        self.init_setup()

    def init_sampling_info(self):
        """Build the MBH :class:`TransformContainer`, prior, periodicity, and waveform kwargs.

        Every piece is filled only when unset, so the stock preparation's values
        (``prepare_mbh_branch``) win: ``transform`` ->
        :func:`~lisatools.globalfit.stock.erebor.transforms.make_mbh_transform_container`;
        ``periodic`` -> ``phi_ref`` 2 pi, ``alpha`` 2 pi, ``psi`` pi; ``priors``
        -> a fallback box (``logM`` in ``[ln 1e5, ln 1e8]``, ``Q`` log-uniform on
        ``[1, 10]`` in the LINEAR column, spins ``(-1, 1)``, ``dist`` ``[1, 150]``
        Gpc, ``t_plunge`` in ``[0, Tobs + 3600]`` s; the stock preparation uses a
        ``dist`` box of ``[0.1, 150]`` Gpc and a ``t_plunge`` box from the data
        start to past the data end by the merger-window buffer); ``betas`` ->
        :func:`~lisatools.globalfit.stock.erebor.common.default_source_ladder`
        (``ntemps`` rungs, capped at ``ladder_tmax``); ``other_tempering_kwargs``
        -> ``permute=False`` (asserted); ``initialize_kwargs`` -> ``{}``;
        ``waveform_kwargs`` -> the legacy BBHx ``modes=[(2, 2)], length=1024``
        (with a WARNING); ``inner_moves`` -> the bare ``inner_move_kind`` stack
        (:func:`~lisatools.globalfit.stock.erebor.common.resolve_inner_moves`,
        no sky hops).

        Raises:
            AssertionError: ``other_tempering_kwargs["permute"]`` is true.
        """

        # ICRS sampling basis (2026-06 run-frame directive): sky and
        # polarization are sampled directly in ICRS (alpha = RA,
        # sin_delta, psi = ICRS polarization), matching
        # ``mbh_catalogue_to_sampling_basis``. No LISA->SSB->ICRS chain;
        # the orbits must be loaded with ``frame='icrs'`` so the response
        # consumes ``(alpha, delta)`` directly.
        input_basis = [
            "logM",
            "Q",
            "s1z",
            "s2z",
            "dist",
            "phi_ref",
            "cos_iota",
            "psi",
            "alpha",
            "sin_delta",
            "t_plunge",
        ]

        if self.transform is None:
            # Stock forward + inverse container (direct ICRS; same basis as
            # ``input_basis`` above). The inverse transforms enable
            # ``both_inverse_transforms`` (full basis -> sampling basis)
            # for injection/diagnostic round trips.
            self.transform = make_mbh_transform_container()

        if self.periodic is None:
            self.periodic = {"mbh": {"phi_ref": 2 * np.pi, "alpha": 2 * np.pi, "psi": np.pi}}

        self.logger.debug("Decide how to treat fdot prior")
        if self.priors is None:
            priors_mbh = {
                "logM": uniform_dist(np.log(1e5), np.log(1e8)),
                # LINEAR-column log-uniform: the transform ``mT_Q`` reads Q as
                # m1/m2 >= 1. eryn's ``log_uniform`` samples/scores in ln-space
                # (support [0, 2.30] here) -- see LogUniformLinear's docstring.
                "Q": LogUniformLinear(1.0, 10.0),
                "s1z": uniform_dist(-0.99999999, +0.99999999),
                "s2z": uniform_dist(-0.99999999, +0.99999999),
                "dist": uniform_dist(1, 150.0), # uniform_dist(0.01, 1000.0),
                "phi_ref": uniform_dist(0.0, 2 * np.pi),
                "cos_iota": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "psi": uniform_dist(0.0, np.pi), #is this right?
                "alpha": uniform_dist(0.0, 2 * np.pi),
                "sin_delta": uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
                "t_plunge": uniform_dist(0.0, self.Tobs + 3600.0),
            }

            self.priors = {"mbh": ProbDistContainer(priors_mbh)}

        if self.betas is None:
            # make_ladder(ndim, ntemps=MBH_NTEMPS), not 1/1.2^k (user ruling
            # 2026-10-13: "the default ladders for non-RJ sources should be
            # from make_ladder"); see common.default_source_ladder. Still no
            # pinned near-prior last rung (the 2026-08-27 ruling: at
            # MBH_NTEMPS=2 ``betas[-1] = 1e-4`` left the branch untempered,
            # job 353). A run that wants another ladder sets ``mbh.betas``
            # explicitly (the field is an override and wins).
            from .common import default_source_ladder

            self.betas = default_source_ladder(self.ndim, input_basis, self.ntemps,
                                               tmax=self.ladder_tmax)

        self.logger.info(f"Using betas: {self.betas} in MBH branch")

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.waveform_kwargs is None:
            self.logger.warning(
                "No waveform kwargs provided for MBHSetup, using defaults. These are the legacy BBHx settings"
            )
            self.waveform_kwargs = dict(
                modes=[(2, 2)],
                length=1024,
            )

        if self.inner_moves is None:
            from .common import resolve_inner_moves

            # Bare ``inner_move_kind`` stack. The stock build path
            # (``source_runtime.prepare_mbh_branch``) fills ``inner_moves``
            # BEFORE this Setup is constructed, with the LISA-frame sky-mode
            # hops mixed in at ``sky_move_*_weight`` (they need the run's
            # orbits for the constellation phase, which only the build has).
            resolve_inner_moves(self)

    def init_setup(self):
        """Run :meth:`init_sampling_info`, then :meth:`init_state_backend_info`."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the MBH state and backend to :class:`MBHState` / :class:`MBHHDFBackend`."""
        if self.branch_state is None:
            self.branch_state = MBHState

        if self.branch_backend is None:
            self.branch_backend = MBHHDFBackend
