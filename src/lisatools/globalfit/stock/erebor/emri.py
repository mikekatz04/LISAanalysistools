"""EMRI (extreme-mass-ratio inspiral) branch blocks for the Erebor recipe.

:class:`EMRISettings` is the EMRI knob block (``fit.emri``) and
:class:`EMRISetup` turns it into the built branch (transform, prior, per-leaf
ladder, inner-move stack). The production ``all_sources`` variant uses the
subclass :class:`~lisatools.globalfit.stock.erebor.source_runtime.SourceEMRISettings`
(likelihood path, batching, warm start), filled by
:func:`~lisatools.globalfit.stock.erebor.source_runtime.prepare_emri_branch`
from the mojito EMRI catalogue (``EMRI_IDS``); the ``emri_pe`` move is the
direct-to-WDM ``EMRIDirectLikeMove`` (``EMRI_LIKELIHOOD=direct``) on the
add-one/remove-one machinery.
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

from ....sampling.prior import EMRIKerrDomainPrior
from ....sampling.snrladder import top_snr_value

from ...engine import Settings, Setup
from ...hdfbackend import EMRIHDFBackend
from ...loginfo import init_logger
from ..base import env_default
from ...state import EMRIState
from .transforms import make_emri_transform_container

@dataclasses.dataclass
class EMRISettings(Settings):
    """Settings dataclass describing the EMRI branch in an Erebor-style recipe.

    Holds parameter ranges for the intrinsic EMRI parameters (primary mass,
    secondary mass, spin, semi-latus rectum, eccentricity), the waveform
    configuration, and search/sampling helpers consumed by :class:`EMRISetup`.

    Fixed-dimensional branch: one leaf per catalogue source
    (``nleaves_min == nleaves_max``, set by ``prepare_emri_branch``), each
    leaf with its own tempering ladder (``ntemps`` rungs; the engine runs
    cold-chain only). Sampling basis (12): ``logm1, m2, a, p0, e0, dist
    [Gpc], cosqS, phiS, cosqK, phiK, Phi_phi0, Phi_r0`` (see
    :func:`~lisatools.globalfit.stock.erebor.transforms.make_emri_transform_container`).
    Env knobs are ``EMRI_<FIELD>``; the launcher values below are those of
    ``submit_gf_9mo_v9.sh`` / ``submit_gf_12mo.sh``. Other launcher EMRI
    knobs (``EMRI_PERMUTE_EVERY``, ``EMRI_COLD_TO_HOT_EVERY``,
    ``EMRI_EIGEN_INFO``, ``EMRI_START_FACTOR``, ``EMRI_CHECK_LL``,
    ``EMRI_IDS``) are read by the move / recipe / driver, and the
    ``EMRI_LIKELIHOOD`` / ``EMRI_DIRECT_*`` / ``EMRI_NUM_PROP_REPEATS`` /
    warm-start knobs are fields of ``SourceEMRISettings``.

    Attributes:
        logm1_lims: ``[lo, hi]`` of the ``ln(m1 / Msol)`` uniform prior; empty
            or ``None`` keeps the wide default ``[ln 3e5, ln 1.5e7]``
            (``prepare_emri_branch`` turns an empty list into ``None``; a
            direct :class:`EMRISetup` needs ``None`` or a 2-list).
        m2_lims: Secondary-mass box (Msol); default ``[1, 200]``.
        ntemps: Per-leaf ladder rungs. Env ``EMRI_NTEMPS``. Default 24;
            launchers 12.
        a_lims: Spin box; default ``[0.01, 0.999]``.
        p0_lims: Initial semi-latus rectum box; default ``[2, 100]``.
        e0_lims: Initial eccentricity box; default ``[0.001, 0.8]``. A joint
            FEW domain-of-validity cut on ``(a, p0, e0)``
            (:class:`~lisatools.sampling.prior.EMRIKerrDomainPrior`) applies on
            top of the boxes.
        waveform_kwargs: Waveform kwargs that double as the EMRI move's
            likelihood kwargs; ``None`` -> ``{}`` at branch prep.
        eps: FEW mode-selection threshold (``mode_selection_threshold``),
            delivered per call through the wave wrap's ``runtime_kwargs``.
            ``None`` keeps FEW's own default (1e-5). Env ``EMRI_EPS``;
            launchers 1e-3.
        injection: ``(nleaves, 12)`` sampling-basis start/truth rows; filled
            from the catalogue by ``prepare_emri_branch``.
        info_matrix_gen: Optional external eigen-table builder for the eigen
            inner move, ``builder(move, leaf, widths) -> (axes, sigmas)``
            (installed as the move's ``eigen_table_builder``); ``None`` keeps
            the move's own information-matrix route (``EMRI_EIGEN_INFO``,
            launchers ``gram``).
        fill_values: Transform fills ``[xI0, Phi_theta0]`` — one pair for every
            leaf or an ``(nleaves, 2)`` per-leaf table (``xI0`` is the
            prograde / retrograde flag). ``None`` -> resolved per leaf from the
            catalogue, with a ``[1.0, 0.0]`` (prograde) fallback.
        betas: Explicit per-leaf ladder (wins over ``ntemps`` /
            ``ladder_tmax``); ``None`` ->
            :func:`~lisatools.globalfit.stock.erebor.common.default_source_ladder`.
        inner_moves: Explicit eryn inner-move stack ``[(move, weight), ...]``;
            ``None`` -> built from ``inner_move_kind`` and the jump knobs
            (:func:`~lisatools.globalfit.stock.erebor.common.make_default_inner_moves`).
        inner_move_kind: ``"eigen"`` (information-matrix eigen-axis jump) or
            ``"stretch"``. Env ``EMRI_INNER_MOVE_KIND``. Default ``"eigen"``.
        eigen_refresh_every: Leaf visits between eigen-table rebuilds. Env
            ``EMRI_EIGEN_REFRESH``. Default 100; launchers 10.
        eigen_table_scope: Where tables are built: ``"walker_max"`` (one table
            at the max-lnL cold walker) or ``"per_walker"`` (one per
            (temperature, walker) point). Env ``EMRI_EIGEN_SCOPE``. Default
            ``"walker_max"``; launchers ``walker_max``.
        num_prop_repeats: Inner-move repeats per leaf visit. Plain default 10
            here; ``SourceEMRISettings`` makes it env-backed
            (``EMRI_NUM_PROP_REPEATS``, default 2; launchers 25).
        emri_search_file_key: Legacy search temp-file key; no reader.
        ladder_reset_on_resume: On resume, replace every stored per-leaf ladder
            with the configured one (logged per leaf). Env
            ``EMRI_LADDER_RESET_ON_RESUME``. Default ``False``.
        ladder_reanchor_nats: Lost-rung guard: a rung more than this many nats
            below its leaf's cold rung (beyond the expected tempering drop) is
            re-anchored to the cold state; 0 disables. Env
            ``EMRI_LADDER_REANCHOR_NATS``. Default 200.0; launchers 100.
        ladder_tmax: Top temperature of the default ladder (``None`` =
            uncapped ~25 % spacing). Env ``EMRI_LADDER_TMAX``; launchers 625.
        ladder_top_snr: SNR-keyed per-leaf ladders: ``> 0`` gives leaf ``j``
            ``Tmax = clip((SNR_j / ladder_top_snr)^2, ladder_tmax_min,
            ladder_tmax)``; 0 = off; on/true/yes = 2.0. Env
            ``EMRI_LADDER_TOP_SNR``. Default 0.0 (launchers unset).
        ladder_tmax_min: Floor of the SNR-keyed top temperature. Env
            ``EMRI_LADDER_TMAX_MIN``. Default 4.0.
        ladder_top_snr_refresh: Proposes between SNR-keyed ladder rebuilds of
            leaves whose SNR moved > 20 %; 0 = never. Env
            ``EMRI_LADDER_TOP_SNR_REFRESH``. Default 0.
        ladder_top_snr_refresh_pe: Allow those rebuilds in PE stages too
            (adaptive, not detailed-balance exact). Env
            ``EMRI_LADDER_TOP_SNR_REFRESH_PE``. Default ``False``.
        eigen_jump_mix: Fraction of eigen draws taken at ``eigen_jump_factor``
            x the table step (0 = plain ``EigenAxisMove``). Env
            ``EMRI_EIGEN_JUMP_MIX``. Default 0.0; launchers 0.1.
        eigen_jump_factor: Step multiplier of those draws. Env
            ``EMRI_EIGEN_JUMP_FACTOR``. Default 3.0; launchers 3.0.
        eigen_axis_adapt: Adapt each table axis's step scale to the chain's
            robust width along it (floor = the matrix sigma). Env
            ``EMRI_EIGEN_AXIS_ADAPT``. Default ``False``; launchers 1.
        eigen_adapt_pool: Adapt from the union of all walkers' rung ensembles
            (walker-block fan-out) instead of the rank's own walker; needs
            ``eigen_axis_adapt``. Env ``EMRI_EIGEN_ADAPT_POOL``. Default
            ``False``; launchers 1.
        eigen_temper_sigmas: Rung ``k`` steps with ``sigma / sqrt(beta_k)``
            (capped) instead of the cold sigma. Env
            ``EMRI_EIGEN_TEMPER_SIGMAS``. Default ``False``; launchers 1.
        eigen_adapt_max_factor: Cap of an adapted sigma, as a multiple of the
            information-matrix sigma (``<= 0`` or ``inf`` = no cap). Env
            ``EMRI_EIGEN_ADAPT_MAX_FACTOR``. Default 100.0; launchers 100.
        eigen_axis_acc_log: Per-axis cold / hot acceptance log, once per table
            refresh (telemetry only). Env ``EMRI_EIGEN_AXIS_ACC_LOG``. Default
            ``False``; launchers 1.
        eigen_axis_acc_tune: Acceptance-driven per-axis step controller (an
            axis below ``eigen_axis_acc_min`` cold acceptance has its step
            divided by ``eigen_axis_acc_factor``, above ``eigen_axis_acc_max``
            multiplied; fixed between refreshes). Env
            ``EMRI_EIGEN_AXIS_ACC_TUNE``. Default ``False``; launchers 1.
        eigen_axis_acc_min: Controller lower acceptance target. Env
            ``EMRI_EIGEN_AXIS_ACC_MIN``. Default 0.1; launchers 0.1.
        eigen_axis_acc_max: Controller upper acceptance target. Env
            ``EMRI_EIGEN_AXIS_ACC_MAX``. Default 0.5; launchers 0.5.
        eigen_axis_acc_factor: Controller step factor. Env
            ``EMRI_EIGEN_AXIS_ACC_FACTOR``. Default 2.0; launchers 2.
        de_jump_mix: Stack weight of the cross-walker difference jump
            (:class:`~lisatools.sampling.moves.walkerdiffjump.WalkerDifferenceJumpMove`);
            0 = not in the stack. Env ``EMRI_DE_JUMP_MIX``. Default 0.0;
            launchers 0.2.
        de_jump_mode_frac: Share of ``gamma = 1`` mode jumps in that move.
            Env ``EMRI_DE_JUMP_MODE_FRAC``. Default 0.5; launchers 0.5.

    Example:
        .. code-block:: python

            fit = erebor.all_sources(nwalkers=4)
            fit.emri.ntemps = 12            # or env EMRI_NTEMPS=12
            fit.emri.ladder_tmax = 625.0
            fit.emri.eps = 1e-3
    """

    logm1_lims: typing.List[float] = dataclasses.field(default_factory=list)
    m2_lims: typing.List[float] = dataclasses.field(default_factory=list)
    # per-leaf ladder size for the add/remove move (engine is cold-chain only)
    ntemps: int = dataclasses.field(default_factory=env_default("EMRI_NTEMPS", 24, int))
    a_lims: typing.List[float] = dataclasses.field(default_factory=list)
    p0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    e0_lims: typing.List[float] = dataclasses.field(default_factory=list)
    waveform_kwargs: Optional[dict] = None
    # Mode-selection threshold (FEW's ``mode_selection_threshold``; ``eps`` was
    # its FEW 1.x name and is what everyone still calls it). Raising it drops
    # harmonics -> faster, less accurate; lowering it keeps more. None keeps
    # FEW's own per-call default (1e-5, ``few/waveform/base.py:143``) -- the
    # LAT constructor pin in ``lisatools.sources.emri.response`` does NOT
    # apply, because FEW passes its call-time default to the selector
    # unconditionally. A set value is delivered as the per-call
    # ``runtime_kwargs`` of the EMRI wave wrap (the one wrap both the
    # engine-side and move-side generators resolve through), NOT through the
    # branch ``waveform_kwargs`` -- those double as the EMRI move's LIKELIHOOD
    # kwargs, where this key is a TypeError in ``inner_product``. An explicit
    # ``waveform_kwargs["mode_selection_threshold"]`` still wins over this
    # field as the source of the value.
    eps: Optional[float] = dataclasses.field(
        default_factory=env_default("EMRI_EPS", None, float)
    )
    injection: Optional[np.ndarray] = None  # AS here only for the starting state
    info_matrix_gen: Optional[Any] = None  # todo change name to info matrix or smth
    # ``[xI0, Phi_theta0]`` transform fills. None -> resolved per leaf from
    # the injection catalogue in ``prepare_emri_branch`` (an ``(nleaves, 2)``
    # table — xI0 is the intrinsic prograde/retrograde flag and can differ
    # per leaf), with a ``[1.0, 0.0]`` prograde fallback in
    # ``init_sampling_info`` when nothing resolves it.
    fill_values: Optional[np.ndarray] = None
    betas: Optional[np.ndarray] = None
    inner_moves: Optional[typing.List[Move]] = None
    # default inner-move stack when ``inner_moves`` is unset: "eigen"
    # (information-matrix eigen-axis jump) or "stretch" (legacy escape)
    inner_move_kind: str = dataclasses.field(
        default_factory=env_default("EMRI_INNER_MOVE_KIND", "eigen", str)
    )
    # leaf-visit cadence for the eigen inner-move table refresh — SPARSE:
    # EMRI likelihood rows are per-row dense (~1 s), so a table build is
    # minutes; it happens on the first visit and then every N-th
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_REFRESH", 100, int)
    )
    # table scope: per-row-dense likelihood -> ONE table, built at the
    # max-lnL cold-chain walker
    eigen_table_scope: str = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_SCOPE", "walker_max", str)
    )
    num_prop_repeats: Optional[int] = 10
    # TODO check: keep/remove/document -- field has no reader anywhere (legacy search temp-file key)
    emri_search_file_key: Optional[str] = "_emri_search_tmp_file"
    # ---- per-leaf ladder + eigen-step knobs (user rulings 2026-10-13) -------
    # RESUME: the store's per-leaf ladders (betas_all) are kept; True replaces
    # every leaf's ladder with this branch's configured ladder (logged per
    # leaf, old -> new). One-off launch-line knob (EMRI_LADDER_RESET_ON_RESUME=1);
    # inert on a fresh store. See recipe.SingleSourcePEBuilder.build.
    ladder_reset_on_resume: bool = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_RESET_ON_RESUME", False, bool)
    )
    # Lost-rung guard of the add/remove move: at each leaf visit a rung whose
    # lnL sits more than this many nats below the leaf's cold rung (beyond
    # the rung's expected tempering drop) is re-anchored to the cold state.
    # 0 disables. See ResidualAddOneRemoveOneMove._reanchor_lost_rungs.
    ladder_reanchor_nats: float = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_REANCHOR_NATS", 200.0, float)
    )
    # Cap on the default ladder's top temperature (None = eryn's uncapped
    # ~25 % spacing, T ~ 120 at 8 rungs). See common.default_source_ladder
    # for why a loud source wants ~10. Ignored when ``betas`` is explicit.
    ladder_tmax: Optional[float] = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_TMAX", None, float)
    )
    # ---- SNR-keyed per-leaf ladders (user ruling 2026-10-18; default OFF) ---
    # > 0: each leaf's ladder is make_ladder(ndim, ntemps, Tmax=T_leaf) with
    # T_leaf = clip((SNR_leaf / ladder_top_snr)^2, ladder_tmax_min,
    # ladder_tmax) -- ladder_tmax is the CEILING then -- so the top rung sees
    # every leaf at effective SNR ~ladder_top_snr. SNR_leaf = sqrt(<h|h>) at the
    # max-lnL cold walker. Acts at a fresh store's ladder construction (deferred
    # to the leaf's first recorded SNR when the store has none) and at
    # EMRI_LADDER_RESET_ON_RESUME=1; a plain resume keeps the stored ladders.
    # 0 = off (bit-identical); on / true / yes = 2.0. See
    # lisatools.sampling.snrladder and recipe.resolve_snr_keyed_ladders.
    ladder_top_snr: float = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_TOP_SNR", 0.0, top_snr_value)
    )
    # the floor of a leaf's SNR-keyed top temperature (prior-only leaves sit here)
    ladder_tmax_min: float = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_TMAX_MIN", 4.0, float)
    )
    # every N proposes, rebuild (on the head, between proposes) the ladder of a
    # leaf whose SNR moved > 20 % since its ladder was built; 0 = never. An
    # ADAPTIVE change (not detailed-balance exact while it happens): search
    # stages only unless ladder_top_snr_refresh_pe. See
    # ResidualAddOneRemoveOneMove._maybe_update_snr_ladders.
    ladder_top_snr_refresh: int = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_TOP_SNR_REFRESH", 0, int)
    )
    ladder_top_snr_refresh_pe: bool = dataclasses.field(
        default_factory=env_default("EMRI_LADDER_TOP_SNR_REFRESH_PE", False, bool)
    )
    # Eigen inner move: this fraction of the proposals draws its axis step at
    # eigen_jump_factor x the table width (0 = every draw bit-identical to the
    # plain EigenAxisMove). See lisatools.sampling.moves.eigenjumpmix.
    eigen_jump_mix: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_JUMP_MIX", 0.0, float)
    )
    eigen_jump_factor: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_JUMP_FACTOR", 3.0, float)
    )
    # Eigen inner move (user ruling 2026-10-14): adapt each table axis's step
    # SCALE to the chain's own robust width along it (the information-matrix
    # directions are kept; floor = the matrix sigma, cap = the table's own
    # cap). Off = every table bit-identical to the information-matrix one.
    # The ring-buffer knobs are move-level env knobs
    # (EMRI_EIGEN_ADAPT_VISITS 50 / EMRI_EIGEN_ADAPT_MIN_VISITS 10). See
    # ResidualAddOneRemoveOneMove._eigen_adapt_table.
    eigen_axis_adapt: bool = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ADAPT", False, bool)
    )
    # Eigen inner move (user ruling 2026-10-15, "just do 1 for now"):
    # under the walker-block fan-out, adapt every rank's step scale from the
    # UNION of all walkers' rung ensembles (the head ships the full
    # ensemble's rung coordinates in every propose payload) instead of the
    # rank's own walker. Acts only with eigen_axis_adapt. Off = the per-rank
    # buffer, bit-identical. See
    # ResidualAddOneRemoveOneMove._note_eigen_pool_sample.
    eigen_adapt_pool: bool = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_ADAPT_POOL", False, bool)
    )
    # Eigen inner move (same ruling): rung k steps with sigma / sqrt(beta_k)
    # (capped at the table's own cap) instead of the cold sigma at every
    # rung; the persisted table stays the cold one. Off = every rung the cold
    # sigma, bit-identical. See
    # ResidualAddOneRemoveOneMove._eigen_visit_rung_scale.
    eigen_temper_sigmas: bool = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_TEMPER_SIGMAS", False, bool)
    )
    # Safety cap of the step-scale adaptation (ruling 2026-10-15, after the
    # 12mo's first tar: rung chains straddling discrete degenerate solutions
    # widened the best-measured axis x600-4900, to the prior box): an adapted
    # sigma never exceeds this multiple of the information-matrix sigma.
    # <= 0 or inf = no cap. Inert unless eigen_axis_adapt is on. See
    # addremovemove.factor_capped_sigmas.
    eigen_adapt_max_factor: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_ADAPT_MAX_FACTOR", 100.0, float)
    )
    # ---- the EMRI sampling audit (user ruling 2026-10-18; docs/emri-sampling-audit.md)
    # Per-axis acceptance accounting of the eigen inner move (cold vs hot
    # rungs, per table axis), logged once per table refresh. Telemetry only:
    # no proposal changes. See ResidualAddOneRemoveOneMove._note_inner_outcome.
    eigen_axis_acc_log: bool = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ACC_LOG", False, bool)
    )
    # Acceptance-driven per-axis step controller on top of the adaptation:
    # after each refresh window an axis whose COLD acceptance fell below
    # eigen_axis_acc_min has its step divided by eigen_axis_acc_factor
    # (never below the Fisher sigma), one above eigen_axis_acc_max multiplied
    # by it (never above the adapted width / the box cap). Fixed between
    # refreshes (each repeat sweep stays a frozen symmetric kernel). Off =
    # bit-identical. See ResidualAddOneRemoveOneMove._eigen_axis_tune_table.
    eigen_axis_acc_tune: bool = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ACC_TUNE", False, bool)
    )
    eigen_axis_acc_min: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ACC_MIN", 0.1, float)
    )
    eigen_axis_acc_max: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ACC_MAX", 0.5, float)
    )
    eigen_axis_acc_factor: float = dataclasses.field(
        default_factory=env_default("EMRI_EIGEN_AXIS_ACC_FACTOR", 2.0, float)
    )
    # Cross-walker difference-vector jump in the inner-move stack (stack
    # weight; 0 = not in the stack, bit-identical) and its share of gamma = 1
    # mode jumps. See lisatools.sampling.moves.walkerdiffjump.
    de_jump_mix: float = dataclasses.field(
        default_factory=env_default("EMRI_DE_JUMP_MIX", 0.0, float)
    )
    de_jump_mode_frac: float = dataclasses.field(
        default_factory=env_default("EMRI_DE_JUMP_MODE_FRAC", 0.5, float)
    )


class EMRISetup(Setup):
    """:class:`Setup` for extreme mass-ratio inspirals in the Erebor recipe.

    Construction copies the settings, opens the ``emri_setup.log`` logger
    (under ``log_dir``) and runs :meth:`init_setup`. Built once per branch by
    :meth:`StockGlobalFit.build <lisatools.globalfit.stock.base.StockGlobalFit.build>`.

    Args:
        emri_settings: Settings dataclass with EMRI parameter ranges,
            waveform configuration, and search helpers.
    """

    def __init__(self, emri_settings: EMRISettings):

        # had a better way to do this but it stopped allowing for pickle
        super().__init__(emri_settings)

        level = logging.DEBUG
        name = "EMRISetup"
        self.logger = init_logger(filename="emri_setup.log", level=level, name=name, log_dir=getattr(self, 'log_dir', None), shared_console=False)

        self.init_setup()

    def init_sampling_info(self):
        """Build the EMRI :class:`TransformContainer`, prior, periodicity, and tempering ladder.

        Fills (only when unset): ``fill_values`` (``[1.0, 0.0]``), ``transform``
        (:func:`~lisatools.globalfit.stock.erebor.transforms.make_emri_transform_container`),
        ``periodic`` (``phiS``, ``phiK``, ``Phi_phi0``, ``Phi_r0`` with period
        ``2 pi``), ``priors`` (:meth:`setup_priors` runs unconditionally, so
        an explicit ``priors`` value is overwritten), ``betas``
        (:func:`~lisatools.globalfit.stock.erebor.common.default_source_ladder`),
        ``other_tempering_kwargs`` (``permute=False``), ``initialize_kwargs``
        and ``inner_moves``.

        Raises:
            AssertionError: When ``other_tempering_kwargs["permute"]`` is true.
        """
        input_basis = [
            "logm1",
            "m2",
            "a",
            "p0",
            "e0",
            "dist",
            "qS",
            "phiS",
            "qK",
            "phiK",
            "Phi_phi0",
            "Phi_r0",
        ]

        if self.fill_values is None:
            # prograde-equatorial fallback (synthetic/no-catalogue runs)
            self.fill_values = np.array([1.0, 0.0])
        if self.transform is None:
            self.transform = make_emri_transform_container(self.fill_values)

        if self.periodic is None:
            self.periodic = {
                "emri": {
                    "phiS": 2 * np.pi,
                    "phiK": 2 * np.pi,
                    "Phi_phi0": 2 * np.pi,
                    "Phi_r0": 2 * np.pi,
                }
            }

        self.setup_priors(input_basis)

        if self.betas is None:
            # make_ladder, not 1/1.2^k (user ruling 2026-10-13: "the default
            # ladders for non-RJ sources should be from make_ladder"): the
            # geometric spacing for ~25 % swap acceptance of an ndim Gaussian,
            # sized by this branch's own EMRI_NTEMPS. An explicit ``betas``
            # still wins (this block never runs then).
            from .common import default_source_ladder

            self.betas = default_source_ladder(self.ndim, input_basis, self.ntemps,
                                               tmax=self.ladder_tmax)

        self.logger.info(f"Using betas: {self.betas} in EMRI branch")

        # TODO: maybe combine this into Setup
        if self.other_tempering_kwargs is None:
            self.other_tempering_kwargs = dict(permute=False)

        if "permute" not in self.other_tempering_kwargs:
            self.other_tempering_kwargs["permute"] = False

        assert not self.other_tempering_kwargs["permute"]

        if self.initialize_kwargs is None:
            self.initialize_kwargs = {}

        if self.inner_moves is None:
            from .common import resolve_inner_moves

            resolve_inner_moves(self)

    def setup_priors(self, input_basis):
        """Build the EMRI prior dictionary, overriding intrinsic ranges from settings.

        Starts from a default uniform prior on each input-basis parameter and
        overrides the intrinsic parameter ranges (``logm1``, ``m2``, ``a``,
        ``p0``, ``e0``) with the corresponding ``*_lims`` attributes when
        those have been set.

        The result, ``self.priors = {"emri": EMRIKerrDomainPrior(...)}``, also
        cuts the joint ``(a, p0, e0)`` FEW domain of validity; with leaves of
        mixed prograde / retrograde ``xI0`` the cut uses the prograde grid
        (logged) and per-leaf validity falls to the likelihood's FEW
        domain-error sentinel.

        Args:
            input_basis (list[str]): Ordered list of EMRI parameter names
                matching the sampler's input basis.
        """

        # The logm1 / m2 / p0 boxes must contain every mojito_light catalogue
        # source (a start point outside the prior makes lnpdiff NaN and
        # freezes the sampler): the catalogue spans M in [3.92e5, 1e7],
        # m2 up to 127.7, p0 down to 2.12.
        priors_emri = {
            input_basis[0]: uniform_dist(np.log(3e5), np.log(1.5e7)),  # log m1
            input_basis[1]: uniform_dist(1, 200),  # m2
            input_basis[2]: uniform_dist(0.01, 0.999),  # a
            input_basis[3]: uniform_dist(2.0, 100.0),  # p0
            input_basis[4]: uniform_dist(0.001, 0.8),  # e0
            input_basis[5]: uniform_dist(0.01, 100.0),  # dist in Gpc
            input_basis[6]: uniform_dist(-0.99999, 0.99999),  # qS
            input_basis[7]: uniform_dist(0.0, 2 * np.pi),  # phiS
            input_basis[8]: uniform_dist(-0.99999, 0.99999),  # qK
            input_basis[9]: uniform_dist(0.0, 2 * np.pi),  # phiK
            input_basis[10]: uniform_dist(0.0, 2 * np.pi),  # Phi_phi0
            input_basis[11]: uniform_dist(0.0, 2 * np.pi),  # Phi_r0
        }

        limits = ["logm1_lims", "m2_lims", "a_lims", "p0_lims", "e0_lims"]
        for i, lims in enumerate(limits):
            if getattr(self, lims) is not None:
                self.logger.info(
                    f"Setting prior for parameter {i} using limits {getattr(self, lims)}"
                )
                priors_emri[input_basis[i]] = uniform_dist(*getattr(self, lims))

        # Joint (a, p0, e0) FEW domain-of-validity cut on top of the boxes:
        # combinations the Kerr ecc-eq grid cannot generate get logpdf -inf
        # (and rvs rejection-resamples), matching the -1e300 likelihood
        # sentinel applied when a waveform call fails on a domain error.
        # Shared prior-level FEW domain gate. With PER-LEAF fills the prior
        # has no leaf identity: a uniform prograde/retrograde population
        # keeps its exact gate; mixed populations fall back to the prograde
        # grid here and rely on the likelihood's FEW domain-error sentinel
        # (-1e300) for per-leaf validity.
        # TODO: per-leaf domain gate in EMRIKerrDomainPrior.
        _fv = np.asarray(self.fill_values, dtype=float)
        _xi_vals = np.unique(_fv[:, 0]) if _fv.ndim == 2 else _fv[:1]
        if len(_xi_vals) == 1:
            _xi_fill = float(_xi_vals[0])
        else:
            self.logger.warning(
                "EMRI leaves mix prograde and retrograde xI0 "
                f"({_xi_vals.tolist()}); the shared prior domain gate uses "
                "the prograde grid (per-leaf validity via the likelihood's "
                "FEW domain-error sentinel)."
            )
            _xi_fill = 1.0
        self.priors = {
            "emri": EMRIKerrDomainPrior(
                priors_emri,
                a_index=input_basis.index("a"),
                p0_index=input_basis.index("p0"),
                e0_index=input_basis.index("e0"),
                xI_fill=_xi_fill,
            )
        }

    def init_setup(self):
        """Run sampling-info and state-backend initialization."""
        self.init_sampling_info()
        self.init_state_backend_info()

    def init_state_backend_info(self):
        """Default the EMRI state / backend to :class:`EMRIState` / :class:`EMRIHDFBackend`.

        Both are per-leaf-ladder module sub-state / sub-backend classes (the
        branch's tempered ensemble lives in its own HDF group).
        """
        if self.branch_state is None:
            self.branch_state = EMRIState

        if self.branch_backend is None:
            self.branch_backend = EMRIHDFBackend
