#!/usr/bin/env python
"""Staged combined run: noise+foreground search -> GB search -> GB PE, with VGBs.

The run this builds toward: mojito data, ``T_obs`` = 3 months, one composition
carrying **gb + vgb + psd + galfor**, driven by a THREE-stage recipe rather
than the single combined PE stage ``all_sources`` ships with.

    1. ``noise_search``  kind="search"  noise_joint_search (psd_pe+galfor_pe)
       The stage is a "search" in the sense of being convergence-gated, but
       its sub-moves are the PE ones: psd and galfor are fixed-dimensional,
       so there is nothing to search FOR and a maximizing proposal would be
       the wrong tool. ``JointMaxLogLSearch`` supplies the criterion --
       ``run_move_max_likelihood`` loops internally until the cold-chain max
       lnL plateaus ACROSS ALL its sub-moves jointly, and ``SearchRecipeStep``
       is done on its first check because the criterion lives INSIDE the
       move. So this stage converges the noise model before a single GB is
       subtracted.

    1b. ``noise_vgb_search`` kind="search" noise_vgb_joint_search
       The same, plus the 55 VGBs. They are KNOWN sources -- seeded from the
       catalogue, fixed-dimensional, no RJ -- so they too are fitted rather
       than searched for, and their power leaves the residual before the GB
       search starts.

    2. ``gb_search``     kind="rj"      psd_pe + galfor_pe + GB search moves
       ``RJRecipeStep`` watches the cold-chain leaf count on ``plateau_branch
       ="gb"`` and advances when it plateaus -- "GB PE when the leaves of
       search converge". Noise keeps sampling underneath (PE mode) so the GB
       search sees a live PSD rather than a frozen one.

    3. ``gb_pe``         kind="pe"      everything, VGBs included
       ``PERecipeStep`` never stops on its own.

Composition is ``all_sources`` MINUS mbh/emri/sobbh -- it already carries vgb,
and ``remove_branch`` drops the moves that sample a removed branch (stock
``bd02f94``), so nothing dangles.

WHY NOT gb_no_fg (default composition): it loads ``source_types = ("GB",)``
with a FIXED PSD, i.e. there is no instrument-noise realization in the data at
all. Sampling a PSD against it would be fitting noise that is not there. This
script defaults ``source_types`` to NOISE + GB + VGB so every sampled branch
has something to fit.

GB_ONLY=1 (2026-08-24, user ruling: "just do GBs, no PSD/foreground, or VGB
modules ... use the injection psd") flips that reasoning around ON PURPOSE:
the fit becomes ``erebor.gb_no_fg`` -- the stock variant DESIGNED for
GB-only running -- but with ``source_types`` kept at NOISE+GB+VGB, so the
DATA is the same full mojito injection (noise brick + GB galaxy + VGBs)
while only the gb branch is SAMPLED. The likelihood sensitivity is
gb_no_fg's fixed-PSD path: ``adjust_general`` fits the analytic 2-parameter
instrument model [Soms_d, Sa_a] to the mojito NOISE brick's tabulated
estimates (``resolve_noise_file_psd_params``; pin PSD_FROM_NOISE_FILE=1 to
make a missing brick a hard error instead of a silent fallback to the stock
levels) -> ``fixed_psd_kwargs`` -> run.py setup_acs's no-psd-branch path
builds every walker AC from it, and the ACA's ``linear_psd_arr`` (built from
each AC's invC) is the ONLY sensitivity the GB band engine, the sig-het
in-model scorer, the in-move F-stat grid fit, and the SNR rejection gates
read. Consequences, accepted by the ruling: the unmodeled confusion
foreground and the VGBs simply sit in the residual (unwhitened -- the fixed
PSD carries no galaxy term), and the reported lnL is source-only
(-1/2 <r|r>, gb_no_fg's ``likelihood_source_only=True``). Stages become
gb_search -> full_pe directly: no noise/vgb stages, no waiting on a PSD fit.
The GB machinery is bit-identical to the all_sources composition --
all_sources' GB branch IS the gb_no_fg stack (``AllSourcesGBSettings``
subclasses ``GBNoFgGBSettings``; ``prepare_gb_branch``/``setup_gb_moves``
shared) -- so every band/cap/sig-het/F-stat env knob applies unchanged.

Run (one GPU)::

    MOJITO_DATA_PATH=/path/to/L1 USE_GPU=1 GPU_BACKEND=cuda12x GPUS=0 \
    NITER=100 NWALKERS=16 \
    python scripts/fstat_proposal/run_combined_staged.py

Smoke mode (COMBINED_SMOKE=1) shrinks every axis and turns the GB/VGB debug
verifications on -- see the knob table at the bottom of this docstring.

Key env knobs
-------------
    COMBINED_SMOKE=1     small band / few iterations / debug on
    GB_ONLY=1            erebor.gb_no_fg composition: gb is the ONLY sampled
                         branch (fixed injection PSD, source-only lnL); the
                         data still carries SOURCE_TYPES in full. Stages:
                         gb_search -> full_pe. Incompatible with the
                         STAGE_NOISE_* flags below.
    SOURCE_TYPES         comma list (default "NOISE,GB,VGB")
    NITER, NWALKERS      sampler shape
    GB_NTEMPS, VGB_NTEMPS, PSD_NTEMPS     per-branch ladders
    GB_DEBUG=1, VGB_DEBUG=1               residual round-trip verification
    GB_DEBUG_PLOT_BAND   ONE band index -- unset means EVERY band, which at
                         ~1150 cells renders thousands of figures per proposal
    GB_WARM_START_COMPONENTS  warm-start components npz (workstream B):
                         when set, the gb_search stage runs the
                         ``rj_warm_search`` birth move IMMEDIATELY BEFORE
                         ``rj_fstat_search`` (user ruling 2026-08-24);
                         unset = stage lists unchanged, bit-identical run.
                         full_pe is NOT touched (the ruling names search;
                         the warm move is search-configured).
    MBHB_IDS / EMRI_IDS / SOBHB_IDS
                         comma id lists arming the mbh/emri/sobbh branches
                         (campaign gate S6, docs/6mo-campaign.md; user
                         ruling 2026-09-02 "we are adding MBHB EMRI
                         SOBHB"). Absent/empty = branch dropped (today's
                         4-branch behavior, bit-identical). Armed: a
                         ``source_search`` stage (joint max-lnL over the
                         armed source PE moves) runs FIRST -- sources
                         converge and subtract before the noise stages
                         fit the PSD ("MBH search first, full_year
                         pattern") -- and the armed PE moves join
                         gb_search + full_pe in the sobbh -> mbh -> emri
                         banking order. Ids land on
                         ``general.mojito_source_ids``; SOURCE_TYPES
                         defaults gain the armed classes (explicit env
                         still wins). Incompatible with GB_ONLY=1.
    GB_SEARCH_RJ_REPLACE gb_search F-stat REPLACEMENT move (default 1,
                         2026-08-24 exact-MH reinstatement): ``rj_replace``
                         runs IMMEDIATELY AFTER ``rj_fstat_search`` in the
                         gb_search cycle -- full 9-column candidates from
                         the fitted F-stat grids + epoch center table,
                         scored at their EXACT likelihood (never
                         phase-maximized), so a higher-SNR table draw can
                         relocate a mis-seated leaf in one accepted move.
                         =0 removes it (stage lists bit-identical to the
                         pre-replace runs). full_pe is NOT touched.
    STAGE_SKIP_NOISE=1   start at stage 2 (noise already converged)
    GB_SEARCH_SOURCE_EVERY   sobbh/mbh/emri gb_search cadence (default 5):
                         they propose on every Nth gb_search iteration,
                         staying subtracted in between; full_pe is NEVER
                         cadenced (2026-09-18 ruling; 09-15 was mbh/emri
                         only, every 10)
    STAGE_SKIP_SOURCE_SEARCH=1  no source_search stage: armed sources
                         start (and stay subtracted) at their seeded
                         coords; source proposals only in gb_search +
                         full_pe (exact-truth-start flow, 2026-09-14)
    SOURCE_SEARCH_POSITION   first (default: before the noise and GB
                         stages) | after_seed (between gb_search_seed and
                         gb_search_1; the 9mo / 1yr "fixed-dim source seed",
                         2026-10-08; refused when no seed stage composes)
    SOURCE_SEARCH_CHECKS / SOURCE_SEARCH_TOL / SOURCE_SEARCH_MAX_ROUNDS
                         source_search's OWN per-walker plateau rule: flat
                         rounds, nats, round ceiling over the stage. Unset =
                         NOISE_SEARCH_CHECKS / MAXLOGL_TOL / MAXLOGL_MAX_ITER
                         (the globals, as before). A round = one pass of the
                         armed source PE moves at their in-model repeats;
                         MAXLOGL_ITERS_PER_STEP rounds per stored row
    GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1  every v9 GB search stage samples
                         the noise, not just gb_search_3: the leading joint
                         psd+galfor+vgb rider plus a noise convergence after
                         EVERY in-model convergence proposal, so the psd and
                         foreground adjust as the GB model fills in. Default
                         OFF -- the 6mo v9 run keeps noise FIXED in
                         gb_search_1/2 (2026-09-26, the 3mo v9 run)
    PER-STAGE RECIPE KNOBS (user ruling 2026-10-07, the 9mo / 1yr recipe:
    seed x3 -> gb_search_1 -> gb_search_2 -> full_pe). Unset = today:
    GB_SEARCH_STAGES     numbered search stages composed, ascending comma
                         list (default 1,2,3); the kept ones keep their names
    GB_SEARCH_{N}_SOURCE_EVERY  sobbh/mbh/emri cadence in stage N: 0 = no
                         source moves there (vgb + warm start stay), 1 =
                         every iteration; unset = GB_SEARCH_SOURCE_EVERY
    GB_SEARCH_{N}_NOISE_MODE  fixed | interleaved | cycle_end. interleaved =
                         the leading rider + 4 noise slots (the table's stage
                         3); cycle_end = ONE psd+galfor max-lnL search run TO
                         CONVERGENCE at the END of every cycle (plateau rule
                         NOISE_SEARCH_CHECKS / MAXLOGL_TOL / MAXLOGL_PER_WALKER,
                         afresh each cycle, at most GB_SEARCH_{N}_NOISE_MAX_ROUNDS
                         rounds, default 5000)
    GB_SEARCH_{N}_RESET_VALVES=1  at a FRESH stage entry (not a resume): the
                         per-(walker, band) cold-lnL max re-learned AND the
                         per-band barren valve revived (the per-walker window
                         is released at every new step anyway)
    GB_SEARCH_{N}_NUM_REPEAT_PROPOSALS / _INMODEL_GROUP_ITERS /
    _INMODEL_CONVERGE_ITERS  the stage's in-model budget (profile key
                         ``inmodel_budget``, 2026-10-16): the pure in-model
                         moves' repeats per pass and group window (passes),
                         the RJ moves' newborn polish window (repeats); set
                         at EVERY entry of the stage, the other stages run
                         the moves as built (gb_search_1 untouched)
    --print-recipe / GF_PRINT_RECIPE=1  print the resolved recipe (stages,
                         stop rules, profiles, noise, source cadence, moves)
                         and exit: no data, no MPI
    STAGE_NOISE_PSD_ONLY=1  the STANDALONE noise stage samples psd ALONE;
                         galfor stays frozen at GALFOR_START_PARAMS (which it
                         REFUSES to run without) until gb_search_3 releases
                         it. gb_search_3's interleaved noise slots are NOT
                         touched (2026-09-26, the 3mo v9 run)
    STAGE_SKIP_NOISE_VGB=1  no noise_vgb_search stage; with
                         VGB_START_FACTOR=0 the VGBs start at exact truth and
                         have nothing to converge, so they sit subtracted
                         through the noise stage and sample from gb_search_1
    STAGE_NOISE_ONLY=1   run only the two noise search stages, then stop
    STAGE_NOISE_VGB_PE=1 searches, then PE-sample psd+galfor+vgb (no GB);
                         bounded by NUM_ITERATIONS
    REMOVE_BRANCHES      comma list of whole branches to drop: gb, galfor,
                         vgb, psd. Removed branches leave the fit, every
                         stage move list, AND the DEFAULT injection stream
                         list ("we will not inject them"). Removing psd
                         (2026-09-14, the TRUTH-INJECTION NULL TEST) also
                         drops both noise stages and the NOISE stream, and
                         switches the likelihood to the FIXED-sensitivity
                         path -- it REQUIRES gb + galfor removed,
                         ADD_INSTRUMENT_NOISE=0 and UNEQUAL_ARM=0, all
                         enforced here.
    PSD_FIXED_PARAMS     comma floats [Soms_d, Sa_a], PHYSICAL (linear)
    GALFOR_FIXED_PARAMS  comma floats (amp, fk, alpha, f_1, f_2), PHYSICAL
                         -> general.fixed_psd_kwargs, the only sensitivity
                         the no-psd-branch path reads. See the basis note at
                         the wiring below before setting them.
"""
from __future__ import annotations

import logging
import os
import sys

# GB stage scoping -- MUST be seeded before ANY lisatools.globalfit.stock
# import: ``erebor``'s module-level default instances snapshot every
# env-backed field at import time and ``erebor.all_sources(...)`` CLONES
# that snapshot, so a setdefault placed after the import is invisible.
# GB_MODE=search arms the SEARCH-stage GB moves (leaf caps from 1, birth
# phase-max, flip 1.0, zero-leaf start -- injection/SNR seeding skipped);
# GB_PE_MOVES_STRICT keeps that scoped: the pe-NAMED instances (rj_prior /
# rj_fstat_mcmc / rj_refit, the full_pe stage) stay strictly PE regardless.
os.environ.setdefault("GB_MODE", "search")
os.environ.setdefault("GB_PE_MOVES_STRICT", "1")
# The gb_search stage lists Move("rj_prior_removal") -- build_gb_moves only
# CONSTRUCTS that stock move when gb.search_prior_removal is on (and mode is
# search), so the knob must default on here or recipe materialization fails
# with "no stock move under this name". Search-stage-only either way: the
# move is search-gated and full_pe never references it.
os.environ.setdefault("GB_SEARCH_PRIOR_REMOVAL", "1")
# GB_RIDGE_GIBBS (default 1): zero-likelihood fiber resample of the exact
# Mc-ratio-distance degeneracy after the RJ cycle in both GB stages -- the
# (Mc, r, dist) marginals are frozen along the curved ridge without it
# (measured 2026-08-20). Registered by build_gb_moves; =0 removes it.
# NITER is the name this script's docs (and muscle memory) use; the stock
# field is general.num_iterations -> env NUM_ITERATIONS (rule 0). Map it
# HERE, before any stock.erebor import snapshots the env -- a bare NITER
# was silently ignored (the smoke "NITER=12" never applied; runs ended at
# whatever NUM_ITERATIONS resolved to, looking like silent deaths).
if os.environ.get("NITER") and not os.environ.get("NUM_ITERATIONS"):
    os.environ["NUM_ITERATIONS"] = os.environ["NITER"]

logger = logging.getLogger("combined_staged")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True")


#: Moves that must never be dropped, whatever ``GB_SEARCH_DROP_MOVES``
#: says. ``rj_fstat_search`` is the DESIGNATED UPDATER
#: (``leaf_cap_update=True``) and the only move that runs the level-3
#: judge and the stage-end check, so dropping it would leave the stage
#: silently unable to END -- the failure would look like "the gate never
#: latches", which is exactly the class of bug this run has already spent
#: days on.
_UNDROPPABLE_SEARCH_MOVES = frozenset({"rj_fstat_search"})

_DROP_LOGGED = set()


def _dropped_search_moves() -> frozenset:
    """``GB_SEARCH_DROP_MOVES`` -- comma-separated move names to omit.

    Cost knob for a long search stage. 6mo v9 job 662 spent 1392 s in
    ``in_model`` plus 1248 s in ``in_model_fstat`` out of a 5564 s
    iteration (47%); the two are the SAME CLASS -- one pure in-model
    polish per RJ move -- so with cold births at ~1e-4 acceptance a
    single group-rule polish per iteration is enough and the second is
    ~1250 s of duplication.

    Applies to the pure in-model SLOTS only, which is where both stage
    assemblies in this file gate their optional moves. A name this
    cannot honour is reported rather than ignored: a knob that silently
    reaches nothing is the recurring failure mode in this run.
    """
    raw = os.environ.get("GB_SEARCH_DROP_MOVES", "").strip()
    if not raw:
        return frozenset()
    want = {p.strip() for p in raw.split(",") if p.strip()}
    bad = want & _UNDROPPABLE_SEARCH_MOVES
    if bad:
        raise SystemExit(
            f"GB_SEARCH_DROP_MOVES lists {sorted(bad)}, which cannot be "
            f"dropped: rj_fstat_search is the designated updater "
            f"(leaf_cap_update=True) and the only move that runs the "
            f"level-3 valve judge and the stage-end check. Dropping it "
            f"leaves gb_search unable to end, and the symptom is "
            f"indistinguishable from a broken gate.")
    return frozenset(want)


def _inmodel_after_removal() -> bool:
    """``GB_SEARCH_INMODEL_AFTER_REMOVAL`` -- default ON.

    ``rj_prior_removal`` is the ONLY RJ move with no in-model partner,
    so until now the saved state -- and the residual the NEXT
    iteration's births are drawn against -- was the post-death state
    with no group-rule pass over it. The surviving twin of a killed
    double is precisely the cell that needs readjusting after the
    death.

    ⚠ NOT A DOUBLES CURE, and the comment is here so nobody reads it as
    one: deaths are still JUDGED on the unadjusted residual. This is
    correctness of the saved state and better births from it.
    """
    return os.environ.get(
        "GB_SEARCH_INMODEL_AFTER_REMOVAL", "1").strip() in (
            "1", "true", "True", "yes", "on")


def _drop_slot(slot: str) -> bool:
    """Is this stage slot dropped by the knob? Logs once per slot."""
    if slot not in _dropped_search_moves():
        return False
    if slot not in _DROP_LOGGED:
        _DROP_LOGGED.add(slot)
        logger.info(
            "[V9-STAGE] GB_SEARCH_DROP_MOVES: %r omitted from every search "
            "stage composition. The move is still BUILT (dropping it from "
            "the build would fail materialization for any stage that still "
            "lists it); it simply never runs.", slot)
    return True


def _pe_combine_kwargs() -> dict:
    """Combine kwargs shared by every PE stage.

    PE cycle style (user ruling 2026-08-26): the SEARCH-style single
    propose per iteration -- the backend stores ONE row per iteration,
    fixing the 1-row-per-move PE storage the random_choice mode produced
    -- but the cycle composition is DRAWN each iteration: ``len(moves)``
    weighted draws WITH replacement, all proposals equal weight by
    default. FULL_PE_WEIGHTED_CYCLE=0 + FULL_PE_RANDOM_CHOICE=1 restores
    the legacy one-move-per-step mode; search stages pass neither flag
    and are untouched.
    """
    return dict(
        share_temperature_control=False,
        weighted_cycle=_env_flag("FULL_PE_WEIGHTED_CYCLE", "1"),
        random_choice=_env_flag("FULL_PE_RANDOM_CHOICE", "0"),
    )


def pe_fstat_moves() -> list:
    """The PE stages' F-stat birth moves, in cycle order (user ruling 2026-10-12).

    "I would like to do two fstat moves that will split their share of the
    total moves": ``rj_fstat_pe`` (the GB-FREE residual, the whole galaxy, fit
    once) and ``rj_fstat_pe_residual`` (the LIVE residual, refit every
    ``GB_FSTAT_REFIT_EVERY_PE`` iterations). ``GB_PE_RJ_FSTAT_RESIDUAL_SPLIT``
    is the residual move's share of ``GB_PE_RJ_FSTAT_FRACTION``; a move with
    NO share (split exactly 0 or 1) is left out of the stage, so the PE stage
    degrades to the one F-stat move it draws (and, with the draw off, the
    weighted cycle cannot run a move the split gave nothing). The same
    resolver feeds the draw (``GFCombineMove._pe_rj_draw_one_config``).
    Used by EVERY PE composition in this file (full_pe and replica_pe, both
    assemblies), so "full_pe" means one thing.
    """
    from lisatools.globalfit.moves.globalfitmove import pe_rj_fstat_residual_split
    from lisatools.globalfit.recipe import PE_FSTAT_RESIDUAL_MOVE, Move

    split = pe_rj_fstat_residual_split()
    out = []
    if split < 1.0:
        out.append(Move("rj_fstat_pe", branch="gb"))
    if split > 0.0:
        out.append(Move(PE_FSTAT_RESIDUAL_MOVE, branch="gb"))
    return out


def _apply_smoke_defaults() -> None:
    """Shrink every cost axis and arm the debug verifications.

    ``setdefault`` throughout: an explicitly-set knob always wins, so the
    same script serves the smoke and the production run and the diff between
    them is exactly this function.
    """
    smoke = {
        # --- shape ---
        # 12, not 3: the chunked noise search (MAXLOGL_ITERS_PER_STEP) now
        # spends real engine iterations per stage -- this is the run-wide
        # total -- and the budget must reach the GB stages. NUM_ITERATIONS
        # is the REAL knob (NITER was a dead name; the module-top shim maps
        # it now).
        "NUM_ITERATIONS": "12",
        "NWALKERS": "8",
        "GB_NTEMPS": "6",
        "VGB_NTEMPS": "4",
        "PSD_NTEMPS": "4",
        # --- band: 5 WDM layers instead of 13 ---
        # _band_klohi snaps INWARD and requires >= 3 whole layers, so the
        # span must clear ~4*layer_df (1.3889e-4 Hz) after snapping. 7.0-7.3
        # mHz gives k_lo=51, k_hi=52 -> ONE layer, and raises.
        # 7.0-7.8 mHz gives k_lo=51, k_hi=56 -> 5 layers, 3 interior.
        "GB_MIN_FREQ": "7.0e-3",
        "GB_MAX_FREQ": "7.8e-3",
        "GB_NUM_REPEAT_PROPOSALS": "20",
        "GB_N_SUBBANDS": "256",
        # --- F-stat fit: every stage still RUNS, each does less work ---
        "FSTAT_N_PER_AXIS": "8",
        "FSTAT_COMB_NSKY_MAX": "32",
        "FSTAT_COMB_NSKY_MIN": "8",
        "FSTAT_PEAKS_PER_BAND": "5",
        "FSTAT_CKPT_SECS": "15",   # so checkpointing actually fires
        # --- verification ---
        "GB_DEBUG": "1",
        "VGB_DEBUG": "1",
        # ONE band: unset renders a WDM tile per source per block.
        "GB_DEBUG_PLOT_BAND": "1",
        "GB_DEBUG_PLOT_WALKER": "0",
    }
    for k, v in smoke.items():
        os.environ.setdefault(k, v)


from lisatools.globalfit.moves.globalfitmove import Move  # noqa: E402


class JointMaxLogLSearch(Move):
    """A set of stock moves converging under ONE max-logl criterion.

    ``PSDMove(max_logl_mode=True)`` converges each noise branch separately,
    to its own plateau. psd and galfor are two parameterizations of the same
    noise model -- galfor moves change the residual the psd move is fitting
    and vice versa -- so maximizing them independently can stall each where
    the pair could still climb together. This promotes the criterion to span
    both.

    Wraps the ``*_pe`` moves DELIBERATELY: ``max_logl_mode`` is consulted in
    exactly one place (``psdmove.py:918``), choosing between the plateau loop
    and a single ``run_move_for_loop``. So the pe move IS the search move
    minus its private loop -- the right inner behaviour here. Wrapping the
    ``*_search`` moves would nest two plateau loops.

    This is also what makes ``Stage(kind="search")`` correct for these
    stages: ``SearchRecipeStep`` reports done on its FIRST check, because the
    stopping criterion is supposed to live INSIDE the move. A stage that
    listed the moves side by side would run one pass and advance -- the
    criterion has to span the whole stage, which is what this object is.

    Module level, not a closure: the pre-build fit must pickle/deepcopy
    (LISA Analysis Tools-wide rule), so a local class would break it.
    """

    def __init__(self, name, inner_names, iters_per_step=None,
                 num_checks=None, restart_each_propose=False, tol=None,
                 max_rounds=None, **kwargs):
        super().__init__(name, **kwargs)
        self.inner_names = list(inner_names)
        # Per-propose inner-iteration cap and plateau length handed to
        # MaxLogLCombineMove. None = the global knobs (MAXLOGL_ITERS_PER_STEP
        # / NOISE_SEARCH_CHECKS, the standalone noise stages); the gb_search
        # rider overrides both (2026-09-11).
        self.iters_per_step = iters_per_step
        self.num_checks = num_checks
        # True = every propose runs the plateau rule AFRESH (the cycle-end
        # noise convergence, GB_SEARCH_{N}_NOISE_MODE=cycle_end): without it a
        # plateaued instance takes one round per call, the rider behaviour.
        self.restart_each_propose = bool(restart_each_propose)
        # This instance's OWN plateau tolerance (nats) and round ceiling (rounds
        # over the instance's life, MaxLogLCombineMove.max_iter). None = the
        # globals MAXLOGL_TOL / MAXLOGL_MAX_ITER, as before. Set on the built
        # move AFTER construction, because its __init__ lets those globals
        # override its constructor arguments (the source_search stage's
        # SOURCE_SEARCH_TOL / SOURCE_SEARCH_MAX_ROUNDS, 2026-10-08).
        self.tol = None if tol is None else float(tol)
        self.max_rounds = None if max_rounds is None else int(max_rounds)

    def stock_dependencies(self):
        """The stock moves this wraps -- without this they are never BUILT.

        Variant setup functions construct only the stock moves the recipe
        asks for (``recipe.stock_names()``). This move resolves by its own
        ``setup``, so its name tells the builder nothing about the moves it
        composes; under STAGE_NOISE_ONLY, where these are the only moves in
        the recipe, that left ``ctx.stock_moves`` completely empty.
        """
        return list(self.inner_names)

    def setup(self, ctx):
        from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

        missing = [n for n in self.inner_names if n not in ctx.stock_moves]
        if missing:
            raise ValueError(
                f"{self.name}: no stock move(s) {missing} (available: "
                f"{sorted(ctx.stock_moves)})."
            )
        # only passed when set, so every other instance is built exactly as before
        _restart = ({"restart_each_propose": True}
                    if getattr(self, "restart_each_propose", False) else {})
        mv = MaxLogLCombineMove(
            [ctx.stock_moves[n] for n in self.inner_names],
            num_checks=(
                int(os.environ.get("NOISE_SEARCH_CHECKS", "5"))
                if self.num_checks is None else int(self.num_checks)
            ),
            share_temperature_control=False,
            iters_per_step=self.iters_per_step,
            **_restart,
        )
        # per-instance tolerance / ceiling, only when set (getattr: a descriptor
        # pickled before these existed builds exactly as before)
        apply_release_tol(mv, getattr(self, "tol", None))
        if getattr(self, "max_rounds", None) is not None:
            mv.max_iter = int(self.max_rounds)
        mv.gf_move_name = self.name
        return mv


def apply_release_tol(inner, tol):
    """Give a built max-logL search ITS OWN plateau tolerance.

    ``MaxLogLCombineMove.__init__`` reads ``MAXLOGL_TOL`` and lets it override
    the ``tol`` it was constructed with, so a per-move tolerance has to be set
    on the instance afterwards. ``tol`` None = keep whatever the move has.
    """
    if tol is None:
        return inner
    tol = float(tol)
    if not tol > 0:
        raise ValueError(f"release tol={tol} must be > 0 nats.")
    inner.tol = tol
    return inner


class GatedNoiseSearch(JointMaxLogLSearch):
    """The ONE noise proposal of a ratcheted search stage (GALFOR_RATCHET=1).

    Wraps the same psd+galfor joint max-logL search as :class:`JointMaxLogLSearch`
    in a :class:`~lisatools.globalfit.noise_ratchet.NoiseRatchetGate`, whose
    per-iteration mode the stage's ``SearchStageProfileStep`` drives from the
    stage-local iteration: NUDGE (force the galfor coordinates down by
    ``delta`` on every rung and walker, then one GB in-model pass), HOLD
    (nothing runs) or RELEASE (the ordinary search, then the same in-model
    pass). User design 2026-09-30: "take out the noise_search modules from gb
    search 3, only put one in there, at the beginning of an iteration, then
    gate that one proposal so that it does the schedule we want"; the in-model
    follow-up "should run anytime the noise is changing whether forced or
    free, just not when it is held fixed."

    Module level, not a closure: the pre-build fit must pickle/deepcopy.
    """

    def __init__(self, name, inner_names, delta, in_model_name="in_model",
                 release_to_convergence=True, release_tol=None, **kwargs):
        super().__init__(name, inner_names, **kwargs)
        self.delta = [float(x) for x in delta]
        self.in_model_name = in_model_name
        # The release's own plateau tolerance in nats (user ruling 2026-10-02:
        # "change tol = 20 to 5. I just want to see it converged"). Applied to
        # the built MaxLogLCombineMove AFTER construction because its __init__
        # lets the process-global MAXLOGL_TOL override the constructor value;
        # None leaves the global in charge.
        self.release_tol = None if release_tol is None else float(release_tol)
        # User ruling 2026-10-02: on a release the psd and galfor branches run
        # in SEARCH mode until the log-likelihood converges -- the wrapped
        # max-logL search starts afresh on every release instead of taking
        # the one round per call a plateaued rider takes (job 675's second
        # release: one round, 2 s). See NoiseRatchetGate.release_to_convergence.
        self.release_to_convergence = bool(release_to_convergence)

    def stock_dependencies(self):
        """The wrapped noise moves PLUS the in-model follow-up, so all are built."""
        deps = list(self.inner_names)
        if self.in_model_name and self.in_model_name not in deps:
            deps.append(self.in_model_name)
        return deps

    def setup(self, ctx):
        from lisatools.globalfit.noise_ratchet import NoiseRatchetGate

        inner = super().setup(ctx)
        apply_release_tol(inner, self.release_tol)
        galfor = ctx.stock_moves.get("galfor_pe")
        if galfor is None:
            raise ValueError(
                f"{self.name}: the galfor ratchet needs the stock 'galfor_pe' "
                f"move to perform the forced step (available: "
                f"{sorted(ctx.stock_moves)}).")
        in_model = None
        if self.in_model_name:
            in_model = ctx.stock_moves.get(self.in_model_name)
            if in_model is None:
                raise ValueError(
                    f"{self.name}: the in-model follow-up move "
                    f"{self.in_model_name!r} is not built (GB_SEARCH_IN_MODEL=0?). "
                    "The ratchet requires it: the GB sources must settle to a "
                    "changed noise before any RJ move scores against it.")
        gate = NoiseRatchetGate(inner, galfor, self.delta, in_model_move=in_model,
                                release_to_convergence=self.release_to_convergence)
        gate.gf_move_name = self.name
        return gate


# mbh/emri/sobbh arming (campaign S6): branch -> (env, mojito class), in the
# sobbh -> mbh -> emri BANKING order (user ruling 2026-08-27: the cheap
# branch banks its progress before the minutes-per-leaf MBH/EMRI proposals
# put the iteration at risk; move order is not a sampling statement).
_SOURCE_BRANCH_ENVS = (
    ("sobbh", "SOBHB_IDS", "SOBHB"),
    ("mbh", "MBHB_IDS", "MBHB"),
    ("emri", "EMRI_IDS", "EMRI"),
)


def _source_ids_from_env() -> dict:
    """Armed source branches: {branch: [ids]} from MBHB_IDS/EMRI_IDS/SOBHB_IDS.

    Absent or empty env = branch NOT armed (it is removed, today's
    behavior). Mirrors source_runtime.default_source_ids' parsing.
    """
    armed = {}
    for branch, env, _cls in _SOURCE_BRANCH_ENVS:
        raw = os.environ.get(env, "")
        ids = [int(x) for x in raw.split(",") if x.strip() != ""]
        if ids:
            armed[branch] = ids
    return armed


#: THE v9 SEARCH STAGE TABLE -- ``(stage name, profile, samples noise)``.
#:
#: User spec 2026-09-24. One ``gb_search`` stage becomes three that differ in
#: exactly four things, and this tuple is the single place all four are
#: stated, so the GB-only composition and the full one cannot drift:
#:
#:   * ``phase_maximize`` -- the RJ BIRTH phase-maximization heuristic.
#:     On in stage 1 (hunting: take the credit), off afterwards.
#:   * ``opt_snr`` -- the GB SNR prior boundary. 8 in stage 1 ("SNR 5 = noise,
#:     keep 8" -- a boundary of 5 measurably ballooned the hot ladder with
#:     noise births), 5 once the loud population is assembled and the faint
#:     tail must become reachable.
#:   * ``peak_min_snr`` -- the F-stat peak-selection floor. Moves with the
#:     opt-SNR boundary; 6.25 is the landed 2026-09-23 knee value.
#:   * samples noise -- whether the psd/galfor moves are in the stage at all.
#:
#: ⚠ ``peak_min_snr`` is consumed when the F-stat grid is FITTED and stamped
#: into the stage-B cache, whose loader refuses a mismatch. The stage entry
#: therefore sets an in-code override AND forces a fresh-epoch refit; see
#: ``recipe.SearchStageProfileStep``. Nothing on disk is deleted.
#: ``full_pe``'s F-stat peak floor, declared for the same reason the search
#: stages declare theirs: the override the search stages install is
#: process-global, so an undeclared full_pe would inherit gb_search_3's value
#: silently. 6.25 is what the design wants here (an assembled model's faint
#: tail must stay reachable) -- this makes choosing it explicit. ``None``
#: would clear the override and hand the stage back to FSTAT_PEAK_MIN_SNR.
#: Since 2026-10-12 it is the floor of the LIVE-residual move
#: (``rj_fstat_pe_residual``); the GB-free galaxy move ``rj_fstat_pe`` carries
#: its OWN floor, GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE (6.5), scoped around its
#: fit and applied to a loaded epoch as a cut on the cached peaks.
_PE_PEAK_MIN_SNR = 6.25


def _pe_rj_flip_fraction() -> float:
    """The GB RJ flip fraction full_pe declares on entry: GB_PE_RJ_FLIP_FRACTION
    (the launcher's 0.1) or the recipe's 0.2 code default.

    Declared per stage because, with the GB branch in search mode, the
    recipe builds the PE-named RJ moves with the SEARCH fraction
    (``_rj_flip_default`` follows ``gb_info.mode``), so the PE knob would
    otherwise reach nothing in a v9 run (user ruling 2026-10-02: "make sure
    for GBs, during full PE all RJ moves sample 0.1 of the available slots").
    """
    from lisatools.globalfit.recipe import _pe_rj_flip_default

    return float(_pe_rj_flip_default())


def _replica_pe_knobs() -> dict:
    """The replica-PE stage's convergence knobs (user design 2026-10-03).

    REPLICA_PE_WINDOW (10): rows the trends are fitted over ("over say 10
    replica pe iterations"); REPLICA_PE_LEAF_TOL (10): the largest total
    change over the window any of min/mean/max cold leaves may show;
    REPLICA_PE_LNL_TOL (100): the same for min/mean/max cold lnL, nats;
    REPLICA_PE_MIN_ITERS (= window): rows in-stage before the check runs.
    """
    window = int(os.environ.get("REPLICA_PE_WINDOW", "10").strip() or 10)
    leaf_tol = float(os.environ.get("REPLICA_PE_LEAF_TOL", "10").strip() or 10)
    lnl_tol = float(os.environ.get("REPLICA_PE_LNL_TOL", "100").strip() or 100)
    min_iters = int(os.environ.get("REPLICA_PE_MIN_ITERS", "").strip() or window)
    if window < 3 or leaf_tol <= 0 or lnl_tol <= 0 or min_iters < window:
        raise ValueError(
            f"REPLICA_PE_WINDOW={window} must be >= 3, REPLICA_PE_LEAF_TOL={leaf_tol} and "
            f"REPLICA_PE_LNL_TOL={lnl_tol} > 0, REPLICA_PE_MIN_ITERS={min_iters} >= the window.")
    return dict(window=window, leaf_tol=leaf_tol, lnl_tol=lnl_tol, min_iters=min_iters)


def _pe_inmodel_repeats():
    """``{branch: n}`` full_pe declares for its moves' in-model repeats, or None.

    ``PE_INMODEL_REPEATS`` (int; unset = leave every move as built) applies to
    the branches in ``PE_INMODEL_REPEATS_BRANCHES`` (comma list; default
    ``gb,vgb,sobbh,mbh,psd,galfor`` -- user ruling 2026-10-02: "in-model
    repeats of 25 for all sources except emris for right now for PE. This
    includes VGBs and GBs ... that way we have a general 'thinning' factor
    of 25", so the noise branches are in the set and EMRI is not). Per
    STAGE, not process-global: the built moves are shared with the search
    stages, whose repeat counts stay what their knobs say.
    """
    raw = os.environ.get("PE_INMODEL_REPEATS", "").strip()
    if not raw:
        return None
    n = int(raw)
    if n < 1:
        raise ValueError(f"PE_INMODEL_REPEATS={n} must be >= 1.")
    branches = [b.strip() for b in os.environ.get(
        "PE_INMODEL_REPEATS_BRANCHES", "gb,vgb,sobbh,mbh,psd,galfor").split(",") if b.strip()]
    if not branches:
        raise ValueError("PE_INMODEL_REPEATS_BRANCHES is empty.")
    return {b: n for b in branches}

V9_SEARCH_STAGE_PROFILES = (
    # prior_births (user ruling 2026-10-02: "set it specifically for gb search
    # 3 ... prior removal only for gb search 1"): what rj_prior_removal
    # proposes in the stage -- False = deaths only (the pruning move), True =
    # births AND deaths from the prior container. Stated for EVERY stage
    # because the move object is shared across them (a stage without the key
    # would inherit whatever the previous stage left); GB_SEARCH_{N}_PRIOR_BIRTHS
    # overrides a row. Applied move-scoped by SearchStageProfileStep, never
    # broadcast (rj_removal_only on rj_fstat_search = no births).
    ("gb_search_1",
     dict(phase_maximize=True, opt_snr=8.0, peak_min_snr=8.0,
          prior_births=False), False),
    # ⚠ STAGE 2 DOES NOT PHASE-MAXIMIZE, and the reason is the row itself.
    # It is the stage that DROPS the floors -- opt SNR 8 -> 5, F-stat peak
    # 8 -> 6.25 -- so it is already reaching for weaker, more marginal
    # sources. Maximizing the phase on top of that would add a second
    # optimistic bias to exactly the population least able to afford one:
    # a maximized delta is an upper bound on what the source can pay, and
    # near the floor that is the difference between a real detection and
    # a noise peak (user ruling 2026-09-25, correcting the same day's
    # "1 and 2" to stage 1 only). Only stage 1 -- the high-floor, high-
    # confidence pass -- maximizes.
    ("gb_search_2",
     dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25,
          prior_births=True), False),
    # reset_band_max (user ruling 2026-09-30): stage 3 is the stage whose noise
    # MOVES, so the shutoff valve's per-(walker, band) cold-lnL max carried in
    # from the fixed-noise stages is re-learned from -inf at entry instead of
    # inherited ("instead of starting from the stored value, let's reset
    # that"). Job 672's stage 3 ended on the shutoff rule after three
    # iterations because every band sat below a max earned under the old
    # noise. The galfor ratchet resets it again at every nudge.
    ("gb_search_3",
     dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25,
          reset_band_max=True, prior_births=True), True),
)

#: Per-stage override of the profile's ``opt_snr`` -- the OPTIMAL-SNR
#: prior boundary (``opt_snr_rej_samp_limit``), NOT the F-stat peak
#: floor, which stays on ``peak_min_snr`` and is not touched here.
#:
#: Added 2026-09-29 for Mike's "let's test a gb search 3 rerun with snr
#: limit 3 and see what happens". Until now the profile row hard-coded
#: 5.0 and ``SearchStageProfileStep._apply_profile`` writes it onto every
#: GB move on stage entry, so ``GB_OPT_SNR_LIMIT_SEARCH`` was overridden
#: and there was no per-stage way in.
#:
#: Stage 2 gets the symmetric knob because it costs one table row and
#: because a stage-3-only knob invites the next experiment to hard-edit
#: the row again -- which is what this replaces.
_SEARCH_STAGE_OPT_SNR_KNOBS = {
    "gb_search_2": "GB_SEARCH_2_OPT_SNR",
    "gb_search_3": "GB_SEARCH_3_OPT_SNR",
}

#: Per-stage override of the profile's ``prior_births`` (1/0): what the
#: shared ``rj_prior_removal`` move proposes in that stage. Resolved by
#: :func:`search_stage_profiles` beside the opt-SNR knobs.
_SEARCH_STAGE_PRIOR_BIRTHS_KNOBS = {
    "gb_search_1": "GB_SEARCH_1_PRIOR_BIRTHS",
    "gb_search_2": "GB_SEARCH_2_PRIOR_BIRTHS",
    "gb_search_3": "GB_SEARCH_3_PRIOR_BIRTHS",
}

# ---- PER-STAGE RECIPE KNOBS (user ruling 2026-10-07, the 9mo / 1yr recipe) --
# "I want to start with 3 iterations of gb search seed ... Then I want to have
# gb search 1 as is currently but with no other source moves ... Then GB search
# 2 with valves reset ... Gb search 2 should include the other sources every
# iteration. And it should allow the PSD and galfor to vary ... After GB search
# 2 will be full pe. No replica pe and no ratcheting." Refined the same day:
# "during gb search 2 the galfor and psd sampling should be required to
# converge before moving on to the other proposals. There should only be one
# noise/galfor proposal though ... at the end of the cycle."
#
# Every knob below unset = today's composition, byte-identical: the running 6mo
# store's recipe is fixed (GFHDFBackend.add_recipe refuses a dropped or
# reordered stage), so the new recipe reaches the 1yr / 9mo launchers only
# through their derive tables.

#: ``GB_SEARCH_{N}_NOISE_MODE`` values. ``interleaved`` = the table's sampled
#: composition (leading psd+galfor+vgb rider + the four noise slots, or the
#: galfor ratchet gate under GALFOR_RATCHET=1); ``cycle_end`` = ONE psd+galfor
#: max-lnL search to convergence at the END of the cycle.
SEARCH_NOISE_MODES = ("fixed", "interleaved", "cycle_end")

#: The cycle-end convergence's round ceiling per propose (``GB_SEARCH_{N}_
#: NOISE_MAX_ROUNDS``). A safety cap, never the stop: the plateau rule ends it.
#: The ratchet release's ceiling, same ruling ("we do not want it to hit that").
_CYCLE_END_MAX_ROUNDS = 5000


def _stage_knob(name: str, suffix: str):
    """``GB_SEARCH_{N}_{suffix}`` for a numbered search stage, else None."""
    head, _, num = name.rpartition("_")
    if head != "gb_search" or not num.isdigit():
        return None
    return f"GB_SEARCH_{num}_{suffix}"


def _env_bool(knob: str):
    """1/0-style env bool; None when unset or empty; raises on anything else."""
    raw = os.environ.get(knob)
    if raw is None or raw.strip() == "":
        return None
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{knob}={raw!r} must be 1/0.")


def _search_stage_numbers():
    """``GB_SEARCH_STAGES`` -> the numbered stages to compose (None = all).

    An ascending comma list of table numbers; refused rather than reordered or
    deduplicated, because a stage list that silently differs from the one typed
    is the failure this whole file guards against.
    """
    raw = os.environ.get("GB_SEARCH_STAGES", "").strip()
    if not raw:
        return None
    table = [int(n.rpartition("_")[2]) for n, _, _ in V9_SEARCH_STAGE_PROFILES]
    try:
        want = [int(p) for p in raw.split(",")]
    except ValueError:
        raise ValueError(f"GB_SEARCH_STAGES={raw!r} must be a comma list of "
                         f"stage numbers from {table}.") from None
    if (not want or any(w not in table for w in want)
            or want != sorted(set(want))):
        raise ValueError(
            f"GB_SEARCH_STAGES={raw!r} must be an ASCENDING list of distinct "
            f"stage numbers from {table} (e.g. 1,2).")
    return tuple(want)


def search_stage_noise_mode(name: str, table_sampled: bool) -> str:
    """The noise composition of one numbered search stage.

    ``GB_SEARCH_{N}_NOISE_MODE`` wins; unset = the table (``interleaved`` for a
    sampled row or under GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1, else ``fixed``).
    """
    knob = _stage_knob(name, "NOISE_MODE")
    raw = (os.environ.get(knob) or "").strip().lower() if knob else ""
    if not raw:
        return ("interleaved"
                if (table_sampled or _env_flag("GB_SEARCH_SAMPLE_NOISE_ALL_STAGES"))
                else "fixed")
    if raw not in SEARCH_NOISE_MODES:
        raise ValueError(f"{knob}={raw!r} must be one of {SEARCH_NOISE_MODES}.")
    return raw


def search_stage_source_every(name: str):
    """``GB_SEARCH_{N}_SOURCE_EVERY``: the stage's sobbh/mbh/emri cadence.

    None = unset (the global GB_SEARCH_SOURCE_EVERY); 0 = no source moves in
    the stage; N >= 1 = every Nth iteration of the stage (1 = every one).
    """
    knob = _stage_knob(name, "SOURCE_EVERY")
    raw = (os.environ.get(knob) or "").strip() if knob else ""
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{knob}={raw!r} must be an integer >= 0.") from None
    if n < 0:
        raise ValueError(f"{knob}={n} must be >= 0 (0 = no source moves).")
    return n


#: ``GB_SEARCH_{N}_<SUFFIX>`` -> the stage profile's ``inmodel_budget`` key
#: (``recipe.INMODEL_BUDGET_KEYS``: the first two reach the pure in-model
#: moves, the third the RJ moves' newborn polish).
_INMODEL_BUDGET_KNOBS = (
    ("NUM_REPEAT_PROPOSALS", "num_repeat_proposals"),
    ("INMODEL_GROUP_ITERS", "inmodel_group_iters"),
    ("INMODEL_CONVERGE_ITERS", "inmodel_converge_iters"),
)


def search_stage_inmodel_budget(name: str):
    """``GB_SEARCH_{N}_NUM_REPEAT_PROPOSALS`` / ``_INMODEL_GROUP_ITERS`` /
    ``_INMODEL_CONVERGE_ITERS``: the stage's in-model budget as the profile
    key ``inmodel_budget`` (``{attribute: int}``, only the knobs that are
    set; None when none is). Applied at the stage entry by
    ``recipe.apply_inmodel_budget``: repeats per pass and the group window
    (passes) on the pure in-model moves, the newborn polish window (repeats)
    on the RJ moves. A stage that sets none runs the moves as built, so
    gb_search_1 is untouched by a gb_search_2 budget (Mike 2026-10-16).
    """
    out = {}
    for suffix, attr in _INMODEL_BUDGET_KNOBS:
        knob = _stage_knob(name, suffix)
        raw = (os.environ.get(knob) or "").strip() if knob else ""
        if not raw:
            continue
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(f"{knob}={raw!r} must be an integer >= 1.") from None
        if n < 1:
            raise ValueError(f"{knob}={n} must be >= 1.")
        out[attr] = n
    return out or None


def search_stage_noise_max_rounds(name: str) -> int:
    """``GB_SEARCH_{N}_NOISE_MAX_ROUNDS``: the cycle-end convergence's ceiling."""
    knob = _stage_knob(name, "NOISE_MAX_ROUNDS")
    raw = (os.environ.get(knob) or "").strip() if knob else ""
    if not raw:
        return _CYCLE_END_MAX_ROUNDS
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{knob}={raw!r} must be an integer >= 1.") from None
    if n < 1:
        raise ValueError(f"{knob}={n} must be >= 1.")
    return n


# ---- THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08) ----------------------
# "since we are seeing the MBHs move quite a bit towards a better likelihood, I
# want to add a new stage to the 9mo run ... 'fixed-dim source seed'. In this
# stage, the galfor and psd will be fixed just like in gb search seed and gb
# search 1. However, it will only run MBHBs, SOBHBs, and EMRIs, until the
# maximum likelihood of each walker converges ... Over say 10 iterations, make
# sure the logL has not changed by more than 10." That stage is ``source_search``
# (the joint max-lnL search over the armed source PE moves); these give it ITS
# OWN plateau rule and a position. Read only when the stage composes (the 6mo
# exports STAGE_SKIP_SOURCE_SEARCH=1); each unset = today's composition.

#: ``SOURCE_SEARCH_POSITION`` values. ``first`` = before the noise and GB stages
#: (the historical place, the default); ``after_seed`` = between gb_search_seed
#: and gb_search_1 (the 9mo / 1yr, main-session placement decision 2026-10-08:
#: the source maxima are found against a residual with the warm-started galaxy
#: already subtracted, rather than one still holding the whole galaxy that the
#: pinned noise model does not account for).
SOURCE_SEARCH_POSITIONS = ("first", "after_seed")


def source_search_position() -> str:
    """``SOURCE_SEARCH_POSITION`` (first | after_seed; unset = first)."""
    raw = (os.environ.get("SOURCE_SEARCH_POSITION") or "").strip().lower()
    if not raw:
        return "first"
    if raw not in SOURCE_SEARCH_POSITIONS:
        raise ValueError(f"SOURCE_SEARCH_POSITION={raw!r} must be one of "
                         f"{SOURCE_SEARCH_POSITIONS}.")
    return raw


def source_search_rule():
    """``(checks, tol, max_rounds)`` of the source_search stage's plateau rule.

    ``SOURCE_SEARCH_CHECKS`` (flat rounds per walker), ``SOURCE_SEARCH_TOL``
    (nats), ``SOURCE_SEARCH_MAX_ROUNDS`` (rounds over the whole stage, a safety
    ceiling). Each unset = None = the global knob the stage has always used
    (NOISE_SEARCH_CHECKS / MAXLOGL_TOL / MAXLOGL_MAX_ITER), so the globals stay
    the gb_search_2 cycle-end noise slot's.
    """
    def _read(knob, cast, ok, what):
        raw = (os.environ.get(knob) or "").strip()
        if not raw:
            return None
        try:
            val = cast(raw)
        except ValueError:
            raise ValueError(f"{knob}={raw!r} must be {what}.") from None
        if not ok(val):
            raise ValueError(f"{knob}={raw!r} must be {what}.")
        return val

    checks = _read("SOURCE_SEARCH_CHECKS", int, lambda v: v >= 1,
                   "an integer >= 1 (consecutive flat rounds per walker)")
    # v == v rejects nan; a finite positive number of nats
    tol = _read("SOURCE_SEARCH_TOL", float,
                lambda v: v == v and 0.0 < v < float("inf"), "a number of nats > 0")
    cap = _read("SOURCE_SEARCH_MAX_ROUNDS", int, lambda v: v >= 1,
                "an integer >= 1 (rounds)")
    return checks, tol, cap


def _seed_profile(profiles):
    """gb_search_seed's profile = gb_search_1's, verbatim (ruling 2026-09-26).

    Looked up by NAME: with GB_SEARCH_STAGES dropping stage 1 the first row is
    another stage's, and seeding under gb_search_2's floors would be a silent
    change of the seed stage. Refused instead.
    """
    for n, p, _ in profiles:
        if n == "gb_search_1":
            return dict(p)
    raise ValueError(
        "gb_search_seed takes gb_search_1's profile, but GB_SEARCH_STAGES="
        f"{os.environ.get('GB_SEARCH_STAGES')!r} drops stage 1. Keep 1 in the "
        "list or export GB_SEARCH_SEED_ITERS=0.")


def search_stage_profiles():
    """:data:`V9_SEARCH_STAGE_PROFILES` with the env overrides applied.

    ⚠ CALL THIS, never the constant. Both v9 stage assemblies (the
    GB_ONLY one and the full composition) build from the table, and a
    knob wired into one of them is the "knob resolves, consuming path
    never runs" shape this file has already produced several defects
    in. One resolver, two call sites, and a test that reads the source
    of both.

    Unset knobs leave the table byte-identical, so the default run is
    unchanged. A value <= 0 raises rather than quietly disabling the
    boundary: ``opt_snr_rej_samp_limit = 0`` is a legal "off" elsewhere
    in the tree, so a fat-fingered 0 here would read as a deliberate
    choice and silently admit every birth.

    Also resolved here, for the same two-call-site reason (2026-10-07):
    ``GB_SEARCH_STAGES`` (which rows compose; the third element stays the
    TABLE's sampled flag, which keys the stage-3 warm cadence) and
    ``GB_SEARCH_{N}_RESET_VALVES`` (the ``reset_valves`` profile key).
    """
    keep = _search_stage_numbers()
    if keep is not None:
        _dropped = [n for n, _, _ in V9_SEARCH_STAGE_PROFILES
                    if int(n.rpartition("_")[2]) not in keep]
        _inert = sorted(k for k in os.environ
                        for n in _dropped if k.startswith(_stage_knob(n, "")))
        print(f"[V9-STAGE] GB_SEARCH_STAGES={','.join(map(str, keep))}: "
              f"{_dropped} NOT composed" + (f"; their knobs {_inert} are inert"
                                            if _inert else ""), flush=True)
    out = []
    for name, prof, sampled in V9_SEARCH_STAGE_PROFILES:
        if keep is not None and int(name.rpartition("_")[2]) not in keep:
            continue
        knob = _SEARCH_STAGE_OPT_SNR_KNOBS.get(name)
        raw = os.environ.get(knob) if knob else None
        if raw is not None and raw.strip() != "":
            try:
                val = float(raw)
            except ValueError:
                raise ValueError(
                    f"{knob}={raw!r} is not a number.") from None
            if val <= 0.0:
                raise ValueError(
                    f"{knob}={val} must be > 0. Zero would read as a "
                    f"deliberate 'no optimal-SNR boundary' and admit "
                    f"every birth; unset the knob to keep the default "
                    f"{prof['opt_snr']}.")
            was = prof["opt_snr"]
            prof = dict(prof, opt_snr=val)
            print(f"[V9-STAGE {name}] profile {prof} -- opt_snr={val} "
                  f"from {knob} (default {was}; the F-stat peak floor "
                  f"peak_min_snr={prof['peak_min_snr']} is NOT affected)",
                  flush=True)
        pknob = _SEARCH_STAGE_PRIOR_BIRTHS_KNOBS.get(name)
        praw = os.environ.get(pknob) if pknob else None
        if praw is not None and praw.strip() != "":
            _v = praw.strip().lower()
            if _v in ("1", "true", "yes", "on"):
                pb = True
            elif _v in ("0", "false", "no", "off"):
                pb = False
            else:
                raise ValueError(f"{pknob}={praw!r} must be 1/0 (prior births on/off).")
            if pb != prof.get("prior_births"):
                print(f"[V9-STAGE {name}] profile prior_births={pb} from {pknob} "
                      f"(table default {prof.get('prior_births')})", flush=True)
            prof = dict(prof, prior_births=pb)
        # reset_valves (2026-10-07): written only when ON, so every other
        # profile dict stays byte-identical; applied by SearchStageProfileStep
        # at a FRESH stage entry (never on a resume inside the stage).
        vknob = _stage_knob(name, "RESET_VALVES")
        if _env_bool(vknob):
            print(f"[V9-STAGE {name}] profile reset_valves=True from {vknob}: "
                  f"at the stage's fresh entry the per-(walker, band) cold-lnL max "
                  f"is re-learned and the per-band barren valve revived", flush=True)
            prof = dict(prof, reset_valves=True)
        # inmodel_budget (2026-10-16): written only when a knob is set, so
        # the other profile dicts stay byte-identical; applied at EVERY entry
        # of the stage (a resume included) by SearchStageProfileStep.
        budget = search_stage_inmodel_budget(name)
        if budget:
            _num = name.rpartition("_")[2]
            print(f"[V9-STAGE {name}] profile inmodel_budget={budget} from "
                  f"GB_SEARCH_{_num}_NUM_REPEAT_PROPOSALS / _INMODEL_GROUP_ITERS / "
                  f"_INMODEL_CONVERGE_ITERS (the in-model moves' repeats per pass "
                  f"and group window in passes, the RJ moves' newborn polish window "
                  f"in repeats; the other stages run the moves as built)", flush=True)
            prof = dict(prof, inmodel_budget=budget)
        out.append((name, prof, sampled))
    return tuple(out)


class FixedIterationStop:
    """Stop a recipe stage after exactly ``n`` sampler iterations.

    ``RJRecipeStep`` takes a ``convergence_fn`` that fully overrides its
    leaf-plateau test, which is what lets ``gb_search_seed`` run a FIXED
    length instead of waiting for convergence (user ruling 2026-09-26:
    "I do not want to wait too long or wait for convergence").

    A module-level CLASS, not a closure: the pre-build fit has to survive
    ``pickle``/``deepcopy`` (the sprint-wide rule in CLAUDE.md), and a
    lambda or a factory-returned closure would not. State is one int plus
    one optional int.

    The start iteration is latched on the first call rather than read from
    the step, because ``convergence_fn`` is invoked as
    ``fn(i, sample, sampler)`` with no access to ``self._stage_start_iter``.
    """

    def __init__(self, n: int):
        if int(n) < 1:
            raise ValueError(f"FixedIterationStop needs n >= 1, got {n}.")
        self.n = int(n)
        self._start = None

    def __call__(self, i, sample, sampler) -> bool:
        cur = int(sampler.backend.iteration)
        if self._start is None:
            self._start = cur
        return (cur - self._start) >= self.n


def _seed_iters() -> int:
    """Length of ``gb_search_seed`` in FULL SAMPLER ITERATIONS (default 5).

    ``GB_SEARCH_SEED_ITERS=0`` drops the stage entirely, which is the
    pre-2026-09-26 recipe exactly.
    """
    n = int(os.environ.get("GB_SEARCH_SEED_ITERS", "5"))
    if n < 0:
        raise ValueError(f"GB_SEARCH_SEED_ITERS={n} must be >= 0.")
    return n


def _v9_search_enabled() -> bool:
    """Is the v9 three-stage GB search on? ``STAGE_V9_SEARCH=0`` restores the
    single legacy ``gb_search`` stage, bit-identically."""
    return os.environ.get("STAGE_V9_SEARCH", "1").strip() not in (
        "0", "false", "False", "off")


def build_fit():
    from lisatools.globalfit.recipe import Move, Recipe, Stage
    from lisatools.globalfit.stock import erebor

    nwalkers = int(os.environ.get("NWALKERS", "16"))
    gb_only = _env_flag("GB_ONLY")
    armed_sources = _source_ids_from_env()
    if gb_only and armed_sources:
        raise ValueError(
            f"GB_ONLY=1 is the gb-only composition; source id envs "
            f"{sorted(armed_sources)} cannot be armed with it."
        )
    # GB_ONLY: the gb_no_fg stock variant IS the GB-only design (no psd /
    # galfor / vgb branch anywhere in it; fixed injection PSD via its
    # adjust_general -> fixed_psd_kwargs -> setup_acs no-psd-branch path).
    # all_sources minus psd would NOT be equivalent: it never fills
    # fixed_psd_params, so the engine would silently fall back to the
    # hardcoded stock levels (engine.py fixed_psd_kwargs default).
    fit = erebor.gb_no_fg(nwalkers=nwalkers) if gb_only \
        else erebor.all_sources(nwalkers=nwalkers)

    def warm():
        # rj_warm_search (workstream B; user ruling 2026-08-24: "BEFORE the
        # fstat proposal"): warm-start births from the previous run's
        # clustered posterior components (GB_WARM_START_COMPONENTS npz;
        # build_gb_moves constructs the stock move only when the knob is
        # set, so the descriptor must be knob-conditional too). Fresh Move
        # descriptor per call (never share one instance across stages).
        return ([Move("rj_warm_search", branch="gb")]
                if os.environ.get("GB_WARM_START_COMPONENTS", "").strip()
                else [])

    def warm_pe():
        # rj_warm_pe (user ruling 2026-09-07: the PE twin, mirroring the
        # fstat search <-> pe differences): warm-start births in the PE
        # cycle, IMMEDIATELY BEFORE rj_fstat_pe -- the same relative
        # position the search twin takes vs rj_fstat_search. Same knob
        # arms both twins; fresh Move descriptor per call.
        return ([Move("rj_warm_pe", branch="gb")]
                if os.environ.get("GB_WARM_START_COMPONENTS", "").strip()
                else [])

    def replace():
        # rj_replace (2026-08-24 exact-MH reinstatement, user directive
        # "after the rj move"): the F-stat REPLACEMENT move IMMEDIATELY
        # AFTER rj_fstat_search in the gb_search cycle. Knob-conditional
        # like warm()/ridge() -- build_gb_moves only constructs the stock
        # move when gb.search_rj_replace is on (same env, same default 1),
        # and a listed-but-unbuilt move fails recipe materialization.
        # Listing it here (rather than relying on the setup_gb_moves
        # auto-insertion, which would add it anyway) keeps the composed
        # stage printout truthful. Fresh Move descriptor per call.
        return ([Move("rj_replace", branch="gb")]
                if os.environ.get("GB_SEARCH_RJ_REPLACE", "1").strip()
                in ("1", "true", "True", "yes", "on")
                else [])

    # TOBS_TARGET honored (2026-08-13): all_sources pins a FIXED WDM grid
    # (legacy nf/nt override, mojito-adjusted to 1440x2160 = 90 d) which by
    # the settings contract BEATS general.tobs_target -- so an explicit
    # TOBS_TARGET was silently ignored (the 23-mo shakedown ran 90 d). When
    # the env asks for a Tobs, clear the fixed grid so the build derives
    # (Nf, Nt) from tobs_target + the wavelet-duration bounds -- the same
    # machinery that yields 1440x2160 at the 90-d default. Unset env ->
    # behavior unchanged.
    if os.environ.get("TOBS_TARGET", "").strip():
        fit.general.nf = None
        fit.general.nt = None
        print(
            f"[combined] TOBS_TARGET={fit.general.tobs_target:.6g} s: "
            "cleared any fixed-grid (nf, nt) override (all_sources pins "
            "one; gb_no_fg's is already None -- no-op there).",
            flush=True,
        )

    # REMOVE_BRANCHES (user ask 2026-09-11: "run for everything except the
    # GBs and galfor. We will not inject them."): comma list of whole
    # branches to drop -- from the fit, from every stage's move list, and
    # from the DEFAULT injection streams below. Only the branches the stage
    # lists know how to drop are accepted. With "gb" removed there is no
    # F-stat machinery and no RJ: the gb stages collapse to ONE full_pe over
    # what remains.
    #
    # "psd" became removable 2026-09-14 evening (the TRUTH-INJECTION NULL
    # TEST ruling: "remove the psd fitting, use best fit values for psd and
    # galfor from the 3mo run"). Removing it drops the joint noise criteria
    # (JointMaxLogLSearch branch="psd") and both noise stages, and hands the
    # likelihood to run.py setup_acs's FIXED-sensitivity path
    # (``sensitivity_backend(..., **general.fixed_psd_kwargs)``) -- see the
    # PSD_FIXED_PARAMS / GALFOR_FIXED_PARAMS block below, which is the only
    # way the 3mo best fit reaches that path. Guarded hard: gb and galfor
    # machinery both assume a SAMPLED psd (the GB band engine reads the
    # per-walker linear_psd_arr the psd branch drives; galfor's foreground is
    # a component OF the psd branch's sensitivity), so psd can only go once
    # those are already gone.
    _removable = ("gb", "galfor", "vgb", "psd")
    remove_branches = tuple(
        b.strip().lower()
        for b in os.environ.get("REMOVE_BRANCHES", "").split(",")
        if b.strip()
    )
    for _b in remove_branches:
        if _b not in _removable:
            raise ValueError(
                f"REMOVE_BRANCHES entry {_b!r} is not removable here "
                f"(supported: {', '.join(_removable)})."
            )
    if gb_only and remove_branches:
        raise ValueError(
            "GB_ONLY=1 is already the gb-only composition; "
            "REMOVE_BRANCHES makes no sense with it."
        )
    if "psd" in remove_branches:
        # Both of these are silent-wrongness traps rather than crashes, so
        # refuse at COMPOSITION time (seconds) instead of discovering it
        # from a null that never reaches zero after hours of allocation.
        for _need in ("galfor", "gb"):
            if _need not in remove_branches:
                raise ValueError(
                    f"REMOVE_BRANCHES removes 'psd' but keeps {_need!r}: "
                    f"the {_need} machinery samples against the psd "
                    "branch's sensitivity (galfor's foreground is a "
                    "component of it; the GB band engine reads the "
                    "per-walker linear_psd_arr it drives). Remove "
                    f"{_need!r} too, or keep psd."
                )
        # "NO INJECTED NOISE" (ruling 2026-09-14): with no psd branch there
        # is nothing to fit a noise realization with, so an injected one
        # sits unmodelled in the residual and the truth null cannot reach
        # zero. all_sources defaults add_instrument_noise=True (auto), so
        # this is the easy way to get a quietly-wrong null.
        if fit.general.add_instrument_noise:
            raise ValueError(
                "REMOVE_BRANCHES removes 'psd' but "
                f"add_instrument_noise={fit.general.add_instrument_noise!r}: "
                "an injected noise realization would sit unmodelled in the "
                "residual (no psd branch fits it), so a truth-injection "
                "null could not reach zero. Export ADD_INSTRUMENT_NOISE=0."
            )
        # The noise-stage flags have no stage to act on any more (same
        # reasoning as the GB_ONLY block below).
        for _flag in ("STAGE_NOISE_ONLY", "STAGE_NOISE_VGB_PE",
                      "STAGE_SKIP_NOISE"):
            if _env_flag(_flag):
                raise ValueError(
                    f"REMOVE_BRANCHES removes 'psd', so there are no noise "
                    f"stages; {_flag}=1 makes no sense here."
                )
        # UNEQUAL_ARM swaps the PSD BRANCH's instrument component
        # (all_sources._wire_unequal_arm raises "unequal_arm=1 requires the
        # psd branch"), so the fixed-sensitivity path runs the plain
        # analytic equal-arm model. Caught here rather than at build.
        if fit.general.unequal_arm:
            raise ValueError(
                "REMOVE_BRANCHES removes 'psd' but UNEQUAL_ARM=1: the "
                "unequal-arm model is installed ON the psd branch's "
                "instrument component, which no longer exists. Export "
                "UNEQUAL_ARM=0 (at the null point the residual cancels, so "
                "the PSD weighting only scales small deviations)."
            )

    # Every sampled branch needs a stream: NOISE for psd/galfor, GB, VGB --
    # plus the armed source classes' streams (their data must contain the
    # signals the branches fit). Explicit SOURCE_TYPES always wins.
    # GB_ONLY keeps the same default DELIBERATELY: the data is the SAME full
    # injection (noise + GB galaxy + VGBs) as the production runs -- only the
    # SAMPLED branch set shrinks to gb. Unmodeled content stays in the
    # residual; that is the accepted trade for not waiting on a noise fit.
    # REMOVE_BRANCHES is the opposite contract ("we will not inject them"):
    # a removed gb/vgb also leaves the DEFAULT stream list. A removed PSD
    # takes the NOISE stream with it -- nothing models a noise realization
    # any more, so injecting one would leave it in the residual and the
    # truth-injection null could never reach zero ("NO INJECTED NOISE").
    _default_src = ",".join(
        (["NOISE"] if "psd" not in remove_branches else [])
        + (["GB"] if "gb" not in remove_branches else [])
        + (["VGB"] if "vgb" not in remove_branches else [])
    ) + "".join(
        f",{cls}" for br, _env, cls in _SOURCE_BRANCH_ENVS
        if br in armed_sources
    )
    src = os.environ.get("SOURCE_TYPES", _default_src)
    fit.general.source_types = tuple(
        s.strip().upper() for s in src.split(",") if s.strip()
    )

    # gb_no_fg carries ONLY the gb branch; remove_branch raises on absent
    # names, so guard. Unarmed source branches drop (the pre-S6 4-branch
    # behavior); armed ones STAY and get their mojito catalogue id lists
    # (mojito_source_ids gates inclusion; GB/VGB entries preserved).
    for branch in ("mbh", "emri", "sobbh"):
        if branch in fit.branches and branch not in armed_sources:
            fit.remove_branch(branch)
    if armed_sources:
        _ids = dict(fit.general.mojito_source_ids)
        for branch, _env, cls in _SOURCE_BRANCH_ENVS:
            if branch in armed_sources:
                _ids[cls] = list(armed_sources[branch])
        fit.general.mojito_source_ids = _ids
        print(
            f"[combined] source branches armed: "
            + ", ".join(f"{br}={armed_sources[br]}"
                        for br, _e, _c in _SOURCE_BRANCH_ENVS
                        if br in armed_sources),
            flush=True,
        )

    for _b in remove_branches:
        if _b in fit.branches:
            fit.remove_branch(_b)
    if remove_branches:
        print(f"[combined] branches REMOVED (not sampled, not in the "
              f"default injection): {list(remove_branches)}", flush=True)

    # FIXED SENSITIVITY (psd removed): PSD_FIXED_PARAMS / GALFOR_FIXED_PARAMS
    # -> general.fixed_psd_kwargs, the ONE thing run.py setup_acs's
    # no-psd-branch path reads (``sensitivity_backend(f"walker_{w}",
    # **general_info.fixed_psd_kwargs)``, run.py:1515-1517).
    #
    # BASIS -- READ THIS BEFORE SETTING THEM. The fixed path applies NO
    # transform: whatever is in the dict goes straight to the backend as
    # ``psd_params`` / ``galfor_params``, so both must already be in the
    # PHYSICAL basis the backend documents -- psd = [Soms_d, Sa_a] as LINEAR
    # (square-root) values, galfor = the 5-vector (amp, fk, alpha, f_1, f_2)
    # linear. (The SAMPLED path is the one that has to call
    # ``both_transforms`` first, run.py:1472-1486; every stock psd/galfor
    # transform is None at the default PSD_LOG_SAMPLING=0 /
    # GALFOR_LOG_SAMPLING=0, which is why the 3mo store's raw chain
    # coordinates ARE physical and can be used verbatim.)
    #
    # The submit script fills these from the 3mo store's best-logL cold
    # walker via ``lisatools.globalfit.warmstart.opt_snr.best_logl_noise``,
    # which returns exactly those raw chain rows (no transform) -- so the
    # two ends agree by construction. If a future run is launched with
    # PSD_LOG_SAMPLING=1 / GALFOR_LOG_SAMPLING=1, that extraction returns
    # LOG values and this wiring would silently under-weight the noise.
    _psd_fixed = os.environ.get("PSD_FIXED_PARAMS", "").strip()
    _gal_fixed = os.environ.get("GALFOR_FIXED_PARAMS", "").strip()
    if _psd_fixed or _gal_fixed:
        import numpy as np

        def _floats(raw, name):
            try:
                return np.array([float(x) for x in raw.split(",")
                                 if x.strip()])
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a comma list of floats, got {raw!r}."
                ) from exc

        _kw = dict(fit.general.fixed_psd_kwargs or {})
        if _psd_fixed:
            _kw["psd_params"] = _floats(_psd_fixed, "PSD_FIXED_PARAMS")
        if _gal_fixed:
            _kw["galfor_params"] = _floats(_gal_fixed, "GALFOR_FIXED_PARAMS")
        _kw.setdefault("galfor_params", None)
        fit.general.fixed_psd_kwargs = _kw
        print(f"[combined] FIXED sensitivity (physical basis): "
              f"psd_params={_kw.get('psd_params')} "
              f"galfor_params={_kw.get('galfor_params')}", flush=True)
        # Which of the two actually reaches the likelihood depends on which
        # BRANCHES are sampled, and the two answers differ -- say both out
        # loud rather than leaving a pin to look live when it is not.
        if "psd" in fit.branches and _psd_fixed:
            print("[combined] WARNING: PSD_FIXED_PARAMS set while the psd "
                  "branch is SAMPLED -- setup_acs takes the sampled "
                  "coordinates and IGNORES psd_params.", flush=True)
        if _gal_fixed:
            if "galfor" in fit.branches:
                print("[combined] WARNING: GALFOR_FIXED_PARAMS set while the "
                      "galfor branch is SAMPLED -- setup_acs takes the "
                      "sampled coordinates and IGNORES galfor_params. Add "
                      "galfor to REMOVE_BRANCHES to pin it.", flush=True)
            else:
                print("[combined] galfor branch REMOVED and galfor_params "
                      "PINNED -- the foreground is held at the value above "
                      "while psd stays sampled (run.py setup_acs). This is "
                      "the loop-breaker configuration: galfor can no longer "
                      "absorb unresolved GB and raise the search's own noise "
                      "floor in the band those sources live in.", flush=True)

    if gb_only:
        # Branch-set sanity: gb only, or the composition is not what the
        # header promises (and setup_acs would sample a PSD we said is fixed).
        _extra = [b for b in fit.branches if b != "gb"]
        if _extra:
            raise ValueError(
                f"GB_ONLY=1 but the fit carries extra branches {_extra}."
            )
        if _env_flag("STAGE_NOISE_ONLY") or _env_flag("STAGE_NOISE_VGB_PE"):
            raise ValueError(
                "GB_ONLY=1 has no noise/vgb stages; STAGE_NOISE_ONLY / "
                "STAGE_NOISE_VGB_PE make no sense here."
            )
        # Same stage names / kinds / GB move stacks as the full composition
        # below (monitor + digests + resume statuses stay compatible), minus
        # every psd/galfor/vgb move. Fresh store starts directly at
        # gb_search; nothing in the package gates on the noise stage names.
        def ridge():
            # Fresh Move descriptor per stage (never share one instance).
            return ([Move("gb_ridge_gibbs", branch="gb")]
                    if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])

        def gb_only_in_model(slot):
            if _drop_slot(slot):
                return []
            if slot == "in_model_removal" and not _inmodel_after_removal():
                return []
            # Same in_model_replace retirement as the full composition --
            # there are TWO stage assemblies in this file and gating only
            # one of them is the "knob resolves, consuming path never runs"
            # shape that has produced several defects in this run.
            if slot == "in_model_replace" and os.environ.get(
                    "GB_SEARCH_IN_MODEL_REPLACE", "0").strip() not in (
                        "1", "true", "True", "yes", "on"):
                return []
            return ([Move(slot, branch="gb")]
                    if _env_flag("GB_SEARCH_IN_MODEL") else [])

        if _v9_search_enabled():
            # Same three-stage v9 cycle as the full composition below, minus
            # every psd/galfor/vgb move. Keeping the two in step matters: the
            # GB-only variant is what the probe scripts and the search-test
            # runbook drive, so a divergence here would mean the probes stop
            # testing the production cycle.
            def _gb_only_pe_moves():
                return warm_pe() + pe_fstat_moves() + [
                    Move("rj_prior_pe", branch="gb"),
                ] + ridge()

            _pe_stage = Stage(
                name="full_pe", kind="pe",
                moves=_gb_only_pe_moves(),
                step_kwargs=dict(peak_min_snr=_PE_PEAK_MIN_SNR,
                                 stage_name="full_pe"),
                combine_kwargs=_pe_combine_kwargs(),
            )
            # replica_pe here TOO (2026-10-03; the two-assemblies rule above):
            # no source moves in the GB-only variant, so the replica is
            # full_pe's moves plus the convergence stop, nothing else.
            _gb_only_replica = []
            if _env_flag("STAGE_REPLICA_PE", "1"):
                _rk = _replica_pe_knobs()
                _gb_only_replica = [Stage(
                    name="replica_pe", kind="replica_pe",
                    moves=_gb_only_pe_moves(),
                    step_kwargs=dict(peak_min_snr=_PE_PEAK_MIN_SNR,
                                     stage_name="replica_pe", **_rk),
                    combine_kwargs={**_pe_combine_kwargs(), "weighted_cycle": False},
                )]
            _warm3 = int(os.environ.get("GB_SEARCH_3_WARM_EVERY", "5"))
            if _warm3 < 1:
                raise ValueError(
                    f"GB_SEARCH_3_WARM_EVERY={_warm3} must be >= 1.")
            # Resolved ONCE per assembly so the override logs once and
            # the seed row and the loop cannot disagree.
            _profiles = search_stage_profiles()
            # No psd/galfor branch here: a per-stage noise mode that asks for
            # noise sampling would reach nothing, so refuse it (the table's
            # stage-3 default, like GB_SEARCH_SAMPLE_NOISE_ALL_STAGES, is
            # simply moot under GB_ONLY, as before).
            for _name, _p, _s in _profiles:
                _k = _stage_knob(_name, "NOISE_MODE")
                if (os.environ.get(_k) or "").strip() and \
                        search_stage_noise_mode(_name, _s) != "fixed":
                    raise ValueError(
                        f"{_k}={os.environ[_k]!r} under GB_ONLY=1: the gb-only "
                        "composition has no psd/galfor branch to sample.")
            _gb_only_stages = []
            # gb_search_seed leads here TOO. There are two v9 stage
            # assemblies in this file -- this GB_ONLY one and the full
            # composition below -- and adding a stage to only one of them
            # is the "knob resolves, consuming path never runs" shape that
            # has produced several defects in this run. Same profile
            # (gb_search_1's), same fixed length, same disarmed valve.
            if _seed_iters() > 0:
                _sp = _seed_profile(_profiles)
                _seed_warm = ([Move("rj_warm_search", branch="gb")]
                              if warm() else [])
                _gb_only_stages.append(Stage(
                    name="gb_search_seed", kind="gb_search",
                    moves=_seed_warm + gb_only_in_model("in_model"),
                    step_kwargs=dict(
                        plateau_branch="gb",
                        stage_name="gb_search_seed", profile=dict(_sp),
                        convergence_fn=FixedIterationStop(_seed_iters()),
                        search_shutoff_per_walker=False,
                    ),
                    combine_kwargs=dict(share_temperature_control=False),
                ))
            for _name, _prof, _sampled in _profiles:
                _every = _warm3 if _sampled else 1
                _warm = ([Move("rj_warm_search", branch="gb", every=_every)]
                         if warm() else [])
                _gb_only_stages.append(Stage(
                    name=_name, kind="gb_search",
                    moves=(_warm + gb_only_in_model("in_model")
                           + [Move("rj_fstat_search", branch="gb")]
                           + gb_only_in_model("in_model_fstat")
                           + replace()
                           + gb_only_in_model("in_model_replace")
                           + [Move("rj_prior_removal", branch="gb")]
                           + gb_only_in_model("in_model_removal")
                           + ridge()),
                    step_kwargs=dict(
                        plateau_branch="gb",
                        convergence_iter=int(
                            os.environ.get("GB_PLATEAU_ITERS", "5")),
                        stage_name=_name, profile=dict(_prof),
                    ),
                    combine_kwargs=dict(share_temperature_control=False),
                ))
            fit.recipe = Recipe(_gb_only_stages + _gb_only_replica + [_pe_stage])
            return fit

        fit.recipe = Recipe([
            Stage(
                name="gb_search", kind="rj",
                # rj_warm_search (when armed) runs IMMEDIATELY BEFORE
                # rj_fstat_search -- warm-start births first, then the
                # F-stat grid births (user ruling 2026-08-24); rj_replace
                # (default on) IMMEDIATELY AFTER them -- the exact-MH
                # F-stat replacement pass before the removal judge.
                moves=warm() + [
                    Move("rj_fstat_search", branch="gb"),
                ] + replace() + [
                    Move("rj_prior_removal", branch="gb"),
                ] + ridge(),
                step_kwargs=dict(
                    plateau_branch="gb",
                    convergence_iter=int(
                        os.environ.get("GB_PLATEAU_ITERS", "5")
                    ),
                ),
                combine_kwargs=dict(share_temperature_control=False),
            ),
            Stage(
                name="full_pe", kind="pe",
                # rj_warm_pe (when armed) runs IMMEDIATELY BEFORE
                # rj_fstat_pe -- the PE mirror of the search-stage order.
                moves=warm_pe() + pe_fstat_moves() + [
                    Move("rj_prior_pe", branch="gb"),
                ] + ridge(),
                combine_kwargs=_pe_combine_kwargs(),
            ),
        ])
        return fit

    # GB move names (2026-08-12 rename, user ruling):
    #   rj_fstat_search  = F-stat grid births, search config (cap updater)
    #   rj_prior_removal = removal-only prior pruning (search cycle)
    #   rj_fstat_pe      = F-stat grid births, strict-PE config
    #   rj_prior_pe      = pure prior births, strict-PE config
    # NOTE ON NAMING (2026-09-08). The noise/VGB STAGES are called "*_search"
    # but their sub-moves are the "*_pe" ones, and that is deliberate, not a
    # leftover: psd (2 params), galfor (5) and vgb (55 KNOWN sources) are all
    # FIXED-DIMENSIONAL with nothing to search FOR. In this codebase a
    # "_search" proposal means maximize-and-skip-detailed-balance, which is
    # the right tool for finding unknown GBs and the wrong one for a
    # 2-parameter noise level. "search" in the STAGE name refers to the
    # stage's role in the ladder -- a convergence-gated burn-in -- and that
    # is what JointMaxLogLSearch supplies: a joint max-logL criterion
    # spanning all its sub-moves, so the stage advances when they have
    # JOINTLY plateaued. Hence `[GF_TIMING] stage=..._search move=psd_pe`,
    # which reads oddly but is correct.
    #
    # (A `noise_search = [Move("psd_search"), Move("galfor_search")]` list
    # used to sit here and was referenced by no stage -- removed 2026-09-08.
    # `setup_recipe` still BUILDS psd_search/galfor_search stock moves for
    # every present noise branch; this driver simply never requests them.)
    # Branch-aware lists (REMOVE_BRANCHES): a removed galfor/vgb drops out
    # of every move list and joint criterion below with no other change.
    _has_galfor = "galfor" in fit.branches
    _has_vgb = "vgb" in fit.branches
    # psd too (2026-09-14): with it removed there is no noise move at all and
    # no noise STAGE -- the sensitivity is fixed, so there is nothing to
    # converge. Same branch-aware pattern as galfor/vgb.
    _has_psd = "psd" in fit.branches
    noise_pe = (
        ([Move("psd_pe", branch="psd")] if _has_psd else [])
        + ([Move("galfor_pe", branch="galfor")] if _has_galfor else []))
    # VGBs are KNOWN sources: fixed-dimensional, no RJ, nothing to search
    # for. They sample from the first stage onward so their power is being
    # fitted while the noise converges, rather than sitting in the residual
    # and biasing the PSD.
    vgb = [Move("vgb_pe", branch="vgb")] if _has_vgb else []
    # VGB ridge-Gibbs, the GB twin (user ruling 2026-09-16, "VGBs get the
    # ridge-gibbs fiber move too"). ``build_vgb_moves`` registers
    # ``vgb_ridge_gibbs`` ONLY when the vgb basis carries dist/Mc/
    # fdot_astro_ratio -- i.e. under the 6-column VGB_CHIRP_MASS_BASIS=1
    # chirp basis -- so the stage must request it only there, or Move.setup
    # raises on the missing stock name. Same GB_RIDGE_GIBBS kill switch as
    # the GB one, and the same stage placement gb_ridge_gibbs has: BOTH
    # gb_search and the PE stages. It rides gb_search as its own Move (not
    # inside the noise_vgb joint-search CRITERION) because it is exactly
    # likelihood-invariant -- zero likelihood calls, prior x fiber-measure
    # MH only -- so it cannot perturb a max-logL plateau test, while
    # WITHOUT it the VGB chirp masses would sit frozen along the fiber for
    # the whole search stage, which is the disease the move exists to cure.
    _vgb_ridge_on = (
        _has_vgb
        and _env_flag("VGB_CHIRP_MASS_BASIS")
        and _env_flag("GB_RIDGE_GIBBS", "1")
    )

    def vgb_ridge():
        """Fresh Move descriptor per stage (never share one instance)."""
        return [Move("vgb_ridge_gibbs", branch="vgb")] if _vgb_ridge_on else []

    _noise_names = (["psd_pe"] if _has_psd else []) + (
        ["galfor_pe"] if _has_galfor else [])

    # ---- STAGE_NOISE_PSD_ONLY (user spec 2026-09-26, the 3mo v9 run) -------
    # The STANDALONE noise stage samples psd ALONE; the foreground stays FROZEN
    # at GALFOR_START_PARAMS -- a deliberately conservative reference, for a run
    # whose first stage fits the psd against a residual that still contains the
    # whole GB galaxy. Letting galfor move there is the 6mo disease: with GB
    # empty the foreground absorbs the unresolved galaxy (measured at 8.5x the
    # true confusion) and the inflated floor then hides the sources the search
    # exists to find.
    #
    # ⚠ SCOPE IS THE STANDALONE STAGE ONLY. gb_search_3's four interleaved
    # noise_joint_search_1..4 slots and the noise_vgb_joint_search rider keep
    # BOTH moves: stage 3 is the stage that RELEASES the foreground, and taking
    # galfor out of it would leave the run with no foreground fit at all. That
    # is why this filters a separate name list rather than _noise_names itself.
    _noise_psd_only = _env_flag("STAGE_NOISE_PSD_ONLY")
    if _noise_psd_only and _has_galfor and not os.environ.get(
            "GALFOR_START_PARAMS", "").strip():
        # A frozen branch sits at its START coordinates, and with no pin those
        # are a PRIOR DRAW. The run would then hunt GBs against a random
        # foreground for the whole of noise_search, gb_search_1 and
        # gb_search_2 -- the same failure the half-pin guard below prevents,
        # one layer down, and just as silent.
        raise ValueError(
            "STAGE_NOISE_PSD_ONLY=1 freezes the galfor branch through the "
            "noise stage and both fixed-noise GB search stages, but "
            "GALFOR_START_PARAMS is unset -- the foreground would be frozen "
            "at a PRIOR DRAW for three whole stages. Set GALFOR_START_PARAMS "
            "(PHYSICAL/linear: amp, fk, alpha, f_1, f_2), or drop "
            "STAGE_NOISE_PSD_ONLY and let the noise stage fit the foreground."
        )
    _standalone_noise_names = (
        [n for n in _noise_names if n != "galfor_pe"] if _noise_psd_only
        else list(_noise_names))
    if _noise_psd_only:
        print("[combined] STAGE_NOISE_PSD_ONLY=1: the standalone noise "
              f"stage samples {_standalone_noise_names} only; galfor stays "
              "FROZEN at GALFOR_START_PARAMS until gb_search_3 releases it.",
              flush=True)

    # Stage 1: noise alone. Stage 2 and the GB search: noise + VGBs, with the
    # max-logl criterion spanning ALL of them -- one object per stage, so the
    # convergence is joint rather than each move plateauing separately.

    # TODO: CLEAN
    noise_only = [JointMaxLogLSearch(
        "noise_joint_search", list(_standalone_noise_names), branch="psd")]

    # ⚠ ONE OBJECT PER SLOT PER STAGE, NEVER A SHARED INSTANCE.
    # JointMaxLogLSearch carries its own plateau state, so two slots sharing
    # one object pool their convergence and the second inherits the first's
    # already-flat verdict. These used to be four module-level singletons,
    # which was safe only because exactly ONE stage (gb_search_3) sampled the
    # noise. Under GB_SEARCH_SAMPLE_NOISE_ALL_STAGES three stages do, so they
    # are FACTORIES now and every stage builds its own -- the same rule the
    # Move descriptors already follow (`fresh Move descriptor per stage`).
    #
    # Slots _1 / _2 bracket the F-stat birth move; _3 / _4 ride AFTER
    # rj_replace and AFTER rj_prior_removal (user ruling 2026-09-25: "add
    # noise_search (psd and foreground) after each rj proposal to convergence
    # ... as we try to remove the last bit of sources").
    def _noise_slot(name):
        """A fresh interleaved noise-convergence slot for ONE stage."""
        return [JointMaxLogLSearch(name, list(_noise_names), branch="psd")]

    # STANDALONE stage -> the psd-only filter applies here too (the gb_search
    # rider below is a search-stage move and deliberately keeps galfor).
    noise_vgb = [JointMaxLogLSearch(
        "noise_vgb_joint_search",
        _standalone_noise_names + (["vgb_pe"] if _has_vgb else []),
        branch="psd")]
    # The SAME joint move riding inside gb_search, with its OWN plateau rule.
    # It never plateaus for good there -- the GB residual moves every
    # iteration -- so under NOISE_SEARCH_CHECKS=5 it took ~6 rounds of 16 s
    # per GB iteration (93 s, 15% of the iteration, 3mo job 473). The
    # [MAXLOGL] trace shows the re-tracking happens in ROUND 1 (the
    # IMPROVED jumps of hundreds of lnL land there); rounds 2+ add ~5 lnL
    # per round, the tol-level wobble of a stretch ensemble near the mode.
    # GB_SEARCH_NOISE_CHECKS (default 1): keep taking rounds while a round
    # still improves by more than tol, stop at the first flat one -- so
    # after a big residual change the noise gets as many rounds as it
    # needs, and otherwise ~2. GB_SEARCH_NOISE_ITERS_PER_STEP (default 0 =
    # the global MAXLOGL_ITERS_PER_STEP ceiling) is only a hard cap. The
    # standalone noise stages above are untouched.
    _gb_noise_checks = int(os.environ.get("GB_SEARCH_NOISE_CHECKS", "1"))
    _gb_noise_cap = int(os.environ.get("GB_SEARCH_NOISE_ITERS_PER_STEP", "0"))

    def _noise_rider():
        """A fresh leading psd+galfor+vgb rider for ONE stage."""
        return [JointMaxLogLSearch(
            "noise_vgb_joint_search",
            _noise_names + (["vgb_pe"] if _has_vgb else []),
            branch="psd", num_checks=(_gb_noise_checks or None),
            iters_per_step=(_gb_noise_cap or None))]

    # ---- GB_SEARCH_SAMPLE_NOISE_ALL_STAGES (user ruling 2026-09-26) -------
    # Every v9 GB search stage samples the noise, not just gb_search_3: the
    # leading joint rider plus a noise convergence after EVERY in-model
    # convergence proposal ("this way the psd and foreground will adjust as
    # it moves").
    #
    # WHY, from the first 3-month run: with the noise FIXED for two whole
    # stages, the four walkers were frozen at psd values that had drifted
    # far apart -- each walker searching against its own noise floor, with
    # no mechanism to reconcile. The standalone stage cannot fix that on its
    # own either: it maximizes against a residual that still holds the ENTIRE
    # GB galaxy, so whatever it converges to is galaxy-contaminated, and
    # freezing it is what makes the contamination permanent. Sampling it
    # throughout lets the noise track back down as the GB model fills in.
    #
    # ⚠ COST, and it is not small. Each JointMaxLogLSearch riding inside
    # gb_search measured ~93 s per GB iteration at 3mo (job 473, ~15% of the
    # iteration) at NOISE_SEARCH_CHECKS=5; GB_SEARCH_NOISE_CHECKS=1 is what
    # keeps it near ~2 rounds. This puts FIVE of them (rider + four slots)
    # into stages 1 and 2 as well, so budget accordingly and watch
    # [GF_TIMING] for the noise moves' share.
    #
    # ⚠ DEFAULT OFF. The 6mo v9 run shares this driver and its stage table is
    # unchanged: noise FIXED in gb_search_1/2, sampled in gb_search_3.
    _noise_all_stages = _env_flag("GB_SEARCH_SAMPLE_NOISE_ALL_STAGES")
    if _noise_all_stages and not (_has_psd or _has_galfor):
        raise ValueError(
            "GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1 but neither a psd nor a "
            "galfor branch is present (REMOVE_BRANCHES?), so there is no "
            "noise for the search stages to sample."
        )

    # gb_search source cadence (user ruling 2026-09-15: "run them in
    # gb_search as before, but make them run every 10 iterations"): the
    # mbh/emri dense rows cost minutes per pass at near-zero GPU, so in
    # gb_search they propose on every Nth stage iteration (stage-local —
    # full_pe runs them every iteration). Between cadence hits they stay
    # SUBTRACTED at their current coords exactly as before.
    #
    # SOBBH JOINED THE CADENCE (user ruling 2026-09-18), superseding the
    # 09-15 "sobbh's cheap chunked-het rows ride every iteration". They are
    # not cheap: the 4-GPU run measured sobbh_pe at 250 s per propose over
    # 47 calls (min 246, max 337) against mbh 147 s and emri 164 s on ~1 in
    # 4 iterations, i.e. sobbh alone was 43% of a 9.2-min iteration and the
    # single largest cost in gb_search. Its chunked-het scorer bills a FLAT
    # 1.73 s per CALL whatever the batch shape (see
    # sobbhspecialmove._LL_SPAN_KEYS' header), so the only lever on it is
    # how many calls run: num_repeats (20 -> 10 the same day) and this
    # cadence. full_pe is untouched — all three run every iteration there.
    _gb_search_src_every = int(os.environ.get("GB_SEARCH_SOURCE_EVERY", "5"))
    if _gb_search_src_every < 1:
        raise ValueError(
            f"GB_SEARCH_SOURCE_EVERY={_gb_search_src_every} must be >= 1.")

    #: Branches the gb_search cadence applies to. All three armed source
    #: branches as of the 2026-09-18 ruling; kept as a named set so the
    #: 09-15 behaviour (mbh/emri only) is one edit away.
    _GB_SEARCH_CADENCED = ("sobbh", "mbh", "emri")

    def source_pe(gb_search_cadence=False, every=None):
        # Fresh Move descriptors per stage (never share one instance):
        # the armed source PE moves, sobbh -> mbh -> emri (banking order).
        # ``gb_search_cadence`` puts them on the 1-in-N schedule; ``every``
        # (a stage's GB_SEARCH_{N}_SOURCE_EVERY >= 1) replaces the global N.
        def _every(br):
            if not (gb_search_cadence and br in _GB_SEARCH_CADENCED):
                return 1
            return _gb_search_src_every if every is None else int(every)
        return [Move(f"{br}_pe", branch=br, every=_every(br))
                for br, _env, _cls in _SOURCE_BRANCH_ENVS
                if br in armed_sources]

    stages = []
    _skip_src_search = _env_flag("STAGE_SKIP_SOURCE_SEARCH")
    if _skip_src_search and not armed_sources:
        raise ValueError(
            "STAGE_SKIP_SOURCE_SEARCH=1 but no source branches are armed "
            "(MBHB_IDS/EMRI_IDS/SOBHB_IDS empty) -- there is no "
            "source_search stage to skip."
        )
    if armed_sources and _skip_src_search:
        # User ruling 2026-09-14 late: with exact-truth starts
        # (*_START_FACTOR=0, valid now that the source inner moves are
        # eigen rather than stretch and need no ensemble spread) there is
        # nothing for the joint source search to converge. The sources
        # stay SUBTRACTED the whole time exactly as before -- their
        # templates enter the residual at setup_acs from the state
        # coords -- and their PE proposals run only where they already
        # ride: gb_search and full_pe.
        print("[combined] STAGE_SKIP_SOURCE_SEARCH=1: no source_search "
              "stage; sources subtracted at their start coords through "
              "the noise stages, proposals armed in gb_search + full_pe.",
              flush=True)
    if armed_sources and not _skip_src_search:
        # SOURCE SEARCH FIRST (campaign S6: "MBH search first, full_year
        # pattern"): the armed source branches converge under ONE joint
        # max-lnL criterion before anything else runs, so their (loud)
        # power is out of the residual before the noise stages fit the
        # PSD. They start near truth (*_START_FACTOR seeding), so this is
        # a refinement loop, not a blind search. Afterwards they keep
        # PE-sampling through gb_search and full_pe (live residual); they
        # sit converged-and-subtracted during the noise stages.
        #
        # THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08) is this stage, with
        # its OWN plateau rule (SOURCE_SEARCH_CHECKS / _TOL / _MAX_ROUNDS; each
        # unset = the global knob, as before) and a position: ``first`` (here,
        # the default) or ``after_seed`` (between gb_search_seed and
        # gb_search_1, inserted below). A round = ONE pass of the armed source
        # PE moves, each at its in-model repeats ({BRANCH}_NUM_PROP_REPEATS);
        # MAXLOGL_ITERS_PER_STEP rounds per sampler iteration (stored row), the
        # per-walker plateau count carried across iterations (no
        # restart_each_propose); the stage ends when every walker's cold lnL
        # has gained no more than tol over its baseline for ``checks``
        # consecutive rounds (MAXLOGL_PER_WALKER=1: the laggard decides), or
        # at the ceiling.
        _src_branch = next(br for br, _e, _c in _SOURCE_BRANCH_ENVS
                           if br in armed_sources)
        _src_checks, _src_tol, _src_cap = source_search_rule()
        _src_position = source_search_position()
        if _src_position == "after_seed":
            # refuse up front rather than drop or move the stage silently
            _why = ("STAGE_NOISE_ONLY=1" if _env_flag("STAGE_NOISE_ONLY")
                    else "STAGE_NOISE_VGB_PE=1" if _env_flag("STAGE_NOISE_VGB_PE")
                    else "no gb branch (REMOVE_BRANCHES)" if "gb" not in fit.branches
                    else "STAGE_V9_SEARCH=0" if not _v9_search_enabled()
                    else "GB_SEARCH_SEED_ITERS=0" if _seed_iters() == 0
                    else None)
            if _why is not None:
                raise ValueError(
                    f"SOURCE_SEARCH_POSITION=after_seed but this composition has no "
                    f"gb_search_seed stage ({_why}). Use SOURCE_SEARCH_POSITION=first "
                    "or compose the seed.")
        _src_stage = Stage(
            name="source_search", kind="search",
            moves=[JointMaxLogLSearch(
                "source_joint_search",
                [f"{br}_pe" for br, _e, _c in _SOURCE_BRANCH_ENVS
                 if br in armed_sources],
                branch=_src_branch, num_checks=_src_checks, tol=_src_tol,
                max_rounds=_src_cap)],
            combine_kwargs=dict(share_temperature_control=False),
        )
        if _src_position == "first":
            stages.append(_src_stage)
        print(f"[combined] source_search ({_src_position}): "
              f"{'+'.join(_src_stage.moves[0].inner_names)} ONLY (psd/galfor fixed), "
              f"to a per-walker max-lnL plateau: "
              f"{_src_checks if _src_checks is not None else 'NOISE_SEARCH_CHECKS'} "
              f"consecutive flat round(s) within "
              f"{f'{_src_tol:g} nats' if _src_tol is not None else 'MAXLOGL_TOL'}, "
              f"ceiling "
              f"{f'{_src_cap} rounds' if _src_cap is not None else 'MAXLOGL_MAX_ITER'}.",
              flush=True)
    else:
        _src_position, _src_stage = None, None
    # SOURCE WARM START (user ruling 2026-10-08): with SOURCE_WARM_START_STORE
    # set, every mbh/emri/sobbh leaf whose catalogue id the seed store holds at
    # SNR_6mo > SOURCE_WARM_START_SNR_MIN starts at the seed's last cold-chain
    # row (lisatools.globalfit.warmstart.sources, applied by run.py load_info on
    # a FRESH start); every other leaf at *_START_FACTOR. Reported here so the
    # recipe printout and the launch log carry it; the decision itself is
    # logged per branch at load ("[SOURCE-WARM] ...").
    if armed_sources:
        _sws = os.environ.get("SOURCE_WARM_START_STORE", "").strip()
        if _sws:
            print(f"[combined] source warm start: {_sws} (seed SNR > "
                  f"{os.environ.get('SOURCE_WARM_START_SNR_MIN', '10')}; seed ids "
                  f"mbh={os.environ.get('MBH_SOURCE_WARM_START_IDS', '?')} "
                  f"emri={os.environ.get('EMRI_SOURCE_WARM_START_IDS', '?')} "
                  f"sobbh={os.environ.get('SOBBH_SOURCE_WARM_START_IDS', '?')}); "
                  "other leaves at *_START_FACTOR", flush=True)
        else:
            print("[combined] source warm start: OFF (every source leaf at "
                  "*_START_FACTOR)", flush=True)
    # CONDITIONAL NOISE STAGES (v9, user spec 2026-09-24): "noise_search /
    # noise_vgb_search run ONLY when there is no psd/foreground estimate from
    # a previous run." The estimate IS the start pin -- {PSD,GALFOR}_START_
    # PARAMS, which the submit script fills from the previous run's maxlogL
    # point, the same folder the warm start comes from. With every sampled
    # noise branch pinned there is nothing for a convergence-gated noise
    # burn-in to find; without one they run exactly as before.
    #
    # ⚠ The gate requires a pin for EVERY sampled noise branch. A half-pinned
    # start (psd seeded, galfor prior-drawn) with the noise stages skipped
    # would enter gb_search_1 -- which does not sample noise at all -- with a
    # random foreground, and stay there for two whole stages.
    _noise_pinned = bool(os.environ.get("PSD_START_PARAMS", "").strip()) and (
        not _has_galfor
        or bool(os.environ.get("GALFOR_START_PARAMS", "").strip()))
    _force_noise = _env_flag("STAGE_FORCE_NOISE_SEARCH")
    if _v9_search_enabled() and _noise_pinned and not _force_noise:
        print(
            "[combined] v9: SKIPPING noise_search / noise_vgb_search -- the "
            "noise model is PINNED at a previous run's estimate "
            "(PSD_START_PARAMS"
            + (" + GALFOR_START_PARAMS" if _has_galfor else "")
            + "). Search stages without noise moves hold it there; the first "
              "noise-sampling stage releases it (gb_search_3 by default; "
              "GB_SEARCH_{N}_NOISE_MODE). Export STAGE_FORCE_NOISE_SEARCH=1 to "
              "run them anyway.",
            flush=True,
        )
    elif _has_psd and not _env_flag("STAGE_SKIP_NOISE"):
        if _v9_search_enabled() and not _noise_pinned:
            # Name the branch(es) that are actually unpinned. The old wording
            # said "PSD_START_PARAMS / GALFOR_START_PARAMS unset" whichever of
            # the two was missing, which reads as false on a deliberate
            # half-pin -- e.g. the 3mo v9 run, which pins galfor to an offline
            # reference precisely so the noise stage can SEARCH the psd.
            _unpinned = [
                n for n, v in (("PSD_START_PARAMS", _has_psd),
                               ("GALFOR_START_PARAMS", _has_galfor))
                if v and not os.environ.get(n, "").strip()
            ]
            print(
                "[combined] v9: the noise stage(s) are KEPT -- "
                f"{' and '.join(_unpinned)} unset, so that branch starts from "
                "a PRIOR DRAW and must be fitted before the fixed-noise "
                "search stages freeze it.",
                flush=True,
            )
        stages.append(Stage(
            name="noise_search", kind="search", moves=noise_only,
            combine_kwargs=dict(share_temperature_control=False),
        ))
        # STAGE_SKIP_NOISE_VGB (user ruling 2026-09-26, the 3mo v9 run): drop
        # the vgb burn-in stage. With VGB_START_FACTOR=0 the VGBs start at
        # EXACT truth, so a max-logL stage has nothing to converge -- they sit
        # subtracted at truth through the noise stage (which is what makes
        # "every source removed except the GBs" true for a psd-only fit) and
        # begin sampling in gb_search_1, where vgb_pe already rides the
        # fixed-noise stages. Deliberately its OWN knob rather than a second
        # job for STAGE_NOISE_PSD_ONLY.
        if _has_vgb and _env_flag("STAGE_SKIP_NOISE_VGB"):
            print("[combined] STAGE_SKIP_NOISE_VGB=1: no noise_vgb_search "
                  "stage; the VGBs stay at their start coords through "
                  "noise_search and begin sampling in the first GB search "
                  "stage.", flush=True)
        elif _has_vgb:
            # without a vgb branch this stage would duplicate noise_search
            stages.append(Stage(
                name="noise_vgb_search", kind="search", moves=noise_vgb,
                combine_kwargs=dict(share_temperature_control=False),
            ))
    if _env_flag("STAGE_NOISE_ONLY"):
        # Stages 1-2 only: watch the joint psd+galfor search converge without
        # paying for the F-stat grid fit (which lives in the gb_search RJ
        # birth move's setup) or any GB work. The noise stages run FIRST in
        # the full recipe too, so nothing here changes their behaviour -- it
        # just stops afterwards.
        if not stages:
            raise ValueError(
                "STAGE_NOISE_ONLY=1 with STAGE_SKIP_NOISE=1 leaves no stages."
            )
        fit.recipe = Recipe(stages)
        return fit

    if _env_flag("STAGE_NOISE_VGB_PE"):
        # Searches, then PE-sample psd+galfor+vgb — the gate between "the
        # noise search converged" and "turn on the GB machinery": posterior
        # sampling of every non-GB branch, no F-stat fit, no RJ. PE never
        # stops on its own, so NUM_ITERATIONS bounds the run.
        stages.append(Stage(
            name="noise_vgb_pe", kind="pe",
            moves=noise_pe + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ))
        fit.recipe = Recipe(stages)
        return fit

    if "gb" not in fit.branches:
        # REMOVE_BRANCHES took gb: no F-stat machinery, no RJ, nothing to
        # search for. Everything that remains PE-samples together in one
        # full_pe -- the same composition as the stage below minus every
        # gb move (name kept so monitor/digests/resume statuses read the
        # same). PE never stops on its own; NUM_ITERATIONS bounds the run.
        stages.append(Stage(
            name="full_pe", kind="pe",
            moves=noise_pe + source_pe() + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ))
        fit.recipe = Recipe(stages)
        return fit

    # ======================================================================
    # v9: THREE GB SEARCH STAGES (user spec 2026-09-24)
    # ======================================================================
    # Design: docs/superpowers/specs/2026-09-24-v9-search-stage-restructure-
    # design.md. One ``gb_search`` stage becomes three, differing in exactly
    # four things:
    #
    #   stage           noise     phase max   opt SNR   F-stat peak
    #   gb_search_1     FIXED     on          8         8
    #   gb_search_2     FIXED     off         5         6.25
    #   gb_search_3     sampled   off         5         6.25
    #
    # and each runs the same per-iteration move cycle:
    #
    #   1  rj_warm_search      (stage 3: every GB_SEARCH_3_WARM_EVERY iters)
    #   2  in_model            (adaptive passes -- see below)
    #   3  rj_fstat_search
    #   4  in_model_fstat
    #   5  rj_replace          (two internal passes: warm, then F-stat)
    #   6  in_model_replace
    #   7  rj_prior_removal    (removal only) -- THE CYCLE ENDS HERE
    #
    # The cycle ENDS on the removal judge (user amendment 2026-09-24), so
    # every birth and every swap has had a full in-model refinement pass
    # before it is judged for death. A source still sitting at its birth or
    # swap coordinates looks far more deletable than the same source after it
    # has walked onto its peak, and the three in-model slots are each named
    # for the RJ move they polish.
    #
    # THE IN-MODEL SLOTS ARE NOT FIXED-LENGTH. Each is ONE ``propose()`` that
    # internally repeats its whole pass -- GB_NUM_REPEAT_PROPOSALS repeats per
    # source per pass, a FIXED number -- until every occupied (walker, band)
    # sub-band's cold-chain logL has plateaued, or GB_INMODEL_GROUP_MAX_PASSES
    # hits. Fixed repeats, adaptive passes (user confirmation 2026-09-24);
    # per-source totals are consequently not uniform across the proposal,
    # which is accepted in search.
    #
    # "FIXED noise" is the ABSENCE of the psd/galfor moves, not a pin: an
    # unsampled branch does not move, and setup_acs rebuilds each walker's
    # sensitivity from the state coords every pass. The VALUE it is fixed at
    # is PSD_START_PARAMS / GALFOR_START_PARAMS (run.py's noise start pin),
    # which the submit script fills from the SAME previous-run folder the warm
    # start comes from, at that run's maxlogL point.
    #
    # STAGE_V9_SEARCH=0 restores the single legacy ``gb_search`` stage.
    _v9_stages = _v9_search_enabled()

    def in_model(slot):
        """One pure in-model slot, when the move exists to resolve.

        ``build_gb_moves`` registers the three slots only under
        GB_MODE=search + GB_SEARCH_IN_MODEL=1, and a listed-but-unbuilt move
        fails recipe materialization -- so the descriptor has to be
        knob-conditional exactly like warm()/replace(). Fresh Move descriptor
        per call (never share one instance across stages).
        """
        if _drop_slot(slot):
            return []
        if slot == "in_model_removal" and not _inmodel_after_removal():
            return []
        # in_model_replace retires with rj_replace (user ruling
        # 2026-09-26: "in_model_replace we do not need this anymore").
        # It was the polish slot for rj_replace's survivors; with that move
        # off (GB_SEARCH_RJ_REPLACE=0, six consecutive zero-yield proposes)
        # it is simply a THIRD pure in-model pass over the same rows
        # in_model and in_model_fstat already polished -- 1156 s = 14% of
        # job 634's gb_search_1 iteration 1, reaching IMGROUP pass 14.
        # GB_SEARCH_IN_MODEL_REPLACE=1 restores it; reinstate it alongside
        # rj_replace in gb_search_3, not on its own.
        if slot == "in_model_replace" and os.environ.get(
                "GB_SEARCH_IN_MODEL_REPLACE", "0").strip() not in (
                    "1", "true", "True", "yes", "on"):
            return []
        return ([Move(slot, branch="gb")]
                if _env_flag("GB_SEARCH_IN_MODEL") else [])

    def _search_stage(name, *, noise_mode, phase_maximize, opt_snr,
                      peak_min_snr, reset_band_max=False, warm_every=1,
                      seed_only=False, prior_births=None, reset_valves=False,
                      source_every=None, noise_max_rounds=_CYCLE_END_MAX_ROUNDS,
                      inmodel_budget=None):
        # the stage profile as the step sees it; reset_band_max is only
        # written when set so the other stages' dicts stay byte-identical
        _profile = dict(phase_maximize=phase_maximize, opt_snr=opt_snr,
                        peak_min_snr=peak_min_snr)
        if reset_band_max:
            _profile["reset_band_max"] = True
        if inmodel_budget:
            # GB_SEARCH_{N}_NUM_REPEAT_PROPOSALS / _INMODEL_GROUP_ITERS /
            # _INMODEL_CONVERGE_ITERS (2026-10-16): applied by the step at
            # EVERY entry of the stage; written only when set
            _profile["inmodel_budget"] = dict(inmodel_budget)
        if prior_births is not None:
            # move-scoped: what the shared rj_prior_removal proposes HERE
            _profile["prior_births"] = bool(prior_births)
        if reset_valves:
            # GB_SEARCH_{N}_RESET_VALVES (2026-10-07): applied by the step at
            # a FRESH stage entry; written only when set (byte-identical else)
            _profile["reset_valves"] = True
        if noise_mode not in SEARCH_NOISE_MODES:
            raise ValueError(f"{name}: noise_mode={noise_mode!r} is not one of "
                             f"{SEARCH_NOISE_MODES}.")
        if noise_mode != "fixed" and not (_has_psd or _has_galfor):
            raise ValueError(
                f"{name}: noise mode {noise_mode!r} but neither a psd nor a galfor "
                "branch is present (REMOVE_BRANCHES?) -- there is no noise to sample.")
        # "interleaved" is the table's sampled composition (the rider + four
        # slots, or the ratchet gate); everything keyed on ``sample_noise``
        # below is exactly that path. "cycle_end" builds its ONE slot after
        # the source moves (see _noise_end below) and is otherwise "fixed".
        sample_noise = noise_mode == "interleaved"
        if noise_mode == "cycle_end":
            from lisatools.globalfit.noise_ratchet import ratchet_from_env as _rfe

            if _rfe() is not None:
                raise ValueError(
                    f"{name}: GALFOR_RATCHET=1 with the cycle-end noise convergence "
                    f"({_stage_knob(name, 'NOISE_MODE')}=cycle_end): two different "
                    "noise designs for one stage. The ruled 9mo/1yr recipe has no "
                    "ratchet; drop one of the two.")
        # SAMPLED noise: the legacy gb_search composition verbatim -- the
        # leading joint psd+galfor+vgb search plus the two extra re-tracking
        # rounds that bracket the F-stat birth move, so the grid is always
        # fitted against a current noise level. FIXED noise: the vgb branch
        # keeps sampling (it is 55 KNOWN sources, nothing to do with the
        # noise model) and the psd/galfor moves are simply absent.
        # ---- the galfor RATCHET (user design 2026-09-30) -------------------
        # GALFOR_RATCHET=1 replaces the leading rider AND the four interleaved
        # noise slots with ONE gated noise proposal at the head of the
        # iteration, with vgb_pe RIGHT AFTER it (user ruling 2026-09-30: the
        # 55 known sources are refined against the noise the gate just set,
        # never against the one it is about to change). The stage step
        # drives the gate -- nudge / hold / release -- from the stage-local
        # iteration; see lisatools.globalfit.noise_ratchet. Unset, the
        # composition below is byte-identical to before.
        from lisatools.globalfit.noise_ratchet import (
            drop_band_from_env, min_drop_from_env, min_gain_from_env,
            nudge_delta_from_env, ratchet_from_env, stop_rule_from_env)

        _ratchet = ratchet_from_env() if sample_noise else None
        _ratchet_delta = ([float(x) for x in nudge_delta_from_env()]
                          if _ratchet is not None else None)
        # the data-driven stop (user design 2026-10-02): GALFOR_RATCHET_MIN_GAIN
        _ratchet_min_gain = min_gain_from_env() if _ratchet is not None else 0.0
        # WHICH rule ends the ratchet (user ruling 2026-10-03, "adjust away from
        # the raw lnL gain"): GALFOR_RATCHET_STOP_RULE galfor (default) /
        # gain / off; the galfor rule's threshold GALFOR_RATCHET_MIN_DROP
        # (fraction) over GALFOR_RATCHET_DROP_BAND (Hz). Resolved only when
        # the ratchet is armed so an unratcheted launch reads no new knobs.
        _ratchet_stop_rule = stop_rule_from_env() if _ratchet is not None else "gain"
        _ratchet_min_drop = min_drop_from_env() if _ratchet is not None else 0.0
        _ratchet_drop_band = drop_band_from_env() if _ratchet is not None else None
        # GALFOR_RATCHET_END_STAGE_ON_STOP (user design 2026-10-03: "make gb
        # search 3 only the ratcheting and foreground convergence"): the stage
        # ends when the stop rule ends the ratchet; the replica-PE stage follows.
        _ratchet_end_stage = (_env_flag("GALFOR_RATCHET_END_STAGE_ON_STOP", "1")
                              if _ratchet is not None else False)
        # GALFOR_RATCHET_MIN_NUDGES: the stop may not fire before this many
        # nudges have run in the process (user ruling 2026-10-02: "I want to
        # force at least 1 more nudge"); 0 = no floor
        _ratchet_min_nudges = 0
        if _ratchet is not None:
            _ratchet_min_nudges = int(os.environ.get("GALFOR_RATCHET_MIN_NUDGES", "0").strip() or 0)
            if _ratchet_min_nudges < 0:
                raise ValueError(f"GALFOR_RATCHET_MIN_NUDGES={_ratchet_min_nudges} must be >= 0.")
        if _ratchet is not None and not _env_flag("GB_SEARCH_IN_MODEL"):
            raise ValueError(
                "GALFOR_RATCHET=1 needs GB_SEARCH_IN_MODEL=1: every noise change "
                "(forced or free) is followed by one GB in-model pass so the "
                "sources settle to the new noise before any RJ move scores "
                "against it, and that pass is the stock 'in_model' move.")
        _slots = sample_noise and _ratchet is None
        if _ratchet is not None:
            # RELEASE = SEARCH MODE TO CONVERGENCE (user ruling 2026-10-02).
            # The gate's inner is the same psd+galfor max-logL search, but
            # with the STANDALONE stage's plateau rule, not the rider's:
            # GB_SEARCH_NOISE_CHECKS=1 ends the search at the first round in
            # which no walker gains more than MAXLOGL_TOL, which on job 675's
            # first release fired after 17 rounds while the chain was still
            # climbing 50-100 lnL per round (walker spread 656). The release
            # criterion is GALFOR_RATCHET_RELEASE_CHECKS consecutive flat
            # rounds per walker (default NOISE_SEARCH_CHECKS, 5), capped at
            # GALFOR_RATCHET_RELEASE_ITERS rounds per release (200; a round
            # is PSD_NUM_PROP_REPEATS + GALFOR_NUM_PROP_REPEATS proposals per
            # walker, ~1-5 s). The search starts afresh on every release.
            _rel_checks = int(os.environ.get(
                "GALFOR_RATCHET_RELEASE_CHECKS",
                os.environ.get("NOISE_SEARCH_CHECKS", "5")))
            # the cap exists so a release cannot run unbounded on a broken
            # likelihood; it must never be the thing that ENDS a healthy one
            # (user ruling 2026-10-02: "make this higher. We do not want it
            # to hit that"). 5000 rounds is 1.5-7 h at 1-5 s per round; job
            # 675's release from the pinned start took 17.
            _rel_cap = int(os.environ.get("GALFOR_RATCHET_RELEASE_ITERS", "5000"))
            # the release's OWN tolerance (user ruling 2026-10-02: 5, not the
            # global MAXLOGL_TOL=20 -- "I just want to see it converged")
            _rel_tol = float(os.environ.get("GALFOR_RATCHET_RELEASE_TOL", "5"))
            if _rel_checks < 1 or _rel_cap < 1 or not _rel_tol > 0:
                raise ValueError(
                    f"GALFOR_RATCHET_RELEASE_CHECKS={_rel_checks} and "
                    f"GALFOR_RATCHET_RELEASE_ITERS={_rel_cap} must both be >= 1 and "
                    f"GALFOR_RATCHET_RELEASE_TOL={_rel_tol} > 0.")
            _noise = [
                GatedNoiseSearch(
                    "noise_ratchet_search", _noise_names, _ratchet_delta,
                    branch="psd", num_checks=_rel_checks,
                    iters_per_step=_rel_cap, release_to_convergence=True,
                    release_tol=_rel_tol)
            ] + ([Move("vgb_pe", branch="vgb")] if _has_vgb else [])
            print(f"[combined] {name}: GALFOR_RATCHET {_ratchet} -- one gated "
                  f"noise proposal at the head of the iteration, nudge "
                  f"{list(_ratchet_delta)} in the sampled galfor basis; the "
                  f"interleaved noise_joint_search_1..4 slots are OUT. A "
                  f"RELEASE runs the psd+galfor search afresh to convergence: "
                  f"{_rel_checks} consecutive flat round(s) per walker within "
                  f"{_rel_tol:g} nats, at most {_rel_cap} rounds per release."
                  + (" The schedule opens with a RELEASE before the first nudge."
                     if _ratchet.release_first else "")
                  + (f" Data-driven stop (GALFOR_RATCHET_STOP_RULE=galfor): after a release "
                     f"whose median-walker foreground at "
                     f"{1e3 * _ratchet_drop_band[0]:g}-{1e3 * _ratchet_drop_band[1]:g} mHz "
                     f"fell by less than {100.0 * _ratchet_min_drop:g}% of the previous "
                     f"release's (each walker against itself), no more nudges (the "
                     f"{_ratchet.cycles}-cycle count is a ceiling; the lnL gain is logged "
                     f"but does not decide)."
                     if _ratchet_stop_rule == "galfor" else
                     f" Data-driven stop (GALFOR_RATCHET_STOP_RULE=gain): after a release "
                     f"whose max cold lnL gains less than {_ratchet_min_gain:.0f} over the "
                     f"previous release, no more nudges (the {_ratchet.cycles}-cycle count "
                     f"is a ceiling)."
                     if _ratchet_stop_rule == "gain" and _ratchet_min_gain > 0 else
                     f" No data-driven stop (GALFOR_RATCHET_STOP_RULE={_ratchet_stop_rule}"
                     f"{'' if _ratchet_stop_rule == 'off' else ', MIN_GAIN 0'}): the "
                     f"{_ratchet.cycles}-cycle count ends the ratchet."),
                  flush=True)
        else:
            _noise = (_noise_rider() if sample_noise
                      else ([Move("vgb_pe", branch="vgb")] if _has_vgb else []))
        _noise_pre = _noise_slot("noise_joint_search_1") if _slots else []
        _noise_post = _noise_slot("noise_joint_search_2") if _slots else []
        # NOISE TO CONVERGENCE AFTER EVERY RJ PROPOSAL -- specifically,
        # after that proposal's IN-MODEL UPDATE (user ruling 2026-09-25,
        # amended the same day: "move the noise joint searches to after the
        # in model updates"). gb_search_3 only; it is the one stage that
        # SAMPLES the noise.
        #
        # The pattern is uniform, RJ -> in-model -> noise:
        #
        #   rj_warm_search   in_model          _noise_pre
        #   rj_fstat_search  in_model_fstat    _noise_post
        #   rj_replace       in_model_replace  _noise_rep
        #   rj_prior_removal (none)            _noise_rem
        #
        # WHY AFTER THE POLISH RATHER THAN STRAIGHT AFTER THE RJ. A newborn
        # sits at its birth coordinates until its in-model slot walks it
        # onto its peak; converging psd+galfor in between would fit the
        # foreground against a GB model that is known to be half-updated,
        # and then immediately invalidate it. Letting the polish finish
        # first means each noise convergence sees a settled GB model.
        #
        # ⚠ _noise_pre KEEPS ITS POSITION and it is load-bearing: it is
        # already "after in_model", and it also lands immediately before
        # rj_fstat_search, which is where the F-stat grid FIT reads the
        # noise level. The amendment made the other three match IT, not the
        # other way round.
        #
        # WHY, in the user's words: "when we are sampling the
        # noise/foreground it needs to move to convergence after each
        # proposal as we try to remove the last bit of sources." By stage 3
        # what is left is marginal, and a stale foreground is exactly what
        # decides whether a marginal source looks real.
        #
        # ⚠ COST. Each JointMaxLogLSearch riding inside gb_search measured
        # ~6 rounds x 16 s = 93 s per GB iteration at 3mo (job 473, ~15% of
        # the iteration). Going from two interleaved to four roughly doubles
        # that share, and 6mo rounds are dearer than 3mo ones.
        # GB_SEARCH_NOISE_CHECKS=1 is what keeps it affordable -- it stops at
        # the first round that fails to improve by more than tol, so a slot
        # whose residual barely moved pays ~2 rounds, not 6.
        # Stages 1-2 are unaffected: they do not sample the noise, so every
        # one of these lists is empty there.
        _noise_rep = _noise_slot("noise_joint_search_3") if _slots else []
        _noise_rem = _noise_slot("noise_joint_search_4") if _slots else []
        # ---- THE CYCLE-END NOISE CONVERGENCE (user ruling 2026-10-07) -------
        # GB_SEARCH_{N}_NOISE_MODE=cycle_end: "during gb search 2 the galfor and
        # psd sampling should be required to converge before moving on to the
        # other proposals. There should only be one noise/galfor proposal
        # though. Not multiple times during the cycle. Just have the galfor/psd
        # convergence proposal at the end of the cycle." ONE psd+galfor joint
        # max-lnL search, LAST in the cycle (after the source moves), run TO
        # CONVERGENCE on every cycle: the standalone noise stage's plateau rule
        # (NOISE_SEARCH_CHECKS flat rounds per walker within MAXLOGL_TOL,
        # MAXLOGL_PER_WALKER), started afresh each propose
        # (restart_each_propose -- a plateaued instance would otherwise take one
        # round per call, the rider behaviour), with a round CEILING that is a
        # safety cap only. vgb_pe keeps leading the cycle as in a fixed-noise
        # stage: the VGBs are 55 known sources, not noise. Under search legs the
        # tail belongs to the last leg, so the row saved after the cycle holds
        # the converged noise.
        _noise_end = []
        if noise_mode == "cycle_end":
            _noise_end = [JointMaxLogLSearch(
                "noise_cycle_end_search", list(_noise_names), branch="psd",
                num_checks=int(os.environ.get("NOISE_SEARCH_CHECKS", "5")),
                iters_per_step=int(noise_max_rounds),
                restart_each_propose=True)]
            print(f"[combined] {name}: noise sampled by ONE {'+'.join(_noise_names)} "
                  f"search TO CONVERGENCE at the END of every cycle "
                  f"({_noise_end[0].num_checks} flat round(s) per walker within "
                  f"MAXLOGL_TOL={os.environ.get('MAXLOGL_TOL', '5.0 (code default)')} "
                  f"nats, afresh each cycle, ceiling {int(noise_max_rounds)} rounds); "
                  f"no leading rider, no interleaved slots.", flush=True)
        # GB_SEARCH_{N}_SOURCE_EVERY: 0 = no sobbh/mbh/emri moves in this stage
        # (vgb, the warm start and the ridge moves stay); N = their cadence;
        # None = the global GB_SEARCH_SOURCE_EVERY, as before
        _sources = ([] if source_every == 0
                    else source_pe(gb_search_cadence=True, every=source_every))
        _warm = ([Move("rj_warm_search", branch="gb", every=warm_every)]
                 if warm() else [])
        if seed_only:
            # ---- gb_search_seed (user ruling 2026-09-26) ----------------
            # "The new GB 1 is just the warmstart RJ + in model to
            # convergence. Run that 5 times (1 per full sampler
            # iteration)." The point is to seed HARD off the warm-start
            # posterior before any F-stat grid is fitted, so this stage
            # deliberately carries NO rj_fstat_search -- which also means
            # GB_FSTAT_REFIT_EVERY=1 cannot trigger a fit here.
            #
            # Everything else is the SAME profile as gb_search_1 (opt_snr
            # 8, phase maximization on, peak floor 8) -- the caller passes
            # that dict verbatim.
            #
            # vgb_pe rides along (~8 s): it keeps the vgb branch sampling
            # and keeps the stage structurally parallel to the others. The
            # source PE moves (sobbh / mbh / emri) are the ones deliberately
            # absent -- they were 2694 s of job 628's first iteration and
            # are exactly the wait this stage exists to avoid.
            #
            # TODO (1-YEAR RUNS): revisit whether the OTHER SOURCES belong
            # in this stage. The 6mo argument for leaving them out is a
            # pure wall-time one -- 2694 s against a seed stage meant to be
            # cheap -- and it assumes the GB warm start is not materially
            # biased by an unconverged MBH/EMRI/SOBBH model sitting in the
            # residual. At 1 yr both halves of that weaken: the source
            # models are dearer AND their residual contribution is larger,
            # so 5 iterations of hard GB seeding against a stale source
            # model could bake in GB leaves that are really mis-subtracted
            # source power. Decide it with a measurement, not by analogy --
            # run the seed stage once with and once without the source
            # moves and compare the GB leaf sets (the 19.668 mHz
            # split-source check and the sub-3-bin pair count above 9 mHz
            # are the discriminators that have worked before). Note the
            # eigen sidecar makes the source moves much cheaper on a
            # resume, which changes the cost side of the trade too.
            return Stage(
                name=name, kind="gb_search",
                moves=_noise + _warm + in_model("in_model"),
                step_kwargs=dict(
                    profile=dict(_profile),
                    stage_name=name,
                    # FIXED length: no plateau test, no convergence wait.
                    convergence_fn=FixedIterationStop(_seed_iters()),
                    # ⚠ The per-(walker, band) RJ shutoff valve is DISARMED
                    # here. It freezes a band's RJ after
                    # GB_SEARCH_BAND_SHUTOFF_CONV_ITER (=3) flat iterations
                    # WITHIN a step -- on a 5-iteration stage whose whole
                    # job is to birth hard from the warm start, that would
                    # cut the seeding short in exactly the bands that are
                    # slowest to pay off. The later stages keep it.
                    search_shutoff_per_walker=False,
                ),
            )
        # ---- search LEGS (user design 2026-09-30) ---------------------------
        # GB_SEARCH_LEGS=1: one stored row per leg of the cycle (after every
        # in_model* move) instead of one per full cycle; the resume position
        # is the leg-ender NAME the last row was saved after. Cadences
        # (warm_every) and the galfor ratchet then count CYCLES. Off = today.
        _legs = _env_flag("GB_SEARCH_LEGS")
        _combine_kwargs = dict(share_temperature_control=False)
        if _legs:
            _combine_kwargs["leg_ends"] = "auto"
        return Stage(
            name=name, kind="gb_search",
            # THE OTHER SOURCES RUN LAST (user ruling 2026-09-25). They used
            # to lead the stage. Three reasons, in order of how much they
            # matter:
            #
            #  1. GIBBS ORDERING. The GB cycle now refines against the
            #     residual the sources left, and the sources are then updated
            #     against the freshly-improved GB model rather than against
            #     one a full iteration old.
            #  2. PREEMPTION COST. Mid-iteration checkpoints sit at every
            #     sub-move boundary, so the block that runs last is the block
            #     a spot kill throws away. Measured on job 620 iteration 1:
            #     sobbh 144 s + mbh 262 s + emri 1266 s = ~28 min, against a
            #     GB cycle that had not started. Losing the sources is much
            #     cheaper than losing the GB cycle plus a comb scan.
            #  3. On the 1-in-10 cadence (GB_SEARCH_SOURCE_EVERY=10) nine
            #     iterations in ten now open on GB work instead of idling
            #     through a source block that is not even scheduled.
            #
            # ⚠ ONLY ``source_pe`` MOVES -- sobbh / mbh / emri. The NOISE
            # slots stay exactly where they are: ``_noise`` leads, and
            # ``_noise_pre`` / ``_noise_post`` BRACKET the F-stat birth move
            # on purpose, so the grid is always fitted against a current
            # noise level. In stages 1-2 ``_noise`` is the lone ``vgb_pe``
            # standing in for the absent psd/galfor moves, which keeps those
            # stages structurally parallel to stage 3's
            # ``noise_vgb_joint_search``. Moving either would be a different
            # and much less safe change.
            moves=(
                _noise
                + _warm + in_model("in_model") + _noise_pre
                + [Move("rj_fstat_search", branch="gb")]
                + in_model("in_model_fstat")
                + _noise_post
                + replace()
                + in_model("in_model_replace")
                + _noise_rep
                + [Move("rj_prior_removal", branch="gb")]
                + in_model("in_model_removal")
                + _noise_rem
                + ([Move("gb_ridge_gibbs", branch="gb")]
                   if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
                + vgb_ridge()
                + _sources
                + _noise_end
            ),
            step_kwargs=dict(
                plateau_branch="gb",
                convergence_iter=int(os.environ.get("GB_PLATEAU_ITERS", "5")),
                stage_name=name,
                profile=dict(_profile),
                # the galfor ratchet schedule (None = off), its nudge and the
                # data-driven stop (0 = schedule alone)
                ratchet=_ratchet,
                ratchet_delta=_ratchet_delta,
                ratchet_min_gain=_ratchet_min_gain,
                ratchet_min_nudges=_ratchet_min_nudges,
                ratchet_stop_rule=_ratchet_stop_rule,
                ratchet_min_drop=_ratchet_min_drop,
                ratchet_drop_band=_ratchet_drop_band,
                ratchet_end_stage_on_stop=_ratchet_end_stage,
                legs=_legs,
            ),
            combine_kwargs=_combine_kwargs,
        )

    if _v9_stages:
        # Warm start every Nth iteration in stage 3 only (user ruling
        # 2026-09-24). By then the previous run's posterior has been mined;
        # what remains is expensive per hit and rarely productive, so it
        # rides a cadence instead of the every-iteration schedule stages 1-2
        # give it.
        _warm3 = int(os.environ.get("GB_SEARCH_3_WARM_EVERY", "5"))
        if _warm3 < 1:
            raise ValueError(
                f"GB_SEARCH_3_WARM_EVERY={_warm3} must be >= 1.")
        # gb_search_seed leads (user ruling 2026-09-26), NOT a renumber of
        # the three that follow: gb_search_1/2/3 keep their names, their
        # knobs (GB_SEARCH_3_WARM_EVERY) and their meaning in every existing
        # log and snapshot. It takes gb_search_1's profile verbatim -- the
        # user's "identical to what is now gb search 2 (e.g. opt snr 8,
        # phase max, etc.)", stage 1 being the one that carries that profile.
        if _noise_all_stages:
            print("[combined] GB_SEARCH_SAMPLE_NOISE_ALL_STAGES=1: every GB "
                  "search stage samples the noise -- leading joint rider + a "
                  "noise convergence after every in-model slot, in "
                  "gb_search_1/2 as well as gb_search_3.", flush=True)
        # Resolved ONCE per assembly, as in the GB_ONLY branch above.
        _profiles = search_stage_profiles()
        _seed = []
        if _seed_iters() > 0:
            _seed_prof = _seed_profile(_profiles)
            _seed = [_search_stage("gb_search_seed", noise_mode="fixed",
                                   seed_only=True, **_seed_prof)]
        # ONE PE composition, used twice (user design 2026-10-03: "an exact
        # replica of pe mode (and I mean exact) ... The only difference between
        # replica pe and full pe is that the other sources (emris, mbhs,
        # sobhbs) are still only run every five iterations"). The move list,
        # the PE declarations and the peak floor are built by these two helpers
        # for BOTH stages, so the replica cannot drift from full_pe except in
        # the one argument that names the difference.
        def _pe_moves(source_cadence: bool):
            # The peak floor is DECLARED rather than inherited: the search
            # stages install a process-global override of it, and the PE
            # stages want the loose 6.25 (an assembled model's faint tail must
            # stay reachable) -- but getting it by accident from gb_search_3 is
            # not the same as choosing it.
            # rj_fstat_pe + rj_fstat_pe_residual (ruling 2026-10-12): the two
            # F-stat moves split the draw's F-stat share (pe_fstat_moves)
            return (noise_pe + source_pe(gb_search_cadence=source_cadence) + warm_pe()
                    + pe_fstat_moves() + [
                Move("rj_prior_pe", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
                + vgb + vgb_ridge())

        def _pe_step_kwargs(stage_name: str):
            return dict(peak_min_snr=_PE_PEAK_MIN_SNR,
                        stage_name=stage_name,
                        # per-stage in-model repeats (PE_INMODEL_REPEATS)
                        pe_repeats=_pe_inmodel_repeats(),
                        # GB_PE_RJ_FLIP_FRACTION, applied on entry: in a
                        # search-mode run the PE RJ moves are BUILT with the
                        # search fraction (1.0)
                        pe_rj_flip_fraction=_pe_rj_flip_fraction())

        _replica = []
        if _env_flag("STAGE_REPLICA_PE", "1"):
            _rk = _replica_pe_knobs()
            # ⚠ weighted_cycle OFF for the replica: the combine refuses a
            # per-move cadence on its drawn-cycle fallback. Under the PE RJ
            # draw-one plan (GB_PE_RJ_DRAW_ONE, the production setting) that
            # fallback is never reached -- full_pe and the replica run the
            # same plan -> cadence filter -> sequential propose -- so the
            # stages are identical there; only with the draw turned off would
            # full_pe draw a weighted cycle while the replica runs the fixed
            # sequence.
            _replica = [Stage(
                name="replica_pe", kind="replica_pe",
                moves=_pe_moves(True),
                step_kwargs=dict(**_pe_step_kwargs("replica_pe"), **_rk),
                combine_kwargs={**_pe_combine_kwargs(), "weighted_cycle": False},
            )]
            print(f"[combined] replica_pe: full_pe's moves and declarations with the "
                  f"source PE moves on the gb_search cadence (every "
                  f"{_gb_search_src_every}); ENDS when min/mean/max cold leaves and "
                  f"min/mean/max cold lnL each trend by less than {_rk['leaf_tol']:g} "
                  f"leaves / {_rk['lnl_tol']:g} nats over the last {_rk['window']} rows "
                  f"(at least {_rk['min_iters']} rows in-stage); full_pe then begins = "
                  f"the start of sample taking (its start_iteration stamp).", flush=True)
        # SOURCE_SEARCH_POSITION=after_seed (2026-10-08): the fixed-dim source
        # seed between gb_search_seed and gb_search_1 (refused above when no
        # seed composes, so it can never land anywhere else).
        _src_after_seed = []
        if _src_position == "after_seed":
            if not _seed:
                raise AssertionError("after_seed with no gb_search_seed composed")
            _src_after_seed = [_src_stage]
        # Per-stage noise mode / source cadence (2026-10-07). ``_sampled`` is
        # the TABLE's flag: it still keys the warm cadence (a stage-3 property,
        # GB_SEARCH_3_WARM_EVERY), whatever GB_SEARCH_{N}_NOISE_MODE says.
        stages += _seed + _src_after_seed + [
            _search_stage(_name,
                          noise_mode=search_stage_noise_mode(_name, _sampled),
                          warm_every=(_warm3 if _sampled else 1),
                          source_every=search_stage_source_every(_name),
                          noise_max_rounds=search_stage_noise_max_rounds(_name),
                          **_prof)
            for _name, _prof, _sampled in _profiles
        ] + _replica + [
            Stage(
                name="full_pe", kind="pe",
                moves=_pe_moves(False),
                step_kwargs=_pe_step_kwargs("full_pe"),
                combine_kwargs=_pe_combine_kwargs(),
            ),
        ]
        fit.recipe = Recipe(stages)
        return fit

    stages += [
        Stage(
            name="gb_search", kind="rj",
            # Noise stays in SEARCH (joint max-logl) mode through the GB
            # search. Per-stage move-name uniqueness (recipe.py ebd8612) is
            # what lets these recur across stages.
            # rj_fstat_search (ex rj_prior_search), NOT rj_fstat_mcmc_search (2026-08-12): the
            # serial-MCMC move scores gb.get_fstat_ll -- an FD kernel --
            # against the parent ACA, which is WDM here (wrong domain), and
            # carries leaf_cap_update=False. rj_prior_search is the
            # GPU-verified WDM birth engine (sig-het F-stat grids via
            # route_sighet_fstat, epoch refits, D/2 caps as the DESIGNATED
            # updater, at-cap skip) with use_prior_removal built in; the
            # removal-only move follows it per the search-cycle order.
            # Re-wiring the serial MCMC onto the sig-het scorer is a
            # post-run item (its batches are not f0-ordered, so the
            # reference-block stash would thrash every step).
            # rj_warm_search (when armed) runs IMMEDIATELY BEFORE
            # rj_fstat_search -- warm-start births first, then the F-stat
            # grid births (user ruling 2026-08-24); rj_replace (default
            # on) IMMEDIATELY AFTER them -- the exact-MH F-stat
            # replacement pass before the removal judge.
            # Armed source PE moves ride between the noise joint search
            # and the GB RJ cycle (sobbh -> mbh -> emri banking order).
            # mbh/emri ride at the 1-in-N gb_search cadence (block above);
            # sobbh every iteration.
            moves=_noise_rider() + source_pe(gb_search_cadence=True) + warm() + _noise_slot("noise_joint_search_1") + [
                Move("rj_fstat_search", branch="gb"),
            ] + _noise_slot("noise_joint_search_2") + replace() + [
                Move("rj_prior_removal", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
            + vgb_ridge(),
            step_kwargs=dict(
                plateau_branch="gb",
                convergence_iter=int(os.environ.get("GB_PLATEAU_ITERS", "5")),
            ),
            combine_kwargs=dict(share_temperature_control=False),
        ),
        Stage(
            name="full_pe", kind="pe",
            # No rj_fstat_mcmc: the pe-named serial MCMC has the same
            # FD-kernel-on-WDM-data fstat scoring as its search twin.
            # rj_fstat_pe + rj_prior_pe are the GB PE moves. Armed source
            # PE moves follow the noise moves (the full_year composition:
            # noise, then sobbh -> mbh -> emri, then gb).
            # rj_warm_pe (when armed) runs IMMEDIATELY BEFORE rj_fstat_pe
            # -- the PE mirror of the search-stage order.
            moves=noise_pe + source_pe() + warm_pe() + pe_fstat_moves() + [
                Move("rj_prior_pe", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
            + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ),
    ]
    fit.recipe = Recipe(stages)
    return fit


# ---- THE RECIPE PRINTOUT (2026-10-07) ---------------------------------------
# ``--print-recipe`` / ``GF_PRINT_RECIPE=1``: compose the fit exactly as a run
# would (build_fit -- construction only: no data, no backend, no MPI) and print
# what it composed. ONE resolver, two consumers: the description is read off the
# built Recipe, so it cannot disagree with what the run would execute.

_SOURCE_MOVES = ("sobbh_pe", "mbh_pe", "emri_pe")


def _envv(knob, default="unset"):
    v = os.environ.get(knob)
    return f"{knob}={v if v not in (None, '') else default}"


def _describe_move(m) -> str:
    label = m.name
    inner = "+".join(getattr(m, "inner_names", ()) or ())
    if isinstance(m, GatedNoiseSearch):
        label += f"  [galfor RATCHET gate over {inner}]"
    elif isinstance(m, JointMaxLogLSearch):
        checks = (m.num_checks if m.num_checks is not None
                  else int(os.environ.get("NOISE_SEARCH_CHECKS", "5")))
        cap = (m.iters_per_step if m.iters_per_step is not None
               else int(os.environ.get("MAXLOGL_ITERS_PER_STEP", "10")))
        tol = os.environ.get("MAXLOGL_TOL", "5 (code default)")
        # the instance's OWN tolerance / ceiling when it carries one (the
        # source_search stage); otherwise the globals, worded as before
        own_tol = getattr(m, "tol", None)
        own_cap = getattr(m, "max_rounds", None)
        if getattr(m, "restart_each_propose", False):
            within = (f"{own_tol:g} nats" if own_tol is not None
                      else f"MAXLOGL_TOL={tol} nats")
            label += (f"  [{inner}: TO CONVERGENCE, afresh every cycle -- {checks} "
                      f"flat round(s) per walker within {within} "
                      f"({_envv('MAXLOGL_PER_WALKER', '1')}), ceiling {cap} rounds]")
        else:
            within = f"{own_tol:g} nats" if own_tol is not None else f"MAXLOGL_TOL={tol}"
            label += (f"  [{inner}: joint max-lnL plateau, {checks} flat round(s) "
                      f"within {within}, at most {cap} round(s) per propose"
                      + (f", ceiling {own_cap} rounds" if own_cap is not None else "")
                      + "]")
    ev = int(getattr(m, "every", 1) or 1)
    if ev != 1:
        label += f"  (every {ev} iterations)"
    return label


def _describe_noise(st) -> str:
    names = [m.name for m in st.moves]
    gate = [m for m in st.moves if isinstance(m, GatedNoiseSearch)]
    end = [m for m in st.moves if isinstance(m, JointMaxLogLSearch)
           and getattr(m, "restart_each_propose", False)]
    slots = [n for n in names if n.startswith("noise_joint_search_")]
    if gate:
        return "SAMPLED: ONE galfor-ratchet gate at the head of the cycle (GALFOR_RATCHET=1)"
    if end:
        where = "LAST" if st.moves[-1] is end[-1] else "NOT last (!)"
        return (f"SAMPLED at the END of the cycle: ONE {'+'.join(end[0].inner_names)} "
                f"search to convergence ({where} in the move list); no rider, no slots")
    if "noise_vgb_joint_search" in names or slots:
        return (f"SAMPLED, interleaved: leading psd+galfor+vgb rider + {len(slots)} "
                f"noise convergence slot(s) after the in-model moves")
    pe = [n for n in names if n in ("psd_pe", "galfor_pe")]
    if pe:
        return f"SAMPLED by the PE moves {pe}"
    if st.kind == "search":
        return "the stage's own joint max-lnL search (see moves)"
    return "FIXED (no psd/galfor move: held where the last noise stage / start pin left it)"


def _describe_sources(st) -> str:
    ev = {m.name: int(getattr(m, "every", 1) or 1) for m in st.moves
          if m.name in _SOURCE_MOVES}
    if not ev:
        return "none (no sobbh/mbh/emri move in this stage)"
    if len(set(ev.values())) == 1:
        e = next(iter(ev.values()))
        return (f"{'/'.join(ev)} " + ("EVERY iteration" if e == 1
                                      else f"every {e} iterations"))
    return ", ".join(f"{n} every {e}" for n, e in ev.items())


def _describe_source_search_stop(st) -> str:
    """The fixed-dim source seed's plateau rule, resolved, with its knobs."""
    m = next(mv for mv in st.moves if isinstance(mv, JointMaxLogLSearch))
    if m.num_checks is not None:
        checks, ck = m.num_checks, f"SOURCE_SEARCH_CHECKS={m.num_checks}"
    else:
        checks = int(os.environ.get("NOISE_SEARCH_CHECKS", "5"))
        ck = f"SOURCE_SEARCH_CHECKS unset -> NOISE_SEARCH_CHECKS={checks}"
    if m.tol is not None:
        tol, tk = f"{m.tol:g}", f"SOURCE_SEARCH_TOL={m.tol:g}"
    else:
        tol = os.environ.get("MAXLOGL_TOL", "5 (code default)")
        tk = f"SOURCE_SEARCH_TOL unset -> MAXLOGL_TOL={tol}"
    _mi = int(os.environ.get("MAXLOGL_MAX_ITER", "0") or 0)
    if m.max_rounds is not None:
        cap = f"{m.max_rounds} rounds (SOURCE_SEARCH_MAX_ROUNDS={m.max_rounds})"
    elif _mi:
        cap = f"{_mi} rounds (SOURCE_SEARCH_MAX_ROUNDS unset -> MAXLOGL_MAX_ITER={_mi})"
    else:
        cap = "none (SOURCE_SEARCH_MAX_ROUNDS and MAXLOGL_MAX_ITER unset: unbounded)"
    per_walker = os.environ.get("MAXLOGL_PER_WALKER", "1").strip() not in (
        "0", "false", "False", "no", "off", "")
    chunk = (m.iters_per_step if m.iters_per_step is not None
             else int(os.environ.get("MAXLOGL_ITERS_PER_STEP", "10")))
    return (f"every walker's cold lnL gains no more than {tol} nats over its baseline "
            f"for {checks} consecutive round(s) ({ck}; {tk}; "
            f"{_envv('MAXLOGL_PER_WALKER', '1')}: "
            + ("the laggard walker decides" if per_walker else "the BEST walker decides")
            + f"), ceiling {cap}; a round = one pass of {'+'.join(m.inner_names)} "
            f"at their in-model repeats, at most {chunk} round(s) per stored row "
            f"(MAXLOGL_ITERS_PER_STEP), the count carried across rows")


def _describe_pe_fstat(st) -> str:
    """A PE stage's F-stat birth moves: what each fits, its cadence and floor,
    and its share of the PE RJ draw (ruling 2026-10-12), resolved."""
    from lisatools.globalfit.moves.globalfitmove import pe_rj_fstat_residual_split
    from lisatools.globalfit.recipe import PE_FSTAT_RESIDUAL_MOVE, pe_fstat_move_config

    names = [m.name for m in st.moves]
    f_tot = float(os.environ.get("GB_PE_RJ_FSTAT_FRACTION", "0.8"))
    split = pe_rj_fstat_residual_split()
    both = "rj_fstat_pe" in names and PE_FSTAT_RESIDUAL_MOVE in names
    draw_one = os.environ.get("GB_PE_RJ_DRAW_ONE", "0") == "1"
    parts = []
    for name, what in (("rj_fstat_pe", "GB-FREE residual (the whole galaxy), root shared"),
                       (PE_FSTAT_RESIDUAL_MOVE,
                        "LIVE residual (GBs subtracted), root shared_pe_residual")):
        if name not in names:
            continue
        cfg = pe_fstat_move_config(name)
        ev = cfg["fstat_refit_every"]
        floor = cfg["fstat_peak_min_snr"]
        p = (f_tot * (split if name == PE_FSTAT_RESIDUAL_MOVE else 1.0 - split)
             if both else f_tot)
        parts.append(
            f"{name} = {what}, one random cold walker per epoch "
            f"({_envv('GB_FSTAT_PE_REF', 'random')}); "
            + ("fit ONCE (first visit / the stored epoch), never refit"
               if ev == 0 else f"refit every {ev} elapsed iterations")
            + "; peak floor "
            + (f"SNR {floor:g} (its own)" if floor is not None
               else "the stage's declared floor")
            + (f"; P(draw)={p:.3f}" if draw_one else ""))
    if not parts:
        return "none"
    return (" | ".join(parts)
            + f"  [GB_PE_RJ_FSTAT_FRACTION={f_tot:g} split "
              f"GB_PE_RJ_FSTAT_RESIDUAL_SPLIT={split:g}]")


def _describe_stop(st) -> str:
    kw = st.step_kwargs or {}
    fn = kw.get("convergence_fn")
    if isinstance(fn, FixedIterationStop):
        return f"FIXED {fn.n} sampler iterations (GB_SEARCH_SEED_ITERS)"
    if st.kind == "gb_search":
        ci = kw.get("convergence_iter")
        out = (f"every OCCUPIED (walker, band) pair shut off "
               f"({_envv('GB_SEARCH_STAGE_END_ON_SHUTOFF', '1 (code default)')}; valve "
               f"{_envv('GB_SEARCH_BAND_SHUTOFF_PER_WALKER')}, "
               f"{_envv('GB_SEARCH_BAND_SHUTOFF_CONV_ITER', '5 (code default)')}, "
               f"{_envv('GB_SEARCH_BAND_SHUTOFF_LL_TOL')}), or the nleaves plateau "
               f"(GB_PLATEAU_ITERS={ci}: window {2 * int(ci or 0) + 1} in-stage rows) "
               f"AND every occupied pair shut")
        if kw.get("ratchet") is not None and kw.get("ratchet_end_stage_on_stop"):
            out += "; OR the galfor ratchet's stop (GALFOR_RATCHET_END_STAGE_ON_STOP)"
        return out
    if st.kind == "replica_pe":
        return (f"min/mean/max cold leaves and lnL each trend < {kw.get('leaf_tol'):g} "
                f"leaves / {kw.get('lnl_tol'):g} nats over {kw.get('window')} rows "
                f"(>= {kw.get('min_iters')} rows in-stage)")
    if st.kind == "pe":
        return "none: PE never stops on its own (NUM_ITERATIONS bounds the run)"
    if st.kind == "search" and st.name == "source_search":
        return _describe_source_search_stop(st)
    if st.kind == "search":
        return "the joint max-lnL plateau inside the stage's move (done on its first check)"
    if st.kind == "rj":
        return f"nleaves plateau (convergence_iter={kw.get('convergence_iter')})"
    return "?"


def describe_recipe(fit) -> str:
    """The resolved recipe of a BUILT-but-not-run fit, as text.

    Per stage, in order: name, kind, stop rule, the search profile (opt_snr,
    peak_min_snr, phase_maximize, prior_births, reset flags), the valves,
    whether the noise is sampled and how, the sobbh/mbh/emri cadence, the PE
    declarations, and the move list in execution order. Read off
    ``fit.recipe`` itself, never re-derived from the knobs.
    """
    stages = list(fit.recipe.stages)
    armed = _source_ids_from_env()
    lines = [
        "==== RESOLVED RECIPE (run_combined_staged.build_fit; no data, no MPI) ====",
        f"branches: {list(fit.branches)}",
        f"source_types: {tuple(fit.general.source_types)}",
        "armed sources: " + (", ".join(f"{b}={ids}" for b, ids in armed.items())
                             if armed else "none"),
        f"{_envv('NUM_ITERATIONS')}  {_envv('NWALKERS', '16 (driver default)')}",
    ]
    for i, st in enumerate(stages, 1):
        kw = st.step_kwargs or {}
        lines.append(f"stage {i}/{len(stages)}  {st.name}  kind={st.kind}")
        lines.append(f"  stop:     {_describe_stop(st)}")
        prof = kw.get("profile")
        if prof is not None:
            lines.append("  profile:  " + " ".join(f"{k}={v}" for k, v in prof.items()))
        if st.kind == "gb_search":
            v = ("per-(walker, band) RJ valve DISARMED in this stage"
                 if kw.get("search_shutoff_per_walker") is False else
                 "per-(walker, band) RJ valve released at step entry (new step serial)")
            if (prof or {}).get("reset_valves"):
                v += ("; reset_valves: at a FRESH entry the per-band cold-lnL max is "
                      "re-learned and the per-band barren valve revived")
            if (prof or {}).get("reset_band_max"):
                v += "; reset_band_max: cold-lnL max re-learned at every entry"
            lines.append(f"  valves:   {v}")
        if st.kind in ("gb_search", "rj", "pe", "replica_pe"):
            lines.append(f"  noise:    {_describe_noise(st)}")
            lines.append(f"  sources:  {_describe_sources(st)}")
        elif st.kind == "search":
            # a joint max-lnL search stage: read what its search WRAPS
            inner = {n for m in st.moves for n in (getattr(m, "inner_names", ()) or ())}
            lines.append("  noise:    " + (
                "SAMPLED inside the stage's joint max-lnL search"
                if inner & {"psd_pe", "galfor_pe"} else
                "FIXED (no psd/galfor move: held at the start pin "
                "PSD_START_PARAMS/GALFOR_START_PARAMS or where the last noise stage left it)"))
            src = [n for n in _SOURCE_MOVES if n in inner]
            lines.append("  sources:  " + (
                f"{'/'.join(src)} inside the joint max-lnL search, EVERY round"
                if src else "none (no sobbh/mbh/emri move in this stage)"))
        if kw.get("ratchet") is not None:
            lines.append(f"  ratchet:  {kw['ratchet']}")
        if st.kind in ("pe", "replica_pe"):
            lines.append(
                f"  declares: peak_min_snr={kw.get('peak_min_snr')} "
                f"pe_repeats={kw.get('pe_repeats')} "
                f"pe_rj_flip_fraction={kw.get('pe_rj_flip_fraction')}")
            lines.append(f"  F-stat:   {_describe_pe_fstat(st)}")
        ck = st.combine_kwargs or {}
        lines.append(
            "  combine:  " + (" ".join(f"{k}={ck[k]}" for k in sorted(ck))
                              or "(GFCombineMove defaults)"))
        lines.append(f"  moves ({len(st.moves)}):")
        for j, m in enumerate(st.moves, 1):
            lines.append(f"      {j:2d}. {_describe_move(m)}")
    return "\n".join(lines)


def print_recipe() -> int:
    """``--print-recipe``: compose (no data, no MPI) and print the recipe."""
    if _env_flag("COMBINED_SMOKE"):
        _apply_smoke_defaults()
    # (one expression: tests/test_monitor_rank.py locates main()'s
    # build_fit assignment by its text, so this must not repeat it)
    print(describe_recipe(build_fit()), flush=True)
    return 0


def main() -> int:
    # Only configure logging if nothing else has: the global-fit framework
    # installs its own handler, and adding a second one duplicates EVERY
    # line (which is what the first full-band run did).
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    if _env_flag("COMBINED_SMOKE"):
        _apply_smoke_defaults()
        print("[combined] SMOKE mode: shrunk axes, GB_DEBUG + VGB_DEBUG on.",
              flush=True)

    # DEDICATED MONITOR RANK (GF_MONITOR_RANK=1, 2026-10-04). FIRST, before
    # anything else touches MPI: the highest world rank leaves the run and
    # builds the page + snapshot tar from disk, so the saver never stops
    # receiving for it. The run gets a communicator of the size it always
    # had (world order kept), so layout, placement and roles are unchanged.
    # Every run rank joins the closing barrier in __main__
    # (finish_run_ranks); the monitor rank leaves when it completes.
    from mpi4py import MPI
    from lisatools.globalfit.monitor import rank as _monitor_rank

    # HOST-MEMORY FLIGHT RECORDER on EVERY process of the run (head, compute
    # ranks, saver, monitor rank), started before the monitor split so the
    # monitor rank has it too: "[GF_MEM] w<rank> pid ... rss / children / node
    # avail" lines on change, the node's largest processes (+ shmem / slab
    # totals) every 10 min, and below 15 % available a table every 15 s plus a
    # faulthandler Python-stack dump (SIGWINCH) of the largest run process.
    # Why: the 9mo jobs 795 and 798 were both OOM-killed on node B after ~8 h
    # with nothing in the log naming the process that grew
    # (lisatools/globalfit/memwatch.py). GF_MEMWATCH=0 off.
    from lisatools.globalfit import memwatch as _memwatch

    _memwatch.start(f"w{MPI.COMM_WORLD.Get_rank()}")

    _run_comm, _mon = _monitor_rank.split_monitor_rank(MPI.COMM_WORLD)
    if _run_comm is None:
        print(f"[combined] rank {MPI.COMM_WORLD.Get_rank()} is the dedicated "
              f"MONITOR rank (GF_MONITOR_RANK=1): page + snapshot tar off the "
              f"saver.", flush=True)
        _monitor_rank.run_monitor_rank(MPI.COMM_WORLD)
        return 0
    if _mon is not None:
        print(f"[combined] GF_MONITOR_RANK=1: world rank {_mon} builds the "
              f"monitor; the run uses {_run_comm.Get_size()} ranks.",
              flush=True)

    fit = build_fit()
    print(f"[combined] branches: {list(fit.branches)}", flush=True)
    for st in fit.recipe.stages:
        print(f"[combined] stage {st.name:>13s} ({st.kind:>6s}): "
              f"{[m.name for m in st.moves]}", flush=True)
    print(f"[combined] source_types: {fit.general.source_types}", flush=True)

    if _env_flag("COMBINED_DRY_RUN"):
        print("[combined] COMBINED_DRY_RUN=1 -- composition only, not built.",
              flush=True)
        return 0

    print("[combined] fit.build() ...", flush=True)
    from lisatools.globalfit.communication.ranks import layout_dry_run, prepare_rank, RankRole

    # The RUN communicator from here on (== COMM_WORLD unless a monitor rank
    # was split off above).
    layout = prepare_rank(fit, _run_comm)
    # GF_LAYOUT_DRY_RUN=1: print every rank's placement and stop before the
    # build allocates anything (a bad layout still raises inside prepare_rank).
    if layout_dry_run(layout, _run_comm):
        print("[combined] GF_LAYOUT_DRY_RUN=1 -- layout only, not built.", flush=True)
        return 0
    fit.build()
    print("[combined] running", flush=True)
    fit.run(comm=_run_comm)
    # LOUD completion marker on stdout: "Residuals saved" goes to the
    # logger FILE only, so a finished run used to look exactly like a
    # silent death on the console.
    #
    # RANK-GUARDED (2026-08-15): under `mpiexec -n 3` (dedicated saver
    # rank) ONLY the main rank runs the MCMC -- run_global_fit gates it on
    # `rank == main_rank` and the spare/saver ranks return immediately.
    # Printed unconditionally, those ranks announced "RUN COMPLETE:
    # num_iterations=2000 reached" seconds after launch, while the real
    # run was still starting its first stage. Alarming and completely
    # false, so say which rank is talking and only claim completion from
    # the rank that actually sampled.
    _rank = _run_comm.Get_rank()
    role = layout.role_of(_rank)
    if role == RankRole.HEAD:
        if _env_flag("NULL_CHECK_ONLY"):
            # NULL_CHECK_ONLY stopped the run at the initial-lnL print
            # (run.py's null_check_only), so claiming "num_iterations
            # reached; residuals saved" would be flatly false.
            print(
                "[combined] NULL CHECK COMPLETE: initial lnL measured, no "
                "iterations run (NULL_CHECK_ONLY=1).",
                flush=True,
            )
        else:
            print(
                f"[combined] RUN COMPLETE: num_iterations="
                f"{fit.general.num_iterations} reached; residuals saved.",
                flush=True,
            )
    elif role == RankRole.COMPUTE:
        # under the walker-block layout a COMPUTE rank's fit.run() only
        # returns once the head has sent the fan-out STOP (review N-6) --
        # i.e. the run is over, not "still continuing on the head".
        print(
            f"[combined] rank {_rank} ({role.value}) released (head sent "
            f"STOP); run complete on the head.",
            flush=True,
        )
    elif role == RankRole.SAVER:
        # the saver's fit.run() returns only after it has handled the
        # head's {"finish_run": True} (review N-6).
        print(f"[combined] rank {_rank} ({role.value}) finished.", flush=True)
    else:  # legacy SPARE
        print(
            f"[combined] rank {_rank} ({role.value}) exiting; the run "
            f"continues on the head.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    # The resolved recipe and exit, BEFORE mpi4py is imported: no MPI, no data.
    if "--print-recipe" in sys.argv[1:] or _env_flag("GF_PRINT_RECIPE"):
        sys.exit(print_recipe())
    from mpi4py import MPI
    from lisatools.globalfit.communication.ranks import install_mpi_abort_on_error

    _comm = install_mpi_abort_on_error(MPI.COMM_WORLD)
    try:
        _rc = main()
        # Every RUN rank joins the closing barrier the dedicated monitor
        # rank waits on (a no-op without GF_MONITOR_RANK=1).
        if not _rc:
            from lisatools.globalfit.monitor import rank as _monitor_rank

            _monitor_rank.finish_run_ranks()
    except SystemExit:
        raise
    except BaseException:
        # The hook above handles the reporting + Abort; this only exists so
        # an exception escaping main() cannot fall through to a clean exit.
        sys.excepthook(*sys.exc_info())
        raise
    # A NONZERO return is also a failure: abort so no rank is left waiting.
    if _rc and _comm is not None:
        print(f"[MPI-ABORT] rank {_comm.Get_rank()} exiting with code {_rc}; "
              f"aborting all ranks.", file=sys.stderr, flush=True)
        _comm.Abort(_rc)
    sys.exit(_rc)
