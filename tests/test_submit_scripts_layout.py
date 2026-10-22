"""The 6mo campaign submit scripts size the rank count from the GPU count.

Both ``submit_gf_6mo_v8.sh`` (main) and ``submit_gf_6mo_v8_nogb_null.sh``
(null test) self-dispatch via ``exec sbatch ...`` before any heavy work
(environment activation, data preflight, ...) whenever ``SLURM_JOB_ID`` is
unset. That makes the pre-submit dispatch block cheap to exercise directly:
shadow ``sbatch`` with a stub that prints its argv and exits 0, run the
script with ``bash``, and inspect what the (never-actually-submitted) job
would have looked like.

Covers the layout invariants (user ruling 2026-09-16, after the WP7
transport gates passed on the cluster): at ``NGPUS=2`` the scripts pin the
walker-block layout (``GF_LEGACY_RANK_LAYOUT=0``: head + 1 compute + saver,
still ``--ntasks=3``); ``GF_LEGACY_RANK_LAYOUT=1`` remains the rollback knob
and must still launch today's single-compute-rank shape; ``NGPUS=4`` must
force the walker-block layout across 2 nodes of 2 GPUs each, since the
legacy layout cannot span nodes. Since V9-30 (2026-10-07) ``NGPUS=4`` is
the DEFAULT of the v9 production launchers (6mo / 1yr / 9mo); the v8 and
3mo scripts keep ``NGPUS=2``.
"""

import os
import re
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8.sh"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8_nogb_null.sh"),
    # the 3-month twin of the 6mo campaign script: same layout machinery,
    # Tobs-derived settings reverted, source branches and warm start off
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_3mo_v8_4gpu.sh"),
    # the v9 campaign script: same dispatch machinery as 6mo_v8, three
    # deliberate deltas (see SixMonthV9DeltaTest)
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v9_4gpu.sh"),
    # the 3-month twin OF THE V9 SCRIPT (2026-09-26): psd-only first stage,
    # inflated foreground reference, no warm start, 2 GPUs. See
    # ThreeMonthV9TwinTest for the declared delta list.
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_3mo_v9_2gpu.sh"),
    # the 1-year twin OF THE V9 SCRIPT (2026-10-03): byte-identical to the
    # 6mo v9 script except the Tobs deltas. See OneYearV9TwinTest.
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_1yr_v9.sh"),
    # the 9-month twin OF THE V9 SCRIPT (2026-10-07): the 1yr settings at
    # Tobs = 270 d. See NineMonthV9TwinTest.
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_9mo_v9.sh"),
]

THREE_MO = SCRIPTS[2]
SIX_MO = SCRIPTS[0]
SIX_MO_V9 = SCRIPTS[3]
THREE_MO_V9 = SCRIPTS[4]
ONE_YR_V9 = SCRIPTS[5]
NINE_MO_V9 = SCRIPTS[6]
# the v9 production launchers: the 6mo script and its two derived twins
V9_PRODUCTION = (SIX_MO_V9, ONE_YR_V9, NINE_MO_V9)

# Env knobs the dispatch block reads; stripped from the inherited environment
# before each scenario applies its own overrides, so a stray value in the
# test runner's shell can never leak into a scenario that expects it unset.
_DISPATCH_ENV_KEYS = (
    "SLURM_JOB_ID",
    "NGPUS",
    "NODES",
    "GPUS_PER_RANK",
    "RANKS_PER_GPU",
    "GF_LEGACY_RANK_LAYOUT",
    "GF_MONITOR_RANK",
    # 2026-10-15: the replica-routing knobs the v9 dispatch now reads
    "GF_GPU_ROUTING",
    "NWALKERS",
    "RANKS_PER_BLOCK",
)

# The v9 6mo/1yr/9mo launchers add ONE task for the dedicated monitor rank
# (GF_MONITOR_RANK, default 1, 2026-10-04): the run's own task count is
# unchanged, the allocation is one larger. tests/test_monitor_rank.py pins
# the block itself.
_MONITOR_RANK_SCRIPTS = V9_PRODUCTION


def _ntasks(script, run_tasks):
    return f"--ntasks={run_tasks + (1 if script in _MONITOR_RANK_SCRIPTS else 0)}"


#: The eigen jump-mixture exports of the v9 production launchers (2026-10-13):
#: {BRANCH}_EIGEN_JUMP_MIX 0.1 / {BRANCH}_EIGEN_JUMP_FACTOR 3.0 for the three
#: add/remove branches. Code defaults stay 0 / 3.0; the launchers turn it on.
EIGEN_JUMP_MIX_LINE = {
    f"{b}_EIGEN_JUMP_{k}": v
    for b in ("EMRI", "MBH", "SOBBH")
    for k, v in (("MIX", "0.1"), ("FACTOR", "3.0"))
}
EIGEN_JUMP_MIX_KNOBS = tuple(EIGEN_JUMP_MIX_LINE)
# 2026-10-14 (Mike, "some of the emris are sampling terribly"): per-axis
# step-scale adaptation of the source eigen tables ON in the launchers
# (code default off).
EIGEN_AXIS_ADAPT_LINE = {f"{b}_EIGEN_AXIS_ADAPT": "1" for b in ("EMRI", "MBH", "SOBBH")}
EIGEN_AXIS_ADAPT_KNOBS = tuple(EIGEN_AXIS_ADAPT_LINE)
# 2026-10-15 (Mike, "just do 1 for now" of the EMRI sampling list, then the
# widening cap after the 1yr tar 2): the adaptation buffer POOLED across
# ranks, the eigen step TEMPERED per rung (sigma / sqrt(beta_k)) and the
# adapted width capped at MAX_FACTOR x the Fisher sigma -- ON in the
# launchers (code defaults off / off / 100).
EIGEN_POOL_TEMPER_LINE = {
    **{f"{b}_EIGEN_ADAPT_POOL": "1" for b in ("EMRI", "MBH", "SOBBH")},
    **{f"{b}_EIGEN_TEMPER_SIGMAS": "1" for b in ("EMRI", "MBH", "SOBBH")},
    **{f"{b}_EIGEN_ADAPT_MAX_FACTOR": "100" for b in ("EMRI", "MBH", "SOBBH")},
}
EIGEN_POOL_TEMPER_KNOBS = tuple(EIGEN_POOL_TEMPER_LINE)
# 2026-10-18 (Mike, "do a fully audited and adjusted EMRI setup", and "we do
# not have enough walkers for the stretch move, so we need to improve what we
# have"; docs/emri-sampling-audit.md): the per-axis acceptance log (all three
# source branches; telemetry only), the acceptance-driven per-axis step
# controller and the cross-walker difference jump (EMRI only) -- ON in the
# launchers (code defaults off / off / 0).
EMRI_AUDIT_LINE = {
    **{f"{b}_EIGEN_AXIS_ACC_LOG": "1" for b in ("EMRI", "MBH", "SOBBH")},
    "EMRI_EIGEN_AXIS_ACC_TUNE": "1", "EMRI_EIGEN_AXIS_ACC_MIN": "0.1",
    "EMRI_EIGEN_AXIS_ACC_MAX": "0.5", "EMRI_EIGEN_AXIS_ACC_FACTOR": "2",
    "EMRI_DE_JUMP_MIX": "0.2", "EMRI_DE_JUMP_MODE_FRAC": "0.5",
    # 2026-10-18 (Mike, "do this for the other sources"): the controller on
    # for MBH / SOBBH (MBH's pooled adaptation had dropped its acceptance to
    # 0.07-0.21), the jump at 0.1 for MBH and 0.2 for the bimodal SOBBH leaves
    "MBH_EIGEN_AXIS_ACC_TUNE": "1", "SOBBH_EIGEN_AXIS_ACC_TUNE": "1",
    "MBH_DE_JUMP_MIX": "0.1", "SOBBH_DE_JUMP_MIX": "0.2",
}
EMRI_AUDIT_KNOBS = tuple(EMRI_AUDIT_LINE)
# 2026-10-19 (Mike, "when we do the sampling moves, can we move in the LISA
# reference frame ... Only the move that produces the new coordinates should be
# moved in the LISA frame"): the MBH inner moves propose in the LISA frame
# (code default "sampling").
PROPOSAL_FRAME_LINE = {"MBH_PROPOSAL_FRAME": "lisa"}
PROPOSAL_FRAME_KNOBS = tuple(PROPOSAL_FRAME_LINE)
#: what the launchers leave at the CODE defaults on the MBH / SOBBH lines (the
#: knobs exist): the controller band / factor and the jump's mode fraction
EMRI_AUDIT_NOT_EXPORTED = tuple(f"{b}_{k}" for b in ("MBH", "SOBBH") for k in (
    "EIGEN_AXIS_ACC_MIN", "EIGEN_AXIS_ACC_MAX", "EIGEN_AXIS_ACC_FACTOR",
    "DE_JUMP_MODE_FRAC"))
# 2026-10-13 (Mike): the GB warm-start SELF refit -- the saver rebuilds the
# mixture from the run's own last 50 rows x every cold walker every 5 rows in
# gb_search_2 and every 250 in full_pe; the warm moves draw 75 % of their
# proposes from it and 25 % from the parent-derived mixture. Code default OFF.
# 2026-10-14 ruling: in gb_search_2 refit every stored row and draw only from
# the refit (GB_WARM_SELF_EVERY_SEARCH 5 -> 1, GB_WARM_SELF_FRACTION_SEARCH 1.0).
# 2026-10-14 (later): gb_search_1 refits every row as well (code: the stage
# gate), and full_pe every 150 (GB_WARM_SELF_EVERY_PE 250 -> 150).
# 2026-10-15: built ON THE HEAD at the warm moves' entry (AT_ENTRY 1), PE every
# 25, three containers "original,full,filtered": search 0,0,1, PE 1,1,1, the
# consensus threshold 0.95; the two FRACTION knobs are gone.
# 2026-10-15 (later): search 0.2,0,0.8 ("20 % old run refit warmstart / 80%
# self refit warmstart filtered", so no leaf of the previous run is left out).
# 2026-10-16: search 0.5,0,0.5 ("Let's split them 0.5/0/0.5 in search").
WARM_SELF_LINE = {
    "GB_WARM_SELF": "1", "GB_WARM_SELF_AT_ENTRY": "1",
    "GB_WARM_SELF_EVERY_SEARCH": "1",
    "GB_WARM_SELF_EVERY_PE": "25", "GB_WARM_SELF_LAST_K": "50",
    "GB_WARM_SELF_SPLIT_SEARCH": "0.5,0,0.5", "GB_WARM_SELF_SPLIT_PE": "1,1,1",
    "GB_WARM_SELF_CONSENSUS": "0.95",
}
WARM_SELF_KNOBS = tuple(WARM_SELF_LINE)
# 2026-10-14 (Mike): the search-stage cold -> hot rotation of the source
# ladders, rotating by walker so no walker receives its own cold chain;
# first every 200 proposals (8 proposes of 25 repeats), then "every 5
# iterations during the search stages rather than 8" = 125. Code default
# 125; the launchers export it for visibility.
COLD_TO_HOT_LINE = {
    "MBH_COLD_TO_HOT_EVERY": "125", "EMRI_COLD_TO_HOT_EVERY": "125",
    "SOBBH_COLD_TO_HOT_EVERY": "125",
}
COLD_TO_HOT_KNOBS = tuple(COLD_TO_HOT_LINE)

# 2026-10-16 (Mike): gb_search_2 spends less time in the in-model steps, the
# checkers (group rule, per-row polish rule) kept and their windows shortened
# in that stage only; gb_search_1 as built (25 / 3 / 250). 2026-10-19 (Mike):
# fewer, fatter passes, 15 x 2 -> 30 x 1 (the same 30-repeat group test).
SEARCH2_INMODEL_LINE = {
    "GB_SEARCH_2_NUM_REPEAT_PROPOSALS": "30",
    "GB_SEARCH_2_INMODEL_GROUP_ITERS": "1",
    "GB_SEARCH_2_INMODEL_CONVERGE_ITERS": "100",
}
SEARCH2_INMODEL_KNOBS = tuple(SEARCH2_INMODEL_LINE)

# 2026-10-19 (Mike): the F-stat refit every second cycle in gb_search_2
# (stage profile fstat_refit_every; refit-clock ticks, 3 per cycle).
SEARCH2_FSTAT_REFIT_LINE = {"GB_SEARCH_2_FSTAT_REFIT_EVERY": "4"}
SEARCH2_FSTAT_REFIT_KNOBS = tuple(SEARCH2_FSTAT_REFIT_LINE)

# 2026-10-16 (Mike): the search F-stat epochs cycle through the cold walkers
# (code default rotate; min = the 2026-09-21 min-lnL rule), exported for
# visibility in every v9 production launcher.
FSTAT_SEARCH_REF_LINE = {"GB_FSTAT_SEARCH_REF": "rotate"}
FSTAT_SEARCH_REF_KNOBS = tuple(FSTAT_SEARCH_REF_LINE)

# 2026-10-17 (Mike): the RJ valve's all-walker rule -- a sub-band shuts
# only when every walker holding a source in it is converged together --
# exported for visibility (code default on; 0 = the per-walker valve).
SHUTOFF_ALL_WALKERS_LINE = {"GB_SEARCH_BAND_SHUTOFF_ALL_WALKERS": "1"}
SHUTOFF_ALL_WALKERS_KNOBS = tuple(SHUTOFF_ALL_WALKERS_LINE)

_STUB_SBATCH = """#!/usr/bin/env bash
for a in "$@"; do printf '%s\\n' "$a"; done
exit 0
"""


def _exports(path):
    """``{KNOB: value}`` as bash resolves the script's export lines in order.

    Resolving them rather than regexing the file is the point: the values
    are ``${K:-default}`` forms and several are overridden further down, so
    a grep answers what the file SAYS and this answers what the run GETS.
    """
    src = open(path).read()
    lines = [l for l in src.split("\n") if re.match(r"^export [A-Z0-9_]+=", l)]
    env = dict(os.environ)
    for k in list(env):
        if k.isupper():
            env.pop(k, None)
    out = subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv | sort\n"],
                         capture_output=True, text=True, env=env).stdout
    return dict(l.split("=", 1) for l in out.split("\n") if "=" in l)


class SixMonthV9DeltaTest(unittest.TestCase):
    """``submit_gf_6mo_v9_4gpu.sh`` is ``submit_gf_6mo_v8.sh`` plus exactly
    three families of change (user spec 2026-09-24).

    Same silent-failure shape as the 3-month twin below: the two files are
    ~99% the same text, and every block that MUST differ looks like an
    ordinary knob. A v9 that quietly kept v8's coarse surrogate, or ran with
    the convergence mode off, would produce entirely plausible output and
    waste the campaign -- so each delta is pinned here, and so is the
    v8 side of it, because a test that only checks v9 would still pass if
    someone "fixed" v8 to match.
    """

    def setUp(self):
        self.v8 = _exports(SIX_MO)
        self.v9 = _exports(SIX_MO_V9)

    # -- V9-2: noise likelihood back to exact-fine ----------------------
    def test_the_noise_model_is_normal_not_coarse(self):
        self.assertEqual(self.v8["COARSE_GPU_MODE"], "delayed_acceptance")
        self.assertEqual(self.v9["COARSE_GPU_MODE"], "off")

    def test_the_other_coarse_knobs_still_match_v8(self):
        """``COARSE_Q`` deliberately does NOT match v8 any more -- see
        :meth:`test_v9_runs_the_EXACT_FINE_noise_likelihood` and job 618.
        This test used to include it, on the same false premise the script
        comment carried ("they select nothing at mode=off"), which is how
        an illegal pair shipped. The two that really are inert stay pinned:
        dropping them would let the stamped noise identity and the
        preflight's "wanted" values disagree."""
        for k in ("COARSE_USE_WS", "COARSE_FIDUCIAL"):
            self.assertEqual(self.v9[k], self.v8[k], k)

    # -- V9-1: per-source in-model convergence --------------------------
    def test_the_convergence_mode_is_armed(self):
        self.assertNotIn("GB_INMODEL_CONVERGE", self.v8)
        self.assertIn(self.v9["GB_INMODEL_CONVERGE"], ("on", "observe"))

    def test_the_convergence_defaults_are_the_ruled_values(self):
        # 100 -> 250 (user, 2026-09-25, "just to be safe" for births).
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_ITERS"], "250")
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_DLL"], "4.0")
        # BOTH. "RJ moves [should] have their sources run to convergence
        # (in the search) whether they are birthed or survived" (user,
        # 2026-09-25). This is the per-ROW rule in the RJ pool; the pure
        # in-model slots converge per SUB-BAND via the GROUP rule and are
        # a separate scope, untouched by this knob.
        self.assertEqual(
            self.v9["GB_INMODEL_CONVERGE_CLASSES"], "newborn,mature")

    def test_the_row_ceiling_leaves_room_ABOVE_the_patience_floor(self):
        """⚠ The floor is W+1: a row cannot retire before the window is
        full. So the ceiling has to sit well above W or it, not the
        evidence, decides when a birth stops polishing.

        This is why raising ITERS alone is wrong: at W=250 the default
        ceiling (0 = 4x the 100-repeat class budget = 400) would leave a
        row just 149 repeats in which to plateau, against a measured p50 of
        260 at W=100. Pinned as a RATIO so the two cannot drift apart."""
        w = int(self.v9["GB_INMODEL_CONVERGE_ITERS"])
        ceil = int(self.v9["GB_INMODEL_CONVERGE_MAX"])
        self.assertGreater(ceil, 0, "0 resolves to 4x the CLASS budget, "
                                    "which does not track the window")
        self.assertGreaterEqual(
            ceil, 3 * w,
            f"ceiling {ceil} is too tight for a {w}-repeat window")

    def test_the_ladder_gate_is_on(self):
        """1.0 would make every rung vote, and the hot rungs never retire."""
        self.assertLess(
            float(self.v9["GB_INMODEL_CONVERGE_GATE_FRAC"]), 1.0)

    def test_the_refill_is_armed_and_can_actually_fire(self):
        """stop_frac 1.0 silently disables the refill: every column would
        finish together, nothing would carry, and the loop degenerates into
        the fixed chunk loop. Arming REFILL without lowering it is the
        no-op combination this pins against."""
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_REFILL"], "1")
        self.assertLess(
            float(self.v9["GB_INMODEL_CONVERGE_STOP_FRAC"]), 1.0)

    # -- V9-5 / V9-6: flip fraction + the in-GROUP convergence ----------
    def test_the_search_flip_fraction_is_one(self):
        self.assertEqual(self.v8["GB_SEARCH_RJ_FLIP_FRACTION"], "0.5")
        self.assertEqual(self.v9["GB_SEARCH_RJ_FLIP_FRACTION"], "1.0")
        # PE is untouched
        self.assertEqual(self.v9["GB_PE_RJ_FLIP_FRACTION"],
                         self.v8["GB_PE_RJ_FLIP_FRACTION"])

    def test_the_in_group_convergence_is_armed_and_distinct(self):
        """V9-6 is a DIFFERENT scope from V9-1 -- (walker, band) on a
        pass clock inside one proposal, vs row on a repeat clock. Pinned
        together so nobody "tidies" one family into the other."""
        self.assertEqual(self.v9["GB_INMODEL_GROUP"], "1")
        self.assertEqual(self.v9["GB_INMODEL_GROUP_SCALE"], "flat")
        # D/2 = 0.5 * GB_LEAF_CAP_NDIM (8.0), and it does NOT scale when
        # the window moves (user ruling 2026-09-26, declining my proposal
        # to derive it as 2.6667 when ITERS went 3 -> 2). It is a PHYSICAL
        # quantity -- the lnL a genuinely new D-parameter source has to
        # buy -- and deliberately the SAME number the leaf cap's
        # improvement gate uses, so the two features cannot disagree about
        # what counts as an improvement worth waiting for. Deriving it
        # from the window would break that tie.
        #
        # ⚠ This means shortening ITERS DOES loosen the per-repeat bar
        # (4.0/(3*25) = 0.0533 -> 4.0/(2*25) = 0.0800). That is chosen,
        # not overlooked: bands shut earlier, which is the point.
        self.assertEqual(self.v9["GB_INMODEL_GROUP_DLL"], "4.0")
        self.assertEqual(
            float(self.v9["GB_INMODEL_GROUP_DLL"]),
            0.5 * float(self.v9.get("GB_LEAF_CAP_NDIM", 8.0)),
            "GB_INMODEL_GROUP_DLL must stay D/2; see the comment above",
        )
        # the group window is in PASSES and must stay far smaller than the
        # row window, which is in repeats -- swapping them would make the
        # group rule unfireable (and the row rule fire on noise)
        self.assertLess(int(self.v9["GB_INMODEL_GROUP_ITERS"]),
                        int(self.v9["GB_INMODEL_CONVERGE_ITERS"]))
        # The window is a FLOOR: passes 1..W shut NOTHING because a band
        # needs W passes of history before it can be judged. Job 634
        # measured those warm-up passes at 62.6% of ALL pure in-model work
        # across the three slots (each runs its own group state, so the
        # warm-up is paid once PER SLOT). >= 2 keeps the test meaningful;
        # 1 would retire a band on a single pass with no history.
        self.assertGreaterEqual(int(self.v9["GB_INMODEL_GROUP_ITERS"]), 2)

    def test_the_group_has_a_hard_pass_ceiling(self):
        """Unbounded by construction: one pass is a full sweep of every
        source, so the ceiling is the only bound."""
        self.assertGreater(int(self.v9["GB_INMODEL_GROUP_MAX_PASSES"]), 0)

    # -- V9-7: permuted ("fancy") swaps off, vertical KEPT ---------------
    def test_fancy_tempering_is_off_but_vertical_survives(self):
        """⚠ Killing run_tempering must NOT take the vertical rung swaps
        with it. They are additive to the permuted swaps, not part of
        them, and they are the transport the in-model convergence depends
        on -- AND, since _adapt_band_temps is only ever called from
        run_tempering, they are now the only thing keeping the temperature
        ladder from freezing for the whole run."""
        self.assertEqual(self.v9["GB_RUN_FANCY_TEMPERING"], "0")
        self.assertEqual(self.v9["GB_TEMPER_VERTICAL"], "1")

    # -- V9-8: the standalone in-model move is actually BUILT -------------
    def test_the_pure_in_model_move_is_armed(self):
        """Without this the group-convergence knobs are exported, ignored,
        and nothing ever logs [GB_IMGROUP] -- the move is not built and not
        scheduled. This pair is what makes V9-6 reachable at all."""
        self.assertEqual(self.v9["GB_SEARCH_IN_MODEL"], "1")
        # Briefly 100 on 2026-09-25, reverted the same day: the group
        # window is 3 PASSES and a pass is this many repeats, so raising it
        # tightens the enforced rate 4x rather than coarsening granularity.
        self.assertEqual(self.v9["GB_NUM_REPEAT_PROPOSALS"], "25")
        self.assertNotIn("GB_SEARCH_IN_MODEL", self.v8)

    # -- V9-9: the stage-convergence valve --------------------------------
    def test_the_stage_shutoff_valve_is_on_and_the_band_schedule_is_not(self):
        """SNR limits move per recipe STAGE in v9, never per band, so the
        per-band stage schedule must stay out of the way. Independent
        flags; pinned together because arming both would have the two
        fighting over the same thresholds."""
        self.assertEqual(self.v9["GB_SEARCH_BAND_SHUTOFF_PER_WALKER"], "1")
        self.assertEqual(self.v9["GB_SEARCH_STAGE_PER_WALKER"], "0")

    # -- V9-10: refit cadence ---------------------------------------------
    def test_the_pe_refit_cadence_is_separate_and_longer(self):
        """full_pe is a random_choice stage; a search-tuned cadence there
        refits far more often in wall-clock terms than the number says.

        250 until 2026-10-02; user ruling that day: "during PE fstat should
        refit every 50 iterations" (the clock counts iterations and the
        decision runs when rj_fstat_pe is drawn, P = 0.45 per iteration).
        Back to 250 on 2026-10-07 (Mike: "make it default during pe to
        updating fstat every 250 iterations"). 150 on 2026-10-12 (Mike:
        "PE fstat fit should be based on a randomly chosen walker residual.
        And we should do a refit on that every 150 iterations"; the walker
        rule is the GB_FSTAT_PE_REF=random code default, not exported).
        """
        self.assertEqual(self.v9["GB_FSTAT_REFIT_EVERY_PE"], "150")
        # SEARCH stages refit 4x more often (user, 2026-09-25): the grid
        # goes stale as the search subtracts what it finds, so an old grid
        # aims births at peaks already claimed. PE is deliberately NOT
        # changed -- the two cadences are separate knobs for that reason.
        # 10 -> 1 (2026-09-25): refit every search iteration, because
        # the residual moves fast enough that the grid goes stale --
        # 628's rj_fstat_search cold yield fell 135 -> 96 -> 54.
        # 1 -> 2 (2026-09-28, Mike): REVERSED once the stage was late enough
        # that cold births had run out. On job 662 the refit cost 1129 s
        # of a 5564 s iteration (20%) and bought ~16 cold births, with
        # acceptance at 25/253k (rj_warm) and 80/236k (rj_fstat). A
        # stale grid cannot cost what it no longer buys. Revert to 1 if
        # cold birth acceptance climbs back.
        self.assertEqual(self.v9["GB_FSTAT_REFIT_EVERY"], "2")
        self.assertLess(int(self.v9["GB_FSTAT_REFIT_EVERY"]),
                        int(self.v9["GB_FSTAT_REFIT_EVERY_PE"]))

    # -- V9-4: cap cells off --------------------------------------------
    def test_the_leaf_cap_is_off_and_its_companions_with_it(self):
        """User ruling 2026-09-24: "no caps. Keep all those options
        available but turn them off."

        The cap is armed ONLY by a non-empty GB_LEAF_CAP_START, so empty is
        the off switch and every other knob stays present but inert. The
        companions are pinned separately because bash resolves exports in
        order -- an early `=0` followed by a later `=1` silently re-arms
        them, which is exactly the bug this test caught during authoring.
        """
        self.assertEqual(self.v9["GB_LEAF_CAP_START"], "")
        self.assertEqual(self.v9["GB_CAP_DRIFT_GATE"], "0")
        self.assertEqual(self.v9["GB_CAP_DRIFT_GATE_EDGE_LEAK"], "0")
        # the two stage-convergence vetoes the cap needed
        self.assertEqual(self.v9["GB_SEARCH_CAP_QUIESCENT"], "0")
        self.assertEqual(self.v9.get("GB_SEARCH_CAP_HEADROOM", "0"), "0")

    def test_the_cap_knobs_are_still_present_not_deleted(self):
        """"Keep all those options available" -- re-arming must be a
        one-line change, not an archaeology exercise."""
        src = open(SIX_MO_V9).read()
        for knob in ("GB_CAP_DIVISOR", "GB_LEAF_CAP_MIN_ITERS",
                     "GB_LEAF_CAP_REQUIRE_IMPROVEMENT"):
            self.assertIn(f"export {knob}=", src, knob)

    # -- V9-3: naming / a fresh store -----------------------------------
    def test_the_store_is_not_v8s(self):
        """The coarse mode is part of noise_model_identity, so a v8 store
        cannot be resumed under V9-2 -- run.py refuses it. The default
        STORE_DIR must therefore be a new path, not v8's."""
        src = open(SIX_MO_V9).read()
        self.assertIn("gf_prod_6mo_v9_4gpu/", src)
        self.assertNotIn(
            "STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v8/", src)

    def test_it_does_not_write_over_v8s_log(self):
        src = open(SIX_MO_V9).read()
        self.assertIn("--job-name=gf6mo_v9_4gpu", src)
        self.assertNotIn("gf6mo_v8_%j.log", src)

    # -- V9-11: the three-stage GB search --------------------------------
    def test_the_three_stage_search_is_armed(self):
        """One gb_search stage becomes gb_search_1/2/3. Without this the
        driver composes the single legacy stage and EVERY per-stage profile
        -- phase max, opt SNR, the F-stat peak floor -- silently never
        applies, while the script's own header claims all three."""
        self.assertEqual(self.v9["STAGE_V9_SEARCH"], "1")
        self.assertNotIn("STAGE_V9_SEARCH", self.v8)

    def test_the_stage1_values_are_what_the_moves_are_BUILT_with(self):
        """phase max / opt SNR / peak floor are per-stage PROFILE values
        now, applied at stage entry. These exports are what is in force
        BEFORE the first profile applies, so they must be stage 1's: if a
        profile ever failed to apply, the run would hold stage 1's
        configuration rather than silently run stage 1 as stage 2."""
        self.assertEqual(self.v9["GB_RJ_PHASE_MAXIMIZE"], "1")
        self.assertEqual(self.v9["GB_OPT_SNR_LIMIT_SEARCH"], "8.0")
        self.assertEqual(self.v9["FSTAT_PEAK_MIN_SNR"], "8.0")

    def test_stage_3_warm_start_rides_a_cadence(self):
        """User ruling 2026-09-24: warm start every 5th iteration in
        gb_search_3 only."""
        self.assertEqual(self.v9["GB_SEARCH_3_WARM_EVERY"], "5")

    def test_rj_replace_is_back_and_two_pass(self):
        """Reinstated 2026-09-24, then switched OFF again 2026-09-25 on
        yield: six consecutive zero-yield proposes in job 628, 8 swaps out
        of ~707k proposals and ZERO at the cold chain, while costing 19.4%
        of the cycle. The two-pass WIRING stays configured so reinstating
        it for gb_search_3 is a one-knob change."""
        self.assertEqual(self.v9["GB_SEARCH_RJ_REPLACE"], "0")
        self.assertEqual(self.v9["GB_REPLACE_WARM_PASS"], "1")
        self.assertEqual(self.v8["GB_SEARCH_RJ_REPLACE"], "0")
        # PE is explicitly NOT part of the reinstatement.
        self.assertEqual(self.v9["GB_PE_RJ_REPLACE"], "0")

    # -- V9-12: one seed store for the warm start AND the noise pin -------
    def test_the_warm_start_and_the_noise_pin_share_one_store(self):
        """User ruling 2026-09-24: "make sure the warmstart and PSD/GB
        frozen start point come from the same folder ... It should use
        maxlogL for PSD/GB." One variable, so overriding one without the
        other is not something a launch can do by accident."""
        src = open(SIX_MO_V9).read()
        self.assertIn("export GF_SEED_STORE=", src)
        self.assertIn("GB_WARM_START_SOURCE_STORE-${GF_SEED_STORE}", src)
        self.assertIn("lisatools.globalfit.warmstart.noise_pin", src)

    def test_the_seed_is_the_3mo_10_WALKER_run_that_actually_EXISTS(self):
        """REGRESSION, job 616 (2026-09-24 19:52). The default used to name
        a `_noisefix` sibling that does not exist on the cluster, and the
        first v9 launch died on it before reaching python -- the script's
        own candidate scan listed eight gf_prod_3mo* runs with no _noisefix
        among them. User ruling on the forensics, 2026-09-25: "yea it is
        just gf_prod_3mo_v8_10walkers".

        A path this test cannot verify (no filesystem here) is exactly the
        kind that has to be pinned by NAME once it has been confirmed by a
        launch, so a plausible-looking edit cannot silently reintroduce it.
        """
        self.assertEqual(
            self.v9["GF_SEED_STORE"],
            "/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/"
            "gf_prod_3mo_testing.h5",
        )
        self.assertNotIn("noisefix", self.v9["GF_SEED_STORE"])

    def test_the_seed_never_falls_back_to_a_DIFFERENT_run(self):
        """The resolver may pick the single .h5 inside a given directory,
        but it must never substitute another run: a seed is a provenance
        statement, and quietly swapping it is the exact "warm start and
        noise pin from different folders" split the ruling forbids.

        Note this is about the RESOLVER, not the default. Changing which
        store the default names is a user ruling; substituting one at run
        time, after the operator named a different one, is the defect.
        """
        src = open(SIX_MO_V9).read()
        self.assertIn("does NOT fall back to a", src)
        # The only substitution allowed is directory -> the ONE .h5 in it.
        self.assertIn('_n_h5=$(find "${_seed_dir}" -maxdepth 1 -name \'*.h5\'',
                      src)
        self.assertIn('if [ "${_n_h5}" = "1" ]; then', src)

    def test_the_shipped_COARSE_pair_PASSES_the_real_validator(self):
        """REGRESSION, job 618 (2026-09-25): all 5 ranks aborted at build
        with ``coarse_Q > 1 in an all-source run requires an explicit
        COARSE_GPU_MODE``.

        The script shipped COARSE_Q=8 with COARSE_GPU_MODE=off on the
        written-down belief that "with mode=off they select nothing". They
        do not: ``run.py`` reads ``coarse_Q`` on its own and builds the
        coarse basis for any Q > 1 without consulting the mode.

        Asserting the two VALUES would only re-state the fix. This runs the
        actual function that rejected the job, so the pair is checked by
        the rule itself and a future edit to either knob is caught here
        rather than on the cluster.
        """
        from lisatools.globalfit.stock.erebor.noise import (
            validate_coarse_settings)

        class _GS:
            pass

        gs = _GS()
        gs.coarse_Q = int(self.v9["COARSE_Q"])
        gs.coarse_gpu_mode = self.v9["COARSE_GPU_MODE"]
        gs.coarse_fiducial = self.v9["COARSE_FIDUCIAL"]
        gs.gpus = [0, 1]
        validate_coarse_settings(gs, all_source=True)   # must not raise

    def test_v9_runs_the_EXACT_FINE_noise_likelihood(self):
        """V9-2 ruling, restated 2026-09-25: "We want the general psd
        computations for galfor and psd. Not the coarse version." Q=1 IS
        that configuration -- there is nothing to coarsen."""
        self.assertEqual(self.v9["COARSE_Q"], "1")
        self.assertEqual(self.v9["COARSE_GPU_MODE"], "off")
        # ...and v8, which this file is rebased on, ran the OTHER legal
        # pair. Pinned so the delta stays deliberate and visible.
        self.assertEqual(self.v8["COARSE_Q"], "8")
        self.assertEqual(self.v8["COARSE_GPU_MODE"], "delayed_acceptance")

    def test_the_noise_pin_output_is_PARSED_never_evald(self):
        """REGRESSION, job 617 (2026-09-25):

            slurm_script: eval: line 3572: syntax error near unexpected
            token `('

        The CLI is clean -- three diagnostics to STDERR, two
        ``export NAME=<digits>`` lines to stdout, verified against a real
        store both bare and under this script's full export environment.
        Something else in the cluster process wrote to stdout (a backend
        banner is the likely culprit; it cannot appear on a laptop with no
        CUDA, which is why local testing could not have caught it).

        ``eval`` of another process's stdout is the defect whoever the
        polluter is: ANY stray line becomes shell. The values must be
        parsed out with an anchored, number-only pattern instead.
        """
        src = open(SIX_MO_V9).read()
        self.assertNotIn('eval "${_pin}"', src)
        self.assertIn("NEVER `eval` this output", src)
        for var in ("PSD", "GALFOR"):
            self.assertIn(
                rf"sed -n 's/^export {var}_START_PARAMS="
                rf"\([-+0-9.eE,]*\)$/\1/p'", src)

    def test_the_pin_satisfies_each_variable_INDEPENDENTLY(self):
        """User ruling 2026-09-25: "the galfor pin should come from our
        estimate. The PSD estimate can come from the 3mo run." Since
        GALFOR_START_PARAMS ships hand-set, requiring BOTH lines to parse
        would discard a good instrument-noise pin over a galfor value this
        run was never going to use."""
        src = open(SIX_MO_V9).read()
        self.assertIn(
            '{ [ -n "${_psd_was}" ] || [ -n "${_psd_pin}" ]; }', src)
        self.assertIn(
            '{ [ -n "${_gal_was}" ] || [ -n "${_gal_pin}" ]; }', src)
        # ...and galfor still ships hand-set, so the store never supplies it.
        self.assertTrue(self.v9["GALFOR_START_PARAMS"])

    def test_unexpected_pin_stdout_is_reported_not_silently_dropped(self):
        """Dropping the junk fixes the crash but loses the evidence. The
        next run has to NAME the polluter, or this comment stays a guess."""
        src = open(SIX_MO_V9).read()
        self.assertIn("unexpected stdout", src)
        self.assertIn("_pin_junk", src)

    def test_a_missing_seed_says_it_will_ALSO_kill_the_warm_start_gate(self):
        """Job 616's log was hard to read because one cause tripped two
        gates that never referenced each other: the [V9-SEED] miss printed
        a SOFT message ("otherwise the noise stages will fit it"), then the
        [WARMSTART] gate hard-exited 10 lines later on the same inherited
        path with no traceback. The seed gate must forward-declare the
        fatality, and the fatal gate must name where its path came from."""
        src = open(SIX_MO_V9).read()
        self.assertIn("AND THIS WILL BE FATAL BELOW", src)
        self.assertIn("there is one problem here, not two", src)
        self.assertIn("that path came from GF_SEED_STORE", src)

    def test_mid_iteration_checkpoints_are_pinned_off(self):
        """User ask 2026-09-24: "checkpoints between all the GB moves for
        resume"; user ruling 2026-10-07 (V9-30): the production line's
        MIDIT_CHECKPOINT=0 is the launcher default. The HOOKS are
        unconditional (GFCombineMove writes after every sub-move); what this
        pins is that the feature is DISARMED by default and that its throttle
        -- the thing that decides how much a spot preemption costs once it is
        re-armed -- stays on the run record rather than a silent default."""
        self.assertEqual(self.v9["MIDIT_CHECKPOINT"], "0")
        self.assertIn("MIDIT_CHECKPOINT_MIN_INTERVAL", self.v9)
        self.assertGreaterEqual(
            int(self.v9["MIDIT_CHECKPOINT_MIN_INTERVAL"]), 0)

    def test_mid_iteration_checkpoints_can_be_switched_from_the_command_line(self):
        """User ruling 2026-10-01: "MIDIT_CHECKPOINT=0 in the launch command
        is good" (a legged search lands a row after every leg, and the
        checkpoint's adoption rule cannot tell a startup copy from a
        half-nudged state -- jobs 673/674/675). The export was a hard
        ``=1`` until then, so the command-line knob reached NOTHING; pin the
        ``${K:-default}`` form in both launchers. The 6mo default became 0
        on 2026-10-07 (V9-30); the 3mo v9 script keeps 1."""
        for path, default in ((SIX_MO_V9, "0"), (THREE_MO_V9, "1")):
            src = open(path).read()
            self.assertRegex(
                src, rf"(?m)^export MIDIT_CHECKPOINT=\$\{{MIDIT_CHECKPOINT:-{default}\}}$",
                f"{path}: MIDIT_CHECKPOINT must be exported as "
                f"${{MIDIT_CHECKPOINT:-{default}}} so the launch command can flip it")

    def test_the_galfor_start_is_the_offline_3mo_estimate(self):
        """User ruling 2026-09-24, chosen explicitly over the 6mo
        alternative. PHYSICAL/LINEAR, amp MODULATION-CORRECTED (the 90 d
        window sits on a dim stretch, <M_XX> = 0.808, so the correction
        RAISES amp -- using the raw number understates by 19%).

        Pinned as numbers against the file they came from, because the
        vector is only meaningful as a whole: two of its five parameters
        are unidentified, and cherry-picking one would silently change the
        curve.
        """
        import json
        import os as _os

        raw = self.v9["GALFOR_START_PARAMS"]
        got = [float(x) for x in raw.split(",")]
        self.assertEqual(len(got), 5)
        src = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "galfor_6mo_figs", "addback_fixedpoint_90d.json")
        if not _os.path.exists(src):
            self.skipTest("addback_fixedpoint_90d.json is untracked scratch")
        fp = json.load(open(src))["history"][-1]["prior_box_refit"]
        want = [fp["amp_modulation_corrected"], fp["params"]["fk"],
                fp["params"]["alpha"], fp["params"]["f_1"],
                fp["params"]["f_2"]]
        for g, w, n in zip(got, want, ("amp", "fk", "alpha", "f_1", "f_2")):
            self.assertAlmostEqual(g / w, 1.0, places=9, msg=n)

    def test_the_galfor_start_is_inside_the_run_prior(self):
        """⚠ The one check that matters: outside the support, every walker
        and every rung prices at log_prior = -inf on iteration 0. alpha 5.0
        is INTERIOR only because GALFOR_ALPHA_MAX=20.0 is also exported --
        the two knobs are load-bearing together."""
        from lisatools.globalfit.stock.erebor.noise import (
            GALFOR_BASIS, GALFOR_PRIOR_RANGE)

        got = [float(x) for x in self.v9["GALFOR_START_PARAMS"].split(",")]
        box = [list(map(float, r)) for r in GALFOR_PRIOR_RANGE]
        box[GALFOR_BASIS.index("alpha")][1] = float(
            self.v9["GALFOR_ALPHA_MAX"])
        for n, v, (lo, hi) in zip(GALFOR_BASIS, got, box):
            self.assertTrue(lo <= v <= hi, f"{n}={v:g} outside [{lo:g},{hi:g}]")

    def test_a_hand_set_galfor_start_survives_the_seed_pin(self):
        """The script hardcodes GALFOR_START_PARAMS and ALSO reads a pin
        from GF_SEED_STORE. The pin block must not clobber it -- that is
        what the per-variable guard is for, and it is the whole reason an
        offline estimate can be supplied at all."""
        src = open(SIX_MO_V9).read()
        self.assertIn("_gal_was=${GALFOR_START_PARAMS+set}", src)
        self.assertIn("was set by hand -- keeping it, not the store's", src)
        self.assertIn("${GALFOR_START_PARAMS:-", src)

    def test_the_noise_pin_is_read_at_MAXLOGL(self):
        """noise_pin reads best_logl_noise, i.e. the best-logL cold walker
        of the last valid row -- not a posterior mean."""
        src = open(SIX_MO_V9).read()
        self.assertIn("maxlogL", src)

    def test_everything_else_is_still_v8(self):
        """The whole point of a copy-with-deltas: any OTHER knob that
        drifted is either an unrecorded change or a bad merge."""
        allowed = {
            # V9-2. COARSE_Q joined this list on 2026-09-25: it is not an
            # extra change but the OTHER HALF of the same one -- mode=off
            # with Q=8 is the illegal pair that aborted job 618.
            "COARSE_GPU_MODE", "COARSE_Q",
            "STORE_DIR", "FILE_STORE_DIR",          # V9-3
            "GB_LEAF_CAP_START", "GB_CAP_DRIFT_GATE",
            "GB_CAP_DRIFT_GATE_EDGE_LEAK", "GB_SEARCH_CAP_QUIESCENT",
            "GB_SEARCH_RJ_FLIP_FRACTION",           # V9-5
            "GB_RUN_FANCY_TEMPERING",               # V9-7
            "GB_SEARCH_IN_MODEL",                   # V9-8
            "GB_NUM_REPEAT_PROPOSALS",              # V9-8
            # source in-model repeats 25 in every stage, "same as pe" (user
            # ruling 2026-10-03); v8 ran MBH 2 / EMRI 2 / SOBBH 10
            "MBH_NUM_PROP_REPEATS", "EMRI_NUM_PROP_REPEATS",
            "SOBBH_NUM_PROP_REPEATS",
            "GB_SEARCH_BAND_SHUTOFF_PER_WALKER",    # V9-9
            "GB_SEARCH_BAND_SHUTOFF_CONV_ITER",     # V9-9
            "GB_SEARCH_STAGE_PER_WALKER",           # V9-9
            "GB_FSTAT_REFIT_EVERY_PE",              # V9-10
            "GB_FSTAT_REFIT_EVERY",                 # V9-14 (40 -> 10)
            # V9-13 (2026-09-25): the RJ schedule itself. The rigid path
            # ran ALL RJ rounds before any polish; the staged scheduler
            # interleaves RJ round -> in-model to convergence -> refill.
            "GB_RJ_DIRECT_BATCH",
            # V9-11, the three-stage search restructure
            "STAGE_V9_SEARCH", "GB_SEARCH_3_WARM_EVERY",
            "GB_SEARCH_RJ_REPLACE", "GB_REPLACE_WARM_PASS",
            "GB_RJ_PHASE_MAXIMIZE", "GB_OPT_SNR_LIMIT_SEARCH",
            "FSTAT_PEAK_MIN_SNR",
            # V9-12, the shared seed store + noise pin
            "GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE",
            # mid-iteration checkpoints, pinned explicitly for v9
            "MIDIT_CHECKPOINT", "MIDIT_CHECKPOINT_MIN_INTERVAL",
            # V9-30 (2026-10-07): Mike's production line IS the launcher
            # default -- 8-rung MBH / EMRI ladders (v8 ran 2), the expected-
            # information eigen tables (gram / fisher; v8 predates the knob)
            # and the source cross-checks off (v8 predates the knob).
            "MBH_NTEMPS", "EMRI_NTEMPS",
            "SOBBH_EIGEN_INFO", "EMRI_EIGEN_INFO", "MBH_EIGEN_INFO",
            "PSD_EIGEN_INFO", "GALFOR_EIGEN_INFO",
            "SOBBH_CHECK_LL", "MBH_CHECK_LL", "EMRI_CHECK_LL",
            # 2026-10-07: stage clocks survive a resume, reset at each
            # boundary (v8 predates the knob)
            "GF_PERSIST_STAGE_START",
            # 2026-10-07: the GB ladder re-spacing (hot half at 1.35 from
            # T10, top rung unpinned); fresh stores only, v8 predates it
            "GB_LADDER_HOT_RATIO", "GB_LADDER_HOT_FROM", "GB_LADDER_PIN_LAST",
            # 2026-10-08: MBH / EMRI eigen tables refresh every 10 visits
            # (Gram from the fast templates; v8 refreshed its ll tables at 100)
            "MBH_EIGEN_REFRESH", "EMRI_EIGEN_REFRESH",
            # the offline 3mo galfor start point
            "GALFOR_START_PARAMS",
            # the galfor RATCHET (2026-09-30): one gated noise proposal at
            # the head of gb_search_3 -- nudge / hold / release -- v9 only,
            # default OFF. See lisatools.globalfit.noise_ratchet.
            "GALFOR_RATCHET", "GALFOR_RATCHET_HOLD", "GALFOR_RATCHET_RELEASE",
            "GALFOR_RATCHET_CYCLES", "GALFOR_RATCHET_DLOG10_AMP",
            "GALFOR_RATCHET_DLOG10_FK", "GALFOR_RATCHET_DLOG10_F2",
            # the ratchet's production plan (2026-10-02, jobs 685/695): release-
            # first two-step cycles, release to convergence (50 flat rounds / 5
            # nats / cap 5000), the min-gain stop (200) with a nudge floor (2),
            # the relaunch clock, and the valve floor below 7 mHz. Until 10-02
            # these lived on the launch line only; the launcher now carries
            # them so a line that drops one cannot silently change the run.
            "GALFOR_RATCHET_RELEASE_FIRST", "GALFOR_RATCHET_RELEASE_CHECKS",
            "GALFOR_RATCHET_RELEASE_TOL", "GALFOR_RATCHET_RELEASE_ITERS",
            "GALFOR_RATCHET_MIN_GAIN", "GALFOR_RATCHET_MIN_NUDGES",
            # 2026-10-03: the stop decides on the released FOREGROUND (median
            # walker's drop over a band), not the raw lnL gain.
            "GALFOR_RATCHET_STOP_RULE", "GALFOR_RATCHET_MIN_DROP", "GALFOR_RATCHET_DROP_BAND",
            # 2026-10-03: gb_search_3 ends at the ratchet stop; replica_pe runs
            # full_pe's composition (sources on the gb_search cadence) until the
            # large-scale metrics converge; then full_pe = sample taking.
            "GALFOR_RATCHET_END_STAGE_ON_STOP", "STAGE_REPLICA_PE", "REPLICA_PE_WINDOW",
            "REPLICA_PE_LEAF_TOL", "REPLICA_PE_LNL_TOL",
            "GALFOR_RATCHET_CLOCK_RESET", "GALFOR_RATCHET_CLOCK_START",
            "GALFOR_RATCHET_SHUTOFF_MIN_FREQ",
            # full_pe declarations (user rulings 2026-10-02): in-model repeats
            # 25 on every branch but emri ("a general thinning factor of 25"),
            # the GB RJ draw at 0.45 prior / 0.45 fstat / 0.1 warm, the PE flip
            # fraction 0.1, the PE refit every 50 iterations.
            "PE_INMODEL_REPEATS", "PE_INMODEL_REPEATS_BRANCHES",
            "GB_PE_RJ_FSTAT_FRACTION", "GB_PE_RJ_WARM_FRACTION",
            "GB_PE_RJ_FLIP_FRACTION",
            # the two PE F-stat moves (user ruling 2026-10-12): the GB-free
            # galaxy grid fitted once at SNR 6.5, the live-residual grid every
            # GB_FSTAT_REFIT_EVERY_PE, splitting the F-stat draw share. v8
            # predates all three; PeTwoFstatMovesLauncherTest pins them.
            "GB_FSTAT_REFIT_EVERY_PE_GB_FREE", "GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE",
            "GB_PE_RJ_FSTAT_RESIDUAL_SPLIT",
            # match-criterion panels back on for the v9 pages (2026-09-30)
            "GF_MONITOR_MATCH_STATS",
            # search legs: one row per in-model leg (2026-09-30)
            "GB_SEARCH_LEGS",
            # f0-adaptive stage-B sky grid (the F-stat fix, 2026-09-24)
            "FSTAT_STAGEB_SKY_ADAPT", "FSTAT_STAGEB_NSKY_MIN",
            "FSTAT_STAGEB_NSKY_MAX", "FSTAT_STAGEB_GROUP_MAX_GB",
            # V9-15 (2026-09-25, off job 628): sig-het reference refresh
            # 25 -> 50. It was the largest single line item inside the
            # in-model polish (167-186 s per RJ move). Trades reference
            # staleness for wall time; [GB_TRUST] and the end-of-block
            # "ll AUDIT vs exact" COLD median are the instruments.
            "GB_SIGHET_REFRESH_EVERY",
            # V9-16 (2026-09-26): gb_search_seed -- a new FIRST search
            # stage, warm-start RJ + in-model only, fixed 5 iterations,
            # before any F-stat grid is fitted. Named rather than
            # renumbered, so gb_search_1/2/3 are untouched.
            "GB_SEARCH_SEED_ITERS",
            # V9-17 (2026-09-26): in_model_replace retires with
            # rj_replace -- a third pure in-model pass over rows the
            # other two slots already polished, 14% of an iteration.
            "GB_SEARCH_IN_MODEL_REPLACE",
            # V9-18 (2026-09-26): pruning back ON. v8 ran 0 (prior
            # births AND deaths, inherited from a 2026-09-21 hand
            # edit); v9 returns to the code default 1 = deaths only,
            # because with rj_replace off there was no dedicated
            # pruning move left anywhere in the cycle.
            "GB_SEARCH_PRIOR_REMOVAL_ONLY",
            # V9-19 (2026-09-26): MOJITO_INFO_PATH -- the single mojito
            # path the monitor page needs (the level holding catalogues/
            # and data/). Read ONLY by the page generator, never by the
            # sampler, so it changes no physics and no timing. v8 predates
            # it entirely, hence the drift; the 3mo v9 twin now carries it
            # too, so this is the only list that still needs the entry.
            "MOJITO_INFO_PATH",
            # V9-20 (2026-09-26): GB_TEMPER_ALL_RUNGS -- the vertical swap
            # now covers EVERY rung of a (walker, band) column, including
            # rungs holding an unpicked source or none at all, so a cell
            # whose whole model is net-harmful can be evicted upward. v8
            # predates the feature entirely. Inactive rungs are priced from
            # a per-block constant and never made resident, so buffer
            # residency is unchanged.
            "GB_TEMPER_ALL_RUNGS",
            # 2026-10-04: GB_VERT_EXACT_PRICE -- price the vertical swap
            # with exact add-deltas instead of the sig-het running value
            # (default 0 until the copy-store comparison; v8 predates it).
            "GB_VERT_EXACT_PRICE",
            # V9-21 (2026-09-28): GB_TEMPER_VERTICAL_AT_REFIT -- the
            # vertical sweep now runs ONLY on repeats that re-anchored
            # the sig-het references, so the swap and the refit land in
            # the same repeat (paired with GB_SIGHET_REFRESH_EVERY
            # 50 -> 25, already on this list). Order within the repeat is
            # unchanged: sample -> refit -> swap. v8 predates both the
            # knob and the all-rungs sweep it gates.
            "GB_TEMPER_VERTICAL_AT_REFIT",
            # V9-22 (2026-09-28, iteration-time package). The level-3
            # valve bar becomes ONE FLOOR-SNR SOURCE (0.5 * 8^2 = 32)
            # instead of the level-1/2 per-row dll of 4.0, and the
            # second pure in-model pass is dropped -- in_model and
            # in_model_fstat are the same class and cost 47% of a job
            # 662 iteration between them. v8 has neither knob.
            # V9-23 (2026-09-28): GB_INMODEL_OBSERVABLE_JUMP 2.0 -> 1.5.
            # Under GB_INMODEL_OBSERVABLE_EIGEN=axis one eigen axis is
            # drawn per repeat, so an axis's accepted/proposed motion
            # ratio is its OWN acceptance: job 663 had f_mid at ~7% and
            # fdot 3-7% against a pooled 0.55-0.79, and the 09-19
            # doubling made accepted f motion SMALLER. 1.5 is an
            # interim compromise -- one global knob cannot suit both
            # halves -- pending the per-axis scaling on the
            # gb-inmodel-axis-adapt branch.
            # V9-24 (2026-09-28): GB_SEARCH_INMODEL_AFTER_REMOVAL --
            # a pure in-model pass after rj_prior_removal, the only RJ
            # move that had no in-model partner, so the SAVED state was
            # the post-death state with no group-rule pass over it. v8
            # predates the slot entirely.
            "GB_SEARCH_INMODEL_AFTER_REMOVAL",
            "GB_INMODEL_OBSERVABLE_JUMP",
            "GB_SEARCH_BAND_SHUTOFF_LL_TOL",
            # GB_SEARCH_DROP_MOVES is deliberately NOT listed: Mike
            # rejected dropping in_model_fstat (2026-09-28) and the
            # export was removed, so a future re-appearance should be
            # caught by this test rather than pre-authorised.
            # --- backfill (2026-09-28) -------------------------------
            # These shipped earlier in the 09-26/09-27 gate rework and
            # their entries were never added, so this test had been RED
            # for reasons unrelated to whatever change was being made.
            # A permanently-failing drift test guards nothing: it trains
            # you to ignore it, which is the opposite of the point.
            #
            # Level-3/4 valve retuning: the per-band RJ shutoff clock and
            # its revival window, shortened once the valve state began
            # persisting so a band can actually reach the threshold
            # within one job.
            "GB_RJ_BAND_SHUTOFF_ITERS",
            "GB_RJ_BAND_SHUTOFF_RESET_ITERS",
            # sobbh/mbh/emri cadence in the SEARCH stages, 10 -> 5.
            "GB_SEARCH_SOURCE_EVERY",
            # The containerised monitor, run on the saver rank after a
            # save. v8 had no in-run monitor at all, so every one of
            # these is new rather than changed.
            "GF_MONITOR_AFTER_SAVE", "GF_MONITOR_ITER",
            "GF_MONITOR_SNAPSHOT", "GF_MONITOR_TIMEOUT",
            # 2026-10-04: the page + tar on a DEDICATED rank (one extra
            # task); tests/test_monitor_rank.py pins the block.
            "GF_MONITOR_RANK",
            # 2026-10-05: host-cached in-model proposal factors + the lookup
            # Gram info matrix + the lookup-built sig-het reference / anchor.
            "GB_CHOL_CACHE", "GB_CHOL_CACHE_EVERY", "GB_CHOL_CACHE_TOL",
            "SIGHET_REF_BUILD", "SIGHET_ANCHOR_ENGINE", "SIGHET_INFOMAT_ENGINE",
            # V9-27 (2026-10-02): the SOBBH lookup scoring path's knobs (default
            # chunked = the v8 path); SixMonthSOBBHLookupTest pins the block.
            "SOBBH_LIKELIHOOD", "SOBBH_LOOKUP_TABLE_PATH", "SOBBH_LOOKUP_EVAL_DT",
            "SOBBH_LOOKUP_ROW_BATCH", "SOBBH_LOOKUP_KERNEL",
            # V9-25 (2026-09-30): mbh_pe on the batched, windowed MBH
            # likelihood at B=8, response order 8 pinned. v8 predates the
            # knobs entirely; SixMonthMBHBatchedTest pins the block.
            "MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
            # V9-28 (2026-10-02, MBH session): the batched window generated at
            # q*dt; default 1 = what job 695 ran. The MBH suites pin the block.
            "MBH_WINDOW_DECIMATE",
            # 2026-10-12: the LISA-frame sky-mode hops back in the MBH inner-move
            # stack (SkyMove in the ICRS basis; 0.02 both / 0.04 lat / 0.02 long
            # of the per-leaf draws, user ruling). New knobs, same code defaults.
            "MBH_SKY_MOVE_BOTH_WEIGHT", "MBH_SKY_MOVE_LAT_WEIGHT",
            "MBH_SKY_MOVE_LONG_WEIGHT",
            # 2026-10-13: the eigen jump mixture ON in the launchers (Mike: "if
            # this is going to improve sampling in general then lets try it");
            # code default mix 0. EigenJumpMixLauncherTest pins the block.
            *EIGEN_JUMP_MIX_KNOBS,
            # 2026-10-14: per-axis step-scale adaptation of the source eigen
            # tables ON (code default off); EigenJumpMixLauncherTest pins it.
            *EIGEN_AXIS_ADAPT_KNOBS,
            # 2026-10-15: pooled / tempered / factor-capped adaptation, same block
            *EIGEN_POOL_TEMPER_KNOBS,
            # 2026-10-18: the EMRI sampling audit (per-axis acceptance log,
            # the acceptance controller, the cross-walker jump); code defaults
            # off / off / 0. EigenJumpMixLauncherTest pins the block.
            *EMRI_AUDIT_KNOBS,
            # 2026-10-19: the MBH proposal frame (code default "sampling");
            # EigenJumpMixLauncherTest pins it.
            *PROPOSAL_FRAME_KNOBS,
            # 2026-10-13: the ladder caps (first ruling: EMRI T_max 100; second
            # ruling the same day: main-mode tempering -- EMRI T_max 6, MBH
            # T_max 10, the MBH lost-rung guard off, the EMRI guard at 50 nats);
            # code defaults None / 200.
            "EMRI_LADDER_TMAX", "EMRI_LADDER_REANCHOR_NATS",
            "MBH_LADDER_TMAX", "MBH_LADDER_REANCHOR_NATS",
            "SOBBH_LADDER_TMAX", "SOBBH_LADDER_REANCHOR_NATS",
            # 2026-10-14: the search-stage cold -> hot rotation cadence of the
            # source ladders (code default 200, exported for visibility).
            *COLD_TO_HOT_KNOBS,
            # 2026-10-13: the GB warm-start self-refit block (code default
            # OFF); WarmSelfLauncherTest pins it.
            *WARM_SELF_KNOBS,
            # 2026-10-16: the gb_search_2 in-model budget (stage profile
            # inmodel_budget; unset = as built).
            *SEARCH2_INMODEL_KNOBS,
            # 2026-10-19: the gb_search_2 F-stat refit cadence
            *SEARCH2_FSTAT_REFIT_KNOBS,
            # 2026-10-16: the search F-stat epochs rotate through the walkers
            *FSTAT_SEARCH_REF_KNOBS,
            # 2026-10-17: the valve's all-walker rule
            *SHUTOFF_ALL_WALKERS_KNOBS,
            # V9-26 (2026-10-02): the EMRI direct-to-WDM scoring path's knobs
            # (default full = the v8 path); SixMonthEMRIDirectTest pins the block.
            "EMRI_LIKELIHOOD", "EMRI_BATCH_MAX_SIZE", "EMRI_DIRECT_TABLE",
            "EMRI_DIRECT_RESPONSE", "EMRI_TRAJ_WORKERS", "EMRI_DIRECT_LOOKUP",
        }
        drift = {
            k: (self.v8.get(k), self.v9.get(k))
            for k in set(self.v8) | set(self.v9)
            if self.v8.get(k) != self.v9.get(k)
            and not k.startswith("GB_INMODEL_CONVERGE")
            and not k.startswith("GB_INMODEL_GROUP")
            and k not in allowed
        }
        self.assertEqual(drift, {}, f"undeclared v8 -> v9 drift: {drift}")


def _mbh_preflight_source(path):
    """The python body of the launcher's MBH preflight heredoc."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("# MBH PREFLIGHT."))
    head = next(i for i in range(start, len(lines))
                if lines[i] == "python - <<'PYEOF' || exit 2")
    end = next(i for i in range(head + 1, len(lines)) if lines[i] == "PYEOF")
    return "\n".join(lines[head + 1:end])


class SixMonthMBHBatchedTest(unittest.TestCase):
    """V9-25 (2026-09-30): mbh_pe runs the batched, windowed MBH likelihood.

    ``MBH_LIKELIHOOD=batched`` makes ``build_mbh_move_runtime`` build
    ``MBHBatchedLikeMove`` on every compute rank. B=8 is the controller's
    ruling off the H100 benchmark on the production 6-month grid (B=8 +8.5 GB
    at 0.037 s/row vs B=16 +18.5 GB at 0.034). An unknown env var is silently
    ignored, so the launcher preflights the knobs through the real settings
    class and the real consistency rule; these tests run that preflight.
    """

    def setUp(self):
        self.v9 = _exports(SIX_MO_V9)
        self.three = _exports(THREE_MO_V9)

    def test_mbh_pe_runs_batched_at_b8_with_order_8(self):
        self.assertEqual(self.v9["MBH_LIKELIHOOD"], "batched")
        self.assertEqual(self.v9["MBH_BATCH_MAX_SIZE"], "8")
        self.assertEqual(self.v9["MBH_RESPONSE_ORDER"], "8")

    def test_every_mbh_knob_is_overridable_from_the_launch_line(self):
        """``MBH_LIKELIHOOD=full`` is the escape and ``MBH_RESPONSE_ORDER=30``
        restores the old order: a hard ``export K=v`` would make both reach
        nothing (the MIDIT_CHECKPOINT lesson)."""
        src = open(SIX_MO_V9).read()
        for knob, default in (("MBH_LIKELIHOOD", "batched"),
                              ("MBH_BATCH_MAX_SIZE", "8"),
                              ("MBH_RESPONSE_ORDER", "8"),
                              ("MBH_WINDOW_DECIMATE", "2")):   # 2 since V9-30 (2026-10-07)
            self.assertRegex(
                src, rf"(?m)^export {knob}=\$\{{{knob}:-{default}\}}$", knob)

    def test_nothing_in_the_launcher_conflicts_with_batched(self):
        """Batched pins BOTH generators to the 90 d window and refuses an
        explicit MBH_WAVEFORM_DURATION or USE_TDIONFLY; the cross-check
        cadence stays at the batched move's own default (every 10th visit),
        and the check itself is OFF by default since V9-30 (2026-10-07:
        MBH_CHECK_LL=0 on the production line), re-armed from the line."""
        for knob in ("MBH_WAVEFORM_DURATION", "USE_TDIONFLY",
                     "MBH_CHECK_LL_EVERY"):
            self.assertNotIn(knob, self.v9, knob)
        self.assertEqual(self.v9["MBH_CHECK_LL"], "0")
        self.assertRegex(open(SIX_MO_V9).read(),
                         r"(?m)^export MBH_CHECK_LL=\$\{MBH_CHECK_LL:-0\}$")

    def _run_preflight(self, **env):
        """Exec the launcher's MBH preflight in-process: (exit code, stdout)."""
        import contextlib
        import io
        from unittest import mock

        # import BEFORE patching the env: erebor's module-level defaults
        # snapshot env-backed fields at import time and must not capture the
        # patched values for the rest of this process
        from lisatools.globalfit.stock.erebor import source_runtime  # noqa: F401

        code = compile(_mbh_preflight_source(SIX_MO_V9), "mbh_preflight", "exec")
        base = {k: self.v9[k] for k in
                ("MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
                 "MBH_WINDOW_DECIMATE")}
        base.update(env)
        out = io.StringIO()
        with mock.patch.dict(os.environ, base):
            for k in ("MBH_WAVEFORM_DURATION", "USE_TDIONFLY"):
                if k not in env:
                    os.environ.pop(k, None)
            with contextlib.redirect_stdout(out):
                try:
                    exec(code, {"__name__": "__main__"})
                except SystemExit as exc:
                    return exc.code, out.getvalue()
        return 0, out.getvalue()

    def test_the_shipped_block_passes_its_own_preflight(self):
        rc, out = self._run_preflight()
        self.assertEqual(rc, 0, out)
        self.assertIn("mbh_pe scoring=batched batch<=8 response_order=8", out)
        self.assertIn("waveform_duration=90.0 d", out)
        self.assertIn("decimate=2", out)   # the V9-30 default (2026-10-07)

    def test_the_preflight_resolves_and_bounds_the_window_decimation(self):
        """MBH_WINDOW_DECIMATE on the launch line reaches the settings (printed);
        a factor that does not divide Nf=1440 is refused before the allocation."""
        rc, out = self._run_preflight(MBH_WINDOW_DECIMATE="2")
        self.assertEqual(rc, 0, out)
        self.assertIn("decimate=2", out)
        rc, out = self._run_preflight(MBH_WINDOW_DECIMATE="7")
        self.assertEqual(rc, 2, out)
        self.assertIn("does not divide Nf=1440", out)

    def test_the_preflight_refuses_what_the_build_would_refuse(self):
        for env in ({"MBH_WAVEFORM_DURATION": "2592000"},
                    {"USE_TDIONFLY": "1"}):
            rc, out = self._run_preflight(**env)
            self.assertEqual(rc, 2, f"{env}: {out}")
            self.assertIn("[MBH-PREFLIGHT] REFUSING", out)

    def test_the_preflight_refuses_an_install_without_the_batched_move(self):
        import sys
        from unittest import mock

        # everything imported up front: patch.dict(sys.modules) drops any
        # module first imported inside it when it restores the dict
        from lisatools.globalfit.stock.erebor import source_runtime  # noqa: F401

        with mock.patch.dict(sys.modules,
                             {"lisatools.globalfit.moves.mbhbatchedmove": None}):
            rc, out = self._run_preflight()
        self.assertEqual(rc, 2, out)
        self.assertIn("SILENTLY IGNORED", out)

    def test_the_preflight_refuses_settings_that_ignore_the_env(self):
        """A settings class that does not read MBH_LIKELIHOOD (an install
        whose field is named differently) resolves to its own default."""
        import functools
        from unittest import mock

        from lisatools.globalfit.stock.erebor import source_runtime as sr

        stale = functools.partial(sr.SourceMBHSettings, likelihood="full")
        with mock.patch.object(sr, "SourceMBHSettings", stale):
            rc, out = self._run_preflight()
        self.assertEqual(rc, 2, out)
        self.assertIn("but the settings resolve", out)

    def test_the_3mo_twin_does_not_inherit_it(self):
        """The 3-month twin does not export the 6-month MBH block, and it does
        not need to: since the batched path became the library default
        (MBH_LIKELIHOOD=auto, 2026-10-01) a window longer than the 90-day grid
        is CLAMPED to the data's active box instead of raising, so the twin's
        ``MBHB_IDS=5`` escape builds on the batched path too. The 6-month
        grid fits the full window with room to spare."""
        import types

        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        for knob in ("MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER"):
            self.assertNotIn(knob, self.three, knob)
        layer_dt = 1440 * 2.5  # Nf x dt on both grids
        win = (90 * 86400.0, 10 * 86400.0, 4 * 86400.0, 86400.0)

        def grid(tobs, crop):
            nt = int(round(float(tobs) / layer_dt))
            return types.SimpleNamespace(layer_dt=layer_dt, t0=0.0, Nt=nt,
                                         ind_min_t=crop, ind_max_t=nt - crop - 1)

        six = grid(self.v9["TOBS_TARGET"], int(self.v9["EDGE_CROP_WAVELETS"]))
        geom = mbh_window_layers(six, 100 * 86400.0, *win)
        self.assertLessEqual(geom["Nt_keep"] + 2 * geom["n_pad"], six.Nt)
        three = grid(self.three["TOBS_TARGET"], 20)
        clamped = mbh_window_layers(three, 60 * 86400.0, *win)
        # kept box clamped inside the active box; segment inside the grid
        self.assertGreaterEqual(clamped["n_start"], three.ind_min_t)
        self.assertLessEqual(clamped["n_start"] + clamped["Nt_keep"], three.ind_max_t + 1)
        self.assertLess(clamped["Nt_keep"], geom["Nt_keep"])


def _emri_preflight_source(path):
    """The python body of the launcher's EMRI preflight heredoc."""
    with open(path) as fh:
        lines = fh.read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("# EMRI PREFLIGHT."))
    head = next(i for i in range(start, len(lines))
                if lines[i] == "python - <<'PYEOF' || exit 2")
    end = next(i for i in range(head + 1, len(lines)) if lines[i] == "PYEOF")
    return "\n".join(lines[head + 1:end])


class SixMonthEMRIDirectTest(unittest.TestCase):
    """V9-26 (2026-10-02) / V9-27 (2026-10-03): emri_pe scores through the direct-to-WDM
    template BY DEFAULT.

    ``EMRI_LIKELIHOOD=direct`` (the launcher default since the H100 speed tests: 25 ms
    single, 15 ms/row batched vs production 88 at 6 months) makes
    ``build_emri_move_runtime`` build ``EMRIDirectLikeMove`` on every compute rank and the
    engine's EMRI template generator the direct template too; ``EMRI_LIKELIHOOD=full`` on
    the launch line restores the per-row production path for both."""

    _KNOBS = ("EMRI_LIKELIHOOD", "EMRI_BATCH_MAX_SIZE", "EMRI_DIRECT_TABLE",
              "EMRI_DIRECT_RESPONSE", "EMRI_TRAJ_WORKERS", "EMRI_DIRECT_LOOKUP")

    def setUp(self):
        self.v9 = _exports(SIX_MO_V9)

    def test_defaults(self):
        self.assertEqual(self.v9["EMRI_LIKELIHOOD"], "direct")
        self.assertEqual(self.v9["EMRI_BATCH_MAX_SIZE"], "8")
        self.assertEqual(self.v9["EMRI_DIRECT_RESPONSE"], "dense")
        self.assertEqual(self.v9["EMRI_TRAJ_WORKERS"], "8")   # 8 since V9-30 (2026-10-07)
        self.assertEqual(self.v9["EMRI_DIRECT_LOOKUP"], "kernel")
        # the preflight proves the fused kernel on the node's GPU (a module without it
        # would silently fall back to the Python lookup)
        self.assertIn('SyntheticEdgesTest("test_gpu_equals_cpu")', open(SIX_MO_V9).read())
        # unset: the canonical table in the run folder, found or built by the preflight
        self.assertEqual(self.v9["EMRI_DIRECT_TABLE"], "")
        self.assertIn("export FILE_STORE_DIR=${STORE_DIR}", open(SIX_MO_V9).read())
        # the grid the table must match, and the edge crop that must clear its edges
        self.assertEqual(self.v9["TOBS_TARGET"], "15552000")
        self.assertEqual(self.v9["EDGE_CROP_WAVELETS"], "60")

    def test_every_emri_knob_is_overridable_from_the_launch_line(self):
        src = open(SIX_MO_V9).read()
        for knob in self._KNOBS:
            self.assertRegex(src, rf"(?m)^export {knob}=\$\{{{knob}:-", knob)

    def test_dispatch_defaults_match_the_exports(self):
        """The self-dispatch sizes --cpus-per-task BEFORE the job sees the EMRI
        block, so it repeats two defaults; they must be the block's."""
        src = open(SIX_MO_V9).read()
        disp = src[: src.index('exec sbatch --partition="${_NGPU_PART}"')]
        self.assertIn('"${EMRI_LIKELIHOOD:-%s}" = "direct"' % self.v9["EMRI_LIKELIHOOD"], disp)
        self.assertIn("${EMRI_TRAJ_WORKERS:-%s}" % self.v9["EMRI_TRAJ_WORKERS"], disp)

    def _dispatch(self, **env):
        with tempfile.TemporaryDirectory() as stub_dir:
            stub = os.path.join(stub_dir, "sbatch")
            with open(stub, "w") as fh:
                fh.write(_STUB_SBATCH)
            os.chmod(stub, 0o755)
            run_env = {k: v for k, v in os.environ.items()
                       if k not in _DISPATCH_ENV_KEYS and k not in self._KNOBS}
            run_env["PATH"] = stub_dir + os.pathsep + run_env.get("PATH", "")
            run_env.update(NGPUS="4", **env)
            res = subprocess.run(["bash", SIX_MO_V9], env=run_env, capture_output=True,
                                 text=True, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)
        return res.stdout.splitlines()

    def test_direct_reserves_a_core_per_trajectory_worker(self):
        # the extra cores feed ONLY the trajectory pool: ranks stay single-threaded
        # (MPI-only policy), and the pool's spawned workers inherit OMP_NUM_THREADS=1
        self.assertEqual(self.v9["OMP_NUM_THREADS"], "1")
        # direct with 8 workers (V9-30, 2026-10-07) = 2 + 8 cores
        self.assertIn("--cpus-per-task=10", self._dispatch())                 # direct: the default
        self.assertIn("--cpus-per-task=2", self._dispatch(EMRI_LIKELIHOOD="full"))
        self.assertIn("--cpus-per-task=10", self._dispatch(EMRI_LIKELIHOOD="direct"))
        self.assertIn("--cpus-per-task=9",
                      self._dispatch(EMRI_LIKELIHOOD="direct", EMRI_TRAJ_WORKERS="7"))
        self.assertIn("--cpus-per-task=2",
                      self._dispatch(EMRI_LIKELIHOOD="direct", EMRI_TRAJ_WORKERS="0"))

    def _run_preflight(self, table=("found",), parity=("ok",), kparity=("ok",), **env):
        """Exec the launcher's EMRI preflight in-process: (exit code, stdout).

        ``table`` stubs ``wdm_lookup_store.ensure_lookup_table``: ("found"|"built",) or
        ("raise", message); the calls land in ``self.ensured``. ``parity`` stubs the dense
        TDI kernel's GPU parity run, ``kparity`` the fused lookup kernel's (``self.kruns``
        counts those)."""
        import contextlib
        import io
        from unittest import mock

        from lisatools.globalfit.stock.erebor import source_runtime  # noqa: F401

        code = compile(_emri_preflight_source(SIX_MO_V9), "emri_preflight", "exec")
        base = {k: self.v9[k] for k in self._KNOBS + ("EDGE_CROP_WAVELETS",)}
        base["FILE_STORE_DIR"] = "/the/run/folder"
        base.update(env)
        self.ensured = []
        self.kruns = 0

        def fake_ensure(path, **kw):
            self.ensured.append((path, kw))
            if table[0] == "raise":
                raise ValueError(table[1])
            return table[0]

        def fake_run(runner, suite):
            res = unittest.TestResult()
            res.testsRun = 1
            lookup = any(type(t).__name__ == "SyntheticEdgesTest" for t in suite)
            self.kruns += lookup
            outcome = (kparity if lookup else parity)[0]
            if outcome == "fail":
                res.failures.append((None, "boom"))
            elif outcome == "skip":
                res.skipped.append((None, "no GPU backend"))
            return res

        patches = [mock.patch.dict(os.environ, base),
                   mock.patch("unittest.TextTestRunner.run", fake_run),
                   mock.patch("lisatools.wdm_lookup_store.ensure_lookup_table", fake_ensure)]
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            with contextlib.redirect_stdout(out):
                try:
                    exec(code, {"__name__": "__main__"})
                except SystemExit as exc:
                    return exc.code, out.getvalue()
        return 0, out.getvalue()

    def test_the_shipped_block_passes_its_own_preflight(self):
        rc, out = self._run_preflight()
        self.assertEqual(rc, 0, out)
        self.assertIn("emri_pe scoring=direct", out)
        rc, out = self._run_preflight(EMRI_LIKELIHOOD="full")
        self.assertEqual(rc, 0, out)
        self.assertIn("emri_pe scoring=full", out)

    def test_direct_passes_with_a_matching_table_and_a_passing_gpu_parity(self):
        rc, out = self._run_preflight(EMRI_LIKELIHOOD="direct")
        self.assertEqual(rc, 0, out)
        self.assertIn("emri_pe scoring=direct batch<=8 response=dense lookup=kernel traj_workers=8", out)
        self.assertEqual(self.kruns, 1)            # the fused kernel was proven on the GPU
        self.assertIn("edge_crop=60>=8", out)
        # unset EMRI_DIRECT_TABLE: the canonical table in the run folder, on this grid
        name = "/the/run/folder/wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5"
        self.assertEqual(self.ensured[0][0], name)
        self.assertEqual((self.ensured[0][1]["Nf"], self.ensured[0][1]["dt"]), (1440, 2.5))
        self.assertIn(f"lookup table {name}: found", out)

    def test_a_missing_table_is_built_by_the_preflight_at_the_pointer(self):
        rc, out = self._run_preflight(table=("built",), EMRI_LIKELIHOOD="direct",
                                      EMRI_DIRECT_TABLE="/shared/tables/mine.h5")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.ensured[0][0], "/shared/tables/mine.h5")
        self.assertIn("lookup table /shared/tables/mine.h5: built", out)

    def test_full_never_touches_the_table(self):
        rc, out = self._run_preflight(EMRI_LIKELIHOOD="full")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.ensured, [])

    def test_direct_refusals(self):
        cases = [
            ({}, dict(table=("raise", "built for Nf=180, dt=20.0")), "built for Nf=180"),
            ({}, dict(parity=("fail",)), "parity test"),
            ({}, dict(parity=("skip",)), "skipped 1"),
            (dict(EDGE_CROP_WAVELETS="4"), {}, "EDGE_CROP_WAVELETS=4"),
            (dict(EMRI_DIRECT_RESPONSE="bogus"), {}, "EMRI_DIRECT_RESPONSE"),
            ({}, dict(kparity=("fail",)), "fused lookup kernel"),
            ({}, dict(kparity=("skip",)), "EMRI_DIRECT_LOOKUP=python"),
            (dict(EMRI_DIRECT_LOOKUP="bogus"), {}, "EMRI_DIRECT_LOOKUP='bogus'"),
        ]
        for env, kw, needle in cases:
            with self.subTest(needle=needle):
                rc, out = self._run_preflight(EMRI_LIKELIHOOD="direct", **kw, **env)
                self.assertEqual(rc, 2, out)
                self.assertIn("[EMRI-PREFLIGHT] REFUSING", out)
                self.assertIn(needle, out)

    def test_python_lookup_skips_the_kernel_parity_run(self):
        rc, out = self._run_preflight(kparity=("fail",), EMRI_LIKELIHOOD="direct",
                                      EMRI_DIRECT_LOOKUP="python")
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.kruns, 0)
        self.assertIn("lookup=python", out)

    def test_spline_response_skips_the_gpu_parity_run(self):
        rc, out = self._run_preflight(parity=("fail",), EMRI_LIKELIHOOD="direct",
                                      EMRI_DIRECT_RESPONSE="spline")
        self.assertEqual(rc, 0, out)

    def test_the_preflight_refuses_an_install_without_the_direct_move(self):
        import sys
        from unittest import mock

        from lisatools.globalfit.stock.erebor import source_runtime  # noqa: F401

        with mock.patch.dict(sys.modules, {"lisatools.globalfit.moves.emridirectmove": None}):
            rc, out = self._run_preflight(EMRI_LIKELIHOOD="direct")
            self.assertEqual(rc, 2, out)
            self.assertIn("SILENTLY IGNORED", out)
            # full is the old install's own path: no refusal
            rc, out = self._run_preflight(EMRI_LIKELIHOOD="full")
            self.assertEqual(rc, 0, out)
            self.assertIn("predates EMRI_LIKELIHOOD", out)

    def test_the_preflight_refuses_settings_that_ignore_the_env(self):
        import functools
        from unittest import mock

        from lisatools.globalfit.stock.erebor import source_runtime as sr

        stale = functools.partial(sr.SourceEMRISettings, likelihood="full")
        with mock.patch.object(sr, "SourceEMRISettings", stale):
            rc, out = self._run_preflight(EMRI_LIKELIHOOD="direct")
        self.assertEqual(rc, 2, out)
        self.assertIn("but the settings resolve", out)


class MpiPlacementTest(unittest.TestCase):
    """⚠ EVERY script that launches `-ppn 1` must also export
    I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0, or the placement silently reverts
    to SLURM's per-node block split.

    This is a LAUNCH BLOCKER, not a tuning knob. At 5 tasks over 2 nodes --
    the v9 4-GPU shape, 4 compute + 1 saver -- SLURM's 3/2 split puts three
    compute ranks on a 2-GPU node and build_layout refuses outright. At 3
    tasks the two placements coincide, which is exactly why it went unnoticed
    until a 5-task launch, and why it is pinned here rather than left to the
    next person to rediscover.

    Found 2026-09-24: the fix lived on an UNMERGED branch, so neither v9 nor
    v8-on-dev carried it.
    """

    def test_every_ppn_launcher_sets_placement_respect_off(self):
        import glob

        checked = 0
        for path in glob.glob(os.path.join(
                ROOT, "scripts", "fstat_proposal", "submit_gf_*.sh")):
            src = open(path).read()
            if "-ppn 1" not in src:
                continue
            checked += 1
            self.assertIn(
                "export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", src,
                f"{os.path.basename(path)} launches `-ppn 1` without the "
                f"placement export; a 5-task 2-node run would be refused by "
                f"build_layout")
        self.assertGreater(checked, 0, "no -ppn launcher found to check")

    def test_the_v9_launcher_is_one_of_them(self):
        src = open(SIX_MO_V9).read()
        self.assertIn("-ppn 1", src)
        self.assertIn("export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", src)


class V9RankLayoutTest(unittest.TestCase):
    """What (NGPUS, NWALKERS) actually resolve to, run against the SCRIPT's
    own arithmetic rather than a reimplementation of it.

    Written 2026-09-24 while the user was testing the NWALKERS/NGPU shape.
    The hazard is not a crash: a NWALKERS that does not divide N_COMPUTE is
    ROUNDED UP rather than refused, and the store then LOCKS to the rounded
    value for its whole life (a resume refuses any change). So testing at one
    value and launching at another silently produces a different run.
    """

    #: The two formulas this pins, quoted from the script so a drift in
    #: either place fails here rather than at launch.
    N_COMPUTE_SRC = ("N_COMPUTE_EFF=$(( ${SLURM_NNODES:-1} * _NGPUS_EFF "
                     "* RANKS_PER_GPU / _k ))")
    ROUND_SRC = "export NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF ))"

    def setUp(self):
        self.src = open(SIX_MO_V9).read()

    def test_the_formulas_are_still_the_ones_modelled_here(self):
        """If either line changes, the expectations below are stale and must
        be re-derived -- fail loudly instead of asserting against fiction."""
        self.assertIn(self.N_COMPUTE_SRC, self.src)
        self.assertIn(self.ROUND_SRC, self.src)

    @staticmethod
    def _resolve(ngpus, nwalkers, ranks_per_gpu=1, gpus_per_rank=1):
        nodes, gpus_per_node = (2, 2) if ngpus == 4 else (1, 2)
        n_compute = nodes * gpus_per_node * ranks_per_gpu // gpus_per_rank
        if n_compute > 1 and nwalkers == 1:
            return n_compute, 1, "replica"
        if n_compute > 0 and nwalkers % n_compute:
            nwalkers = (nwalkers // n_compute + 1) * n_compute
            return n_compute, nwalkers, "rounded"
        return n_compute, nwalkers, "exact"

    def test_the_intended_4gpu_shape_needs_no_rounding(self):
        """NGPUS=4, NWALKERS=4 -> N_COMPUTE 4, one walker per rank, exact."""
        n_compute, nw, how = self._resolve(4, int(self.v9_nwalkers()))
        self.assertEqual((n_compute, nw, how), (4, 4, "exact"))

    def v9_nwalkers(self):
        return _exports(SIX_MO_V9)["NWALKERS"]

    def test_the_script_ships_the_shape_that_divides(self):
        self.assertEqual(int(self.v9_nwalkers()), 4)

    def test_non_multiples_are_ROUNDED_not_refused(self):
        """⚠ The quiet one. Only a [SUBMIT] line marks it, and the store then
        locks to the rounded number."""
        for asked, got in ((2, 4), (3, 4), (5, 8), (6, 8), (10, 12)):
            n_compute, nw, how = self._resolve(4, asked)
            self.assertEqual((nw, how), (got, "rounded"),
                             f"NGPUS=4 NWALKERS={asked}")

    def test_one_walker_is_replica_mode_not_a_rounding(self):
        self.assertEqual(self._resolve(4, 1)[2], "replica")
        self.assertEqual(self._resolve(2, 1)[2], "replica")

    def test_ranks_per_gpu_2_gives_EIGHT_compute_ranks_not_five(self):
        """The script's old comment claimed 'NGPUS=4 needs 5 compute ranks
        with RANKS_PER_GPU=2'. It does not -- that gives 8, and 5 is not
        reachable from 4 GPUs at all. Pinned so the corrected comment cannot
        quietly revert."""
        self.assertEqual(self._resolve(4, 8, ranks_per_gpu=2)[0], 8)
        # The script still QUOTES the old claim in order to correct it, so
        # test for the correction rather than the absence of the string.
        self.assertIn("that is wrong", self.src)
        self.assertIn("RANKS_PER_GPU=2 gives N_COMPUTE=8, not 5", self.src)


class SixMonthSOBBHLookupTest(unittest.TestCase):
    """The 6mo v9 SOBBH lookup block: ``SOBBH_LIKELIHOOD=lookup`` (the default) puts the lookup
    comp inside the existing SOBBH add/remove move (docs/sobbh-wdm-lookup.md); chunked = the v8
    path."""

    def setUp(self):
        self.v9 = _exports(SIX_MO_V9)
        self.text = open(SIX_MO_V9).read()

    def test_the_block_defaults(self):
        self.assertEqual(self.v9["SOBBH_LIKELIHOOD"], "lookup")  # the default (2026-10-03)
        self.assertEqual(self.v9["SOBBH_LOOKUP_TABLE_PATH"], "")   # the run folder's table
        self.assertEqual(self.v9["SOBBH_LOOKUP_EVAL_DT"], "43200")  # the sparse 12-h response
        self.assertEqual(self.v9["SOBBH_LOOKUP_ROW_BATCH"], "32")
        # kernel, not auto (SOBBH session, 2026-10-03): a build without the
        # fused kernel is REFUSED in the preflight instead of silently
        # scoring through the Python lookup.
        self.assertEqual(self.v9["SOBBH_LOOKUP_KERNEL"], "kernel")

    def test_the_preflight_guards(self):
        for needle in (
            "resolve_sobbh_lookup_table",          # find or build the table before mpiexec
            "SOBBH_LOOKUP_KERNEL=kernel but",      # a missing compiled kernel is refused
            "the settings resolve",                 # a silently ignored knob is refused
        ):
            self.assertIn(needle, self.text)


class ThreeMonthTwinTest(unittest.TestCase):
    """``submit_gf_3mo_v8_4gpu.sh`` is the 6mo script at 3 months.

    Written 2026-09-18 to the spec "use all the updates and base it on the
    6mo run, but make sure the high level 3mo things are there (Tobs, no
    emris/sobhbs/mbhbs)" plus "(no warmstart)".

    The failure this guards is a silent one. The two files are 97% the
    same text, so the obvious way to carry a 6mo fix across is to copy the
    block -- and the blocks that must NOT be copied are exactly the ones
    that look like ordinary knobs (``TOBS_TARGET``, ``GB_N_SUBBANDS``). A
    3-month run that quietly analysed 6 months of data, or armed the
    source branches, would produce plausible output and waste the
    campaign.
    """

    def setUp(self):
        self.three = _exports(THREE_MO)
        self.six = _exports(SIX_MO)

    def test_tobs_and_its_derived_settings_are_the_3mo_values(self):
        self.assertEqual(self.three["TOBS_TARGET"], "7776000")
        self.assertEqual(self.three["GB_NLEAVES_MAX"], "10000")
        self.assertEqual(self.three["GB_N_SUBBANDS"], "32768")
        self.assertEqual(self.three["GB_RJ_INMODEL_CHUNK"], "65536")
        self.assertEqual(self.three["COARSE_Q"], "1")
        self.assertEqual(self.three["COARSE_GPU_MODE"], "off")
        self.assertEqual(self.three["BASE_FILE_NAME"], "gf_prod_3mo")
        # 6mo-only; the 3mo arm takes the defaults
        self.assertNotIn("SIGHET_NT_LAYER", self.three)
        self.assertNotIn("EDGE_CROP_WAVELETS", self.three)

    def test_source_id_vars_are_UNSET_not_set_empty(self):
        """Set-empty looks equivalent and breaks the import (2026-09-18).

        Two readers disagree about what "empty" means:

        * ``source_runtime.default_source_ids`` seeds the SETTINGS and
          tests ``os.environ.get(f"{cls}_IDS") is not None`` -- so a
          set-but-empty var REPLACES the class default
          ``{"MBHB": [], "EMRI": [1], "SOBHB": []}`` with three empty
          lists;
        * ``run_combined_staged._source_ids_from_env`` ARMS the branches
          and reads ``os.environ.get(env, "")`` -- unset and set-empty are
          identical to it.

        With all three set-empty, importing
        ``lisatools.globalfit.stock.erebor`` FAILS outright: the module
        builds its stock registry eagerly, and ``FullYearCombinedGlobalFit``
        -- a variant this run never uses -- raises "mojito_source_ids must
        inject at least 1 source total". The run died before it built
        anything.
        """
        for knob in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS"):
            self.assertNotIn(
                knob, self.three,
                f"{knob} must be UNSET, not set-empty: a set-empty value "
                f"overrides the settings default and makes `import erebor` "
                f"raise from an unrelated variant's constructor")

    def test_unset_ids_still_leave_every_source_branch_unarmed(self):
        """The other half: unset must not accidentally ARM anything."""
        import sys
        sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))
        from run_combined_staged import _source_ids_from_env

        from lisatools.globalfit.stock.erebor.source_runtime import (
            default_source_ids,
        )

        saved = {k: os.environ.pop(k, None)
                 for k in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS")}
        try:
            self.assertEqual(_source_ids_from_env(), {},
                             "no source branch may be armed")
            # ... while the SETTINGS default still injects something, which
            # is what keeps the eager registry constructible.
            self.assertGreaterEqual(
                sum(len(v) for v in default_source_ids().values()), 1,
                "the settings default must stay non-empty or `import "
                "erebor` raises again")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_the_data_carries_no_source_streams(self):
        self.assertEqual(self.three["SOURCE_TYPES"], "NOISE,GB,VGB")

    def test_the_source_search_skip_is_not_set(self):
        """It is an ERROR, not a no-op, with no armed sources.

        ``run_combined_staged`` raises "STAGE_SKIP_SOURCE_SEARCH=1 but no
        source branches are armed -- there is no source_search stage to
        skip." The 6mo script sets it; carrying it across would kill this
        run at startup, which is how it was found.
        """
        self.assertNotIn("STAGE_SKIP_SOURCE_SEARCH", self.three)

    def test_no_warm_start(self):
        self.assertEqual(
            self.three.get("GB_WARM_START_COMPONENTS", ""), "",
            "a 3-month run is the SOURCE of warm-start components, not a "
            "consumer; empty is the documented off switch")

    def test_every_other_knob_matches_the_6mo_script(self):
        """The whole point of the merge: only the listed knobs differ."""
        allowed = {
            "TOBS_TARGET", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS",
            "COARSE_Q", "COARSE_GPU_MODE", "BASE_FILE_NAME",
            "SOURCE_TYPES", "MBHB_IDS", "EMRI_IDS", "SOBHB_IDS",
            "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_STORE",
            "STAGE_SKIP_SOURCE_SEARCH",
            # sig-het memory sizing, split 2026-09-18. The 6-month run OOM'd
            # in gb_search inside bin_fold_real (the fold chunker sizes each
            # chunk to FILL the byte cap, so an 8 GiB cap builds an ~8 GiB
            # transient) and took the revert its own comment block prescribes
            # for the full_pe handoff. The 3-month run's slots are ~0.25 MB
            # against the 6mo ~0.5 MB and it has been healthy through
            # iteration 210, so it keeps the aggressive sizing. All four are
            # transient knobs -- no stored number depends on them -- and both
            # scripts make them env-overridable, so the split is a default,
            # not a fork.
            # GF_MONITOR_PAGE: 0 on 3mo, unset (page ON) on 6mo
            # (2026-09-28). The in-process page render leaked the live
            # store handle when the generator raised, killing the 3mo
            # saver; a4 fixed that, but the in-process path still has
            # no timeout wired, so a page that HANGS rather than raises
            # would still block the saver. 3mo is tar-only until that
            # is wired and one clean stage has run on the 6mo, which
            # keeps the page. Remove this entry when 3mo takes it back.
            # (TWO twin tests, TWO allowed lists -- authorising only one
            # leaves the other failing.)
            "GF_MONITOR_PAGE",
            "GB_INMODEL_SETUP_BATCH", "GB_SIGHET_FOLD_MAX_BYTES",
            "GB_INFOMAT_MEMPOOL_FREE", "GB_INMODEL_BATCH_MEMPOOL_FREE",
            # 6mo-v8 RUN tuning, 2026-09-20..23, after this 3mo v8 arm's last
            # sync (09-19) and never carried back: both v8 scripts were then
            # superseded by the v9 pair (submit_gf_3mo_v9_2gpu.sh 09-26,
            # ThreeMonthV9TwinTest). Recorded here, not ported, because the
            # 3mo v8 arm is retired and porting would change what it runs.
            #   2c815dd6 09-20  GB_OPT_SNR_LIMIT_SEARCH 5.0, GB_SEARCH_SOURCE_EVERY 10
            #   17d0b1dc 09-20  GB_RJ_PHASE_MAXIMIZE 1 -> 0 (6mo test)
            #   369f4904 09-21  GB_FSTAT_REFIT_EVERY 40, FSTAT_GRID_MEM_MB 512
            #   b73a6ab4 09-22  GB_SEARCH_NOISE_ITERS_PER_STEP 100,
            #                   GB_PE_RJ_FSTAT_FRACTION 0.5
            #   52eb1e38 09-23  GB_SEARCH_PRIOR_REMOVAL_ONLY (new knob)
            #   a456fa27 09-23  MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=0 (pins the
            #                   6mo v8-lineage store's arm model)
            #   129d8b78 09-23  FSTAT_PEAK_MIN_SNR 6.25
            "GB_OPT_SNR_LIMIT_SEARCH", "GB_SEARCH_SOURCE_EVERY",
            "GB_RJ_PHASE_MAXIMIZE", "GB_FSTAT_REFIT_EVERY", "FSTAT_GRID_MEM_MB",
            "GB_SEARCH_NOISE_ITERS_PER_STEP", "GB_PE_RJ_FSTAT_FRACTION",
            "GB_SEARCH_PRIOR_REMOVAL_ONLY", "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM",
            "FSTAT_PEAK_MIN_SNR",
        }
        keys = (set(self.three) | set(self.six)) - {"_", "SHLVL", "PWD"}
        diff = {k for k in keys
                if self.three.get(k, "<unset>") != self.six.get(k, "<unset>")}
        unexpected = diff - allowed
        self.assertEqual(
            unexpected, set(),
            f"these knobs drifted apart and are not on the 3mo reversion "
            f"list: {sorted(unexpected)}")

    def test_the_multirank_and_correctness_updates_came_across(self):
        """The reason to derive from the 6mo file rather than the old 3mo one."""
        for knob, value in (("VGB_CHIRP_MASS_BASIS", "1"),
                            ("VGB_SIGHET_INMODEL", "1"),
                            ("VGB_INMODEL_PROPOSAL", "observable"),
                            # full -> axis in BOTH v8 scripts (f8ab2f33,
                            # 2026-09-19: "full" let one ill-conditioned
                            # eigen axis contaminate every joint step)
                            ("GB_INMODEL_OBSERVABLE_EIGEN", "axis"),
                            ("GB_LEAF_CAP_MIN_ITERS", "3"),
                            ("NWALKERS", "4"),
                            ("SIGHET_TUKEY_ALPHA", "0.01"),
                            ("SOBBH_EIGEN_SCOPE", "walker_max")):
            self.assertEqual(self.three.get(knob), value, knob)


class ThreeMonthV9TwinTest(unittest.TestCase):
    """``submit_gf_3mo_v9_2gpu.sh`` is ``submit_gf_6mo_v9_4gpu.sh`` at 3 months.

    Same silent-failure shape the v8 twin above guards, and the same remedy:
    the two files are ~99% the same text, so the obvious way to carry a fix
    across is to copy the block -- and the blocks that must NOT be copied are
    exactly the ones that look like ordinary knobs. A 3-month run that quietly
    analysed 6 months of data, armed the source branches, or fitted the
    foreground in its first stage would produce entirely plausible output and
    waste the allocation.

    Design + the provenance of every number:
    ``docs/superpowers/specs/2026-09-26-3mo-v9-2gpu-design.md``.
    """

    # the 90-day add-back fixed point, before inflation
    GALFOR_BASE_AMP = 1.436180605904e-44
    GALFOR_BASE_FK = 2.533915614978e-03

    def setUp(self):
        self.three = _exports(THREE_MO_V9)
        self.six = _exports(SIX_MO_V9)

    # -- 3MO-1: Tobs and only its derived settings ----------------------
    def test_tobs_and_its_derived_settings_are_the_3mo_values(self):
        self.assertEqual(self.three["TOBS_TARGET"], "7776000")
        self.assertEqual(self.three["GB_NLEAVES_MAX"], "10000")
        self.assertEqual(self.three["GB_N_SUBBANDS"], "32768")
        self.assertEqual(self.three["GB_RJ_INMODEL_CHUNK"], "65536")
        self.assertEqual(self.three["BASE_FILE_NAME"], "gf_prod_3mo")
        # 6mo-only; at Nt=2160 the CODE DEFAULTS are the validated values
        # (nt_layer 64 -> snaps to 60, stride 36 = 36 h; taper 11 + margin 8
        # = 19 <= the default crop 20).
        self.assertNotIn("SIGHET_NT_LAYER", self.three)
        self.assertNotIn("EDGE_CROP_WAVELETS", self.three)

    def test_the_sighet_STAGING_knobs_do_NOT_revert(self):
        """The v8 3mo arm's aggressive sizing is deliberately NOT taken.

        ``GB_N_SUBBANDS=32768`` is only safe because these four keep the 6mo
        post-OOM values: the fold chunker sizes each chunk to FILL
        ``GB_SIGHET_FOLD_MAX_BYTES``, so an 8 GiB cap builds an ~8 GiB
        transient by design, and that plus one-block staging is what killed
        the 6mo run in ``bin_fold_real``. Pinned as a PAIR with the slot
        count above, because raising one without the other is the failure.
        """
        for knob, want in (("GB_INMODEL_SETUP_BATCH", "2048"),
                           ("GB_SIGHET_FOLD_MAX_BYTES", "1073741824"),
                           ("GB_INFOMAT_MEMPOOL_FREE", "1"),
                           ("GB_INMODEL_BATCH_MEMPOOL_FREE", "1")):
            self.assertEqual(self.three[knob], want, knob)
            self.assertEqual(self.three[knob], self.six[knob], knob)

    # -- 3MO-2: no warm start, no seed stage, no seed store -------------
    def test_no_warm_start(self):
        self.assertEqual(
            self.three.get("GB_WARM_START_COMPONENTS", "<unset>"), "",
            "a 3-month run is the SOURCE of warm-start components, not a "
            "consumer; EMPTY (not unset) is the documented off switch, and "
            "the 6mo `${VAR-default}` form would re-arm it if the line went")

    def test_no_seed_stage_and_no_cadence_knob_left_dangling(self):
        self.assertEqual(self.three["GB_SEARCH_SEED_ITERS"], "0")
        self.assertNotIn(
            "GB_SEARCH_3_WARM_EVERY", self.three,
            "it cadences rj_warm_search, and there is no warm move to "
            "cadence -- a knob that resolves and reaches nothing")

    def test_the_seed_store_is_gone_entirely(self):
        for knob in ("GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE",
                     "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_LAST_K",
                     "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES"):
            self.assertNotIn(knob, self.three, knob)

    def test_replace_cannot_run_a_warm_pass_it_has_no_container_for(self):
        self.assertEqual(self.three["GB_REPLACE_WARM_PASS"], "0")

    # -- 3MO-3: psd-only first stage, galfor frozen at the reference -----
    def test_the_noise_evolves_throughout(self):
        """Revised 2026-09-26 off the first 3-month run.

        The first attempt fitted the psd ALONE in stage 0 and then FROZE
        psd+galfor through gb_search_1/2. Two failures at once: the four
        walkers' psds diverged (S_oms 1.74x, S_tm 7.81x spread, leaf counts
        ordering exactly inversely to each walker's S_tm) and whatever stage 0
        converged to was contaminated by the entire unsubtracted GB galaxy.
        Freezing was the defect, so nothing is frozen now.
        """
        self.assertEqual(self.three["GB_SEARCH_SAMPLE_NOISE_ALL_STAGES"], "1")
        self.assertNotIn(
            "STAGE_NOISE_PSD_ONLY", self.three,
            "galfor must sample from iteration 0 -- the psd-only stage is "
            "exactly what this run learned not to do")
        self.assertEqual(self.three["STAGE_SKIP_NOISE_VGB"], "1")
        self.assertEqual(self.three["NOISE_SEARCH_CHECKS"], "5")
        # global to JointMaxLogLSearch, so it governs this stage too
        self.assertEqual(self.three["MAXLOGL_TOL"], "20")

    def test_the_plateau_is_per_walker_and_freezes_the_converged(self):
        """The laggard decides, not the best walker.

        ``JointMaxLogLSearch`` tested ``state.log_like[0].max()``, so the
        stage advanced as soon as the luckiest walker stopped climbing --
        measured on job 636 row 0, plateau declared at best=52494067.95 with
        the other three cold walkers 71,050 / 128,080 / 1,600 lnL below it.
        """
        self.assertEqual(self.three["MAXLOGL_PER_WALKER"], "1")
        # OFF: the freeze has to restore the sub-state as well as the
        # main state, and shipping it on killed the first relaunch.
        self.assertEqual(self.three["MAXLOGL_FREEZE_CONVERGED"], "0")

    def test_PSD_START_PARAMS_is_unset_or_the_first_stage_disappears(self):
        """Pinning it flips ``_noise_pinned`` and the noise stages are SKIPPED.

        That would delete this run's entire first stage while every log line
        still read as though the psd had been fitted.
        """
        self.assertNotIn("PSD_START_PARAMS", self.three)

    def test_the_foreground_reference_is_the_INFLATED_3mo_fixed_point(self):
        vals = [float(x) for x in
                self.three["GALFOR_START_PARAMS"].split(",")]
        self.assertEqual(len(vals), 5)
        amp, fk, alpha, f_1, f_2 = vals
        self.assertAlmostEqual(amp / self.GALFOR_BASE_AMP, 1.25, places=9,
                               msg="amp inflation (user ruling 2026-09-26)")
        self.assertAlmostEqual(fk / self.GALFOR_BASE_FK, 1.10, places=9,
                               msg="knee inflation: x1.06 is the "
                                   "self-consistent G^(3/11) partner of "
                                   "x1.25 and x1.20 reproduces the 6mo "
                                   "k~2.0 pathology, so 1.10 is the point")
        self.assertEqual(alpha, 5.0)
        self.assertAlmostEqual(f_2, 1.405721657329e-03, places=15)
        # every value strictly INSIDE its prior box -- f_1 in particular is
        # nudged off the 1e-2 rail the v9 launch audit flagged.
        for name, v, lo, hi in (("amp", amp, 1e-47, 1e-41),
                                ("fk", fk, 0.8e-3, 1e-2),
                                ("f_1", f_1, 1e-5, 1e-2),
                                ("f_2", f_2, 1e-5, 1e-2)):
            self.assertGreater(v, lo, name)
            self.assertLess(v, hi, f"{name} is ON or past its prior rail")

    # -- 3MO-4: data + source branches -----------------------------------
    def test_the_data_is_the_COMBINED_stream_with_no_source_branches(self):
        self.assertEqual(self.three["SOURCE_TYPES"], "COMBINED,GB,VGB")

    def test_source_id_vars_are_UNSET_not_set_empty(self):
        """Set-empty looks equivalent and breaks ``import erebor`` outright.

        ``source_runtime.default_source_ids`` tests ``... is not None``, so a
        set-but-empty value replaces the class default with three empty lists
        and ``FullYearCombinedGlobalFit`` -- a variant this run never uses --
        raises from the eagerly-built stock registry.
        """
        for knob in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS"):
            self.assertNotIn(knob, self.three, knob)

    def test_the_source_search_skip_is_not_set(self):
        """It is an ERROR, not a no-op, with no armed sources."""
        self.assertNotIn("STAGE_SKIP_SOURCE_SEARCH", self.three)

    # -- 3MO-6: the fresh store takes the correct reference fit ----------
    def test_the_unequal_arm_psd_reference_fit_is_taken(self):
        self.assertEqual(
            self.six["MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM"], "0",
            "the 6mo pin exists to keep its v8-lineage store resumable")
        self.assertNotIn(
            "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM", self.three,
            "this store is FRESH, so it takes the default (=1, unequal arm) "
            "-- general.psd_injection is the truth line every monitor "
            "compares against, and the estimator was fitting EQUAL-arm while "
            "the run ran unequal")

    def test_every_other_knob_matches_the_6mo_v9_script(self):
        """The whole point of deriving from 6mo: only the listed knobs differ."""
        allowed = {
            # 3MO-1 Tobs + derived
            # GF_MONITOR_PAGE: 0 on 3mo, unset (page ON) on 6mo
            # (2026-09-28). The in-process page render leaked the live
            # store handle when the generator raised, killing the 3mo
            # saver; a4 fixed that, but the in-process path still has
            # no timeout wired, so a page that HANGS rather than raises
            # would still block the saver. 3mo is tar-only until that
            # is wired and one clean stage has run on the 6mo, which
            # keeps the page. Remove this entry when 3mo takes it back.
            # (TWO twin tests, TWO allowed lists -- authorising only one
            # leaves the other failing.)
            "GF_MONITOR_PAGE",
            # 2026-10-04: the dedicated monitor rank and its 10-save
            # cadence went into the 6mo/1yr v9 launchers only (one extra
            # task); the 3mo v9 twin keeps the saver-rank hook at 3.
            "GF_MONITOR_RANK", "GF_MONITOR_ITER",
            "TOBS_TARGET", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS",
            "BASE_FILE_NAME", "STORE_DIR", "SLURM_LOG",
            # 3MO-2 no warm start / no seed stage / no seed store
            "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_STORE",
            "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_LAST_K",
            "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES",
            "GF_SEED_STORE", "GB_SEARCH_SEED_ITERS", "GB_SEARCH_3_WARM_EVERY",
            "GB_REPLACE_WARM_PASS",
            # 3MO-3 the noise evolves throughout + the inflated start point
            "GB_SEARCH_SAMPLE_NOISE_ALL_STAGES", "STAGE_NOISE_PSD_ONLY",
            "STAGE_SKIP_NOISE_VGB", "NOISE_SEARCH_CHECKS",
            "MAXLOGL_PER_WALKER", "MAXLOGL_FREEZE_CONVERGED",
            "GALFOR_START_PARAMS",
            # 3MO-4 data + source branches
            "SOURCE_TYPES", "MBHB_IDS", "EMRI_IDS", "SOBHB_IDS",
            "STAGE_SKIP_SOURCE_SEARCH",
            # 3MO-6 the unequal-arm reference fit
            "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM",
            # 3MO-9 (2026-10-02): nor the 6mo SOBBH lookup block (its preflight is
            # written for the 6-month grid).
            "SOBBH_LIKELIHOOD", "SOBBH_LOOKUP_TABLE_PATH", "SOBBH_LOOKUP_EVAL_DT",
            "SOBBH_LOOKUP_ROW_BATCH", "SOBBH_LOOKUP_KERNEL",
            # 3MO-7 (2026-09-30): the 6mo batched-MBH block is NOT carried
            # across. No MBH branch is armed here, AND the batched default
            # window (90 + 10 + 2 x 1 d kept + 2 x 4 d pad) does not fit a
            # 90-day grid, so the documented MBHB_IDS=5 escape would die at
            # build. See SixMonthMBHBatchedTest.test_the_3mo_twin_does_not_inherit_it.
            "MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
            # V9-28 (2026-10-02, MBH session): the batched window generated at
            # q*dt; default 1 = what job 695 ran. The MBH suites pin the block.
            "MBH_WINDOW_DECIMATE",
            # 2026-10-12: the LISA-frame sky-mode hops back in the MBH inner-move
            # stack (SkyMove in the ICRS basis; 0.02 both / 0.04 lat / 0.02 long
            # of the per-leaf draws, user ruling). New knobs, same code defaults.
            "MBH_SKY_MOVE_BOTH_WEIGHT", "MBH_SKY_MOVE_LAT_WEIGHT",
            "MBH_SKY_MOVE_LONG_WEIGHT",
            # 2026-10-13: the eigen jump mixture block of the 6mo / 1yr / 9mo
            # launchers (no source branch is armed in the 3mo v9 twin)
            *EIGEN_JUMP_MIX_KNOBS,
            # 2026-10-14: the per-axis step-scale adaptation switches, same reason
            *EIGEN_AXIS_ADAPT_KNOBS,
            # 2026-10-15: pooled / tempered / factor-capped adaptation, same block
            *EIGEN_POOL_TEMPER_KNOBS,
            # 2026-10-18: the EMRI sampling audit block, same reason (no source
            # branch is armed in the 3mo v9 twin, which carries no EMRI export)
            *EMRI_AUDIT_KNOBS,
            # 2026-10-19: the MBH proposal frame, same reason
            *PROPOSAL_FRAME_KNOBS,
            # 2026-10-13: the ladder cap exports, mirrored in the 3mo twin
            "EMRI_LADDER_TMAX", "EMRI_LADDER_REANCHOR_NATS",
            "MBH_LADDER_TMAX", "MBH_LADDER_REANCHOR_NATS",
            "SOBBH_LADDER_TMAX", "SOBBH_LADDER_REANCHOR_NATS",
            # 2026-10-14: the cold -> hot rotation cadence, mirrored in the
            # 3mo twin (inert there: no source branch is armed)
            *COLD_TO_HOT_KNOBS,
            # 2026-10-13: the GB warm-start self-refit block, mirrored in the
            # 3mo twin (inert there: no GB_WARM_START_COMPONENTS)
            *WARM_SELF_KNOBS,
            # 2026-10-16: the gb_search_2 in-model budget, mirrored in the
            # 3mo twin
            *SEARCH2_INMODEL_KNOBS,
            # 2026-10-19: the gb_search_2 F-stat refit cadence, mirrored
            *SEARCH2_FSTAT_REFIT_KNOBS,
            # 2026-10-16: the rotating search F-stat reference, mirrored
            *FSTAT_SEARCH_REF_KNOBS,
            # 2026-10-17: the valve's all-walker rule, mirrored
            *SHUTOFF_ALL_WALKERS_KNOBS,
            # 3MO-8 (2026-10-02): nor is the 6mo EMRI direct-to-WDM block: no
            # EMRI branch is armed here, and its lookup table is built for the
            # 6-month grid's Nf and dt only.
            "EMRI_LIKELIHOOD", "EMRI_BATCH_MAX_SIZE", "EMRI_DIRECT_TABLE",
            "EMRI_DIRECT_RESPONSE", "EMRI_TRAJ_WORKERS", "EMRI_DIRECT_LOOKUP",
            # V9-30 (2026-10-07): Mike's 6mo production line became the
            # 6mo / 1yr / 9mo launcher DEFAULT (MIDIT_CHECKPOINT 0, 8-rung
            # MBH / EMRI ladders, gram / fisher eigen tables, source
            # cross-checks off). The 3mo v9 twin was not on that line and
            # keeps its own values (no source branch is armed there).
            "MIDIT_CHECKPOINT", "MBH_NTEMPS", "EMRI_NTEMPS",
            "SOBBH_EIGEN_INFO", "EMRI_EIGEN_INFO", "MBH_EIGEN_INFO",
            "PSD_EIGEN_INFO", "GALFOR_EIGEN_INFO",
            "SOBBH_CHECK_LL", "MBH_CHECK_LL", "EMRI_CHECK_LL",
            # 2026-10-07: stage clocks survive a resume (reset at each
            # boundary); the 3mo v9 script does not export it
            "GF_PERSIST_STAGE_START",
            # 2026-10-07: the GB ladder re-spacing (fresh stores only); the
            # 3mo v9 script keeps the stock ladder
            "GB_LADDER_HOT_RATIO", "GB_LADDER_HOT_FROM", "GB_LADDER_PIN_LAST",
        }
        keys = (set(self.three) | set(self.six)) - {"_", "SHLVL", "PWD"}
        diff = {k for k in keys
                if self.three.get(k, "<unset>") != self.six.get(k, "<unset>")}
        unexpected = diff - allowed
        self.assertEqual(
            unexpected, set(),
            f"these knobs drifted apart and are not on the 3mo v9 delta "
            f"list: {sorted(unexpected)}")

    def test_the_v9_search_machinery_came_across_intact(self):
        """The reason to derive from the 6mo v9 file rather than 3mo_v9.sh."""
        for knob in ("GB_INMODEL_CONVERGE", "GB_INMODEL_CONVERGE_ITERS",
                     "GB_INMODEL_CONVERGE_ITERS_SURVIVOR",
                     "GB_INMODEL_CONVERGE_CLASSES", "GB_INMODEL_GROUP",
                     "GB_NUM_REPEAT_PROPOSALS", "GB_RJ_DIRECT_BATCH",
                     "GB_RUN_FANCY_TEMPERING", "GB_TEMPER_VERTICAL",
                     "GB_SEARCH_BAND_SHUTOFF_PER_WALKER",
                     "GB_SEARCH_BAND_SHUTOFF_CONV_ITER",
                     "GB_FSTAT_REFIT_EVERY", "FSTAT_PEAK_MIN_SNR",
                     "GB_OPT_SNR_LIMIT_SEARCH", "GB_RJ_PHASE_MAXIMIZE",
                     "GB_LEAF_CAP_START", "STAGE_V9_SEARCH",
                     "COARSE_Q", "COARSE_GPU_MODE",
                     # sig-het accuracy: Tobs-independent, and staleness is
                     # LESS harmful at shorter Tobs, so 50 is safer here
                     "GB_SIGHET_REFRESH_EVERY", "SIGHET_N_CP",
                     "SIGHET_TUKEY_ALPHA", "GB_SIGHET_TRUST_PHASE_C",
                     "NWALKERS", "GB_NTEMPS"):
            self.assertEqual(self.three.get(knob), self.six.get(knob), knob)


class WarmStartPathSeparatorTest(unittest.TestCase):
    """The warm-start npz must land INSIDE the run store (2026-09-18).

    The default was written ``${STORE_DIR}warmstart/...`` with no
    separator, which relies on ``STORE_DIR`` ending in a slash. The
    built-in default does; a command-line override
    (``STORE_DIR=/.../gf_prod_6mo_v8_4gpu ./submit...``) does not, so the
    npz went to a SIBLING directory ``gf_prod_6mo_v8_4gpuwarmstart/`` --
    outside the snapshot zips, and not rebuilt when the store is reset.
    """

    def test_store_dir_and_warmstart_are_separated(self):
        for path in SCRIPTS:
            with open(path) as fh:
                src = fh.read()
            for m in re.finditer(r"\$\{STORE_DIR\}(?!/)(\S{0,24})", src):
                self.assertNotIn(
                    "warmstart", m.group(1),
                    f"{os.path.basename(path)}: "
                    f"${{STORE_DIR}}{m.group(1)} has no path separator, so "
                    f"an override without a trailing slash puts the warm "
                    f"start outside the store")


class SobbhKnobsSingleExportTest(unittest.TestCase):
    """One effective export per SOBBH knob (2026-09-17).

    ``SOBBH_EIGEN_SCOPE=walker_max`` (user ruling 2026-09-16) was exported
    once and then silently overridden by a second, older
    ``export SOBBH_EIGEN_SCOPE=per_walker`` further down the same file, so
    the ruling never reached a run. ``SOBBH_NTEMPS`` must be overridable
    from the environment (a store born at 8 rungs is resumed with
    ``SOBBH_NTEMPS=8``; the stored count wins either way, but the knob
    should not lie in run_settings.log).
    """

    def _exports(self, path, name):
        with open(path) as fh:
            return [
                line.strip() for line in fh
                if line.lstrip().startswith(f"export {name}=")
            ]

    def test_sobbh_eigen_scope_exported_once(self):
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_EIGEN_SCOPE")
            self.assertLessEqual(
                len(lines), 1,
                f"{path}: SOBBH_EIGEN_SCOPE exported {len(lines)} times: {lines}",
            )

    def test_campaign_sobbh_eigen_scope_is_walker_max(self):
        # The ruling applies to the campaign script; the null-test script
        # keeps its own (per-walker) setting and is only held to one export.
        lines = self._exports(SCRIPTS[0], "SOBBH_EIGEN_SCOPE")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("walker_max", lines[0])
        self.assertNotIn("per_walker", lines[0])

    def test_sobbh_repeats_is_env_overridable_and_defaults_to_10(self):
        # User ruling 2026-09-18 (the v8 6mo script). Repeats are the ONLY
        # knob that moves the SOBBH cost: [SOBBH_LL_TIMING] measured a flat
        # 1.73 s per scoring call regardless of rows, and calls come from
        # repeats, not walkers or rungs. 20 -> 10 halves the dominant
        # per-iteration cost.
        lines = self._exports(SCRIPTS[0], "SOBBH_NUM_PROP_REPEATS")
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(
            lines[0], "export SOBBH_NUM_PROP_REPEATS=${SOBBH_NUM_PROP_REPEATS:-10}")

    def test_v9_source_repeats_are_env_overridable_and_default_to_25(self):
        # User ruling 2026-10-03 (the v9 6mo script): "adjust the in-model
        # repeats during search for mbhs emris and sobhbs to 25. same as pe"
        # -- the BUILT values, so every stage runs them (sources every 5
        # iterations in gb_search_3 and replica_pe). Yielding exports:
        # MBH/EMRI were hard `=2` lines a launch-line value could not reach.
        # (SOBBH history: 25 -> 20 on 2026-09-16 -> 10 on 2026-09-18 -> 25.)
        for knob in ("MBH_NUM_PROP_REPEATS", "EMRI_NUM_PROP_REPEATS",
                     "SOBBH_NUM_PROP_REPEATS"):
            lines = self._exports(SIX_MO_V9, knob)
            self.assertEqual(len(lines), 1, (knob, lines))
            self.assertEqual(lines[0], f"export {knob}=${{{knob}:-25}}", knob)

    def test_v9_sobbh_check_cadence_is_env_overridable(self):
        # SOBBH session, 2026-10-03: the cadence was a hard `=30` export, so a
        # SOBBH_CHECK_LL_EVERY on the launch line was silently overwritten
        # (the first lookup segment wants 10, the long-run value stays 30).
        lines = self._exports(SIX_MO_V9, "SOBBH_CHECK_LL_EVERY")
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(
            lines[0], "export SOBBH_CHECK_LL_EVERY=${SOBBH_CHECK_LL_EVERY:-30}")

    def test_v9_pe_inmodel_repeats_cover_emri(self):
        # User ruling 2026-10-03 ("same as pe" for mbhs, emris and sobhbs):
        # emri joined the PE declaration list (excluded on 2026-10-02).
        lines = self._exports(SIX_MO_V9, "PE_INMODEL_REPEATS_BRANCHES")
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(
            lines[0],
            "export PE_INMODEL_REPEATS_BRANCHES=${PE_INMODEL_REPEATS_BRANCHES:-"
            "gb,vgb,sobbh,mbh,emri,psd,galfor}")

    def test_sobbh_ntemps_is_env_overridable_and_defaults_to_8(self):
        # User ruling 2026-09-17: 8 in the scripts (the 4-GPU store is an
        # 8-rung store; resumed 12-rung stores keep 12 via the store-wins
        # rule). Both campaign scripts.
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_NTEMPS")
            self.assertEqual(len(lines), 1, f"{path}: {lines}")
            self.assertEqual(
                lines[0], "export SOBBH_NTEMPS=${SOBBH_NTEMPS:-8}", path)


class SubmitScriptsSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        for path in SCRIPTS:
            result = subprocess.run(
                ["bash", "-n", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0, f"{path}: bash -n failed: {result.stderr}"
            )


class SubmitScriptsInJobBlockTest(unittest.TestCase):
    """The IN-JOB rank-layout block (after the `exec sbatch` self-dispatch).

    The stub-sbatch scenarios below can only observe the PRE-submit dispatch
    block -- everything after `exec sbatch` never runs without a real job --
    so these invariants are checked as text. They are the ones that only bite
    a launch that bypassed the dispatch block entirely (a manual
    `sbatch --nodes=2 --ntasks=5 <script>`), which is exactly the path no
    other test covers.
    """

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def test_multi_node_forces_walker_block_layout_in_job(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'if [ "${SLURM_NNODES:-1}" -gt 1 ] && '
                    '[ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then',
                    text,
                )
                self.assertIn("FORCING GF_LEGACY_RANK_LAYOUT=0", text)

    def test_in_job_exports_all_three_layout_knobs(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn("export GPUS_PER_RANK RANKS_PER_GPU", text)
                self.assertIn("export GF_LEGACY_RANK_LAYOUT", text)

    def test_nwalkers_modulo_is_guarded_against_zero_compute_ranks(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${N_COMPUTE_EFF}" -gt 0 ]', text)
                # the division itself must not be reachable unguarded
                self.assertNotIn(
                    'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
                    "[ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then",
                    text,
                )

    def test_one_walker_is_exempt_from_the_divisibility_fix(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${NWALKERS}" -eq 1 ]', text)
                self.assertIn("one-walker replica mode", text)
                # The sampler-shape export must honour the submitting shell's
                # NWALKERS (carried by --export=ALL), or the branch below it is
                # unreachable: a hard `export NWALKERS=<n>` above the `-eq 1`
                # test would silently run n walkers for `NWALKERS=1 ./submit`.
                # The DEFAULT VALUE is deliberately not pinned here -- it is a
                # campaign choice that moves (10 at the 2026-09-11 rebase, 4
                # for the walker-block store on 2026-09-18); what this test
                # protects is the overridable FORM and its position.
                m = re.search(r"^export NWALKERS=\$\{NWALKERS:-\d+\}",
                              text, re.M)
                self.assertIsNotNone(
                    m, "NWALKERS must be exported as ${NWALKERS:-<default>}")
                self.assertNotRegex(text, r"\nexport NWALKERS=\d+\s")
                self.assertLess(
                    m.start(), text.index('[ "${NWALKERS}" -eq 1 ]'))
                # the rounding branch survives for NWALKERS > 1
                self.assertIn(
                    "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF", text
                )
                # the exemption is tested BEFORE the rounding branch
                self.assertLess(
                    text.index('[ "${NWALKERS}" -eq 1 ]'),
                    text.index(
                        "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF"
                    ),
                )

    def test_multinode_launch_line_is_hydra_round_robin(self):
        # WP7 Step 0 (2026-09-16): on this cluster `srun --mpi=pmix` bootstraps
        # Intel MPI but its OFI address exchange fails, and a bare `srun` gives
        # three size-1 worlds; the launcher that works is hydra with the SLURM
        # bootstrap, `-ppn 1` (round-robin over hosts = cyclic placement) and
        # the tcp fabric provider. Pin all of it so a refactor cannot drift
        # back to `srun`.
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 python', text,
                )
                self.assertIn(
                    "export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi "
                    "FI_PROVIDER=tcp",
                    text,
                )
                # ... and the knob that makes `-ppn 1` bind at all (2026-09-18):
                # with the SLURM bootstrap hydra otherwise honours the
                # scheduler's per-node task counts, so 5 tasks over 2 nodes
                # land 3/2 (block) instead of A,B,A,B,A (cyclic) and
                # build_layout refuses -- 3 compute ranks on a 2-GPU node.
                self.assertIn(
                    "export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", text,
                )
                self.assertNotIn('srun --ntasks="${SLURM_NTASKS', text)


class SubmitScriptsNwalkersBlockTest(unittest.TestCase):
    """Execute the in-job NWALKERS divisibility/exemption block with bash.

    The text checks in ``SubmitScriptsInJobBlockTest`` confirm the exemption
    line and message exist and come first; this class actually runs the
    extracted `if ... fi` block under bash to confirm the one-walker branch
    leaves NWALKERS untouched and the rounding branch still fires (and still
    rounds correctly) for every other NWALKERS value.
    """

    _START_MARKER = (
        'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
        '[ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then'
    )

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def _extract_block(self, path):
        text = self._text(path)
        start = text.index(self._START_MARKER)
        end = text.index("\nfi", start)
        return text[start : end + len("\nfi")]

    def _run_block(self, path, overrides):
        block = self._extract_block(path)
        env = {"PATH": os.environ.get("PATH", "")}
        env.update(overrides)
        result = subprocess.run(
            ["bash", "-c", block + "\necho NW=${NWALKERS}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{path} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout

    def test_nwalkers_1_is_one_walker_replica_mode_and_unchanged(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "1",
                    },
                )
                self.assertIn("one-walker replica mode", stdout)
                self.assertIn("NW=1", stdout.splitlines())

    def test_nwalkers_6_still_rounds_up_to_8(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "6",
                    },
                )
                self.assertIn("is not a multiple of N_COMPUTE", stdout)
                self.assertIn("NW=8", stdout.splitlines())


class SubmitScriptsDispatchTest(unittest.TestCase):
    """Exercise the pre-submit `exec sbatch ...` dispatch block via a stub."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="submit_scripts_stub_")
        cls.stub_dir = cls._tmpdir.name
        stub_path = os.path.join(cls.stub_dir, "sbatch")
        with open(stub_path, "w") as fh:
            fh.write(_STUB_SBATCH)
        os.chmod(stub_path, 0o755)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run_dispatch(self, script, overrides):
        env = dict(os.environ)
        for key in _DISPATCH_ENV_KEYS:
            env.pop(key, None)
        env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        result = subprocess.run(
            ["bash", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{script} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout.splitlines()

    def _assert_export_contains(self, lines, needle):
        export_lines = [l for l in lines if l.startswith("--export=")]
        self.assertTrue(
            export_lines, f"no --export= argv line found; stdout lines: {lines}"
        )
        self.assertTrue(
            any(needle in l for l in export_lines),
            f"{needle!r} not found in --export= line(s): {export_lines}",
        )

    def test_ngpus_2_explicit_legacy_still_launches_today_shape(self):
        # GF_LEGACY_RANK_LAYOUT=1 is the rollback knob: same resource request,
        # --ntasks=3, mpiexec -n 3, the single-compute-rank layout.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "1"}
                )
                self.assertIn(_ntasks(script, 3), lines)
                self.assertIn("--nodes=1", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=1")

    def test_ngpus_2_default_is_walker_block(self):
        # user ruling 2026-09-16 (WP7 Steps 0-2/4 green): the walker-block
        # layout is the default at NGPUS=2 -- head + 1 compute + saver, still
        # --ntasks=3, but GF_LEGACY_RANK_LAYOUT=0 is what the job sees.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn(_ntasks(script, 3), lines)
                self.assertIn("--nodes=1", lines)
                # The 3mo v9 run goes ON-DEMAND at every GPU count
                # (user ruling 2026-09-26): its job 633 lost 43 min of
                # gb_search_1 to a silent spot kill. Every other script
                # keeps spot at NGPUS=2, where 1 node is cheap to lose.
                want = ("gpu-80-ondemand" if script == THREE_MO_V9
                        else "gpu-80-spot")
                self.assertIn(f"--partition={want}", lines)
                self.assertIn("--gres=gpu:2", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_4_forces_walker_block_two_nodes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "4"})
                self.assertIn(_ntasks(script, 5), lines)
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:2", lines)
                # NGPUS=4 is 2 nodes, so a spot preemption of EITHER node
                # kills the whole MPI world -- twice the exposure of the
                # 1-node flow. It goes to ON-DEMAND (user 2026-09-18);
                # NGPUS=2 above stays on spot deliberately.
                self.assertIn("--partition=gpu-80-ondemand", lines)
                self.assertNotIn("--partition=gpu-80-spot", lines)

    def test_ngpus_8_four_nodes_walker_block(self):
        """2026-10-14 (Mike, timing tests): NGPUS=8 = 4 nodes x 2 GPUs on
        the on-demand partition, 8 compute ranks + the saver (the walker
        count must then be a multiple of 8: a cloned store). Anything else
        is still refused. The v9 production launchers and the 3mo v9 twin
        carry it; the v8 scripts are frozen history and do not."""
        for script in (*V9_PRODUCTION, THREE_MO_V9):
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "8"})
                self.assertIn(_ntasks(script, 9), lines)
                self.assertIn("--nodes=4", lines)
                self.assertIn("--gres=gpu:2", lines)
                self.assertIn("--partition=gpu-80-ondemand", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")
                src = open(script).read()
                self.assertIn("unsupported (2, 4 or 8)", src)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_the_v9_production_launchers_default_to_four_gpus(self):
        # V9-30 (user ruling 2026-10-07): `./submit...` with NO NGPUS on the
        # line IS the production shape -- NGPUS=4: 2 nodes x 2 GPUs,
        # on-demand, 4 compute + saver + monitor. The 3mo v9 twin keeps its
        # 1-node NGPUS=2 default.
        for script in V9_PRODUCTION:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {})
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:2", lines)
                self.assertIn(_ntasks(script, 5), lines)
                self.assertIn("--partition=gpu-80-ondemand", lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "NGPUS=4")
        lines = self._run_dispatch(THREE_MO_V9, {})
        self.assertIn("--nodes=1", lines)
        self.assertIn(_ntasks(THREE_MO_V9, 3), lines)

    def test_the_4gpu_partition_is_overridable_without_editing(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "4", "PARTITION": "gpu-80-spot"})
                self.assertIn("--partition=gpu-80-spot", lines)

    def test_nodes_knob_spreads_the_gpus_one_per_node(self):
        # one-walker replica gates (user ruling 2026-09-16: test ACROSS nodes):
        # NGPUS=2 NODES=2 -> 2 nodes x gpu:1, head + 1 compute + saver, cyclic
        for script in SCRIPTS:
            with self.subTest(script=script, nodes=2):
                lines = self._run_dispatch(script, {"NGPUS": "2", "NODES": "2"})
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn(_ntasks(script, 3), lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")
            with self.subTest(script=script, nodes=4):
                lines = self._run_dispatch(script, {"NGPUS": "4", "NODES": "4"})
                self.assertIn("--nodes=4", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn(_ntasks(script, 5), lines)
                self.assertIn("--distribution=cyclic", lines)
            with self.subTest(script=script, nodes="unset"):
                # unset NODES = the NGPUS table, unchanged
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--nodes=1", lines)
                self.assertIn("--gres=gpu:2", lines)

    def test_nodes_knob_must_divide_ngpus(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                env = {k: v for k, v in os.environ.items() if k not in _DISPATCH_ENV_KEYS}
                env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
                env.update({"NGPUS": "2", "NODES": "3"})
                result = subprocess.run(
                    ["bash", script], env=env, capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("must be >= 1 and divide NGPUS=2", result.stdout)

    def test_the_monitor_rank_task_is_dropped_with_the_knob_off(self):
        for script in _MONITOR_RANK_SCRIPTS:
            with self.subTest(script=script):
                on = self._run_dispatch(script, {"NGPUS": "4"})
                self.assertIn("--ntasks=6", on)
                self._assert_export_contains(on, "GF_MONITOR_RANK=1")
                off = self._run_dispatch(
                    script, {"NGPUS": "4", "GF_MONITOR_RANK": "0"})
                self.assertIn("--ntasks=5", off)
                self._assert_export_contains(off, "GF_MONITOR_RANK=0")

    def test_ngpus_2_explicit_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "0"}
                )
                self.assertIn(_ntasks(script, 3), lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_2_gpus_per_rank_2_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script,
                    {
                        "NGPUS": "2",
                        "GPUS_PER_RANK": "2",
                        "GF_LEGACY_RANK_LAYOUT": "0",
                    },
                )
                self.assertIn(_ntasks(script, 2), lines)


#: THE 9MO / 1YR RECIPE (user ruling 2026-10-07; the fixed-dim source seed
#: 2026-10-08): the three Tobs-independent entries BOTH generators
#: (derive_1yr_v9_launcher.py, derive_9mo_v9_launcher.py) append to their
#: tables, verbatim. The running 6mo store's recipe is fixed
#: (GFHDFBackend.add_recipe refuses a dropped or reordered stage), so the recipe
#: reaches the 1yr / 9mo launchers only here.
_RECIPE_REPLACEMENTS = (
    ("export GB_SEARCH_SEED_ITERS=5\n",
     "# ---- THE 9MO / 1YR RECIPE (user ruling 2026-10-07) -------------------------\n"
     "# gb_search_seed (3 iterations) -> gb_search_1 -> gb_search_2 -> full_pe. No\n"
     "# gb_search_3, no replica_pe (STAGE_REPLICA_PE above), no galfor ratchet\n"
     "# (GALFOR_RATCHET stays 0). The seed: the 6mo's, 3 iterations not 5, no\n"
     "# source moves. gb_search_1: its table row (opt SNR 8, phase max, F-stat peak\n"
     "# 8, prior removal only), noise FIXED, NO sobbh/mbh/emri moves. gb_search_2:\n"
     "# its table row (opt SNR 5, no phase max, peak 6.25, prior births AND\n"
     "# deaths), the valves reset at its fresh entry, the sources EVERY iteration,\n"
     "# and ONE psd+galfor search to CONVERGENCE at the END of every cycle (\"only\n"
     "# one noise/galfor proposal ... at the end of the cycle\"): the standalone\n"
     "# noise stage's plateau rule (NOISE_SEARCH_CHECKS flat rounds per walker\n"
     "# within MAXLOGL_TOL), afresh each cycle. Then full_pe, declared as today.\n"
     "# The resolved recipe, no data / no MPI, under this script's exports:\n"
     "#   python scripts/fstat_proposal/run_combined_staged.py --print-recipe\n"
     "export GB_SEARCH_SEED_ITERS=3\n"
     "export GB_SEARCH_STAGES=${GB_SEARCH_STAGES:-1,2}\n"
     "export GB_SEARCH_1_SOURCE_EVERY=${GB_SEARCH_1_SOURCE_EVERY:-1}   # 2026-10-10 ruling: sobbh/mbh/emri every cycle in gb_search_1 too (was 0)\n"
     "export GB_SEARCH_2_SOURCE_EVERY=${GB_SEARCH_2_SOURCE_EVERY:-1}\n"
     "export GB_SEARCH_2_NOISE_MODE=${GB_SEARCH_2_NOISE_MODE:-cycle_end}\n"
     "export GB_SEARCH_2_RESET_VALVES=${GB_SEARCH_2_RESET_VALVES:-1}\n", 1),
    ("export STAGE_REPLICA_PE=${STAGE_REPLICA_PE:-1}\n",
     "export STAGE_REPLICA_PE=${STAGE_REPLICA_PE:-0}   # 9mo/1yr recipe (ruling 2026-10-07): \"No replica pe\" -- gb_search_2 hands over to full_pe\n", 1),
    # the fixed-dim source seed (user ruling 2026-10-08; placed after the seed by
    # the main session the same day)
    ("export STAGE_SKIP_SOURCE_SEARCH=1\n",
     "# ---- 9MO / 1YR: THE FIXED-DIM SOURCE SEED (user ruling 2026-10-08) ---------\n"
     "# \"since we are seeing the MBHs move quite a bit towards a better likelihood\n"
     "# ... 'fixed-dim source seed' ... the galfor and psd will be fixed just like in\n"
     "# gb search seed and gb search 1. However, it will only run MBHBs, SOBHBs, and\n"
     "# EMRIs, until the maximum likelihood of each walker converges.\" The existing\n"
     "# source_search stage, RUN here (STAGE_SKIP_SOURCE_SEARCH 1 -> 0) and placed\n"
     "# AFTER gb_search_seed (SOURCE_SEARCH_POSITION=after_seed, main session\n"
     "# 2026-10-08: the source maxima are found against a residual with the\n"
     "# warm-started galaxy already subtracted; =first puts it before the seed).\n"
     "# Moves: sobbh_pe / mbh_pe / emri_pe ONLY, from their start points (the 6mo cold chain for SNR > 10 sources, exact truth otherwise: the SOURCE WARM START block below);\n"
     "# psd + galfor FIXED at the start pin, the GB leaves held at the seed's\n"
     "# warm-start births, the VGBs held. Stop: every walker's cold lnL gains no\n"
     "# more than SOURCE_SEARCH_TOL nats over SOURCE_SEARCH_CHECKS consecutive\n"
     "# rounds (a round = one pass of the three moves at their in-model repeats;\n"
     "# MAXLOGL_PER_WALKER=1, the laggard walker decides), at most\n"
     "# SOURCE_SEARCH_MAX_ROUNDS rounds, MAXLOGL_ITERS_PER_STEP (10) rounds per\n"
     "# stored row. MAXLOGL_TOL=20 / NOISE_SEARCH_CHECKS stay the gb_search_2\n"
     "# cycle-end noise slot's.\n"
     "export STAGE_SKIP_SOURCE_SEARCH=${STAGE_SKIP_SOURCE_SEARCH:-0}\n"
     "export SOURCE_SEARCH_POSITION=${SOURCE_SEARCH_POSITION:-after_seed}\n"
     "export SOURCE_SEARCH_CHECKS=${SOURCE_SEARCH_CHECKS:-10}\n"
     "export SOURCE_SEARCH_TOL=${SOURCE_SEARCH_TOL:-10}\n"
     "export SOURCE_SEARCH_MAX_ROUNDS=${SOURCE_SEARCH_MAX_ROUNDS:-200}\n", 1),
    # the galfor frequency prior (user ruling 2026-10-08; 9mo / 1yr only -- the
    # 6mo relaunch keeps the stock box)
    ("export GALFOR_ALPHA_MAX=20.0\n",
     "export GALFOR_ALPHA_MAX=20.0\n"
     "# ---- 9MO / 1YR: GALFOR FREQUENCY PRIOR (user ruling 2026-10-08) -----------\n"
     "# \"adjust the frequency parameters (fk, f1, f2) in the foreground model ...\n"
     "# in its prior ... to go from 1e-4 to 1e-2\": the three frequency columns\n"
     "# share one box, 0.1-10 mHz (stock: fk 0.8-10 mHz, f_1 / f_2 10 uHz-10 mHz).\n"
     "# One resolver (galfor_prior_ranges) feeds the prior, the start-pin window\n"
     "# and the noise-pin refusal, so the GALFOR_START_PARAMS pin (fk 2.53 mHz,\n"
     "# f_1 10 mHz, f_2 1.41 mHz) is inside the box. Fresh store only: a chain\n"
     "# outside the box would price at log_prior = -inf on resume.\n"
     "# 2026-10-09 (Mike): \"adjust the foreground prior for f_k and f_2 to go from\n"
     "# 0.8 mHz to 10 mHz. Keep f_1 as is.\" -- per-column boxes, in Hz; f_1 keeps the\n"
     "# 2026-10-08 box. The start pin (fk 2.53 mHz, f_1 10 mHz, f_2 1.41 mHz) is inside.\n"
     "export GALFOR_FK_PRIOR=${GALFOR_FK_PRIOR:-0.8e-3,1e-2}\n"
     "export GALFOR_F1_PRIOR=${GALFOR_F1_PRIOR:-1e-4,1e-2}\n"
     "export GALFOR_F2_PRIOR=${GALFOR_F2_PRIOR:-0.8e-3,1e-2}\n", 1),
    # the source warm start from the 6mo cold chain (user ruling 2026-10-08;
    # the same entry in both derive tables)
    ("export MBH_START_FACTOR=0.0\nexport EMRI_START_FACTOR=0.0\nexport SOBBH_START_FACTOR=0.0\nexport VGB_START_FACTOR=0.0\n",
     "export MBH_START_FACTOR=0.0\nexport EMRI_START_FACTOR=0.0\nexport SOBBH_START_FACTOR=0.0\nexport VGB_START_FACTOR=0.0\n"
     "# ---- 9MO / 1YR: SOURCE WARM START from the 6mo cold chain (user ruling\n"
     "# 2026-10-08) -------------------------------------------------------------\n"
     "# \"start the 9mo sources from the 6mo cold chain's final positions instead of\n"
     "# catalogue truth ... for any source over SNR 10 at 6 mo. Otherwise start how\n"
     "# we did at 6mo [exact truth] for sources under SNR 10 (computed at six\n"
     "# months)\". lisatools.globalfit.warmstart.sources reads the seed store's last\n"
     "# written cold-chain row (sub_backend/<branch>/chain + h_h; the running backup\n"
     "# copy if the primary is torn), maps leaves by CATALOGUE ID (the seed's own id\n"
     "# lists below, NOT this run's: MBHB 7 and 12 were never fitted at 6mo and\n"
     "# start at truth), takes SNR_6mo = median over the cold walkers of sqrt(h_h),\n"
     "# and for SNR > SOURCE_WARM_START_SNR_MIN copies walker w -> w and the whole\n"
     "# 8-rung ladder (the rung counts match). Below the threshold, or absent from\n"
     "# the seed: exact truth as above. A NaN record REFUSES the launch\n"
     "# (SOURCE_WARM_START_SNR_UNKNOWN=truth|warm overrides); a start outside this\n"
     "# run's prior refuses it. Fresh start only (a resume keeps its chain). DRY RUN\n"
     "# on the cluster BEFORE launching, once per branch (prints each id's 6mo SNR\n"
     "# and the warm / truth decision):\n"
     "#   python -m lisatools.globalfit.warmstart.sources --store $GF_SEED_STORE \\\n"
     "#       --branch mbh --ids 2,5,16,18\n"
     "# OFF BY DEFAULT SINCE 2026-10-22 (Mike: the non-GB sources \"always start at\n"
     "# the catalog injected values rather than the last run's values\"): every\n"
     "# source leaf starts at exact truth (*_START_FACTOR=0 above). Opt back in for\n"
     "# one launch with SOURCE_WARM_START_STORE=$GF_SEED_STORE on the line; to push a\n"
     "# stopped run's sources back to the injection use\n"
     "# scripts/flagship/relaunch_sources_at_injection.sh.\n"
     "export SOURCE_WARM_START_STORE=${SOURCE_WARM_START_STORE-}\n"
     "export SOURCE_WARM_START_SNR_MIN=${SOURCE_WARM_START_SNR_MIN:-10}\n"
     "export MBH_SOURCE_WARM_START_IDS=${MBH_SOURCE_WARM_START_IDS:-2,5,16,18}\n"
     "export EMRI_SOURCE_WARM_START_IDS=${EMRI_SOURCE_WARM_START_IDS:-0,1,2,3,4,5,6,7}\n"
     "export SOBBH_SOURCE_WARM_START_IDS=${SOBBH_SOURCE_WARM_START_IDS:-0,1,2,3,4,5}\n", 1),
    # the PSD reference fit (user rulings 2026-09-26 [3mo] and 2026-10-04 [unequal
    # arm only]; the 6mo line's own comment: "DELETE THIS LINE for a fresh store")
    ("export MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=0\n",
     "# ---- 9MO / 1YR: the reference fit is the UNEQUAL-ARM fit (2026-10-08) --------\n"
     "# The =0 pin above exists ONLY so the resumed 6mo store keeps the equal-arm\n"
     "# pair its chain was started with (coarse_fiducial_digest). A fresh store has\n"
     "# no such debt: =1 (the code default when unset; the 3mo v9 dropped the line,\n"
     "# ruling 2026-09-26) makes general.psd_injection -- the reference / 'truth'\n"
     "# pair the pages and diagnostics compare against, NOT the noise model, which\n"
     "# is UNEQUAL_ARM=1 either way -- the unequal-arm fit 1.500004e-11 /\n"
     "# 3.000107e-15. Ruling 2026-10-04: no default may rely on an equal-arm path.\n"
     "export MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM=${MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM:-1}\n", 1),
    # the stage-B F-stat sky cap (pre-launch check 2026-10-09; Mike: "if you think
    # ... for the 1yr run, then yes do that")
    ("export FSTAT_STAGEB_NSKY_MAX=0       # 0 = no node cap (user ruling)\n",
     "# ---- 9MO / 1YR: cap the stage-B sky grid at the 6mo's 8-mHz node count ------\n"
     "# The adaptive sky grid asks for ~(f0 Tobs v/c)^2 nodes per peak: at 6 mo that\n"
     "# is 64 (< 4 mHz) / 256 (8 mHz) / 1024 (15 mHz); at 9 mo 128 / 512 / 2048 and\n"
     "# at 1 yr 256 / 1024 / 4096, on top of the fdot axis (13 -> 28 -> 48 nodes at\n"
     "# 8 mHz) and the f0 axis (79 -> 96). The 6mo stacked grid was ~12 GB raw and\n"
     "# 24-35 GB resident per GPU, so uncapped the 9mo grid lands at 2-3x that and\n"
     "# the 1yr at ~8x (the 1yr v8 OOMed on its first F-stat search). 512 never\n"
     "# touches a peak below 8 mHz at 9 mo (their natural counts are <= 512) and\n"
     "# bounds the > 8 mHz tail; the host-mapped float32 grid (TODO) is the real\n"
     "# fix, and at 1 yr the fdot axis alone needs it. FSTAT_STAGEB_NSKY_MAX=0 on\n"
     "# the line restores the uncapped grid.\n"
     "export FSTAT_STAGEB_NSKY_MAX=${FSTAT_STAGEB_NSKY_MAX:-512}\n", 1),
    # the stage-B fit's per-group transient (pre-launch check 2026-10-09; Mike:
    # "adjust the FSTAT a little bit if you can for memory ... no changes that
    # sacrifice needed accuracy"): the grouping only batches the evaluation
    ("export FSTAT_STAGEB_GROUP_MAX_GB=2.0\n",
     "export FSTAT_STAGEB_GROUP_MAX_GB=${FSTAT_STAGEB_GROUP_MAX_GB:-1.0}   # 9mo/1yr: halve the stage-B per-group transient (batching only, no grid change; 2.0 at 6mo)\n", 1),
)

#: THE 6MO RUN ON THE 9MO / 1YR MONITOR PAGE (user request 2026-10-12): the one
#: monitor-only entry BOTH generators append to their tables, verbatim. The 6mo
#: launcher never adds itself.
ADD_INFO_DEFAULT = "/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu"
_MONITOR_ADD_INFO_REPLACEMENTS = (
    ("export GF_MONITOR_TIMEOUT=1800\n# Requires MOJITO_INFO_PATH, exported above.\n",
     "export GF_MONITOR_TIMEOUT=1800\n# Requires MOJITO_INFO_PATH, exported above.\n"
     "# ---- 9MO / 1YR: THE 6MO RUN ON THIS RUN'S MONITOR PAGE (user request 2026-10-12)\n"
     "# \"add a second run to the same html only in specific places where it is\n"
     "# reasonable visually to compare between runs ... plot the phase maximized\n"
     "# overlap plots, a comparison of the final psd and foreground measurements\n"
     "# (final for the previous run and current for the ongoing run)\". The page stays\n"
     "# this run's; each run directory listed here (':'-joined) is read at ITS last\n"
     "# stored row and drawn onto the overlap CDF / survival, recovered-vs-SNR and\n"
     "# recovered-vs-frequency figures (matched against its own gb_truth_*.npz) and\n"
     "# the noise + foreground posteriors and curves. A missing dir / store / truth\n"
     "# set costs only its overlay (a caption note + a MISSING line). The monitor rank\n"
     "# inherits it; the CLI twin is python -m lisatools.globalfit.monitor --add-info.\n"
     "# GF_MONITOR_ADD_INFO=none on the launch line drops it.\n"
     "export GF_MONITOR_ADD_INFO=${GF_MONITOR_ADD_INFO:-" + ADD_INFO_DEFAULT + "}\n", 1),
)

#: the recipe as the 1yr / 9mo launchers resolve it (and the 6mo's values)
_RECIPE_EXPORTS = {
    "GB_SEARCH_SEED_ITERS": ("3", "5"),
    "GB_SEARCH_STAGES": ("1,2", None),
    "GB_SEARCH_1_SOURCE_EVERY": ("1", None),   # 2026-10-10 ruling (was 0)
    "GB_SEARCH_2_SOURCE_EVERY": ("1", None),
    "GB_SEARCH_2_NOISE_MODE": ("cycle_end", None),
    "GB_SEARCH_2_RESET_VALVES": ("1", None),
    "STAGE_REPLICA_PE": ("0", "1"),
    # the fixed-dim source seed (2026-10-08)
    "STAGE_SKIP_SOURCE_SEARCH": ("0", "1"),
    "SOURCE_SEARCH_POSITION": ("after_seed", None),
    "SOURCE_SEARCH_CHECKS": ("10", None),
    "SOURCE_SEARCH_TOL": ("10", None),
    "SOURCE_SEARCH_MAX_ROUNDS": ("200", None),
    # the galfor frequency prior (2026-10-08): fk / f_1 / f_2 in one 0.1-10 mHz box
    "GALFOR_FK_PRIOR": ("0.8e-3,1e-2", None),
    "GALFOR_F1_PRIOR": ("1e-4,1e-2", None),
    "GALFOR_F2_PRIOR": ("0.8e-3,1e-2", None),
    # the source warm start from the 6mo cold chain (2026-10-08); the STORE
    # export is ${K-${GF_SEED_STORE}} and is asserted separately
    "SOURCE_WARM_START_SNR_MIN": ("10", None),
    "MBH_SOURCE_WARM_START_IDS": ("2,5,16,18", None),
    "EMRI_SOURCE_WARM_START_IDS": ("0,1,2,3,4,5,6,7", None),
    "SOBBH_SOURCE_WARM_START_IDS": ("0,1,2,3,4,5", None),
    # the PSD reference fit: the unequal-arm pair on a fresh store (2026-10-08)
    "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM": ("1", "0"),
    # the stage-B F-stat sky cap (2026-10-09): bounds the > 8 mHz tail at 9 mo / 1 yr
    "FSTAT_STAGEB_NSKY_MAX": ("512", "0"),
    "FSTAT_STAGEB_GROUP_MAX_GB": ("1.0", "2.0"),
}


def _assert_the_recipe(test, exports, six, text):
    """The 2026-10-07 recipe, resolved; the 6mo keeps today's; no ratchet; and
    every new knob yields to the launch line (``${K:-v}``) except the seed
    count, which keeps the 6mo line's hard form."""
    for knob, (want, six_want) in _RECIPE_EXPORTS.items():
        test.assertEqual(exports.get(knob), want, knob)
        test.assertEqual(six.get(knob), six_want, knob)
        if knob != "GB_SEARCH_SEED_ITERS":
            test.assertRegex(text, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}", knob)
    test.assertEqual(exports["GALFOR_RATCHET"], "0")
    # the source warm start is OFF by default since 2026-10-22 (Mike: the non-GB
    # sources always start at the catalogue injection); opt in on the launch line
    test.assertEqual(exports["SOURCE_WARM_START_STORE"], "")
    test.assertNotIn("SOURCE_WARM_START_STORE", six)
    test.assertEqual(exports["GB_SEARCH_SOURCE_EVERY"], six["GB_SEARCH_SOURCE_EVERY"])
    for knob in exports:
        if knob.startswith("GB_SEARCH_3_") or knob == "GB_SEARCH_SAMPLE_NOISE_ALL_STAGES":
            # inherited from the 6mo and inert without stage 3 -- but never NEW
            test.assertEqual(exports[knob], six.get(knob), knob)


class OneYearV9TwinTest(unittest.TestCase):
    """``submit_gf_1yr_v9.sh`` IS ``submit_gf_6mo_v9_4gpu.sh`` at 12 months.

    User ruling 2026-10-03: "identical runs with the exception of changing
    the usual Tobs changes ... the most minimal amount of changes needed for
    the waveforms and other areas due to 12 mo instead of 6". The guard is
    byte-level: the 1yr file must equal the 6mo file with EXACTLY the
    replacements below applied (plus its own comment header), so any fix that
    lands in one script and not the other, or any extra knob, fails here.
    The three waveform windows (MBH / EMRI / SOBBH, 2026-10-03) declared NO
    export-line changes at 1 yr beyond MBHB_IDS and the preflights' grid.
    Since 2026-10-07 the table also carries the 9mo / 1yr RECIPE
    (``_RECIPE_REPLACEMENTS``: seed x3 -> gb_search_1 -> gb_search_2 ->
    full_pe), the one declared delta that is not a Tobs change.
    """

    # (6mo text, 1yr text, occurrences) -- the generator's table, verbatim
    REPLACEMENTS = (
        ("#SBATCH --job-name=gf6mo_v9_4gpu     # job name",
         "#SBATCH --job-name=gf1yr_v9          # job name", 1),
        ("/shared/data/global_fit_output/gf6mo_v9_4gpu_%j.log",
         "/shared/data/global_fit_output/gf1yr_v9_%j.log", 1),
        ("SLURM_LOG=/shared/data/global_fit_output/gf6mo_v9_4gpu_${SLURM_JOB_ID:-manual}.log",
         "SLURM_LOG=/shared/data/global_fit_output/gf1yr_v9_${SLURM_JOB_ID:-manual}.log", 1),
        ("STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/}",
         "STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_1yr_v9/}", 1),
        ("export BASE_FILE_NAME=gf_prod_6mo\n", "export BASE_FILE_NAME=gf_prod_1yr\n", 1),
        ("export TOBS_TARGET=15552000        # 180 d; grid resolves Nf 1440 x Nt 4320 x dt 2.5 (exact factor-2 of 3 mo in Nt)",
         "export TOBS_TARGET=31104000        # 360 d; grid resolves Nf 1440 x Nt 8640 x dt 2.5 (exact factor-2 of 6 mo in Nt, as 6 mo was of 3 mo)", 1),
        ("export SIGHET_NT_LAYER=120\n", "export SIGHET_NT_LAYER=240\n", 1),
        ("export GB_NLEAVES_MAX=15000        # 6 mo: deeper confusion resolved; 3-mo ran 10000",
         "export GB_NLEAVES_MAX=20000        # 1 yr: deeper confusion resolved again; 6-mo ran 15000, 3-mo 10000", 1),
        ("export GB_N_SUBBANDS=8192   # PER GPU; total = x n_gpus. Slab ~0.5 MB/slot",
         "export GB_N_SUBBANDS=4096   # PER GPU; total = x n_gpus. Slab ~1.0 MB/slot at 1 yr", 1),
        ("export GB_RJ_INMODEL_CHUNK=32768  # byte-parity with the 3mo twin's 65536 (6mo cells ~2x bytes); floored to ntemps multiples by the column-atomic staging",
         "export GB_RJ_INMODEL_CHUNK=16384  # byte-parity with the 6mo 32768 (1yr cells ~2x bytes); floored to ntemps multiples by the column-atomic staging", 1),
        ("export MBHB_IDS=2,5,16,18          # t_c 173.3 / 104.7 / 111.4 / 92.0 d",
         "export MBHB_IDS=0,2,3,4,5,7,9,12,15,16,18   # t_c 300.4/173.3/336.8/286.4/104.7/263.8/285.9/243.8/318.0/111.4/92.0 d (MBH session 2026-10-03). src 10 (2.8e6 Msun, SNR~1963) merges at 369.5 d, 9.5 d past the window, so the 7-d MBH_MERGER_TIME_BUFFER leaves it out; to model its in-window inspiral add 10 here and export MBH_MERGER_TIME_BUFFER=1209600", 1),
        ("make_factory(1440, 4320)", "make_factory(1440, 8640)", 2),
        ("Nf 1440 x Nt 4320 at dt 2.5 s", "Nf 1440 x Nt 8640 at dt 2.5 s", 3),
        ("${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz}",
         "${STORE_DIR}/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz}", 1),
        ("GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5}",
         "GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5}", 1),
        ("GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-7776000}",
         "GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-15552000}   # the 6-month parent", 1),
        ("export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-10}\n",
         "export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-300}   # 9mo/1yr (Mike 2026-10-08): the last 300 stored cold-chain rows x 4 walkers = 1,200 samples of the 6mo full_pe (~2.9M leaf rows into the fit; build the npz by hand on a login node if the in-job build is slow)\n", 1),
    ) + _RECIPE_REPLACEMENTS + _MONITOR_ADD_INFO_REPLACEMENTS

    def setUp(self):
        self.six_text = open(SIX_MO_V9).read()
        self.one_text = open(ONE_YR_V9).read()
        self.six = _exports(SIX_MO_V9)
        self.one = _exports(ONE_YR_V9)

    def test_the_1yr_script_is_the_6mo_script_plus_exactly_these_replacements(self):
        body = self.six_text
        self.assertTrue(body.startswith("#!/bin/bash\n"))
        body = body[len("#!/bin/bash\n"):]
        for old, new, n in self.REPLACEMENTS:
            self.assertEqual(body.count(old), n, old)
            body = body.replace(old, new)
        # the 1yr file = shebang + its own comment-only header + that body
        self.assertTrue(self.one_text.startswith("#!/bin/bash\n"))
        self.assertTrue(self.one_text.endswith(body),
                        "the 1yr script drifted from the 6mo script beyond the "
                        "declared Tobs replacements (diff the two files)")
        header = self.one_text[:-len(body)]
        for line in header.splitlines():
            self.assertTrue(line.startswith("#") or line == "",
                            f"non-comment line in the 1yr header: {line!r}")

    def test_tobs_and_its_derived_settings_are_the_1yr_values(self):
        self.assertEqual(self.one["TOBS_TARGET"], "31104000")
        self.assertEqual(self.one["SIGHET_NT_LAYER"], "240")
        self.assertEqual(self.one["GB_NLEAVES_MAX"], "20000")
        self.assertEqual(self.one["GB_N_SUBBANDS"], "4096")
        self.assertEqual(self.one["GB_RJ_INMODEL_CHUNK"], "16384")
        self.assertEqual(self.one["BASE_FILE_NAME"], "gf_prod_1yr")
        # STORE_DIR is a plain (non-export) assignment the resolver skips:
        # pin the line itself
        self.assertIn(
            "\nSTORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_1yr_v9/}\n",
            self.one_text)
        self.assertNotIn("gf_prod_6mo_v9_4gpu/}", self.one_text)
        # the warm start + noise pin come from the 6mo v9 parent, together
        self.assertEqual(
            self.one["GF_SEED_STORE"],
            "/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5")
        self.assertEqual(self.one["GB_WARM_START_SOURCE_STORE"], self.one["GF_SEED_STORE"])
        self.assertEqual(self.one["GB_WARM_START_SOURCE_TOBS"], "15552000")
        self.assertEqual(self.one["GB_WARM_START_LAST_K"], "300")   # 1,200 samples (2026-10-08)
        self.assertEqual(self.six["GB_WARM_START_LAST_K"], "10")
        self.assertTrue(self.one["GB_WARM_START_COMPONENTS"].endswith(
            "/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz"),
            self.one["GB_WARM_START_COMPONENTS"])

    def test_the_sources_inside_360_days(self):
        # MBH session 2026-10-03: 11 MBHBs merge inside 360 d + the 7 d buffer;
        # src 10 (369.5 d) is left out. EMRI / SOBHB sets unchanged.
        self.assertEqual(self.one["MBHB_IDS"], "0,2,3,4,5,7,9,12,15,16,18")
        self.assertEqual(self.one["EMRI_IDS"], self.six["EMRI_IDS"])
        self.assertEqual(self.one["SOBHB_IDS"], self.six["SOBHB_IDS"])
        self.assertNotIn("MBH_MERGER_TIME_BUFFER", self.one)

    def test_the_preflights_name_the_1yr_grid(self):
        # the MBH + EMRI resolvers, plus the header's 1YR-4 note
        self.assertEqual(self.one_text.count("make_factory(1440, 8640)"), 3)
        self.assertNotIn("make_factory(1440, 4320)", self.one_text)
        # the three "this run's grid" preflight comments follow the resolvers;
        # the 6mo lineage comments (sig-het stride derivation etc.) keep their
        # historical 4320 verbatim, as the byte-identity test requires
        self.assertEqual(self.one_text.count("Nf 1440 x Nt 8640 at dt 2.5 s"), 3)
        self.assertNotIn("Nf 1440 x Nt 4320 at dt 2.5 s", self.one_text)

    def test_every_other_export_is_identical(self):
        # (STORE_DIR is a plain assignment the resolver skips, and
        # FILE_STORE_DIR=${STORE_DIR} therefore resolves empty in both files;
        # the STORE_DIR line is pinned as text above.)
        allowed = {
            "TOBS_TARGET", "SIGHET_NT_LAYER", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "BASE_FILE_NAME",
            "MBHB_IDS", "GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE",
            "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_COMPONENTS",
            # 2026-10-08: the warm-start fit takes the last 300 rows x 4
            # walkers = 1,200 samples of the 6mo (the 6mo took 10 rows of the 3mo)
            "GB_WARM_START_LAST_K",
            "SOURCE_WARM_START_STORE",      # 2026-10-08: the source warm start (empty = off since 2026-10-22)
            "GF_MONITOR_ADD_INFO",          # 2026-10-12: the 6mo run on this run's page (1YR-10)
        } | set(_RECIPE_EXPORTS)          # the 2026-10-07 recipe (1YR-6)
        keys = (set(self.one) | set(self.six)) - {"_", "SHLVL", "PWD"}
        drift = {k: (self.six.get(k), self.one.get(k))
                 for k in sorted(keys - allowed)
                 if self.six.get(k) != self.one.get(k)}
        self.assertEqual(drift, {}, f"undeclared 6mo -> 1yr drift: {drift}")
        # and every declared knob really differs (a stale allowlist hides drift)
        same = sorted(k for k in allowed if self.six.get(k) == self.one.get(k))
        self.assertEqual(same, [], f"declared deltas that do not differ: {same}")

    def test_the_recipe_is_the_2026_10_07_ruling(self):
        _assert_the_recipe(self, self.one, self.six, self.one_text)

    def test_the_name_carries_no_gpu_count(self):
        # user ruling 2026-10-03: "I do not want to add '_4gpu' because ... I may
        # use more later" -- NGPUS picks the layout at launch, as on the 6mo
        self.assertEqual(os.path.basename(ONE_YR_V9), "submit_gf_1yr_v9.sh")
        self.assertIn("#SBATCH --job-name=gf1yr_v9", self.one_text)
        self.assertIn("/shared/data/global_fit_output/gf1yr_v9_%j.log", self.one_text)
        self.assertNotIn("gf1yr_v9_4gpu", self.one_text)


class NineMonthV9TwinTest(unittest.TestCase):
    """``submit_gf_9mo_v9.sh`` IS ``submit_gf_1yr_v9.sh`` at 9 months.

    User ruling 2026-10-07: "run a 9mo run instead of 1yr ... keep all the
    accuracy settings for the 1yr run the same ... keep everything the same
    as the 1 yr run except for any specific necessary changes due to the
    observation". Two guards. Byte-level against the 6mo script: the 9mo
    file must equal the 6mo file with EXACTLY the replacements below applied
    (plus its own comment-only header), so any fix that lands in one script
    and not the other fails here. Export-level against the 1yr script: only
    the Tobs-driven knobs (TOBS_TARGET, SIGHET_NT_LAYER, BASE_FILE_NAME,
    MBHB_IDS, MBH_MERGER_TIME_BUFFER) may resolve differently -- the 1yr
    capacity sizing (GB_NLEAVES_MAX, GB_N_SUBBANDS, GB_RJ_INMODEL_CHUNK) is
    taken verbatim, as ruled, not re-derived for 270 d.
    """

    # (6mo text, 9mo text, occurrences) -- the generator's table, verbatim
    REPLACEMENTS = (
        ("#SBATCH --job-name=gf6mo_v9_4gpu     # job name",
         "#SBATCH --job-name=gf9mo_v9          # job name", 1),
        ("/shared/data/global_fit_output/gf6mo_v9_4gpu_%j.log",
         "/shared/data/global_fit_output/gf9mo_v9_%j.log", 1),
        ("SLURM_LOG=/shared/data/global_fit_output/gf6mo_v9_4gpu_${SLURM_JOB_ID:-manual}.log",
         "SLURM_LOG=/shared/data/global_fit_output/gf9mo_v9_${SLURM_JOB_ID:-manual}.log", 1),
        ("STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/}",
         "STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_9mo_v9/}", 1),
        ("export BASE_FILE_NAME=gf_prod_6mo\n", "export BASE_FILE_NAME=gf_prod_9mo\n", 1),
        ("export TOBS_TARGET=15552000        # 180 d; grid resolves Nf 1440 x Nt 4320 x dt 2.5 (exact factor-2 of 3 mo in Nt)",
         "export TOBS_TARGET=23328000        # 270 d; grid resolves Nf 1440 x Nt 6480 x dt 2.5 (1.5x the 6 mo Nt; the 1yr runs Nt 8640)", 1),
        ("export SIGHET_NT_LAYER=120\n", "export SIGHET_NT_LAYER=180\n", 1),
        ("export GB_NLEAVES_MAX=15000        # 6 mo: deeper confusion resolved; 3-mo ran 10000",
         "export GB_NLEAVES_MAX=20000        # 9 mo: the 1yr value (user ruling 2026-10-07, 1yr settings); 6-mo ran 15000, 3-mo 10000", 1),
        ("export GB_N_SUBBANDS=8192   # PER GPU; total = x n_gpus. Slab ~0.5 MB/slot",
         "export GB_N_SUBBANDS=4096   # PER GPU; total = x n_gpus. Slab ~0.75 MB/slot at 9 mo (the 1yr value, user ruling 2026-10-07)", 1),
        ("export GB_RJ_INMODEL_CHUNK=32768  # byte-parity with the 3mo twin's 65536 (6mo cells ~2x bytes); floored to ntemps multiples by the column-atomic staging",
         "export GB_RJ_INMODEL_CHUNK=16384  # the 1yr value (user ruling 2026-10-07, 1yr settings; 9mo cells ~1.5x the 6mo bytes); floored to ntemps multiples by the column-atomic staging", 1),
        ("export MBHB_IDS=2,5,16,18          # t_c 173.3 / 104.7 / 111.4 / 92.0 d",
         "export MBHB_IDS=2,5,7,12,16,18     # t_c 173.3/104.7/263.8/243.8/111.4/92.0 d (MBH session 2026-10-03 table): every catalogue MBHB merging inside 270 d + the 14-d MBH_MERGER_TIME_BUFFER below. None merges in (270, 284] d; the next are srcs 9 (285.9 d) and 4 (286.4 d), 15.9 / 16.4 d past the end, then 0 (300.4), 15 (318.0), 3 (336.8), 10 (369.5)\n"
         "# User rule 2026-10-07: keep any MBHB merging within 2 weeks of the end of the\n"
         "# observation (its in-window inspiral is modelled, cut at the data end by the\n"
         "# batched window's active-box clamp and the stock generator's data lattice).\n"
         "# 14 d replaces the code default of 7 d; the t_plunge prior's upper edge\n"
         "# follows it (obs_end + buffer + t_plunge_pad).\n"
         "export MBH_MERGER_TIME_BUFFER=1209600   # 14 d", 1),
        ("make_factory(1440, 4320)", "make_factory(1440, 6480)", 2),
        ("Nf 1440 x Nt 4320 at dt 2.5 s", "Nf 1440 x Nt 6480 at dt 2.5 s", 3),
        ("${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz}",
         "${STORE_DIR}/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz}", 1),
        ("GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5}",
         "GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5}", 1),
        ("GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-7776000}",
         "GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-15552000}   # the 6-month parent", 1),
        ("export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-10}\n",
         "export GB_WARM_START_LAST_K=${GB_WARM_START_LAST_K:-300}   # 9mo/1yr (Mike 2026-10-08): the last 300 stored cold-chain rows x 4 walkers = 1,200 samples of the 6mo full_pe (~2.9M leaf rows into the fit; build the npz by hand on a login node if the in-job build is slow)\n", 1),
    ) + _RECIPE_REPLACEMENTS + _MONITOR_ADD_INFO_REPLACEMENTS

    # the only exports that may resolve differently from the 1yr script
    TOBS_DRIVEN = {"TOBS_TARGET", "SIGHET_NT_LAYER", "BASE_FILE_NAME",
                   "MBHB_IDS", "MBH_MERGER_TIME_BUFFER"}

    def setUp(self):
        self.six_text = open(SIX_MO_V9).read()
        self.nine_text = open(NINE_MO_V9).read()
        self.six = _exports(SIX_MO_V9)
        self.one = _exports(ONE_YR_V9)
        self.nine = _exports(NINE_MO_V9)

    def test_the_9mo_script_is_the_6mo_script_plus_exactly_these_replacements(self):
        body = self.six_text
        self.assertTrue(body.startswith("#!/bin/bash\n"))
        body = body[len("#!/bin/bash\n"):]
        for old, new, n in self.REPLACEMENTS:
            self.assertEqual(body.count(old), n, old)
            body = body.replace(old, new)
        self.assertTrue(self.nine_text.startswith("#!/bin/bash\n"))
        self.assertTrue(self.nine_text.endswith(body),
                        "the 9mo script drifted from the 6mo script beyond the "
                        "declared Tobs replacements (diff the two files, or "
                        "re-run derive_9mo_v9_launcher.py after a 6mo edit)")
        header = self.nine_text[:-len(body)]
        for line in header.splitlines():
            self.assertTrue(line.startswith("#") or line == "",
                            f"non-comment line in the 9mo header: {line!r}")

    def test_tobs_and_its_derived_settings_are_the_9mo_values(self):
        # 270 d: WDMSettings.adjust_to_even_bins -> Nf 1440 x Nt 6480 x dt 2.5
        # exactly (checked 2026-10-07); the sig-het stride keeps the 36-h
        # parity (23328000 / 180 = 129600 s) and 180 divides 6480.
        self.assertEqual(self.nine["TOBS_TARGET"], "23328000")
        self.assertEqual(int(self.nine["TOBS_TARGET"]), 1440 * 6480 * 25 // 10)
        self.assertEqual(self.nine["SIGHET_NT_LAYER"], "180")
        self.assertEqual(int(self.nine["TOBS_TARGET"]) // int(self.nine["SIGHET_NT_LAYER"]),
                         36 * 3600)
        self.assertEqual(6480 % int(self.nine["SIGHET_NT_LAYER"]), 0)
        self.assertEqual(self.nine["BASE_FILE_NAME"], "gf_prod_9mo")
        self.assertIn(
            "\nSTORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_9mo_v9/}\n",
            self.nine_text)
        self.assertNotIn("gf_prod_6mo_v9_4gpu/}", self.nine_text)
        self.assertNotIn("gf_prod_1yr_v9", self.nine_text)
        # the warm start + noise pin come from the 6mo v9 parent, as the 1yr's
        self.assertEqual(
            self.nine["GF_SEED_STORE"],
            "/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5")
        self.assertEqual(self.nine["GB_WARM_START_SOURCE_STORE"], self.nine["GF_SEED_STORE"])
        self.assertEqual(self.nine["GB_WARM_START_SOURCE_TOBS"], "15552000")
        # user ruling 2026-10-08: "the last 300 samples ... so 1200 total" =
        # 300 stored cold-chain rows x the 6mo's 4 walkers
        self.assertEqual(self.nine["GB_WARM_START_LAST_K"], "300")
        self.assertEqual(self.nine["GB_WARM_START_LAST_K"], self.one["GB_WARM_START_LAST_K"])
        self.assertTrue(self.nine["GB_WARM_START_COMPONENTS"].endswith(
            "/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz"),
            self.nine["GB_WARM_START_COMPONENTS"])

    def test_the_1yr_capacity_knobs_are_taken_verbatim(self):
        # the ruling: the 1yr values, not a 270-d re-derivation (and not the
        # 6mo values either)
        for knob in ("GB_NLEAVES_MAX", "GB_N_SUBBANDS", "GB_RJ_INMODEL_CHUNK"):
            self.assertEqual(self.nine[knob], self.one[knob], knob)
            self.assertNotEqual(self.nine[knob], self.six[knob], knob)
        self.assertEqual(self.nine["GB_NLEAVES_MAX"], "20000")
        self.assertEqual(self.nine["GB_N_SUBBANDS"], "4096")
        self.assertEqual(self.nine["GB_RJ_INMODEL_CHUNK"], "16384")

    def test_the_sources_inside_270_days_plus_two_weeks(self):
        # MBH session table 2026-10-03: the catalogue MBHBs merging before
        # 367 d are the 1yr eleven; the 9mo set is those with t_c < 284 d.
        # Srcs 9 (285.9 d) and 4 (286.4 d) are the next and stay out.
        self.assertEqual(self.nine["MBHB_IDS"], "2,5,7,12,16,18")
        nine_ids = set(self.nine["MBHB_IDS"].split(","))
        one_ids = set(self.one["MBHB_IDS"].split(","))
        self.assertTrue(nine_ids <= one_ids, nine_ids - one_ids)
        self.assertTrue({"9", "4"} <= one_ids - nine_ids)
        # user rule 2026-10-07: a merger up to two weeks past the end is kept
        self.assertEqual(self.nine["MBH_MERGER_TIME_BUFFER"], str(14 * 86400))
        self.assertNotIn("MBH_MERGER_TIME_BUFFER", self.one)
        self.assertNotIn("MBH_MERGER_TIME_BUFFER", self.six)
        self.assertEqual(self.nine["EMRI_IDS"], self.six["EMRI_IDS"])
        self.assertEqual(self.nine["SOBHB_IDS"], self.six["SOBHB_IDS"])

    def test_the_preflights_name_the_9mo_grid(self):
        # the MBH + EMRI resolvers, plus the header's 9MO-4 note
        self.assertEqual(self.nine_text.count("make_factory(1440, 6480)"), 3)
        self.assertNotIn("make_factory(1440, 4320)", self.nine_text)
        self.assertNotIn("make_factory(1440, 8640)", self.nine_text)
        self.assertEqual(self.nine_text.count("Nf 1440 x Nt 6480 at dt 2.5 s"), 3)
        self.assertNotIn("Nf 1440 x Nt 4320 at dt 2.5 s", self.nine_text)

    def test_every_other_export_matches_the_1yr_script(self):
        # "keep everything the same as the 1 yr run except ... the
        # observation": every resolved export outside TOBS_DRIVEN is the
        # 1yr value (so the V9-30 defaults, the GB knobs, the waveform
        # knobs, the seed/warm start all agree)
        keys = (set(self.one) | set(self.nine)) - {"_", "SHLVL", "PWD"}
        drift = {k: (self.one.get(k), self.nine.get(k))
                 for k in sorted(keys - self.TOBS_DRIVEN)
                 if self.one.get(k) != self.nine.get(k)}
        self.assertEqual(drift, {}, f"undeclared 1yr -> 9mo drift: {drift}")
        same = sorted(k for k in self.TOBS_DRIVEN if self.one.get(k) == self.nine.get(k))
        self.assertEqual(same, [], f"declared deltas that do not differ: {same}")

    def test_the_recipe_is_the_2026_10_07_ruling(self):
        # the SAME recipe as the 1yr (the export-level 1yr identity above
        # already holds it; this names it)
        _assert_the_recipe(self, self.nine, self.six, self.nine_text)

    def test_the_name_carries_no_gpu_count(self):
        self.assertEqual(os.path.basename(NINE_MO_V9), "submit_gf_9mo_v9.sh")
        self.assertIn("#SBATCH --job-name=gf9mo_v9", self.nine_text)
        self.assertIn("/shared/data/global_fit_output/gf9mo_v9_%j.log", self.nine_text)
        self.assertNotIn("gf9mo_v9_4gpu", self.nine_text)


class MonitorAddInfoLauncherTest(unittest.TestCase):
    """The 6mo v9 run overlaid on the 9mo / 1yr monitor pages (user request
    2026-10-12: "we want to be able to add the 6mo v9 run ... something like
    --add-info /path/to/gf_prod_6mo_v9_4gpu"). The in-run monitor reads
    GF_MONITOR_ADD_INFO from the environment the launcher exports; the 6mo
    launcher never adds itself."""

    def test_the_9mo_and_1yr_overlay_the_6mo_run_and_the_line_can_change_it(self):
        six_exports = _exports(SIX_MO_V9)
        for path in (NINE_MO_V9, ONE_YR_V9):
            exports = _exports(path)
            src = open(path).read()
            with self.subTest(script=os.path.basename(path)):
                self.assertEqual(exports.get("GF_MONITOR_ADD_INFO"), ADD_INFO_DEFAULT)
                self.assertRegex(src, r"(?m)^export GF_MONITOR_ADD_INFO="
                                      r"\$\{GF_MONITOR_ADD_INFO:-/shared/data/"
                                      r"global_fit_output/gf_prod_6mo_v9_4gpu\}$")
                self.assertEqual(src.count("export GF_MONITOR_ADD_INFO="), 1)
        self.assertNotIn("GF_MONITOR_ADD_INFO", six_exports)
        six = open(SIX_MO_V9).read()
        self.assertNotIn("GF_MONITOR_ADD_INFO", six)
        # the default IS the 6mo launcher's store directory
        self.assertIn("STORE_DIR=${STORE_DIR:-" + ADD_INFO_DEFAULT + "/}", six)

    def test_a_launch_line_value_wins(self):
        src = open(NINE_MO_V9).read()
        line = next(l for l in src.split("\n") if l.startswith("export GF_MONITOR_ADD_INFO="))
        env = {k: v for k, v in os.environ.items() if not k.isupper()}
        env["GF_MONITOR_ADD_INFO"] = "/a/run:/b/run"
        out = subprocess.run(["bash", "-c", line + "\necho \"$GF_MONITOR_ADD_INFO\""],
                             capture_output=True, text=True, env=env).stdout.strip()
        self.assertEqual(out, "/a/run:/b/run")

    def test_the_knob_is_the_monitors_own_name(self):
        from lisatools.globalfit.monitor import _add_info

        self.assertEqual(_add_info.ENV, "GF_MONITOR_ADD_INFO")


class PeTwoFstatMovesLauncherTest(unittest.TestCase):
    """The two PE F-stat moves (user ruling 2026-10-12) on every v9 launcher.

    "I would like to do two fstat moves that will split their share of the
    total moves. One fstat move fits the GB-free residual so the whole galaxy
    down to SNR 6.5. [...] The second fstat move will fit the gb-inclusive
    residual every 150 iterations." -- "set the gb-free residual refit (with
    the full galaxy found) to not refitting during full pe. It should just fit
    once the first time it is visiting in full pe."

    Exported explicitly in the 6mo derivation source, carried verbatim into
    the 1yr / 9mo (derive_*), mirrored in the 3mo twin.
    """

    WANT = {
        "GB_FSTAT_REFIT_EVERY_PE": "150",          # now the RESIDUAL move's
        "GB_FSTAT_REFIT_EVERY_PE_GB_FREE": "0",    # the galaxy: fit once
        "GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE": "6.5",
        # 2026-10-18 (Mike): split 0.5 -> 0.9 and warm 0.1 -> 0.4 -- on the
        # 9mo job 773 the galaxy grid accepted 0.4 cold births per draw, the
        # residual grid 2.5, the prior 2.5 and the warm self refit 18.6
        "GB_PE_RJ_FSTAT_RESIDUAL_SPLIT": "0.9",
        # the F-stat total the split divides is unchanged
        "GB_PE_RJ_FSTAT_FRACTION": "0.45",
        "GB_PE_RJ_WARM_FRACTION": "0.4",
    }

    def test_every_v9_launcher_exports_the_ruling(self):
        for path in (SIX_MO_V9, ONE_YR_V9, NINE_MO_V9, THREE_MO_V9):
            exports = _exports(path)
            for knob, want in self.WANT.items():
                if path == THREE_MO_V9 and knob in ("GB_PE_RJ_FSTAT_FRACTION",
                                                    "GB_PE_RJ_WARM_FRACTION"):
                    continue    # the 3mo twin's pre-existing PE draw (its drift list)
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)

    def test_the_ruling_is_quoted_in_the_derivation_source(self):
        src = open(SIX_MO_V9).read()
        for words in ("two fstat moves that will split their share",
                      "fit once the first time it is visiting in full pe",
                      "just specific to the one walker that was",
                      "rj_fstat_pe_residual"):
            self.assertIn(words, src)

    def test_the_split_yields_to_the_launch_line_and_the_galaxy_knobs_do_not(self):
        # the split is the GB_PE_RJ_FSTAT_FRACTION sibling (${K:-v}); the
        # galaxy cadence / floor are hard, like GB_FSTAT_REFIT_EVERY_PE
        src = open(NINE_MO_V9).read()
        self.assertRegex(src, r"(?m)^export GB_PE_RJ_FSTAT_RESIDUAL_SPLIT="
                              r"\$\{GB_PE_RJ_FSTAT_RESIDUAL_SPLIT:-0\.9\}$")
        self.assertRegex(src, r"(?m)^export GB_FSTAT_REFIT_EVERY_PE_GB_FREE=0$")
        self.assertRegex(src, r"(?m)^export GB_FSTAT_PEAK_MIN_SNR_PE_GB_FREE=6\.5$")


class V930ProductionDefaultsTest(unittest.TestCase):
    """V9-30 (user ruling 2026-10-07): Mike's 6mo production line

        GB_FSTAT_FORCE_REFIT=1 GB_FSTAT_PE_REF=max PSD_EIGEN_INFO=fisher
        GALFOR_EIGEN_INFO=fisher SOBBH_EIGEN_INFO=gram EMRI_EIGEN_INFO=gram
        MBH_EIGEN_INFO=gram SOBBH_CHECK_LL=0 MBH_CHECK_LL=0 EMRI_CHECK_LL=0
        MIDIT_CHECKPOINT=0 MBH_NTEMPS=8 EMRI_NTEMPS=8 EMRI_TRAJ_WORKERS=8
        MBH_WINDOW_DECIMATE=2 NGPUS=4 ./submit_gf_6mo_v9_4gpu.sh

    is the launcher DEFAULT in the 6mo script and its 1yr / 9mo twins --
    every knob still in the ``${K:-v}`` form so the line can flip it back --
    EXCEPT the two F-stat refit knobs: GB_FSTAT_FORCE_REFIT and
    GB_FSTAT_PE_REF stay one-off line knobs (the PE refit walker keeps the
    code default, random). NGPUS=4 is pinned by the dispatch test
    ``test_the_v9_production_launchers_default_to_four_gpus``.
    """

    LINE = {
        "MIDIT_CHECKPOINT": "0",
        # 2026-10-18 (the EMRI sampling audit): EMRI 8 -> 12 rungs (T_max 625)
        "MBH_NTEMPS": "8", "EMRI_NTEMPS": "12",
        "EMRI_TRAJ_WORKERS": "8",
        "MBH_WINDOW_DECIMATE": "2",
        "SOBBH_EIGEN_INFO": "gram", "EMRI_EIGEN_INFO": "gram", "MBH_EIGEN_INFO": "gram",
        "PSD_EIGEN_INFO": "fisher", "GALFOR_EIGEN_INFO": "fisher",
        "SOBBH_CHECK_LL": "0", "MBH_CHECK_LL": "0", "EMRI_CHECK_LL": "0",
        # not on the line; added the same day by ruling ("keep
        # GF_PERSIST_STAGE_START, but it should reset at the end of a stage")
        "GF_PERSIST_STAGE_START": "1",
        # not on the line; ruling 2026-10-07 "Set these values in the v9
        # script" -- the GB ladder re-spacing, effective on a fresh store
        "GB_LADDER_HOT_RATIO": "1.35", "GB_LADDER_HOT_FROM": "10",
        "GB_LADDER_PIN_LAST": "0",
        # ruling 2026-10-08: "update them now with fast waveforms to do the
        # same thing as the sobhbs psd and galfor" (was 100)
        "MBH_EIGEN_REFRESH": "10", "EMRI_EIGEN_REFRESH": "10",
    }

    def test_the_line_is_the_default_in_every_v9_production_launcher(self):
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in self.LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    # overridable: a hard `export K=v` would make the line
                    # reach nothing (the MIDIT_CHECKPOINT lesson)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}")

    def test_the_fstat_refit_knobs_are_not_defaults(self):
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            with self.subTest(script=os.path.basename(path)):
                for knob in ("GB_FSTAT_FORCE_REFIT", "GB_FSTAT_PE_REF"):
                    self.assertNotIn(knob, exports, knob)
                    self.assertNotRegex(src, rf"(?m)^export {knob}=", knob)


class EigenJumpMixLauncherTest(unittest.TestCase):
    """2026-10-13 (Mike, on the eigen jump mixture): "if this is going to
    improve sampling in general then lets try it. We can always rewind. but
    let's make sure it takes effect for EMRIs". The code default stays mix 0
    (bit-identical draws); the 6mo v9 launcher and its derived 1yr / 9mo
    twins export 10 % of the draws at 3 sigma for EMRI, MBH and SOBBH, each
    overridable from the launch line (the rollback is ``..._MIX=0``).

    Of the per-leaf ladder knobs of the same day, ``{BRANCH}_LADDER_RESET_ON_RESUME``
    is a one-off launch-line knob (a standing export would reset the ladders
    at every relaunch), ``SOBBH_LADDER_REANCHOR_NATS`` is exported at 60 (20
    from 2026-10-15 to 2026-10-18, when the 9mo's bimodal leaves showed the
    guard re-anchoring the other mass mode)
    (since 2026-10-15; the code default 200 before), ``MBH_LADDER_REANCHOR_NATS`` is exported OFF (0) and
    ``EMRI_LADDER_REANCHOR_NATS`` at 50 (100 since 2026-10-18, the EMRI
    sampling audit): the second ruling of 2026-10-13
    ("focus on sampling the main modes (+ the sky move for MBHs) in terms of
    tempering") after the guard was measured re-anchoring MBHB hot rungs that
    merely sat in a secondary sky mode, while 200 missed the lost rungs of
    every EMRI but the two loudest.
    """

    def test_the_block_is_on_in_every_v9_production_launcher(self):
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in EIGEN_JUMP_MIX_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")

    def test_the_ladder_knobs_are_not_exported(self):
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            with self.subTest(script=os.path.basename(path)):
                for b in ("EMRI", "MBH", "SOBBH"):
                    knob = f"{b}_LADDER_RESET_ON_RESUME"
                    self.assertNotIn(knob, exports, knob)
                    self.assertNotRegex(src, rf"(?m)^export {knob}=", knob)

    def test_the_ladder_caps_are_main_mode(self):
        """User ruling 2026-10-13 (the second of the day, after the T_max 100
        EMRI ladder was measured putting 5 of 8 rungs in the prior and the
        MBH guard re-anchoring hot rungs out of secondary sky modes): "I want
        to focus on sampling the main modes (+ the sky move for MBHs) in terms
        of tempering". EMRI T_max 6 (an SNR-15 EMRI keeps SNR 6 on the top
        rung), MBH T_max 10 (SNR 180-1000 keeps SNR 56-330), the MBH lost-rung
        guard off and the EMRI guard at 50 nats (a lost EMRI rung is SNR^2/2
        down, 72-1150; 200 only caught the two loudest); SOBBH joined on
        2026-10-15 (T_max 6, guard 20). Exported in every v9 production
        launcher, overridable on the line."""
        # 2026-10-18 (the EMRI sampling audit, item 6): EMRI guard 50 -> 100 --
        # the 9mo job 773 firings were the top rungs' own tail (72-100 nats on
        # leaves lost at 118-1262), not lost rungs
        # 2026-10-18 (Mike, with the audit): EMRI T_max 6 -> 625 on 12 rungs,
        # the same make_ladder spacing extended upward so the hot rungs can
        # find islands no walker holds (EMRI_NTEMPS 8 -> 12 in LINE above)
        want = {"EMRI_LADDER_TMAX": "625", "EMRI_LADDER_REANCHOR_NATS": "100",
                "MBH_LADDER_TMAX": "10", "MBH_LADDER_REANCHOR_NATS": "0",
                # 2026-10-15 (Mike, "update the SOBHBs"): the SOBBH ladder
                # capped like the EMRI one (its default make_ladder spacing
                # reached beta 0.007; hot pairs swapped at 0.8-0.95 = prior
                # sampling on the 1yr), guard 20 nats (a lost SNR-10 source).
                # 2026-10-18 (Mike, "do this for the other sources"): SOBBH
                # guard 20 -> 60 -- on the 9mo job 773 it fired 60 times on
                # the bimodal leaves 0/1, re-anchoring hot rungs in the OTHER
                # mass mode (45-55 nats below cold), not lost sources
                "SOBBH_LADDER_TMAX": "6", "SOBBH_LADDER_REANCHOR_NATS": "60"}
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            with self.subTest(script=os.path.basename(path)):
                for knob, val in want.items():
                    self.assertEqual(exports.get(knob), val, knob)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{val}\}}$", knob)

    def test_the_axis_adaptation_is_on_in_every_v9_production_launcher(self):
        """2026-10-14 (Mike: "some of the emris are sampling terribly"): the
        per-axis step-scale adaptation of the EMRI / MBH / SOBBH eigen tables
        is ON in every v9 production launcher ({BRANCH}_EIGEN_AXIS_ADAPT=1,
        overridable on the line); the code default stays OFF."""
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in EIGEN_AXIS_ADAPT_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")

    def test_the_pooled_tempered_capped_adaptation_is_on_in_every_v9_production_launcher(self):
        """2026-10-15 (Mike, "just do 1 for now" + the widening cap): the
        adaptation buffer is POOLED across ranks, the eigen step TEMPERED per
        rung and the adapted width capped at 100 x the Fisher sigma in every
        v9 production launcher ({BRANCH}_EIGEN_ADAPT_POOL=1,
        {BRANCH}_EIGEN_TEMPER_SIGMAS=1, {BRANCH}_EIGEN_ADAPT_MAX_FACTOR=100,
        overridable on the line); the code defaults stay off / off / 100."""
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in EIGEN_POOL_TEMPER_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")

    def test_the_emri_audit_block_is_on_in_every_v9_production_launcher(self):
        """2026-10-18 (Mike, the EMRI sampling audit; docs/emri-sampling-audit.md):
        the per-axis acceptance log ({BRANCH}_EIGEN_AXIS_ACC_LOG=1, all three
        source branches), the acceptance-driven per-axis step controller
        (EMRI_EIGEN_AXIS_ACC_TUNE=1, band 0.1-0.5, factor 2) and the
        cross-walker difference jump (EMRI_DE_JUMP_MIX=0.2, mode frac 0.5) in
        every v9 production launcher, overridable on the line. 2026-10-18
        (Mike, "do this for the other sources"): MBH / SOBBH get the
        controller too and the jump at 0.1 / 0.2, with the controller band and
        the mode fraction left at the code defaults (not exported)."""
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in EMRI_AUDIT_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")
            for knob in EMRI_AUDIT_NOT_EXPORTED:
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertNotIn(knob, exports)

    def test_the_mbh_proposal_frame_is_lisa_in_every_v9_production_launcher(self):
        """2026-10-19 (Mike, examining the MBHB sampling: "keep the same
        sampling/waveform parameters on the MBHB, but when we do the sampling
        moves, can we move in the LISA reference frame"): MBH_PROPOSAL_FRAME=lisa
        in every v9 production launcher, overridable on the line (rollback
        =sampling), placed in the MBH sky-hop block with the ruling and the
        9mo numbers; the code default stays "sampling" (bit-identical)."""
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in PROPOSAL_FRAME_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")
            block = src[src.index("# ---- MBH PROPOSALS IN THE LISA FRAME (2026-10-19)"):
                        src.index("export MBH_PROPOSAL_FRAME=")]
            for quote in ("Only the move that produces the", "304 -> 3.7",
                          "20-75x narrower", "linear 0.87-0.998"):
                with self.subTest(script=os.path.basename(path), quote=quote):
                    self.assertIn(quote, block)
            self.assertLess(src.index("export MBH_SKY_MOVE_LONG_WEIGHT="),
                            src.index("export MBH_PROPOSAL_FRAME="))
            for b in ("EMRI", "SOBBH"):
                self.assertNotIn(f"{b}_PROPOSAL_FRAME", exports)

    def test_the_warm_self_refit_block_is_on_in_every_v9_production_launcher(self):
        """2026-10-13 (Mike): the GB warm-start SELF refit -- every v9
        production launcher turns it on with the ruled settings (2026-10-15:
        built on the head at the warm moves' entry, every row in search and
        25 in full_pe, the 50-row table, splits 0,0,1 / 1,1,1, consensus
        0.95); each knob overridable on the line (rollback GB_WARM_SELF=0).
        The code default stays OFF so stock runs are bit-identical."""
        for path in V9_PRODUCTION:
            exports = _exports(path)
            src = open(path).read()
            for knob, want in WARM_SELF_LINE.items():
                with self.subTest(script=os.path.basename(path), knob=knob):
                    self.assertEqual(exports.get(knob), want)
                    self.assertRegex(
                        src, rf"(?m)^export {knob}=\$\{{{knob}:-{want}\}}$")

    def test_the_launch_line_can_switch_it_off(self):
        lines = [l for l in open(NINE_MO_V9).read().split("\n")
                 if re.match(r"^export [A-Z]+_EIGEN_JUMP_(MIX|FACTOR)=", l)]
        env = {k: v for k, v in os.environ.items() if not k.isupper()}
        env["EMRI_EIGEN_JUMP_MIX"] = "0"
        out = subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv | sort\n"],
                             capture_output=True, text=True, env=env).stdout
        got = dict(l.split("=", 1) for l in out.split("\n") if "=" in l)
        self.assertEqual(got["EMRI_EIGEN_JUMP_MIX"], "0")
        self.assertEqual(got["MBH_EIGEN_JUMP_MIX"], "0.1")


#: every launcher carrying the GF_GPU_ROUTING wiring (2026-10-15): the v9
#: production scripts, the hand-mirrored 3mo v9 twin, and the six flagship
#: scripts built from scripts/flagship/launcher.sh.in
FLAGSHIP_SCRIPTS = tuple(
    os.path.join(ROOT, "scripts", "flagship", f"submit_gf_{r}.sh")
    for r in ("3mo", "6mo", "9mo", "12mo", "18mo", "24mo"))
ROUTING_SCRIPTS = (*V9_PRODUCTION, THREE_MO_V9) + FLAGSHIP_SCRIPTS
_MONITOR_OR_FLAGSHIP = set(V9_PRODUCTION) | set(FLAGSHIP_SCRIPTS)


def _routing_ntasks(script, run_tasks):
    return f"--ntasks={run_tasks + (1 if script in _MONITOR_OR_FLAGSHIP else 0)}"


def _shell_block(text, start, end_marker):
    """``text[start : end]`` where ``end`` is the first ``end_marker`` after it."""
    i = text.index(start)
    return text[i: text.index(end_marker, i)]


class GpuRoutingDispatchTest(unittest.TestCase):
    """``NGPUS=8 GF_GPU_ROUTING=1`` on the 4-walker store (Mike, 2026-10-15).

    9mo job 764 ran ``NGPUS=8 GPUS_PER_RANK=2``: 4 compute ranks with two
    GPUs and ONE walker each, and died at the first GB gather on an empty
    second shard -- GPUS_PER_RANK packs walkers, it never splits one. The
    within-walker split is the replica routing: 8 compute ranks, one GPU
    each, 4 walker blocks x R = 2. The launchers now (a) carry the knob
    through ``--export`` and print the factorization, (b) never round NWALKERS
    with it on, (c) refuse the job-764 trap before ``sbatch``, and (d) leave
    every shape bit-identical with it unset.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="submit_routing_stub_")
        stub = os.path.join(cls._tmp.name, "sbatch")
        with open(stub, "w") as fh:
            fh.write(_STUB_SBATCH)
        os.chmod(stub, 0o755)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _run(self, script, overrides, rc=0):
        env = {k: v for k, v in os.environ.items() if k not in _DISPATCH_ENV_KEYS}
        env["PATH"] = self._tmp.name + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        r = subprocess.run(["bash", script], env=env, capture_output=True,
                           text=True, timeout=60)
        self.assertEqual(r.returncode, rc, f"{script} {overrides}:\n{r.stdout}\n{r.stderr}")
        return r.stdout.splitlines()

    @staticmethod
    def _export(lines):
        return [l for l in lines if l.startswith("--export=")]

    def test_ngpus_8_routing_is_four_blocks_of_two_replicas(self):
        for script in ROUTING_SCRIPTS:
            with self.subTest(script=os.path.basename(script)):
                lines = self._run(script, {"NGPUS": "8", "GF_GPU_ROUTING": "1"})
                # 8 compute ranks + the saver (+ the monitor rank), 4 nodes
                self.assertIn(_routing_ntasks(script, 9), lines)
                self.assertIn("--nodes=4", lines)
                self.assertIn("--gres=gpu:2", lines)
                (exp,) = self._export(lines)
                self.assertTrue(exp.endswith(",GF_GPU_ROUTING=1"), exp)
                text = "\n".join(lines)
                self.assertIn(
                    "replica factorization NWALKERS=4 on N_COMPUTE=8 -> n_blocks=4 "
                    "walker block(s) of 1 walker(s) x R=2 replica rank(s) per block, "
                    "one GPU per rank", text)
                self.assertIn("NWALKERS is NOT rounded", text)
                self.assertIn("GB/VGB by band range", text)

    def test_every_spelling_python_accepts_and_none_it_rejects(self):
        script = NINE_MO_V9
        for on in ("1", "true", "TRUE", "yes", "On", " on "):
            with self.subTest(value=on):
                lines = self._run(script, {"NGPUS": "8", "GF_GPU_ROUTING": on})
                self.assertTrue(self._export(lines)[0].endswith(",GF_GPU_ROUTING=1"))
        for off in ("0", "no", "off", "false", ""):
            with self.subTest(value=off):
                lines = self._run(script, {"NGPUS": "8", "GF_GPU_ROUTING": off})
                self.assertNotIn("GF_GPU_ROUTING", self._export(lines)[0])
                self.assertNotIn("replica factorization", "\n".join(lines))
        from lisatools.globalfit.communication.ranks import gpu_routing_enabled
        for on in ("1", "true", "TRUE", "yes", "On", " on "):
            self.assertTrue(gpu_routing_enabled({"GF_GPU_ROUTING": on}), on)
        for off in ("0", "no", "off", "false", ""):
            self.assertFalse(gpu_routing_enabled({"GF_GPU_ROUTING": off}), off)

    def test_the_unset_knob_leaves_every_shape_unchanged(self):
        for script in ROUTING_SCRIPTS:
            for ngpus, tasks, nodes in (("2", 3, 1), ("4", 5, 2), ("8", 9, 4)):
                with self.subTest(script=os.path.basename(script), ngpus=ngpus):
                    lines = self._run(script, {"NGPUS": ngpus})
                    self.assertIn(_routing_ntasks(script, tasks), lines)
                    self.assertIn(f"--nodes={nodes}", lines)
                    (exp,) = self._export(lines)
                    self.assertNotIn("GF_GPU_ROUTING", exp)
                    self.assertTrue(
                        exp.endswith(',GF_LEGACY_RANK_LAYOUT=0,GF_MONITOR_RANK=1')
                        or exp.endswith(',GF_LEGACY_RANK_LAYOUT=0'), exp)
                    self.assertNotIn("replica factorization", "\n".join(lines))

    def test_the_job_764_trap_is_refused_before_sbatch(self):
        for script in ROUTING_SCRIPTS:
            with self.subTest(script=os.path.basename(script)):
                lines = self._run(script, {"NGPUS": "8", "GPUS_PER_RANK": "2"}, rc=2)
                text = "\n".join(lines)
                self.assertEqual(self._export(lines), [], "sbatch must not run")
                self.assertIn("REFUSED: GPUS_PER_RANK=2 but each compute rank holds "
                              "1 walker(s) (NWALKERS=4 over N_COMPUTE=4)", text)
                self.assertIn("set GF_GPU_ROUTING=1", text)
                self.assertIn("NWALKERS a multiple of 8", text)
                # ...and the same with the routing on: still 1 walker per 2 GPUs
                self._run(script, {"NGPUS": "8", "GPUS_PER_RANK": "2",
                                   "GF_GPU_ROUTING": "1"}, rc=2)
        # enough walkers per rank is fine (8 walkers: 2 per 2-GPU rank)
        lines = self._run(NINE_MO_V9, {"NGPUS": "8", "GPUS_PER_RANK": "2",
                                       "NWALKERS": "8"})
        self.assertIn(_routing_ntasks(NINE_MO_V9, 5), lines)

    def test_routing_refuses_what_build_layout_refuses(self):
        for over in ({"RANKS_PER_BLOCK": "3"},          # does not divide 8
                     {"RANKS_PER_BLOCK": "8", "NWALKERS": "3"}):  # 1 block ok...
            with self.subTest(**over):
                env = {"NGPUS": "8", "GF_GPU_ROUTING": "1", **over}
                if over.get("NWALKERS") == "3":
                    # R=8 -> one block of 3 walkers: legal (no rounding)
                    lines = self._run(NINE_MO_V9, env)
                    self.assertIn("n_blocks=1 walker block(s) of 3 walker(s) x R=8",
                                  "\n".join(lines))
                else:
                    lines = self._run(NINE_MO_V9, env, rc=2)
                    self.assertIn("REFUSED: RANKS_PER_BLOCK=3", "\n".join(lines))

    def test_the_shell_factorization_is_pythons(self):
        from lisatools.globalfit.communication import factorize_layout

        funcs = _shell_block(open(NINE_MO_V9).read(), "_gf_routing_on() {",
                             'if [ -z "${SLURM_JOB_ID:-}" ]; then')
        for nw in (1, 2, 3, 4, 6, 8, 10, 24):
            for nc in (1, 2, 4, 8, 16, 32):
                r = subprocess.run(
                    ["bash", "-c", funcs + f"\n_gf_routing_layout {nw} {nc} 1"],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                m = re.search(r"n_blocks=(\d+) walker block\(s\) of (\d+) walker\(s\) "
                              r"x R=(\d+)", r.stdout)
                self.assertIsNotNone(m, r.stdout)
                nb, per, R = (int(x) for x in m.groups())
                self.assertEqual((nb, R, per), factorize_layout(nw, nc), (nw, nc))

    def _in_job(self, script, **env):
        text = open(script).read()
        funcs = _shell_block(text, "_gf_routing_on() {", 'if [ -z "${SLURM_JOB_ID:-}" ]; then')
        block = _shell_block(text, "_GF_ROUTING=0\n", "export NUM_ITERATIONS=")
        full = {"PATH": os.environ.get("PATH", ""), "_k": "1",
                "GF_LEGACY_RANK_LAYOUT": "0"}
        full.update({k: str(v) for k, v in env.items()})
        r = subprocess.run(
            ["bash", "-c", funcs + "\n" + block
             + "\necho NW=${NWALKERS} GF=${GF_GPU_ROUTING:-unset}"],
            env=full, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout

    def test_in_job_the_walker_count_is_never_rounded_with_the_knob_on(self):
        for script in ROUTING_SCRIPTS:
            with self.subTest(script=os.path.basename(script)):
                rc, out = self._in_job(script, N_COMPUTE_EFF=8, NWALKERS=4,
                                       GF_GPU_ROUTING="1")
                self.assertEqual(rc, 0, out)
                self.assertIn("NW=4 GF=1", out.splitlines())
                self.assertIn("n_blocks=4 walker block(s) of 1 walker(s) x R=2", out)
                self.assertNotIn("is not a multiple of", out)
                rc, out = self._in_job(script, N_COMPUTE_EFF=4, NWALKERS=6,
                                       GF_GPU_ROUTING="1")
                self.assertIn("NW=6 GF=1", out.splitlines())
                self.assertIn("n_blocks=2 walker block(s) of 3 walker(s) x R=2", out)
                # unset: the rounding the existing tests pin, unchanged
                rc, out = self._in_job(script, N_COMPUTE_EFF=8, NWALKERS=4)
                self.assertEqual(rc, 0, out)
                self.assertIn("NW=8 GF=unset", out.splitlines())
                self.assertIn("is not a multiple of N_COMPUTE=8", out)
                # an impossible granted shape stops the job before mpiexec
                rc, out = self._in_job(script, N_COMPUTE_EFF=8, NWALKERS=4,
                                       GF_GPU_ROUTING="1", RANKS_PER_BLOCK="3")
                self.assertEqual(rc, 2, out)


if __name__ == "__main__":
    unittest.main()
