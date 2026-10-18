"""SNR-keyed per-leaf ladders: ``{BRANCH}_LADDER_TOP_SNR`` (user ruling 2026-10-18, default off).

Mike: the EMRI ladders went to a uniform top temperature so the loudest leaf
(EMRI 3, SNR 50) sees an effective SNR ~2 at the top rung; a uniform T_max
is right for one SNR only. "While that runs, you can add an option (default
off) to set the ladder by SNR." Leaf k's ladder is ``make_ladder(ndim,
ntemps, Tmax=T_k)``, ``T_k = clip((SNR_k / top)^2, TMAX_MIN, TMAX)``. With
top 2, floor 4, ceiling 625: SNR 50 -> 625, 34 -> 289, 21 -> 110, 12 -> 36.

Pinned here (CPU):

* the formula, the clip, the prior-only / dead case, the knob casts;
* the per-leaf SNR (max-lnL cold walker, sqrt(<h|h>), the fallback);
* the ladder per leaf (rung count, geometric spacing, the ratio note);
* construction through the REAL ``SingleSourcePEBuilder``: fresh store,
  reset-on-resume, plain resume, the rung-count mismatch, knob off;
* the move's between-propose update: deferred construction, the 20 %
  refresh trigger, its cadence, its PE gate, the clipped no-op;
* the REAL ``EMRIDirectLikeMove``: deferred construction single process,
  the 4-rank walker-block fan-out equal to it, ``betas_all`` persisted
  through the HDF backend, and the lost-rung guard reading each leaf's OWN
  ladder;
* ``rerung_store_branches.py --top-snr``: the dry-run table, apply, the
  ``--ntemps-from-tmax`` count, the refusals;
* the knobs (env, settings) and the default-off path bit-identical to the
  golden stream of ``tests/test_eigen_pool_temper.py``.
"""

from __future__ import annotations

import copy
import io
import logging
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import h5py
import numpy as np
from eryn.moves.tempering import make_ladder

from lisatools.sampling import snrladder as S

try:
    from tests import test_source_default_ladders as D
except ImportError:  # pragma: no cover - run from tests/
    import test_source_default_ladders as D

LOGGER_MOVE = "lisatools.globalfit.moves.addremovemove"
LOGGER_RECIPE = "lisatools.globalfit.recipe"

#: every knob this file reads: cleared so a developer shell cannot leak in
_KNOBS = tuple(
    f"{b}_{k}" for b in ("EMRI", "MBH", "SOBBH")
    for k in ("LADDER_TOP_SNR", "LADDER_TMAX", "LADDER_TMAX_MIN", "LADDER_TOP_SNR_REFRESH",
              "LADDER_TOP_SNR_REFRESH_PE", "LADDER_RESET_ON_RESUME", "LADDER_REANCHOR_NATS",
              "NTEMPS")
)


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


# ---------------------------------------------------------------------------
# the formula and the casts
# ---------------------------------------------------------------------------

class FormulaTest(unittest.TestCase):
    TOP, LO, HI = 2.0, 4.0, 625.0

    def test_the_rulings_examples(self):
        """Mike's numbers (2026-10-18): EMRI 3 (SNR 50) -> 625, EMRI 6 (34) -> 289,
        EMRI 0 (21) -> 110, an SNR 12 leaf -> 36."""
        for snr, want, clip in ((50.0, 625.0, ""), (50.2, 625.0, "ceiling"),
                                (34.0, 289.0, ""), (21.0, 110.25, ""), (12.0, 36.0, "")):
            with self.subTest(snr=snr):
                t, raw, c = S.snr_tmax(snr, self.TOP, self.LO, self.HI)
                self.assertAlmostEqual(t, want, places=10)
                self.assertAlmostEqual(raw, (snr / 2.0) ** 2, places=10)
                self.assertEqual(c, clip)
        self.assertEqual(round(S.snr_tmax(21.0, 2, 4, 625)[0]), 110)

    def test_the_floor_and_the_prior_only_case(self):
        for snr in (3.9, 1.0, 0.0, -2.0, None, float("nan"), float("inf") * 0):
            with self.subTest(snr=snr):
                t, raw, c = S.snr_tmax(snr, self.TOP, self.LO, self.HI)
                self.assertEqual((t, c), (4.0, "floor"))
        # SNR 4 at top 2 is exactly the floor, unclipped
        self.assertEqual(S.snr_tmax(4.0, 2, 4, 625), (4.0, 4.0, ""))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            S.snr_tmax(10, 0.0, 4, 625)            # the knob off is not a top SNR
        for lo, hi in ((1.0, 625), (0.5, 625), (700, 625), (4, float("nan"))):
            with self.subTest(lo=lo, hi=hi), self.assertRaises(ValueError):
                S.check_clip(lo, hi)

    def test_the_knob_cast(self):
        for raw, want in (("on", 2.0), ("TRUE", 2.0), ("yes", 2.0), (True, 2.0), ("off", 0.0),
                          ("false", 0.0), ("", 0.0), (None, 0.0), (False, 0.0), ("0", 0.0),
                          ("1", 1.0), ("2.5", 2.5), (3, 3.0), ("-1", 0.0)):
            with self.subTest(raw=raw):
                self.assertEqual(S.top_snr_value(raw), want)
        self.assertEqual(S.LADDER_TOP_SNR_DEFAULT, 2.0)
        for bad in ("two", "nan", "inf"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                S.top_snr_value(bad)

    def test_the_ceiling(self):
        self.assertEqual(S.ceiling_tmax(12, 11, 625), 625.0)
        self.assertAlmostEqual(S.ceiling_tmax(12, 8, None),
                               1.0 / make_ladder(12, ntemps=8)[-1], places=9)
        # the coordinator's count: make_ladder(12, Tmax=625) is 11 rungs, ratio 1.90
        self.assertEqual(S.make_ladder_count(12, 625.0), 11)
        self.assertAlmostEqual(S.rung_ratio(make_ladder(12, Tmax=625.0)), 625 ** 0.1, places=12)

    def test_snr_changed_is_twenty_percent(self):
        self.assertFalse(S.snr_changed(59.9, 50.0))
        self.assertTrue(S.snr_changed(60.1, 50.0))
        self.assertTrue(S.snr_changed(39.9, 50.0))
        self.assertFalse(S.snr_changed(40.1, 50.0))
        self.assertFalse(S.snr_changed(float("nan"), 50.0))
        self.assertFalse(S.snr_changed(10.0, None))


class LeafSNRTest(unittest.TestCase):
    def test_the_max_lnl_cold_walker_not_the_loudest(self):
        h_h = np.array([2500.0, 900.0, 1600.0, 3600.0])
        cold = np.array([-10.0, -3.0, -50.0, -1e300])     # walker 3 is the sentinel
        snr, w, kind = S.leaf_snr(h_h * 0.9, h_h, cold)
        self.assertEqual((w, kind), (1, "optimal"))
        self.assertAlmostEqual(snr, 30.0)

    def test_without_lnl_the_largest_hh(self):
        self.assertEqual(S.leaf_snr(None, [4.0, 100.0, 9.0])[:2], (10.0, 1))

    def test_the_detection_fallback_only_when_hh_is_unusable(self):
        snr, w, kind = S.leaf_snr([400.0, 100.0], [np.nan, 4.0], [0.0, -1.0])
        self.assertEqual((w, kind), (0, "detection"))
        self.assertAlmostEqual(snr, 20.0)                 # sqrt(<d|h>) at the maximum
        self.assertTrue(np.isnan(S.leaf_snr([np.nan], [np.nan], [0.0])[0]))
        self.assertIsNone(S.leaf_snr([-1.0, np.nan], [0.0, np.nan])[2])

    def test_per_leaf_from_substate_shaped_arrays(self):
        nw, nl, nt = 3, 2, 4
        h_h = np.array([[100.0, 4.0], [400.0, 9.0], [225.0, np.nan]])
        ll = np.zeros((nl, nt, nw))
        ll[0, 0] = [0.0, -5.0, 1.0]                        # leaf 0: walker 2
        ll[1, 0] = [0.0, 3.0, 9.0]                         # leaf 1: walker 2 has no h_h -> 1
        out = S.leaf_snrs(None, h_h, ll, nl)
        self.assertEqual([(round(s, 9), w) for s, w, _ in out], [(15.0, 2), (3.0, 1)])
        sub = SimpleNamespace(d_h=None, h_h=h_h, log_like=ll, nleaves_max=nl)
        self.assertEqual(S.substate_leaf_snrs(sub), out)
        self.assertTrue(all(np.isnan(s) for s, _, _ in S.leaf_snrs(None, None, None, 3)))


class LadderTest(unittest.TestCase):
    def test_one_ladder_per_leaf_with_the_rung_count_and_spacing(self):
        for t in (4.0, 36.0, 289.0, 625.0):
            for nt in (2, 11, 18):
                with self.subTest(T=t, ntemps=nt):
                    b = S.snr_keyed_ladder(12, nt, t)
                    np.testing.assert_array_equal(b, make_ladder(12, ntemps=nt, Tmax=t))
                    self.assertEqual(b.shape, (nt,))
                    self.assertEqual(b[0], 1.0)
                    self.assertAlmostEqual(1.0 / b[-1], t, places=8)
                    ratios = b[:-1] / b[1:]
                    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)
                    self.assertAlmostEqual(S.rung_ratio(b), ratios[0], places=10)

    def test_a_leaf_below_the_ceiling_is_denser(self):
        """The count is per branch, so T < ceiling means a smaller ratio."""
        loud, quiet = S.snr_keyed_ladder(12, 11, 625.0), S.snr_keyed_ladder(12, 11, 36.0)
        self.assertGreater(S.rung_ratio(loud), S.rung_ratio(quiet))
        self.assertAlmostEqual(S.rung_ratio(loud), 1.9036539387, places=8)

    def test_the_log_line(self):
        b = S.snr_keyed_ladder(12, 11, 625.0)
        line = S.describe_leaf("[EMRI_LADDER]", 3, 50.2, 625.0, 630.01, "ceiling", 2.0, 4, 625,
                               b, "fresh store", walker=1)
        self.assertEqual(
            line, "[EMRI_LADDER] leaf 3: SNR 50.2 -> T_max 630.0 (top SNR 2.0, ceiling 625 -> "
                  "625.0), ladder [1 ... 0.0016], ratio 1.90 over 11 rungs (fresh store; optimal "
                  "SNR at walker 1)")
        dead = S.describe_leaf("[EMRI_LADDER]", 5, float("nan"), 4.0, 0.0, "floor", 2.0, 4, 625,
                               S.snr_keyed_ladder(12, 11, 4.0), "fresh store")
        self.assertIn("leaf 5: no source -> T_max 4 (top SNR 2.0, floor 4 -> 4.0)", dead)


# ---------------------------------------------------------------------------
# construction through the REAL builder (test_source_default_ladders' fake move)
# ---------------------------------------------------------------------------

NW, NL, ND = D.NW, D.NLEAVES, D.NDIM                 # 4 walkers, 3 leaves, ndim 12
SNRS = (50.2, 34.0, 12.0)


def _info(configured, *, top=2.0, reset=False, tmax=625.0, lo=None, refresh=None):
    info = D._info(configured, reset=reset)
    info.ladder_top_snr = top
    info.ladder_tmax = tmax
    info.ladder_tmax_min = lo
    info.ladder_top_snr_refresh = refresh
    info.ladder_top_snr_refresh_pe = None
    return info


def _record_snrs(sub, snrs=SNRS, best=1):
    """Leaf k recorded at SNR snrs[k] on walker ``best`` (the max-lnL cold walker);
    the other walkers louder, so the walker choice is visible."""
    for leaf, snr in enumerate(snrs):
        sub.h_h[:, leaf] = (snr + 7.0) ** 2
        sub.h_h[best, leaf] = snr ** 2
        sub.d_h[:, leaf] = sub.h_h[:, leaf]
        sub.log_like[leaf, 0, :] = -100.0
        sub.log_like[leaf, 0, best] = -1.0
    return sub


class BuilderTest(unittest.TestCase):
    NT = 11

    def _build(self, info, sub):
        with _env():
            return D._build(info, sub)

    def _want(self, snrs=SNRS):
        return np.stack([make_ladder(ND, ntemps=self.NT, Tmax=S.snr_tmax(s, 2, 4, 625)[0])
                         for s in snrs])

    def test_fresh_store_with_recorded_snrs(self):
        cfg = make_ladder(ND, ntemps=self.NT, Tmax=625.0)
        with self.assertLogs(LOGGER_RECIPE, level="INFO") as cm:
            move, sub = self._build(_info(cfg), _record_snrs(D._sub(self.NT)))
        np.testing.assert_array_equal(sub.betas_all, self._want())
        np.testing.assert_array_equal(move.kwargs["betas_all"], sub.betas_all)
        self.assertEqual(sub.betas_all.shape, (NL, self.NT))
        np.testing.assert_allclose(1.0 / sub.betas_all[:, -1], [625.0, 289.0, 36.0])
        self.assertEqual(move.snr_ladder_plan, {"ref": {0: 50.2, 1: 34.0, 2: 12.0}, "pending": set()})
        text = "\n".join(cm.output)
        self.assertIn("[EMRI_LADDER] leaf 0: SNR 50.2 -> T_max 630.0 (top SNR 2.0, ceiling 625 -> "
                      "625.0), ladder [1 ... 0.0016], ratio 1.90 over 11 rungs (fresh store; "
                      "optimal SNR at walker 1)", text)
        self.assertIn("leaf 2: SNR 12.0 -> T_max 36.0 (top SNR 2.0, within [4, 625])", text)

    def test_fresh_store_without_snrs_is_pending_on_the_configured_ladder(self):
        cfg = make_ladder(ND, ntemps=self.NT, Tmax=625.0)
        with self.assertLogs(LOGGER_RECIPE, level="INFO") as cm:
            move, sub = self._build(_info(cfg), D._sub(self.NT))
        np.testing.assert_array_equal(sub.betas_all, np.tile(cfg, (NL, 1)))
        self.assertEqual(move.snr_ladder_plan, {"ref": {}, "pending": {0, 1, 2}})
        self.assertIn("leaf 1: no SNR recorded yet -> the configured ladder (T_max 625.0)",
                      "\n".join(cm.output))

    def test_a_dead_leaf_gets_the_floor(self):
        sub = _record_snrs(D._sub(self.NT), snrs=(50.0, np.nan, 12.0))
        sub.inds[:, :, 1] = False
        move, sub = self._build(_info(make_ladder(ND, ntemps=self.NT)), sub)
        self.assertAlmostEqual(1.0 / sub.betas_all[1, -1], 4.0, places=10)
        self.assertEqual(move.snr_ladder_plan["pending"], set())
        self.assertNotIn(1, move.snr_ladder_plan["ref"])

    def test_reset_on_resume_rebuilds_from_the_stores_snrs(self):
        stored = D._adapted(self.NT)
        cfg = make_ladder(ND, ntemps=self.NT, Tmax=625.0)
        with self.assertLogs(LOGGER_RECIPE, level="INFO") as cm:
            move, sub = self._build(_info(cfg, reset=True), _record_snrs(D._sub(self.NT, stored)))
        np.testing.assert_array_equal(sub.betas_all, self._want())
        text = "\n".join(cm.output)
        self.assertIn("leaf 0 ladder reset: old ", text)                   # the existing lines
        self.assertIn("(EMRI_LADDER_RESET_ON_RESUME=1; optimal SNR at walker 1)", text)

    def test_a_plain_resume_keeps_the_stored_ladders(self):
        stored = D._adapted(self.NT)
        with self.assertLogs(LOGGER_RECIPE, level="INFO") as cm:
            move, sub = self._build(_info(make_ladder(ND, ntemps=self.NT)),
                                    _record_snrs(D._sub(self.NT, stored)))
        np.testing.assert_array_equal(sub.betas_all, stored)
        self.assertEqual(move.snr_ladder_plan, {"ref": {}, "pending": set()})
        text = "\n".join(cm.output)
        self.assertIn("keeps the STORED", text)
        self.assertIn("each leaf's refresh baseline is its first measured SNR", text)

    def test_a_rung_count_mismatch_keys_at_the_stores_count(self):
        stored = D._adapted(6)
        with self.assertLogs(LOGGER_RECIPE, level="INFO"):
            move, sub = self._build(_info(make_ladder(ND, ntemps=8), reset=True),
                                    _record_snrs(D._sub(6, stored)))
        self.assertEqual(move.coords_shape[0], 6)
        np.testing.assert_array_equal(
            sub.betas_all[1], make_ladder(ND, ntemps=6, Tmax=289.0))

    def test_knob_off_is_the_previous_build(self):
        cfg = make_ladder(ND, ntemps=self.NT)
        for top in (0.0, None):
            with self.subTest(top=top):
                move, sub = self._build(_info(cfg, top=top), _record_snrs(D._sub(self.NT)))
                np.testing.assert_array_equal(sub.betas_all, np.tile(cfg, (NL, 1)))
                self.assertFalse(hasattr(move, "snr_ladder_plan"))

    def test_the_env_knob_reaches_a_namespace_info(self):
        info = D._info(make_ladder(ND, ntemps=self.NT))            # no ladder_top_snr field
        with _env(EMRI_LADDER_TOP_SNR="on", EMRI_LADDER_TMAX="625"), \
                self.assertLogs(LOGGER_RECIPE, level="INFO"):
            move, sub = D._build(info, _record_snrs(D._sub(self.NT)))
        np.testing.assert_array_equal(sub.betas_all, self._want())

    def test_the_knobs_reach_the_move(self):
        info = _info(make_ladder(ND, ntemps=4), lo=6.0, refresh=5)
        info.ladder_top_snr_refresh_pe = True
        move, _ = self._build(info, D._sub(4))
        self.assertEqual(
            {k: move.kwargs[k] for k in ("ladder_top_snr", "ladder_tmax", "ladder_tmax_min",
                                         "ladder_top_snr_refresh", "ladder_top_snr_refresh_pe")},
            dict(ladder_top_snr=2.0, ladder_tmax=625.0, ladder_tmax_min=6.0,
                 ladder_top_snr_refresh=5, ladder_top_snr_refresh_pe=True))

    def test_a_ceiling_below_the_floor_is_refused(self):
        with self.assertRaises(ValueError):
            self._build(_info(make_ladder(ND, ntemps=4), tmax=3.0), D._sub(4))


# ---------------------------------------------------------------------------
# the move's between-propose update (a data-free stub)
# ---------------------------------------------------------------------------

def _stub_move(nt=11, nl=3, **attrs):
    from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove

    mv = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    mv.branch_name = "emri"
    mv.ntemps, mv.nwalkers, mv.nleaves_max, mv.ndim = nt, NW, nl, ND
    mv.temperature_controls = [SimpleNamespace(betas=np.ones(nt)) for _ in range(nl)]
    for k in ("ladder_top_snr", "ladder_tmax", "ladder_tmax_min", "ladder_top_snr_refresh",
              "ladder_top_snr_refresh_pe"):
        setattr(mv, k, None)
    mv.ladder_top_snr, mv.ladder_tmax = 2.0, 625.0
    for k, v in attrs.items():
        setattr(mv, k, v)
    return mv


def _stub_state(snrs, nt=11, betas=None):
    nl = len(snrs)
    sub = SimpleNamespace(
        tempered_initialized=True, nleaves_max=nl,
        betas_all=(np.tile(make_ladder(ND, ntemps=nt, Tmax=625.0), (nl, 1))
                   if betas is None else np.array(betas, dtype=float)),
        inds=np.ones((nt, NW, nl), dtype=bool),
        coords=np.random.default_rng(0).normal(size=(nt, NW, nl, ND)),
        d_h=np.full((NW, nl), np.nan), h_h=np.full((NW, nl), np.nan),
        log_like=np.zeros((nl, nt, NW)))
    _set_snrs(sub, snrs)
    return SimpleNamespace(sub_states={"emri": sub})


def _set_snrs(sub, snrs):
    for leaf, s in enumerate(snrs):
        sub.h_h[:, leaf] = np.nan if s is None else float(s) ** 2


def _top(sub, leaf):
    return float(1.0 / sub.betas_all[leaf, -1])


class MoveUpdateTest(unittest.TestCase):
    def setUp(self):
        p = _env()
        p.start()
        self.addCleanup(p.stop)

    def test_off_touches_nothing(self):
        mv = _stub_move(ladder_top_snr=0.0, ladder_top_snr_refresh=1)
        mv.gf_stage_kind = "search"
        state = _stub_state((50.0, 30.0, 12.0))
        before = copy.deepcopy(state.sub_states["emri"].__dict__)
        self.assertEqual(mv._maybe_update_snr_ladders(state), [])
        for k, v in before.items():
            np.testing.assert_array_equal(getattr(state.sub_states["emri"], k), v, err_msg=k)
        self.assertIsNone(mv.snr_ladder_plan)

    def test_deferred_construction_at_the_first_recorded_snr_in_any_stage(self):
        mv = _stub_move()
        mv.snr_ladder_plan = {"ref": {}, "pending": {0, 1, 2}}
        mv.gf_stage_kind = "pe"
        state = _stub_state((None, None, None))
        sub = state.sub_states["emri"]
        coords0 = sub.coords.copy()
        self.assertEqual(mv._maybe_update_snr_ladders(state), [])        # nothing recorded yet
        _set_snrs(sub, (50.0, None, 12.0))
        with self.assertLogs(LOGGER_MOVE, level="INFO") as cm:
            out = mv._maybe_update_snr_ladders(state)
        self.assertEqual([(l, t) for l, _, t in out], [(0, 625.0), (2, 36.0)])
        self.assertAlmostEqual(_top(sub, 2), 36.0, places=9)
        self.assertAlmostEqual(_top(sub, 1), 625.0, places=9)             # still pending
        self.assertEqual(mv.snr_ladder_plan["pending"], {1})
        np.testing.assert_array_equal(mv.temperature_controls[2].betas, sub.betas_all[2])
        np.testing.assert_array_equal(sub.coords, coords0)               # rung STATES untouched
        text = "\n".join(cm.output)
        self.assertIn("construction, deferred to the leaf's first recorded SNR", text)
        # leaf 1 still unmeasured at the second propose: ONE warning, never repeated
        self.assertIn("WARNING", text)
        self.assertIn("leaves 1 have no recorded <h|h> yet", text)
        with self.assertLogs(LOGGER_MOVE, level="DEBUG") as cm:
            mv._maybe_update_snr_ladders(state)
            logging.getLogger(LOGGER_MOVE).debug("sentinel")
        self.assertNotIn("WARNING", "\n".join(cm.output))

    def _refresh_case(self, kind, new_snr, *, every=1, pe=None, calls=1):
        mv = _stub_move(ladder_top_snr_refresh=every, ladder_top_snr_refresh_pe=pe)
        mv.snr_ladder_plan = {"ref": {0: 50.0}, "pending": set()}
        mv.gf_stage_kind = kind
        state = _stub_state((50.0,))
        _set_snrs(state.sub_states["emri"], (new_snr,))
        outs = [mv._maybe_update_snr_ladders(state) for _ in range(calls)]
        return mv, state.sub_states["emri"], outs

    def test_the_refresh_fires_on_a_twenty_percent_change_in_search(self):
        for kind in ("search", "rj"):
            with self.subTest(kind=kind):
                mv, sub, outs = self._refresh_case(kind, 30.0)
                self.assertEqual(outs[0], [(0, 30.0, 225.0)])
                np.testing.assert_array_equal(sub.betas_all[0], make_ladder(ND, ntemps=11, Tmax=225.0))
                self.assertEqual(mv.snr_ladder_plan["ref"], {0: 30.0})
        _, sub, outs = self._refresh_case("search", 41.0)                 # -18 %: kept
        self.assertEqual(outs[0], [])
        self.assertAlmostEqual(_top(sub, 0), 625.0, places=9)

    def test_the_pe_gate(self):
        _, sub, outs = self._refresh_case("pe", 30.0)
        self.assertEqual(outs[0], [])
        self.assertAlmostEqual(_top(sub, 0), 625.0, places=9)
        _, _, outs = self._refresh_case(None, 30.0)                       # unknown stage
        self.assertEqual(outs[0], [])
        _, sub, outs = self._refresh_case("pe", 30.0, pe=True)            # ruled in
        self.assertEqual(outs[0], [(0, 30.0, 225.0)])
        _, _, outs = self._refresh_case("search", 30.0, every=0)         # never
        self.assertEqual(outs[0], [])

    def test_the_cadence_counts_proposes(self):
        _, _, outs = self._refresh_case("search", 30.0, every=3, calls=3)
        self.assertEqual([len(o) for o in outs], [0, 0, 1])

    def test_a_clipped_change_only_moves_the_baseline(self):
        mv = _stub_move(ladder_top_snr_refresh=1)
        mv.snr_ladder_plan = {"ref": {0: 60.0}, "pending": set()}
        mv.gf_stage_kind = "search"
        state = _stub_state((80.0,))
        before = state.sub_states["emri"].betas_all.copy()
        with self.assertLogs(LOGGER_MOVE, level="INFO") as cm:
            self.assertEqual(mv._maybe_update_snr_ladders(state), [])
        np.testing.assert_array_equal(state.sub_states["emri"].betas_all, before)
        self.assertEqual(mv.snr_ladder_plan["ref"], {0: 80.0})
        self.assertIn("but T_max stays 625.0 (clipped)", "\n".join(cm.output))

    def test_a_plain_resume_sets_the_baseline_first(self):
        mv = _stub_move(ladder_top_snr_refresh=1)
        mv.snr_ladder_plan = {"ref": {}, "pending": set()}
        mv.gf_stage_kind = "search"
        state = _stub_state((30.0,))
        before = state.sub_states["emri"].betas_all.copy()
        with self.assertLogs(LOGGER_MOVE, level="INFO") as cm:
            self.assertEqual(mv._maybe_update_snr_ladders(state), [])
        np.testing.assert_array_equal(state.sub_states["emri"].betas_all, before)
        self.assertEqual(mv.snr_ladder_plan["ref"], {0: 30.0})
        self.assertIn("refresh baseline SNR 30.0 (the stored ladder is kept, T_max 625.0)",
                      "\n".join(cm.output))

    def test_a_move_built_outside_the_builder_treats_every_leaf_as_pending(self):
        mv = _stub_move()
        state = _stub_state((34.0, 12.0))
        out = mv._maybe_update_snr_ladders(state)
        self.assertEqual([(l, t) for l, _, t in out], [(0, 289.0), (1, 36.0)])

    def test_the_move_knobs_resolve_kwarg_then_env_then_default(self):
        mv = _stub_move()
        for k in ("ladder_top_snr", "ladder_tmax"):
            setattr(mv, k, None)
        self.assertEqual(mv._ladder_top_snr(), 0.0)
        self.assertEqual(mv._ladder_tmax_min(), 4.0)
        self.assertEqual(mv._ladder_top_snr_refresh(), 0)
        self.assertFalse(mv._ladder_top_snr_refresh_pe())
        self.assertAlmostEqual(mv._ladder_tmax_ceiling(8), 1.0 / make_ladder(ND, ntemps=8)[-1])
        with _env(EMRI_LADDER_TOP_SNR="yes", EMRI_LADDER_TMAX="625", EMRI_LADDER_TMAX_MIN="6",
                  EMRI_LADDER_TOP_SNR_REFRESH="10", EMRI_LADDER_TOP_SNR_REFRESH_PE="1"):
            self.assertEqual(mv._ladder_top_snr(), 2.0)
            self.assertEqual(mv._ladder_tmax_ceiling(8), 625.0)
            self.assertEqual(mv._ladder_tmax_min(), 6.0)
            self.assertEqual(mv._ladder_top_snr_refresh(), 10)
            self.assertTrue(mv._ladder_top_snr_refresh_pe())
            mv.ladder_top_snr, mv.ladder_top_snr_refresh = 3.0, 0
            self.assertEqual((mv._ladder_top_snr(), mv._ladder_top_snr_refresh()), (3.0, 0))


# ---------------------------------------------------------------------------
# the REAL EMRI move (test_emri_direct_fanout's toy WDM world)
# ---------------------------------------------------------------------------

try:
    from tests import test_eigen_pool_temper as P
    from tests import test_emri_direct_fanout as E
except ImportError:  # pragma: no cover - run from tests/
    import test_eigen_pool_temper as P
    import test_emri_direct_fanout as E

H = E.H


def _staged(builder, kind="search"):
    def make(acs, nwalkers, **kw):
        move = builder(acs, nwalkers, **kw)
        move.gf_stage_kind = kind
        return move

    return make


class RealMoveTest(E._Case):
    N_PROPOSES = 3
    ENV = dict(EMRI_LADDER_TOP_SNR="2", EMRI_LADDER_TMAX="625")

    def setUp(self):
        super().setUp()
        for k in _KNOBS:
            os.environ.pop(k, None)

    def _expected(self, state):
        """The ladders the construction must build from ``state``'s records."""
        sub = state.sub_states["emri"]
        out = []
        for snr, _, kind in S.substate_leaf_snrs(sub, H.NLEAVES):
            self.assertEqual(kind, "optimal")
            out.append(make_ladder(H.NDIM, ntemps=H.NTEMPS, Tmax=S.snr_tmax(snr, 2, 4, 625)[0]))
        return np.stack(out), [s for s, _, _ in S.substate_leaf_snrs(sub, H.NLEAVES)]

    def test_deferred_construction_single_process_and_the_block_fanout_equals_it(self):
        kw = dict(scope="per_walker", adaptive=False, refresh=10, rng=H._RowLocalRandom,
                  permute_every=0)
        with mock.patch.object(E, "_make_move", _staged(E._make_move)), \
                mock.patch.dict(os.environ, self.ENV), H._row_local_numpy():
            with self.assertLogs(LOGGER_MOVE, level="INFO") as cm:
                ref = self._single(4, **kw)
            out = self._world(4, 4, **kw)
        states = ref["states"]
        # propose 1 ran on the configured ladder; propose 2's prepare built every
        # leaf's ladder from propose 1's records, and the ladder stays (no adaptation)
        want, snrs = self._expected(states[0])
        self.assertTrue(np.all(np.isfinite(snrs)))
        for k in (1, 2):
            np.testing.assert_array_equal(states[k].sub_states["emri"].betas_all, want)
        text = "\n".join(cm.output)
        for leaf in range(H.NLEAVES):
            self.assertIn(f"[EMRI_LADDER] leaf {leaf}: SNR {snrs[leaf]:.1f} -> T_max", text)
        self.assertEqual(text.count("construction, deferred"), H.NLEAVES)
        # the 4-rank walker blocks: the head's prepare edits the ladder before it
        # slices, so every rank proposes on it and the merge equals single process
        self._assert_match(out, ref)
        for k, s in enumerate(out[0]["states"]):
            np.testing.assert_allclose(s.sub_states["emri"].betas_all,
                                       states[k].sub_states["emri"].betas_all, rtol=1e-12, atol=0)

    def test_the_rebuilt_row_is_persisted_by_the_backend(self):
        from lisatools.globalfit.hdfbackend import EMRIHDFBackend, GFHDFBackend
        from lisatools.globalfit.state import EMRIState

        kw = dict(scope="walker_max", adaptive=False, refresh=10,
                  rng=lambda: np.random.RandomState(H.SEED), permute_every=0)
        with mock.patch.object(E, "_make_move", _staged(E._make_move)), \
                mock.patch.dict(os.environ, self.ENV):
            ref = self._single(4, **kw)
        last = ref["states"][-1]
        want, _ = self._expected(ref["states"][0])
        np.testing.assert_array_equal(last.sub_states["emri"].betas_all, want)
        self.assertFalse(np.allclose(want, np.tile(H.BETAS, (H.NLEAVES, 1))))
        tmp = tempfile.mkdtemp(prefix="snrladder_")
        self.addCleanup(shutil.rmtree, tmp, True)
        backend = GFHDFBackend(os.path.join(tmp, "store.h5"), sub_backend={"emri": EMRIHDFBackend},
                               sub_state_bases={"emri": EMRIState})
        backend.reset(4, {"emri": H.NDIM}, nleaves_max={"emri": H.NLEAVES}, ntemps=1,
                      branch_names=["emri"], nbranches=1, rj=False, moves=None,
                      sub_reset_kwargs={"emri": dict(nleaves_max=H.NLEAVES, ndim=H.NDIM,
                                                     num_emris=H.NLEAVES, ntemps=H.NTEMPS)})
        backend.grow(1, None)
        last.random_state = np.random.get_state()            # eryn's saver stores it
        backend.save_step(last, np.ones((1, 4)))
        with h5py.File(os.path.join(tmp, "store.h5"), "r") as f:
            np.testing.assert_array_equal(f["global_fit/sub_backend/emri/betas_all"][0], want)
        np.testing.assert_array_equal(backend.get_a_sample(0).sub_states["emri"].betas_all, want)


class GuardPerLeafLadderTest(E._Case):
    """The lost-rung guard's expected tempering drop uses each LEAF's own ladder."""

    N_PROPOSES = 1

    def test_the_guard_reads_the_visited_leafs_row(self):
        rows = np.array([[1.0, 0.5], [1.0, 1.0 / 625.0]])
        real = E._state

        def _state(walkers):
            state = real(walkers)
            state.sub_states["emri"].betas_all = rows.copy()
            return state

        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove as M

        real_guard = M._reanchor_lost_rungs
        calls = []

        def spy(move, work, leaf, prev_logl, prev_logp, betas):
            calls.append((int(leaf), np.array(betas, dtype=float, copy=True)))
            return real_guard(move, work, leaf, prev_logl, prev_logp, betas)

        kw = dict(scope="walker_max", adaptive=False, refresh=10,
                  rng=lambda: np.random.RandomState(H.SEED), permute_every=0)
        with mock.patch.object(E, "_state", _state), \
                mock.patch.object(M, "_reanchor_lost_rungs", spy):
            self._single(4, **kw)
        self.assertEqual(sorted(l for l, _ in calls), [0, 1])
        for leaf, betas in calls:
            np.testing.assert_array_equal(betas, rows[leaf], err_msg=f"leaf {leaf}")
        # and the margin it implies differs per leaf: 5.5 x (2 - 1) vs 5.5 x (625 - 1)
        margins = {leaf: 0.5 * H.NDIM * (1.0 / b[1] - 1.0) for leaf, b in calls}
        self.assertAlmostEqual(margins[0], 5.5)
        self.assertAlmostEqual(margins[1], 5.5 * 624.0)


# ---------------------------------------------------------------------------
# rerung_store_branches.py --top-snr
# ---------------------------------------------------------------------------

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))
import rerung_store_branches as R  # noqa: E402

try:
    from tests import test_rerung_store_branches as RT
except ImportError:  # pragma: no cover - run from tests/
    import test_rerung_store_branches as RT

#: mbh of the synthetic store: 2 rungs, 4 leaves (the last DEAD), ndim 11
MBH_SNRS = (50.2, 34.0, 12.0)


class RerungTopSNRTest(RT._Base):
    def setUp(self):
        super().setUp()
        with h5py.File(self.path, "r+") as f:
            g = f["global_fit/sub_backend/mbh"]
            h_h = np.full(g["h_h"].shape, np.nan)
            ll = g["log_like"][...]
            for leaf, snr in enumerate(MBH_SNRS):
                h_h[:, :, leaf] = (snr + 5.0) ** 2
                h_h[:, 2, leaf] = snr ** 2
                ll[:, leaf, 0, :] = -50.0
                ll[:, leaf, 0, 2] = -1.0               # walker 2 is the max-lnL cold walker
            h_h[-1, :, 1] = np.nan                     # leaf 1: no record on the last row
            g["h_h"][...] = h_h
            g["d_h"][...] = np.where(np.isfinite(h_h), h_h, np.nan)
            g["log_like"][...] = ll

    def _main(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = R.main([self.path, *args])
        return rc, out.getvalue()

    def _want(self, nt):
        rows = [make_ladder(11, ntemps=nt, Tmax=S.snr_tmax(s, 2, 4, 625)[0]) for s in MBH_SNRS]
        rows.append(make_ladder(11, ntemps=nt, Tmax=4.0))       # the dead leaf: the floor
        return np.stack(rows)

    def test_dry_run_prints_the_table_and_writes_nothing(self):
        before = self._read("mbh", "betas_all")
        rc, text = self._main("--set", "mbh=2", "--top-snr", "--ladder-tmax", "625")
        self.assertEqual(rc, 0, text)
        np.testing.assert_array_equal(self._read("mbh", "betas_all"), before)
        self.assertIn("mbh: 2 -> 2 rung(s)   (count kept: betas_all only)", text)
        self.assertIn("T_max = clip((SNR / 2)^2, 4, 625), 2 rungs", text)
        lines = [l.split() for l in text.splitlines() if l.strip()[:1].isdigit()]
        table = {int(l[0]): l for l in lines if len(l) >= 8}
        self.assertEqual(table[0][1:4], [str(RT.ROWS - 1), "2", "50.2"])
        self.assertEqual(table[0][6:8], ["625.0", "ceiling"])
        self.assertEqual(table[1][1:4], [str(RT.ROWS - 2), "2", "34.0"])     # an earlier row
        self.assertEqual(table[1][6], "289.0")
        self.assertEqual(table[2][6], "36.0")
        self.assertIn("dead leaf (no source): the floor", text)
        self.assertIn("DRY RUN", text)
        self.assertIn("MBH_NTEMPS=2 MBH_LADDER_TOP_SNR=2 MBH_LADDER_TMAX=625", text)

    def test_apply_writes_the_per_leaf_ladders_on_every_row_and_nothing_else(self):
        chain = self._read("mbh", "chain")
        rc, text = self._main("--set", "mbh=2", "--top-snr", "2", "--ladder-tmax", "625",
                              "--apply", "--no-backup")
        self.assertEqual(rc, 0, text)
        b = self._read("mbh", "betas_all")
        self.assertEqual(b.shape, (RT.ROWS, 4, 2))
        np.testing.assert_allclose(b, np.broadcast_to(self._want(2), b.shape), rtol=1e-15)
        np.testing.assert_array_equal(self._read("mbh", "chain"), chain)
        self.assertIn("OK", text)

    def test_ntemps_from_tmax_grows_to_make_ladders_count(self):
        n = S.make_ladder_count(11, 625.0)
        self.assertGreater(n, 2)
        rc, text = self._main("--set", "mbh=auto", "--ntemps-from-tmax", "--top-snr",
                              "--ladder-tmax", "625", "--apply", "--no-backup")
        self.assertEqual(rc, 0, text)
        self.assertIn(f"make_ladder(ndim=11, Tmax=625) = {n}", text)
        self.assertEqual(self._read("mbh", "chain").shape, (RT.ROWS, n, RT.NW, 4, 11))
        b = self._read("mbh", "betas_all")
        np.testing.assert_allclose(b[-1], self._want(n), rtol=1e-15)
        self.assertEqual(int(self._attr("mbh", "ntemps")), n)
        self.assertIn(f"launch with: MBH_NTEMPS={n} MBH_LADDER_TOP_SNR=2", text)

    def test_a_leaf_never_recorded_keeps_the_ladder_without_top_snr(self):
        with h5py.File(self.path, "r+") as f:
            g = f["global_fit/sub_backend/mbh"]
            g["h_h"][:, :, 2] = np.nan
            g["d_h"][:, :, 2] = np.nan
        plans = R.plan(self.path, {"mbh": 3}, top_snr=2.0, ladder_tmax=625.0)
        np.testing.assert_allclose(plans[0].ladder[2], 1.0 / 1.2 ** np.arange(3))
        self.assertIn("no SNR recorded", plans[0].snr_table[2]["note"])

    def test_refusals(self):
        for args in (["--set", "psd=3", "--top-snr"],
                     ["--set", "mbh=auto", "--ntemps-from-tmax", "--top-snr"],
                     ["--set", "mbh=auto", "--top-snr"],
                     ["--set", "mbh=2", "--top-snr", "--ladder-tmax", "3"]):
            with self.subTest(args=args):
                rc, text = self._main(*args)
                self.assertEqual(rc, 2, text)
                self.assertIn("REFUSING", text)

    def test_without_top_snr_the_tool_is_unchanged(self):
        plans = R.plan(self.path, {"mbh": 5})
        self.assertIsNone(plans[0].snr_table)
        np.testing.assert_allclose(plans[0].ladder, 1 / 1.2 ** np.arange(5))
        self.assertTrue(R.plan(self.path, {"mbh": 2})[0].unchanged)


# ---------------------------------------------------------------------------
# knobs, and the default-off path bit-identical
# ---------------------------------------------------------------------------

class KnobTest(unittest.TestCase):
    def test_settings_default_off_and_read_the_env(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        for cls, b in ((EMRISettings, "EMRI"), (MBHSettings, "MBH"), (SOBBHSettings, "SOBBH")):
            with self.subTest(branch=b), _env():
                s = cls()
                self.assertEqual((s.ladder_top_snr, s.ladder_tmax_min, s.ladder_top_snr_refresh,
                                  s.ladder_top_snr_refresh_pe), (0.0, 4.0, 0, False))
            with self.subTest(branch=b, env=True), _env(**{
                    f"{b}_LADDER_TOP_SNR": "on", f"{b}_LADDER_TMAX_MIN": "5",
                    f"{b}_LADDER_TOP_SNR_REFRESH": "10", f"{b}_LADDER_TOP_SNR_REFRESH_PE": "1"}):
                s = cls()
                self.assertEqual((s.ladder_top_snr, s.ladder_tmax_min, s.ladder_top_snr_refresh,
                                  s.ladder_top_snr_refresh_pe), (2.0, 5.0, 10, True))
            with self.subTest(branch=b, number=True), _env(**{f"{b}_LADDER_TOP_SNR": "1.5"}):
                self.assertEqual(cls().ladder_top_snr, 1.5)

    def test_settings_survive_deepcopy_and_pickle(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings

        with _env(EMRI_LADDER_TOP_SNR="2", EMRI_LADDER_TOP_SNR_REFRESH="5"):
            s = EMRISettings()
        t = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertEqual((t.ladder_top_snr, t.ladder_top_snr_refresh), (2.0, 5))


class DefaultOffGoldenTest(unittest.TestCase):
    """The knob at 0 (and unset, GoldenStreamTest) leaves the real EMRI move's stream
    bit-identical to HEAD aaace7ab's golden: single process and the 4-rank fan-out."""

    def test_the_knob_at_zero_is_the_golden_stream(self):
        ref = np.load(P.GOLDEN)
        scen = dict(P.GOLDEN_SCENARIOS)
        off = dict(EMRI_LADDER_TOP_SNR="0", EMRI_LADDER_TOP_SNR_REFRESH="1",
                   EMRI_LADDER_TOP_SNR_REFRESH_PE="1", EMRI_LADDER_TMAX_MIN="4")
        for base in ("single_off", "block4_off"):
            single, nw, env, kw = scen[base]
            scen[base + "_snr0"] = (single, nw, dict(env, **off), kw)
        with mock.patch.object(P, "GOLDEN_SCENARIOS", scen), _env():
            for base in ("single_off", "block4_off"):
                got = P.golden_stream(base + "_snr0")
                want = {k.split("/", 1)[1]: ref[k] for k in ref.files if k.startswith(base + "/")}
                with self.subTest(scenario=base):
                    self.assertEqual(sorted(got), sorted(want))
                    for key in want:
                        np.testing.assert_array_equal(got[key], want[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()
