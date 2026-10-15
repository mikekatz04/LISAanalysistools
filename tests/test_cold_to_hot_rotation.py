"""The search-stage cold -> hot rotation of the source in-model moves.

User ruling 2026-10-14: "during all search parts of the sampler, when source
search is run (EMRI, MBHB, SOBHB), we should be copying the cold-chain into
the hot chain rotating by walker so it does not pass to itself ... every 200
proposals for a source (which includes the 25 repeats so 8 iterations)".

Pure-helper tests (the rotation, the never-self offset, the proposal clock)
plus the move-level hook on a bare ``ResidualAddOneRemoveOneMove`` instance
(no constructor: the hook reads only the attributes set here) over a fake
per-leaf ladder sub-state, and the mixin seam that calls the hook before the
state is sliced per rank. Stub-based, seconds.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.moves import addremovemove as arm
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin

NT, NW, NL, ND = 4, 4, 3, 5


class _FakeSub:
    """A per-leaf ladder sub-state: coords (ntemps, nwalkers, nleaves, ndim),
    inds (ntemps, nwalkers, nleaves), log_prior / log_like (nleaves, ntemps,
    nwalkers) -- the PerLeafLadderState layout."""

    tempered_initialized = True

    def __init__(self, nt=NT, nw=NW, nl=NL, nd=ND, seed=0):
        rng = np.random.default_rng(seed)
        self.coords = rng.normal(size=(nt, nw, nl, nd))
        self.inds = np.ones((nt, nw, nl), dtype=bool)
        self.log_prior = rng.normal(size=(nl, nt, nw))
        self.log_like = rng.normal(size=(nl, nt, nw))


class _FakeState:
    def __init__(self, sub, name="mbh"):
        self.sub_states = {name: sub}


def _bare_move(stage="search", repeats=25, every=125, ntemps=NT, name="mbh"):
    m = arm.ResidualAddOneRemoveOneMove.__new__(arm.ResidualAddOneRemoveOneMove)
    m.branch_name = name
    m.num_repeats = repeats
    m.ntemps = ntemps
    m.gf_stage_kind = stage
    m.cold_to_hot_every = every
    return m


class OffsetTest(unittest.TestCase):
    def test_never_zero_and_cycles_through_every_other_walker(self):
        for nw in (2, 3, 4, 7):
            seen = set()
            for event in range(3 * nw):
                k = arm.cold_to_hot_offset(event, nw)
                self.assertGreaterEqual(k, 1, (nw, event))
                self.assertLess(k, nw, (nw, event))
                seen.add(k)
            self.assertEqual(seen, set(range(1, nw)), nw)

    def test_single_walker_has_no_partner(self):
        self.assertEqual(arm.cold_to_hot_offset(0, 1), 0)
        self.assertEqual(arm.cold_to_hot_offset(5, 0), 0)


class ClockTest(unittest.TestCase):
    def test_twenty_five_repeats_fire_every_fifth_propose(self):
        clock, fired = 0, []
        for propose in range(1, 25):
            fires, clock = arm.cold_to_hot_due(clock, 25, 125)
            if fires:
                fired.append(propose)
        self.assertEqual(fired, [5, 10, 15, 20])

    def test_the_original_200_cadence_fires_every_eighth(self):
        clock, fired = 0, []
        for propose in range(1, 25):
            fires, clock = arm.cold_to_hot_due(clock, 25, 200)
            if fires:
                fired.append(propose)
        self.assertEqual(fired, [8, 16, 24])

    def test_default_is_five_proposes_of_25(self):
        self.assertEqual(arm.COLD_TO_HOT_EVERY_DEFAULT, 125)

    def test_first_propose_never_fires(self):
        self.assertEqual(arm.cold_to_hot_due(0, 25, 125)[0], False)
        self.assertEqual(arm.cold_to_hot_due(0, 1, 125)[0], False)

    def test_disabled_cadence_never_fires(self):
        for cad in (0, -1):
            fires, clock = arm.cold_to_hot_due(175, 25, cad)
            self.assertFalse(fires)
            self.assertEqual(clock, 0)

    def test_repeats_larger_than_cadence_fire_every_propose_and_keep_the_remainder(self):
        fires, clock = arm.cold_to_hot_due(0, 25, 10)
        self.assertTrue(fires)
        self.assertEqual(clock, 5)
        fires, clock = arm.cold_to_hot_due(clock, 25, 10)
        self.assertTrue(fires)
        self.assertEqual(clock, 0)


class RotationTest(unittest.TestCase):
    def test_cold_of_w_lands_in_the_hot_rung_of_w_plus_k(self):
        sub = _FakeSub()
        before = sub.coords.copy()
        n = arm.rotate_cold_into_hot(sub.coords, sub.inds, sub.log_prior, hot=NT - 1, offset=1)
        self.assertEqual(n, NW * NL)
        for w in range(NW):
            dst = (w + 1) % NW
            np.testing.assert_array_equal(sub.coords[NT - 1, dst], before[0, w])
            np.testing.assert_array_equal(sub.log_prior[:, NT - 1, dst], np.asarray([sub.log_prior[l, 0, w] for l in range(NL)]))
        # the cold rung and the middle rungs are untouched
        np.testing.assert_array_equal(sub.coords[: NT - 1], before[: NT - 1])

    def test_no_walker_receives_its_own_cold(self):
        sub = _FakeSub()
        before = sub.coords.copy()
        for offset in (1, 2, 3):
            sub.coords[:] = before
            arm.rotate_cold_into_hot(sub.coords, sub.inds, sub.log_prior, hot=NT - 1, offset=offset)
            for w in range(NW):
                self.assertFalse(np.array_equal(sub.coords[NT - 1, w], before[0, w]), (offset, w))

    def test_dead_leaves_are_skipped(self):
        sub = _FakeSub()
        sub.inds[:, :, 1] = False
        before = sub.coords.copy()
        n = arm.rotate_cold_into_hot(sub.coords, sub.inds, sub.log_prior, hot=NT - 1, offset=1)
        self.assertEqual(n, NW * (NL - 1))
        np.testing.assert_array_equal(sub.coords[:, :, 1], before[:, :, 1])

    def test_one_rung_or_one_walker_copies_nothing(self):
        sub = _FakeSub(nt=1)
        before = sub.coords.copy()
        self.assertEqual(arm.rotate_cold_into_hot(sub.coords, sub.inds, sub.log_prior, hot=0, offset=1), 0)
        np.testing.assert_array_equal(sub.coords, before)
        sub = _FakeSub(nw=1)
        before = sub.coords.copy()
        self.assertEqual(arm.rotate_cold_into_hot(sub.coords, sub.inds, sub.log_prior, hot=NT - 1, offset=0), 0)
        np.testing.assert_array_equal(sub.coords, before)

    def test_without_log_prior(self):
        sub = _FakeSub()
        n = arm.rotate_cold_into_hot(sub.coords, sub.inds, None, hot=NT - 1, offset=2)
        self.assertEqual(n, NW * NL)


class MoveHookTest(unittest.TestCase):
    def _run(self, move, state, n):
        fired = []
        for _ in range(n):
            fired.append(bool(move._maybe_rotate_cold_into_hot(state)))
        return fired

    def test_search_stage_fires_on_the_fifth_propose_with_logging(self):
        sub = _FakeSub()
        state = _FakeState(sub)
        move = _bare_move("search")
        before = sub.coords.copy()
        with self.assertLogs(arm.logger, level="INFO") as cm:
            fired = self._run(move, state, 5)
        self.assertEqual(fired, [False] * 4 + [True])
        self.assertEqual(move.n_cold_to_hot_events, 1)
        # event 0 -> offset 1: walker w's cold sits in the hottest rung of w+1
        for w in range(NW):
            np.testing.assert_array_equal(sub.coords[NT - 1, (w + 1) % NW], before[0, w])
        text = "\n".join(cm.output)
        self.assertIn("cold->hot", text)
        self.assertIn("offset 1", text)

    def test_rj_stage_fires_and_rotates_the_offset_across_events(self):
        sub = _FakeSub()
        state = _FakeState(sub)
        move = _bare_move("rj")
        fired = self._run(move, state, 15)
        self.assertEqual([i + 1 for i, f in enumerate(fired) if f], [5, 10, 15])
        self.assertEqual(move.n_cold_to_hot_events, 3)
        self.assertEqual(move._cold_to_hot_offsets_seen, [1, 2, 3])

    def test_pe_stage_never_fires(self):
        sub = _FakeSub()
        state = _FakeState(sub)
        before = sub.coords.copy()
        move = _bare_move("pe")
        self.assertFalse(any(self._run(move, state, 40)))
        np.testing.assert_array_equal(sub.coords, before)
        self.assertEqual(move.n_cold_to_hot_events, 0)

    def test_unknown_stage_never_fires(self):
        sub = _FakeSub()
        move = _bare_move(None)
        self.assertFalse(any(self._run(move, _FakeState(sub), 40)))

    def test_disabled_knob_never_fires(self):
        sub = _FakeSub()
        move = _bare_move("search", every=0)
        self.assertFalse(any(self._run(move, _FakeState(sub), 40)))

    def test_hottest_sampled_rung_is_the_target(self):
        """The move samples ``self.ntemps`` rungs of a possibly longer stored ladder."""
        sub = _FakeSub(nt=6)
        state = _FakeState(sub)
        before = sub.coords.copy()
        move = _bare_move("search", ntemps=4)
        self._run(move, state, 5)
        for w in range(NW):
            np.testing.assert_array_equal(sub.coords[3, (w + 1) % NW], before[0, w])
        np.testing.assert_array_equal(sub.coords[4:], before[4:])

    def test_missing_substate_is_a_noop(self):
        move = _bare_move("search")
        state = _FakeState(None)
        self.assertFalse(any(self._run(move, state, 10)))

    def test_env_knob_resolution(self):
        move = _bare_move("search", every=None)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MBH_COLD_TO_HOT_EVERY", None)
            self.assertEqual(move._cold_to_hot_every(), 125)
            os.environ["MBH_COLD_TO_HOT_EVERY"] = "50"
            self.assertEqual(move._cold_to_hot_every(), 50)
            move.cold_to_hot_every = 7                      # the kwarg wins
            self.assertEqual(move._cold_to_hot_every(), 7)


class MixinSeamTest(unittest.TestCase):
    def test_propose_calls_prepare_state_before_the_body(self):
        order = []

        class _M(WalkerFanoutMixin):
            fanout = None

            def fanout_prepare_state(self, model, state):
                order.append(("prepare", state))

            def propose_local(self, model, state):
                order.append(("body", state))
                return state, None

        m = _M()
        m.propose("model", "state")
        self.assertEqual(order, [("prepare", "state"), ("body", "state")])

    def test_default_prepare_is_a_noop(self):
        class _M(WalkerFanoutMixin):
            fanout = None

            def propose_local(self, model, state):
                return state, None

        self.assertIsNone(_M().fanout_prepare_state("model", "state"))


if __name__ == "__main__":
    unittest.main()
