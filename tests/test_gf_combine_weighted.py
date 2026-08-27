"""GFCombineMove weighted-cycle mode (GB PE storage fix, 2026-08-26).

USER RULING: the GB PE stage must use the same combine-move style as GB
search -- ALL sub-moves execute inside ONE propose call, so the backend
stores ONE row per iteration -- but with the cycle composition DRAWN each
iteration: ``len(moves)`` draws WITH replacement from the wrapped moves,
weighted (equal weights by default). Search stages keep the fixed
sequential order (default flags) and are untouched.

These tests exercise GFCombineMove directly with stub moves; no sampler,
no GPU, no data.
"""

import unittest

import numpy as np

from lisatools.globalfit.moves.globalfitmove import GFCombineMove


class _StubMove:
    """Minimal move: records its call order on the shared log."""

    temperature_control = None
    periodic = None

    def __init__(self, tag, log):
        self.tag = tag
        self.log = log
        self.accepted = np.zeros((2, 4))

    def propose(self, model, state):
        self.log.append(self.tag)
        return state, np.ones((2, 4))


class _StubModel:
    def __init__(self, seed=1234):
        self.random = np.random.RandomState(seed)


class _StubState:
    sub_states = None


def _make(n, log, **kwargs):
    moves = [_StubMove(i, log) for i in range(n)]
    kwargs.setdefault("share_temperature_control", False)
    return moves, GFCombineMove(moves=moves, **kwargs)


class WeightedCycleTest(unittest.TestCase):
    def test_draws_len_moves_with_replacement_equal_weights(self):
        """One propose = len(moves) weighted draws, executed in drawn order.

        Default weights are EQUAL; the draw must come from ``model.random``
        so a fixed seed reproduces the composition exactly.
        """
        log = []
        _, comb = _make(3, log, weighted_cycle=True)
        model = _StubModel(seed=99)
        comb.propose(model, _StubState())
        mirror = np.random.RandomState(99)
        expected = list(
            mirror.choice(3, size=3, replace=True, p=np.full(3, 1.0 / 3.0))
        )
        self.assertEqual(log, expected)
        self.assertEqual(len(log), 3)

    def test_replacement_actually_happens(self):
        """Across many proposes some cycle must repeat a move (replacement)."""
        log = []
        _, comb = _make(3, log, weighted_cycle=True)
        model = _StubModel(seed=7)
        saw_repeat = False
        for _ in range(50):
            log.clear()
            comb.propose(model, _StubState())
            if len(set(log)) < 3:
                saw_repeat = True
                break
        self.assertTrue(saw_repeat)

    def test_weights_bias_the_draw(self):
        """A delta weight vector runs only the weighted move."""
        log = []
        _, comb = _make(3, log, weighted_cycle=True,
                        move_weights=[1.0, 0.0, 0.0])
        model = _StubModel()
        for _ in range(10):
            comb.propose(model, _StubState())
        self.assertEqual(set(log), {0})
        self.assertEqual(len(log), 30)

    def test_weights_normalize(self):
        """Unnormalized weights are accepted and normalized to sum 1."""
        log = []
        _, comb = _make(3, log, weighted_cycle=True,
                        move_weights=[2.0, 1.0, 1.0])
        model = _StubModel(seed=5)
        comb.propose(model, _StubState())
        mirror = np.random.RandomState(5)
        expected = list(
            mirror.choice(3, size=3, replace=True,
                          p=np.array([2.0, 1.0, 1.0]) / 4.0)
        )
        self.assertEqual(log, expected)

    def test_bad_weights_raise(self):
        log = []
        with self.assertRaises(ValueError):
            _make(3, log, weighted_cycle=True, move_weights=[1.0, 1.0])
        with self.assertRaises(ValueError):
            _make(3, log, weighted_cycle=True,
                  move_weights=[1.0, -1.0, 1.0])
        with self.assertRaises(ValueError):
            _make(3, log, weighted_cycle=True,
                  move_weights=[0.0, 0.0, 0.0])

    def test_accepted_accumulates_over_cycle(self):
        log = []
        _, comb = _make(3, log, weighted_cycle=True)
        model = _StubModel(seed=3)
        _, accepted = comb.propose(model, _StubState())
        self.assertTrue(np.all(accepted == 3))


class SearchModeUntouchedTest(unittest.TestCase):
    def test_default_is_sequential_all_moves_once(self):
        """SEARCH semantics: default flags run every move once, in order."""
        log = []
        _, comb = _make(3, log)
        comb.propose(_StubModel(), _StubState())
        self.assertEqual(log, [0, 1, 2])

    def test_random_choice_legacy_one_per_step(self):
        """Legacy random_choice mode still runs exactly ONE move per step."""
        log = []
        _, comb = _make(3, log, random_choice=True)
        comb.propose(_StubModel(), _StubState())
        self.assertEqual(len(log), 1)

    def test_weighted_cycle_wins_over_random_choice(self):
        """Both flags set: the weighted cycle governs (documented order)."""
        log = []
        _, comb = _make(3, log, weighted_cycle=True, random_choice=True)
        comb.propose(_StubModel(), _StubState())
        self.assertEqual(len(log), 3)


if __name__ == "__main__":
    unittest.main()
